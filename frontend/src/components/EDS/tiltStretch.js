/**
 * Display stretch that undoes the tilt foreshortening of an .h5oina
 * (backend services/tilt_distortion.py, `tiltCheck` in the data store).
 *
 * AZtec stores images and maps of a tilted sample with square pixels, so one
 * direction is foreshortened by k = 1/cos(tilt) on the sample (2.92 at 70°).
 * Drawn with this stretch, the picture has the sample's true aspect ratio.
 * Only the DISPLAY changes: the bitmap and its pixel grid stay as they are, and
 * pointer → (row, col) mapping works on fractions of the box, so a click still
 * lands on the same pixel.
 *
 * The stretch is applied only when the check confirmed the direction
 * (befund "X gestaucht" / "Y gestaucht") — an unclear finding is never
 * corrected by a guess. Electron images always follow the finding. Maps
 * (element, BC, …) and the composite overlay, which sits on the map grid,
 * follow it too: they are rastered by the same beam over the same tilted
 * surface. The one exception is a map the check found on a raster of its own
 * (`erbt_bildverzerrung: false`) -- that one is left as it is. When the check
 * could not tell (no `Relative Size`), the maps are corrected with the image,
 * as the page already draws the image over the whole map.
 */

export const NO_STRETCH = Object.freeze({ sx: 1, sy: 1 });

/** `{ sx, sy, axis, factor }` of a confirmed finding, or null. */
export function tiltFinding(check) {
  if (!check || check.fehler) return null;
  const k = Number(check.faktor_korrektur);
  if (!(k > 1.001) || !Number.isFinite(k)) return null;
  if (check.befund === 'X gestaucht') return { sx: k, sy: 1, axis: 'X', factor: k };
  if (check.befund === 'Y gestaucht') return { sx: 1, sy: k, axis: 'Y', factor: k };
  return null;
}

/** False only when the check found the maps on a raster of their own. */
export function mapsInherit(check) {
  const maps = check?.karten || [];
  const eds = maps.find((m) => m.typ === 'EDS');
  const ebsd = maps.find((m) => m.typ === 'EBSD');
  return (eds ?? ebsd)?.erbt_bildverzerrung !== false;
}

/**
 * The factor to suggest for a manual correction: k = 1/cos(tilt) from the
 * header when it says the sample was tilted, otherwise 1/cos 70° (2.92), the
 * usual EBSD tilt.
 */
export function suggestedManualFactor(check) {
  const k = Number(check?.faktor_k_theoretisch);
  if (Number.isFinite(k) && k > 1.001) return Math.round(k * 100) / 100;
  return 2.92;
}

/**
 * A manual correction the user chose (`{ axis: 'X'|'Y', factor }`) as a stretch,
 * or null when there is none or it is not usable. Manual means the user judged
 * the image: it applies to images and maps alike.
 */
export function manualStretch(manual) {
  const k = Number(manual?.factor);
  if (!manual || !(k > 1.001) || !Number.isFinite(k) || k > 20) return null;
  if (manual.axis === 'X') return { sx: k, sy: 1 };
  if (manual.axis === 'Y') return { sx: 1, sy: k };
  return null;
}

/**
 * Stretch for one layer kind ('electron', 'eds-element', 'bc', 'vbse', 'mask', …),
 * or for the composite overlay ('composite', on the map grid).
 *
 * `manual` (optional): a correction the user set by hand for a file the check
 * could not confirm. It is used only when there is no confirmed finding — a
 * confirmed one always wins, so a hand setting cannot contradict a measurement.
 */
export function stretchFor(check, kind, enabled = true, manual = null) {
  if (!enabled) return NO_STRETCH;
  const f = tiltFinding(check);
  if (!f) return manualStretch(manual) || NO_STRETCH;
  if (kind === 'electron') return { sx: f.sx, sy: f.sy };
  return mapsInherit(check) ? { sx: f.sx, sy: f.sy } : NO_STRETCH;
}

/** CSS aspect-ratio of a w × h box drawn with `stretch`. */
export function stretchedAspect(w, h, stretch = NO_STRETCH) {
  const sx = stretch?.sx || 1;
  const sy = stretch?.sy || 1;
  const W = w * sx;
  const H = h * sy;
  // Rounded to keep the CSS value short; four decimals are far below a pixel.
  return `${Math.round(W * 1e4) / 1e4}/${Math.round(H * 1e4) / 1e4}`;
}
