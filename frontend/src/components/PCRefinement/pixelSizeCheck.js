/**
 * Is this detector pixel size physically possible?
 *
 * User report (2026-08-20): 0.1 mm was entered for a 78 px wide pattern,
 * giving 1.28 um/px, and the app accepted it in silence. Real EBSD phosphor
 * screens are 25-30 mm wide and land at roughly 50-70 um/px; 1.28 um/px is
 * not a detector, it is a typo.
 *
 * Every reviewer of this problem said the same thing: the danger is not that
 * the value is wrong, it is that nothing looks wrong afterwards. Detector
 * geometry feeds the pattern centre, every forward simulation and every
 * refinement — the maps stay colourful and the numbers stay plausible while
 * being wrong. So the check is deliberately loud, and deliberately wide
 * enough that an unusual but real detector does not trip it.
 */

// Comfortably brackets every EBSD detector in normal use, including heavy
// binning (which raises um/px) and large-screen low-binning setups.
export const TYPICAL_MIN_UM = 20;
export const TYPICAL_MAX_UM = 150;

/**
 * @param {number} umPerPx  detector width in um divided by width in pixels
 * @returns {{level: 'ok'|'warn'|'unset', umPerPx: number|null}}
 *   'warn' means "no EBSD detector looks like this — check the input".
 */
export function checkPixelSize(umPerPx) {
  const v = Number(umPerPx);
  if (!Number.isFinite(v) || v <= 0) return { level: 'unset', umPerPx: null };
  if (v < TYPICAL_MIN_UM || v > TYPICAL_MAX_UM) return { level: 'warn', umPerPx: v };
  return { level: 'ok', umPerPx: v };
}

/**
 * The detector width that WOULD give a typical pixel size, for a "did you
 * mean…" hint. Returns null when the pattern width is unknown.
 */
export function plausibleWidthMm(unbinnedW, targetUmPerPx = 60) {
  const w = Number(unbinnedW);
  if (!Number.isFinite(w) || w <= 0) return null;
  return (w * targetUmPerPx) / 1000;
}
