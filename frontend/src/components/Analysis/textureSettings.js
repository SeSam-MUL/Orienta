// The Texture tab's preset and tolerance angle: what the page offers, the defaults, and how a typed tolerance is read.
// The backend (POST /api/analysis/texture, the batch) takes the same presets and tolerances in (0, 30] degrees.

/** The texture presets the page offers (the names the backend knows). */
export const TEXTURE_PRESET_NAMES = ['FCC_Rolling', 'BCC_Rolling'];
export const DEFAULT_TEXTURE_PRESET = 'FCC_Rolling';
export const DEFAULT_TEXTURE_TOLERANCE = '10.0';
export const MAX_TEXTURE_TOLERANCE = 30;

/** The tolerance angle as a number of degrees (more than 0, at most 30), else undefined. */
export function textureToleranceOf(typed) {
  const n = parseFloat(typed);
  return Number.isFinite(n) && n > 0 && n <= MAX_TEXTURE_TOLERANCE ? n : undefined;
}

/** A preset name the page offers, else the default. */
export function texturePresetOf(name) {
  return TEXTURE_PRESET_NAMES.includes(name) ? name : DEFAULT_TEXTURE_PRESET;
}

/**
 * The tolerance field after the backend said which tolerance it holds: the typed text stays when it already shows the
 * same number ("10.0" for 10), a number that is out of range or not a number is the default.
 */
export function textureToleranceFromBackend(typed, backendValue) {
  const n = Number(backendValue);
  if (!Number.isFinite(n) || n <= 0 || n > MAX_TEXTURE_TOLERANCE) return DEFAULT_TEXTURE_TOLERANCE;
  return parseFloat(typed) === n ? typed : String(n);
}

/**
 * What the batch is sent for its texture step: the Texture tab's preset and tolerance as they are now, or undefined for
 * the tolerance when the field does not hold a usable number (the batch then uses its own 10 degrees).
 */
export function textureBatchArgs(preset, typedTolerance) {
  return { preset: texturePresetOf(preset), toleranceDeg: textureToleranceOf(typedTolerance) };
}
