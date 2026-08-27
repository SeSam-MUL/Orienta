/**
 * A picture of the app window for a problem report.
 *
 * Only the Electron shell can do this: a browser cannot screenshot itself
 * without asking the user to pick a window, which is worse than no picture.
 * So this returns null in a browser and the report simply has no screenshot —
 * everything else in it still works.
 *
 * Capture BEFORE the report dialog opens, or the picture shows the dialog
 * instead of the screen the user is complaining about.
 */

const MAX_BYTES = 4 * 1024 * 1024;

/**
 * @returns {Promise<string|null>} base64 PNG (no data: prefix), or null when
 *   unavailable, too large, or the capture failed.
 */
export async function captureScreenshot() {
  try {
    if (!window.electronAPI?.captureScreen) return null;
    const base64 = await window.electronAPI.captureScreen();
    if (typeof base64 !== 'string' || !base64) return null;
    // base64 is ~4/3 of the byte size; a huge window on a 4K screen could
    // otherwise dwarf the logs it is meant to accompany.
    if (base64.length * 0.75 > MAX_BYTES) return null;
    return base64;
  } catch {
    return null;
  }
}

/** True when a screenshot can be taken at all (Electron only). */
export function canCaptureScreenshot() {
  return Boolean(window.electronAPI?.captureScreen);
}
