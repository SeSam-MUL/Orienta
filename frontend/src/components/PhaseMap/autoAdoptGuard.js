/**
 * Guard for the Phase-Map visit-sync's AUTOMATIC result adoption.
 *
 * Activating a result auto-switches the backend's loaded file to that
 * result's source file. That side effect is wanted when the user clicks a
 * gallery chip — the click IS the intent — but the visit-sync adopts on a
 * mere page visit. "Nothing active" can also mean "the user just loaded a
 * DIFFERENT file" (loading resets the active result), and adopting then
 * silently replaced that file with the old result's source
 * (user-hit 2026-08-17).
 *
 * Rule: auto-adoption is allowed only when it cannot switch the file —
 * either no file is loaded (nothing to lose) or the result belongs to the
 * file that is already loaded.
 */

/** Normalise a filesystem path for comparison (Windows-tolerant). */
function normPath(p) {
  return String(p || '').replace(/\//g, '\\').toLowerCase();
}

/**
 * @param {string|null|undefined} resultSourceFile  The result's source_file.
 * @param {string|null|undefined} currentFile       The backend's loaded file.
 * @returns {boolean} true when activating the result would replace the
 *   currently loaded file with a different one.
 */
export function wouldSwitchFile(resultSourceFile, currentFile) {
  if (!resultSourceFile || !currentFile) return false;
  return normPath(resultSourceFile) !== normPath(currentFile);
}

export default wouldSwitchFile;
