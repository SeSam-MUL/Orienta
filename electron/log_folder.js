/**
 * Open the folder that holds Orienta's log files.
 *
 * Kept out of main.js so it can be tested: main.js spawns the backend when it is
 * loaded. The folder is the one `openBackendLog()` writes backend-console.log
 * to and the backend writes orienta.log to - `<project root>/logs`, where the
 * project root is the installed runtime folder in a packaged app and the
 * checkout in development.
 *
 * Never throws: it runs inside an IPC handler, where a throw reaches the
 * renderer as an opaque rejection.
 */
const fs = require('node:fs');
const path = require('node:path');

/**
 * @param {{ shell: { openPath: (p: string) => Promise<string> }, projectRoot: string }} deps
 * @returns {Promise<{ ok: boolean, path: string, error?: string }>}
 */
async function openLogFolder({ shell, projectRoot }) {
  const dir = path.join(projectRoot, 'logs');
  try {
    // A fresh install has no log folder until the backend has started once;
    // opening a folder that does not exist would fail with a bare error.
    fs.mkdirSync(dir, { recursive: true });
  } catch { /* openPath reports it */ }
  try {
    // shell.openPath resolves to '' on success and to an error message otherwise.
    const error = await shell.openPath(dir);
    return error ? { ok: false, path: dir, error } : { ok: true, path: dir };
  } catch (err) {
    return { ok: false, path: dir, error: err && err.message ? err.message : String(err) };
  }
}

module.exports = { openLogFolder };
