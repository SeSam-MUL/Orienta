/**
 * Where a save dialog opens.
 *
 * Every save dialog used to open in the Downloads folder (the M5 tester:
 * "Speicherdialoge starten in Downloads"), so a user writing a series of
 * figures into a report folder navigated there once per file. The folder of
 * the last file saved is remembered in `<orienta home>/electron/last-save-dir.txt`
 * and used as the starting directory next time — across restarts, because a
 * report is written over days.
 *
 * Pure functions plus two tiny file helpers; nothing here touches Electron, so
 * the module is tested by running it.
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const FILE_NAME = 'last-save-dir.txt';

function lastDirFile(home) {
  return path.join(home, 'electron', FILE_NAME);
}

/** The remembered folder, or null when there is none or it no longer exists. */
function readLastDir(home) {
  if (!home) return null;
  try {
    const dir = fs.readFileSync(lastDirFile(home), 'utf8').trim();
    if (!dir) return null;
    // A UNC path is not probed: stat on an unreachable share blocks the main
    // process for the SMB timeout, right when the user clicked Save. The
    // dialog copes with a starting folder that is not there.
    if (dir.startsWith('\\\\')) return dir;
    return fs.statSync(dir).isDirectory() ? dir : null;
  } catch {
    return null;
  }
}

/** Remember the folder of a file the user just saved (or a folder they chose). */
function rememberDir(home, filePath, { isDirectory = false } = {}) {
  if (!home || !filePath) return null;
  const dir = isDirectory ? path.resolve(filePath) : path.dirname(path.resolve(filePath));
  try {
    fs.mkdirSync(path.dirname(lastDirFile(home)), { recursive: true });
    fs.writeFileSync(lastDirFile(home), dir + '\n', 'utf8');
    return dir;
  } catch {
    return null;
  }
}

/**
 * The `defaultPath` for a save dialog.
 *
 * - An absolute path from the caller wins: the phase-map export names its
 *   file, the crop export names its file next to the source.
 * - A bare file name (`image.png`) is placed in the remembered folder, else
 *   in the fallback (Downloads), else left to the dialog.
 * - No name at all: the folder alone, so the dialog opens there with an
 *   empty name (the Browse buttons).
 */
function resolveDefaultPath({ requested, lastDir, fallbackDir } = {}) {
  const req = requested == null ? '' : String(requested);
  if (req && path.isAbsolute(req)) return req;
  const base = lastDir || fallbackDir || null;
  if (!req) return base || undefined;
  // A relative path with directories in it is the caller's own layout; do
  // not re-root it under the remembered folder.
  if (req !== path.basename(req)) return req;
  return base ? path.join(base, req) : req;
}

module.exports = { FILE_NAME, lastDirFile, readLastDir, rememberDir, resolveDefaultPath };
