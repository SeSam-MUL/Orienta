/**
 * What counts as Orienta's own in a folder, decided by SHAPE and not by name.
 *
 * Names such as `logs`, `pending`, `electron`, `python` or `runtime` are
 * everyday folder names. A rule "a folder holding only these names is
 * Orienta's" adopted a user's own `D:\Work\Orienta` that had a `logs\` of
 * measurements in it, and the uninstaller then deleted that folder by name. So:
 *
 *   * An entry is Orienta's LEFTOVER only if it has the SHAPE of what Orienta
 *     itself writes there (`electron\` holding only the folders Chromium is
 *     given, `logs\` holding only Orienta's log files, ...).
 *   * `python`, `runtime` and the conda folders are NEVER accepted by name in a
 *     folder that is not recognisably Orienta's data folder: they are accepted
 *     only through the marker (`isOrientaHome`).
 *
 * The NSIS uninstaller (electron/nsis/uninstall.nsh) cannot import this file, so
 * it carries the same rules by hand. tests/leftover_shapes.json is ONE table of
 * cases that both implementations are run against (tests/electron/leftovers
 * and tests/test_uninstall_nsis.py), and a test compares the name lists.
 */
const fs = require('node:fs');
const path = require('node:path');

const platform = require('./platform.js');

/** The one marker setup writes; it is what makes a folder Orienta's own. */
const MARKER = '.orienta-home';

/** Files at the top of a home that only Orienta writes (specific names). */
const LEFTOVER_FILES = [
  '.python_path', '.install_mode', '.install_incomplete',
  '.packages_lock.json', '.packages_sync.json', '.packages_sync_failed.json',
];
/** Folders at the top whose SHAPE is checked (see `isLeftoverEntry`). */
const LEFTOVER_DIRS = ['electron', 'logs', 'pending', 'setup-tmp'];
/** `pending.json` is Orienta's only when its `pending` folder is beside it. */
const PENDING_RECORD = 'pending.json';
/** Accepted ONLY inside a recognised data folder. */
const OWNED_ONLY = ['python', 'runtime', platform.CONDA_ROOT_DIR, platform.CONDA_ENVS_DIR];

/** Every name Orienta creates at the top of its home (what a removal may touch). */
const ALL_NAMES = [
  ...LEFTOVER_DIRS, 'python', 'runtime', PENDING_RECORD, MARKER, ...LEFTOVER_FILES,
  platform.CONDA_ROOT_DIR, platform.CONDA_ENVS_DIR,
];

/** What Electron is given under `electron\`: userData, sessionData, temp, and the write probe. */
const ELECTRON_CHILDREN = ['userdata', 'sessiondata', 'temp', '.writable'];
/** Prefixes of what the setup puts in `setup-tmp\` while it runs. */
const SETUP_TMP_PREFIXES = [
  'cpython-', 'orienta-', 'python-unpacked', 'micromamba',
  'extract-result', 'apply-result', 'package-sync',
];

const LITTER = new Set(['.ds_store', 'thumbs.db', 'desktop.ini']);
const isLitter = (name) => LITTER.has(String(name).toLowerCase());

/** `dir` without trailing separators -- except a root, which keeps its one. */
function trimSeps(dir) {
  const text = String(dir);
  const root = path.parse(text).root;
  let end = text.length;
  while (end > root.length && (text[end - 1] === '/' || text[end - 1] === '\\')) end--;
  return text.slice(0, end);
}

/** The same folder, the way Windows sees it: case and trailing separators ignored. */
function sameFolder(a, b) {
  const norm = (p) => trimSeps(path.resolve(String(p))).toLowerCase();
  return norm(a) === norm(b);
}

const isChainName = (name) => String(name).toLowerCase() === 'orienta';

function lstatOrNull(target, io) {
  try { return io.lstatSync(target); } catch { return null; }
}

/**
 * Is `dir` an Orienta data folder that has been set up: the marker, or -- an
 * install from before the marker -- runtime/VERSION together with a second sign
 * (.python_path, or an interpreter). ONE definition: the uninstaller, the
 * in-app removal, the folder chooser and the setup all use this rule.
 */
function isOrientaHome(dir, { exists = fs.existsSync } = {}) {
  const has = (...parts) => { try { return Boolean(exists(path.join(dir, ...parts))); } catch { return false; } };
  if (has(MARKER)) return true;
  return has('runtime', 'VERSION')
    && (has('.python_path') || has('python', 'python.exe') || has('python', 'bin', 'python3'));
}

/** Name of a log file Orienta writes: orienta*.log, backend-console.log, and their numbered rotations. */
function isOrientaLogName(name) {
  const lower = String(name).toLowerCase();
  if (!(lower.startsWith('orienta') || lower.startsWith('backend-console'))) return false;
  // Rotations are .log.1 ... .log.9 (file_log.py keeps three, the shell log one).
  return lower.endsWith('.log') || /\.log\.\d$/.test(lower);
}

/** The children of `dir`, or null if it cannot be listed. */
function childrenOf(dir, io) {
  try { return io.readdirSync(dir); } catch { return null; }
}

/**
 * Does the entry `name` in `home` have the shape of something Orienta wrote
 * there (and nothing of anybody's)? `python` and `runtime` never do.
 */
function isLeftoverEntry(home, name, io = fs) {
  const lower = String(name).toLowerCase();
  const full = path.join(home, name);
  if (LEFTOVER_FILES.includes(lower)) return true;
  if (lower === PENDING_RECORD) {
    const st = lstatOrNull(path.join(home, 'pending'), io);
    return Boolean(st && st.isDirectory() && !st.isSymbolicLink());
  }
  if (!LEFTOVER_DIRS.includes(lower)) return false;
  const st = lstatOrNull(full, io);
  if (!st || !st.isDirectory() || st.isSymbolicLink()) return false;
  const kids = childrenOf(full, io);
  if (!kids) return false;
  return kids.every((kid) => {
    const k = kid.toLowerCase();
    if (lower === 'electron') return ELECTRON_CHILDREN.includes(k);
    if (lower === 'pending') return k.startsWith('orienta-runtime-');
    if (lower === 'setup-tmp') return SETUP_TMP_PREFIXES.some((p) => k.startsWith(p)) || k.endsWith('.tmp');
    // logs: Orienta's own log files, and nothing else (no folders)
    const kst = lstatOrNull(path.join(full, kid), io);
    return Boolean(kst && kst.isFile() && isOrientaLogName(kid));
  });
}

/**
 * Does `dir` hold nothing but (nested) empty folders? A link is never empty;
 * bounded, so a huge tree cannot stall the wizard (past the budget: not empty).
 */
function isEmptyDirTree(dir, budget = { left: 500 }, io = fs) {
  const st = lstatOrNull(dir, io);
  if (!st || !st.isDirectory() || st.isSymbolicLink()) return false;
  const kids = childrenOf(dir, io);
  if (!kids) return false;
  for (const kid of kids) {
    if (--budget.left < 0) return false;
    if (isLitter(kid)) {
      const kst = lstatOrNull(path.join(dir, kid), io);
      if (kst && kst.isFile()) continue;     // a FILE of that name only
    }
    if (!isEmptyDirTree(path.join(dir, kid), budget, io)) return false;
  }
  return true;
}

/**
 * Is the folder `dir` (a level of an Orienta/Orienta/... chain) made of
 * Orienta's leftovers only: entries of the shape above, operating-system litter,
 * empty folders, and the next chain level -- recursively?
 */
function isLeftoverTree(dir, io = fs, depth = 0) {
  if (depth > 16) return false;
  const st = lstatOrNull(dir, io);
  if (!st || !st.isDirectory() || st.isSymbolicLink()) return false;
  const kids = childrenOf(dir, io);
  if (!kids) return false;
  return kids.every((kid) => {
    const full = path.join(dir, kid);
    if (isLitter(kid)) {
      const kst = lstatOrNull(full, io);
      if (kst && kst.isFile()) return true;
    }
    if (isLeftoverEntry(dir, kid, io)) return true;
    if (isChainName(kid)) return isLeftoverTree(full, io, depth + 1);
    return isEmptyDirTree(full, { left: 500 }, io);
  });
}

/**
 * The entries of `home` that Orienta did not put there.
 *
 * In a recognised data folder every name Orienta creates is accepted (the
 * repair and resume runs). Anywhere else only the SHAPE of a leftover is: a
 * user's `logs\` of measurements, or a `python\myscript.py`, is theirs.
 */
function foreignEntries(home, entries = null, io = fs) {
  let names = entries;
  if (!names) {
    names = childrenOf(home, io);
    if (!names) return [];   // unreadable: the steps below will say so in their own words
  }
  const owned = isOrientaHome(home, { exists: (p) => lstatOrNull(p, io) !== null });
  return names.filter((name) => {
    if (owned && (ALL_NAMES.includes(name.toLowerCase()) || /^python\.old-/.test(name))) return false;
    if (isLeftoverEntry(home, name, io)) return false;
    const full = path.join(home, name);
    if (isLitter(name)) {
      const st = lstatOrNull(full, io);
      if (st && st.isFile()) return false;
    }
    if (isEmptyDirTree(full, { left: 500 }, io)) return false;
    if (isChainName(name) && isLeftoverTree(full, io)) return false;
    return true;
  });
}

module.exports = {
  MARKER,
  LEFTOVER_FILES,
  LEFTOVER_DIRS,
  PENDING_RECORD,
  OWNED_ONLY,
  ALL_NAMES,
  ELECTRON_CHILDREN,
  SETUP_TMP_PREFIXES,
  isLitter,
  trimSeps,
  sameFolder,
  isChainName,
  isOrientaHome,
  isOrientaLogName,
  isLeftoverEntry,
  isEmptyDirTree,
  isLeftoverTree,
  foreignEntries,
};
