/**
 * Choosing the data folder, and what to do with a choice that no longer holds.
 *
 * Two problems lead here:
 *   1. Every click on the button that chooses the storage location added another
 *      Orienta subfolder (…/Orienta/Orienta/Orienta).
 *   2. After uninstalling and deleting everything by hand, Orienta could not be
 *      installed again.
 *
 * The first one was the wizard's own doing. The folder dialog opened AT the
 * current data folder (…\Orienta); picking what was pre-selected is the natural
 * click, and the code then appended "Orienta" to whatever was picked. Each
 * round trip therefore nested one more level, and the pointer file
 * (%APPDATA%\Orienta\home.txt) recorded the deepest one. These functions keep
 * that step idempotent and say what happens when the recorded folder is gone.
 *
 * Nothing here deletes a data folder, and nothing deletes the pointer: a pointer
 * to a folder that is demonstrably gone is renamed to home.txt.previous.
 */
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const leftovers = require('./leftovers.js');
const volumeId = require('./volume_id.js');

const marker = leftovers.MARKER;
const { trimSeps, sameFolder, isOrientaHome } = leftovers;

/** The last path segment is "Orienta", in any letter case (Windows paths are
 *  case-insensitive, and a user may well have typed "orienta"). */
function endsWithOrienta(dir) {
  return path.basename(trimSeps(dir)).toLowerCase() === 'orienta';
}

/**
 * Is `dir` absent, empty, or made only of Orienta's own leftovers by SHAPE (see
 * leftovers.js -- the rule the setup refuses to share a folder by)? Empty
 * directory trees count as nothing, as they do there.
 */
function holdsOnlyOrientaNames(dir) {
  return leftovers.foreignEntries(dir).length === 0;
}

/**
 * The data folder that results from picking `picked` in the dialog.
 *
 * Orienta keeps its data in a folder of its own, never scattered across the
 * root of whatever was picked (someone choosing D:\ must not get python\,
 * runtime\ and logs\ across the drive, and an uninstall has to know what is
 * ours). So a subfolder named Orienta is added — UNLESS the picked folder
 * already is Orienta's: it carries the marker setup writes, or it is named
 * Orienta AND holds nothing but Orienta's own names (or nothing at all). That
 * makes the step idempotent: picking the folder that was offered gives the
 * folder that was already in use, not another level below it.
 *
 * The NAME alone is not enough: a user's own non-empty folder (D:/Work/orienta)
 * would otherwise become the data folder, and the shell would write electron/
 * and logs/ into it.
 */
function homeFromPick(picked, { exists = fs.existsSync } = {}) {
  const clean = trimSeps(picked);
  let marked = false;
  try { marked = Boolean(exists(path.join(clean, marker))); } catch { /* unreadable: not marked */ }
  if (marked) return clean;
  if (endsWithOrienta(clean) && holdsOnlyOrientaNames(clean)) return clean;
  return path.join(clean, 'Orienta');
}

/**
 * The installed data folder inside `candidate`, following nested Orienta
 * folders downward -- or null.
 *
 * A 0.4.6 user whose real installation sits at X/Orienta/Orienta picks X in the
 * repaired dialog and gets candidate X/Orienta, which holds nothing but the
 * next Orienta. That installation is not "foreign files" in the way: it is
 * the data folder, and the choice has to land on it.
 */
function findExistingHome(candidate, { exists = fs.existsSync } = {}) {
  let dir = candidate;
  for (let level = 0; level < 16; level++) {
    if (isOrientaHome(dir, { exists })) return dir;
    let next = null;
    try {
      for (const name of fs.readdirSync(dir)) {
        if (name.toLowerCase() !== 'orienta') continue;
        const child = path.join(dir, name);
        const st = fs.lstatSync(child);
        if (st.isDirectory() && !st.isSymbolicLink()) { next = child; break; }
      }
    } catch { return null; }
    if (!next) return null;
    dir = next;
  }
  return null;
}

/**
 * Everything the "change..." button decides, BEFORE anything is written.
 *
 *   home       the folder to record
 *   adopted    an installation was found below the pick and is used as it is
 *   unchanged  that is the folder already in use: nothing to record
 *   refusal    the folder cannot be used (null if it can); the pointer must not
 *              be touched, or a working installation is orphaned behind a
 *              message that arrived too late
 *
 * An installed folder is never refused: it holds user notes and a library, and
 * "contains other files" is not a reason to leave it. The folder already in
 * use IS asked about when nothing says it is installed: "Orienta keeps using
 * it" followed by an Install that refuses it was a loop, and nothing is
 * written for an unchanged pick, so telling the user costs nothing.
 */
function decidePick(picked, { current, refuse = defaultRefusal } = {}) {
  const base = homeFromPick(picked);
  const existing = findExistingHome(base);
  const home = existing || base;
  const adopted = Boolean(existing) && existing !== base;
  const unchanged = Boolean(current) && sameFolder(home, current);
  const refusal = existing ? null : refuse(home);
  return { home, adopted, unchanged, refusal };
}

function defaultRefusal(home) {
  return require('./setup/installer').homeRefusal(home);
}

/**
 * Where the dialog should open for the current data folder.
 *
 * In the folder that CONTAINS it. The dialog selects the folder it opens in;
 * opening inside …\Orienta made "OK" mean "this one", and the old code then
 * added another level. Every trailing Orienta segment is stripped, so a
 * pointer a 0.4.6 user already nested three deep opens at the real parent. If
 * that parent is not there any more, the nearest one that is.
 */
function pickerStart(current, { exists = fs.existsSync } = {}) {
  let dir = trimSeps(current || '');
  if (!dir) return undefined;
  while (endsWithOrienta(dir) && path.dirname(dir) !== dir) dir = path.dirname(dir);
  for (let guard = 0; guard < 64; guard++) {
    let there = false;
    try { there = Boolean(exists(dir)); } catch { /* treat as absent */ }
    if (there) return dir;
    const up = path.dirname(dir);
    if (up === dir) return undefined;
    dir = up;
  }
  return undefined;
}

/** Same drive letter (the system drive is Fixed without asking). */
function sameDriveAs(root, other) {
  const a = String(root).slice(0, 2).toLowerCase();
  const b = String(path.parse(String(other)).root).slice(0, 2).toLowerCase();
  return /^[a-z]:$/.test(a) && a === b;
}

/**
 * What kind of drive is `root` (E:\\)? 'Fixed', 'Removable', 'Network',
 * 'CDRom', 'Ram' -- or null if it cannot be told. A UNC path is a network
 * drive without asking. The cheapest reliable answer without a native module is
 * .NET's DriveInfo through PowerShell; it runs only on the rare path where a
 * recorded folder is gone and its drive is not the system drive.
 */
function windowsDriveType(root) {
  const text = String(root);
  if (text.startsWith('\\\\')) return 'Network';
  const letter = text.slice(0, 1);
  if (!/^[A-Za-z]$/.test(letter)) return null;
  try {
    const out = require('node:child_process').execFileSync(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', `(New-Object System.IO.DriveInfo '${letter}').DriveType`],
      { encoding: 'utf8', timeout: 8000, windowsHide: true });
    return String(out).trim() || null;
  } catch {
    return null;
  }
}

/**
 * Is the recorded data folder still there?
 *
 *   none        no pointer, or an environment override that wins anyway
 *   present     the folder exists
 *   stale       it is gone, and what it sat in is a plain directory that
 *               answers: somebody deleted it
 *   unreachable it cannot be told apart from an unplugged drive, an offline
 *               share, a link whose target went away, or a folder that is
 *               merely not readable right now
 *
 * `stale` and `unreachable` call for opposite answers: a stale pointer sends
 * every later start to a folder nobody wants, an unreachable one is the only
 * record of where the user's crystal library is. So `stale` has to be PROVEN:
 *
 *   - the stat of the folder fails with ENOENT or ENOTDIR and nothing else
 *     (EACCES, EPERM, EIO, a share's UNKNOWN: the folder may well be there);
 *   - the nearest ancestor that exists is a directory and not a link (a
 *     junction or symlink whose target is gone looks exactly like a deleted
 *     folder from below);
 *   - off Windows, that ancestor lies inside the user's home on the same
 *     device as the home. Anything else may be a mount point that is not
 *     mounted right now -- an empty directory on the system disk.
 *
 * Known limits, not fixable cheaply: on Linux an UNMOUNTED mount point inside
 * the user's own home (an empty directory on the home device) cannot be told
 * from a deleted folder without reading the mount table, which this does not
 * do, so it reads as stale -- the pointer is then kept as home.txt.previous, not
 * lost. And a stat of a dead network host blocks until the operating system
 * gives up (tens of seconds on Windows) before the first window can open; that
 * predates this check and has no cheap timeout in synchronous code.
 *
 * The probes are injectable (`fsImpl`, `homedir`) so each refusal is tested
 * with the failure it stands for.
 */
function pointerState({
  platformName = process.platform, env = process.env, fsImpl = fs, homedir = os.homedir(),
  driveType = windowsDriveType, volumeSerial = volumeId.windowsVolumeSerial,
} = {}) {
  const endpoints = require('./update_endpoints');
  if ((env[endpoints.HOME_ENV] || '').trim()) return { state: 'none' };
  let pointer = '';
  const pointerFile = endpoints.homePointerFile({ platformName, env });
  try {
    pointer = endpoints.readPointer(pointerFile, fsImpl);
  } catch (err) {
    // No pointer file is "no choice recorded". A file that is THERE but cannot
    // be read (an antivirus scan holds it, a permission) is not: treating it as
    // "none" silently switched to the default folder, and a second 2.5 GB
    // install followed. Nothing is decided on what could not be read.
    if (err && (err.code === 'ENOENT' || err.code === 'ENOTDIR')) return { state: 'none' };
    return {
      state: 'unreachable', pointer: null, pointerFile, kind: 'unreadable',
      why: `could not read ${pointerFile}: ${err && err.code}`,
    };
  }
  if (!pointer || !path.isAbsolute(pointer)) return { state: 'none' };

  try {
    fsImpl.statSync(pointer);
    return { state: 'present', pointer };
  } catch (err) {
    if (!err || (err.code !== 'ENOENT' && err.code !== 'ENOTDIR')) {
      return { state: 'unreachable', pointer, why: `stat failed: ${err && err.code}` };
    }
  }

  // The pointer may ITSELF be a link whose target went away (stat follows it
  // and fails, lstat finds the link): that is a medium that is not there, not
  // a deleted folder.
  try {
    fsImpl.lstatSync(pointer);
    return { state: 'unreachable', pointer, why: 'the folder is a link that no longer leads anywhere' };
  } catch (err) {
    if (!err || (err.code !== 'ENOENT' && err.code !== 'ENOTDIR')) {
      return { state: 'unreachable', pointer, why: `lstat failed: ${err && err.code}` };
    }
  }

  // The nearest ancestor that exists, by lstat: a link is seen AS a link.
  let dir = path.dirname(pointer);
  let ancestor = null;
  for (let guard = 0; guard < 64 && !ancestor; guard++) {
    try {
      ancestor = { dir, stat: fsImpl.lstatSync(dir) };
    } catch (err) {
      if (!err || (err.code !== 'ENOENT' && err.code !== 'ENOTDIR')) {
        return { state: 'unreachable', pointer, why: `lstat failed: ${err && err.code}` };
      }
      const up = path.dirname(dir);
      if (up === dir) return { state: 'unreachable', pointer, why: 'no ancestor answers' };
      dir = up;
    }
  }
  if (!ancestor) return { state: 'unreachable', pointer, why: 'no ancestor answers' };
  if (ancestor.stat.isSymbolicLink()) {
    return { state: 'unreachable', pointer, why: 'a link on the way no longer leads anywhere' };
  }
  if (!ancestor.stat.isDirectory()) return { state: 'unreachable', pointer, why: 'not a directory' };

  if (platformName === 'win32') {
    // A drive letter can be handed to a different disk: a folder recorded as
    // E:\\Orienta is "gone" once E: is another disk, although nobody deleted it.
    // Two answers are asked for, and ANY failure of either means "cannot tell":
    //   1. the volume record written when the folder was chosen (volume_id.js):
    //      the same volume must answer for the letter now. This is what
    //      identifies an external disk -- DriveInfo calls a USB hard disk Fixed,
    //      like the system disk, so the type alone cannot.
    //   2. the drive type: only a fixed disk answers for its folders.
    // KNOWN LIMIT: a pointer written before the record existed has no record, so
    // for it only the type counts, and a re-assigned EXTERNAL hard disk still
    // reads as deleted. It is retired as home.txt.previous, never lost.
    const driveRoot = path.parse(ancestor.dir).root;
    if (!sameDriveAs(driveRoot, homedir)) {
      const pointerFile = endpoints.homePointerFile({ platformName, env });
      const record = volumeId.readVolumeRecord(pointerFile, fsImpl);
      if (record && sameDriveAs(record.root, pointer)) {
        let now = null;
        try { now = volumeSerial(driveRoot); } catch { now = null; }
        if (!now || String(now).toUpperCase() !== String(record.serial).toUpperCase()) {
          return { state: 'unreachable', pointer, why: `the volume on ${driveRoot} is not the one the folder was recorded on` };
        }
      }
      let type = null;
      try { type = driveType(driveRoot); } catch { type = null; }
      if (type !== 'Fixed') {
        return { state: 'unreachable', pointer, why: `the drive ${driveRoot} is ${type || 'of unknown type'}, not a fixed disk` };
      }
    }
  } else {
    try {
      const rel = path.relative(homedir, ancestor.dir);
      const inside = rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
      if (!inside || fsImpl.lstatSync(homedir).dev !== ancestor.stat.dev) {
        return { state: 'unreachable', pointer, why: 'outside the home volume' };
      }
    } catch (err) {
      return { state: 'unreachable', pointer, why: `home unreadable: ${err && err.code}` };
    }
  }
  return { state: 'stale', pointer };
}

/**
 * The name an old pointer is kept under: home.txt.previous, and when that is
 * taken home.txt.previous.2, .3 ... -- a second stale event must not overwrite
 * the record of where the first folder was.
 */
function freeCopyName(pointerFile, { exists = fs.existsSync } = {}) {
  let name = `${pointerFile}.previous`;
  for (let n = 2; exists(name) && n < 1000; n++) name = `${pointerFile}.previous.${n}`;
  return name;
}

/**
 * Settle a pointer at startup, before anything reads the data folder.
 *
 * Returns `{action, pointer, pointerFile, previousFile, home}`:
 *   none        nothing to do
 *   fellBack    the pointer named a folder that was deleted. The pointer file
 *               is NOT deleted: it is renamed to `home.txt.previous` (or the
 *               next free `.previous.N`, never over an earlier copy), so the
 *               old setting can be read back, and the default folder is used.
 *   unreachable the pointer is kept; the caller explains.
 */
function reconcileHomePointer({
  platformName = process.platform, env = process.env, fsImpl = fs, homedir = os.homedir(),
  driveType = windowsDriveType, volumeSerial = volumeId.windowsVolumeSerial,
} = {}) {
  const endpoints = require('./update_endpoints');
  const found = pointerState({ platformName, env, fsImpl, homedir, driveType, volumeSerial });
  const pointerFile = endpoints.homePointerFile({ platformName, env });
  if (found.state === 'stale') {
    const previousFile = freeCopyName(pointerFile, { exists: (p) => fsImpl.existsSync(p) });
    try {
      fsImpl.renameSync(pointerFile, previousFile);
      try { fsImpl.rmSync(volumeId.volumeRecordFile(pointerFile), { force: true }); } catch { /* none */ }
    } catch (err) {
      // The old setting could not be kept, so it is not given up either: the
      // pointer stays, and -- see mustNotCreateHome -- nothing is created at
      // the folder it names, which would rebuild the very chain this is about.
      return {
        action: 'unreachable', kind: 'retire', pointer: found.pointer, pointerFile,
        home: found.pointer, why: `could not keep the old setting: ${err && err.code}`,
      };
    }
    return {
      action: 'fellBack', pointer: found.pointer, pointerFile, previousFile,
      home: endpoints.orientaHome({ platformName, env }),
    };
  }
  if (found.state === 'unreachable') {
    return {
      action: 'unreachable', kind: found.kind || 'drive', pointer: found.pointer, pointerFile,
      home: found.pointer, why: found.why,
    };
  }
  return { action: 'none', pointer: found.pointer || null, pointerFile, home: null };
}

/**
 * What the shell does about the pointer when it starts: nothing in development,
 * and in a packaged build whatever `reconcile` decides. Returns the notice the
 * wizard shows, or null. A failure to check is reported to `onError` and is
 * never the reason the application does not start.
 *
 * A function of its inputs, not a block of main.js, so that "packaged builds
 * only" and "the notice carries the file the old setting was kept in" are tested
 * by running it.
 */
function settleHomeAtStartup({
  isPackaged, reconcile = reconcileHomePointer, onError = () => {},
} = {}) {
  if (!isPackaged) return null;
  let settled;
  try {
    settled = reconcile();
  } catch (err) {
    onError(err);
    // A check that FAILED is not a check that found nothing: with no notice the
    // shell went on to create the recorded folder -- the chain it exists to
    // stop -- so the failure itself blocks that, and the page says so.
    return {
      code: 'unreachable', kind: 'check', dir: null, pointerFile: null, previousFile: null,
      why: String((err && err.message) || err),
    };
  }
  if (!settled || settled.action === 'none') return null;
  // A pointer that named the DEFAULT folder itself falls back to the very same
  // place: "the folder chosen earlier ... the default folder instead" would be
  // nonsense there.
  let code = 'unreachable';
  if (settled.action === 'fellBack') {
    code = sameFolder(settled.pointer, settled.home) ? 'staleDefault' : 'staleFallback';
  }
  return {
    code,
    dir: settled.pointer,
    pointerFile: settled.pointerFile,
    previousFile: settled.previousFile,
    why: settled.why,
    // drive: not connected; retire: the old setting could not be kept;
    // unreadable: the pointer file could not be read at all
    kind: settled.kind,
  };
}

/**
 * The lines the shell log gets about the pointer at startup. The log opens
 * AFTER the check ran, so what the check found (or why it failed) is carried
 * here and written once the log exists.
 */
function startupLogLines({ notice, checkError, home }) {
  const lines = [];
  if (checkError) {
    lines.push(`Could not check the recorded data folder: ${(checkError && checkError.message) || checkError}`);
  }
  if (notice && notice.code === 'staleDefault') {
    lines.push(`The recorded data folder ${notice.dir} is the default folder and no longer exists; `
      + `it will be set up again there, and the pointer ${notice.pointerFile} was renamed to ${notice.previousFile}`);
  } else if (notice && notice.code === 'staleFallback') {
    lines.push(`The recorded data folder ${notice.dir} no longer exists; using the default folder ${home} `
      + `and renaming the pointer ${notice.pointerFile} to ${notice.previousFile}`);
  } else if (notice && notice.code === 'unreachable' && notice.kind === 'unreadable') {
    lines.push(`The data folder pointer ${notice.pointerFile} could not be read, so nothing is decided `
      + `and no folder is created [${notice.why}]`);
  } else if (notice && notice.code === 'unreachable') {
    lines.push(`The recorded data folder ${notice.dir} is not reachable (drive not connected?); the pointer is kept`
      + (notice.why ? ` [${notice.why}]` : ''));
  }
  return lines;
}

/**
 * Must the shell leave the recorded data folder alone at startup -- create
 * nothing in it? True while the pointer could not be proven stale. Creating
 * `<home>/electron` there would rebuild a deleted chain when its retirement
 * failed, and would write into an unmounted mount point on the system disk.
 */
function mustNotCreateHome(notice) {
  return Boolean(notice && notice.code === 'unreachable');
}

/**
 * The English detail shown under the translated sentence when the data folder
 * cannot be used. For a recorded folder on a drive that is not connected the
 * translated sentence already says what to do (connect it, or delete the
 * pointer file); the detail must not contradict it with the advice for a
 * different cause, so it only states the fact.
 */
function unusableHomeReason({ home, error, notice }) {
  if (notice && notice.code === 'unreachable' && notice.kind === 'check') {
    return `Orienta could not check where its data folder is (${notice.why}).`;
  }
  if (notice && notice.code === 'unreachable' && notice.kind === 'unreadable') {
    return `Orienta cannot read ${notice.pointerFile} (${notice.why}).`;
  }
  if (notice && notice.code === 'unreachable' && notice.kind === 'retire') {
    return `The data folder ${notice.dir} no longer exists and ${notice.pointerFile} could not be updated (${notice.why}).`;
  }
  if (notice && notice.code === 'unreachable') {
    return `The recorded data folder ${notice.dir} is not reachable (${error}).`;
  }
  return `Orienta cannot write to ${home} (${error}). `
    + 'This is usually a drive that is not connected, or a user folder kept '
    + 'on a network share. Set ORIENTA_HOME to a folder on this computer '
    + 'and start Orienta again.';
}

/** The part of the notice `setup:probe` hands to the wizard: only an unreachable folder. */
function unreachableForProbe(notice) {
  return notice && notice.code === 'unreachable'
    ? { dir: notice.dir, pointerFile: notice.pointerFile, kind: notice.kind || 'drive', why: notice.why } : null;
}

module.exports = {
  windowsDriveType,
  mustNotCreateHome,
  unusableHomeReason,
  startupLogLines,
  isOrientaHome,
  findExistingHome,
  decidePick,
  sameFolder,
  settleHomeAtStartup,
  unreachableForProbe,
  endsWithOrienta,
  homeFromPick,
  pickerStart,
  pointerState,
  reconcileHomePointer,
};
