/**
 * Removing Orienta's data folder, on the platforms that have no uninstaller.
 *
 * Windows asks these questions in its uninstaller (electron/nsis/uninstall.nsh).
 * macOS and Linux have nothing of the sort: the user drags the app to the Trash
 * or deletes an AppImage, and two to eight gigabytes stay behind for ever, with
 * the crystal library inside them. This is the same decision, taken inside the
 * application instead.
 *
 * It is written as plan-then-execute, and the planning half is pure, because
 * the failure mode here is not "it did not work" -- it is "it deleted the
 * wrong thing", and that cannot be tested after the fact. Every rule below is
 * the NSIS uninstaller's, kept deliberately in step with it:
 *
 *   * only a folder that is recognisably ours (the marker the setup writes, or
 *     an older install's runtime/VERSION plus a second sign of our own);
 *   * never a repository, never a home directory, never a filesystem root;
 *   * never through a symlink: a link is unlinked, never followed;
 *   * only the entries we created -- anything the user put there stays;
 *   * the crystal library only on a separate, explicit yes;
 *   * the pointer file last, so an interrupted removal can be resumed.
 */
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const platform = require('./platform.js');

/**
 * Top-level entries the installer creates. Nothing else is touched.
 *
 * The macOS names come from `platform.js` rather than being written out here.
 * They were written out here once, from the SPEC, and the implementation then
 * chose different ones: this list carried `mamba-root`, which nothing ever
 * creates, and lacked `envs`, which holds the conda environment. So "remove
 * all Orienta data" on a Mac left the largest thing on disk — a 2-5 GB
 * environment — and reported it as the user's own files it had kindly kept.
 */
const OURS = [
  'electron', 'logs', 'setup-tmp', 'python', 'runtime', 'pending', 'pending.json',
  '.orienta-home', '.python_path', '.install_mode', '.install_incomplete',
  // package_sync.js: what the packages were brought to, an update in flight, and
  // the last failure of one. Left behind, "remove all data" would keep files
  // that name a lock for packages that are no longer there.
  '.packages_lock.json', '.packages_sync.json', '.packages_sync_failed.json',
  platform.CONDA_ROOT_DIR, platform.CONDA_ENVS_DIR,
];

/** Inside runtime/, this one is the user's own data. */
const LIBRARY = path.join('runtime', 'Database');

const REASONS = {
  noFolder: 'noFolder',
  notOurs: 'notOurs',
  isRepo: 'isRepo',
  tooShallow: 'tooShallow',
  isHome: 'isHome',
  isLink: 'isLink',
};

function lstatOrNull(target, io) {
  try { return io.lstatSync(target); } catch { return null; }
}

function exists(target, io) {
  return lstatOrNull(target, io) !== null;
}

/**
 * May this folder be removed, and what is inside it that is ours?
 *
 * Returns `{ ok, reason, entries, foreign, library }`. `ok: false` never
 * carries a list: a caller must not be able to delete something we refused.
 */
function planRemoval(home, io = fs, homedir = os.homedir()) {
  const plan = { ok: false, reason: null, home, entries: [], foreign: [], library: null };
  if (!home || !path.isAbsolute(home)) {
    plan.reason = REASONS.notOurs;
    return plan;
  }
  const resolved = path.resolve(home);

  // The rules that depend only on the PATH come first, so they hold whether
  // or not the folder is there: a refusal must not turn into a different
  // refusal because something was mounted or unmounted.
  //
  // Never a root, never the home directory, never the top of a mounted
  // volume -- /Volumes/Backup and /mnt/data are two components and would
  // otherwise pass. (Windows refuses share roots the same way.)
  const parts = resolved.split(path.sep).filter(Boolean);
  if (parts.length < 2) { plan.reason = REASONS.tooShallow; return plan; }
  // Read from the path as GIVEN, not as resolved: on a Windows machine
  // path.resolve('/Volumes/Backup') becomes 'C:\Volumes\Backup' and the rule
  // would be untestable anywhere but on a Mac -- which is where it matters.
  const posix = home.split('\\').join('/');
  if (posix.startsWith('/')) {
    const posixParts = posix.split('/').filter(Boolean);
    if (posixParts.length < 3 && ['Volumes', 'mnt', 'media', 'run'].includes(posixParts[0])) {
      plan.reason = REASONS.tooShallow;
      return plan;
    }
  }
  if (homedir && path.resolve(homedir) === resolved) { plan.reason = REASONS.isHome; return plan; }

  // A link is never followed. Removing "the folder" would then remove
  // somebody else's, and the link would survive pointing at the wreckage.
  const stat = lstatOrNull(resolved, io);
  if (!stat) { plan.reason = REASONS.noFolder; return plan; }
  if (stat.isSymbolicLink()) { plan.reason = REASONS.isLink; return plan; }
  if (!stat.isDirectory()) { plan.reason = REASONS.notOurs; return plan; }

  // A checkout is never a data folder, however many markers it carries.
  if (exists(path.join(resolved, '.git'), io)) { plan.reason = REASONS.isRepo; return plan; }

  // Recognisably ours: the marker the setup writes, or an older install's
  // runtime/VERSION together with a second sign of our own.
  let ours = exists(path.join(resolved, '.orienta-home'), io);
  if (!ours && exists(path.join(resolved, 'runtime', 'VERSION'), io)) {
    ours = exists(path.join(resolved, '.python_path'), io)
      || exists(path.join(resolved, 'python'), io);
  }
  if (!ours) { plan.reason = REASONS.notOurs; return plan; }

  let names = [];
  try { names = io.readdirSync(resolved); } catch { names = []; }
  plan.entries = OURS.filter((name) => names.includes(name));
  plan.foreign = names.filter((name) => !OURS.includes(name) && !/^python\.old-/.test(name));
  plan.entries.push(...names.filter((name) => /^python\.old-/.test(name)));

  // The program folder may be a LINK, to a developer checkout for instance.
  // Emptying it would then delete somebody's work outside this folder
  // entirely; the Windows uninstaller removes the link and says so.
  const runtime = path.join(resolved, 'runtime');
  const runtimeStat = lstatOrNull(runtime, io);
  plan.runtimeIsLink = Boolean(runtimeStat && runtimeStat.isSymbolicLink());

  // The library's name as the FILESYSTEM spells it. APFS and NTFS are
  // case-insensitive, so `exists()` finds "database" while a later
  // `entry === 'Database'` does not -- and the library the user asked to keep
  // would be deleted while the report said it was kept.
  plan.library = null;
  plan.libraryIsLink = false;
  if (!plan.runtimeIsLink) {
    let runtimeNames = [];
    try { runtimeNames = io.readdirSync(runtime); } catch { runtimeNames = []; }
    const realName = runtimeNames.find((n) => n.toLowerCase() === 'database');
    if (realName) {
      const library = path.join(runtime, realName);
      plan.libraryName = realName;
      const libStat = lstatOrNull(library, io);
      plan.libraryIsLink = Boolean(libStat && libStat.isSymbolicLink());
      // An empty library is nothing to warn about; Windows asks only when
      // there is something in it.
      let contents = [];
      try { contents = plan.libraryIsLink ? ['link'] : io.readdirSync(library); } catch { contents = []; }
      if (contents.length > 0) plan.library = library;
    }
  }
  plan.ok = true;
  return plan;
}

/**
 * Carry out a plan.
 *
 * `removeLibrary` is a separate argument rather than part of the plan so that
 * the question the user answered and the thing that happens cannot drift
 * apart. The pointer file goes last: until it is gone, a resumed removal can
 * still find the folder.
 */
function removeData(home, { removeLibrary = false, io = fs, homedir = os.homedir() } = {}) {
  const plan = planRemoval(home, io, homedir);
  if (!plan.ok) return { ok: false, reason: plan.reason, removed: [], kept: [] };

  const removed = [];
  const kept = [...plan.foreign];
  const failed = [];
  const rm = (target) => {
    const stat = lstatOrNull(target, io);
    if (!stat) return;
    try {
      // Measured, not assumed: fs.rmSync unlinks a symlink and does not walk
      // into it, even with recursive -- so the link is removed and its target
      // is untouched. The branch is kept because it states that intent where
      // the next person will look, and because a future implementation that
      // shells out to `rm -rf` WOULD follow a trailing slash.
      if (stat.isSymbolicLink() || !stat.isDirectory()) io.rmSync(target, { force: true });
      else io.rmSync(target, { recursive: true, force: true });
      removed.push(target);
    } catch (err) {
      failed.push({ target, error: String(err && err.message ? err.message : err) });
    }
  };

  const resolved = path.resolve(home);
  const marker = path.join(resolved, '.orienta-home');

  let libraryKept = false;
  for (const name of plan.entries) {
    if (name === '.orienta-home') continue;          // last, see below
    const target = path.join(resolved, name);
    if (name === 'runtime' && plan.runtimeIsLink) {
      // A link is removed as a link, never enumerated. What actually prevents
      // the accident is the PLAN: a linked runtime reports no library, so the
      // branch below that walks the folder is never taken (checked by
      // mutating each half in turn). This branch states the rule where the
      // next person will look for it.
      rm(target);
      continue;
    }
    if (name === 'runtime' && plan.library && !removeLibrary) {
      let inner = [];
      try { inner = io.readdirSync(target); } catch { inner = []; }
      for (const entry of inner) {
        if (entry === plan.libraryName) continue;    // the name the disk uses
        rm(path.join(target, entry));
      }
      kept.push(plan.library);
      libraryKept = true;
      continue;
    }
    rm(target);
  }

  // The marker is what makes this folder recognisable. Removing it while
  // something of ours is still here would leave the rest unreachable: a later
  // attempt is told the folder is not Orienta's.
  const finished = failed.length === 0 && !libraryKept;
  if (finished) rm(marker);

  return {
    ok: true,
    reason: null,
    removed,
    kept,
    failed,
    finished,
    // A linked library is NOT deleted: the link goes, the files it points at
    // stay. Reporting it as removed would be the opposite of the truth.
    libraryRemoved: Boolean(removeLibrary && plan.library && !plan.libraryIsLink),
    libraryWasLink: Boolean(removeLibrary && plan.libraryIsLink),
  };
}

module.exports = { planRemoval, removeData, OURS, LIBRARY, REASONS };
