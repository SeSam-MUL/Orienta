/**
 * Bringing an installation's Python packages to the lock that shipped with the
 * program files, on the first start after an update.
 *
 * WHY THIS EXISTS. Installing a newer .exe over an existing installation
 * replaces the shell, and `bundled_update.js` brings `runtime/` (the program
 * files, including the lock files) to the new release. Nothing ever touched
 * `python/`: the packages stayed exactly what the first-install wizard put
 * there, for ever. What the 0.4.7 locks move against a 0.4.6 installation:
 * kikuchipy 0.11.3 -> 0.13.1, orix 0.14.1 -> 0.15.0, PyEBSDIndex 0.3.9.1 ->
 * 0.3.10.1, and nothing else (the locks carry threadpoolctl 3.7.0, which every
 * 0.4.6 installation already has; `tests/test_lock_delta_budget.py` holds it).
 *
 * THE SHAPE is the one `bundled_update.js` set: decisions are pure functions of
 * data, every side effect goes through an injected runner, and the order in
 * `main.js` is pinned by a test that reads `main.js`. Nothing here imports
 * Electron.
 *
 * WHAT IT DOES, in one paragraph. It asks PIP what it would change
 * (`pip install --dry-run --report`), not what a stored list says -- that works
 * for every installation history and for the Windows locks, which pin only the
 * top-level requirements. An empty answer ends the matter. A non-empty answer
 * is checked against a guard (nothing that is PyTorch, NumPy, SciPy or CUDA; the
 * size of a release's delta is held by `tests/test_lock_delta_budget.py`, against
 * the committed locks, before anything ships), then the wizard's own `pip install -r <lock>` runs, then the
 * result is VERIFIED (versions equal the lock's, the three scientific libraries
 * import), and only then is it recorded. pip downloads everything before it
 * writes, so an unreachable index leaves the environment byte-identical; the
 * window in which files are being replaced is seconds long and is covered by an
 * in-flight marker that makes the next start re-run and re-verify.
 *
 * WHAT IT NEVER DOES: block the start for a network problem. A failure is a
 * log line, one dialog per (lock digest, reason), and the previous packages --
 * which this release also supports. The one exception is an environment that
 * fails verification twice: that state is untested, so the repair wizard is
 * shown and the backend is not started.
 */

const fsDefault = require('node:fs');
const pathDefault = require('node:path');
const crypto = require('node:crypto');
const childProcess = require('node:child_process');

/** Lazy, like `bundled_update.js`: the installer pulls in the whole wizard, and
 *  it in turn records through this module at the end of a first install. */
function installer() { return require('./installer'); }
function macosEnv() { return require('./macos_env'); }
function macosSync() { return require('./package_sync_macos'); }
function platformModule() { return require('../platform.js'); }

// --------------------------------------------------------------------------
// the three files this module owns in <home>
// --------------------------------------------------------------------------

const SCHEMA = 1;
/** What the installed packages were last brought to. */
const RECORD_FILE = '.packages_lock.json';
/** Present while a sync is running: written before pip writes, removed after
 *  the result is verified. A marker on disk at start means a sync was
 *  interrupted and the environment is not to be trusted. */
const MARKER_FILE = '.packages_sync.json';
/** The last failure, so a dialog is shown once and a dead network is not asked
 *  about at every start. */
const FAILURE_FILE = '.packages_sync_failed.json';
/** Every name that has to be known to the uninstallers and the foreign-entries
 *  check. A test holds the three lists to this one. */
const OWN_FILES = [RECORD_FILE, MARKER_FILE, FAILURE_FILE];

const SKIP_ENV = 'ORIENTA_SKIP_PACKAGE_SYNC';

const MB = 1024 * 1024;
/** Below this, pip may fail while writing, which is the one place it can leave
 *  a half-changed package behind. */
const MIN_FREE_BYTES = 500 * MB;
/** Packages whose replacement is never a silent affair: they are large, they
 *  hold native libraries a running process has locked, or they are the part of
 *  the stack where a mismatch is silent (CUDA, BLAS). */
const GUARDED_NAMES = [
  'torch', 'torchvision', 'numpy', 'scipy',
  /^cupy/, /^nvidia-/, /^triton/,
];

const NETWORK_FAILURES_BEFORE_BACKOFF = 3;
const BACKOFF_MS = 24 * 60 * 60 * 1000;

const TIMEOUT_MS = {
  // Cheap when the index is reachable (4 s measured), and a firewall that
  // black-holes packets must not hold the window for minutes.
  dryRun: 20 * 1000,
  install: 10 * 60 * 1000,
  versions: 30 * 1000,
  imports: 120 * 1000,
  check: 60 * 1000,
};

/** What `import` must be able to do for the update to count as done. */
const IMPORT_MODULES = ['kikuchipy', 'orix', 'pyebsdindex'];

// --------------------------------------------------------------------------
// small pure helpers
// --------------------------------------------------------------------------

/** PEP 503 normal form: how pip, the report and the lock may each spell it. */
function canonicalName(name) {
  return String(name).toLowerCase().replace(/[-_.]+/g, '-');
}

/**
 * SHA-256 of the lock text with line endings normalised to LF.
 *
 * This repository has CRLF/LF differences between trees and between a zip and a
 * checkout; the digest answers "is this the same lock", and a line ending is
 * not a different lock.
 */
function lockDigest(text) {
  return crypto.createHash('sha256')
    .update(String(text).replace(/\r\n/g, '\n'), 'utf8')
    .digest('hex');
}

function optedOut(env = process.env) {
  const raw = String((env && env[SKIP_ENV]) || '').trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
}

/** `gpu` or `cpu`; null when `.install_mode` says nothing at all.
 *  Anything else is cpu, as `lockFileFor` already treats it. */
function modeFrom(text) {
  const raw = String(text == null ? '' : text).trim().toLowerCase();
  if (!raw) return null;
  return raw === 'gpu' ? 'gpu' : 'cpu';
}

/** Is `child` inside `parent`? Path arithmetic only; no filesystem. */
function isInside(parent, child, path = pathDefault) {
  if (!parent || !child) return false;
  const rel = path.relative(path.resolve(parent), path.resolve(child));
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SAFE_VERSION = /^[A-Za-z0-9][A-Za-z0-9._+!-]*$/;

/** A requirement spec we are willing to hand to pip: `name==version`, nothing
 *  that can be read as an option or a URL. The marker file is ours, but it is
 *  also a file on a user's disk. */
function specFor(name, version) {
  if (!SAFE_NAME.test(String(name)) || !SAFE_VERSION.test(String(version))) return null;
  return `${name}==${version}`;
}

// --------------------------------------------------------------------------
// the lock: versions it pins
// --------------------------------------------------------------------------

/**
 * `{name -> version}` for every `name==version` line of a pip lock, canonical
 * names, extras and comments dropped.
 *
 * The CPU lock names torch twice, once per `platform_system` marker. The two
 * simple markers it uses are evaluated for `platformSystem`; any other marker is
 * ignored (the line counts). That is enough to answer "which version does this
 * lock want for kikuchipy", which is the only thing the verification asks.
 */
function parseLockVersions(text, platformSystem = 'Windows') {
  const out = new Map();
  for (const raw of String(text || '').split(/\r?\n/)) {
    const noComment = raw.split('#')[0].trim();
    if (!noComment || noComment.startsWith('-')) continue;
    const [requirement, marker = ''] = noComment.split(';');
    const m = /^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s]+)/.exec(requirement.trim());
    if (!m) continue;
    const system = /platform_system\s*(==|!=)\s*["']([^"']+)["']/.exec(marker);
    if (system) {
      const equal = system[2] === platformSystem;
      if ((system[1] === '==') !== equal) continue;
    }
    out.set(canonicalName(m[1]), m[2]);
  }
  return out;
}

function platformSystemOf(platformName) {
  return { win32: 'Windows', darwin: 'Darwin', linux: 'Linux' }[platformName] || 'Windows';
}

// --------------------------------------------------------------------------
// the delta
// --------------------------------------------------------------------------

/**
 * What pip would install, from `pip install --dry-run --report`.
 *
 * `[{name, version, url, sha256}]` sorted by name. Throws on a report that does
 * not look like one: a half-written file after a kill must not read as "nothing
 * to do".
 */
function deltaFromReport(report) {
  if (!report || typeof report !== 'object' || !Array.isArray(report.install)) {
    throw new Error('the pip report has no install list');
  }
  const out = report.install.map((item) => {
    const meta = item && item.metadata;
    if (!meta || !meta.name || !meta.version) {
      throw new Error('the pip report names a package without a name or version');
    }
    const info = item.download_info || {};
    const archive = info.archive_info || {};
    return {
      name: canonicalName(meta.name),
      version: String(meta.version),
      url: info.url || null,
      sha256: (archive.hashes && archive.hashes.sha256) || null,
    };
  });
  return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/**
 * May this delta be applied without asking anybody?
 *
 * By NAME: a release that moves torch, numpy or the CUDA libraries is never a
 * quiet affair. There is no size rule at run time: the size is a property of the
 * release, and `tests/test_lock_delta_budget.py` holds it against the committed
 * locks before the release is built, where a number can still be acted on.
 */
function guardDelta(delta) {
  const names = delta.map((d) => d.name);
  const guarded = names.filter((n) => GUARDED_NAMES.some((g) => (
    g instanceof RegExp ? g.test(n) : g === n)));
  if (guarded.length) {
    return {
      ok: false,
      reason: 'guard',
      detail: `the update would replace ${guarded.join(', ')}, which is never done without being asked`,
      guarded,
    };
  }
  return { ok: true };
}

// --------------------------------------------------------------------------
// argument builders (everything after the interpreter)
// --------------------------------------------------------------------------

const NO_VERSION_CHECK = '--disable-pip-version-check';

/** pip's own question: what would you do for this lock? Capped hard, because
 *  it runs at start-up. */
function dryRunArgs({ mode, lockPath, reportPath }) {
  return [
    '-m', 'pip', 'install', '--dry-run', '--report', reportPath, '--quiet',
    NO_VERSION_CHECK, '--retries', '1', '--timeout', '8',
    ...installer().pipIndexArgs(mode),
    '-r', lockPath,
  ];
}

/** The wizard's own command (`installer.js`, step 4), with a bounded patience. */
function installArgs({ mode, lockPath }) {
  return [
    '-m', 'pip', 'install', NO_VERSION_CHECK, '--retries', '2', '--timeout', '15',
    ...installer().pipIndexArgs(mode),
    '-r', lockPath,
  ];
}

/** Put exactly these packages back, whatever pip believes is installed: the
 *  repair for an interrupted or failed verification. `--no-deps` because the
 *  specs are the whole delta already. */
function reinstallArgs({ mode, specs }) {
  return [
    '-m', 'pip', 'install', NO_VERSION_CHECK, '--retries', '2', '--timeout', '15',
    '--force-reinstall', '--no-deps',
    ...installer().pipIndexArgs(mode),
    ...specs,
  ];
}

const VERSIONS_SCRIPT = [
  'import sys, json',
  'from importlib import metadata',
  'out = {}',
  'for n in sys.argv[1:]:',
  '    try:',
  '        out[n] = metadata.version(n)',
  '    except metadata.PackageNotFoundError:',
  '        out[n] = None',
  'print(json.dumps(out))',
].join('\n');

// `-I`: the probes judge the ENVIRONMENT. Without it a PYTHONPATH, a PYTHONHOME or
// a package in the user's own site-packages (the user-site of the machine the
// installation lives on) would decide whether "import kikuchipy" works.
function versionsArgs(names) {
  return ['-I', '-c', VERSIONS_SCRIPT, ...names];
}

function importArgs() {
  return ['-I', '-c', `import ${IMPORT_MODULES.join(', ')}`];
}

function checkArgs() {
  return ['-m', 'pip', 'check', NO_VERSION_CHECK];
}

// --------------------------------------------------------------------------
// what went wrong, in four words
// --------------------------------------------------------------------------

const SPACE_PATTERNS = [
  /No space left on device/i,
  /\[Errno 28\]/,
  /\[WinError 112\]/,
  /There is not enough space on the disk/i,
  /ENOSPC/,
  /Disk quota exceeded/i,
];
const NETWORK_PATTERNS = [
  /Failed to establish a new connection/i,
  /NewConnectionError/,
  /Max retries exceeded/i,
  /Temporary failure in name resolution/i,
  /Name or service not known/i,
  /getaddrinfo failed/i,
  /Connection (?:refused|reset|aborted|timed out)/i,
  /ConnectionError/,
  /Read timed out/i,
  /ReadTimeoutError|ConnectTimeoutError/,
  /ProxyError/,
  /SSLError/,
  /Network is unreachable/i,
  /Could not fetch URL/i,
  /Remote end closed connection/i,
  /Retrying \(Retry\(/,
  /HTTP error 5\d\d|5\d\d Server Error/i,
];
const RESOLVER_PATTERNS = [
  /ResolutionImpossible/,
  /conflicting dependencies/i,
  /No matching distribution found/i,
  /Could not find a version that satisfies/i,
  /requires a different Python/i,
];

/** Words pip prints only once it is replacing files: from here on the
 *  environment may differ from the one it started with, whatever else it says. */
const WRITE_PHASE_PATTERNS = [
  /Installing collected packages/i,
  /Attempting uninstall/i,
  /Successfully uninstalled/i,
];

function reachedWritePhase(text) {
  const body = String(text || '');
  return WRITE_PHASE_PATTERNS.some((re) => re.test(body));
}

/**
 * `network | space | resolver | other` from what pip printed.
 *
 * Space is looked for first (an out-of-disk failure is never a network one, and
 * it is the one that may have left a package half-written); network before
 * resolver, because pip's last line after an unreachable index is "No matching
 * distribution found", which is true and says nothing about the cause.
 */
function classifyPipFailure({ text = '', timedOut = false } = {}) {
  if (timedOut) return 'network';
  const body = String(text);
  if (SPACE_PATTERNS.some((re) => re.test(body))) return 'space';
  if (NETWORK_PATTERNS.some((re) => re.test(body))) return 'network';
  if (RESOLVER_PATTERNS.some((re) => re.test(body))) return 'resolver';
  return 'other';
}

// --------------------------------------------------------------------------
// the records
// --------------------------------------------------------------------------

function readJsonFile(file, io) {
  let text;
  try {
    text = io.readFileSync(file, 'utf8');
  } catch (err) {
    if (err && err.code === 'ENOENT') return { present: false, data: null };
    return { present: true, data: null };
  }
  try {
    const data = JSON.parse(text);
    return { present: true, data: data && typeof data === 'object' ? data : null };
  } catch {
    return { present: true, data: null };
  }
}

/**
 * Write via a sibling temp file and a rename, so a kill leaves the old content
 * or the new, never half of one. The fast path trusts this file; a torn write
 * that parses as something else would be a wrong answer at every later start.
 *
 * The temp file lives in `setup-tmp/` beside the target (same volume, so the
 * rename stays atomic), not in the home itself: a kill between the write and the
 * rename leaves it behind, and in the home it would be a file the repair wizard
 * and "remove all data" take for somebody else's (`homeNotEmpty`). `setup-tmp`
 * is Orienta's on every list that matters.
 */
function writeJsonAtomic(file, value, io = fsDefault, path = pathDefault) {
  const tmpDir = path.join(path.dirname(file), 'setup-tmp');
  io.mkdirSync(tmpDir, { recursive: true });
  const tmp = path.join(tmpDir, `${path.basename(file)}.${process.pid}.tmp`);
  io.writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  io.renameSync(tmp, file);
}

function removeFile(file, io = fsDefault) {
  try { io.rmSync(file, { force: true }); } catch { /* absent, or cannot tell */ }
}

/**
 * The three files as the decision sees them.
 *
 *   record   the parsed object, or null (absent or unreadable)
 *   marker   null when absent; an object when present -- `{}` when it is there
 *            but unreadable, because PRESENT is the information
 *   failure  the parsed object, or null
 */
function readState(home, { fs = fsDefault, path = pathDefault } = {}) {
  const record = readJsonFile(path.join(home, RECORD_FILE), fs);
  const marker = readJsonFile(path.join(home, MARKER_FILE), fs);
  const failure = readJsonFile(path.join(home, FAILURE_FILE), fs);
  return {
    record: record.data,
    marker: marker.present ? (marker.data || {}) : null,
    failure: failure.data,
  };
}

function recordFor({ lockName, digest, mode, platformName, runtimeTag, how, now }) {
  return {
    schema: SCHEMA,
    lock: lockName,
    sha256: digest,
    mode,
    platform: platformName,
    runtimeTag: runtimeTag || null,
    syncedAt: new Date(now).toISOString(),
    how,
  };
}

/**
 * The failure record that follows `prev` after another failure of `reason`
 * against lock `digest`.
 *
 * `count` counts consecutive failures of the SAME reason against the SAME lock;
 * anything else starts again at one. `shown` carries over only with them, so a
 * different reason, or a new lock, is allowed its own dialog.
 */
function nextFailure(prev, { digest, reason, now, detail = '' }) {
  const same = Boolean(prev && prev.lockSha256 === digest && prev.reason === reason);
  return {
    schema: SCHEMA,
    lockSha256: digest,
    reason,
    detail: String(detail).slice(0, 500),
    at: new Date(now).toISOString(),
    count: same ? (Number(prev.count) || 0) + 1 : 1,
    shown: same ? Boolean(prev.shown) : false,
  };
}

/**
 * Written by the first-install wizard (and by a repair), next to `.install_mode`,
 * so that the first start after an install takes the fast path instead of asking
 * pip a question whose answer is "nothing".
 *
 * It also removes a marker and a failure record: a fresh installation
 * supersedes whatever an interrupted sync left, and the repair wizard is the
 * very thing that is shown for such a marker.
 */
function recordFirstInstall({
  home, mode, lockName, lockText, runtimeTag = null,
  platformName = process.platform, now = Date.now(), fs = fsDefault, path = pathDefault,
}) {
  const record = recordFor({
    lockName, digest: lockDigest(lockText), mode, platformName, runtimeTag, how: 'install', now,
  });
  removeFile(path.join(home, MARKER_FILE), fs);
  removeFile(path.join(home, FAILURE_FILE), fs);
  writeJsonAtomic(path.join(home, RECORD_FILE), record, fs, path);
  return record;
}

// --------------------------------------------------------------------------
// the decision
// --------------------------------------------------------------------------

/**
 * `{action, reason}` -- the whole decision, as data.
 *
 *   skip      do nothing; `reason` says why in one line for the log
 *   fastpath  the packages were brought to exactly this lock; no pip, no network
 *   dryrun    ask pip what would change, and apply it
 *   repair    a sync was interrupted; put the packages it named back and verify
 *
 * The marker comes BEFORE the fast path. A record that matches the lock beside a
 * marker means the process died between "verified" and "cleaned up", or that
 * something wrote the record too early; either way the environment is not
 * trusted until a run has verified it.
 */
function decide({
  optedOut: skipRequested = false, mode, lockDigest: digest, lockName,
  record = null, marker = null, failure = null, now = Date.now(),
}) {
  if (skipRequested) {
    return { action: 'skip', reason: `${SKIP_ENV} is set; leaving the Python packages as they are` };
  }
  if (!mode) return { action: 'skip', reason: 'no .install_mode; will not guess the flavour' };
  if (!digest) return { action: 'skip', reason: 'no lock file to compare against' };
  if (marker) {
    return { action: 'repair', reason: 'a previous update of the packages was interrupted' };
  }
  if (record && record.sha256 === digest && record.mode === mode
      && (!lockName || record.lock === lockName)) {
    return { action: 'fastpath', reason: 'the packages were brought to this lock already' };
  }
  if (failure && failure.lockSha256 === digest) {
    if (failure.reason === 'guard') {
      // The delta will not shrink by trying again. A new lock is a new digest.
      return {
        action: 'skip',
        reason: 'this lock was already refused because its update is too large or touches '
          + 'a guarded package; waiting for a different lock',
      };
    }
    if (failure.reason === 'network'
        && (Number(failure.count) || 0) >= NETWORK_FAILURES_BEFORE_BACKOFF) {
      const age = now - Date.parse(failure.at);
      if (Number.isFinite(age) && age >= 0 && age < BACKOFF_MS) {
        return {
          action: 'skip',
          reason: `${failure.count} network failures in a row; trying again once in 24 hours `
            + `(last at ${failure.at})`,
        };
      }
    }
  }
  return {
    action: 'dryrun',
    reason: record
      ? 'the lock or the flavour differs from what the packages were brought to'
      : 'no record of what the packages were brought to',
  };
}

/**
 * May a sync run here at all -- the conditions that are about the machine, not
 * about the lock.
 */
function preconditions({
  decisionMode, projectRootEnv, python, home, installMode, optedOut: skipRequested,
  path = pathDefault,
}) {
  if (skipRequested) {
    return { ok: false, reason: 'opt-out', message: `${SKIP_ENV} is set; leaving the Python packages as they are` };
  }
  if (decisionMode !== 'run') {
    return { ok: false, reason: 'not-run-mode', message: `the startup mode is ${decisionMode}, not an installed run` };
  }
  if (projectRootEnv) {
    return { ok: false, reason: 'project-root-override', message: 'ORIENTA_PROJECT_ROOT points at a checkout' };
  }
  if (!isInside(home, python, path)) {
    // A developer who pointed .python_path at their own conda environment. Its
    // packages are theirs, and the lock is not their business.
    return { ok: false, reason: 'python-outside-home', message: `the interpreter (${python}) is not inside ${home}` };
  }
  if (!modeFrom(installMode)) {
    return { ok: false, reason: 'no-install-mode', message: 'no .install_mode; will not guess the flavour' };
  }
  return { ok: true };
}

// --------------------------------------------------------------------------
// running processes: one place, with its own children
// --------------------------------------------------------------------------

const KEEP_CHARS = 64 * 1024;

/**
 * The only thing here that spawns. It keeps its OWN set of children --
 * `runSetup.children` belongs to the wizard, which `before-quit` consults only
 * while a setup is running -- and `killAll()` ends their whole process trees.
 *
 * `run()` never rejects: a failure is `{code, timedOut, ...}`, because the
 * caller classifies by what pip printed and a rejection would have to be
 * unwrapped to find it.
 */
function createRunner({
  spawn = childProcess.spawn,
  execFile = childProcess.execFile,
  platformName = process.platform,
  killTree = null,
  env = null,
} = {}) {
  const children = new Set();
  const state = { cancelled: false };

  function kill(child) {
    if (!child || !child.pid) return;
    try {
      const plan = (killTree || platformModule().killTree)(child.pid, platformName);
      if (plan.kind === 'command') {
        execFile(plan.command, plan.args, { windowsHide: true }, () => {});
      } else {
        process.kill(plan.target, plan.signal);
      }
    } catch { /* already gone */ }
    try { child.kill(); } catch { /* already gone */ }
  }

  function run(exe, args, { timeoutMs = 0, onLine = null, cwd, env: callEnv = null } = {}) {
    return new Promise((resolve) => {
      const result = {
        code: -1, signal: null, timedOut: false, cancelled: false,
        stdout: '', stderr: '', output: '', error: null,
      };
      if (state.cancelled) {
        result.cancelled = true;
        resolve(result);
        return;
      }
      let child;
      try {
        child = spawn(exe, args, {
          cwd, env: callEnv || env || process.env, windowsHide: true, detached: platformName !== 'win32',
        });
      } catch (err) {
        result.error = err.message;
        resolve(result);
        return;
      }
      children.add(child);
      const tails = { out: '', err: '' };
      const feed = (which) => (buf) => {
        const text = buf.toString();
        if (which === 'out') {
          if (result.stdout.length < 1024 * 1024) result.stdout += text;
        } else {
          result.stderr = (result.stderr + text).slice(-KEEP_CHARS);
        }
        result.output = (result.output + text).slice(-KEEP_CHARS);
        tails[which] += text;
        const lines = tails[which].split(/\r?\n/);
        tails[which] = lines.pop();
        if (onLine) for (const line of lines) if (line.trim()) onLine(line.trim());
      };
      if (child.stdout) child.stdout.on('data', feed('out'));
      if (child.stderr) child.stderr.on('data', feed('err'));
      let timer = null;
      if (timeoutMs > 0) {
        timer = setTimeout(() => { result.timedOut = true; kill(child); }, timeoutMs);
      }
      let finished = false;
      const finish = (code, signal) => {
        if (finished) return;
        finished = true;
        if (timer) clearTimeout(timer);
        children.delete(child);
        if (onLine) for (const rest of [tails.out, tails.err]) if (rest.trim()) onLine(rest.trim());
        result.code = code === null || code === undefined ? -1 : code;
        result.signal = signal || null;
        result.cancelled = state.cancelled;
        resolve(result);
      };
      child.on('error', (err) => { result.error = err.message; finish(-1, null); });
      child.on('close', finish);
    });
  }

  return {
    run,
    killAll() {
      state.cancelled = true;
      for (const child of children) kill(child);
      children.clear();
    },
    get cancelled() { return state.cancelled; },
    get size() { return children.size; },
  };
}

// --------------------------------------------------------------------------
// the orchestration
// --------------------------------------------------------------------------

function tailOf(text, lines = 6) {
  return String(text || '').trim().split(/\r?\n/).slice(-lines).join(' | ').slice(0, 600);
}

/**
 * Bring the packages to the lock, if they are not there. Never throws.
 *
 * @param opts   data about this launch
 *   home            the Orienta home
 *   python          the interpreter `.python_path` names
 *   decisionMode    `startupDecision().mode`; only 'run' syncs
 *   projectRootEnv  ORIENTA_PROJECT_ROOT, a developer's override
 *   platformName    process.platform
 *   env             process.env
 *
 * @param deps   everything with a side effect, injectable
 *   run(exe, args, {timeoutMs, onLine})    -> {code, timedOut, stdout, stderr, output, cancelled}
 *   isCancelled()                          the app is quitting
 *   log(line)
 *   freeBytes(dir)                         -> {bytes, known}
 *   onPhase('syncing')                     the waiting page's text; before pip is asked
 *   notify({reason, detail})               the one dialog; awaited
 *   syncMacos(ctx)                         tests only: replaces package_sync_macos.js
 *   macos {…}                              tests and CI only: see package_sync_macos.js
 *   runtimeTag()                           what runtime/VERSION says
 *   now(), fs, path
 *
 * @returns one of
 *   {ok: true, action: 'skip'|'fastpath'|'noop'|'synced'|'failed'|'macos', ...}
 *   {repair: true, message}     the environment is in an untested state
 *   {cancelled: true}           the app is quitting; nothing was decided
 */
async function syncPackages(opts, deps) {
  const d = {
    fs: fsDefault, path: pathDefault, now: () => Date.now(),
    log: () => {}, isCancelled: () => false,
    freeBytes: () => ({ bytes: 0, known: false }),
    onPhase: () => {}, notify: async () => {},
    syncMacos: null, runtimeTag: () => null,
    ...deps,
  };
  const say = (line) => d.log(`Package sync: ${line}`);
  try {
    return await syncInner(opts, d, say);
  } catch (err) {
    // The start must not depend on this. A marker that was written stays, so
    // the next start verifies; everything else is the previous packages.
    say(`failed unexpectedly (${err && err.message ? err.message : err}); `
      + 'starting with the current packages');
    return { ok: true, action: 'failed', reason: 'exception', detail: String(err && err.message) };
  }
}

async function syncInner(opts, d, say) {
  const { fs, path } = d;
  const {
    home, python, decisionMode, projectRootEnv,
    platformName = process.platform, env = process.env,
  } = opts;

  const skipRequested = optedOut(env);
  let installModeText = null;
  try { installModeText = fs.readFileSync(path.join(home, '.install_mode'), 'utf8'); } catch { /* absent */ }

  const pre = preconditions({
    decisionMode, projectRootEnv, python, home, installMode: installModeText,
    optedOut: skipRequested, path,
  });
  if (!pre.ok) {
    say(`skipped (${pre.reason}) — ${pre.message}`);
    return { ok: true, action: 'skip', reason: pre.reason };
  }
  const mode = modeFrom(installModeText);

  // ---- the lock this runtime shipped ---------------------------------------
  const darwin = platformName === 'darwin';
  let lockName;
  try {
    lockName = darwin ? macosEnv().MACOS_LOCK_FILE : installer().lockFileFor(mode, platformName);
  } catch (err) {
    say(`skipped — ${err.message}`);
    return { ok: true, action: 'skip', reason: 'no-lock-for-platform' };
  }
  const lockPath = path.join(home, 'runtime', lockName);
  let lockText;
  try {
    lockText = fs.readFileSync(lockPath, 'utf8');
  } catch (err) {
    say(`skipped — ${lockName} is not in the runtime (${err.code || err.message})`);
    return { ok: true, action: 'skip', reason: 'lock-missing' };
  }
  if (!darwin) {
    const verdict = installer().lockFileIsSafe(lockText);
    if (!verdict.ok) {
      // A packaging mistake, not something the user can act on: log, no dialog.
      say(`skipped — ${verdict.detail}`);
      return { ok: true, action: 'skip', reason: 'lock-unsafe' };
    }
  }
  const digest = lockDigest(lockText);
  const state = readState(home, { fs, path });
  const verdict = decide({
    optedOut: false, mode, lockDigest: digest, lockName,
    record: state.record, marker: state.marker, failure: state.failure, now: d.now(),
  });
  say(`${verdict.action} — ${verdict.reason}`);
  if (verdict.action === 'skip') return { ok: true, action: 'skip', reason: verdict.reason };
  if (verdict.action === 'fastpath') return { ok: true, action: 'fastpath' };

  const context = {
    home, python, mode, lockName, lockPath, lockText, digest, state, platformName, env,
  };

  if (darwin) {
    // micromamba, not pip: `package_sync_macos.js`. It gets everything decided
    // above and answers in the same shapes as `syncPip`. The hook is only for
    // tests; by default the real thing runs.
    const hook = d.syncMacos || ((ctx) => macosSync().syncMacos(ctx, d));
    const result = await hook({ ...context, decision: verdict, log: say });
    say(`macOS: ${JSON.stringify(result)}`);
    return result;
  }

  return syncPip(context, verdict, d, say);
}

/**
 * Record a failure, show the one dialog, and let the start go on. Shared by the
 * pip and the micromamba flow: the rule "one dialog per (lock digest, reason),
 * never one per start" is a property of the shell, not of the package manager.
 */
async function failWith({ home, digest, state, d, say }, reason, detail, { unverified = false } = {}) {
  const { fs, path } = d;
  const failureFile = path.join(home, FAILURE_FILE);
  const rec = nextFailure(state.failure, { digest, reason, now: d.now(), detail });
  say(`could not update the packages (${reason}) — ${detail}`);
  try { writeJsonAtomic(failureFile, rec, fs, path); } catch (err) { say(`could not write the failure record (${err.message})`); }
  if (!rec.shown) {
    try {
      await d.notify({ reason, detail, unverified });
      rec.shown = true;
      try { writeJsonAtomic(failureFile, rec, fs, path); } catch { /* shown twice at worst */ }
    } catch (err) {
      say(`could not show the message (${err.message})`);
    }
  }
  return { ok: true, action: 'failed', reason, detail, ...(unverified ? { unverified: true } : {}) };
}

async function syncPip(ctx, verdict, d, say) {
  const { fs, path } = d;
  const { home, python, mode, lockName, lockPath, lockText, digest, state, platformName } = ctx;
  const repairing = verdict.action === 'repair';
  const markerFile = path.join(home, MARKER_FILE);
  const failureFile = path.join(home, FAILURE_FILE);
  const recordFile = path.join(home, RECORD_FILE);
  const lockVersions = parseLockVersions(lockText, platformSystemOf(platformName));

  // pip reads PIP_* from the environment of the process that starts it: another
  // index, a proxy, a constraints file. Often the reason a sync fails or installs
  // something unexpected, and invisible in the log. Names only: an index URL can
  // carry a token.
  const pipEnv = Object.keys(ctx.env || {}).filter((k) => /^PIP_/i.test(k)).sort();
  if (pipEnv.length) {
    say(`the environment sets ${pipEnv.join(', ')}; pip will read ${pipEnv.length === 1 ? 'it' : 'them'} `
      + '(values are not logged)');
  }

  const cancelled = () => d.isCancelled();

  /** pip's own words go to the log, except "Requirement already satisfied": a
   *  full lock prints one per installed package, which would bury the dozen
   *  lines that say what actually happened. */
  const pipLine = (line) => {
    if (/^Requirement already satisfied/.test(line)) return;
    say(`  pip: ${line}`);
  };

  /** Record a failure, show the one dialog, and let the start go on. */
  const fail = (reason, detail, extra) => failWith({ home, digest, state, d, say }, reason, detail, extra);

  // ---- room to write -------------------------------------------------------
  const free = d.freeBytes(home) || { bytes: 0, known: false };
  if (free.known && free.bytes < MIN_FREE_BYTES) {
    return fail('space', `only ${(free.bytes / MB).toFixed(0)} MB free, ${MIN_FREE_BYTES / MB} MB needed`);
  }
  if (!free.known) say('free space could not be measured; going on');

  // ---- what would change ---------------------------------------------------
  // The page says "syncing" from here, not from the install: the question to pip
  // takes seconds of its own, and until now the page said "Loading the analysis
  // engine" all through it. The fast path and the skips never get here.
  d.onPhase('syncing');
  const reportPath = path.join(home, 'setup-tmp', 'package-sync-report.json');
  try { fs.mkdirSync(path.dirname(reportPath), { recursive: true }); } catch { /* run will say */ }
  removeFile(reportPath, fs);
  const dry = await d.run(
    python, dryRunArgs({ mode, lockPath, reportPath }),
    { timeoutMs: TIMEOUT_MS.dryRun, onLine: pipLine },
  );
  if (cancelled() || dry.cancelled) return { cancelled: true };

  let delta = null;
  let dryFailure = null;
  if (dry.code === 0) {
    try {
      delta = deltaFromReport(JSON.parse(fs.readFileSync(reportPath, 'utf8')));
    } catch (err) {
      dryFailure = { reason: 'other', detail: `the pip report could not be read (${err.message})` };
    }
  } else {
    dryFailure = {
      reason: classifyPipFailure({ text: `${dry.stdout}\n${dry.stderr}`, timedOut: dry.timedOut }),
      detail: dry.timedOut ? `pip did not answer within ${TIMEOUT_MS.dryRun / 1000} s`
        : (dry.error || tailOf(dry.output) || `pip exited with ${dry.code}`),
    };
  }
  removeFile(reportPath, fs);

  // What an interrupted sync said it was doing, as specs pip may be handed.
  const markerSpecs = new Map();
  if (repairing && state.marker && state.marker.to && typeof state.marker.to === 'object') {
    for (const [name, version] of Object.entries(state.marker.to)) {
      if (specFor(name, version)) markerSpecs.set(canonicalName(name), String(version));
    }
  }

  if (dryFailure) {
    if (!repairing) return fail(dryFailure.reason, dryFailure.detail);
    if (!markerSpecs.size) {
      // An interrupted update, nothing in the marker to put back, and pip cannot
      // be asked: the environment is unverified. Look at it once -- libraries that
      // import mean start (the marker stays); libraries that do not mean the
      // repair wizard, not a backend that dies on its first import.
      say(`the question to pip failed (${dryFailure.reason}) and the interrupted update named nothing; `
        + 'checking whether the libraries import');
      const probe = await d.run(python, importArgs(), { timeoutMs: TIMEOUT_MS.imports });
      if (cancelled() || probe.cancelled) return { cancelled: true };
      if (probe.code === 0 || probe.timedOut) {
        return fail(dryFailure.reason, `${dryFailure.detail} (the earlier update is still unverified)`,
          { unverified: true });
      }
      return needsRepair(`${dryFailure.detail}; and the libraries do not import`);
    }
    say(`the question to pip failed (${dryFailure.reason}); repairing from the interrupted update's own list`);
    delta = [];
  }

  // The packages to bring over: pip's answer, plus whatever the interrupted
  // update had named (pip may think those are fine and be wrong about it).
  const target = new Map(delta.map((x) => [x.name, x.version]));
  for (const [name, version] of markerSpecs) if (!target.has(name)) target.set(name, version);

  if (!target.size) {
    if (repairing) {
      // A marker, nothing named in it, and pip sees nothing to do: all that is
      // left to do is to look at the result.
      say('nothing named by the interrupted update; verifying only');
    } else {
      removeFile(failureFile, fs);
      writeJsonAtomic(recordFile, recordFor({
        lockName, digest, mode, platformName, runtimeTag: d.runtimeTag(), how: 'noop', now: d.now(),
      }), fs, path);
      say('nothing to change; recorded');
      return { ok: true, action: 'noop' };
    }
  }

  // ---- the guard -----------------------------------------------------------
  if (delta.length) {
    const guard = guardDelta(delta);
    if (!guard.ok) return fail(guard.reason, guard.detail);
    say(`delta: ${delta.map((x) => `${x.name} ${x.version}`).join(', ')}`);
  }
  if (repairing) {
    say(`repairing: ${[...target].map(([n, v]) => `${n} ${v}`).join(', ') || '(verification only)'}`);
  }

  const names = [...target.keys()];
  const specs = [...target].map(([n, v]) => specFor(n, v)).filter(Boolean);
  const expected = {};
  for (const [name, version] of target) expected[name] = lockVersions.get(name) || version;

  // ---- what is installed now, and the marker -------------------------------
  const before = await readVersions(d, python, names);
  if (cancelled()) return { cancelled: true };
  try {
    writeJsonAtomic(markerFile, {
      schema: SCHEMA, startedAt: new Date(d.now()).toISOString(), lockSha256: digest, mode,
      from: before.versions || {}, to: Object.fromEntries(target),
    }, fs, path);
  } catch (err) {
    // Without the marker a kill in the next few seconds would leave an
    // environment nobody knows about. Do not start what cannot be undone.
    return fail('other', `could not write ${MARKER_FILE} (${err.message})`);
  }

  // ---- the install ---------------------------------------------------------
  const installArgsList = repairing && specs.length
    ? reinstallArgs({ mode, specs })
    : installArgs({ mode, lockPath });
  say(`running: pip install ${repairing && specs.length ? specs.join(' ') : `-r ${lockName}`}`);
  const install = await d.run(python, installArgsList, {
    timeoutMs: TIMEOUT_MS.install, onLine: pipLine,
  });
  if (cancelled() || install.cancelled) {
    // The marker stays: the next start re-runs and verifies.
    return { cancelled: true };
  }

  const unchangedNow = async () => {
    const now = await readVersions(d, python, names);
    if (!before.versions || !now.versions) return false;
    return names.every((n) => before.versions[n] === now.versions[n]);
  };

  const finish = async (how) => {
    // The marker goes FIRST. Killed between the two, there is no marker and no
    // record: the next start asks pip, sees nothing to do, and records. The
    // other order leaves a record AND a marker, which is a state to explain.
    removeFile(markerFile, fs);
    removeFile(failureFile, fs);
    writeJsonAtomic(recordFile, recordFor({
      lockName, digest, mode, platformName, runtimeTag: d.runtimeTag(), how, now: d.now(),
    }), fs, path);
    say(`done (${how}); recorded`);
    return { ok: true, action: 'synced', how, packages: Object.fromEntries(target) };
  };

  /** The imports could not be judged in time. Not "broken": the packages are in
   *  place and nobody has seen them fail. Start, keep the marker (the next start
   *  verifies again), say so. */
  const unverified = (detail) => fail('other', detail, { unverified: true });

  let check;
  if (install.code === 0) {
    check = await verifyInstall(d, python, expected, say);
    if (cancelled()) return { cancelled: true };
    if (check.ok) return finish(repairing ? 'repair' : 'sync');
    if (check.unverified) return unverified(check.detail);
    say(`verification failed (${check.detail}); reinstalling the packages once`);
  } else {
    const reason = classifyPipFailure({
      text: `${install.stdout}\n${install.stderr}`, timedOut: install.timedOut,
    });
    const detail = install.timedOut ? `pip did not finish within ${TIMEOUT_MS.install / 60000} min`
      : (tailOf(install.output) || `pip exited with ${install.code}`);
    say(`pip install failed (${reason}) — ${detail}`);

    // The classification is for the record and the dialog, NEVER for the
    // question "did pip write anything". A `Retrying` warning in front of a
    // `[WinError 5] Access is denied` reads as a network failure and is a failure
    // while writing; a timeout can be the kill of a pip that was mid-write.
    // Whether the environment is still the old one is asked of the environment.
    let touched = reachedWritePhase(`${install.stdout}\n${install.stderr}`);
    if (touched) say('pip reached the write phase; checking the environment');
    if (!repairing && !touched) {
      if (await unchangedNow()) {
        // pip downloads before it writes, and this one never got that far.
        removeFile(markerFile, fs);
        return fail(reason, detail);
      }
      touched = true;
      say('the failed install changed packages; checking and repairing');
    }
    // Reached when pip wrote something, or when this IS the repair of an
    // earlier interruption -- where "pip changed nothing just now" says nothing
    // about the state the interruption left. Look at the environment itself.
    const cannotRetryNow = !touched && (reason === 'network' || reason === 'resolver');
    check = await verifyInstall(d, python, expected, say);
    if (cancelled()) return { cancelled: true };
    if (check.ok) return finish('repair');
    if (check.unverified) return unverified(check.detail);
    if (cannotRetryNow) {
      // Nothing can be reinstalled without a network. If the libraries still
      // import, the start goes ahead and the marker stays for the next try;
      // if they do not, the backend cannot start anyway.
      return startIfImportable(
        reason, `${detail} (the earlier update is still unverified)`,
        `${check.detail}; and ${reason === 'network' ? 'there is no network' : 'pip cannot resolve'} to fix it`,
      );
    }
    say(`verification failed (${check.detail}); reinstalling the packages once`);
  }

  const again = await d.run(python, reinstallArgs({ mode, specs }), {
    timeoutMs: TIMEOUT_MS.install, onLine: pipLine,
  });
  if (cancelled() || again.cancelled) return { cancelled: true };
  check = again.code === 0
    ? await verifyInstall(d, python, expected, say)
    : { ok: false, detail: `the reinstall failed (${tailOf(again.output) || `pip exited with ${again.code}`})` };
  if (cancelled()) return { cancelled: true };
  if (check.ok) return finish('repair');
  if (check.unverified) return unverified(check.detail);
  return needsRepair(check.detail);

  /** An environment nobody can fix right now and nobody has seen fail: if the
   *  libraries import, start and leave the marker for the next try; if they do
   *  not, the backend cannot start anyway. A probe that does not finish in time
   *  is not a failed import. */
  async function startIfImportable(reason, detail, brokenDetail) {
    const probe = await d.run(python, importArgs(), { timeoutMs: TIMEOUT_MS.imports });
    if (cancelled() || probe.cancelled) return { cancelled: true };
    if (probe.code === 0 || probe.timedOut) return fail(reason, detail, { unverified: true });
    return needsRepair(brokenDetail);
  }

  /** Still wrong after the one retry. Nobody tested this combination: the repair
   *  wizard replaces the whole environment. The marker STAYS, so even a user who
   *  closes the wizard is not started against it. */
  function needsRepair(detail) {
    say(`REPAIR NEEDED — ${detail}; not starting the backend`);
    return { ok: false, repair: true, detail, message: detail };
  }
}

/** `{versions: {name: version|null}}`, or `{versions: null}` when it could not be read. */
async function readVersions(d, python, names) {
  if (!names.length) return { versions: {} };
  const out = await d.run(python, versionsArgs(names), { timeoutMs: TIMEOUT_MS.versions });
  if (out.code !== 0) return { versions: null, detail: tailOf(out.output) };
  const last = out.stdout.trim().split(/\r?\n/).filter((l) => l.trim()).pop() || '';
  try {
    const parsed = JSON.parse(last);
    const versions = {};
    for (const [k, v] of Object.entries(parsed)) versions[canonicalName(k)] = v;
    return { versions };
  } catch {
    return { versions: null, detail: `unreadable: ${last.slice(0, 100)}` };
  }
}

/**
 * Is what pip wrote what the lock asked for, and does it run?
 *
 * Versions first (metadata only: cheap, and exactly what the lock states), then
 * the three imports -- `pip check` reads metadata and would pass a package whose
 * files are half-copied; an import does not. `pip check` is LOGGED and does not
 * gate: it has not been measured on the GPU environment, and a verdict nobody
 * has seen is not a gate to put in front of a start.
 */
async function verifyInstall(d, python, expected, say) {
  const names = Object.keys(expected);
  const got = await readVersions(d, python, names);
  if (!got.versions) return { ok: false, detail: `the installed versions could not be read (${got.detail || 'no answer'})` };
  const wrong = names.filter((n) => got.versions[n] !== expected[n]);
  if (wrong.length) {
    return {
      ok: false,
      detail: `installed ${wrong.map((n) => `${n} ${got.versions[n] || '(missing)'} instead of ${expected[n]}`).join(', ')}`,
    };
  }
  const imports = await d.run(python, importArgs(), { timeoutMs: TIMEOUT_MS.imports });
  if (imports.timedOut) {
    // A first import after an install can be slow (a virus scanner reading every
    // new file, a cold disk). "Did not answer in time" is not "failed": nothing
    // has been seen to be wrong, so the caller starts and verifies again later.
    return {
      ok: false,
      unverified: true,
      detail: `importing ${IMPORT_MODULES.join(', ')} did not finish within `
        + `${TIMEOUT_MS.imports / 1000} s; the packages are in place but not verified`,
    };
  }
  if (imports.code !== 0) {
    return {
      ok: false,
      detail: `importing ${IMPORT_MODULES.join(', ')} failed: ${tailOf(imports.output)}`,
    };
  }
  try {
    const check = await d.run(python, checkArgs(), { timeoutMs: TIMEOUT_MS.check });
    say(`pip check (informational): exit ${check.code} ${tailOf(check.output, 3)}`);
  } catch { /* informational */ }
  return { ok: true, detail: '' };
}

module.exports = {
  SCHEMA,
  RECORD_FILE,
  MARKER_FILE,
  FAILURE_FILE,
  OWN_FILES,
  SKIP_ENV,
  MIN_FREE_BYTES,
  GUARDED_NAMES,
  NETWORK_FAILURES_BEFORE_BACKOFF,
  BACKOFF_MS,
  TIMEOUT_MS,
  IMPORT_MODULES,
  canonicalName,
  lockDigest,
  optedOut,
  modeFrom,
  isInside,
  specFor,
  parseLockVersions,
  deltaFromReport,
  guardDelta,
  dryRunArgs,
  installArgs,
  reinstallArgs,
  versionsArgs,
  importArgs,
  checkArgs,
  classifyPipFailure,
  reachedWritePhase,
  readState,
  recordFor,
  nextFailure,
  recordFirstInstall,
  writeJsonAtomic,
  decide,
  preconditions,
  createRunner,
  syncPackages,
  failWith,
  removeFile,
  tailOf,
  verifyInstall,
  readVersions,
};
