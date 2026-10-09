/**
 * Bringing a macOS installation's conda environment to the lock that shipped
 * with the program files: the micromamba half of `package_sync.js`.
 *
 * WHY IT IS NOT `micromamba install -f <full lock>`. Measured with micromamba
 * 2.9.0 on a win-64 stand-in (2026-10-09), and not documented anywhere:
 *
 *   install -f <full lock>, existing prefix
 *       every package is listed as an install; afterwards conda-meta holds TWO
 *       records for each package that changed and `micromamba list` reports the
 *       old one. A lock-file transaction has no removal list.
 *   create -y -f <full lock>, existing prefix
 *       correct, and it deletes the prefix first (a file planted in it was
 *       gone): not atomic, 722 MB again.
 *
 * WHAT IT DOES INSTEAD, each step measured on the same stand-in:
 *
 *   1. Read the delta from the DISK: which `conda-meta/<name>-<version>-<build>`
 *      records differ from the lock's packages (same name, other file name).
 *      No stored list is trusted, so it works for every installation history,
 *      for an update that was interrupted, and for the broken state the
 *      full-lock install above leaves behind.
 *   2. Write a DELTA LOCK: the new lock's header and only those packages.
 *   3. `install --download-only -f <delta lock>`. The network phase. Nothing is
 *      linked: with the channel unreachable the prefix is byte-identical
 *      (`Download error (7)` / `(6)`, exit 1; a wrong sha256 is refused with
 *      "SHA256 doesn't match" before anything is linked).
 *   4. The marker, then `remove --force <every delta name installed>`. Exactly
 *      those records are unlinked and NOT their dependents (openssl stayed when
 *      ca-certificates went). `remove` does not accept `--platform`.
 *   5. `install -f <delta lock>`, from the package cache, no network (measured
 *      with the proxy pointed at a closed port: "Linking ...", exit 0).
 *   6. VERIFY: `micromamba list --json` shows the lock's version for each
 *      package of the delta, conda-meta holds exactly one record for each, the
 *      three scientific libraries import, and the project's own OpenMP gate
 *      (`scripts/check_runtime_health.py --gate files`) finds exactly one
 *      runtime.
 *
 * The delta is recomputed from the disk each time it is applied, and the
 * removal covers every delta name that is installed in any form, so the local
 * phase can be run again after a kill at any point (the packages are in the
 * cache, so even without a network).
 *
 * WHAT IS NOT PROVEN HERE: osx-arm64 itself. Code signing of a relinked binary
 * (openssl is in the 0.4.7 delta), the real lock, the real size. These are
 * exercised by the project's macOS build, which is not part of this repository.
 *
 * NOT DONE, ON PURPOSE: `micromamba clean`. The wizard's `cleanArgs` passes
 * `-r <root>`, and micromamba 2.9.0's `clean` rejects it ("The following
 * arguments were not expected: -r"), so the cache has never been emptied by
 * the wizard either. A working `clean -a` would also sweep every other
 * writable package cache on the machine. The delta's cache is a few MB.
 *
 * Nothing here imports Electron. Every process goes through the injected `run`.
 */

const fsDefault = require('node:fs');
const pathDefault = require('node:path');

/** Lazy, like the rest of the sync: `package_sync.js` requires this file lazily
 *  and this file needs it back. */
function sync() { return require('./package_sync'); }
function macosEnv() { return require('./macos_env'); }
function platformModule() { return require('../platform.js'); }

const CONDA_PLATFORM = 'osx-arm64';
/** Must end in `-lock.yml`: micromamba decides by the NAME (see macos_env.js). */
const DELTA_LOCK_FILE = 'orienta-macos-delta-lock.yml';
/** More than this is not "a few libraries moved". */
const MAX_DELTA_PACKAGES = 60;

const MB = 1024 * 1024;
const MAC_TIMEOUT_MS = {
  probe: 60 * 1000,
  list: 60 * 1000,
  download: 10 * 60 * 1000,
  remove: 3 * 60 * 1000,
  install: 5 * 60 * 1000,
  health: 120 * 1000,
};

/**
 * Names whose replacement is never a quiet affair on macOS, beyond what
 * `package_sync.js` already guards (torch, numpy, scipy, cupy, nvidia-*,
 * triton): the interpreter and the native runtime layer, where a mismatch is
 * silent at install time and an abort at run time -- the OpenMP runtime above
 * all, which is the reason this platform has its own environment at all.
 * conda-forge spells PyTorch `pytorch`/`libtorch`.
 */
const GUARDED_NAMES_MACOS = [
  'python', 'python_abi',
  /^pytorch/, /^libtorch/, 'torchvision',
  'llvm-openmp', '_openmp_mutex', /^libomp/, /^libiomp/,
  'libblas', 'libcblas', 'liblapack', 'liblapacke', 'libopenblas', /^libgfortran/,
  'libcxx',
];

/** A conda package name we are willing to put on a command line. */
const SAFE_CONDA_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]*$/;

// --------------------------------------------------------------------------
// the lock
// --------------------------------------------------------------------------

function unquote(value) {
  const v = String(value).trim();
  if (v.length >= 2 && (v[0] === "'" || v[0] === '"') && v[v.length - 1] === v[0]) return v.slice(1, -1);
  return v;
}

/** `ca-certificates-2026.7.22-h4c7d964_0` from the package's URL. */
function stemFromUrl(url) {
  const base = String(url || '').split(/[?#]/)[0].split('/').pop();
  return base.replace(/\.(conda|tar\.bz2)$/, '');
}

/** The version a `name-version-build` stem carries. */
function versionFromStem(stem) {
  const m = /^(.+)-([^-]+)-([^-]+)$/.exec(String(stem));
  return m ? m[2] : null;
}

/**
 * The lock, split without a YAML parser (there is none in the shipped app; see
 * `lockFileComplaint`): the header up to and including `package:`, and one text
 * block per package, each with the few scalar fields the sync reads.
 *
 * `packages` holds the packages of `condaPlatform` only; anything else in the
 * file counts in `foreign`. A pip-managed package is `unsupported`: this flow
 * knows how to move conda packages and nothing else.
 */
function parseLock(text, condaPlatform = CONDA_PLATFORM) {
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  const start = lines.findIndex((l) => /^package:\s*$/.test(l));
  if (start < 0) throw new Error('the lock file has no `package:` list');
  const header = `${lines.slice(0, start + 1).join('\n')}\n`;
  const raw = [];
  let current = null;
  for (const line of lines.slice(start + 1)) {
    if (/^- name:/.test(line)) { current = [line]; raw.push(current); } else if (current) current.push(line);
  }
  const packages = [];
  const unsupported = [];
  let foreign = 0;
  for (const block of raw) {
    while (block.length && block[block.length - 1].trim() === '') block.pop();
    const fields = {};
    fields.name = unquote(block[0].replace(/^- name:/, ''));
    for (const line of block.slice(1)) {
      const m = /^ {2}(version|manager|platform|url):\s*(.*)$/.exec(line);
      if (m && !(m[1] in fields)) fields[m[1]] = unquote(m[2]);
    }
    if (fields.manager && fields.manager !== 'conda') { unsupported.push(fields.name); continue; }
    if (fields.platform !== condaPlatform) { foreign += 1; continue; }
    packages.push({
      name: fields.name,
      version: fields.version,
      url: fields.url,
      stem: stemFromUrl(fields.url),
      text: `${block.join('\n')}\n`,
    });
  }
  return { header, packages, unsupported, foreign };
}

/** The delta lock: the new lock's header and the delta packages' blocks, in the
 *  lock's own order. */
function buildDeltaLock(parsed, names) {
  const wanted = new Set(names);
  const body = parsed.packages.filter((p) => wanted.has(p.name)).map((p) => p.text).join('');
  return parsed.header + body;
}

// --------------------------------------------------------------------------
// what is installed, and the delta
// --------------------------------------------------------------------------

/**
 * `Map(name -> [stem, ...])` from `<prefix>/conda-meta/*.json` -- file NAMES
 * only. A stem (`name-version-build`) is the package's identity, independent of
 * which channel URL spelling it came from. More than one stem for a name is the
 * duplicate-record state the full-lock install leaves behind.
 */
function readInstalled(prefix, { fs = fsDefault, path = pathDefault } = {}) {
  const out = new Map();
  for (const file of fs.readdirSync(path.join(prefix, 'conda-meta'))) {
    const m = /^(.+)-([^-]+)-([^-]+)\.json$/.exec(file);
    if (!m) continue;
    const stems = out.get(m[1]) || [];
    stems.push(file.slice(0, -'.json'.length));
    out.set(m[1], stems);
  }
  return out;
}

/**
 * What differs between the lock and the disk.
 *
 *   delta       [{name, version, stem, url, from: [stem, ...]}] sorted by name;
 *               `from` is empty for a package that is new to the environment
 *   duplicates  names with more than one record on disk
 *   unchanged   how many lock packages are installed exactly as locked
 *
 * Packages installed but not in the lock are left alone: from here we cannot
 * tell the user's own from an old release's.
 */
function computeDelta(packages, installed) {
  const delta = [];
  let unchanged = 0;
  for (const p of packages) {
    const have = installed.get(p.name) || [];
    if (have.length === 1 && have[0] === p.stem) { unchanged += 1; continue; }
    delta.push({
      name: p.name, version: p.version, stem: p.stem, url: p.url, from: [...have],
    });
  }
  delta.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const duplicates = [...installed].filter(([, stems]) => stems.length > 1).map(([n]) => n).sort();
  return { delta, duplicates, unchanged };
}

/**
 * May this delta be applied without asking anybody? The pip rules (large, or
 * torch/numpy/scipy/CUDA) plus the macOS ones above, plus a package count.
 */
function guardDelta(delta, sizes = {}) {
  const names = delta.map((d) => d.name);
  const guarded = names.filter((n) => GUARDED_NAMES_MACOS.some((g) => (
    g instanceof RegExp ? g.test(n) : g === n)));
  if (guarded.length) {
    return {
      ok: false,
      reason: 'guard',
      detail: `the update would replace ${guarded.join(', ')}, which is never done without being asked`,
      guarded,
    };
  }
  if (delta.length > MAX_DELTA_PACKAGES) {
    return {
      ok: false,
      reason: 'guard',
      detail: `the update would replace ${delta.length} packages, more than the `
        + `${MAX_DELTA_PACKAGES} that count as a few libraries moving`,
    };
  }
  return sync().guardDelta(delta.map((d) => ({ name: d.name })), sizes);
}

// --------------------------------------------------------------------------
// argument lists (everything after the micromamba executable)
// --------------------------------------------------------------------------

function checkedNames(names) {
  for (const n of names) {
    if (!SAFE_CONDA_NAME.test(String(n))) throw new Error(`"${n}" is not a package name this flow will pass to micromamba`);
  }
  return names;
}

function common({ root, prefix }) {
  return ['--no-rc', '--no-env', '-y', '-r', root, '-p', prefix];
}

function deltaLockName(lockFile) {
  if (!macosEnv().isLockFileName(lockFile)) {
    throw new Error(`micromamba only reads a file named *-lock.yml as a lock; "${pathDefault.basename(String(lockFile))}" would install nothing`);
  }
  return lockFile;
}

/** The network phase: fetch and unpack, link nothing. */
function downloadArgs({ root, prefix, lockFile, platform = CONDA_PLATFORM }) {
  return ['install', ...common({ root, prefix }), '--platform', platform, '--download-only', '-f', deltaLockName(lockFile)];
}

/** Unlink exactly these records. `remove` has no `--platform`. */
function removeArgs({ root, prefix, names }) {
  return ['remove', ...common({ root, prefix }), '--force', ...checkedNames(names)];
}

/** Link the delta from the cache. */
function installDeltaArgs({ root, prefix, lockFile, platform = CONDA_PLATFORM }) {
  return ['install', ...common({ root, prefix }), '--platform', platform, '-f', deltaLockName(lockFile)];
}

function listArgs({ root, prefix }) {
  return ['list', '--no-rc', '--no-env', '-r', root, '-p', prefix, '--json'];
}

// --------------------------------------------------------------------------
// what went wrong, from micromamba's own words
// --------------------------------------------------------------------------

const NETWORK_PATTERNS = [
  // MEASURED: "Download error (7) Could not connect to server" (proxy to a
  // closed port), "Download error (6) Could not resolve hostname" (unknown
  // host). The number is libcurl's; any download error is a transport failure.
  /Download error \(\d+\)/,
  /Could not resolve host/i,
  /Could not connect to server/i,
  /Failed to connect to/i,
  /Connection (?:refused|reset|timed out)/i,
  /Operation timed out|Timeout was reached/i,
  /SSL connect error/i,
  /Network is unreachable/i,
];
const CHECKSUM_PATTERNS = [
  // MEASURED: "File not valid: SHA256 doesn't match expectation",
  // "tarball has incorrect SHA256", "Found incorrect downloads. Aborting".
  /SHA256 doesn't match/i,
  /MD5 doesn't match/i,
  /incorrect (?:SHA256|MD5)/i,
  /Found incorrect downloads/i,
];
const SIGNING_PATTERNS = [
  // From the wizard's own notes (`macos_env.js`): micromamba throws "Could not
  // codesign executable" in the middle of a transaction. Not measured here.
  /Could not codesign/i,
];

/**
 * `network | space | checksum | signing | other` from what micromamba printed.
 *
 * Space first (an out-of-disk failure is never a network one, and it is the one
 * that may have left a package half-written), then the checksum (a refused
 * download is not a connection problem), then the network.
 */
function classifyMicromambaFailure({ text = '', timedOut = false } = {}) {
  if (timedOut) return 'network';
  const body = String(text);
  if (sync().classifyPipFailure({ text: body }) === 'space') return 'space';
  if (CHECKSUM_PATTERNS.some((re) => re.test(body))) return 'checksum';
  if (SIGNING_PATTERNS.some((re) => re.test(body))) return 'signing';
  if (NETWORK_PATTERNS.some((re) => re.test(body))) return 'network';
  return 'other';
}

// --------------------------------------------------------------------------
// verification, as data
// --------------------------------------------------------------------------

/** `Map(name -> [version, ...])` from `micromamba list --json` (an object with
 *  a `packages` array; MEASURED). Null when the text is not that. */
function versionsFromList(text) {
  const body = String(text || '').trim();
  let data = null;
  try {
    data = JSON.parse(body);
  } catch {
    const i = body.indexOf('{');
    if (i >= 0) { try { data = JSON.parse(body.slice(i)); } catch { data = null; } }
  }
  if (!data || !Array.isArray(data.packages)) return null;
  const out = new Map();
  for (const p of data.packages) {
    if (!p || !p.name) continue;
    const versions = out.get(p.name) || [];
    versions.push(String(p.version));
    out.set(p.name, versions);
  }
  return out;
}

/**
 * What is wrong with the delta's packages on disk, as sentences. Empty is right.
 * Two independent views, because they fail differently: `micromamba list` is
 * what the user would see, conda-meta is what the next install would act on.
 */
function mismatches(delta, listed, installed) {
  const out = [];
  for (const p of delta) {
    const versions = listed ? (listed.get(p.name) || []) : null;
    if (versions && !(versions.length === 1 && versions[0] === p.version)) {
      out.push(`${p.name}: micromamba list shows ${versions.length ? versions.join(' and ') : 'nothing'}, the lock says ${p.version}`);
    }
    const stems = installed.get(p.name) || [];
    if (!(stems.length === 1 && stems[0] === p.stem)) {
      out.push(`${p.name}: ${stems.length} record(s) in conda-meta (${stems.join(', ') || 'none'}), expected exactly ${p.stem}`);
    }
  }
  return out;
}

// --------------------------------------------------------------------------
// the orchestration
// --------------------------------------------------------------------------

/**
 * Bring the conda environment to the lock. Never throws (the caller wraps it).
 *
 * @param ctx   what `package_sync.js` decided: home, python, lockName,
 *              lockText, digest, state, platformName, env, decision, log
 * @param d     the sync's dependencies (see `syncPackages`), plus optionally
 *   d.macos   { condaPlatform, micromamba, root, prefix, importArgs,
 *               healthScript (null = skip), codesign (false = skip, or a
 *               function), minLockPackages } -- tests and CI; production
 *               uses the defaults
 *
 * @returns the shapes `syncPip` answers in:
 *   {ok: true, action: 'noop'|'synced'|'failed'|'skip', ...}
 *   {ok: false, repair: true, message}   verification failed twice
 *   {cancelled: true}
 */
async function syncMacos(ctx, d) {
  const fs = d.fs || fsDefault;
  const path = d.path || pathDefault;
  const S = sync();
  const { home, python, mode, lockName, lockText, digest, state, platformName } = ctx;
  const say = ctx.log || (() => {});
  const repairing = Boolean(ctx.decision && ctx.decision.action === 'repair');
  const plat = platformModule();
  const env = macosEnv();
  const cfg = {
    condaPlatform: CONDA_PLATFORM,
    micromamba: plat.micromambaExeIn(home),
    root: plat.condaRootIn(home),
    prefix: path.dirname(path.dirname(python)),
    importArgs: null,
    healthScript: path.join(home, 'runtime', 'scripts', 'check_runtime_health.py'),
    codesign: true,
    ...(d.macos || {}),
  };
  const mmEnv = env.scrubbedEnv(ctx.env || process.env);
  const cancelled = () => d.isCancelled();
  const fail = (reason, detail) => S.failWith({ home, digest, state, d: { ...d, fs, path }, say }, reason, detail);
  const mm = (args, opts = {}) => d.run(cfg.micromamba, args, { env: mmEnv, ...opts });
  // micromamba prints a table with rules, blank lines and a standing security
  // warning; what the log needs is the package lines and the verdicts.
  const pipLine = (line) => {
    if (/^[-\s]*$/.test(line) || /Security Warning/.test(line)) return;
    say(`  micromamba: ${line}`);
  };

  // ---- is this a conda prefix, and is the lock one we can read -------------
  if (!fs.existsSync(path.join(cfg.prefix, 'conda-meta'))) {
    say(`skipped — ${cfg.prefix} is not a conda environment`);
    return { ok: true, action: 'skip', reason: 'not-a-conda-prefix' };
  }
  const complaint = env.lockFileComplaint(lockText, {
    platform: cfg.condaPlatform, ...(cfg.minLockPackages ? { minPackages: cfg.minLockPackages } : {}),
  });
  if (complaint) {
    // A packaging mistake, not something the user can act on: log, no dialog.
    say(`skipped — ${complaint}`);
    return { ok: true, action: 'skip', reason: 'lock-unsafe' };
  }
  let parsed;
  try {
    parsed = parseLock(lockText, cfg.condaPlatform);
  } catch (err) {
    say(`skipped — ${err.message}`);
    return { ok: true, action: 'skip', reason: 'lock-unsafe' };
  }
  if (parsed.unsupported.length) {
    say(`skipped — the lock holds pip-managed packages (${parsed.unsupported.slice(0, 5).join(', ')}), which this flow does not move`);
    return { ok: true, action: 'skip', reason: 'lock-has-pip' };
  }
  const badName = parsed.packages.find((p) => !SAFE_CONDA_NAME.test(p.name) || !p.version || !p.stem);
  if (badName) {
    say(`skipped — the lock names a package this flow will not handle (${badName.name})`);
    return { ok: true, action: 'skip', reason: 'lock-unsafe' };
  }

  // ---- the delta, from the disk --------------------------------------------
  let installed;
  try {
    installed = readInstalled(cfg.prefix, { fs, path });
  } catch (err) {
    return fail('other', `could not read ${path.join(cfg.prefix, 'conda-meta')} (${err.message})`);
  }
  const { delta, duplicates, unchanged } = computeDelta(parsed.packages, installed);
  if (duplicates.length) say(`duplicate records on disk for: ${duplicates.join(', ')}`);
  say(`${unchanged} of ${parsed.packages.length} packages are as locked; ${delta.length} differ`);

  const recordFile = path.join(home, S.RECORD_FILE);
  const markerFile = path.join(home, S.MARKER_FILE);
  const failureFile = path.join(home, S.FAILURE_FILE);
  const record = (how) => S.writeJsonAtomic(recordFile, S.recordFor({
    lockName, digest, mode, platformName, runtimeTag: d.runtimeTag(), how, now: d.now(),
  }), fs, path);

  const probeImports = async () => {
    const r = await d.run(python, cfg.importArgs || S.importArgs(), { timeoutMs: S.TIMEOUT_MS.imports });
    return r;
  };

  /** Still wrong after the retry: an untested state. The marker STAYS. */
  const needsRepair = (detail) => {
    say(`REPAIR NEEDED — ${detail}; not starting the backend`);
    return { ok: false, repair: true, detail, message: detail };
  };

  /**
   * A failure BEFORE anything was changed. With no marker that is `fail`. With
   * a marker (an earlier update was interrupted) the environment is unverified:
   * if the libraries still import the start goes ahead and the marker stays for
   * the next try; if they do not, there is nothing to start.
   */
  const cannotProceed = async (reason, detail) => {
    if (!repairing) return fail(reason, detail);
    const probe = await probeImports();
    if (cancelled() || probe.cancelled) return { cancelled: true };
    if (probe.code === 0) return fail(reason, `${detail} (the earlier update is still unverified)`);
    return needsRepair(`${detail}; and the libraries do not import`);
  };

  /** The checks of the delta's packages, as a list of sentences. */
  const verify = async () => {
    const problems = [];
    const listRun = await mm(listArgs(cfg), { timeoutMs: MAC_TIMEOUT_MS.list });
    if (cancelled() || listRun.cancelled) return { cancelled: true };
    const listed = listRun.code === 0 ? versionsFromList(listRun.stdout) : null;
    if (!listed) problems.push(`micromamba list could not be read (${S.tailOf(listRun.output) || `exit ${listRun.code}`})`);
    let now;
    try {
      now = readInstalled(cfg.prefix, { fs, path });
    } catch (err) {
      return { ok: false, detail: `conda-meta could not be read (${err.message})` };
    }
    problems.push(...mismatches(delta, listed, now));
    if (problems.length) return { ok: false, detail: problems.slice(0, 6).join('; ') };

    const imports = await probeImports();
    if (cancelled() || imports.cancelled) return { cancelled: true };
    if (imports.code !== 0) {
      return {
        ok: false,
        detail: imports.timedOut
          ? `the imports did not finish within ${S.TIMEOUT_MS.imports / 1000} s`
          : `the imports failed: ${S.tailOf(imports.output)}`,
      };
    }

    // Exactly one OpenMP runtime: the project's own gate, not a second opinion.
    if (cfg.healthScript === null) {
      say('OpenMP check skipped (no script configured)');
    } else if (!fs.existsSync(cfg.healthScript)) {
      say(`OpenMP check skipped (${cfg.healthScript} is not there)`);
    } else {
      const gate = await d.run(
        python, [cfg.healthScript, '--prefix', cfg.prefix, '--gate', 'files'],
        { timeoutMs: MAC_TIMEOUT_MS.health },
      );
      if (cancelled() || gate.cancelled) return { cancelled: true };
      if (gate.code !== 0) {
        return { ok: false, detail: `the OpenMP check failed: ${S.tailOf(gate.output, 4)}` };
      }
      say(`OpenMP check: ${S.tailOf(gate.output, 2)}`);
    }
    return { ok: true, detail: '' };
  };

  const finish = (how) => {
    // The marker goes FIRST (see syncPip): killed between the steps, there is
    // no marker and no record, and the next start finds nothing to do.
    S.removeFile(markerFile, fs);
    S.removeFile(failureFile, fs);
    record(how);
    say(`done (${how}); recorded`);
    return {
      ok: true, action: 'synced', how,
      packages: Object.fromEntries(delta.map((p) => [p.name, p.version])),
    };
  };

  // ---- nothing differs ----------------------------------------------------
  if (!delta.length) {
    if (!repairing) {
      S.removeFile(failureFile, fs);
      record('noop');
      say('nothing to change; recorded');
      return { ok: true, action: 'noop' };
    }
    say('nothing differs after the interrupted update; verifying only');
    if (!fs.existsSync(cfg.micromamba)) return needsRepair(`micromamba is missing at ${cfg.micromamba}`);
    const check = await verify();
    if (check.cancelled) return { cancelled: true };
    return check.ok ? finish('repair') : needsRepair(check.detail);
  }

  // ---- the guard ------------------------------------------------------------
  const sizes = {};
  if (d.headSize) {
    await Promise.all(delta.map(async (p) => {
      if (!p.url) return;
      try {
        const n = await d.headSize(p.url);
        if (Number.isFinite(n) && n >= 0) sizes[p.name] = n;
      } catch { /* unknown, which the guard reports */ }
    }));
    if (cancelled()) return { cancelled: true };
  }
  const guard = guardDelta(delta, sizes);
  if (!guard.ok) return fail(guard.reason, guard.detail);
  if (guard.unknownSize && guard.unknownSize.length) {
    say(`size check partial: no size for ${guard.unknownSize.join(', ')}`);
  }
  say(`delta: ${delta.map((p) => `${p.name} ${p.from.map(versionFromStem).join('+') || '(new)'} -> ${p.version}`).join(', ')} `
    + `(${(guard.bytes / MB).toFixed(2)} MB known)`);

  // ---- room to write; the tools we depend on ---------------------------------
  const free = d.freeBytes(home) || { bytes: 0, known: false };
  if (free.known && free.bytes < S.MIN_FREE_BYTES) {
    return cannotProceed('space', `only ${(free.bytes / MB).toFixed(0)} MB free, ${S.MIN_FREE_BYTES / MB} MB needed`);
  }
  if (!free.known) say('free space could not be measured; going on');

  if (!fs.existsSync(cfg.micromamba)) {
    return cannotProceed('other', `micromamba is missing at ${cfg.micromamba}`);
  }
  const version = await mm(['--version'], { timeoutMs: MAC_TIMEOUT_MS.probe });
  if (cancelled() || version.cancelled) return { cancelled: true };
  say(`micromamba ${version.stdout.trim() || '(version unknown)'}; pinned ${plat.MICROMAMBA_VERSION}`);

  // micromamba's own platform is a compile-time fact: an x86_64 build under
  // Rosetta would "update" an arm64 environment with x86_64 packages.
  const info = await mm(env.infoArgs({ root: cfg.root, prefix: cfg.prefix }), { timeoutMs: MAC_TIMEOUT_MS.probe });
  if (cancelled() || info.cancelled) return { cancelled: true };
  const built = env.platformFromInfo(info.stdout);
  if (built !== cfg.condaPlatform) {
    return cannotProceed('other', `micromamba reports ${built || 'no platform'}, not ${cfg.condaPlatform} (running under Rosetta?)`);
  }

  // Relinking a binary makes micromamba re-sign it, and it throws if signing
  // fails. Find out before anything is removed, not in the middle.
  if (cfg.codesign) {
    try {
      const tmpDir = path.join(home, 'setup-tmp');
      fs.mkdirSync(tmpDir, { recursive: true });
      const probe = typeof cfg.codesign === 'function' ? cfg.codesign : env.codesignWorks;
      await probe({
        tmpDir,
        copyFile: (from, to) => fs.promises.copyFile(from, to),
        run: async (exe, ...args) => {
          const r = await d.run(exe, args, { timeoutMs: MAC_TIMEOUT_MS.probe });
          if (r.code !== 0) throw new Error(S.tailOf(r.output) || `exit ${r.code}`);
          return r.stdout;
        },
        remove: (file) => fs.promises.rm(file, { force: true }),
      });
    } catch (err) {
      return cannotProceed('signing', `this Mac cannot sign what the update has to relink (${err.message})`);
    }
  }
  if (cancelled()) return { cancelled: true };

  // ---- the delta lock ---------------------------------------------------------
  const tmp = path.join(home, 'setup-tmp');
  const lockFile = path.join(tmp, DELTA_LOCK_FILE);
  const deltaText = buildDeltaLock(parsed, delta.map((p) => p.name));
  const deltaComplaint = env.lockFileComplaint(deltaText, {
    platform: cfg.condaPlatform, minPackages: delta.length, minBytes: 200,
  });
  if (deltaComplaint) return fail('other', `the delta lock is not valid (${deltaComplaint})`);
  try {
    fs.mkdirSync(tmp, { recursive: true });
    fs.writeFileSync(lockFile, deltaText, 'utf8');
  } catch (err) {
    return fail('other', `could not write ${lockFile} (${err.message})`);
  }

  try {
    d.onPhase('syncing');
    const args = { root: cfg.root, prefix: cfg.prefix, lockFile, platform: cfg.condaPlatform };

    // ---- the network phase: nothing is linked --------------------------------
    say('downloading the changed packages');
    const dl = await mm(downloadArgs(args), { timeoutMs: MAC_TIMEOUT_MS.download, onLine: pipLine });
    if (cancelled() || dl.cancelled) return { cancelled: true };
    if (dl.code !== 0) {
      const reason = classifyMicromambaFailure({ text: `${dl.stdout}\n${dl.stderr}`, timedOut: dl.timedOut });
      const detail = dl.timedOut ? `micromamba did not finish downloading within ${MAC_TIMEOUT_MS.download / 60000} min`
        : (S.tailOf(dl.output) || `micromamba exited with ${dl.code}`);
      // Nothing was linked: the prefix is as it was.
      return cannotProceed(reason, detail);
    }

    // ---- the marker, then the local phase ---------------------------------------
    try {
      S.writeJsonAtomic(markerFile, {
        schema: S.SCHEMA, startedAt: new Date(d.now()).toISOString(), lockSha256: digest, mode,
        from: Object.fromEntries(delta.map((p) => [p.name, p.from.map(versionFromStem).join('+') || null])),
        to: Object.fromEntries(delta.map((p) => [p.name, p.version])),
      }, fs, path);
    } catch (err) {
      // Without the marker a kill in the next minute would leave an
      // environment nobody knows about. Do not start what cannot be undone.
      return fail('other', `could not write ${S.MARKER_FILE} (${err.message})`);
    }

    /** Remove every delta package that is installed in ANY form, then link the
     *  delta. Idempotent: a retry after a half-done first attempt must not
     *  link a second copy of what already landed. */
    const apply = async (label) => {
      let now;
      try { now = readInstalled(cfg.prefix, { fs, path }); } catch (err) { return { ok: false, detail: `conda-meta unreadable (${err.message})` }; }
      const names = delta.map((p) => p.name).filter((n) => (now.get(n) || []).length > 0);
      if (names.length) {
        say(`${label}: removing ${names.join(', ')}`);
        const rm = await mm(removeArgs({ root: cfg.root, prefix: cfg.prefix, names }), {
          timeoutMs: MAC_TIMEOUT_MS.remove, onLine: pipLine,
        });
        if (cancelled() || rm.cancelled) return { cancelled: true };
        if (rm.code !== 0) {
          return { ok: false, detail: `remove failed (${S.tailOf(rm.output) || `exit ${rm.code}`})`, output: rm.output };
        }
      }
      say(`${label}: linking the delta from the package cache`);
      const inst = await mm(installDeltaArgs(args), { timeoutMs: MAC_TIMEOUT_MS.install, onLine: pipLine });
      if (cancelled() || inst.cancelled) return { cancelled: true };
      if (inst.code !== 0) {
        return {
          ok: false, output: inst.output,
          detail: inst.timedOut ? `install did not finish within ${MAC_TIMEOUT_MS.install / 60000} min`
            : `install failed (${S.tailOf(inst.output) || `exit ${inst.code}`})`,
        };
      }
      return { ok: true };
    };

    let step = await apply('first attempt');
    if (step.cancelled) return { cancelled: true };
    let check = step;
    if (step.ok) {
      check = await verify();
      if (check.cancelled) return { cancelled: true };
      if (check.ok) return finish(repairing ? 'repair' : 'sync');
    }
    say(`not right yet (${check.detail}); doing it once more`);

    step = await apply('second attempt');
    if (step.cancelled) return { cancelled: true };
    check = step;
    if (step.ok) {
      check = await verify();
      if (check.cancelled) return { cancelled: true };
      if (check.ok) return finish('repair');
    }
    return needsRepair(check.detail);
  } finally {
    S.removeFile(lockFile, fs);
  }
}

module.exports = {
  CONDA_PLATFORM,
  DELTA_LOCK_FILE,
  MAX_DELTA_PACKAGES,
  MAC_TIMEOUT_MS,
  GUARDED_NAMES_MACOS,
  SAFE_CONDA_NAME,
  stemFromUrl,
  versionFromStem,
  parseLock,
  buildDeltaLock,
  readInstalled,
  computeDelta,
  guardDelta,
  downloadArgs,
  removeArgs,
  installDeltaArgs,
  listArgs,
  classifyMicromambaFailure,
  versionsFromList,
  mismatches,
  syncMacos,
};
