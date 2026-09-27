'use strict';

/**
 * Building the macOS environment with micromamba.
 *
 * macOS is the one platform that does not install from a pip lock. The reason
 * is a bug report: on a Mac the pip wheels of torch, scikit-learn and faiss
 * each bring their own OpenMP runtime, and the second one to initialise aborts
 * the backend about thirty seconds after launch. conda-forge builds all of
 * them against one llvm-openmp, so the whole environment comes from there.
 *
 * Everything here is argv construction and output parsing, kept apart from the
 * running of it so the decisions can be tested without a Mac. What the
 * commands DO can only be checked on one (T11).
 */

const path = require('path');
const fsp = require('fs').promises;
const pathJoin = path.join;

/**
 * The lock file the wizard feeds to micromamba.
 *
 * The name is load-bearing and looks arbitrary. micromamba decides whether a
 * file is a lockfile by its NAME, not its content:
 *
 *     is_conda_env_lockfile_name(f) = f.ends_with("-lock.yml")
 *                                  || f.ends_with("-lock.yaml")
 *
 * anything else is parsed as a plain environment YAML. Measured with
 * micromamba 2.9.0 on this repository's own lock, same bytes, two names:
 * `conda-lock-macos.yml` gave exit 0, no output and an EMPTY environment;
 * `orienta-macos-lock.yml` gave a transaction of 467 packages. A silent
 * success is worse than a failure, so `isLockFileName` guards it.
 */
const MACOS_LOCK_FILE = 'orienta-macos-lock.yml';

function isLockFileName(name) {
  return /-lock\.ya?ml$/.test(String(name || ''));
}

/**
 * Pull ONE named file out of the runtime package.
 *
 * macOS has a circularity: the program files are unpacked by `apply_update.py`
 * (which carries the security checks — it refuses any member with a `Database`
 * path component), and on a Mac there is no interpreter to run it until the
 * conda environment exists, which needs the lock file that is inside those
 * same program files.
 *
 * So exactly one member comes out early, by a FIXED name we supply, with
 * `unzip -p` writing to stdout — nothing is written to a path taken from the
 * archive, so none of the applier's checks are being worked around. The full
 * unpack still goes through `apply_update.py` afterwards, unchanged.
 *
 * `/usr/bin/unzip` by absolute path, and it is part of the macOS base system;
 * `unzipProbeArgs` checks it before anything is downloaded rather than after.
 */
function unzipMemberArgs(archive, member) {
  return ['/usr/bin/unzip', '-p', archive, member];
}

function unzipProbeArgs() {
  return ['/usr/bin/unzip', '-v'];
}

/** The smallest a real lock could plausibly be; ours is ~200 KB. */
const MIN_LOCK_BYTES = 1024;
/** Floor for package records of our platform; the real lock has 467. */
const MIN_PACKAGES = 100;

/**
 * Is this actually the lock file, or something that will quietly do nothing?
 *
 * A shape check, NOT a YAML parse — and deliberately so. There is no YAML
 * parser available here: the shell declares no YAML dependency, and the one
 * resolvable transitively would not be in the shipped app at all, because
 * electron-builder excludes `node_modules/**`. A check that worked in
 * development and vanished in the build would be worse than none.
 *
 * What it looks for is what micromamba itself requires, because the failure
 * mode on this path is never an error message:
 *
 *  - the file is not empty or truncated (a partial `unzip -p` writes a prefix
 *    and exits non-zero, but callers have been known to ignore that);
 *  - lockfile version 1 — micromamba reads `case 1:` and nothing else;
 *  - `metadata.platforms` includes the platform we are installing, and at
 *    least one package record is for it. Selecting an absent platform is a
 *    WARNING in micromamba ("Selected packages ... are empty"), followed by a
 *    successful, empty transaction — the same silent success as the wrong
 *    file name.
 *
 * Returns null when it is fine, or a sentence naming what is wrong.
 */
function lockFileComplaint(text, { platform = 'osx-arm64', minBytes = MIN_LOCK_BYTES } = {}) {
  const body = String(text || '');
  if (body.trim().length === 0) return 'the lock file came out empty';
  if (Buffer.byteLength(body, 'utf8') < minBytes) {
    return `the lock file is only ${Buffer.byteLength(body, 'utf8')} bytes, so it is truncated`;
  }
  const version = /^version:\s*(\d+)\s*$/m.exec(body);
  if (!version) return 'the lock file has no `version:` line, so it is not a conda lock file';
  if (version[1] !== '1') {
    return `the lock file is version ${version[1]}; micromamba reads version 1 only`;
  }
  if (!/^\s*content_hash:\s*$/m.test(body)) return 'the lock file has no content_hash';
  if (!/^package:\s*$/m.test(body)) return 'the lock file lists no packages';
  if (!new RegExp(`^\\s*-\\s*${platform}\\s*$`, 'm').test(body)) {
    return `the lock file does not cover ${platform}`;
  }
  // Not "at least one": a lock with a single matching record installs one
  // package and reports success, which is the empty-environment failure this
  // check exists to prevent, only slightly less empty. The real environment
  // is 467 packages; any scientific stack is hundreds.
  const forUs = (body.match(new RegExp(`^\\s*platform:\\s*${platform}\\s*$`, 'gm')) || []).length;
  if (forUs < MIN_PACKAGES) {
    return `the lock file has only ${forUs} package(s) for ${platform}, where a `
      + `complete environment has several hundred; installing it would `
      + 'succeed and produce an unusable environment';
  }
  return null;
}

/**
 * Environment variables micromamba reads, which must not leak in from the
 * user's own conda install.
 *
 * The list is from libmamba's configuration table, plus the generic rule:
 * EVERY config key is also settable as `MAMBA_<UPPERCASE_KEY>`. So the named
 * ones are belt, and the `MAMBA_`/`CONDA_` prefix sweep is braces.
 */
const MAMBA_ENV_VARS = [
  'MAMBA_ROOT_PREFIX', 'MAMBA_DEFAULT_ROOT_PREFIX', 'MAMBA_PLATFORM',
  'CONDARC', 'MAMBARC',
  'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_ENVS_DIRS', 'CONDA_ENVS_PATH',
  'CONDA_PKGS_DIRS', 'CONDA_SUBDIR', 'CONDA_CHANNELS',
  'CONDA_SAFETY_CHECKS', 'CONDA_EXTRA_SAFETY_CHECKS',
  // Not conda's, but it is where micromamba looks for a config file.
  'XDG_CONFIG_HOME',
];

/** A copy of `env` with everything micromamba might read removed. */
function scrubbedEnv(env = process.env) {
  const out = {};
  for (const [key, value] of Object.entries(env)) {
    if (MAMBA_ENV_VARS.includes(key)) continue;
    if (/^(MAMBA|CONDA)_/.test(key)) continue;
    out[key] = value;
  }
  return out;
}

/**
 * `micromamba create` for a lock file, isolated from the user's own setup.
 *
 * Why each flag, because every one of them is load-bearing:
 *
 *  --no-rc      no .condarc/.mambarc from anywhere on the machine
 *  --no-env     no configuration from environment variables either
 *  -y           REQUIRED: micromamba's prompt has no tty detection, so
 *               without this a non-interactive run can block for ever
 *  -r <root>    the package cache defaults to <root_prefix>/pkgs, and without
 *               -r the root prefix is the USER's — we would download several
 *               gigabytes into their conda install. `-p` alone is not enough.
 *  -p <prefix>  where the environment goes
 *  --platform   micromamba's own platform is decided at COMPILE time, so an
 *               x86_64 build running under Rosetta on an M-series Mac would
 *               silently build a complete x86_64 environment. Stating it is
 *               the cheap half of the guard; `infoArgs` is the other half.
 *  -f <lock>    the lock file, whose NAME has to end in -lock.yml
 *
 * `--override-channels` is deliberately NOT passed: it constrains a solve, and
 * a lockfile install does not solve — every URL comes from the lock.
 */
function createArgs({ root, prefix, lockFile, platform = 'osx-arm64' }) {
  if (!isLockFileName(lockFile)) {
    throw new Error(
      `micromamba only recognises a lock file whose name ends in -lock.yml or `
      + `-lock.yaml; "${path.basename(String(lockFile))}" would be read as a plain `
      + `environment file and would install NOTHING, without an error`);
  }
  return [
    'create', '--no-rc', '--no-env', '-y',
    '-r', root,
    '-p', prefix,
    '--platform', platform,
    '-f', lockFile,
  ];
}

/**
 * `micromamba info`, for the other half of the Rosetta guard.
 *
 * Measured: `info` does NOT accept `--platform` ("The following arguments were
 * not expected"), and without it reports the platform the BINARY was built
 * for. That is exactly what we need to know — an osx-64 micromamba on an
 * M-series Mac answers `osx-64`, and everything it builds would be x86_64.
 */
function infoArgs({ root, prefix }) {
  return ['info', '--no-rc', '--no-env', '-r', root, '-p', prefix];
}

/** The platform `micromamba info` reports, or null if it said nothing. */
function platformFromInfo(stdout) {
  const match = /^\s*platform\s*:\s*(\S+)\s*$/m.exec(String(stdout || ''));
  return match ? match[1] : null;
}

/**
 * Remove the downloaded packages.
 *
 * Safe by default: micromamba hard-links a package into the environment and
 * falls back to a COPY, never a symlink, so unlinking the cache name leaves
 * the environment whole. (On osx-arm64 any binary whose embedded prefix was
 * rewritten is a fresh copy anyway, and re-signed.)
 *
 * `clean -p` only removes packages nothing uses, which here is close to
 * nothing, so the directory goes as well. Root and prefix must be on the same
 * volume or the hard links become copies -- correct, but twice the disk.
 */
function cleanArgs({ root }) {
  return ['clean', '--no-rc', '--no-env', '-y', '-r', root, '-a'];
}

function packageCacheDir(root) {
  return path.join(root, 'pkgs');
}

/**
 * Is `/usr/bin/codesign` there?
 *
 * micromamba re-signs every arm64 binary whose embedded install prefix it had
 * to rewrite, by spawning `/usr/bin/codesign -s - -f <path>`, and it THROWS if
 * that fails rather than carrying on. Apple Silicon refuses to load an
 * unsigned arm64 binary at all (SIGKILL, no dialog), so this is not optional.
 *
 * Whether `/usr/bin/codesign` works without the Xcode command line tools is
 * unverified -- it may be one of the shims that prompts to install them. On
 * the machine this ships to (the head of institute's Mac) that would strand
 * the install half-finished, so it is checked BEFORE the transaction starts
 * rather than discovered in the middle of one.
 */
function codesignProbeArgs() {
  return ['/usr/bin/codesign', '--version'];
}

/**
 * A file we may copy and sign to prove codesign actually works.
 *
 * Small, present on every macOS, and a Mach-O — which `/bin/echo` is.
 */
const CODESIGN_PROBE_SUBJECT = '/bin/echo';

/** Ad-hoc sign a throwaway copy: `-s -` is the ad-hoc identity. */
function codesignSignArgs(target) {
  return ['/usr/bin/codesign', '-s', '-', '-f', target];
}

/**
 * Does ad-hoc signing actually WORK here, not merely exist?
 *
 * Researched 2026-09-23 (spec appendix A): `/usr/bin/codesign` is a
 * base-system binary on every macOS we support, and ad-hoc signing has needed
 * no Xcode tools since macOS 12, when Apple moved signature allocation
 * in-process. So "is the file there" tests the one thing that cannot fail.
 *
 * What CAN fail, and what this catches instead, by signing a copy of a small
 * system binary before the transaction rather than inside it:
 *
 *  - a target directory codesign cannot write to (it writes `<path>.cstemp`
 *    beside the file);
 *  - stray extended attributes, which produce "resource fork, Finder
 *    information, or similar detritus not allowed" — a documented consequence
 *    of building inside an iCloud- or Dropbox-synced folder;
 *  - micromamba's own spawn bug. The version pinned here, 2.9.0-0 (published
 *    2026-08-07), predates the fix for it (merged 2026-08-21, in no release
 *    yet) and throws "Could not codesign executable: Invalid argument" in the
 *    middle of a transaction. This probe uses the same binary the same way, so
 *    it fails first, before anything has been installed.
 */
async function codesignWorks({ tmpDir, copyFile, run, remove }) {
  const target = path.join(tmpDir, 'orienta-codesign-probe');
  await copyFile(CODESIGN_PROBE_SUBJECT, target);
  try {
    await run(...codesignSignArgs(target));
  } finally {
    await remove(target);
  }
}

/**
 * Build the macOS environment, start to finish.
 *
 * Orchestration with its runners injected, so the ORDER is testable without a
 * Mac — which matters, because the order is the part that was decided rather
 * than derived, and every failure it guards against is silent.
 *
 * `run` streams a command's output (for the long one); `capture` returns
 * stdout. Both take (exe, args, options) and throw on a non-zero exit.
 */
async function installMacosEnvironment({
  micromamba, root, prefix, archive,
  lockMember = MACOS_LOCK_FILE,
  lockPath,
  platform = 'osx-arm64',
  run, capture, writeFile, onLine = () => {},
  exists = async (p) => { try { await fsp.stat(p); return true; } catch { return false; } },
  codesign = codesignWorks,
}) {
  const say = (line) => { onLine(line); };

  // 1. Before anything else: the two tools we depend on and cannot install.
  //    A machine without them must find out now, not after a download or --
  //    worse -- in the middle of a transaction.
  try {
    await capture(...unzipProbeArgs());
  } catch (err) {
    throw new Error(
      '/usr/bin/unzip is missing, so the lock file cannot be read out of the '
      + `package. [${err.message}]`);
  }
  try {
    // Not "is codesign there" -- it always is (spec appendix A.1). This signs
    // a throwaway copy of a system binary, which is what micromamba is about
    // to do many times over, and so fails HERE instead of mid-transaction.
    await codesign({
      tmpDir: path.dirname(lockPath),
      copyFile: (from, to) => fsp.copyFile(from, to),
      run: capture,
      remove: (file) => fsp.rm(file, { force: true }),
    });
  } catch (err) {
    // Deliberately NOT "install the developer tools". codesign needs none,
    // and sending somebody to `xcode-select --install` for a problem it
    // cannot solve costs an afternoon and several gigabytes for nothing.
    throw new Error(
      'This Mac cannot sign the files the installation has to sign, so the '
      + 'environment would be built and then refuse to start. '
      + `[${err.message}]`);
  }

  // 2. One named member out of the package, to stdout.
  say(`reading ${lockMember} from the package`);
  let lockText;
  try {
    lockText = await capture(...unzipMemberArgs(archive, lockMember));
  } catch (err) {
    // Every package built before this feature existed has no lock in it, and
    // the version picker lets a user choose any published release. Unwrapped,
    // that reads as `Command failed: /usr/bin/unzip -p ...`.
    throw new Error(
      `this version of Orienta has no macOS environment in it (${lockMember} is `
      + 'not in the package), so it cannot be installed on a Mac. Choose a newer '
      + `version. [${err.message}]`);
  }
  const complaint = lockFileComplaint(lockText, { platform });
  if (complaint) {
    throw new Error(`${complaint}. The downloaded package is damaged or is not `
      + 'the one this installer expects.');
  }
  await writeFile(lockPath, lockText);

  // 3. The environment. The long step.
  say('creating the Python environment (this takes a few minutes)');
  await run(micromamba, createArgs({ root, prefix, lockFile: lockPath, platform }),
    { onLine });

  // 4. Did we get the architecture we asked for? micromamba's own platform is
  //    compile-time, so an x86_64 binary under Rosetta would have built a
  //    complete x86_64 environment without complaining.
  const info = await capture(micromamba, ...infoArgs({ root, prefix }));
  const built = platformFromInfo(info);
  if (built !== platform) {
    throw new Error(
      `the environment was built for ${built || 'an unknown platform'} instead of `
      + `${platform}. This happens when the installer runs under Rosetta; open `
      + 'Orienta directly rather than through a translated terminal.');
  }

  // 5. Did it actually produce an interpreter?
  //    `micromamba info` prints a platform for an empty prefix too, so the
  //    check above does not cover this. Without it, an environment that
  //    installed nothing is discovered two calls later, when the unpacker is
  //    spawned and Node reports ENOENT -- which the wizard then reports as
  //    "unpacking the program files failed", blaming the wrong step.
  const interpreter = pathJoin(prefix, 'bin', 'python');
  if (!(await exists(interpreter))) {
    throw new Error(
      `the environment was created but has no interpreter at ${interpreter}. `
      + 'The lock file installed nothing, which usually means it is for another '
      + 'platform.');
  }

  // 6. The cache. Safe: packages are hard-linked or copied into the
  //    environment, never symlinked, so removing the cache leaves it whole.
  say('removing the downloaded packages');
  try {
    await run(micromamba, cleanArgs({ root }), { onLine });
  } catch (err) {
    // Wasted disk, not a broken install. Refusing here would strand a user
    // who has a working environment.
    say(`could not clean the package cache (harmless): ${err.message}`);
  }
  return { prefix, platform: built };
}

module.exports = {
  installMacosEnvironment,
  MACOS_LOCK_FILE,
  MAMBA_ENV_VARS,
  isLockFileName,
  unzipMemberArgs,
  unzipProbeArgs,
  lockFileComplaint,
  MIN_LOCK_BYTES,
  MIN_PACKAGES,
  scrubbedEnv,
  createArgs,
  infoArgs,
  platformFromInfo,
  cleanArgs,
  packageCacheDir,
  codesignProbeArgs,
  codesignSignArgs,
  codesignWorks,
  CODESIGN_PROBE_SUBJECT,
};
