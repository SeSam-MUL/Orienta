/**
 * The package that shipped inside this installer, and whether what is installed
 * is older than it.
 *
 * WHY THIS EXISTS. Installing a newer .exe over an existing installation
 * replaces the SHELL and nothing else. `installedRuntime()` asks only whether
 * `runtime/VERSION` is a file, never which version it names, so a v0.4.4 runtime
 * satisfies a v0.4.6 shell; and the wizard's unpack runs on a FIRST install
 * only. Measured on 2026-09-27, installing 0.4.6 over 0.4.4: the registry said
 * "Orienta 0.4.6", `Orienta.exe` was from that build, the bundled
 * `resources/orienta-runtime-v0.4.6.zip` lay beside it -- and `runtime/VERSION`
 * still said `v0.4.4`, the backend imported that tree, and Settings -> About
 * reported v0.4.4. Every fix between the two releases was absent, with nothing
 * on screen to suggest it. That is the whole bug this module closes.
 *
 * IT UNPACKS NOTHING ITSELF. It PARKS the bundled package the way the updater
 * was designed for and lets `apply_update.py` do the work, because that file is
 * where the guarantees live: the member checks, the refusal of anything landing
 * under `Database/`, prune-before-extract, and idempotence after a kill. A
 * second unpacker would mean a second copy of those guarantees, and only one of
 * them would be the audited one.
 *
 * `extract_only()` is deliberately NOT the path taken: it exists for the first
 * install, has no old manifest, and therefore cannot prune -- a file a release
 * deletes would survive on every updated machine for ever.
 *
 * Everything here is injectable and none of it touches Electron, so the
 * decisions can be tested without a packaged app, an interpreter or a zip.
 */

const fsDefault = require('fs');
const pathDefault = require('path');

/** `orienta-runtime-v1.2.3.zip` -> `v1.2.3`. */
const PACKAGE_RE = /^orienta-runtime-(.+)\.zip$/i;

/** Lazy, and from the installer: `compareTagsDesc` is the function that knows
 *  v0.10.0 outranks v0.9.0, and a second copy of that subtlety is how the
 *  0.9 -> 0.10 boundary gets shipped wrong. Required late so that loading this
 *  module costs nothing at startup. */
function compareTagsDesc(a, b) {
  return require('./setup/installer').compareTagsDesc(a, b);
}

function tagOf(fileName) {
  const m = PACKAGE_RE.exec(fileName);
  return m ? m[1] : null;
}

/**
 * The runtime package inside this installation, or null.
 *
 * A directory holding two packages yields the NEWEST, not the first: a
 * repackaged installer or a leftover from a previous build would otherwise
 * decide the version by directory order. The `.sha256` is required, exactly as
 * it is for a sideloaded package -- the applier verifies the digest, and a
 * package we cannot state a digest for is one it will refuse anyway.
 */
function bundledPackage(resourcesPath, deps = {}) {
  const fs = deps.fs || fsDefault;
  const path = deps.path || pathDefault;
  if (!resourcesPath) return null;
  let names;
  try {
    names = fs.readdirSync(resourcesPath);
  } catch {
    return null;                    // not a packaged app, or nothing bundled
  }
  const candidates = names.filter((n) => PACKAGE_RE.test(n) && tagOf(n));
  candidates.sort((a, b) => compareTagsDesc(tagOf(a), tagOf(b)));
  for (const name of candidates) {
    const zip = path.join(resourcesPath, name);
    const sum = `${zip}.sha256`;
    try {
      if (fs.statSync(zip).isFile() && fs.statSync(sum).isFile()) {
        return { zip, sum, name, tag: tagOf(name) };
      }
    } catch { /* unreadable; try the next one */ }
  }
  return null;
}

/**
 * What `runtime/VERSION` says, or null when there is no readable answer.
 *
 * First line only, trimmed: the file is written by the package and read here,
 * and a trailing newline must not become part of a version number.
 */
function installedTag(home, deps = {}) {
  const fs = deps.fs || fsDefault;
  const path = deps.path || pathDefault;
  if (!home) return null;
  try {
    const text = fs.readFileSync(path.join(home, 'runtime', 'VERSION'), 'utf8');
    const first = String(text).split(/\r?\n/)[0].trim();
    return first || null;
  } catch {
    return null;
  }
}

/**
 * The digest of the package the installed runtime came from, or null.
 *
 * `runtime/VERSION` answers "which version", and from 0.4.6 onwards that is not
 * enough: a release may be rebuilt under the same tag, and then the tag says
 * "nothing to do" while the program files are a build behind — the exact failure
 * this module exists to close, coming back through the version number.
 *
 * null is the honest answer for every installation made before the applier began
 * recording this, and the decision below treats it as "cannot tell" rather than
 * as "identical".
 */
function installedDigest(home, deps = {}) {
  const fs = deps.fs || fsDefault;
  const path = deps.path || pathDefault;
  if (!home) return null;
  try {
    const text = fs.readFileSync(path.join(home, 'runtime', 'SOURCE_SHA256'), 'utf8');
    const token = String(text).trim().split(/\s+/)[0] || '';
    return /^[0-9a-f]{64}$/i.test(token) ? token.toLowerCase() : null;
  } catch {
    return null;
  }
}

/** Is a package already parked? Then it, not the bundle, is what applies. */
function alreadyParked(home, deps = {}) {
  const fs = deps.fs || fsDefault;
  const path = deps.path || pathDefault;
  try {
    return fs.statSync(path.join(home, 'pending.json')).isFile();
  } catch {
    return false;
  }
}

/**
 * `{action, reason}` — the whole decision, as data.
 *
 *   park    the bundle is newer; park it and run the applier
 *   apply   something is parked already; run the applier, park nothing
 *   none    nothing to do, and `reason` says why in one line for the log
 *
 * A bundle that is OLDER never downgrades. Someone who installs an older .exe
 * on purpose -- to reproduce a bug, to pin a version in a lab -- is not asking
 * for their program files to be rolled back underneath them, and doing it
 * silently would be the same class of surprise as this module exists to fix.
 * It is logged, so "why does About still say the newer number" has an answer.
 */
function updateDecision({
  bundled, installed, parked, bundledDigest = null, installedDigest = null,
}) {
  if (parked) {
    return { action: 'apply', reason: 'a package is already parked; applying that' };
  }
  if (!bundled) {
    return { action: 'none', reason: 'no runtime package shipped beside this shell' };
  }
  if (!installed) {
    // Not our business: with no runtime there is nothing to update, and the
    // startup decision routes this to the wizard, which unpacks the bundle.
    return { action: 'none', reason: 'nothing installed yet; the wizard unpacks the package' };
  }
  const order = compareTagsDesc(bundled.tag, installed);
  if (order < 0) {
    return {
      action: 'park',
      reason: `the bundled ${bundled.tag} is newer than the installed ${installed}`,
    };
  }
  if (order === 0) {
    // SAME TAG, which since 0.4.6 no longer means same build: that release was
    // rebuilt four times, and Sebastian's machine holds Build 3 as v0.4.6. Tags
    // alone would answer "nothing to do" and leave the program files a build
    // behind under a new shell — this module's own bug, returning through the
    // version number. So compare what the builds actually are.
    if (bundledDigest && installedDigest && bundledDigest === installedDigest) {
      return {
        action: 'none',
        reason: `the installed ${installed} is the same build as this shell carries`,
      };
    }
    if (bundledDigest && installedDigest) {
      return {
        action: 'park',
        reason: `the installed ${installed} is a DIFFERENT build of the same version `
              + `(${installedDigest.slice(0, 12)} vs ${bundledDigest.slice(0, 12)})`,
      };
    }
    if (bundledDigest && !installedDigest) {
      // Every installation made before the applier recorded this. Apply once and
      // let it record; the applier is idempotent, prunes by manifest diff and
      // refuses anything under `Database/`, so an unnecessary apply costs one
      // unpack and nothing else. Skipping it is what would be expensive.
      return {
        action: 'park',
        reason: `the installed ${installed} does not say which build it is; `
              + 'applying once so that it does',
      };
    }
    // No digest for the BUNDLE — a malformed `.sha256` beside it. Parking would
    // repeat on every start, because nothing would ever be learned. Say why and
    // leave the runtime alone.
    return {
      action: 'none',
      reason: `the installed ${installed} matches this shell's tag, and the bundled `
            + 'package states no digest to compare builds by',
    };
  }
  return {
    action: 'none',
    reason: `the installed ${installed} is newer than the bundled ${bundled.tag}; `
          + 'leaving it alone',
  };
}

/** The digest as the `.sha256` file states it: first whitespace-delimited token. */
function digestFrom(sumFile, deps = {}) {
  const fs = deps.fs || fsDefault;
  const text = fs.readFileSync(sumFile, 'utf8');
  const token = String(text).trim().split(/\s+/)[0] || '';
  if (!/^[0-9a-f]{64}$/i.test(token)) {
    throw new Error(`${sumFile} does not contain a sha256`);
  }
  return token.toLowerCase();
}

/**
 * Park the bundled package: the zip under `<home>/pending/`, the record beside
 * it, in the shape `apply_update.py` reads.
 *
 * COPIED rather than referenced. The applier deletes the parked file when it
 * discards a package it refuses, and the original here lives in the
 * installation directory -- on Windows under `%LOCALAPPDATA%\Programs`, which
 * the next installer run replaces wholesale. Five megabytes buys the applier
 * the right to own what it was handed.
 *
 * The record is written LAST. A record naming a file that is not fully copied
 * is the one state that would make the applier report a corrupt package on a
 * perfectly good build.
 */
function parkBundled({ home, bundled, deps = {} }) {
  const fs = deps.fs || fsDefault;
  const path = deps.path || pathDefault;
  const sha256 = digestFrom(bundled.sum, { fs });
  const dir = path.join(home, 'pending');
  fs.mkdirSync(dir, { recursive: true });
  const target = path.join(dir, bundled.name);
  fs.copyFileSync(bundled.zip, target);
  const record = { tag: bundled.tag, file: bundled.name, sha256 };
  fs.writeFileSync(path.join(home, 'pending.json'), `${JSON.stringify(record, null, 2)}\n`, 'utf8');
  return { ...record, path: target };
}

/**
 * The verdict of a run of `apply_update.py`, read from its `--result-json`.
 *
 * The exit code is not enough and never was: it cannot distinguish "a member
 * tried to escape into Database" from "Python did not start". A missing or
 * unreadable verdict after a non-zero exit is itself dirty -- we do not know
 * what the tree looks like, and guessing "fine" is how a half-applied runtime
 * gets a backend started against it.
 */
function readVerdict({ file, exitCode, deps = {} }) {
  const fs = deps.fs || fsDefault;
  let parsed = null;
  try {
    parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch { /* handled below */ }
  if (parsed && typeof parsed === 'object') {
    return {
      applied: Boolean(parsed.applied),
      tag: parsed.tag || null,
      error: parsed.error || null,
      dirty: Boolean(parsed.dirty),
      terminal: Boolean(parsed.terminal),
    };
  }
  if (exitCode === 0) {
    // Exit 0 is the applier's own promise that the shell may proceed. Believe
    // it over a file we failed to read.
    return { applied: false, tag: null, error: null, dirty: false, terminal: false };
  }
  return {
    applied: false,
    tag: null,
    error: 'the updater wrote no readable verdict',
    dirty: true,
    terminal: false,
  };
}

module.exports = {
  PACKAGE_RE,
  tagOf,
  bundledPackage,
  installedTag,
  installedDigest,
  alreadyParked,
  updateDecision,
  digestFrom,
  parkBundled,
  readVerdict,
};
