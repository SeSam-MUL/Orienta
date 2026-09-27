/**
 * The JavaScript twin of everything in github_releases.py that the FIRST
 * INSTALL needs — which runs before any backend exists and so cannot ask Python
 * for any of it.
 *
 * Every function here has a Python counterpart, and
 * tests/test_update_endpoint_contract.py runs both and compares. Revision 1 of
 * this work shipped an asset name that could never match because each side was
 * asserted only against itself; a later revision repeated it with a digest
 * parser that refused the checksum file this project publishes.
 *
 * The environment-variable names are duplicated deliberately — a require of the
 * Python module is impossible here — and the contract test sets each variable
 * and checks that BOTH sides honour it.
 */
const fs = require('node:fs');
const platform = require('./platform.js');
const path = require('node:path');
const os = require('node:os');

const DEFAULT_REPO = 'SeSam-MUL/Orienta';
const DEFAULT_API_ROOT = 'https://api.github.com';
const DEFAULT_WEB_ROOT = 'https://github.com';

const REPO_SLUG_ENV = 'ORIENTA_UPDATE_REPO';
const API_ROOT_ENV = 'ORIENTA_GITHUB_API';
const RELEASES_URL_ENV = 'ORIENTA_RELEASES_URL';
const HOME_ENV = 'ORIENTA_HOME';

function repoSlug() {
  return (process.env[REPO_SLUG_ENV] || '').trim() || DEFAULT_REPO;
}

function githubApiRoot() {
  return ((process.env[API_ROOT_ENV] || '').trim() || DEFAULT_API_ROOT).replace(/\/+$/, '');
}

/** Where to send a user in a BROWSER. Not derived from the API root, which
 *  would show them JSON. */
function releasesPageUrl() {
  const override = (process.env[RELEASES_URL_ENV] || '').trim();
  if (override) return override.replace(/\/+$/, '');
  return `${DEFAULT_WEB_ROOT}/${repoSlug()}/releases`;
}

function releaseByTagUrl(tag) {
  return `${githubApiRoot()}/repos/${repoSlug()}/releases/tags/${tag}`;
}

/**
 * The newest published release.
 *
 * What a bootstrap installer must ask for. Asking for its OWN version instead
 * pins a 100 MB download to one release for ever: once that release is
 * superseded or removed, every copy of that installer still in existence is
 * dead, with a 404 naming a tag the user never typed. The whole point of a
 * small starter is that it stays valid while the thing it installs moves on.
 *
 * GitHub excludes drafts AND pre-releases from this endpoint, which is why
 * `releaseListUrl` exists beside it.
 */
function latestReleaseUrl() {
  return `${githubApiRoot()}/repos/${repoSlug()}/releases/latest`;
}

/**
 * Every release, newest first, for the "install a different version" list.
 *
 * Also the fallback when `latest` returns nothing: a repository whose only
 * releases are pre-releases has no "latest" at all, and answering that with
 * "this version has not been published yet" would be wrong twice over.
 */
function releaseListUrl(perPage = 30) {
  return `${githubApiRoot()}/repos/${repoSlug()}/releases?per_page=${perPage}`;
}

function runtimeAssetNames(tag) {
  const pkg = `orienta-runtime-${tag}.zip`;
  return { package: pkg, checksum: `${pkg}.sha256` };
}

/**
 * Exact match, never a substring: the checksum asset's name CONTAINS the
 * package's, so `find(a => a.name.includes(...))` returns the wrong one.
 */
function assetUrl(release, exactName) {
  // A mirror or GitHub Enterprise instance may answer with a differently shaped
  // body — this module is built to run against an arbitrary ORIENTA_GITHUB_API —
  // so `[null]`, a list of bare strings, or a non-object release must yield
  // null rather than throwing out of a function whose contract is "url or null".
  if (!release || typeof release !== 'object' || !Array.isArray(release.assets)) {
    return null;
  }
  const asset = release.assets.find(
    (a) => a && typeof a === 'object' && a.name === exactName,
  );
  if (!asset) return null;
  // Explicitly null, not undefined: the Python twin returns None, and a
  // contract test compares the two as strings.
  const url = asset.browser_download_url;
  return typeof url === 'string' && url ? url : null;
}

/**
 * One entry of a checksum document: a digest, optionally followed by a file
 * name ("*" marks binary mode in the sha256sum convention).
 *
 * The whitespace class is spelled out as [ \t] rather than \s, because
 * JavaScript and Python do not agree on what whitespace is and this function
 * has a Python twin. Measured: JS's \s matches U+FEFF (the BOM) and Python's
 * does not, while Python's str.split() treats U+001C..1F and U+0085 as
 * separators and JS's \s does not. PowerShell writes a BOM by default, so that
 * divergence was reachable simply by regenerating a .sha256 file on Windows.
 */
const ENTRY_RE = /^[ \t]*([0-9a-fA-F]{64})(?:[ \t]+\*?([^\r\n]*?))?[ \t]*$/gm;

/**
 * The digest out of a .sha256 (or SHA256SUMS) document.
 *
 * Three producers disagree. The release driver writes "<hex>  <filename>";
 * python-build-standalone publishes per-asset files of bare hex AND an
 * aggregate SHA256SUMS listing every asset. Pass `filename` whenever the
 * document might be an aggregate: without it a multi-entry document is REFUSED
 * rather than answered with its first line, which would be the digest of a
 * different file and would fail every install with a mismatch that blames the
 * download.
 *
 * Throws on anything else, including the HTML login page a captive portal
 * serves instead. Accepts a Buffer as well as a string.
 */
function parseSha256Document(text, filename = null) {
  // Strip ALL leading BOMs and fold every line terminator either language
  // recognises to "\n". Python's re.MULTILINE honours only "\n"; this /m regex
  // also honours "\r", U+2028 and U+2029, and Python's lstrip(BOM) removes a
  // run rather than one. Each was a measured divergence, found by a
  // differential test rather than by reading the code.
  const doc = String(text == null ? '' : text)
    .replace(/^\uFEFF+/, '')
    .replace(/\r\n|\r|\u2028|\u2029/g, '\n');

  const entries = [];
  ENTRY_RE.lastIndex = 0;          // the /g flag makes this regex stateful
  let m = ENTRY_RE.exec(doc);
  while (m !== null) {
    entries.push({ digest: m[1].toLowerCase(), name: (m[2] || '').trim() });
    m = ENTRY_RE.exec(doc);
  }

  if (entries.length === 0) {
    throw new Error(`not a SHA-256 digest document (starts with "${doc.slice(0, 40)}")`);
  }

  const baseName = (p) => String(p).replace(/\\/g, '/').split('/').pop();

  if (filename != null) {
    const want = baseName(filename);
    const hit = entries.find((e) => e.name && baseName(e.name) === want);
    if (!hit) {
      throw new Error(
        `the checksum document lists ${entries.length} file(s), none of them "${want}"`);
    }
    return hit.digest;
  }

  if (entries.length > 1) {
    throw new Error(
      `the checksum document lists ${entries.length} files; say which one is `
      + 'wanted by passing filename=');
  }
  return entries[0].digest;
}

/**
 * Where the installed environment lives. The single JavaScript definition of
 * it: the shell and the setup both take it from here rather than each building
 * their own. The Python twin is bundle_update.orienta_home(), which does not
 * exist yet — when it does, the contract test must gain a row for this pair, as
 * it has for every other value computed in both languages.
 *
 * An override is validated, not trusted: this path is the parent of a directory
 * the updater recursively deletes and of the user's crystal library, and a
 * relative path would resolve against whatever cwd the process happens to have.
 */
/**
 * Where the pointer to a chosen data folder lives.
 *
 * Deliberately NOT inside the Orienta home — that would be circular — and not
 * inside the application directory, which an update or a reinstall deletes.
 * A few bytes in the roaming profile, which is exactly what this is: a
 * per-user preference that must survive everything else.
 */
function homePointerFile({ platformName = process.platform, env = process.env } = {}) {
  return path.join(platform.configFolder(platformName, env), 'home.txt');
}

/**
 * The folder holding Python, the program and the crystal library.
 *
 * This is the one that gets big: 2.5 GB for the processor build, 8 GB for the
 * graphics-card build, plus a crystal library that grows for years. On a
 * laptop with a small system drive that is precisely the thing a user needs to
 * put somewhere else — and until the pointer file existed, the only way to
 * move it was an environment variable, which is to say: not a way at all.
 *
 * Precedence: the environment wins (scripts and CI set it), then the user's
 * recorded choice, then the default beside every other application's data.
 */
function orientaHome({ platformName = process.platform, env = process.env } = {}) {
  const override = (env[HOME_ENV] || '').trim();
  if (override) {
    if (!path.isAbsolute(override)) {
      throw new Error(`${HOME_ENV} must be an absolute path, got "${override}"`);
    }
    return override;
  }
  try {
    const recorded = readPointer(homePointerFile({ platformName, env }));
    // An unreadable or relative pointer falls through to the default rather
    // than throwing: this function runs at module load, and a bad byte in a
    // preference file must not be the reason the application cannot start.
    if (recorded && path.isAbsolute(recorded)) return recorded;
  } catch { /* no choice recorded */ }
  return platform.defaultDataFolder(platformName, env);
}

/**
 * Read the pointer file, in either encoding it may have been written in.
 *
 * Written as UTF-16LE with a byte-order mark, because the UNINSTALLER reads it
 * too and NSIS cannot read UTF-8: it decodes through the ANSI code page, so a
 * folder under "OneDrive - Montanuniversität Leoben" would come back mangled,
 * the uninstaller would look in a folder that does not exist, and the data
 * would be left behind with no explanation. UTF-8 is still accepted on the way
 * in, so a hand-edited file works.
 */
function readPointer(file) {
  const raw = fs.readFileSync(file);
  let text;
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    text = raw.subarray(2).toString('utf16le');
  } else {
    text = raw.toString('utf8').replace(/^\uFEFF/, '');
  }
  return text.replace(/[\r\n]+/g, '').trim();
}

/** Record a chosen data folder, or forget it with null. */
/** Do `a` and `b` overlap — equal, or one inside the other? Case-insensitive,
 *  as Windows paths are. */
function overlaps(a, b) {
  const x = path.resolve(a).toLowerCase().replace(/[\\/]+$/, '');
  const y = path.resolve(b).toLowerCase().replace(/[\\/]+$/, '');
  return x === y || x.startsWith(y + path.sep) || y.startsWith(x + path.sep);
}

/**
 * Record a chosen data folder, or forget it with null.
 *
 * `forbidden` lists further folders the data must not overlap, supplied by the
 * caller that knows them — the application's own install directory, which the
 * application uninstaller removes wholesale.
 */
function setOrientaHome(dir, {
  forbidden = [], platformName = process.platform, env = process.env,
} = {}) {
  const file = homePointerFile({ platformName, env });
  fs.mkdirSync(path.dirname(file), { recursive: true });
  if (!dir) {
    fs.rmSync(file, { force: true });
    return orientaHome({ platformName, env });
  }
  if (!path.isAbsolute(dir)) throw new Error(`the folder must be an absolute path: ${dir}`);
  // Never the folder holding this pointer. electron-builder's uninstaller has
  // a `--delete-app-data` switch that runs `RMDir /r` on exactly that folder,
  // AFTER Orienta's own careful step — so a data folder there would be deleted
  // library and all, whatever the user answered. Picking %APPDATA% in the
  // "change…" dialog was enough to get there.
  for (const place of [path.dirname(file), ...forbidden]) {
    if (place && overlaps(dir, place)) {
      throw new Error(`Orienta's data cannot be kept in ${dir}: it overlaps ${place}, `
        + 'which is removed when Orienta itself is uninstalled');
    }
  }
  // Proved writable BEFORE it is recorded. Recording a folder that cannot be
  // written would hand Chromium an unusable path on the next launch, and this
  // project has already seen what that does: the app does not start at all.
  fs.mkdirSync(dir, { recursive: true });
  const probe = path.join(dir, '.writable');
  fs.writeFileSync(probe, '');
  fs.rmSync(probe, { force: true });
  // UTF-16LE with a BOM, for the uninstaller's sake — see readPointer.
  fs.writeFileSync(file, Buffer.concat([
    Buffer.from([0xff, 0xfe]),
    Buffer.from(`${dir}\r\n`, 'utf16le'),
  ]));
  return dir;
}

module.exports = {
  DEFAULT_REPO,
  DEFAULT_API_ROOT,
  DEFAULT_WEB_ROOT,
  REPO_SLUG_ENV,
  API_ROOT_ENV,
  RELEASES_URL_ENV,
  HOME_ENV,
  repoSlug,
  githubApiRoot,
  releasesPageUrl,
  releaseByTagUrl,
  latestReleaseUrl,
  releaseListUrl,
  runtimeAssetNames,
  assetUrl,
  parseSha256Document,
  orientaHome,
  setOrientaHome,
  homePointerFile,
  readPointer,
};
