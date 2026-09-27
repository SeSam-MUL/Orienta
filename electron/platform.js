/**
 * Everything the shell has to answer differently per operating system.
 *
 * One file, pure functions, no Electron: the setup wizard, `main.js` and
 * `update_endpoints.js` each used to carry their own Windows assumptions
 * (`python.exe`, `System32\tar.exe`, `%LOCALAPPDATA%`, `taskkill`, an NVIDIA
 * driver number that is a Windows version). Three copies of the same question
 * is how one of them ends up answered wrongly on a platform nobody ran.
 *
 * Every function takes the platform, the environment and the home directory as
 * arguments, defaulting to the running process, so a test can ask for macOS on
 * a Windows machine. That is the only way any of this is testable before we
 * have a Mac.
 *
 * WINDOWS BEHAVIOUR MUST NOT MOVE. The Windows answers here are the ones that
 * shipped in v0.4.5, and `tests/electron/platform.test.js` pins them.
 */
const path = require('node:path');
const os = require('node:os');

/** The three we build for. Anything else is refused, loudly and early. */
const SUPPORTED = ['win32', 'darwin', 'linux'];

/** python-build-standalone, pinned: an interpreter of another minor version
 *  would fail at pip install time -- after the download -- against lock files
 *  that pin cp311 wheels. */
const PYTHON_RELEASE = '20260901';
const PYTHON_VERSION = '3.11.16';

/** The triple python-build-standalone names its assets by. */
const PYTHON_TRIPLE = {
  win32: { x64: 'x86_64-pc-windows-msvc' },
  linux: { x64: 'x86_64-unknown-linux-gnu' },
  // macOS does not use python-build-standalone at all: the stack comes from
  // conda-forge through micromamba, because pip wheels of torch, scikit-learn
  // and faiss each bring their own OpenMP runtime and the second one to start
  // aborts the backend ("OMP: Error #15", the macOS bug report).
  darwin: {},
};

/** micromamba, pinned with its published checksum. macOS only. */
const MICROMAMBA_VERSION = '2.9.0-0';
/** Only arm64 is supported; the x64 asset exists (`micromamba-osx-64`) but an
 *  Intel Mac has no torch, so it is not built for. */
const MICROMAMBA_ASSET = { arm64: 'micromamba-osx-arm64' };
const MICROMAMBA_SHA256 = {
  'darwin-arm64': 'ec2a072f028e1a7cf20f3e2e74d5a8127cf5a5f27636375b5359811565f4e5be',
};

/**
 * CUDA 12 will not initialise below these.
 *
 * Two numbering schemes for the same CUDA release: Windows says 527.41, Linux
 * says 525.60.13. And a driver version is NOT a number -- 525.105.17 is newer
 * than 525.60.13, while as floats 525.105 < 525.6. The first revision of this
 * file stored the Linux floor as 525.6013 and would have rejected four real
 * R525 drivers, i.e. installed the CPU stack on a working CUDA machine, which
 * is the silent failure this whole probe exists to prevent. So: components,
 * compared component by component, plus the string NVIDIA itself prints, for
 * the message the user has to act on.
 */
const MIN_DRIVER_FOR_CUDA12 = {
  win32: { parts: [527, 41], display: '527.41' },
  linux: { parts: [525, 60, 13], display: '525.60.13' },
  darwin: null,
};

/** A driver version as comparable components; null when it is unreadable. */
function parseDriverVersion(text) {
  const raw = String(text ?? '').trim();
  if (!/^\d+(\.\d+)*$/.test(raw)) return null;
  return raw.split('.').map((n) => Number.parseInt(n, 10));
}

/** -1, 0, 1 -- shorter versions compare as if padded with zeroes. */
function compareVersions(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    if (x !== y) return x < y ? -1 : 1;
  }
  return 0;
}

const PYTHON_URL_ENV = 'ORIENTA_PYTHON_URL';
const MICROMAMBA_URL_ENV = 'ORIENTA_MICROMAMBA_URL';

function current(platform = process.platform) {
  return platform;
}

/** Is this something we ship for at all? Returns a reason when not. */
function supportStatus(platform = process.platform, arch = process.arch, release = null) {
  if (!SUPPORTED.includes(platform)) {
    return { supported: false, reason: `Orienta does not run on ${platform}.` };
  }
  if (platform === 'darwin') {
    if (arch !== 'arm64') {
      return {
        supported: false,
        reason: 'Orienta needs a Mac with Apple Silicon (M1 or newer). '
          + 'Intel Macs are not supported: PyTorch stopped publishing builds for them.',
      };
    }
    // Darwin 23 is macOS 14. The packages of the macOS environment need it.
    const major = Number.parseInt(String(release || os.release()).split('.')[0], 10);
    if (Number.isFinite(major) && major < 23) {
      return { supported: false, reason: 'Orienta needs macOS 14 (Sonoma) or newer.' };
    }
    return { supported: true, reason: null };
  }
  if (arch !== 'x64') {
    return { supported: false, reason: `Orienta needs a 64-bit Intel or AMD processor on ${platform}.` };
  }
  return { supported: true, reason: null };
}

/**
 * The big folder: Python, the program files, the crystal library.
 *
 * Windows keeps `%LOCALAPPDATA%\Orienta` exactly as it shipped. macOS uses
 * Application Support; Linux follows XDG. Both fall back to the home directory
 * when the variable is unset, which is normal on a fresh account.
 */
function defaultDataFolder(platform = process.platform, env = process.env, home = os.homedir()) {
  if (platform === 'win32') {
    return path.join(env.LOCALAPPDATA || path.join(home, '.local', 'share'), 'Orienta');
  }
  if (platform === 'darwin') {
    return path.join(home, 'Library', 'Application Support', 'Orienta');
  }
  return path.join(env.XDG_DATA_HOME || path.join(home, '.local', 'share'), 'Orienta');
}

/**
 * The small folder: the pointer to the data folder, and the wizard's language.
 *
 * It MUST NOT be the data folder or contain it: `setOrientaHome` refuses a data
 * folder that overlaps the pointer's folder, so a user who picks the default
 * location back would otherwise be told no. On macOS that means a sibling
 * directory, not `~/Library/Preferences` -- Apple reserves that for preference
 * files its own tools manage.
 */
function configFolder(platform = process.platform, env = process.env, home = os.homedir()) {
  if (platform === 'win32') {
    return path.join(env.APPDATA || path.join(home, '.config'), 'Orienta');
  }
  if (platform === 'darwin') {
    return path.join(home, 'Library', 'Application Support', 'Orienta Settings');
  }
  return path.join(env.XDG_CONFIG_HOME || path.join(home, '.config'), 'Orienta');
}

/** What the runtime is built from, and where it comes from. */
function runtimeSource(platform = process.platform, arch = process.arch, env = process.env) {
  if (platform === 'darwin') {
    const key = `darwin-${arch}`;
    const file = MICROMAMBA_ASSET[arch];
    if (!file) throw new Error(`no micromamba build for darwin/${arch}`);
    const override = (env[MICROMAMBA_URL_ENV] || '').trim();
    return {
      kind: 'conda',
      file,
      url: override
        || `https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/${file}`,
      sha256: MICROMAMBA_SHA256[key] || null,
      version: MICROMAMBA_VERSION,
    };
  }
  const triple = (PYTHON_TRIPLE[platform] || {})[arch];
  if (!triple) {
    throw new Error(`no Python build for ${platform}/${arch}`);
  }
  const file = `cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${triple}-install_only.tar.gz`;
  const override = (env[PYTHON_URL_ENV] || '').trim();
  return {
    kind: 'pbs',
    file,
    url: override
      || `https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${file}`,
    // python-build-standalone publishes one SHA256SUMS document per release,
    // not a file per asset; the wizard parses it by archive name.
    sha256: null,
    version: PYTHON_VERSION,
    release: PYTHON_RELEASE,
  };
}

/**
 * Where the conda environment lives on macOS.
 *
 * macOS has no `python/` directory at all: micromamba builds a conda
 * environment and the interpreter is inside it. Kept beside the root prefix
 * rather than inside it, so removing the package cache (`<root>/pkgs`) can
 * never reach the environment.
 */
function condaPrefixIn(home) {
  return path.join(home, CONDA_ENVS_DIR, 'orienta');
}

/**
 * The two directory names the macOS install adds to the data folder.
 *
 * Exported as NAMES, not only as paths, because two separate allowlists have
 * to know them: the setup refuses a data folder containing anything it does
 * not recognise, and the uninstaller removes only what it recognises. Both
 * were written for a Windows layout, so on a Mac the setup blamed the user
 * for `micromamba`/`envs` it had written itself thirty seconds earlier, and
 * the uninstaller left the environment — the largest thing on disk — behind
 * while reporting it as the user's own files. One list of names, so they
 * cannot drift apart again.
 */
const CONDA_ROOT_DIR = 'micromamba';
const CONDA_ENVS_DIR = 'envs';

/** The root prefix micromamba downloads into. */
function condaRootIn(home) {
  return path.join(home, CONDA_ROOT_DIR);
}

/**
 * The micromamba binary itself.
 *
 * Kept, not deleted after the install: the wizard's repair mode has to build
 * the environment again, and re-downloading a pinned binary to repair an
 * installation that may have no network is the wrong way round.
 */
function micromambaExeIn(home) {
  return path.join(condaRootIn(home), 'bin', 'micromamba');
}

/**
 * The interpreter inside an installed runtime.
 *
 * Three answers, not two. macOS installs from conda-forge -- its wheels of
 * torch, scikit-learn and faiss each carry an OpenMP runtime and the second
 * to initialise aborts the backend -- so there is no python-build-standalone
 * tree there and `<home>/python/bin/python3` would never exist.
 */
function interpreterIn(home, platform = process.platform) {
  if (platform === 'win32') return path.join(home, 'python', 'python.exe');
  if (platform === 'darwin') return path.join(condaPrefixIn(home), 'bin', 'python');
  return path.join(home, 'python', 'bin', 'python3');
}

/**
 * tar, by absolute path where that matters.
 *
 * On Windows `tar` on PATH may be GNU tar (Git for Windows, MSYS2, Cygwin),
 * which cannot read this archive and fails confusingly -- after a 48 MB
 * download. Elsewhere /usr/bin/tar is bsdtar (macOS) or GNU tar (Linux) and
 * both read it.
 */
function tarExe(platform = process.platform, env = process.env) {
  if (platform !== 'win32') return '/usr/bin/tar';
  const root = env.SystemRoot;
  if (!root) throw new Error('SystemRoot is not set; cannot locate the system tar');
  return path.join(root, 'System32', 'tar.exe');
}

/** The minimum NVIDIA driver for CUDA 12 as NVIDIA prints it, or null where
 *  there is no CUDA at all. This is the string for the user's message. */
function minDriverDisplay(platform = process.platform) {
  return (MIN_DRIVER_FOR_CUDA12[platform] || {}).display ?? null;
}

/**
 * Is this driver new enough for the CUDA 12 wheels?
 *
 * Fails closed in both directions that matter: an unreadable version is "no",
 * and a platform without CUDA is "no" -- a comparison against a missing floor
 * would otherwise say yes to everything (`x >= null` is `x >= 0`).
 */
function driverMeetsCuda12(driverText, platform = process.platform) {
  const floor = MIN_DRIVER_FOR_CUDA12[platform];
  if (!floor) return false;
  const parts = parseDriverVersion(driverText);
  if (!parts) return false;
  return compareVersions(parts, floor.parts) >= 0;
}

/** Can this platform have an NVIDIA card we would use? */
function canUseCuda(platform = process.platform) {
  return platform !== 'darwin';
}

/**
 * How to kill a process and everything it started.
 *
 * pip spawns children; killing only the parent leaves them running and the
 * installation half-finished. Windows has no process groups, so `taskkill /T`
 * walks the tree; elsewhere the child is started detached and the negative pid
 * signals its whole group.
 */
function killTree(pid, platform = process.platform) {
  if (platform === 'win32') {
    return { kind: 'command', command: 'taskkill', args: ['/PID', String(pid), '/T', '/F'] };
  }
  return { kind: 'signal', target: -Math.abs(pid), signal: 'SIGKILL' };
}

/**
 * Folders the data must never be put inside, because the program lives there.
 *
 * On Windows and Linux that is the directory holding the executable. On macOS
 * the executable sits at Orienta.app/Contents/MacOS/Orienta, so its own
 * directory is three levels below the thing the user drags to the Trash: the
 * BUNDLE is what has to be forbidden, or a data folder chosen inside it would
 * be deleted with the app and nobody would connect the two.
 */
function programRoots(execPath, platform = process.platform) {
  const exeDir = path.dirname(execPath);
  if (platform !== 'darwin') return [exeDir];
  const bundle = exeDir.replace(/\/Contents\/MacOS$/, '');
  return bundle === exeDir ? [exeDir] : [bundle, exeDir];
}

/**
 * What `app.relaunch()` has to be told, per platform.
 *
 * Under an AppImage the running executable is a temporary mount that vanishes
 * with the process, so relaunching it starts nothing at all. The AppImage
 * runtime puts the path of the .AppImage file in APPIMAGE, and that is what
 * has to be started again. Everywhere else the default is right.
 */
function relaunchOptions(env = process.env) {
  const image = (env.APPIMAGE || '').trim();
  return image ? { execPath: image, args: [] } : undefined;
}

/**
 * Where to open the window so it lands on the display the wizard was on.
 *
 * The setup wizard finishes by relaunching the process (app.relaunch), and the
 * new one creates its window with no coordinates — so the OS puts it on the
 * primary display. The M5 tester, who has three monitors, ran the wizard on
 * one screen and found Orienta on another.
 *
 * `point` is a spot that was inside the old window; whichever display's work
 * area contains it is the display to use. Returns null when no display does,
 * which is what happens if that monitor has since been unplugged — then the
 * caller passes no coordinates and the OS decides, exactly as before.
 *
 * Pure: takes the display list rather than reaching for Electron's `screen`.
 */
function windowPositionFor(displays, point, size) {
  if (!point || !Array.isArray(displays) || !size) return null;
  const { x: px, y: py } = point;
  if (!Number.isFinite(px) || !Number.isFinite(py)) return null;

  const home = displays.find((d) => {
    const a = (d && (d.workArea || d.bounds)) || null;
    return a && px >= a.x && px < a.x + a.width && py >= a.y && py < a.y + a.height;
  });
  if (!home) return null;

  const area = home.workArea || home.bounds;
  // Centred, but never off the top-left of that display: a window taller than
  // the work area would otherwise get a negative offset and lose its title bar.
  return {
    x: Math.round(area.x + Math.max(0, (area.width - size.width) / 2)),
    y: Math.round(area.y + Math.max(0, (area.height - size.height) / 2)),
  };
}

module.exports = {
  SUPPORTED,
  PYTHON_RELEASE,
  PYTHON_VERSION,
  MICROMAMBA_VERSION,
  MIN_DRIVER_FOR_CUDA12,
  parseDriverVersion,
  compareVersions,
  minDriverDisplay,
  driverMeetsCuda12,
  current,
  supportStatus,
  defaultDataFolder,
  configFolder,
  runtimeSource,
  interpreterIn,
  condaPrefixIn,
  condaRootIn,
  micromambaExeIn,
  CONDA_ROOT_DIR,
  CONDA_ENVS_DIR,
  tarExe,
  canUseCuda,
  killTree,
  programRoots,
  relaunchOptions,
  windowPositionFor,
};
