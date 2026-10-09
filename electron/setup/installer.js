/**
 * First-run setup: build the Python side of the application.
 *
 * Sizes are MEASURED, not estimated (2026-09-17, this project's own stack):
 * CPU environment 1.87 GB, GPU environment 7.97 GB, runtime package 4.4 MB,
 * standalone interpreter 48 MB.
 *
 * The GPU decision is pre-made but never hidden. CUDA 12 requires NVIDIA driver
 * 527.41 or newer, and installing the CUDA stack onto an older driver produces
 * an application that runs, reports no error, and computes everything on the
 * CPU — measured in this repository at 11.2 TFLOP/s against 0.71, a factor of
 * 16 that nobody would recognise as a defect. So the driver is read, the
 * recommendation is explained in terms the person can act on, and they can
 * still choose the other option.
 */
const { execFile } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { promisify } = require('node:util');

const platform = require('../platform.js');
const macosEnv = require('./macos_env.js');

const execFileAsync = promisify(execFile);

/** CUDA 12 will not initialise below this. The number is load-bearing: it is
 *  the difference between a working GPU install and a silent CPU one -- and it
 *  is a WINDOWS driver version; Linux numbers the same CUDA release 525.60.13,
 *  so it comes from the platform resolver rather than from a constant here. */
const MIN_DRIVER_FOR_CUDA12 = platform.minDriverDisplay();

const GB = 1024 ** 3;
/**
 * 1.87 GB measured + headroom for pip's build/unpack space.
 *
 * WINDOWS AND LINUX ONLY. macOS installs a different stack a different way,
 * and 2.5 GB there is not a headroom question but a wrong answer — see
 * `cpuBytesNeeded`.
 */
const CPU_BYTES_NEEDED = Math.round(2.5 * GB);

/**
 * macOS needs its own floor. This one is still an ESTIMATE -- but an
 * estimate with one real measurement under it, which the 7 GB it replaces
 * did not have.
 *
 * The Windows number comes from watching a pip install. On macOS the peak is
 * the conda environment PLUS the package cache, which is only deleted after
 * the transaction finishes — so the two exist at once, at the worst moment.
 *
 * The 7 GB that shipped in v0.4.5 was an estimate, and it asked to be replaced
 * by a real number. The M5 tester supplied one on 2026-09-25:
 * `~/Library/Application Support/Orienta` held **9.89 GB** (10.48 GB on disk,
 * 267,079 objects). That is already above the floor the wizard was promising.
 *
 * What that 9.89 GB does NOT separate: he had by then built an Al master, a Si
 * master and their Monte-Carlo files, which live in the same folder. So the
 * install alone is less than 9.89 GB — how much less, nobody has measured. The
 * number that would settle it is `du -sh` on that folder immediately after the
 * wizard finishes and before anything is simulated; MAC-TEST.md now asks for
 * it.
 *
 * Until then: 12 GB = the measured folder plus room for the package cache that
 * coexists with the environment during the transaction. It stays deliberately
 * generous because the two errors are not symmetric — too low means a failure
 * part way through a long install, too high means a clear refusal before
 * anything is downloaded.
 */
const MACOS_CPU_BYTES_NEEDED = Math.round(12 * GB);

function cpuBytesNeeded(platformName = process.platform) {
  return platformName === 'darwin' ? MACOS_CPU_BYTES_NEEDED : CPU_BYTES_NEEDED;
}

/**
 * Lines that belong in the log but not on the wizard's progress line.
 *
 * Only micromamba's own `warning libmamba ...` channel, which opens every
 * transaction with the same paragraph about package scripts being able to
 * contain arbitrary code. It is boilerplate, it is English, and it appeared
 * mid-install under a German heading. Nothing is dropped: `record` still
 * writes every line, so the setup log and the diagnostics zip are unchanged.
 *
 * Narrow in consequence, not in pattern: this hides EVERY line on
 * micromamba's warning channel, not only the package-scripts paragraph --
 * a bad cache or a missing channel would be hidden from the progress line
 * too. They stay in the log, and an ERROR from micromamba is not a
 * `warning libmamba` line, so failures still reach the screen.
 */
function isSetupNoise(line) {
  return /^warning\s+libmamba\b/i.test(String(line || '').trim());
}
/** 7.97 GB measured + headroom; the CUDA wheels unpack larger than they download. */
const GPU_BYTES_NEEDED = Math.round(9 * GB);

// python-build-standalone. Pinned, because an interpreter of another minor
// version would fail at pip install time — after the download — against lock
// files that pin cp311 wheels.
const PYTHON_RELEASE = platform.PYTHON_RELEASE;
// Resolved lazily: requiring this module on a platform it has no build for
// must not throw a stack trace before the wizard can show its refusal.
let _source = null;
function runtimeSourceOrNull() {
  if (_source === null) {
    try { _source = platform.runtimeSource(); } catch { _source = false; }
  }
  return _source || null;
}
const PYTHON_FILE = (runtimeSourceOrNull() || {}).file || null;
const DEFAULT_PYTHON_URL = (() => {
  try { return platform.runtimeSource(process.platform, process.arch, {}).url; }
  catch { return null; }
})();

const PYTHON_URL_ENV = 'ORIENTA_PYTHON_URL';
const PYTHON_SHA256_URL_ENV = 'ORIENTA_PYTHON_SHA256_URL';

/**
 * Where the interpreter comes from. Overridable, like every other external URL
 * here: a pinned URL the project does not control is a single point of failure
 * for every future install — the release can be yanked, the host blocked, a
 * lab can need a mirror.
 */
/**
 * The interpreter path of this file only knows python-build-standalone.
 *
 * macOS builds its runtime from conda-forge through micromamba (T3), which is
 * a different archive, a different checksum document and a different unpack.
 * Until that path exists, say so instead of running a Mach-O binary through
 * `tar xzf` and blaming the download for the mismatch.
 */
function requirePythonBuildStandalone() {
  const source = platform.runtimeSource();
  if (source.kind !== 'pbs') {
    throw new Error(
      `the interpreter install path does not support ${process.platform} yet `
      + `(it needs ${source.kind}); this build is Windows and Linux only`);
  }
  return source;
}

function pythonUrl() {
  const override = (process.env[PYTHON_URL_ENV] || '').trim();
  if (override) return override;
  return requirePythonBuildStandalone().url;
}

/** The name of the archive inside the checksum document. */
function pythonArchiveName() {
  return pythonUrl().split('/').pop().split('?')[0];
}

/**
 * Where the interpreter's digest comes from.
 *
 * MEASURED 2026-09-20 against release 20260901: that release publishes 871
 * assets, exactly ZERO of them `.sha256`, and one aggregate `SHA256SUMS`.
 * Deriving `<archive>.sha256` gave a 404, and since the setup refuses to
 * install without a digest, that would have made every install impossible —
 * after the 48 MB download.
 *
 * So the default is the aggregate, and callers MUST pass
 * `pythonArchiveName()` to the parser: answering a multi-entry document with
 * its first line returns the digest of a different file and fails with a
 * mismatch that blames the download.
 */
function pythonSha256Url() {
  const override = (process.env[PYTHON_SHA256_URL_ENV] || '').trim();
  if (override) return override;
  requirePythonBuildStandalone();
  return `https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/SHA256SUMS`;
}

/**
 * What `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`
 * told us, or "no GPU".
 *
 * An unreadable driver version is reported as NOT cuda-capable. A machine whose
 * driver we cannot read is not a machine to hand the CUDA stack to.
 */
function parseNvidiaSmi(stdout, platformName = process.platform) {
  const line = String(stdout ?? '').split('\n').map((s) => s.trim()).filter(Boolean)[0];
  if (!line) return { present: false, cudaCapable: false, name: null, driver: null };

  const [name, driver] = line.split(',').map((s) => s.trim());
  // A driver version is not a number: 525.105.17 is NEWER than 525.60.13, and
  // as floats it is smaller. The comparison happens on components, in the
  // platform resolver, which also knows that a Mac has no CUDA at all.
  //
  // The platform is an ARGUMENT, like every neighbour in this file. It was the
  // only one resolving the host silently, and the first macOS run showed why
  // that matters: on a Mac `cudaCapable` is correctly false for every driver
  // string, so tests written against the CUDA rules had no way to ask for
  // them. Default unchanged, so nothing shipping behaves differently.
  const parts = platform.parseDriverVersion(driver);
  const known = parts !== null;
  return {
    present: true,
    name: name || null,
    driver: known ? driver : null,
    cudaCapable: known && platform.driverMeetsCuda12(driver, platformName),
  };
}

/**
 * Free bytes on the volume holding `dir`, or 0 when we cannot tell.
 *
 * ZERO, not Infinity. An unknown amount of space is not unlimited space, and
 * the permissive answer here recommends an 8 GB install to a machine that
 * cannot take it — which fails part-way, after the download.
 */
function freeBytesOn(dir) {
  return measureFree(dir).bytes;
}

/** The same measurement, keeping WHETHER it succeeded.
 *
 * `freeBytesOn` returning 0 is a deliberate fail-closed choice, but rendering
 * that 0 as "Only 0.0 GB free" tells a user with 500 GB that their disk is
 * full. On a university-managed profile with LOCALAPPDATA redirected to a UNC
 * share, that is the DEFAULT outcome.
 */
function measureFree(dir) {
  if (!dir) return { bytes: 0, known: false };
  try {
    const stats = fs.statfsSync(dir);
    return { bytes: stats.bavail * stats.bsize, known: true };
  } catch {
    return { bytes: 0, known: false };
  }
}

/**
 * Which flavour to install, and why — in words the person can act on.
 *
 * `blocked` means there is not room for the smaller of the two, so there is
 * nothing to offer and the wizard must say so before downloading anything.
 */
function chooseRecommendation(gpu, freeBytes, freeKnown = true,
                              platformName = process.platform) {
  const free = Number.isFinite(freeBytes) ? freeBytes : 0;
  const asGb = (bytes) => (bytes / GB).toFixed(1);
  // macOS installs a different stack a different way and needs a
  // different floor; everywhere else this is the number that shipped.
  const CPU_NEEDED = cpuBytesNeeded(platformName);

  if (free < CPU_NEEDED) {
    return {
      recommendation: 'cpu',
      blocked: true,
      freeBytes: free,
      freeKnown,
      neededBytes: CPU_NEEDED,
      reason: freeKnown
        ? `Only ${asGb(free)} GB free — Orienta needs at least `
          + `${asGb(CPU_NEEDED)} GB. Free some space and try again.`
        : 'The free space on this drive could not be read, so Orienta cannot '
          + 'tell whether it will fit. This happens when the user folder is on '
          + 'a network drive. Set ORIENTA_HOME to a local folder and try again.',
      reasonCode: freeKnown ? 'tooLittleSpace' : 'freeSpaceUnknown',
      reasonValues: freeKnown ? { free: asGb(free), needed: asGb(CPU_NEEDED) } : {},
    };
  }

  const card = gpu || { present: false };

  // canUseCuda(platformName), not canUseCuda(). Called bare it answered for
  // the HOST while this function had been handed a platform to answer for, so
  // the single most important branch quietly ignored its own argument: asked
  // about darwin from Windows it fell through to "update your NVIDIA driver",
  // and asked about win32 from a Mac it returned the Mac text. Harmless while
  // the default is the host -- and it made the Mac behaviour impossible to
  // check anywhere except on a Mac, which is how it survived.
  if (!platform.canUseCuda(platformName)) {
    // A Mac has no CUDA at all, so there is nothing to detect and nothing to
    // update. Saying "no NVIDIA card found — update your driver" to someone on
    // an Apple machine reads as a fault they could fix.
    return {
      recommendation: 'cpu',
      freeBytes: free,
      freeKnown,
      neededBytes: CPU_NEEDED,
      reason: `Orienta uses the processor on this Mac (${asGb(CPU_NEEDED)} GB). `
        + `Graphics-card acceleration needs an NVIDIA card, which Apple computers do not have.`,
      // The wizard has four languages; this sentence had one. The code lets
      // the renderer say it in the user's, the same way `warningCode` already
      // does for cudaUnverified. The prose above stays as the fallback for an
      // older renderer and for the log.
      reasonCode: 'macosCpuOnly',
      reasonValues: { size: asGb(CPU_NEEDED) },
    };
  }

  if (!card.present) {
    if (card.driverBroken) {
      return {
        recommendation: 'cpu',
        freeBytes: free,
      freeKnown,
        neededBytes: CPU_NEEDED,
        reason: 'An NVIDIA driver is installed but is not responding, so Orienta '
          + 'cannot use the graphics card. Reinstalling or updating the NVIDIA '
          + 'driver usually fixes this. Orienta will use the processor for now.',
      };
    }
    return {
      recommendation: 'cpu',
      freeBytes: free,
      freeKnown,
      neededBytes: CPU_NEEDED,
      reason: `No NVIDIA graphics card found — Orienta will use the processor `
        + `(${asGb(CPU_NEEDED)} GB).`,
    };
  }

  if (!card.cudaCapable) {
    const driver = card.driver === null ? 'an unreadable version' : card.driver;
    return {
      recommendation: 'cpu',
      freeBytes: free,
      freeKnown,
      neededBytes: CPU_NEEDED,
      // The number is in front of them because updating the driver is a real
      // option, and they can only take it if they know which number is wrong.
      reason: `${card.name || 'This graphics card'} reports driver ${driver}. `
        + `GPU acceleration needs ${MIN_DRIVER_FOR_CUDA12} or newer. Update the `
        + `NVIDIA driver to use the card, or continue on the processor.`,
    };
  }

  if (free < GPU_BYTES_NEEDED) {
    return {
      recommendation: 'cpu',
      freeBytes: free,
      freeKnown,
      neededBytes: GPU_BYTES_NEEDED,
      reason: `${card.name} found, but only ${asGb(free)} GB free disk space — the `
        + `GPU version needs ${asGb(GPU_BYTES_NEEDED)} GB. The processor version `
        + `needs ${asGb(CPU_NEEDED)} GB.`,
    };
  }

  return {
    recommendation: 'gpu',
    freeBytes: free,
      freeKnown,
    neededBytes: GPU_BYTES_NEEDED,
    reason: `${card.name}, driver ${card.driver} — GPU acceleration recommended `
      + `(${asGb(GPU_BYTES_NEEDED)} GB).`,
  };
}

/** Ask the driver, or conclude there is no NVIDIA card. Never throws. */
async function detectGpu() {
  try {
    const { stdout } = await execFileAsync(
      'nvidia-smi',
      ['--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
      // windowsHide, like every child process this project spawns: without it
      // the probe flashes a console window over the wizard.
      { timeout: 15000, windowsHide: true },
    );
    return parseNvidiaSmi(stdout);
  } catch (err) {
    // nvidia-smi MISSING is the answer on a machine with no NVIDIA card.
    // nvidia-smi PRESENT AND FAILING is a different thing entirely — a stale
    // driver after a Windows update, a laptop in hybrid mode, a driver that
    // cannot talk to its card. Reporting that as "no graphics card" routes a
    // working RTX 4090 to the CPU and silently costs the factor of 16 this
    // module's header is written about.
    if (err && err.code === 'ENOENT') {
      return { present: false, cudaCapable: false, name: null, driver: null };
    }
    return {
      present: false,
      cudaCapable: false,
      name: null,
      driver: null,
      driverBroken: true,
      detail: String((err && err.message) || err).split('\n')[0].slice(0, 200),
    };
  }
}

/**
 * Everything the first screen needs, measured rather than assumed.
 *
 * The Orienta home does not exist on a first run, so the free-space question is
 * asked of the nearest ancestor that does — measuring a directory that is not
 * there returns 0, which would block every first install.
 */
async function probe(home) {
  const gpu = await detectGpu();

  let target = home;
  while (target) {
    try {
      if (fs.statSync(target).isDirectory()) break;
    } catch {
      /* not there, or cannot tell — walk up */
    }
    const parent = path.dirname(target);
    if (parent === target) break;
    target = parent;
  }

  const { bytes, known } = measureFree(target);
  return { gpu, ...chooseRecommendation(gpu, bytes, known) };
}

module.exports = {
  MIN_DRIVER_FOR_CUDA12,
  measureFree,
  CPU_BYTES_NEEDED,
  MACOS_CPU_BYTES_NEEDED,
  cpuBytesNeeded,
  isSetupNoise,
  GPU_BYTES_NEEDED,
  PYTHON_URL_ENV,
  PYTHON_SHA256_URL_ENV,
  DEFAULT_PYTHON_URL,
  pythonUrl,
  pythonArchiveName,
  pythonSha256Url,
  parseNvidiaSmi,
  chooseRecommendation,
  freeBytesOn,
  detectGpu,
  probe,
};

// ==========================================================================
// installing
// ==========================================================================

const {
  parseSha256Document,
  releaseByTagUrl,
  latestReleaseUrl,
  releaseListUrl,
  runtimeAssetNames,
  assetUrl,
} = require('../update_endpoints');

const DEFAULT_PYTORCH_GPU = 'https://download.pytorch.org/whl/cu126';
const DEFAULT_PYTORCH_CPU = 'https://download.pytorch.org/whl/cpu';
const PYTORCH_GPU_ENV = 'ORIENTA_PYTORCH_INDEX_GPU';
const PYTORCH_CPU_ENV = 'ORIENTA_PYTORCH_INDEX_CPU';

/** 'gpu' or anything else, which means cpu — the smaller, safer install is the
 *  right answer to a question we could not read. */
/** Installed only by the GPU lock file, and never removed by pip on the way
 *  back down, because the CPU lock file does not mention them. */
const CUDA_ONLY_PACKAGES = [
  'cupy-cuda12x',
  'nvidia-cublas-cu12', 'nvidia-cuda-cupti-cu12', 'nvidia-cuda-nvrtc-cu12',
  'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12', 'nvidia-cufft-cu12',
  'nvidia-curand-cu12', 'nvidia-cusolver-cu12', 'nvidia-cusparse-cu12',
  'nvidia-nccl-cu12', 'nvidia-nvjitlink-cu12', 'nvidia-nvtx-cu12',
  // Linux resolves several CUDA packages Windows never sees; without them a
  // switch from the graphics-card install to the processor one leaves
  // gigabytes behind and the user is told it was undone.
  'nvidia-cusparselt-cu12', 'nvidia-nvshmem-cu12', 'cuda-bindings',
  'cuda-pathfinder', 'cuda-toolkit', 'nvidia-cufile-cu12', 'triton',
  // The same package under the name Windows installs it by. It has been
  // missing since the list was written, on the platform that ships.
  'triton-windows',
];

/**
 * The lock file for this machine: the flavour AND the platform.
 *
 * The Windows locks are frozen from an environment somebody ran; the Linux
 * ones are resolved for Linux by scripts/make_linux_locks.py. Installing
 * Windows's answer on Linux fails on the first CUDA wheel, so an unknown
 * platform is refused rather than given the Windows file.
 */
function lockFileFor(mode, platformName = process.platform) {
  const flavour = mode === 'gpu' ? 'gpu' : 'cpu';
  if (platformName === 'win32') return `requirements-lock-${flavour}.txt`;
  if (platformName === 'linux') return `requirements-lock-linux-${flavour}.txt`;
  // macOS builds its environment from conda-forge (T3), not from a pip lock.
  throw new Error(`no pip lock file for ${platformName}`);
}

/**
 * May a resumed run skip straight to the end?
 *
 * Only macOS can answer no. There, a resume skips steps 2, 3 AND 4 — and step
 * 3 is the only thing that creates an interpreter, because the environment is
 * built inside it. Windows and Linux are covered further down by step 4's
 * "the lock file is missing" check; macOS has no step 4 at all, so without
 * this a resumed run walks to the end and writes `.python_path` for an
 * interpreter that is not there, reporting a finished installation that
 * cannot start.
 *
 * Not reachable today — every darwin failure returns resumable=false — but
 * `resumable = true` is written unconditionally, so this is one new failure
 * path away from being reachable, and it fails silently when it is.
 */
function resumeHasSomethingToContinueFrom(home, platformName = process.platform,
                                          exists = fs.existsSync) {
  if (platformName !== 'darwin') return true;
  return exists(platform.interpreterIn(home, platformName));
}

function pythonExeIn(home) {
  return platform.interpreterIn(home);
}

/**
 * Windows' own bsdtar, by absolute path, derived from the environment.
 *
 * `tar` on PATH may be GNU tar (Git for Windows, MSYS2, Cygwin), which cannot
 * read this archive and fails confusingly — after a 48 MB download.
 */
function tarExe() {
  return platform.tarExe();
}

/** Overridable, like every other external URL here: 2 to 8 GB of every install
 *  comes through these two, and a lab may need a mirror. */
function pytorchIndex(mode) {
  return mode === 'gpu'
    ? (process.env[PYTORCH_GPU_ENV] || '').trim() || DEFAULT_PYTORCH_GPU
    : (process.env[PYTORCH_CPU_ENV] || '').trim() || DEFAULT_PYTORCH_CPU;
}

/**
 * pip arguments for one flavour.
 *
 * The CUDA index must NOT be offered in cpu mode. `torch==2.11.0+cu126`
 * satisfies `==2.11.0` AND sorts above it, so merely offering the index hands
 * the ~8 GB wheel to exactly the user who was routed to CPU for lack of disk
 * space.
 */
function pipIndexArgs(mode) {
  return [
    '--extra-index-url', pytorchIndex(mode),
    // Closes exactly one thing: arbitrary code execution from an sdist's
    // setup.py during resolution. Not a general safety measure.
    '--only-binary', ':all:',
    // pip suppresses its bar on a pipe anyway; saying so keeps the output
    // parseable and makes the heartbeat the single source of "still working".
    '--progress-bar', 'off',
    // pip's HTTP cache defaults to %LOCALAPPDATA%\pip\Cache — the SAME volume
    // freeBytesOn just measured. Without this, peak usage is the download PLUS
    // the installed tree, so a machine that passes the 9 GB check by a few GB
    // still fails part-way through an 8 GB install. The cache is never reused
    // here: the setup runs once.
    '--no-cache-dir',
  ];
}

/**
 * Reject a lock file that would redirect pip or fetch from a URL.
 *
 * This defends against a carelessly edited lock file. NOT against a hostile
 * release: whoever can publish a release also ships the Python that gets
 * imported, so auditing one text file while importing unaudited code proves
 * nothing. The control that WOULD bind an install to exact bytes is
 * `--require-hashes` against a lock generated with hashes.
 *
 * None of the four lock files carries hashes today, and that is a decision,
 * not an oversight: hashes are all-or-nothing (one `--hash=` puts pip into
 * hash-checking mode, where every requirement needs one and no unpinned
 * dependency may appear), and the Windows locks are frozen from a working
 * environment rather than resolved, so they have no hashes to freeze. Adding
 * them would have to happen for all four at once, together with
 * `--require-hashes` here. Until then this function is a guard against a
 * mistake, and says so. The `--hash=` lines it already tolerates are there so
 * that adopting hashes does not also require changing this check.
 */
function lockFileIsSafe(text) {
  const allowed = [
    DEFAULT_PYTORCH_GPU, DEFAULT_PYTORCH_CPU,
    pytorchIndex('gpu'), pytorchIndex('cpu'),
  ];
  for (const raw of String(text || '').split('\n')) {
    const line = raw.split('#')[0].trim();
    if (!line) continue;
    if (line.startsWith('--hash=') || line.includes(' --hash=')) continue;

    if (line.startsWith('-')) {
      const url = line.split(/[\s=]+/)[1] || '';
      if (!allowed.some((prefix) => url.startsWith(prefix))) {
        return { ok: false, detail: `lock file redirects pip: ${line}` };
      }
      continue;
    }
    const withoutHashes = line.replace(/\s--hash=\S+/g, '');
    if (withoutHashes.includes(' @ ') || withoutHashes.includes('://')) {
      return { ok: false, detail: `lock file installs from a URL: ${line}` };
    }
  }
  return { ok: true, detail: '' };
}

/**
 * Judge torch's own answer about itself.
 *
 * `pip install` exits 0 while installing a CPU-only wheel, so the exit code
 * proves nothing. What matters is whether a device is usable, which is why an
 * untagged version with no device is a failure too.
 */
function verifyCudaFromOutput(stdout) {
  const text = String(stdout ?? '').trim();
  if (!text) return { ok: false, detail: 'torch produced no output' };

  // The LAST line: torch, or something it imports, may print a warning to
  // stdout first, and taking the first two tokens of the whole output then
  // parses that warning as a version and reports a working card as broken.
  const lastLine = text.split(/\r?\n/).filter((l) => l.trim()).pop() || '';
  const [version = '', available = ''] = lastLine.trim().split(/\s+/);
  if (version.includes('+cpu')) {
    return { ok: false, detail: `a CPU-only build of PyTorch is installed (${version})` };
  }
  if (available.toLowerCase() !== 'true') {
    return {
      ok: false,
      detail: `PyTorch ${version || '(unknown version)'} reports no usable CUDA device — `
        + `the NVIDIA driver is probably older than ${MIN_DRIVER_FOR_CUDA12}`,
    };
  }
  return { ok: true, detail: text };
}

/** Ask the interpreter we just installed what it actually has. */
async function verifyCuda(exe) {
  try {
    // Registered like any other child: this one holds ~2.5 GB of mapped CUDA
    // DLLs and now has a five-minute budget, so without this it outlives the
    // window by minutes when someone closes the wizard during verification.
    const pending = execFileAsync(
      exe,
      ['-c', 'import torch;print(torch.__version__, torch.cuda.is_available(),'
           + 'torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'],
      { timeout: 300000, windowsHide: true },
    );
    if (pending.child) runSetup.children.add(pending.child);
    let stdout;
    try {
      ({ stdout } = await pending);
    } finally {
      if (pending.child) runSetup.children.delete(pending.child);
    }
    return verifyCudaFromOutput(stdout);
  } catch (err) {
    // A first `import torch` on a cold tree, with Defender scanning 8 GB of
    // freshly written wheels, can take minutes. Killing it and reporting "your
    // GPU does not work" invites the user to throw away an install that was
    // fine, so a timeout says so and is not offered as a CUDA failure.
    if (err && (err.killed || err.signal)) {
      return {
        ok: false,
        timedOut: true,
        detail: 'PyTorch took too long to start, so Orienta could not check the '
          + 'graphics card. The installation itself is finished — start Orienta '
          + 'again and it will use the card if it works.',
      };
    }
    return { ok: false, detail: `PyTorch could not be imported: ${err.message}` };
  }
}

function sha256File(file) {
  const crypto = require('node:crypto');
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(file);
    stream.on('error', reject);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

/**
 * Electron's `net`, required LAZILY.
 *
 * Two reasons. `node:https` ignores HTTP_PROXY/HTTPS_PROXY while Python's
 * urllib honours them, so on a university network with a mandatory proxy the
 * in-app update would work and the first install could not — Electron's net
 * uses Chromium's stack and picks up the system proxy, including PAC files,
 * with no code. And `electron` is installed only under frontend/node_modules,
 * so a top-level require from here cannot resolve and would take every
 * pure-function test in this module down with it.
 */
function electronNet() {
  return require('electron').net;
}

/**
 * Compare two release tags as VERSIONS, newest first.
 *
 * Not `Array.sort()`: lexically "v0.4.0" sorts above "v0.10.0", because "4"
 * beats "1" one character in. Every project meets this exactly once, at the
 * 0.9 -> 0.10 boundary, and by then the wrong answer has been shipping for a
 * while. A tag that does not parse sorts last rather than throwing: it is a
 * folder listing, not a contract.
 */
function compareTagsDesc(a, b) {
  const parts = (tag) => String(tag).replace(/^v/i, '').split(/[.+-]/)
    .map((piece) => (/^\d+$/.test(piece) ? Number(piece) : NaN));
  const left = parts(a);
  const right = parts(b);
  for (let i = 0; i < Math.max(left.length, right.length); i += 1) {
    const x = left[i];
    const y = right[i];
    if (Number.isNaN(x) && Number.isNaN(y)) continue;
    if (Number.isNaN(x)) return 1;      // unparseable sorts last
    if (Number.isNaN(y)) return -1;
    if ((x || 0) !== (y || 0)) return (y || 0) - (x || 0);
  }
  return 0;
}

/** Is this somewhere we are willing to send a request in clear text? */
function isLoopback(parsed) {
  return parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost'
    || parsed.hostname === '[::1]';
}

/**
 * May we follow this hop?
 *
 * A pure function on purpose. The wiring below needs an Electron runtime and
 * cannot be unit-tested, so the part that can be WRONG is kept out of it --
 * otherwise the only available test is one that greps the source for a string,
 * which is how the previous version of this policy shipped a bug that killed
 * every download while its test stayed green.
 *
 * @returns {{ok: true} | {ok: false, reason: string}}
 */
function redirectVerdict(fromUrl, redirectUrl, hops, limit = 5) {
  let next;
  try {
    next = new URL(redirectUrl, fromUrl);
  } catch {
    return { ok: false, reason: 'the redirect address could not be read' };
  }
  // The scheme, not the host: GitHub legitimately sends asset downloads to
  // objects.githubusercontent.com, so a same-host rule would break every real
  // install. What must not happen is dropping to plain text, where an
  // intercepted checksum document and an intercepted payload agree with each
  // other and the digest proves nothing about either.
  if (next.protocol !== 'https:' && !isLoopback(next)) {
    return {
      ok: false,
      reason: `refusing a redirect to ${next.protocol}// -- downloads must stay encrypted`,
    };
  }
  if (hops > limit) return { ok: false, reason: 'too many redirects' };
  return { ok: true };
}

function requestOnce(url, onResponse, onError, redirectsLeft) {
  // 'manual' does NOT mean "the 3xx arrives as a response". In Electron it
  // means the request emits 'redirect' and is CANCELLED unless
  // followRedirect() is called -- which is how an earlier version of this
  // function killed every download with "Redirect was cancelled", since
  // python-build-standalone's asset URL redirects once to a CDN.
  //
  // The reason for taking manual control at all is redirectVerdict above:
  // with the default ('follow') Chromium follows every hop internally and no
  // hop is ever inspected.
  const request = electronNet().request({ url, redirect: 'manual' });
  let hops = 0;

  request.on('redirect', (status, method, redirectUrl) => {
    hops += 1;
    const verdict = redirectVerdict(url, redirectUrl, hops, redirectsLeft || 5);
    if (!verdict.ok) {
      request.abort();
      onError(new Error(verdict.reason));
      return;
    }
    request.followRedirect();
  });

  request.on('response', (response) => {
    const status = response.statusCode;
    if (status < 200 || status >= 300) {
      // 3xx cannot arrive here: it goes to 'redirect' above.
      onError(Object.assign(new Error(`HTTP ${status} for ${url}`), { status }));
      response.resume();
      return;
    }
    onResponse(response);
  });
  request.on('error', onError);
  request.end();
}

function fetchText(url) {
  return new Promise((resolve, reject) => {
    requestOnce(url, (response) => {
      const chunks = [];
      response.on('data', (c) => chunks.push(c));
      response.on('error', reject);
      response.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    }, reject, 5);
  });
}

/** No byte for this long means the connection is dead, whatever it claims. */
const STALL_TIMEOUT_MS = 120000;

function fetchToFile(url, dest, onBytes) {
  return new Promise((resolve, reject) => {
    requestOnce(url, (response) => {
      const total = Number(response.headers['content-length'] || 0);
      let done = 0;
      let settled = false;
      const out = fs.createWriteStream(dest);

      // Without this a download that stops mid-stream hangs for ever while the
      // heartbeat keeps saying "still working" — which on a network with a
      // captive portal or a PAC-file proxy is the DEFAULT outcome, and is
      // exactly the case Electron's net was chosen for.
      let stall = null;
      const done_ = () => { if (stall) clearTimeout(stall); };
      const arm = () => {
        done_();
        stall = setTimeout(() => {
          fail(Object.assign(
            new Error(
              `the download stopped responding after ${Math.round(done / 1e6)} MB. `
              + 'This usually means a proxy or a firewall is interfering.'),
            { code: 'stalled' },
          ));
        }, STALL_TIMEOUT_MS);
      };

      const fail = (err) => {
        if (settled) return;
        settled = true;
        done_();
        out.destroy();
        try { response.destroy(); } catch { /* ignore */ }
        // Never leave a partial file behind: the next run must not mistake it
        // for a download that completed.
        try { fs.rmSync(dest, { force: true }); } catch { /* ignore */ }
        reject(err);
      };

      out.on('error', fail);
      response.on('error', fail);
      response.on('data', (chunk) => {
        done += chunk.length;
        arm();
        if (onBytes) onBytes(done, total);
      });
      response.pipe(out);
      out.on('finish', () => {
        if (settled) return;
        // A server or proxy that closes early produces a perfectly normal
        // `finish`. The digest catches it, but blames it on antivirus tampering
        // — so name it here, where the shortfall is still known, and let it
        // classify as a network error the user should simply retry.
        if (total > 0 && done !== total) {
          fail(Object.assign(
            new Error(`the download ended early (${done} of ${total} bytes)`),
            { code: 'ECONNRESET' },
          ));
          return;
        }
        settled = true;
        done_();
        resolve();
      });
      arm();
    }, reject, 5);
  });
}

/**
 * Fetch JSON, and name the one failure that does not look like a failure.
 *
 * A hotel, conference or campus captive portal answers 200 with an HTML
 * sign-in page. Unwrapped, JSON.parse throws "Unexpected token <" and lands on
 * "something unexpected" -- for the exact environment the checksum message
 * already names out loud.
 */
async function fetchJson(url) {
  const body = await fetchText(url);
  try {
    return JSON.parse(body);
  } catch {
    throw Object.assign(
      new Error('the reply was not a release description; a network sign-in '
        + 'page may be in the way'),
      { code: 'captive' });
  }
}

/**
 * Which release to install, and where its two files are.
 *
 * `wanted` is a tag, or null for "whatever is newest". Null is the DEFAULT and
 * the whole reason this function exists: an installer that asks for its own
 * build number can only ever install that one release, so every copy of it
 * dies the day that release is superseded.
 *
 * `/releases/latest` hides drafts and pre-releases, so a repository whose only
 * releases are pre-releases has no "latest" at all. Falling through to the
 * list rather than reporting "not published yet" is the difference between a
 * pre-release being installable and not.
 */
async function resolveRelease(wanted) {
  const usable = (release) => {
    if (!release || !release.tag_name) return null;
    const names = runtimeAssetNames(release.tag_name);
    const pkg = assetUrl(release, names.package);
    const sum = assetUrl(release, names.checksum);
    return pkg && sum ? { tag: release.tag_name, packageUrl: pkg, checksumUrl: sum, names } : null;
  };

  if (wanted && wanted !== 'latest') {
    const found = usable(await fetchJson(releaseByTagUrl(wanted)));
    if (found) return found;
    throw Object.assign(
      new Error(`the release ${wanted} does not carry `
        + `${runtimeAssetNames(wanted).package}`),
      { status: 404 });
  }

  try {
    const found = usable(await fetchJson(latestReleaseUrl()));
    if (found) return found;
  } catch (err) {
    // A 404 here means "no non-prerelease exists", which the list can still
    // answer. Anything else -- no network, a proxy, a rate limit -- must not
    // be retried as a second failing request.
    if (!err || err.status !== 404) throw err;
  }

  const list = await fetchJson(releaseListUrl());
  if (Array.isArray(list)) {
    for (const release of list) {
      if (release && release.draft) continue;
      const found = usable(release);
      if (found) return found;
    }
  }
  throw Object.assign(
    new Error('no published release carries an Orienta package yet'),
    { status: 404 });
}

/** Every release a user could choose, newest first. */
async function listReleases() {
  const list = await fetchJson(releaseListUrl());
  if (!Array.isArray(list)) return [];
  return list
    .filter((r) => r && r.tag_name && !r.draft)
    .map((r) => ({
      tag: r.tag_name,
      name: r.name || r.tag_name,
      prerelease: Boolean(r.prerelease),
      publishedAt: r.published_at || null,
      // A release with no package cannot be installed, and offering it in the
      // list would hand the user a choice that always fails.
      installable: Boolean(assetUrl(r, runtimeAssetNames(r.tag_name).package)
        && assetUrl(r, runtimeAssetNames(r.tag_name).checksum)),
    }));
}

/**
 * Download `url` to `dest` and refuse anything whose digest does not match.
 *
 * `filename` is passed to the shared parser because a checksum document may be
 * an AGGREGATE naming hundreds of files — python-build-standalone publishes
 * exactly that — and answering it with its first line returns the digest of a
 * different file, failing the install with a mismatch that blames the download.
 *
 * A digest served from the same host proves integrity of TRANSFER, not
 * authenticity. It catches a truncated download, a proxy rewriting content and
 * a corrupted mirror; it does not catch a compromised release.
 */
/**
 * A download whose digest we already know, rather than fetching alongside it.
 *
 * micromamba is pinned by version AND digest in `platform.js`, so there is
 * nothing to fetch: the checksum file published beside the asset would be
 * fetched over the same connection as the asset, which is not a second
 * opinion. (It is also wrong upstream for the tarball assets — the generator
 * writes the bare binary's digest into the tarball's `.sha256` — so the
 * pinned value is the only trustworthy one.)
 */
async function downloadPinned(url, expected, dest, filename, onBytes) {
  await fetchToFile(url, dest, onBytes);
  await assertSha256(dest, expected, filename || url);
  return dest;
}

/** The original: fetch the digest from its own document, then download. */
async function downloadVerified(url, sha256Url, dest, filename, onBytes) {
  const document = await fetchText(sha256Url);
  const expected = parseSha256Document(document, filename || null);
  await fetchToFile(url, dest, onBytes);
  await assertSha256(dest, expected, filename || url);
  return dest;
}

async function assertSha256(dest, expected, filename) {
  const actual = await sha256File(dest);
  if (actual.toLowerCase() !== String(expected).toLowerCase()) {
    try { fs.rmSync(dest, { force: true }); } catch { /* ignore */ }
    const error = new Error(
      `checksum mismatch for ${filename}: expected ${expected}, got ${actual}. `
      + 'The download was altered in transit — antivirus software and captive '
      + 'portals both do this.',
    );
    error.code = 'checksum';
    throw error;
  }
  return dest;
}

module.exports.redirectVerdict = redirectVerdict;
module.exports.compareTagsDesc = compareTagsDesc;
module.exports.foreignEntries = foreignEntries;
module.exports.localPackage = localPackage;
module.exports.localPackageSearch = localPackageSearch;
module.exports.resolveRelease = resolveRelease;
module.exports.listReleases = listReleases;
module.exports.fetchJson = fetchJson;
module.exports.lockFileFor = lockFileFor;
module.exports.CUDA_ONLY_PACKAGES = CUDA_ONLY_PACKAGES;
module.exports.pythonExeIn = pythonExeIn;
module.exports.resumeHasSomethingToContinueFrom = resumeHasSomethingToContinueFrom;
module.exports.tarExe = tarExe;
module.exports.pytorchIndex = pytorchIndex;
module.exports.pipIndexArgs = pipIndexArgs;
module.exports.lockFileIsSafe = lockFileIsSafe;
module.exports.verifyCudaFromOutput = verifyCudaFromOutput;
module.exports.verifyCuda = verifyCuda;
module.exports.sha256File = sha256File;
module.exports.fetchText = fetchText;
module.exports.fetchToFile = fetchToFile;
module.exports.downloadVerified = downloadVerified;
module.exports.PYTORCH_GPU_ENV = PYTORCH_GPU_ENV;
module.exports.PYTORCH_CPU_ENV = PYTORCH_CPU_ENV;

// ==========================================================================
// runSetup — the whole job
// ==========================================================================

const { spawn } = require('node:child_process');  // execFile is imported at the top
const os = require('node:os');

/**
 * A runtime package lying beside the installer, or null.
 *
 * Measurement PCs in this field are routinely kept off the network, and a
 * lab that can reach GitHub from a desk machine but not from the microscope
 * has no way in otherwise. Dropping `orienta-runtime-<tag>.zip` and its
 * `.sha256` next to the .exe -- or into `<home>` -- is that way.
 *
 * The checksum file is REQUIRED even here. A local file is not automatically
 * a trustworthy one: it arrives on the same USB stick that carries everything
 * else in a shared lab, and skipping the check for the offline path would put
 * the weakest verification exactly where the least supervision is.
 */
function localPackage(home, releaseTag, extraDirs = []) {
  return localPackageSearch(home, releaseTag, extraDirs).found;
}

/**
 * The same search, but reporting WHERE it looked.
 *
 * Because the failure it produces is otherwise a lie by omission: a user who
 * was told to put a file somewhere gets a sentence about an unpublished
 * GitHub release, and no hint that a local search happened at all, let alone
 * which folders it read. Listing them turns "why does this not work" into one
 * glance.
 */
function localPackageSearch(home, releaseTag, extraDirs = []) {
  // A null tag means "newest", and there is no filename to build from it. In
  // a folder the user pointed at, the package that is THERE is the answer, so
  // the tag is read off the file rather than demanded in advance.
  const names = releaseTag ? runtimeAssetNames(releaseTag) : null;
  const userProfile = process.env.USERPROFILE || process.env.HOME || '';
  const candidates = [
    // The copy that shipped INSIDE the installer, before anything else. It is
    // the one location that cannot be wrong: nobody had to put it there, no
    // folder can have been moved, and it works on a machine that cannot reach
    // GitHub at all. Everything below is for installing a DIFFERENT version
    // than the one that came in the box.
    (() => {
      try {
        return process.resourcesPath || null;
      } catch { return null; }
    })(),
    // `extraDirs` carries the REAL Downloads and Desktop, resolved by the main
    // process through Electron's own lookup. Building them from %USERPROFILE%
    // is wrong on any machine where OneDrive has moved the known folders --
    // which is the default on a managed university laptop, and is precisely
    // the machine most likely to be handed a file and told where to put it.
    ...extraDirs,
    (process.env.ORIENTA_PACKAGE_DIR || '').trim() || null,
    home,
    // The %USERPROFILE% guesses stay as a fallback for when the main process
    // could not resolve the real ones.
    userProfile ? path.join(userProfile, 'Downloads') : null,
    userProfile ? path.join(userProfile, 'Desktop') : null,
    // The directory the app was launched FROM, which is the portable case.
    (() => { try { return process.cwd(); } catch { return null; } })(),
    // NOT where the setup was double-clicked: after installation this is
    // `...\Programs\Orienta\`, and the setup is gone by then anyway.
    (() => { try { return path.dirname(process.execPath); } catch { return null; } })(),
  ].filter(Boolean);
  const searched = [...new Set(candidates)];

  for (const dir of searched) {
    if (names) {
      const zip = path.join(dir, names.package);
      const sum = `${zip}.sha256`;
      try {
        if (fs.statSync(zip).isFile() && fs.statSync(sum).isFile()) {
          return { found: { zip, sum, dir, tag: releaseTag }, searched };
        }
      } catch { /* not here; try the next one */ }
      continue;
    }
    // Unpinned: take the newest-looking package that has its checksum beside
    // it. Sorted descending so a folder holding two versions gives the later
    // one, which is what "newest" has to mean here.
    let found = null;
    try {
      const entries = fs.readdirSync(dir)
        .filter((n) => /^orienta-runtime-.+\.zip$/i.test(n))
        .sort((x, y) => compareTagsDesc(
          x.replace(/^orienta-runtime-/i, '').replace(/\.zip$/i, ''),
          y.replace(/^orienta-runtime-/i, '').replace(/\.zip$/i, '')));
      for (const name of entries) {
        const zip = path.join(dir, name);
        const sum = `${zip}.sha256`;
        if (fs.statSync(zip).isFile() && fs.existsSync(sum)) {
          const tag = name.replace(/^orienta-runtime-/i, '').replace(/\.zip$/i, '');
          found = { zip, sum, dir, tag };
          break;
        }
      }
    } catch { /* unreadable directory; try the next one */ }
    if (found) return { found, searched };
  }
  return { found: null, searched };
}

/** Every failure the user can meet has its own code, so the wizard can say
 *  something translated and specific rather than printing a Node error. */
const CODES = {
  blocked: 'space',
  // Not produced by runSetup: the shell raises it from `setup:probe` when the
  // Orienta home cannot be written at all. It lives here because this list is
  // what the wizard's vocabulary is checked against, and a code with no
  // sentence is a user reading a raw identifier.
  homeUnusable: 'homeUnusable',
  homeNotEmpty: 'homeNotEmpty',
  network: 'network',
  notFound: 'notFound',
  rateLimited: 'rateLimited',
  checksum: 'checksum',
  timeout: 'timeout',
  inUse: 'inUse',
  package: 'package',
  pip: 'pip',
  cuda: 'cudaUnavailable',
  // This machine is not one we ship for: macOS below 14, an Intel Mac,
  // or an operating system that is not one of the three.
  unsupported: 'unsupported',
  interrupted: 'interrupted',
  unexpected: 'unexpected',
};

/** Classify a thrown error into one of the codes above. */
function classify(err, step) {
  if (err && err.code === 'checksum') return CODES.checksum;
  if (err && err.code === 'stalled') return CODES.timeout;
  if (err && err.code === 'captive') return CODES.network;
  const status = err && err.status;
  // A 404 means "this release was never published" ONLY while we are asking
  // about the release. In the `python` step the same status means
  // python-build-standalone yanked or retagged an asset -- blaming Orienta's
  // release there sends the user to look at the wrong page entirely.
  if (status === 404) return step === 'resolve' ? CODES.notFound : CODES.unexpected;
  // 403 is GitHub's rate limit AND what a university proxy returns when it
  // refuses the request outright. Telling the second user to "try again in a
  // few minutes" has them waiting for something that will never change; 429 is
  // unambiguous, 403 is not.
  if (status === 429) return CODES.rateLimited;
  if (status === 403) return CODES.network;
  const message = String((err && err.message) || '');
  // Disk-full has its own message, and it is the one failure a user can act on
  // immediately. Landing it on "unexpected" wastes the only actionable error
  // this module produces.
  if (/ENOSPC|no space left|not enough space|disk is full/i.test(message)
      || (err && err.code === 'ENOSPC')) {
    return CODES.blocked;
  }
  if (/ENOTFOUND|ECONNREFUSED|ECONNRESET|ETIMEDOUT|net::|getaddrinfo|socket/i.test(message)) {
    return CODES.network;
  }
  return CODES.unexpected;
}

/**
 * Append one line to <home>/logs/orienta-setup.log.
 *
 * An installation is the one part of this product a user cannot retry cheaply:
 * twenty minutes and eight gigabytes. When it fails, the only thing that can
 * say why is the output the failing program already produced — which, until
 * this existed, was fed to a function that updated a timestamp and dropped it.
 * The log lands beside the application's own logs/, so the diagnostics export
 * picks it up with no extra wiring.
 *
 * appendFileSync, not a stream: the line that matters most is the last one
 * before a crash, and a stream may still be holding it.
 */
function logSetupLine(home, line) {
  try {
    const dir = path.join(home, 'logs');
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, 'orienta-setup.log'),
                      `${new Date().toISOString()} ${line}\n`, 'utf8');
  } catch { /* a log that cannot be written must not fail the install */ }
}

/**
 * Entries in `home` that Orienta did not put there.
 *
 * Everything on this list is something the app or the setup creates — note
 * `electron` and `logs`, which the shell writes at startup, BEFORE setup ever
 * runs, so a fresh home is never quite empty. Anything else means the folder
 * is somebody's, and setting up in it would hand that somebody's files to the
 * uninstaller.
 */
const OURS = new Set([
  'electron', 'logs', 'setup-tmp', 'python', 'runtime', 'pending', 'pending.json',
  '.orienta-home', '.python_path', '.install_mode', '.install_incomplete',
  // What the packages were last brought to, the marker of an update in flight,
  // and the last failure to bring them there (package_sync.js). Missing here, a
  // repair after an interrupted update would refuse with "this folder contains
  // files that are not Orienta's", naming files the app wrote itself.
  '.packages_lock.json', '.packages_sync.json', '.packages_sync_failed.json',
  // macOS writes these two itself, in step 2 and step 3. Without them the
  // SECOND run of the setup — a retry after any failure, the "install the
  // processor version instead" button, or repair mode — refuses with "this
  // folder already contains files that are not Orienta's (envs, micromamba)",
  // blaming the user for files the installer wrote a minute earlier, with no
  // way out. Taken from platform.js so the list cannot drift from the paths.
  platform.CONDA_ROOT_DIR, platform.CONDA_ENVS_DIR,
]);
function foreignEntries(home, entries = null) {
  let names = entries;
  if (!names) {
    try {
      names = fs.readdirSync(home);
    } catch {
      return [];   // unreadable: the steps below will say so in their own words
    }
  }
  return names.filter((n) => !OURS.has(n) && !/^python\.old-/.test(n));
}

/** Record what the run was doing, for a launch that happens after it dies.
 *
 * Best effort by design: failing to write a breadcrumb must never be the reason
 * an installation fails.
 */
function writeMarker(file, fields) {
  try {
    fs.writeFileSync(file, `${JSON.stringify({ startedAt: new Date().toISOString(), ...fields })}\n`);
  } catch { /* ignore */ }
}

function rmrf(target) {
  try { fs.rmSync(target, { recursive: true, force: true }); } catch { /* ignore */ }
}

/**
 * Run a child and stream its output, line by line, to `onLine`.
 *
 * `spawn`, not `execFile`: execFile buffers, and this runs for tens of minutes
 * with a 1 MiB default cap that a verbose pip exceeds. `windowsHide` because
 * this project has a recorded incident of console windows opening over the
 * desktop.
 */
function runStreaming(exe, args, { onLine, cwd, env } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(exe, args, { cwd, env, windowsHide: true });
    // One buffer PER STREAM: stdout and stderr interleave, and a shared tail
    // concatenates half a pip progress line onto the next stderr chunk, so the
    // wizard shows garbled text exactly when something is going wrong.
    const tails = { out: '', err: '' };
    const feed = (which) => (buf) => {
      tails[which] += buf.toString();
      const lines = tails[which].split(/\r?\n/);
      tails[which] = lines.pop();
      for (const line of lines) if (line.trim() && onLine) onLine(line.trim());
    };
    child.stdout.on('data', feed('out'));
    child.stderr.on('data', feed('err'));
    child.on('error', reject);
    child.on('close', (code) => {
      for (const rest of [tails.out, tails.err]) {
        if (rest.trim() && onLine) onLine(rest.trim());
      }
      if (code === 0) resolve();
      else reject(Object.assign(new Error(`${path.basename(exe)} exited with ${code}`), { exitCode: code }));
    });
    runSetup.children.add(child);
    child.on('close', () => runSetup.children.delete(child));
  });
}

/**
 * Record which lock the packages were just installed from, so that the first
 * start after this install takes the fast path in `package_sync.js` instead of
 * asking pip a question whose answer is "nothing".
 *
 * Never throws and never fails an install: a missing record costs one dry run
 * at the next start, not a working installation.
 *
 * The lock is read from the unpacked runtime on Windows and Linux -- the very
 * file step 4 installed from -- and, on macOS, from the text the environment was
 * created from (`macosLockText`); a resumed macOS run falls back to the unpacked
 * file, which is the same lock.
 */
function recordPackagesLock(home, mode, runtimeTag, macosLockText = null) {
  try {
    const sync = require('./package_sync');
    const darwin = platform.current() === 'darwin';
    const lockName = darwin ? macosEnv.MACOS_LOCK_FILE : lockFileFor(mode);
    const lockText = darwin && macosLockText
      ? macosLockText
      : fs.readFileSync(path.join(home, 'runtime', lockName), 'utf8');
    sync.recordFirstInstall({
      home, mode, lockName, lockText, runtimeTag, platformName: platform.current(),
    });
  } catch (err) {
    logSetupLine(home, `could not record the package lock (${err.message}); `
      + 'the next start will ask pip instead');
  }
}

/**
 * Build the Python side of the application.
 *
 * Never throws: every failure is a returned `{ok:false, code, error}` so the
 * wizard always has something translated to show.
 *
 * `resume` skips the interpreter and the runtime package, which are already
 * there — it exists so the CPU fallback after a failed CUDA verification does
 * not re-download 48 MB and 4 MB for nothing.
 */
async function runSetup(options, onProgress) {
  const inner = await runSetupInner(options, onProgress);
  if (inner.ok || !options || !options.home) return inner;
  // Attached ONCE, here. Set per-return it reached two of the twelve failure
  // paths, so "Open the log" was hidden for the other ten -- including
  // `package` and `cuda`, the two where the log IS the story -- while
  // logSetupLine had been writing the file the whole time.
  return {
    log: tailOf(options.home),
    logFile: path.join(options.home, 'logs', 'orienta-setup.log'),
    ...inner,
  };
}

/** The last tail `runSetupInner` recorded, kept outside it so the wrapper can
 *  reach it without every return carrying it. */
let lastTail = '';
function tailOf() { return lastTail; }

async function runSetupInner(options, onProgress) {
  const {
    mode = 'cpu', home, releaseTag, resume = false,
    packageDir = null, knownFolders = null,
  } = options || {};
  const emit = (step, message, extra = {}) => {
    if (onProgress) onProgress({ step, message, ...extra });
  };

  if (!home) return { ok: false, code: CODES.unexpected, error: 'no installation directory' };

  // Step 0: is this machine one we ship for at all?
  //
  // Before the release is resolved and before anything is downloaded. Without
  // it, macOS 13 got no refusal whatsoever — micromamba downloaded and the
  // environment failed somewhere inside a transaction against packages that
  // need 14 — and an Intel Mac got `no micromamba build for darwin/x64`
  // thrown out of platform.js two steps later, as a developer sentence under
  // "something unexpected". Both are answerable questions, and the answer
  // belongs in front of the work, in a sentence the user can act on.
  const support = platform.supportStatus();
  if (!support.supported) {
    return { ok: false, code: CODES.unsupported, error: support.reason };
  }

  const temp = path.join(home, 'setup-tmp');
  const incomplete = path.join(home, '.install_incomplete');
  let heartbeat = null;
  let cudaUnverified = '';
  // The lock text the macOS environment was created from, kept for the record
  // written at the end (package_sync.js); null elsewhere and on a resumed run.
  let macosLockText = null;
  // True once the interpreter AND the program files are on disk. `resume` is a
  // property of the failure, not a constant the caller may assume: offering
  // "install the processor version instead" with resume=true after a failure
  // that happened BEFORE the download leaves step 4 looking for a lock file in
  // a tree that does not exist, and answering "something unexpected".
  let resumable = resume;
  let lastActivity = Date.now();
  const started = Date.now();
  let currentStep = 'probe';
  // Per-step, because "22 minutes" against a step that normally takes two is
  // information, and against one that normally takes twenty-five is noise.
  let stepStarted = Date.now();

  const touch = () => { lastActivity = Date.now(); };

  // The last 50 lines of the CURRENT step. Bounded because pip on the GPU path
  // emits thousands, and an error message is read by a person.
  let tail = [];
  const record = (line) => {
    touch();
    logSetupLine(home, line);
    tail.push(line);
    if (tail.length > 50) tail.shift();
    lastTail = tail.join('\n');
  };
  const tailText = () => tail.join('\n');
  lastTail = '';

  try {
    runSetup.cancelled = false;
    fs.mkdirSync(home, { recursive: true });
    // Never into a folder that holds someone else's files. The marker written
    // just below is what the uninstaller later trusts to say "this is
    // Orienta's", and it would then remove `python\`, `logs\`, `runtime\*`
    // from — say — an ORIENTA_HOME=D:\Work that happened to have a folder of
    // those names.
    const foreign = foreignEntries(home);
    if (foreign.length) {
      return {
        ok: false,
        code: CODES.homeNotEmpty,
        error: `${home} already contains files that are not Orienta's `
          + `(${foreign.slice(0, 5).join(', ')}${foreign.length > 5 ? ', …' : ''}).`,
      };
    }
    // The one thing the UNINSTALLER trusts to say "this folder is Orienta's
    // data folder". Before it, recognition rested on files a developer
    // checkout can also have (.python_path), so an ORIENTA_HOME pointing at a
    // repository could have been "cleaned". Written first, so even a setup
    // that fails part-way leaves a folder the uninstaller will recognise.
    try {
      fs.writeFileSync(path.join(home, '.orienta-home'),
        'This folder holds Orienta\'s data. The uninstaller looks for this file\n'
        + 'before removing anything here; please leave it in place.\n');
    } catch { /* recognition then falls back to runtime\\VERSION plus a second marker */ }
    // Written first and removed ONLY on success. A setup that fails in step 4
    // leaves a complete python/ and a complete runtime/ but no .python_path, so
    // startupDecision reads "the interpreter is missing" — which is false, it
    // is sitting right there. This file is what lets the next launch say what
    // really happened, and which step to resume from.
    writeMarker(incomplete, { mode, step: 'probe', tag: releaseTag || null });

    // pip prints almost nothing on a pipe: on the GPU path the user would see
    // one line and then silence for twenty minutes. This project already
    // learned that lesson once, in simulation_controller.py.
    heartbeat = setInterval(() => {
      // Guarded: this runs on the timer queue, outside the try/catch around the
      // install, so a throwing onProgress callback would become an uncaught
      // exception in the main process and take the window with it.
      try {
        if (Date.now() - lastActivity < 4000) return;
        emit(currentStep, '', {
          elapsedS: Math.round((Date.now() - started) / 1000),
          stepElapsedS: Math.round((Date.now() - stepStarted) / 1000),
        });
      } catch { /* ignore */ }
    }, 3000);

    // ---- 1. what can this machine take? ---------------------------------
    currentStep = 'probe';
    emit('probe', '');
    const measured = await probe(home);
    touch();
    if (measured.blocked) {
      return { ok: false, code: CODES.blocked, error: measured.reason };
    }
    // `blocked` only means "below the CPU floor". The user may have chosen GPU
    // from a wizard that is deliberately allowed to offer it against advice, so
    // the floor for the mode ACTUALLY chosen has to be tested here — this is
    // the only place that knows both the measurement and the choice. Without
    // it, a machine with 5 GB free downloads, extracts, and then dies eight
    // gigabytes into pip: exactly the outcome the measurement exists to avoid.
    const needed = mode === 'gpu' ? GPU_BYTES_NEEDED : cpuBytesNeeded();
    if (measured.freeKnown !== false && measured.freeBytes < needed) {
      return {
        ok: false,
        code: CODES.blocked,
        error: `The ${mode === 'gpu' ? 'graphics card' : 'processor'} version needs `
          + `about ${(needed / GB).toFixed(1)} GB and this drive has `
          + `${(measured.freeBytes / GB).toFixed(1)} GB free.`
          + (mode === 'gpu'
            ? ' Free some space, or install the processor version instead.'
            : ' Free some space and try again.'),
        canFallBackToCpu: mode === 'gpu' && measured.freeBytes >= cpuBytesNeeded(),
        // Nothing has been downloaded yet, so the fallback must NOT skip the
        // downloads: step 4 would look for a lock file in a tree that does not
        // exist and answer "something unexpected".
        resumable: false,
      };
    }

    rmrf(temp);
    fs.mkdirSync(temp, { recursive: true });

    // ---- 1b. does the release we are about to install actually exist? ----
    // One small GET, and the single most likely thing to be wrong: a tag that
    // was never published, an asset that failed to upload, a rate limit, a
    // proxy. Asking LAST meant a user on a wrong tag downloaded 48 MB, unpacked
    // an interpreter and only then heard that there was nothing to install.
    let packageUrl = null;
    let checksumUrl = null;
    // Which version is actually being installed. Reported back so the wizard
    // can NAME it -- "Orienta 0.4.1" -- instead of leaving the user to find
    // out afterwards.
    let resolvedTag = releaseTag || null;
    let assetNames = runtimeAssetNames(resolvedTag || 'v0.0.0');

    // What is already on this machine — above all the copy that shipped
    // inside the installer.
    const extraDirs = [];
    if (packageDir) extraDirs.push(packageDir);
    if (Array.isArray(knownFolders)) extraDirs.push(...knownFolders.filter(Boolean));
    const search = resume
      ? { found: null, searched: [] }
      : localPackageSearch(home, releaseTag, extraDirs);
    let beside = search.found;

    // A LOCAL COPY IS A FALLBACK, NOT A DECISION.
    //
    // Taking it merely because it exists re-creates, by another route, the
    // exact fault this installer was just freed from: the bundled package is
    // whatever shipped in the .exe, so an .exe kept for a year would install a
    // year-old Orienta for ever and never once ask whether something newer
    // exists. The file name would say v0.4.0, and so would the result.
    //
    // So when no version was pinned, ask anyway, and keep the local copy only
    // if it is at least as new — or if the question cannot be asked at all,
    // which is what keeps an offline machine working.
    if (!resume && beside && !releaseTag) {
      currentStep = 'resolve';
      stepStarted = Date.now();
      emit('resolve', '');
      try {
        const newest = await resolveRelease(null);
        if (compareTagsDesc(newest.tag, beside.tag) < 0) {
          logSetupLine(home,
            `${newest.tag} is newer than the bundled ${beside.tag}; downloading it`);
          resolvedTag = newest.tag;
          assetNames = newest.names;
          packageUrl = newest.packageUrl;
          checksumUrl = newest.checksumUrl;
          beside = null;
        } else {
          logSetupLine(home,
            `the bundled ${beside.tag} is current (newest published: ${newest.tag})`);
        }
      } catch (err) {
        // Cannot ask: no network, a proxy, or nothing published yet. That is
        // precisely the case the bundled copy exists for, so it is not a
        // failure and must not be reported as one.
        logSetupLine(home, `could not check for a newer release (${err.message}); `
          + `using the bundled ${beside ? beside.tag : 'package'}`);
      }
      touch();
    }

    if (!resume && !beside && !packageUrl) {
      logSetupLine(home,
        `no Orienta package found in: ${search.searched.join(' | ')}`);
    }
    if (beside) {
      resolvedTag = beside.tag;
      assetNames = runtimeAssetNames(beside.tag);
      logSetupLine(home, `using the package found at ${beside.zip}`);
      emit('resolve', '', { tag: resolvedTag });
    }
    if (!resume && !beside && !packageUrl) {
      currentStep = 'resolve';
      stepStarted = Date.now();
      emit('resolve', '');
      let chosen;
      try {
        chosen = await resolveRelease(releaseTag);
      } catch (err) {
        // The folders searched go WITH the failure. Without them the screen
        // talks about GitHub to a user whose whole task was to put a file in
        // a folder, and never says which folders were read.
        if (search.searched.length) err.searchedDirs = search.searched;
        throw err;
      }
      resolvedTag = chosen.tag;
      assetNames = chosen.names;
      packageUrl = chosen.packageUrl;
      checksumUrl = chosen.checksumUrl;
      logSetupLine(home, `installing release ${resolvedTag}`);
      emit('resolve', '', { tag: resolvedTag });
      touch();
    }

    // ---- 2. the interpreter ---------------------------------------------
    if (!resume) {
      currentStep = 'python';
      stepStarted = Date.now();
      tail = [];
      logSetupLine(home, `--- step: python (mode=${mode}) ---`);
      writeMarker(incomplete, { mode, step: 'python', tag: releaseTag || null });
      emit('python', '');

      // macOS does not download an interpreter here at all: it downloads
      // micromamba, and the interpreter appears later, inside the conda
      // environment micromamba builds. The environment cannot be built yet —
      // its lock file is inside the program files, which are step 3 — so this
      // step ends with a 14 MB binary and no Python.
      const runtime = platform.runtimeSource();
      if (runtime.kind === 'conda') {
        const exe = platform.micromambaExeIn(home);
        fs.mkdirSync(path.dirname(exe), { recursive: true });
        if (!runtime.sha256) {
          return {
            ok: false, code: CODES.unexpected,
            error: `no pinned checksum for ${runtime.file}; refusing to run an `
              + 'unverified binary',
          };
        }
        await downloadPinned(runtime.url, runtime.sha256, exe, runtime.file,
          (done, total) => { touch(); emit('python', '', { done, total }); });
        // Apple Silicon refuses to run a file without the executable bit, and
        // a downloaded file does not have one.
        fs.chmodSync(exe, 0o755);
        touch();
      } else {
      const archive = path.join(temp, pythonArchiveName());
      await downloadVerified(
        pythonUrl(), pythonSha256Url(), archive, pythonArchiveName(),
        (done, total) => { touch(); emit('python', '', { done, total }); },
      );
      touch();

      // Extracted into a temp directory and MOVED, never straight into `home`:
      // home is the parent of runtime/, and this is the one archive with no
      // member validation of its own, so a tarball containing runtime/Database
      // would overwrite measured data.
      const unpacked = path.join(temp, 'python-unpacked');
      fs.mkdirSync(unpacked, { recursive: true });
      await runStreaming(tarExe(), ['xzf', archive, '-C', unpacked], { onLine: record });

      // The archive's own top-level directory is `python/`.
      const source = path.join(unpacked, 'python');
      const destination = path.join(home, 'python');
      // NOT rmrf-then-rename. A repair run meets a `python` directory whose
      // python.exe the backend may still have open (quit is not synchronous,
      // and antivirus holding one DLL is enough on its own). A recursive delete
      // is depth-first: it removes Lib/ and site-packages/, THEN hits the
      // locked exe and throws — gutting a working installation while leaving
      // python.exe in place, so the next launch still reads as "installed" and
      // dies with a Python bootstrap error nobody can connect to setup.
      //
      // Renaming the whole directory aside is atomic: with a file inside it
      // open, it fails having changed NOTHING.
      if (fs.existsSync(destination)) {
        const aside = `${destination}.old-${Date.now()}`;
        try {
          fs.renameSync(destination, aside);
        } catch (err) {
          return {
            ok: false,
            code: CODES.inUse,
            error: 'The existing Python folder is in use, so Orienta cannot replace '
              + 'it. Close Orienta completely (check the task bar) and run the '
              + `setup again. [${err.code || err.message}]`,
          };
        }
        rmrf(aside);   // best effort; a leftover .old- directory is harmless
      }
      fs.renameSync(source, destination);

      if (!fs.existsSync(pythonExeIn(home))) {
        return {
          ok: false, code: CODES.unexpected,
          error: `the interpreter archive did not contain ${pythonExeIn(home)}`,
        };
      }
      touch();
      }
    }

    // ---- 3. the program files -------------------------------------------
    if (!resume) {
      currentStep = 'runtime';
      stepStarted = Date.now();
      tail = [];
      logSetupLine(home, `--- step: runtime (mode=${mode}) ---`);
      writeMarker(incomplete, { mode, step: 'runtime', tag: releaseTag || null });
      emit('runtime', '');
      // Resolved in step 1b, before anything was downloaded.
      let zip;
      if (beside) {
        // Checked exactly as a downloaded one is: same parser, same digest,
        // same refusal. `parse_sha256_document` is given the filename because
        // the document may name several files.
        zip = beside.zip;
        const expected = parseSha256Document(
          fs.readFileSync(beside.sum, 'utf8'), assetNames.package);
        const actual = await sha256File(zip);
        if (actual.toLowerCase() !== expected.toLowerCase()) {
          return {
            ok: false,
            code: CODES.checksum,
            error: `${assetNames.package} in ${beside.dir} does not match its `
              + 'checksum file. The copy is damaged or incomplete.',
            resumable: false,
          };
        }
        emit('runtime', '', { done: 1, total: 1 });
      } else {
        zip = path.join(temp, assetNames.package);
        await downloadVerified(packageUrl, checksumUrl, zip, assetNames.package,
          (done, total) => { touch(); emit('runtime', '', { done, total }); });
      }
      touch();

      // Unpacked through the APPLIER, so the setup and the updater cannot hold
      // two different ideas of what a package may contain. `runtime/` is never
      // deleted: it holds the crystal library.
      // macOS builds its environment HERE, between the download and the
      // unpack, because of a circularity: the unpack runs `apply_update.py`
      // (which carries the security checks) and there is no interpreter to
      // run it until the environment exists -- and the environment needs the
      // lock file that is inside this very package. So one named member comes
      // out early, to stdout, and the full unpack follows unchanged.
      if (platform.current() === 'darwin') {
        currentStep = 'packages';
        emit('packages', '');
        writeMarker(incomplete, { mode, step: 'packages', tag: releaseTag || null });
        try {
          await macosEnv.installMacosEnvironment({
            micromamba: platform.micromambaExeIn(home),
            root: platform.condaRootIn(home),
            prefix: platform.condaPrefixIn(home),
            archive: zip,
            lockPath: path.join(temp, macosEnv.MACOS_LOCK_FILE),
            // The scrub is belt to `--no-rc --no-env`'s braces: micromamba
            // reads MAMBA_* and CONDA_* from the environment, and a user with
            // their own conda install has several of them set. Passing the
            // scrubbed copy means the isolation does not rest on two flags
            // alone.
            run: (exe, args, opts) => runStreaming(
              exe, args, { ...opts, env: macosEnv.scrubbedEnv() }),
            // 32 MB because the lock file comes back through this: it is
            // ~500 KB and the default buffer is 1 MB, which is close enough
            // to fail on a bigger environment rather than on this one.
            capture: async (exe, ...args) => (
              await execFileAsync(exe, args, {
                maxBuffer: 32 * 1024 * 1024, env: macosEnv.scrubbedEnv(),
              })).stdout,
            writeFile: async (file, text) => fs.writeFileSync(file, text, 'utf8'),
            // Logged always, shown selectively: the progress line is a sign of
            // life, not a console. micromamba opens every transaction with a
            // paragraph about package scripts containing arbitrary code --
            // boilerplate, in English, which the M5 tester met sitting in the
            // middle of a German wizard on 2026-09-25 and reported as an
            // alarming raw message.
            onLine: (line) => {
              record(line);
              if (!isSetupNoise(line)) emit('packages', line);
              touch();
            },
          });
        } catch (err) {
          // CODES.pip, not a new code: its sentence is "Installing the Python
          // packages failed", which is exactly what happened, and it already
          // exists in all four languages. The identifier is internal; adding
          // one for a distinction the user cannot see would mean four
          // translations of the same sentence.
          return {
            ok: false, code: CODES.pip, resumable,
            error: err.message, log: tailText(),
          };
        }
        touch();
        try {
          macosLockText = fs.readFileSync(path.join(temp, macosEnv.MACOS_LOCK_FILE), 'utf8');
        } catch { /* the record then reads the unpacked lock, which is the same file */ }
        currentStep = 'runtime';
      }

      const applier = path.join(__dirname, '..', 'apply_update.py');
      // --result-json, because the applier's exit code cannot distinguish "a
      // member tried to escape into Database" from "Python failed to start".
      // Without it, every refusal reads `python.exe exited with 1` and the one
      // sentence explaining what happened is printed and dropped.
      const verdictFile = path.join(temp, 'extract-result.json');
      try {
        await runStreaming(
          pythonExeIn(home),
          [applier, '--home', home, '--extract', zip,
           '--into', path.join(home, 'runtime'), '--result-json', verdictFile],
          { onLine: record },
        );
      } catch (err) {
        let detail = '';
        try {
          detail = String(JSON.parse(fs.readFileSync(verdictFile, 'utf8')).error || '');
        } catch { /* the verdict may not have been written at all */ }
        return {
          ok: false,
          code: CODES.package,
          resumable,
          error: detail
            || `unpacking the program files failed: ${err.message}`,
          log: tailText(),
        };
      }
      touch();
    }

    // A resumed run skipped steps 2, 3 AND 4, so on macOS it skipped the only
    // thing that creates an interpreter. Windows and Linux are covered further
    // down by step 4's "the lock file is missing" check; macOS has no step 4,
    // so without this a resumed run would walk to the end and write
    // `.python_path` for an interpreter that is not there — reporting a
    // finished installation that cannot start.
    //
    // Not reachable today (every darwin failure returns resumable=false), but
    // `resumable = true` below is unconditional, so it is one new failure path
    // away from being reachable, and it fails silently when it is.
    if (resume && !resumeHasSomethingToContinueFrom(home)) {
      return {
        ok: false, code: CODES.pip, resumable: false,
        error: 'the Python environment is missing, so there is nothing to '
          + 'continue from. Run the setup again from the beginning.',
      };
    }

    resumable = true;   // interpreter and program files are both on disk now

    // ---- 4. the packages -------------------------------------------------
    // On macOS this already happened, inside step 3: the environment had to
    // exist before the program files could be unpacked, because the unpacker
    // is a Python script. There is no pip step there at all -- the whole
    // point of the conda environment is that its packages come from
    // conda-forge, built against ONE OpenMP runtime.
    if (platform.current() !== 'darwin') {
    currentStep = 'packages';
    stepStarted = Date.now();
    tail = [];
    logSetupLine(home, `--- step: packages (mode=${mode}) ---`);
    writeMarker(incomplete, { mode, step: 'packages', tag: releaseTag || null });
    emit('packages', '');
    const lockPath = path.join(home, 'runtime', lockFileFor(mode));
    if (!fs.existsSync(lockPath)) {
      return { ok: false, code: CODES.unexpected, error: `${lockFileFor(mode)} is missing` };
    }
    const verdict = lockFileIsSafe(fs.readFileSync(lockPath, 'utf8'));
    if (!verdict.ok) {
      return { ok: false, code: CODES.unexpected, error: verdict.detail };
    }
    // Coming back as CPU after a failed CUDA check, the GPU stack is still
    // installed and pip will not remove it: nothing in the CPU lock file
    // mentions cupy or the nvidia-* wheels, so they simply stay — about 5 GB,
    // for ever, with .install_mode reading `cpu` so nothing downstream ever
    // knows to clean up. Worse, torch+cpu beside a working cupy is precisely
    // the configuration behind the 2026-08-03 `fused_max_only requires CUDA
    // tensors` incident: cupy's wheel answers runtimeGetVersion() with no
    // hardware behind it.
    if (resume && mode !== 'gpu') {
      emit('packages', 'removing the graphics-card packages');
      try {
        await runStreaming(
          pythonExeIn(home),
          ['-m', 'pip', 'uninstall', '-y', ...CUDA_ONLY_PACKAGES],
          { onLine: (line) => { record(line); emit('packages', line); } },
        );
      } catch {
        // Not fatal: leftover CUDA wheels waste disk, they do not break a CPU
        // install, and refusing to continue here would strand the user with no
        // working install at all.
        emit('packages', 'could not remove all graphics-card packages (harmless)');
      }
      touch();
    }
    try {
      await runStreaming(
        pythonExeIn(home),
        ['-m', 'pip', 'install', ...pipIndexArgs(mode), '-r', lockPath],
        { onLine: (line) => { record(line); emit('packages', line); } },
      );
    } catch (err) {
      // pip's own last words, not just its exit code. `python.exe exited with
      // 1` is true of a network failure, a disk-full, an unresolvable pin and a
      // corrupt wheel alike, and it is what a tester would otherwise paste into
      // an email twenty minutes into their first install.
      return {
        ok: false,
        code: runSetup.cancelled ? CODES.interrupted
          : (classify(err) === CODES.blocked ? CODES.blocked : CODES.pip),
        error: runSetup.cancelled
          ? 'the installation was stopped'
          : `installing the Python packages failed: ${err.message}`,
        resumable,
      };
    }
    touch();
    }

    // ---- 5. did the GPU build actually land? -----------------------------
    if (mode === 'gpu') {
      currentStep = 'verify';
      // Reset, or the heartbeat reports the elapsed time of the TWENTY-MINUTE
      // packages step against a step that began ten seconds ago -- at the very
      // end of a successful 8 GB install, where a user who concludes it has
      // hung and kills the window destroys the whole thing.
      stepStarted = Date.now();
      emit('verify', '');
      // `pip install` exits 0 on a CPU-only wheel, so the exit code proves
      // nothing. An override exists so this branch is testable on a machine
      // whose driver is fine — otherwise the fallback ships unexercised.
      const forced = process.env.ORIENTA_FORCE_CUDA_FAIL === '1';
      const cuda = forced
        ? { ok: false, detail: 'CUDA verification forced to fail (ORIENTA_FORCE_CUDA_FAIL=1)' }
        : await verifyCuda(pythonExeIn(home));
      touch();
      if (!cuda.ok && cuda.timedOut) {
        // The check timed out; the INSTALL is finished and correct. Offering
        // the CPU fallback here would throw away a working CUDA install over a
        // slow first import, and `.install_mode` would then record `cpu`
        // permanently. Finish as GPU and say the check could not be completed.
        // A code, not the English sentence: `result.warning` is rendered
        // verbatim on the success screen, so passing prose here puts an
        // English paragraph under a German heading.
        cudaUnverified = 'cudaUnverified';
      } else if (!cuda.ok) {
        return {
          ok: false,
          code: CODES.cuda,
          error: cuda.detail,
          canFallBackToCpu: true,
          // Python and the program files are both on disk, and the driver has
          // not changed -- updating one needs a reboot, which ends the wizard.
          // Coming back without this re-downloads 48 MB to reach the identical
          // failure.
          resumable: true,
          // pip DOES replace torch+cu126 with torch+cpu, so there is no second
          // torch. What it does not do is remove packages absent from the new
          // lock file — cupy-cuda12x and the eight nvidia-*-cu12 wheels, which
          // are the bulk of the 8 GB. `runSetup` uninstalls them explicitly on
          // the fallback path, so what the fallback needs is only headroom for
          // the CPU torch while they are still there.
          extraBytesForCpu: 2 * GB,
        };
      }
    }

    // ---- 6. record what we built -----------------------------------------
    fs.writeFileSync(path.join(home, '.python_path'), `${pythonExeIn(home)}\n`);
    fs.writeFileSync(path.join(home, '.install_mode'), `${mode}\n`);
    recordPackagesLock(home, mode, resolvedTag, macosLockText);
    fs.mkdirSync(path.join(home, 'runtime', 'Database'), { recursive: true });
    // Only here. Removing it in `finally` would erase the record of every
    // failure the moment it happened, which is the one case it is for.
    try { fs.rmSync(incomplete, { force: true }); } catch { /* ignore */ }

    currentStep = 'done';
    logSetupLine(home,
      `--- finished: ${resolvedTag || 'unknown'} (${mode}) in `
      + `${Math.round((Date.now() - started) / 1000)}s ---`);
    emit('done', '');
    return {
      ok: true, mode, home, tag: resolvedTag,
      warningCode: cudaUnverified || undefined,
    };
  } catch (err) {
    const message = String((err && err.message) || err);
    logSetupLine(home, `--- failed in step ${currentStep}: ${message} ---`);
    return {
      ok: false,
      code: runSetup.cancelled ? CODES.interrupted : classify(err, currentStep),
      error: runSetup.cancelled ? 'the installation was stopped' : message,
      searchedDirs: err && err.searchedDirs ? err.searchedDirs : undefined,
    };
  } finally {
    // Each cleanup in its own try: a throw in `finally` REPLACES the value the
    // function was about to return, turning a successful install into a frozen
    // wizard.
    try { if (heartbeat) clearInterval(heartbeat); } catch { /* ignore */ }
    try { rmrf(temp); } catch { /* ignore */ }
  }
}

/**
 * Children spawned by the setup, so the app can kill them on quit.
 *
 * Without this, closing the window during setup leaves a multi-gigabyte pip
 * writing into site-packages after the app is gone — and the next launch, whose
 * in-flight guard is per-process, starts a second one into the same tree.
 */
runSetup.children = new Set();

/** Set by killChildren, read by runSetup's catch.
 *
 * Without it, cancelling produces `taskkill` -> `python.exe exited with 1` ->
 * no pattern matches -> "Something unexpected went wrong", shown to the one
 * user who knows exactly what went wrong because they just asked for it. Every
 * signal on that screen then says the installer broke.
 */
runSetup.cancelled = false;
runSetup.killChildren = function killChildren() {
  runSetup.cancelled = true;
  for (const child of runSetup.children) {
    // `child.kill()` ends ONE pid on Windows, and the child that matters here
    // is pip, which spawns its own workers — so the multi-gigabyte write this
    // function exists to stop would carry on under a killed parent. This
    // project already settled the question for the backend (taskkill /T,
    // 2026-08-05); the setup needs the same answer.
    try {
      if (child.pid) {
        const plan = platform.killTree(child.pid);
        if (plan.kind === 'command') {
          execFile(plan.command, plan.args, { windowsHide: true }, () => {});
        } else {
          // the children are in the spawned process group; the negative pid
          // reaches all of them, where `child.kill()` would reach only pip
          process.kill(plan.target, plan.signal);
        }
      }
    } catch { /* ignore */ }
    try { child.kill(); } catch { /* ignore */ }
  }
  runSetup.children.clear();
};

module.exports.logSetupLine = logSetupLine;
module.exports.recordPackagesLock = recordPackagesLock;
module.exports.runSetup = runSetup;
module.exports.CODES = CODES;
module.exports.classify = classify;
