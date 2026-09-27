/**
 * What the setup decides on its own, before it downloads anything.
 *
 * The GPU choice is the one that has hurt users of this project before: CUDA 12
 * needs NVIDIA driver 527.41 or newer, and installing the CUDA stack onto an
 * older driver produces an application that runs, reports no error, and
 * silently computes everything on the CPU. Measured elsewhere in this repo:
 * 11.2 TFLOP/s against 0.71 — a factor of 16 nobody would notice as a defect.
 *
 * "Cannot tell" must never become the permissive answer. An unreadable drive
 * reporting unlimited space would recommend an ~8 GB install that dies
 * part-way, on a machine that could not have taken it.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createRequire } from 'node:module';

const resolver = createRequire(import.meta.url)('../../electron/platform.js');
import os from 'node:os';
import path from 'node:path';

import {
  MIN_DRIVER_FOR_CUDA12,
  CPU_BYTES_NEEDED,
  GPU_BYTES_NEEDED,
  parseNvidiaSmi,
  chooseRecommendation,
  freeBytesOn,
  pythonUrl,
  pythonArchiveName,
  pythonSha256Url,
  probe,
} from '../../electron/setup/installer.js';

const GB = 1024 ** 3;
// Every CUDA call below names a platform. These describe rules that exist on
// Windows and Linux and deliberately do NOT on a Mac, where
// chooseRecommendation answers "Orienta uses the processor on this Mac" and
// driverMeetsCuda12 is false for any driver string. Called bare they asked the
// HOST, so the first macOS run failed five of them for the Mac logic working
// exactly as designed.
const PC = 'win32';


// --------------------------------------------------------------------------
// reading nvidia-smi
// --------------------------------------------------------------------------

describe('parseNvidiaSmi', () => {
  it('reads name and driver from the csv output', () => {
    const gpu = parseNvidiaSmi('NVIDIA GeForce RTX 4070, 591.86, 12282 MiB\n', PC);
    expect(gpu.present).toBe(true);
    expect(gpu.name).toBe('NVIDIA GeForce RTX 4070');
    expect(gpu.driver).toBe('591.86');
    expect(gpu.cudaCapable).toBe(true);
  });

  it('marks a driver below the CUDA 12 floor as not capable', () => {
    expect(parseNvidiaSmi('NVIDIA GeForce GTX 1060, 470.05, 6144 MiB\n', PC).cudaCapable).toBe(false);
  });

  it('has a different CUDA 12 floor per platform, and both are the real one', () => {
    // NVIDIA numbers the same CUDA release differently: Windows says 527.41,
    // Linux says 525.60.13. MIN_DRIVER_FOR_CUDA12 resolves against the HOST,
    // so a test pinning 527.41 with toBeCloseTo compared a string to a number
    // on Linux and failed with "difference is NaN" -- for a constant that was
    // correct on both.
    expect(resolver.minDriverDisplay('win32')).toBe('527.41');
    expect(resolver.minDriverDisplay('linux')).toBe('525.60.13');
    // A Mac has no CUDA floor at all, and says so with null rather than a
    // number a user could act on.
    expect(resolver.minDriverDisplay('darwin')).toBeNull();
    // ...and whatever the module re-exports for THIS host is the host's own
    // answer. Derived rather than enumerated: the first version of this line
    // listed the values it expected and went red on macOS, where the right
    // answer is null and was simply not in the list.
    expect(MIN_DRIVER_FOR_CUDA12).toEqual(resolver.minDriverDisplay(process.platform));
  });

  it('accepts a driver exactly at the floor', () => {
    // Written out, not built from the constant: a test that stringifies the
    // threshold it compares against passes for ANY threshold, however encoded
    // -- which is how a Linux floor of 525.6013 survived its own guard.
    expect(parseNvidiaSmi('X, 527.41, 8192 MiB', PC).cudaCapable).toBe(true);
  });

  it('reads a three-part Linux driver version as a version, not a number', () => {
    // 525.105.17 is NEWER than the 525.60.13 floor; as floats 525.105 < 525.6,
    // so a float comparison rejects four real R525 drivers and installs the
    // CPU stack on a working CUDA machine.
    expect(resolver.compareVersions([525, 105, 17], [525, 60, 13])).toBe(1);
    for (const v of ['525.60.13', '525.105.17', '525.125.06', '525.147.05', '535.104.05']) {
      expect(resolver.driverMeetsCuda12(v, 'linux')).toBe(true);
    }
    for (const v of ['525.60.12', '520.61.05', 'not-a-version', '']) {
      expect(resolver.driverMeetsCuda12(v, 'linux')).toBe(false);
    }
  });

  it('never calls a Mac CUDA-capable, however new the driver reads', () => {
    // `version >= null` is `version >= 0`: a missing floor must fail closed.
    expect(resolver.driverMeetsCuda12('999.99.99', 'darwin')).toBe(false);
    expect(resolver.minDriverDisplay('darwin')).toBeNull();
  });

  it('treats empty or missing output as no GPU rather than throwing', () => {
    for (const junk of ['', '   ', '\n\n', null, undefined]) {
      expect(parseNvidiaSmi(junk, PC).present).toBe(false);
      expect(parseNvidiaSmi(junk, PC).cudaCapable).toBe(false);
    }
  });

  it('treats an unparseable driver as not CUDA capable, not as capable', () => {
    // A machine whose driver version we cannot read is not a machine we should
    // hand the CUDA stack to.
    const gpu = parseNvidiaSmi('NVIDIA GeForce RTX 4070, N/A, 12282 MiB', PC);
    expect(gpu.present).toBe(true);
    expect(gpu.driver).toBeNull();
    expect(gpu.cudaCapable).toBe(false);
  });

  it('reads the first GPU when there are several', () => {
    const gpu = parseNvidiaSmi('NVIDIA A100, 550.54, 40960 MiB\nNVIDIA T400, 550.54, 4096 MiB\n', PC);
    expect(gpu.name).toBe('NVIDIA A100');
  });
});

// --------------------------------------------------------------------------
// the recommendation
// --------------------------------------------------------------------------

const GOOD_GPU = { present: true, cudaCapable: true, name: 'RTX 4070', driver: 591.86 };

describe('chooseRecommendation', () => {
  it('recommends the GPU when the driver is new enough and there is room', () => {
    const r = chooseRecommendation(GOOD_GPU, 50 * GB, true, PC);
    expect(r.recommendation).toBe('gpu');
    expect(r.blocked).toBeFalsy();
    expect(r.reason).toContain('RTX 4070');
  });

  it('recommends CPU and names the driver when it is too old', () => {
    const r = chooseRecommendation(
      { present: true, cudaCapable: false, name: 'GTX 1060', driver: 470.05 }, 50 * GB, true, PC);
    expect(r.recommendation).toBe('cpu');
    // The person can act on this: updating the driver is a real option, and
    // they will only know to do it if the number is in front of them.
    expect(r.reason).toMatch(/470|527/);
  });

  it('recommends CPU when there is no NVIDIA GPU', () => {
    const r = chooseRecommendation({ present: false }, 50 * GB, true, PC);
    expect(r.recommendation).toBe('cpu');
    expect(r.blocked).toBeFalsy();
  });

  it('recommends CPU when the disk cannot hold the GPU stack', () => {
    const r = chooseRecommendation(GOOD_GPU, 4 * GB, true, PC);
    expect(r.recommendation).toBe('cpu');
    expect(r.reason.toLowerCase()).toContain('space');
  });

  it('blocks outright when there is not even room for the CPU stack', () => {
    const r = chooseRecommendation({ present: false }, 1 * GB, true, PC);
    expect(r.blocked).toBe(true);
    expect(r.reason).toMatch(/\d/);      // says how much there is
  });

  it('blocks on zero free bytes — "cannot tell" is not "plenty"', () => {
    expect(chooseRecommendation(GOOD_GPU, 0, true, PC).blocked).toBe(true);
  });

  it('reports the space it measured, so the numbers can be checked', () => {
    const r = chooseRecommendation(GOOD_GPU, 4 * GB, true, PC);
    expect(r.freeBytes).toBe(4 * GB);
    expect(r.neededBytes).toBe(GPU_BYTES_NEEDED);
  });

  it('never recommends gpu without a cuda-capable gpu', () => {
    const cases = [
      [{ present: false }, 500 * GB],
      [{ present: true, cudaCapable: false, name: 'X', driver: 400 }, 500 * GB],
      [GOOD_GPU, 3 * GB],
    ];
    for (const [gpu, free] of cases) {
      expect(chooseRecommendation(gpu, free, true, PC).recommendation).toBe('cpu');
    }
  });

  it('answers for the platform it was GIVEN, from any host', () => {
    // chooseRecommendation takes a platformName and, for the one branch that
    // matters most, ignored it: `platform.canUseCuda()` was called bare, so it
    // answered for the HOST. Asked about darwin from Windows the Mac branch
    // never fired and a Mac user's text could not be checked from anywhere but
    // a Mac -- which is exactly how it survived until this suite first ran on
    // one.
    //
    // A Mac must never be told to update an NVIDIA driver: Apple machines have
    // no NVIDIA card and nothing the user could do would change that.
    const none = { present: false, cudaCapable: false };
    for (const pc of ['win32', 'linux']) {
      expect(chooseRecommendation(none, 500e9, true, pc).reason)
        .toMatch(/No NVIDIA graphics card found/);
    }
    const mac = chooseRecommendation(none, 500e9, true, 'darwin');
    expect(mac.recommendation).toBe('cpu');
    expect(mac.reason).toMatch(/on this Mac/);
    expect(mac.reason).not.toMatch(/[Uu]pdate the NVIDIA driver/);
    // and the Mac floor is the bigger one -- conda keeps its package cache
    // alongside the environment it is building
    expect(mac.neededBytes).toBeGreaterThan(
      chooseRecommendation(none, 500e9, true, 'win32').neededBytes);
  });

  it('survives a missing gpu object', () => {
    expect(() => chooseRecommendation(null, 50 * GB, true, PC)).not.toThrow();
    expect(chooseRecommendation(null, 50 * GB, true, PC).recommendation).toBe('cpu');
  });

  it('the thresholds are ordered and documented in bytes', () => {
    expect(CPU_BYTES_NEEDED).toBeLessThan(GPU_BYTES_NEEDED);
    // Measured 2026-09-17: CPU environment 1.87 GB, GPU environment 7.97 GB.
    expect(CPU_BYTES_NEEDED).toBeGreaterThan(1.87 * GB);
    expect(GPU_BYTES_NEEDED).toBeGreaterThan(7.97 * GB);
  });
});

// --------------------------------------------------------------------------
// free space
// --------------------------------------------------------------------------

describe('freeBytesOn', () => {
  it('returns a positive number for a real directory', () => {
    expect(freeBytesOn(os.tmpdir())).toBeGreaterThan(0);
  });

  it('returns 0 rather than infinity when it cannot tell', () => {
    expect(freeBytesOn('\\\\?\\no-such-volume\\x')).toBe(0);
    expect(freeBytesOn(null)).toBe(0);
  });
});

// --------------------------------------------------------------------------
// the whole probe
// --------------------------------------------------------------------------

describe('probe', () => {
  it('answers with a recommendation and a reason even for a path that does not exist', async () => {
    // The Orienta home does not exist yet on a first run, so the probe has to
    // walk up to a parent that does before it can measure anything.
    const result = await probe(path.join(os.tmpdir(), 'orienta-probe', 'not', 'there'));
    expect(['cpu', 'gpu']).toContain(result.recommendation);
    expect(typeof result.reason).toBe('string');
    expect(result.reason.length).toBeGreaterThan(0);
    expect(result.freeBytes).toBeGreaterThan(0);
  });

  it('never throws, whatever nvidia-smi does', async () => {
    await expect(probe(os.tmpdir())).resolves.toBeTruthy();
  });
});

// --------------------------------------------------------------------------
// external URLs
// --------------------------------------------------------------------------

// Windows and Linux only, and the skip is the assertion.
//
// macOS does not download a python-build-standalone interpreter at all: it
// builds its environment with micromamba from a conda lock, because the pip
// wheels of torch, scikit-learn and faiss each carry their own OpenMP runtime
// and the second one to initialise aborts the process. So
// requirePythonBuildStandalone THROWS on darwin, on purpose, and every test
// below asks about a download that does not exist there.
//
// Skipped rather than made platform-agnostic, for the same reason the NSIS
// tests skip off Windows: the boundary is real, and a skip states it.
describe.skipIf(process.platform === 'darwin')('the interpreter download', () => {
  const saved = {};
  beforeEach(() => {
    for (const k of ['ORIENTA_PYTHON_URL', 'ORIENTA_PYTHON_SHA256_URL']) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
  });
  afterEach(() => {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  });

  it('has https defaults', () => {
    expect(pythonUrl()).toMatch(/^https:\/\//);
    expect(pythonSha256Url()).toMatch(/^https:\/\//);
  });

  it('points at the aggregate SHA256SUMS, not a per-asset .sha256', () => {
    // MEASURED 2026-09-20 against release 20260901: 871 assets, exactly ZERO
    // of them .sha256, and one aggregate SHA256SUMS. Deriving
    // `<archive>.sha256` gave a 404 — and because the setup refuses to install
    // without a digest, that would have made every install impossible, after
    // the 48 MB download.
    expect(pythonSha256Url()).toMatch(/SHA256SUMS$/);
    expect(pythonSha256Url()).not.toMatch(/\.sha256$/);
  });

  it('names the archive so the aggregate can be read for the right line', () => {
    // An aggregate answered with its FIRST line gives the digest of a
    // different file, and the install then fails with a mismatch that blames
    // the download. The shared parser refuses a multi-entry document unless it
    // is told which file is wanted.
    expect(pythonArchiveName()).toBe(pythonUrl().split('/').pop());
    expect(pythonArchiveName()).toMatch(/\.tar\.gz$/);
  });

  it('is overridable, like every other external URL in this project', () => {
    // A pinned URL the project does not control is a single point of failure
    // for every future install: the release can be yanked, the host can be
    // blocked, a lab can need a mirror.
    process.env.ORIENTA_PYTHON_URL = 'https://mirror.invalid/python.tar.gz';
    expect(pythonUrl()).toBe('https://mirror.invalid/python.tar.gz');
    expect(pythonArchiveName()).toBe('python.tar.gz');

    process.env.ORIENTA_PYTHON_SHA256_URL = 'https://mirror.invalid/SHA256SUMS';
    expect(pythonSha256Url()).toBe('https://mirror.invalid/SHA256SUMS');
  });

  it('names a cp311 build for each platform that installs one', () => {
    // The lock files pin cp311 wheels; an interpreter of another version would
    // fail at pip install time, after the download.
    //
    // Asked of a NAMED platform. `pythonUrl()` resolves against the host, so
    // on Linux this asserted /windows|win/ against a linux-gnu tarball and
    // went red for a function doing exactly its job. Both are checked now,
    // which is more than the Windows-only version ever did.
    const win = resolver.runtimeSource('win32', 'x64', {});
    expect(win.url).toMatch(/3\.11/);
    expect(win.url).toMatch(/windows|win/i);

    const linux = resolver.runtimeSource('linux', 'x64', {});
    expect(linux.url).toMatch(/3\.11/);
    expect(linux.url).toMatch(/linux/i);
  });
});
