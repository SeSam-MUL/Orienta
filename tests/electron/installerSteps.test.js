/**
 * The setup's contract with the rest of the application.
 *
 * Two things here decide whether an install is right or silently wrong:
 *
 * The CUDA verification, because `pip install` exits 0 while installing a
 * CPU-only wheel. The only way to know what landed is to ask torch, and the
 * cost of not asking is a factor of 16 nobody recognises as a defect.
 *
 * The index choice, because `torch==2.11.0+cu126` satisfies `==2.11.0` AND
 * sorts above it, so offering the CUDA index in cpu mode hands the ~8 GB wheel
 * to exactly the user who was routed to CPU for lack of disk space.
 */
import { describe, it, expect, afterEach } from 'vitest';
import path from 'node:path';
import fs from 'node:fs';

import * as installer from '../../electron/setup/installer.js';
import * as P from '../../electron/platform.js';

const {
  lockFileFor,
  pythonExeIn,
  pipIndexArgs,
  pytorchIndex,
  lockFileIsSafe,
  verifyCudaFromOutput,
  verifyCuda,
  sha256File,
  fetchText,
  fetchToFile,
  downloadVerified,
  runSetup,
  tarExe,
} = installer;

describe('every name the rest of the setup depends on', () => {
  it('is exported and callable', () => {
    // Appending functions to a module whose module.exports was already closed
    // leaves them undefined at the one place that calls them — and the failure
    // surfaces as "runSetup is not a function" on a user's first install.
    const required = {
      lockFileFor, pythonExeIn, pipIndexArgs, pytorchIndex, lockFileIsSafe,
      verifyCudaFromOutput, verifyCuda, sha256File, fetchText, fetchToFile,
      downloadVerified, runSetup, tarExe,
    };
    for (const [name, fn] of Object.entries(required)) {
      expect(typeof fn, `${name} is not exported`).toBe('function');
    }
  });
});

// --------------------------------------------------------------------------
// which packages, from which index
// --------------------------------------------------------------------------

describe('lockFileFor', () => {
  // Asked of a NAMED platform, never of the host.
  //
  // These read `lockFileFor('gpu')` and pinned the Windows answer. The default
  // argument is `process.platform`, so the first Linux run of this suite
  // failed with "expected requirements-lock-gpu.txt, received
  // requirements-lock-linux-gpu.txt" -- the function being exactly right and
  // the test being written on one platform.
  //
  // The comment above `pythonExeIn` in this same file predicted this, for that
  // function, and that fix did not reach this block. Which is the lesson worth
  // keeping: a habit fixed at one call site is not a habit fixed.
  it('picks the lock file for the flavour, per platform', () => {
    expect(lockFileFor('gpu', 'win32')).toBe('requirements-lock-gpu.txt');
    expect(lockFileFor('cpu', 'win32')).toBe('requirements-lock-cpu.txt');
    expect(lockFileFor('gpu', 'linux')).toBe('requirements-lock-linux-gpu.txt');
    expect(lockFileFor('cpu', 'linux')).toBe('requirements-lock-linux-cpu.txt');
  });

  it('has no pip lock for macOS, and says so', () => {
    // macOS builds its environment from conda-forge. Returning a Windows name
    // there would send the wizard looking for a file that cannot help it.
    expect(() => lockFileFor('cpu', 'darwin')).toThrow(/darwin/);
  });

  it('treats anything unrecognised as cpu', () => {
    // The smaller, safer install is the right default for an unknown answer.
    expect(lockFileFor(undefined, 'win32')).toBe('requirements-lock-cpu.txt');
    expect(lockFileFor('GPU ', 'win32')).toBe('requirements-lock-cpu.txt');
  });

  it('names lock files that actually exist in this repository', () => {
    // Both platforms' names, from whichever platform runs the suite: the
    // package carries every lock, so every name must resolve to a real file
    // wherever this runs.
    const root = path.resolve(import.meta.dirname, '..', '..');
    for (const platform of ['win32', 'linux']) {
      for (const mode of ['cpu', 'gpu']) {
        const name = lockFileFor(mode, platform);
        expect(fs.existsSync(path.join(root, name)), name).toBe(true);
      }
    }
  });
});

describe('pytorchIndex', () => {
  const vars = ['ORIENTA_PYTORCH_INDEX_GPU', 'ORIENTA_PYTORCH_INDEX_CPU'];
  afterEach(() => vars.forEach((v) => delete process.env[v]));

  it('has a default per flavour', () => {
    expect(pytorchIndex('gpu')).toContain('cu126');
    expect(pytorchIndex('cpu')).toMatch(/whl\/cpu$/);
  });

  it('is overridable — 2 to 8 GB of every install comes through it', () => {
    process.env.ORIENTA_PYTORCH_INDEX_GPU = 'https://mirror.invalid/gpu';
    process.env.ORIENTA_PYTORCH_INDEX_CPU = 'https://mirror.invalid/cpu';
    expect(pytorchIndex('gpu')).toBe('https://mirror.invalid/gpu');
    expect(pytorchIndex('cpu')).toBe('https://mirror.invalid/cpu');
    expect(pipIndexArgs('gpu')).toContain('https://mirror.invalid/gpu');
  });
});

describe('pipIndexArgs', () => {
  it('never offers the CUDA index in cpu mode', () => {
    // torch==2.11.0+cu126 satisfies ==2.11.0 and sorts above it, so merely
    // OFFERING the index is enough to install the 8 GB wheel.
    expect(pipIndexArgs('cpu').join(' ')).not.toMatch(/cu\d{3}/);
  });

  it('offers the CUDA index in gpu mode', () => {
    expect(pipIndexArgs('gpu').join(' ')).toContain('cu126');
  });

  it('forbids source distributions in both modes', () => {
    for (const mode of ['cpu', 'gpu']) {
      expect(pipIndexArgs(mode)).toContain('--only-binary');
    }
  });

  it('turns off pip\'s progress bar', () => {
    // pip suppresses it on a pipe anyway; saying so keeps the output parseable
    // and makes the heartbeat the single source of "still working".
    expect(pipIndexArgs('cpu').join(' ')).toContain('--progress-bar off');
  });
});

// --------------------------------------------------------------------------
// where the interpreter lands
// --------------------------------------------------------------------------

describe('pythonExeIn', () => {
  // Asked of a named platform rather than of the host. These pinned the
  // Windows answers by running ON Windows, which means they would have FAILED
  // — not skipped — the first time the suite ran on the macOS runner, and the
  // port would have been blamed for it. platform.js takes the platform as an
  // argument precisely so a test can ask for another one; asking keeps the
  // Windows contract pinned everywhere the suite runs.
  it('points inside the managed environment on Windows', () => {
    expect(P.interpreterIn('C:\\x', 'win32')).toBe(path.join('C:\\x', 'python', 'python.exe'));
  });

  it('is the same answer the installer uses on this host', () => {
    // The wrapper adds nothing; if it ever does, that is worth knowing.
    expect(pythonExeIn('C:\\x')).toBe(P.interpreterIn('C:\\x'));
  });
});

describe('tarExe', () => {
  it('names Windows\' own bsdtar by absolute path', () => {
    // `tar` on PATH may be GNU tar (Git for Windows, MSYS2, Cygwin), which
    // cannot read this archive and fails confusingly after a 48 MB download.
    const win = P.tarExe('win32', { SystemRoot: 'C:\\Windows' });
    // path.win32, not path: node's `path` follows the HOST, so
    // path.isAbsolute('C:\\Windows\\System32\\tar.exe') is FALSE on Linux and
    // this went red for a function that was entirely right.
    expect(path.win32.isAbsolute(win)).toBe(true);
    expect(win.toLowerCase()).toMatch(/system32[\\/]tar\.exe$/);
  });

  it('uses the system tar on macOS and Linux', () => {
    // bsdtar on macOS, GNU tar on Linux — both read this archive, and neither
    // has the PATH problem Windows has.
    expect(P.tarExe('darwin', {})).toBe('/usr/bin/tar');
    expect(P.tarExe('linux', {})).toBe('/usr/bin/tar');
  });

  // The "derives the path from the environment" test that used to sit here
  // compared tarExe() against the same expression it had just assigned to
  // SystemRoot, so it held for any implementation that concatenates, and it
  // restored the variable in a way that wrote the string "undefined" when it
  // had been unset. Its real contract -- that tarExe THROWS without
  // SystemRoot -- is tested in installerHardening.test.js.
});

// --------------------------------------------------------------------------
// the lock file we are about to execute
// --------------------------------------------------------------------------

describe('lockFileIsSafe', () => {
  it('accepts an ordinary lock file', () => {
    expect(lockFileIsSafe('kikuchipy==0.11.3\ntorch==2.11.0+cpu\n').ok).toBe(true);
  });

  it('accepts the real lock files this repository ships, all four', () => {
    // All four, not "whichever two this host would install". Reading
    // lockFileFor(mode) without a platform meant Windows checked the Windows
    // pair and Linux the Linux pair -- each run covering half, and neither
    // saying so. The package carries every lock, so every lock is checked.
    const root = path.resolve(import.meta.dirname, '..', '..');
    for (const platform of ['win32', 'linux']) {
      for (const mode of ['cpu', 'gpu']) {
        const name = lockFileFor(mode, platform);
        const verdict = lockFileIsSafe(fs.readFileSync(path.join(root, name), 'utf8'));
        expect(verdict.ok, `${name}: ${verdict.detail}`).toBe(true);
      }
    }
  });

  it('refuses a lock that redirects pip somewhere else', () => {
    expect(lockFileIsSafe('--index-url http://evil/\n').ok).toBe(false);
    expect(lockFileIsSafe('--index-url=http://evil/\n').ok).toBe(false);
    expect(lockFileIsSafe('--extra-index-url http://evil/\n').ok).toBe(false);
  });

  it('refuses a lock that installs from a direct URL', () => {
    expect(lockFileIsSafe('thing @ https://evil/x.tar.gz\n').ok).toBe(false);
    expect(lockFileIsSafe('https://evil/x-1.0-py3-none-any.whl\n').ok).toBe(false);
  });

  it('allows the PyTorch indexes we ship, including an overridden one', () => {
    expect(lockFileIsSafe('--extra-index-url https://download.pytorch.org/whl/cpu\n').ok).toBe(true);
    process.env.ORIENTA_PYTORCH_INDEX_CPU = 'https://mirror.invalid/cpu';
    try {
      // Otherwise a mirrored lock file is rejected as "redirects pip".
      expect(lockFileIsSafe('--extra-index-url https://mirror.invalid/cpu\n').ok).toBe(true);
    } finally {
      delete process.env.ORIENTA_PYTORCH_INDEX_CPU;
    }
  });

  it('allows --hash, which a hashed lock would carry', () => {
    // Otherwise the documented hardening (--require-hashes) is unusable.
    expect(lockFileIsSafe('kikuchipy==0.11.3 --hash=sha256:abc\n').ok).toBe(true);
  });

  it('ignores comments and blank lines', () => {
    expect(lockFileIsSafe('# --index-url http://evil/\n\n  \nkikuchipy==1.0\n').ok).toBe(true);
  });

  it('says WHICH line it refused', () => {
    const verdict = lockFileIsSafe('kikuchipy==1.0\n--index-url http://evil/\n');
    expect(verdict.detail).toContain('evil');
  });
});

// --------------------------------------------------------------------------
// what actually landed
// --------------------------------------------------------------------------

describe('verifyCudaFromOutput', () => {
  it('accepts a CUDA build that reports a usable device', () => {
    expect(verifyCudaFromOutput('2.11.0+cu126 True NVIDIA GeForce RTX 4070').ok).toBe(true);
  });

  it('rejects a CPU wheel and says so plainly', () => {
    const r = verifyCudaFromOutput('2.14.0+cpu False');
    expect(r.ok).toBe(false);
    expect(r.detail).toMatch(/cpu/i);
  });

  it('rejects a CUDA build whose driver cannot run it', () => {
    const r = verifyCudaFromOutput('2.11.0+cu126 False');
    expect(r.ok).toBe(false);
    expect(r.detail).toMatch(/driver/i);
  });

  it('treats unparseable or empty output as a failure, not a pass', () => {
    for (const junk of ['', '   ', null, undefined, 'Traceback (most recent call last):']) {
      expect(verifyCudaFromOutput(junk).ok).toBe(false);
    }
  });

  it('is not fooled by a plain version with no build tag', () => {
    // PyPI's Linux torch IS the CUDA build and reports no +suffix; on Windows
    // the same string is the CPU one. Without a device it is not usable either
    // way, which is what we actually care about.
    expect(verifyCudaFromOutput('2.11.0 False').ok).toBe(false);
  });
});

// --------------------------------------------------------------------------
// downloads
// --------------------------------------------------------------------------

describe('sha256File', () => {
  it('matches a known digest', async () => {
    const crypto = await import('node:crypto');
    const os = await import('node:os');
    const file = path.join(os.tmpdir(), `orienta-sha-${process.pid}`);
    fs.writeFileSync(file, 'hello');
    try {
      const expected = crypto.createHash('sha256').update('hello').digest('hex');
      await expect(sha256File(file)).resolves.toBe(expected);
    } finally {
      fs.unlinkSync(file);
    }
  });
});

describe('the digest parser is the shared one', () => {
  it('is required from update_endpoints, not reimplemented', async () => {
    const source = fs.readFileSync(
      path.resolve(import.meta.dirname, '..', '..', 'electron', 'setup', 'installer.js'),
      'utf8',
    );
    expect(source).toMatch(/require\(['"]\.\.\/update_endpoints['"]\)/);
    // A second hex pattern here would be a second parser, and the two would
    // drift — which is how a .sha256 written on Windows came to parse on one
    // side and be refused on the other.
    expect(source).not.toMatch(/\[0-9a-f\]\{64\}/i);
    expect(source).not.toMatch(/\[a-fA-F0-9\]\{64\}/);
    // ...and it is actually CALLED. Importing the module while hand-rolling
    // the parse beside it would satisfy the require assertion on its own.
    expect(source).toMatch(/parseSha256Document\(/);
  });
});
