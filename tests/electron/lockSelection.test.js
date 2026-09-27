/**
 * Which lock file an installation uses — flavour AND platform.
 *
 * The Windows locks are frozen from an environment somebody ran; the Linux
 * ones are resolved for Linux. Handing a Linux machine the Windows answer
 * fails on the first CUDA wheel, after the download, and the lock files are
 * what an installer executes — so the choice is worth pinning.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const requireCjs = createRequire(import.meta.url);
const installer = requireCjs('../../electron/setup/installer.js');
const REPO = path.resolve(import.meta.dirname, '..', '..');

describe('lockFileFor', () => {
  it('gives Windows what v0.4.5 shipped', () => {
    expect(installer.lockFileFor('gpu', 'win32')).toBe('requirements-lock-gpu.txt');
    expect(installer.lockFileFor('cpu', 'win32')).toBe('requirements-lock-cpu.txt');
  });

  it('gives Linux its own, because the Windows one cannot install there', () => {
    expect(installer.lockFileFor('gpu', 'linux')).toBe('requirements-lock-linux-gpu.txt');
    expect(installer.lockFileFor('cpu', 'linux')).toBe('requirements-lock-linux-cpu.txt');
  });

  it('refuses a platform it has no lock for instead of guessing', () => {
    // macOS builds from conda-forge (T3); handing it a pip lock would bring a
    // second OpenMP runtime, which is the bug this whole port started from.
    expect(() => installer.lockFileFor('cpu', 'darwin')).toThrow(/no pip lock file/);
  });

  it('treats anything that is not "gpu" as the processor flavour', () => {
    expect(installer.lockFileFor('', 'win32')).toBe('requirements-lock-cpu.txt');
    expect(installer.lockFileFor(undefined, 'linux')).toBe('requirements-lock-linux-cpu.txt');
  });
});

describe('the lock files themselves', () => {
  const read = (name) => fs.readFileSync(path.join(REPO, name), 'utf8');

  it('every one of them passes the installer own guard', () => {
    for (const name of ['requirements-lock-cpu.txt', 'requirements-lock-gpu.txt',
      'requirements-lock-linux-cpu.txt', 'requirements-lock-linux-gpu.txt']) {
      expect(installer.lockFileIsSafe(read(name))).toEqual({ ok: true, detail: '' });
    }
  });

  it('the Linux GPU lock asks for the CUDA version torch actually wants', () => {
    // The Windows lock pins 12.9 for every non-Darwin platform, and on Linux
    // torch 2.11.0+cu126 requires 12.6.x — that lock cannot resolve there.
    const text = read('requirements-lock-linux-gpu.txt');
    expect(text).toMatch(/^nvidia-cuda-runtime-cu12==12\.6\./m);
    expect(text).not.toMatch(/^nvidia-cuda-runtime-cu12==12\.9\./m);
    expect(text).toMatch(/^torch==2\.11\.0\+cu126$/m);
  });

  it('the Linux CPU lock carries no CUDA at all', () => {
    const text = read('requirements-lock-linux-cpu.txt');
    expect(text).toMatch(/^torch==2\.11\.0\+cpu$/m);
    expect(text).not.toMatch(/^nvidia-/m);
    expect(text).not.toMatch(/^cupy/m);
  });

  it('says out loud that it was resolved, not measured', () => {
    // A resolved lock and a frozen one look identical and are not: one has
    // been installed by somebody, the other has not.
    for (const name of ['requirements-lock-linux-cpu.txt', 'requirements-lock-linux-gpu.txt']) {
      expect(read(name)).toMatch(/RESOLVED, NOT MEASURED/);
      expect(read(name)).toMatch(/Resolved with: uv \d/);   // the tool version, for repeatability
    }
  });
});

describe('Linux installs the stack this project is measured on', () => {
  /**
   * Windows is where every result in this repository was produced. Linux
   * resolves its own lock, and an unpinned range resolves to whatever is
   * newest on the day the file is made — so without a guard the two drift
   * apart silently and nobody finds out, because the Linux install works.
   *
   * It was not theoretical: the first Linux locks carried kikuchipy 0.11.5,
   * numpy 2.4.6, numba 0.67.0, scikit-learn 1.9.1, orix 0.14.3 and a pymatgen
   * a year apart from the Windows ones. `safe_loader.py` patches kikuchipy's
   * Oxford reader with a body copied verbatim from 0.11.3 for anything below
   * 0.12 — on 0.11.5 that silently swaps a newer reader for the older copy.
   *
   * The CUDA packages are exempt: Linux torch wants 12.6 where Windows
   * resolved 12.9, and constraining them has no solution.
   */
  const CUDA = /^(torch|torchvision|cupy-cuda12x|nvidia-[\w.-]+|cuda-[\w.-]+|triton|triton-windows)$/;

  const pins = (file) => {
    const out = new Map();
    for (const raw of fs.readFileSync(path.join(REPO, file), 'utf8').split('\n')) {
      const body = raw.split('#')[0].split(';')[0].trim();
      if (!body || body.startsWith('-') || !body.includes('==')) continue;
      const [name, version] = body.split('==');
      out.set(name.trim().toLowerCase(), version.trim());
    }
    return out;
  };

  it.each([['gpu'], ['cpu']])('%s: every Windows pin is matched on Linux', (flavour) => {
    const win = pins(`requirements-lock-${flavour}.txt`);
    const lin = pins(`requirements-lock-linux-${flavour}.txt`);
    const drifted = [];
    for (const [name, version] of win) {
      if (CUDA.test(name)) continue;
      const there = lin.get(name);
      // Absent is fine — Windows-only packages (pywin32) do not exist there.
      if (there !== undefined && there !== version) {
        drifted.push(`${name}: linux ${there}, windows ${version}`);
      }
    }
    expect(drifted).toEqual([]);
  });

  it.each([['gpu'], ['cpu']])('%s: only torch comes from the PyTorch index', (flavour) => {
    /**
     * The guard for the defect this file's history is about.
     *
     * `--extra-index-url` does not mean "get torch from here": uv's default
     * strategy takes the FIRST index that has a package, so the mirror won
     * for everything it happens to host — certifi at 2022.12.7, urllib3 1.26,
     * pymatgen a year stale. Nothing caught it, because a lock resolved off
     * the wrong index looks exactly like one that was not.
     *
     * The version-parity test above cannot see this: certifi and urllib3 are
     * transitive, so the Windows lock does not pin them and has nothing to
     * compare against. So the generator records the index per package and
     * this reads it back.
     */
    const text = fs.readFileSync(path.join(REPO, `requirements-lock-linux-${flavour}.txt`), 'utf8');
    const lines = text.split('\n');
    const fromMirror = [];
    let current = null;
    for (const raw of lines) {
      const body = raw.trim();
      if (!raw.startsWith(' ') && body && !body.startsWith('#') && body.includes('==')) {
        current = body.split('==')[0].trim().toLowerCase();
      } else if (body.startsWith('# from https://download.pytorch.org') && current) {
        fromMirror.push(current);
      }
    }
    // It has to have found the annotations at all, or this passes vacuously
    // on a lock generated without --emit-index-annotation.
    expect(text).toMatch(/# from https:\/\/pypi\.org\/simple/);
    expect(fromMirror.sort()).toEqual(['torch', 'torchvision']);
  });

  it.each([['gpu'], ['cpu']])('%s: records no path from the machine that made it', (flavour) => {
    // A recorded path is the maintainer's directory layout, shipped inside
    // the runtime package — and it makes the file depend on the checkout it
    // came from, so `--check` can never report "up to date".
    const lines = fs.readFileSync(path.join(REPO, `requirements-lock-linux-${flavour}.txt`), 'utf8')
      .split('\n')
      .filter((l) => !l.includes('://'))              // package index urls
      .filter((l) => /(AppData|[Tt]emp[\\/]|\b[A-Za-z]:[\\/]|\/home\/|\/Users\/)/.test(l));
    expect(lines).toEqual([]);
  });
});

describe('the GPU cleanup list', () => {
  /**
   * Both pairs, not just the new one.
   *
   * The first version of this test read the Linux pair alone, and so could not
   * see that `triton-windows` — the name the platform that actually ships
   * installs triton under — had never been in the list at all.
   */
  const PAIRS = [
    ['requirements-lock-gpu.txt', 'requirements-lock-cpu.txt'],
    ['requirements-lock-linux-gpu.txt', 'requirements-lock-linux-cpu.txt'],
  ];

  it.each(PAIRS)('covers every CUDA package %s installs', (gpuFile, cpuFile) => {
    // Switching from the graphics-card install to the processor one uninstalls
    // this list. A package missing from it stays on disk — gigabytes the user
    // was told had been removed.
    const text = fs.readFileSync(path.join(REPO, gpuFile), 'utf8');
    const cpuText = fs.readFileSync(path.join(REPO, cpuFile), 'utf8');
    const names = (t) => t.split('\n')
      .map((l) => l.split('#')[0].trim())
      .filter((l) => l && !l.startsWith('-'))
      .map((l) => l.split(/[=<>! ;]/)[0].toLowerCase());
    const cpuNames = new Set(names(cpuText));
    const onlyOnGpu = names(text).filter((n) => !cpuNames.has(n));
    const cleaned = new Set(installer.CUDA_ONLY_PACKAGES.map((n) => n.toLowerCase()));
    const forgotten = onlyOnGpu.filter((n) => !cleaned.has(n));
    expect(forgotten).toEqual([]);
  });
});
