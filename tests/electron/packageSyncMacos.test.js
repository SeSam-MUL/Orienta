/**
 * Does an update bring a Mac's conda environment to the lock it shipped?
 *
 * This is the micromamba half of the package sync (`package_sync_macos.js`).
 * What is pinned here is every DECISION (what differs between the lock and the
 * disk, what the guard refuses), every ARGUMENT list, the classification of
 * micromamba's own words (the sentences are the ones it printed in the
 * measurements of 2026-10-09), and the ORDER of the steps with a fake
 * micromamba that mutates a real conda-meta directory the way the real one
 * does: nothing is unlinked before the download has succeeded and the marker
 * is on disk, and a run that fails before that leaves the directory as it was.
 *
 * What this cannot show is what osx-arm64 does; the rehearsal on a win-64
 * stand-in is described in the commit that added it.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const requireCjs = createRequire(import.meta.url);
const sync = requireCjs('../../electron/setup/package_sync.js');
const mac = requireCjs('../../electron/setup/package_sync_macos.js');
const macosEnv = requireCjs('../../electron/setup/macos_env.js');
const platform = requireCjs('../../electron/platform.js');

const REPO = path.resolve(import.meta.dirname, '..', '..');
// A Windows checkout with core.autocrlf=true has CRLF in the lock and the
// source; the tests build LF text from them and add their own CRLF case.
const readText = (p) => fs.readFileSync(p, 'utf8').replace(/\r\n/g, '\n');
const OLD_LOCK = readText(
  path.join(REPO, 'tests', 'fixtures', 'locks', 'v0.4.6', 'orienta-macos-lock.yml'));
const NEW_LOCK = readText(path.join(REPO, 'orienta-macos-lock.yml'));
const MAC_SRC = readText(path.join(REPO, 'electron', 'setup', 'package_sync_macos.js'));

/** What the 0.4.7 lock changes, written out (the same list as tests/test_lock_delta_budget.py). */
const EXPECTED = {
  kikuchipy: ['0.11.3', '0.13.1'],
  'kikuchipy-base': ['0.11.3', '0.13.1'],
  orix: ['0.14.1', '0.15.0'],
  'orix-base': ['0.14.1', '0.15.0'],
  pyebsdindex: ['0.3.9.1', '0.3.10.1'],
  'pyebsdindex-base': ['0.3.9.1', '0.3.10.1'],
  openssl: ['3.6.4', '3.6.5'],
  psygnal: [null, '0.16.1'],
  lazy_loader: [null, '0.5'],
};

let tmp;
beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-macsync-')); });
afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

const stemsOf = (lockText) => [...lockText.matchAll(/^ {2}url: \S*\/([^/\s]+?)\.(?:conda|tar\.bz2)\s*$/gm)].map((m) => m[1]);
const nameOfStem = (stem) => /^(.+)-[^-]+-[^-]+$/.exec(stem)[1];
const versionOfStem = (stem) => /^.+-([^-]+)-[^-]+$/.exec(stem)[1];

// --------------------------------------------------------------------------
// the real locks
// --------------------------------------------------------------------------

describe('reading the lock', () => {
  const parsed = mac.parseLock(NEW_LOCK);

  it('finds every package of the real lock, and keeps the header up to `package:`', () => {
    expect(parsed.packages).toHaveLength(469);
    expect(parsed.unsupported).toEqual([]);
    expect(parsed.foreign).toBe(0);
    expect(parsed.header.endsWith('\npackage:\n')).toBe(true);
    expect(parsed.header).toContain('osx-arm64: ');
    expect(NEW_LOCK.startsWith(parsed.header)).toBe(true);
  });

  it('takes name, version, URL and the file-name stem as written', () => {
    const k = parsed.packages.find((p) => p.name === 'kikuchipy');
    expect(k).toMatchObject({ version: '0.13.1' });
    expect(k.url).toMatch(/^https:\/\/conda\.anaconda\.org\/conda-forge\/noarch\/kikuchipy-0\.13\.1-/);
    expect(k.stem).toMatch(/^kikuchipy-0\.13\.1-[^-]+$/);
    expect(mac.stemFromUrl('https://x/y/openssl-3.6.5-h55eecbc_0.conda')).toBe('openssl-3.6.5-h55eecbc_0');
    expect(mac.stemFromUrl('https://x/y/six-1.16.0-py_0.tar.bz2?x=1')).toBe('six-1.16.0-py_0');
    expect(mac.versionFromStem('ca-certificates-2026.7.22-h4c7d964_0')).toBe('2026.7.22');
  });

  it('the stems it finds are the ones an independent reading of the URLs finds', () => {
    expect(parsed.packages.map((p) => p.stem)).toEqual(stemsOf(NEW_LOCK));
  });

  it('refuses a lock with no package list', () => {
    expect(() => mac.parseLock('version: 1\nmetadata:\n')).toThrow(/package:/);
  });

  it('sets aside a package that is not from conda: this flow moves nothing else', () => {
    const text = `${NEW_LOCK}- name: leftpad\n  version: '1.0'\n  manager: pip\n  platform: osx-arm64\n  url: https://x/leftpad-1.0.tar.gz\n`;
    expect(mac.parseLock(text).unsupported).toEqual(['leftpad']);
  });

  it('copes with CRLF', () => {
    const crlf = NEW_LOCK.replace(/\n/g, '\r\n');
    expect(mac.parseLock(crlf).packages).toHaveLength(469);
  });
});

describe('the delta', () => {
  const parsed = mac.parseLock(NEW_LOCK);
  const installedFrom = (lockText) => {
    const m = new Map();
    for (const stem of stemsOf(lockText)) m.set(nameOfStem(stem), [...(m.get(nameOfStem(stem)) || []), stem]);
    return m;
  };

  it('between the 0.4.6 environment and the 0.4.7 lock is exactly the nine packages the release notes name', () => {
    const { delta, duplicates, unchanged } = mac.computeDelta(parsed.packages, installedFrom(OLD_LOCK));
    const got = Object.fromEntries(delta.map((p) => [
      p.name, [p.from.length ? versionOfStem(p.from[0]) : null, p.version]]));
    expect(got).toEqual(EXPECTED);
    expect(duplicates).toEqual([]);
    expect(unchanged).toBe(469 - 9);
    expect(delta.map((p) => p.name)).toEqual([...delta.map((p) => p.name)].sort());
  });

  it('is empty for an environment that is already as locked', () => {
    const { delta, unchanged } = mac.computeDelta(parsed.packages, installedFrom(NEW_LOCK));
    expect(delta).toEqual([]);
    expect(unchanged).toBe(469);
  });

  it('leaves alone what is installed and not in the lock', () => {
    const have = installedFrom(NEW_LOCK);
    have.set('users-own-package', ['users-own-package-1.0-0']);
    expect(mac.computeDelta(parsed.packages, have).delta).toEqual([]);
  });

  it('counts a missing record as new, and two records as a difference even when one is right', () => {
    const have = installedFrom(NEW_LOCK);
    have.delete('psygnal');
    const k = have.get('kikuchipy');
    have.set('kikuchipy', ['kikuchipy-0.11.3-pyhd8ed1ab_0', ...k]);
    const { delta, duplicates } = mac.computeDelta(parsed.packages, have);
    expect(delta.map((p) => [p.name, p.from.length])).toEqual([['kikuchipy', 2], ['psygnal', 0]]);
    expect(duplicates).toEqual(['kikuchipy']);
  });

  it('the delta lock holds the header and exactly the delta blocks, byte for byte, in the lock order', () => {
    const { delta } = mac.computeDelta(parsed.packages, installedFrom(OLD_LOCK));
    const text = mac.buildDeltaLock(parsed, delta.map((p) => p.name));
    expect(text.startsWith(parsed.header)).toBe(true);
    const blocks = mac.parseLock(text);
    expect(blocks.packages.map((p) => p.name)).toEqual(
      parsed.packages.filter((p) => p.name in EXPECTED).map((p) => p.name));
    for (const p of blocks.packages) {
      expect(NEW_LOCK).toContain(p.text);
      expect(p.text.startsWith(`- name: ${p.name}\n`)).toBe(true);
    }
    expect(text.endsWith('\n')).toBe(true);
  });

  it('is a lock micromamba accepts as one -- and only with the floor lowered for a delta', () => {
    const { delta } = mac.computeDelta(parsed.packages, installedFrom(OLD_LOCK));
    const text = mac.buildDeltaLock(parsed, delta.map((p) => p.name));
    expect(macosEnv.lockFileComplaint(text, { minPackages: delta.length })).toBeNull();
    expect(macosEnv.lockFileComplaint(text)).toMatch(/only 9 package/);
    expect(macosEnv.lockFileComplaint(NEW_LOCK)).toBeNull();
    expect(macosEnv.isLockFileName(mac.DELTA_LOCK_FILE)).toBe(true);
  });
});

// --------------------------------------------------------------------------
// the guard
// --------------------------------------------------------------------------

describe('the guard', () => {
  const d = (...names) => names.map((name) => ({ name }));

  it('lets the 0.4.7 delta through', () => {
    const verdict = mac.guardDelta(d(...Object.keys(EXPECTED)));
    expect(verdict.ok).toBe(true);
  });

  it.each([
    'python', 'python_abi', 'llvm-openmp', '_openmp_mutex', 'libblas', 'libopenblas', 'libgfortran5',
    'libcxx', 'pytorch', 'libtorch', 'numpy', 'scipy', 'nvidia-cudnn', 'cupy',
  ])('refuses a delta that names %s', (name) => {
    const verdict = mac.guardDelta(d('kikuchipy', name));
    expect(verdict).toMatchObject({ ok: false, reason: 'guard' });
    expect(verdict.detail).toContain(name);
  });

  it('refuses a delta of more than sixty packages, and allows sixty', () => {
    const many = (n) => d(...Array.from({ length: n }, (_, i) => `pkg${i}`));
    expect(mac.guardDelta(many(mac.MAX_DELTA_PACKAGES)).ok).toBe(true);
    expect(mac.guardDelta(many(mac.MAX_DELTA_PACKAGES + 1)).reason).toBe('guard');
  });

  it('has no size rule at run time, as on the other platforms', () => {
    expect(mac.guardDelta(d('kikuchipy'), { kikuchipy: 10 ** 12 }).ok).toBe(true);
  });
});

// --------------------------------------------------------------------------
// argument lists
// --------------------------------------------------------------------------

describe('the arguments', () => {
  const root = '/home/Orienta/micromamba';
  const prefix = '/home/Orienta/envs/orienta';
  const lockFile = '/home/Orienta/setup-tmp/orienta-macos-delta-lock.yml';

  it('download: the install command with --download-only and the same isolation as the wizard', () => {
    expect(mac.downloadArgs({ root, prefix, lockFile })).toEqual([
      'install', '--no-rc', '--no-env', '-y', '-r', root, '-p', prefix,
      '--platform', 'osx-arm64', '--download-only', '-f', lockFile,
    ]);
  });

  it('remove: --force, exactly the names, and no --platform (remove does not take it)', () => {
    const args = mac.removeArgs({ root, prefix, names: ['kikuchipy', 'orix-base'] });
    expect(args).toEqual([
      'remove', '--no-rc', '--no-env', '-y', '-r', root, '-p', prefix, '--force', 'kikuchipy', 'orix-base',
    ]);
    expect(args).not.toContain('--platform');
  });

  it('install: the delta lock, from the cache', () => {
    const args = mac.installDeltaArgs({ root, prefix, lockFile });
    expect(args).toEqual([
      'install', '--no-rc', '--no-env', '-y', '-r', root, '-p', prefix,
      '--platform', 'osx-arm64', '-f', lockFile,
    ]);
    expect(args).not.toContain('--download-only');
  });

  it('list: JSON, in the same prefix', () => {
    expect(mac.listArgs({ root, prefix })).toEqual([
      'list', '--no-rc', '--no-env', '-r', root, '-p', prefix, '--json']);
  });

  it('the install and the download differ only by --download-only (so the cache the first fills is the one the second reads)', () => {
    const a = mac.downloadArgs({ root, prefix, lockFile }).filter((x) => x !== '--download-only');
    expect(a).toEqual(mac.installDeltaArgs({ root, prefix, lockFile }));
  });

  it.each(['--prefix', '-rf', 'a b', 'x;y', 'a/b', '', 'a$b'])('refuses %j as a package name', (bad) => {
    expect(() => mac.removeArgs({ root, prefix, names: ['kikuchipy', bad] })).toThrow(/not a package name/);
  });

  it('accepts every package name of the real lock', () => {
    const names = mac.parseLock(NEW_LOCK).packages.map((p) => p.name);
    expect(names.filter((n) => !mac.SAFE_CONDA_NAME.test(n))).toEqual([]);
    expect(() => mac.removeArgs({ root, prefix, names })).not.toThrow();
  });

  it('refuses a lock file whose name micromamba would not read as a lock (it would install NOTHING)', () => {
    expect(() => mac.downloadArgs({ root, prefix, lockFile: '/x/delta.yml' })).toThrow(/-lock\.yml/);
    expect(() => mac.installDeltaArgs({ root, prefix, lockFile: '/x/delta.yaml' })).toThrow(/-lock\.yml/);
  });
});

// --------------------------------------------------------------------------
// micromamba's own words
// --------------------------------------------------------------------------

describe('what went wrong', () => {
  // Verbatim from micromamba 2.9.0 (win-64), 2026-10-09.
  const CONNECT = 'warning  libmamba Download error (7) Could not connect to server [https://conda.anaconda.org/conda-forge/noarch/ca-certificates-2026.7.22-h4c7d964_0.conda]\n'
    + '    Failed to connect to conda.anaconda.org:443 over proxy 127.0.0.1 after 2007 ms: Could not connect to server\n'
    + 'warning  libmamba Retrying in 2 seconds';
  const DNS = 'critical libmamba Download error (6) Could not resolve hostname [https://nonexistent.invalid/conda-forge/noarch/x.conda]\n'
    + '    Could not resolve host: nonexistent.invalid';
  const SHA = 'error    libmamba File not valid: SHA256 doesn\'t match expectation "C:\\\\x\\\\pkgs\\\\ca.conda"\n'
    + '    Expected: 05e8\n    Actual: 95e8\nca-certificates-2026.7.22-h4c7d964_0.conda tarball has incorrect SHA256\n'
    + 'critical libmamba Found incorrect downloads. Aborting';

  it.each([
    ['a refused connection', CONNECT, 'network'],
    ['an unknown host', DNS, 'network'],
    ['a refused checksum', SHA, 'checksum'],
    ['a disk that is full', 'error libmamba [Errno 28] No space left on device', 'space'],
    ['a failed signature', 'critical libmamba Could not codesign executable: Invalid argument', 'signing'],
    ['something else', 'critical libmamba something unforeseen', 'other'],
    ['nothing at all', '', 'other'],
  ])('%s', (_label, text, expected) => {
    expect(mac.classifyMicromambaFailure({ text })).toBe(expected);
  });

  it('a timeout is a network failure whatever it printed', () => {
    expect(mac.classifyMicromambaFailure({ text: 'quiet', timedOut: true })).toBe('network');
  });

  it('a full disk is not mistaken for the network, and a bad checksum not for either', () => {
    expect(mac.classifyMicromambaFailure({ text: `${CONNECT}\nNo space left on device` })).toBe('space');
    expect(mac.classifyMicromambaFailure({ text: `${SHA}\n${CONNECT}` })).toBe('checksum');
  });
});

describe('reading `micromamba list --json`', () => {
  const list = JSON.stringify({
    log_history: [],
    packages: [
      { name: 'kikuchipy', version: '0.13.1', dist_name: 'kikuchipy-0.13.1-pyhd8ed1ab_0' },
      { name: 'openssl', version: '3.6.5' },
      { name: 'openssl', version: '3.6.4' },
    ],
  });

  it('maps names to every version listed, and reads through leading noise', () => {
    const m = mac.versionsFromList(list);
    expect(m.get('kikuchipy')).toEqual(['0.13.1']);
    expect(m.get('openssl')).toEqual(['3.6.5', '3.6.4']);
    expect(mac.versionsFromList(`warning something\n${list}`).get('kikuchipy')).toEqual(['0.13.1']);
  });

  it('answers null for anything that is not that', () => {
    expect(mac.versionsFromList('')).toBeNull();
    expect(mac.versionsFromList('[]')).toBeNull();
    expect(mac.versionsFromList('{"packages": 3}')).toBeNull();
  });

  it('the mismatches name a stale version, a missing one, and a second record', () => {
    const delta = [
      { name: 'kikuchipy', version: '0.13.1', stem: 'kikuchipy-0.13.1-pyhd8ed1ab_0' },
      { name: 'openssl', version: '3.6.5', stem: 'openssl-3.6.5-h55eecbc_0' },
      { name: 'psygnal', version: '0.16.1', stem: 'psygnal-0.16.1-py_0' },
    ];
    const listed = mac.versionsFromList(list);
    const installed = new Map([
      ['kikuchipy', ['kikuchipy-0.13.1-pyhd8ed1ab_0']],
      ['openssl', ['openssl-3.6.4-h_0', 'openssl-3.6.5-h55eecbc_0']],
    ]);
    const got = mac.mismatches(delta, listed, installed);
    expect(got.join('\n')).toMatch(/openssl: micromamba list shows 3\.6\.5 and 3\.6\.4/);
    expect(got.join('\n')).toMatch(/openssl: 2 record\(s\)/);
    expect(got.join('\n')).toMatch(/psygnal: micromamba list shows nothing/);
    expect(got.join('\n')).toMatch(/psygnal: 0 record\(s\)/);
    expect(got.join('\n')).not.toMatch(/kikuchipy/);
  });
});

// --------------------------------------------------------------------------
// the whole flow, with a micromamba that is a function
// --------------------------------------------------------------------------

/** A home that looks like a 0.4.6 macOS installation, and the lock 0.4.7 shipped. */
function macHome({ installed = OLD_LOCK, lock = NEW_LOCK, mode = 'cpu', files = {} } = {}) {
  const home = path.join(tmp, 'home');
  const prefix = path.join(home, 'envs', 'orienta');
  const python = path.join(prefix, 'bin', 'python');
  fs.mkdirSync(path.join(prefix, 'bin'), { recursive: true });
  fs.mkdirSync(path.join(prefix, 'conda-meta'), { recursive: true });
  fs.mkdirSync(path.join(home, 'runtime', 'scripts'), { recursive: true });
  fs.mkdirSync(path.dirname(platform.micromambaExeIn(home)), { recursive: true });
  fs.writeFileSync(python, '');
  fs.writeFileSync(platform.micromambaExeIn(home), '');
  fs.writeFileSync(path.join(home, 'runtime', 'scripts', 'check_runtime_health.py'), '');
  fs.writeFileSync(path.join(home, 'runtime', 'orienta-macos-lock.yml'), lock);
  if (mode !== null) fs.writeFileSync(path.join(home, '.install_mode'), `${mode}\n`);
  for (const stem of stemsOf(installed)) fs.writeFileSync(path.join(prefix, 'conda-meta', `${stem}.json`), '{}');
  for (const [name, body] of Object.entries(files)) {
    fs.writeFileSync(path.join(home, name), typeof body === 'string' ? body : JSON.stringify(body));
  }
  return { home, prefix, python, mm: platform.micromambaExeIn(home) };
}

const metaFiles = (h) => fs.readdirSync(path.join(h.prefix, 'conda-meta')).sort();
const exists = (h, name) => fs.existsSync(path.join(h.home, name));
const readJson = (h, name) => JSON.parse(fs.readFileSync(path.join(h.home, name), 'utf8'));

const reply = (over = {}) => ({
  code: 0, signal: null, timedOut: false, cancelled: false,
  stdout: '', stderr: '', output: '', error: null, ...over,
});

const CONNECT_TEXT = 'warning  libmamba Download error (7) Could not connect to server [https://conda.anaconda.org/conda-forge/noarch/kikuchipy-0.13.1-pyhd8ed1ab_0.conda]\n'
  + '    Failed to connect to conda.anaconda.org:443: Could not connect to server';

/**
 * micromamba and python as one function. It mutates the real conda-meta
 * directory the way micromamba does -- `remove` deletes records, `install -f`
 * writes one per package of the lock file it is handed, read from disk at the
 * moment of the call -- so a test can look at the directory between steps.
 * `scenario[kind]` is a function (n, ctx) => reply | undefined, n counting the
 * calls of that kind from 1; returning a reply replaces the whole step.
 */
function fakeMicromamba(h, scenario = {}) {
  const calls = [];
  const counts = {};
  const seen = { downloadLock: null, markerDuringRemove: null, metaDuringDownload: null };

  const classify = (exe, args) => {
    if (exe === h.mm) {
      if (args[0] === '--version') return 'version';
      if (args[0] === 'info') return 'info';
      if (args[0] === 'list') return 'list';
      if (args[0] === 'remove') return 'remove';
      if (args[0] === 'install') return args.includes('--download-only') ? 'download' : 'install';
      return 'mm-other';
    }
    if (args.includes('-c')) return 'imports';
    if (String(args[0]).endsWith('check_runtime_health.py')) return 'health';
    return 'other';
  };

  const lockStems = (file) => stemsOf(fs.readFileSync(file, 'utf8'));

  async function run(exe, args, opts = {}) {
    const kind = classify(exe, args);
    counts[kind] = (counts[kind] || 0) + 1;
    calls.push({ kind, exe, args, opts });
    const custom = scenario[kind] && scenario[kind](counts[kind], { exe, args, opts });
    if (custom) return custom;
    const dir = path.join(h.prefix, 'conda-meta');
    switch (kind) {
      case 'version': return reply({ stdout: '2.9.0\n' });
      case 'info': return reply({ stdout: '    platform : osx-arm64\n' });
      case 'download': {
        seen.downloadLock = fs.readFileSync(args[args.indexOf('-f') + 1], 'utf8');
        seen.metaDuringDownload = metaFiles(h);
        return reply({ output: 'Download only - packages download and extraction is done.' });
      }
      case 'remove': {
        seen.markerDuringRemove = exists(h, '.packages_sync.json');
        const names = args.slice(args.indexOf('--force') + 1);
        for (const f of fs.readdirSync(dir)) {
          if (names.includes(nameOfStem(f.replace(/\.json$/, '')))) fs.rmSync(path.join(dir, f));
        }
        return reply();
      }
      case 'install': {
        for (const stem of lockStems(args[args.indexOf('-f') + 1])) {
          fs.writeFileSync(path.join(dir, `${stem}.json`), '{}');
        }
        return reply();
      }
      case 'list': {
        const packages = fs.readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => {
          const stem = f.replace(/\.json$/, '');
          return { name: nameOfStem(stem), version: versionOfStem(stem) };
        });
        return reply({ stdout: JSON.stringify({ log_history: [], packages }) });
      }
      case 'imports': return reply();
      case 'health': return reply({ stdout: 'PASS  openmp files  exactly one runtime\n', output: 'PASS  openmp files' });
      default: throw new Error(`the test micromamba does not know ${exe} ${args.join(' ')}`);
    }
  }
  return { run, calls, counts, seen, kinds: () => calls.map((c) => c.kind) };
}

function runMac(h, fake, over = {}, ctx = {}) {
  const clock = { t: Date.parse('2026-10-09T12:00:00Z') };
  const log = [];
  const notify = vi.fn(async () => {});
  const onPhase = vi.fn();
  const deps = {
    run: fake.run,
    isCancelled: () => false,
    log: (line) => log.push(line),
    freeBytes: () => ({ bytes: 10 * 1024 ** 3, known: true }),
    now: () => clock.t,
    runtimeTag: () => 'v0.4.7',
    notify,
    onPhase,
    macos: { codesign: async () => {} },
    ...over,
  };
  const opts = {
    home: h.home, python: h.python, decisionMode: 'run', projectRootEnv: undefined,
    platformName: 'darwin', env: { PATH: '/usr/bin', MAMBA_ROOT_PREFIX: '/users/own', CONDA_PREFIX: '/users/own/env' },
    ...ctx,
  };
  return { result: sync.syncPackages(opts, deps), log, notify, onPhase, clock };
}

describe('the update of a 0.4.6 environment', () => {
  it('the page says "syncing" before the first process runs, not in the middle of the flow', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    const onPhase = vi.fn();
    const seen = [];
    const spy = async (exe, args, opts) => {
      seen.push(onPhase.mock.calls.map((c) => c[0]));
      return fake.run(exe, args, opts);
    };
    await runMac(h, fake, { run: spy, onPhase }).result;
    expect(fake.calls[0].kind).toBe('version');
    expect(seen[0]).toEqual(['syncing']);
  });

  it('an environment that is already as locked does not touch the page', async () => {
    const h = macHome({ installed: NEW_LOCK });
    const { result, onPhase } = runMac(h, fakeMicromamba(h));
    expect(await result).toMatchObject({ action: 'noop' });
    expect(onPhase).not.toHaveBeenCalled();
  });

  it('brings the nine packages to the lock, and only those', async () => {
    const h = macHome();
    const before = metaFiles(h);
    const fake = fakeMicromamba(h);
    const { result, log, onPhase } = runMac(h, fake);
    const res = await result;

    expect(res).toMatchObject({ ok: true, action: 'synced', how: 'sync' });
    expect(res.packages).toEqual(Object.fromEntries(Object.entries(EXPECTED).map(([k, v]) => [k, v[1]])));

    const after = metaFiles(h);
    expect(after).toEqual(stemsOf(NEW_LOCK).map((s) => `${s}.json`).sort());
    const removed = before.filter((f) => !after.includes(f));
    const added = after.filter((f) => !before.includes(f));
    expect(removed.map((f) => nameOfStem(f.slice(0, -5))).sort())
      .toEqual(Object.keys(EXPECTED).filter((n) => EXPECTED[n][0] !== null).sort());
    expect(added.map((f) => nameOfStem(f.slice(0, -5))).sort()).toEqual(Object.keys(EXPECTED).sort());

    expect(onPhase).toHaveBeenCalledWith('syncing');
    expect(exists(h, '.packages_sync.json')).toBe(false);
    expect(exists(h, '.packages_sync_failed.json')).toBe(false);
    expect(fs.existsSync(path.join(h.home, 'setup-tmp', mac.DELTA_LOCK_FILE))).toBe(false);
    expect(readJson(h, '.packages_lock.json')).toMatchObject({
      schema: 1, lock: 'orienta-macos-lock.yml', sha256: sync.lockDigest(NEW_LOCK), mode: 'cpu',
      platform: 'darwin', runtimeTag: 'v0.4.7', how: 'sync',
    });
    expect(log.join('\n')).toMatch(/460 of 469 packages are as locked; 9 differ/);
  });

  it('the steps run in this order: probes, download, remove, install, then the verification', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    await runMac(h, fake).result;
    expect(fake.kinds()).toEqual([
      'version', 'info', 'download', 'remove', 'install', 'list', 'imports', 'health']);
  });

  it('nothing is unlinked before the download is done, and the marker is on disk by then', async () => {
    const h = macHome();
    const before = metaFiles(h);
    const fake = fakeMicromamba(h);
    await runMac(h, fake).result;
    expect(fake.seen.metaDuringDownload).toEqual(before);
    expect(fake.seen.markerDuringRemove).toBe(true);
    expect(fake.kinds().indexOf('download')).toBeLessThan(fake.kinds().indexOf('remove'));
  });

  it('removes the packages that were there and not the ones that are new', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    await runMac(h, fake).result;
    const rm = fake.calls.find((c) => c.kind === 'remove').args;
    const names = rm.slice(rm.indexOf('--force') + 1);
    expect(names.sort()).toEqual(
      Object.keys(EXPECTED).filter((n) => EXPECTED[n][0] !== null).sort());
    expect(names).not.toContain('psygnal');
  });

  it('hands micromamba a delta lock of only the delta, at the time it downloads', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    await runMac(h, fake).result;
    const parsed = mac.parseLock(fake.seen.downloadLock);
    expect(parsed.packages.map((p) => p.name).sort()).toEqual(Object.keys(EXPECTED).sort());
    const dl = fake.calls.find((c) => c.kind === 'download').args;
    expect(path.basename(dl[dl.indexOf('-f') + 1])).toBe(mac.DELTA_LOCK_FILE);
  });

  it('runs micromamba in the wizard\'s isolation: the user\'s own conda settings never reach it', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    await runMac(h, fake).result;
    const mmCalls = fake.calls.filter((c) => c.exe === h.mm);
    expect(mmCalls.length).toBeGreaterThan(5);
    for (const c of mmCalls) {
      expect(c.opts.env).toBeTruthy();
      expect(Object.keys(c.opts.env).filter((k) => /^(MAMBA|CONDA)_/.test(k))).toEqual([]);
      expect(c.opts.env.PATH).toBe('/usr/bin');
    }
    const ints = fake.calls.filter((c) => c.kind === 'download' || c.kind === 'install' || c.kind === 'remove');
    for (const c of ints) {
      expect(c.args).toEqual(expect.arrayContaining(['--no-rc', '--no-env', '-y', '-r', path.join(h.home, 'micromamba')]));
    }
  });

  it('verifies with the project\'s own OpenMP gate, run by the environment\'s interpreter', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    await runMac(h, fake).result;
    const gate = fake.calls.find((c) => c.kind === 'health');
    expect(gate.exe).toBe(h.python);
    expect(gate.args).toEqual([
      path.join(h.home, 'runtime', 'scripts', 'check_runtime_health.py'),
      '--prefix', h.prefix, '--gate', 'files']);
    const imp = fake.calls.find((c) => c.kind === 'imports');
    expect(imp.exe).toBe(h.python);
    expect(imp.args).toEqual(['-I', '-c', 'import kikuchipy, orix, pyebsdindex']);
  });

  it('a second start takes the fast path: no process at all', async () => {
    const h = macHome();
    await runMac(h, fakeMicromamba(h)).result;
    const again = fakeMicromamba(h);
    const res = await runMac(h, again).result;
    expect(res).toEqual({ ok: true, action: 'fastpath' });
    expect(again.calls).toEqual([]);
  });

  it('an environment that is already as locked is recorded without a process', async () => {
    const h = macHome({ installed: NEW_LOCK });
    const fake = fakeMicromamba(h);
    const res = await runMac(h, fake).result;
    expect(res).toEqual({ ok: true, action: 'noop' });
    expect(fake.calls).toEqual([]);
    expect(readJson(h, '.packages_lock.json').how).toBe('noop');
  });

  it('the opt-out is honoured on a Mac too', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h);
    const res = await runMac(h, fake, {}, { env: { ORIENTA_SKIP_PACKAGE_SYNC: '1' } }).result;
    expect(res).toMatchObject({ ok: true, action: 'skip', reason: 'opt-out' });
    expect(fake.calls).toEqual([]);
  });

  it('repairs a state the full-lock install leaves behind: two records for a package', async () => {
    const h = macHome();
    const meta = path.join(h.prefix, 'conda-meta');
    fs.writeFileSync(path.join(meta, `${stemsOf(NEW_LOCK).find((s) => s.startsWith('orix-0.'))}.json`), '{}');
    expect(metaFiles(h).filter((f) => /^orix-\d/.test(f))).toHaveLength(2);
    const fake = fakeMicromamba(h);
    const { result, log } = runMac(h, fake);
    expect(await result).toMatchObject({ action: 'synced' });
    expect(metaFiles(h).filter((f) => /^orix-\d/.test(f))).toHaveLength(1);
    expect(log.join('\n')).toMatch(/duplicate records on disk for: orix/);
  });
});

describe('when the network is down', () => {
  const offline = (n) => reply({ code: 1, output: CONNECT_TEXT, stderr: CONNECT_TEXT });

  it('the environment is byte for byte as it was, no marker, one failure record, one dialog', async () => {
    const h = macHome();
    const before = metaFiles(h);
    const fake = fakeMicromamba(h, { download: offline });
    const { result, notify } = runMac(h, fake);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'network' });
    expect(metaFiles(h)).toEqual(before);
    expect(exists(h, '.packages_sync.json')).toBe(false);
    expect(exists(h, '.packages_lock.json')).toBe(false);
    expect(fake.kinds()).not.toContain('remove');
    expect(fake.kinds()).not.toContain('install');
    expect(readJson(h, '.packages_sync_failed.json')).toMatchObject({
      lockSha256: sync.lockDigest(NEW_LOCK), reason: 'network', count: 1, shown: true,
    });
    expect(notify).toHaveBeenCalledTimes(1);
    expect(notify.mock.calls[0][0].reason).toBe('network');
    expect(fs.existsSync(path.join(h.home, 'setup-tmp', mac.DELTA_LOCK_FILE))).toBe(false);
  });

  it('the next start asks again but does not show the dialog again; three failures in a row back off for a day', async () => {
    const h = macHome();
    const first = runMac(h, fakeMicromamba(h, { download: offline }));
    await first.result;
    const second = runMac(h, fakeMicromamba(h, { download: offline }), { now: () => first.clock.t + 60_000 });
    await second.result;
    expect(second.notify).not.toHaveBeenCalled();
    expect(readJson(h, '.packages_sync_failed.json').count).toBe(2);
    const third = runMac(h, fakeMicromamba(h, { download: offline }), { now: () => first.clock.t + 120_000 });
    await third.result;
    expect(readJson(h, '.packages_sync_failed.json').count).toBe(3);
    const fourth = fakeMicromamba(h, { download: offline });
    const res = await runMac(h, fourth, { now: () => first.clock.t + 180_000 }).result;
    expect(res).toMatchObject({ action: 'skip' });
    expect(fourth.calls).toEqual([]);
  });

  it('a later start with a network goes through and clears the failure', async () => {
    const h = macHome();
    await runMac(h, fakeMicromamba(h, { download: offline })).result;
    expect(exists(h, '.packages_sync_failed.json')).toBe(true);
    const res = await runMac(h, fakeMicromamba(h)).result;
    expect(res).toMatchObject({ action: 'synced' });
    expect(exists(h, '.packages_sync_failed.json')).toBe(false);
  });

  it('a download that micromamba refuses for its checksum is a failure of its own kind, and changes nothing', async () => {
    const h = macHome();
    const before = metaFiles(h);
    const text = 'error libmamba File not valid: SHA256 doesn\'t match expectation\ncritical libmamba Found incorrect downloads. Aborting';
    const res = await runMac(h, fakeMicromamba(h, { download: () => reply({ code: 1, output: text, stderr: text }) })).result;
    expect(res).toMatchObject({ action: 'failed', reason: 'checksum' });
    expect(metaFiles(h)).toEqual(before);
    expect(exists(h, '.packages_sync.json')).toBe(false);
  });

  it('a download that never finishes is a network failure', async () => {
    const h = macHome();
    const res = await runMac(h, fakeMicromamba(h, { download: () => reply({ code: -1, timedOut: true }) })).result;
    expect(res).toMatchObject({ action: 'failed', reason: 'network' });
    expect(res.detail).toMatch(/did not finish downloading within 10 min/);
  });
});

describe('what stops it before anything is touched', () => {
  async function untouched(h, fake, over, expected) {
    const before = metaFiles(h);
    const res = await runMac(h, fake, over).result;
    expect(res).toMatchObject(expected);
    expect(metaFiles(h)).toEqual(before);
    expect(exists(h, '.packages_sync.json')).toBe(false);
    expect(fake.kinds()).not.toContain('download');
    return res;
  }

  it('a delta that names python is refused by the guard, with no process run at all', async () => {
    const lock = NEW_LOCK;
    const pyStem = stemsOf(lock).find((s) => /^python-3/.test(s));
    const h = macHome();
    fs.rmSync(path.join(h.prefix, 'conda-meta', `${pyStem}.json`));
    fs.writeFileSync(path.join(h.prefix, 'conda-meta', 'python-3.10.1-h_0_cpython.json'), '{}');
    const fake = fakeMicromamba(h);
    const res = await untouched(h, fake, {}, { action: 'failed', reason: 'guard' });
    expect(res.detail).toMatch(/python/);
    expect(fake.calls).toEqual([]);
  });

  it('a guard refusal is not retried at every start: it waits for another lock', async () => {
    const h = macHome();
    const pyStem = stemsOf(NEW_LOCK).find((s) => /^python-3/.test(s));
    fs.rmSync(path.join(h.prefix, 'conda-meta', `${pyStem}.json`));
    await runMac(h, fakeMicromamba(h)).result;
    const res = await runMac(h, fakeMicromamba(h)).result;
    expect(res).toMatchObject({ action: 'skip' });
    expect(res.reason).toMatch(/already refused/);
  });

  it('too little disk space', async () => {
    const h = macHome();
    await untouched(h, fakeMicromamba(h), { freeBytes: () => ({ bytes: 100 * 1024 * 1024, known: true }) },
      { action: 'failed', reason: 'space' });
  });

  it('micromamba missing from the home', async () => {
    const h = macHome();
    fs.rmSync(h.mm);
    const res = await untouched(h, fakeMicromamba(h), {}, { action: 'failed', reason: 'other' });
    expect(res.detail).toMatch(/micromamba is missing/);
  });

  it('a micromamba for another architecture (Rosetta) would update arm64 with x86_64 packages', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h, { info: () => reply({ stdout: '    platform : osx-64\n' }) });
    const res = await untouched(h, fake, {}, { action: 'failed', reason: 'other' });
    expect(res.detail).toMatch(/osx-64.*Rosetta/);
  });

  it('a Mac that cannot sign what micromamba relinks, found before anything is removed', async () => {
    const h = macHome();
    const res = await untouched(h, fakeMicromamba(h), {
      macos: { codesign: async () => { throw new Error('Operation not permitted'); } },
    }, { action: 'failed', reason: 'signing' });
    expect(res.detail).toMatch(/cannot sign.*Operation not permitted/);
  });

  it('runs the signing probe (a function the wizard also uses) before the download', async () => {
    const h = macHome();
    const order = [];
    const fake = fakeMicromamba(h, { download: () => { order.push('download'); return undefined; } });
    await runMac(h, fake, { macos: { codesign: async () => { order.push('codesign'); } } }).result;
    expect(order).toEqual(['codesign', 'download']);
  });

  it('a lock that is not a lock (truncated, or another platform) is left alone and logged', async () => {
    const h = macHome({ lock: 'version: 1\n' });
    const fake = fakeMicromamba(h);
    const { result, log } = runMac(h, fake);
    expect(await result).toMatchObject({ action: 'skip', reason: 'lock-unsafe' });
    expect(fake.calls).toEqual([]);
    expect(log.join('\n')).toMatch(/truncated|no `package:`|has no/);
  });

  it('a lock that parses but is far too small to be an environment is not acted on', async () => {
    const parsed = mac.parseLock(NEW_LOCK);
    const small = parsed.header + parsed.packages.slice(0, 50).map((p) => p.text).join('');
    const h = macHome({ lock: small });
    const fake = fakeMicromamba(h);
    const { result, log } = runMac(h, fake);
    expect(await result).toMatchObject({ action: 'skip', reason: 'lock-unsafe' });
    expect(fake.calls).toEqual([]);
    expect(exists(h, '.packages_lock.json')).toBe(false);
    expect(log.join('\n')).toMatch(/only 50 package/);
  });

  it('a directory that is no conda environment is left alone', async () => {
    const h = macHome();
    fs.rmSync(path.join(h.prefix, 'conda-meta'), { recursive: true });
    const fake = fakeMicromamba(h);
    expect(await runMac(h, fake).result).toMatchObject({ action: 'skip', reason: 'not-a-conda-prefix' });
    expect(fake.calls).toEqual([]);
  });

  it('no .install_mode: it will not guess', async () => {
    const h = macHome({ mode: null });
    expect(await runMac(h, fakeMicromamba(h)).result).toMatchObject({ action: 'skip', reason: 'no-install-mode' });
  });
});

describe('when the local phase goes wrong', () => {
  it('a verification that fails is followed by one more attempt, and then it is recorded as a repair', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h, { imports: (n) => (n === 1 ? reply({ code: 1, output: 'ImportError: x' }) : undefined) });
    const { result, log } = runMac(h, fake);
    const res = await result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(fake.counts.remove).toBe(2);
    expect(fake.counts.install).toBe(2);
    expect(log.join('\n')).toMatch(/not right yet \(the imports failed/);
    expect(readJson(h, '.packages_lock.json').how).toBe('repair');
    expect(exists(h, '.packages_sync.json')).toBe(false);
    // The second attempt removed what the first had linked: no second copy.
    expect(metaFiles(h)).toEqual(stemsOf(NEW_LOCK).map((s) => `${s}.json`).sort());
  });

  it('a second record that appears is caught by the verification and removed by the second attempt', async () => {
    const h = macHome();
    let planted = false;
    const fake = fakeMicromamba(h, {
      install: (n, ctx) => {
        if (n !== 1) return undefined;
        // The first link also leaves the old orix behind (as the full-lock install does).
        fs.writeFileSync(path.join(h.prefix, 'conda-meta', 'orix-0.14.1-pyhd8ed1ab_0.json'), '{}');
        planted = true;
        return undefined;
      },
    });
    const res = await runMac(h, fake).result;
    expect(planted).toBe(true);
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(metaFiles(h).filter((f) => /^orix-\d/.test(f))).toHaveLength(1);
  });

  it('a verification that fails twice sends the user to the repair wizard, and the marker stays', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h, { imports: () => reply({ code: 1, output: 'ImportError: x' }) });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ ok: false, repair: true });
    expect(res.message).toMatch(/imports failed/);
    expect(exists(h, '.packages_sync.json')).toBe(true);
    expect(exists(h, '.packages_lock.json')).toBe(false);
    expect(fake.counts.install).toBe(2);
  });

  it('a failing OpenMP gate fails the verification', async () => {
    const h = macHome();
    const text = 'FAIL  openmp files  2 OpenMP runtimes';
    const fake = fakeMicromamba(h, { health: () => reply({ code: 1, output: text, stdout: text }) });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ repair: true });
    expect(res.message).toMatch(/OpenMP check failed.*2 OpenMP runtimes/);
  });

  it('a gate script that is not there is logged and skipped, and so is one configured as null', async () => {
    const h = macHome();
    fs.rmSync(path.join(h.home, 'runtime', 'scripts', 'check_runtime_health.py'));
    const a = runMac(h, fakeMicromamba(h));
    expect(await a.result).toMatchObject({ action: 'synced' });
    expect(a.log.join('\n')).toMatch(/OpenMP check skipped \(.*is not there\)/);

    const h2 = macHome({ installed: OLD_LOCK });
    const fake = fakeMicromamba(h2);
    fs.rmSync(path.join(h2.home, '.packages_lock.json'), { force: true });
    const b = runMac(h2, fake, { macos: { codesign: false, healthScript: null } });
    expect(await b.result).toMatchObject({ action: 'synced' });
    expect(fake.kinds()).not.toContain('health');
    expect(b.log.join('\n')).toMatch(/OpenMP check skipped \(no script configured\)/);
  });

  it('a second record that micromamba list does not show is still caught (measured: list reports ONE entry for two records)', async () => {
    const h = macHome();
    const lockVersions = (() => {
      const m = new Map();
      for (const stem of stemsOf(NEW_LOCK)) m.set(nameOfStem(stem), [versionOfStem(stem)]);
      return m;
    })();
    const fake = fakeMicromamba(h, {
      install: (n) => {
        if (n === 1) fs.writeFileSync(path.join(h.prefix, 'conda-meta', 'orix-0.14.1-pyhd8ed1ab_0.json'), '{}');
        return undefined;
      },
      list: () => reply({
        stdout: JSON.stringify({ packages: [...lockVersions].map(([name, [version]]) => ({ name, version })) }),
      }),
    });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(fake.counts.install).toBe(2);
    expect(metaFiles(h).filter((f) => /^orix-\d/.test(f))).toHaveLength(1);
  });

  it('a version that micromamba list does not show as the lock\'s fails the verification even if conda-meta looks right', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h, {
      list: () => reply({ stdout: JSON.stringify({ packages: [{ name: 'orix', version: '0.14.1' }] }) }),
    });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ repair: true });
    expect(res.message).toMatch(/orix: micromamba list shows 0\.14\.1/);
  });

  it('a remove that fails is tried once more, and a second failure is a repair', async () => {
    const h = macHome();
    const failing = () => reply({ code: 1, output: 'critical libmamba could not unlink' });
    const fake = fakeMicromamba(h, { remove: failing });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ repair: true });
    expect(res.message).toMatch(/remove failed/);
    expect(fake.counts.remove).toBe(2);
    expect(fake.counts.install).toBeUndefined();
    expect(exists(h, '.packages_sync.json')).toBe(true);
  });

  it('an install that fails once and then works is a repair, not a failure', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h, { install: (n) => (n === 1 ? reply({ code: 1, output: 'critical libmamba Could not codesign executable' }) : undefined) });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
  });

  it('network words from a failed LOCAL phase are never a network failure: the marker stays, the wizard is shown', async () => {
    // The download is the only network phase. A relink that fails is a half-changed
    // environment whatever it printed, so it is never answered by deleting the marker.
    const h = macHome();
    const offlineLooking = () => reply({ code: 1, output: CONNECT_TEXT, stderr: CONNECT_TEXT });
    const fake = fakeMicromamba(h, { install: offlineLooking });
    const { result, notify } = runMac(h, fake);
    const res = await result;
    expect(res).toMatchObject({ ok: false, repair: true });
    expect(exists(h, '.packages_sync.json')).toBe(true);
    expect(exists(h, '.packages_sync_failed.json')).toBe(false);
    expect(notify).not.toHaveBeenCalled();
    expect(fake.counts.install).toBe(2);
  });

  it('imports that do not answer in time: start, marker kept, told once as unverified, no second attempt', async () => {
    const h = macHome();
    const fake = fakeMicromamba(h, { imports: () => reply({ code: -1, timedOut: true }) });
    const { result, notify } = runMac(h, fake);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', unverified: true });
    expect(res.repair).toBeUndefined();
    expect(fake.counts.install).toBe(1);
    expect(fake.counts.remove).toBe(1);
    expect(exists(h, '.packages_sync.json')).toBe(true);
    expect(exists(h, '.packages_lock.json')).toBe(false);
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ unverified: true }));
  });
});

describe('an update that was interrupted', () => {
  it('a marker with nothing left to do and imports that do not answer in time is unverified, not broken', async () => {
    const h = macHome({ installed: NEW_LOCK, files: { '.packages_sync.json': { schema: 1 } } });
    const res = await runMac(h, fakeMicromamba(h, { imports: () => reply({ code: -1, timedOut: true }) })).result;
    expect(res).toMatchObject({ ok: true, action: 'failed', unverified: true });
    expect(exists(h, '.packages_sync.json')).toBe(true);
  });

  it('without a network and imports that do not answer in time: starts, unverified', async () => {
    const h = macHome({ files: { '.packages_sync.json': { schema: 1 } } });
    const fake = fakeMicromamba(h, {
      download: () => reply({ code: 1, output: CONNECT_TEXT, stderr: CONNECT_TEXT }),
      imports: () => reply({ code: -1, timedOut: true }),
    });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'network', unverified: true });
  });

  it('is finished from the disk: what was unlinked is linked again, and the marker goes', async () => {
    const h = macHome({ files: { '.packages_sync.json': { schema: 1, to: { kikuchipy: '0.13.1' } } } });
    // The kill came after `remove` and before `install`.
    for (const f of metaFiles(h)) {
      if (/^(kikuchipy|orix|openssl)(-base)?-\d/.test(f)) fs.rmSync(path.join(h.prefix, 'conda-meta', f));
    }
    const fake = fakeMicromamba(h);
    const { result, log } = runMac(h, fake);
    const res = await result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(log.join('\n')).toMatch(/repair — a previous update/);
    expect(metaFiles(h)).toEqual(stemsOf(NEW_LOCK).map((s) => `${s}.json`).sort());
    expect(exists(h, '.packages_sync.json')).toBe(false);
  });

  it('works without a network when the packages are still in the cache', async () => {
    // The download step finds them there: micromamba succeeds offline.
    const h = macHome({ files: { '.packages_sync.json': { schema: 1 } } });
    const res = await runMac(h, fakeMicromamba(h)).result;
    expect(res).toMatchObject({ how: 'repair' });
  });

  it('without a network and without the cache: starts if the libraries import, and stays unverified', async () => {
    const h = macHome({ files: { '.packages_sync.json': { schema: 1 } } });
    const offline = () => reply({ code: 1, output: CONNECT_TEXT, stderr: CONNECT_TEXT });
    const res = await runMac(h, fakeMicromamba(h, { download: offline })).result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'network' });
    expect(res.detail).toMatch(/still unverified/);
    expect(exists(h, '.packages_sync.json')).toBe(true);
  });

  it('without a network and with libraries that do not import: the repair wizard', async () => {
    const h = macHome({ files: { '.packages_sync.json': { schema: 1 } } });
    const fake = fakeMicromamba(h, {
      download: () => reply({ code: 1, output: CONNECT_TEXT, stderr: CONNECT_TEXT }),
      imports: () => reply({ code: 1, output: 'ImportError' }),
    });
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ ok: false, repair: true });
    expect(exists(h, '.packages_sync.json')).toBe(true);
  });

  it('a marker with nothing left to do is only verified', async () => {
    const h = macHome({ installed: NEW_LOCK, files: { '.packages_sync.json': { schema: 1 } } });
    const fake = fakeMicromamba(h);
    const res = await runMac(h, fake).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(fake.kinds()).toEqual(['list', 'imports', 'health']);
    expect(exists(h, '.packages_sync.json')).toBe(false);
  });

  it('a marker with nothing left to do and a broken environment is a repair', async () => {
    const h = macHome({ installed: NEW_LOCK, files: { '.packages_sync.json': { schema: 1 } } });
    const res = await runMac(h, fakeMicromamba(h, { imports: () => reply({ code: 1, output: 'ImportError' }) })).result;
    expect(res).toMatchObject({ repair: true });
  });
});

describe('closing the window', () => {
  it('before the download ends: nothing was changed and no marker was written', async () => {
    const h = macHome();
    const before = metaFiles(h);
    let cancelled = false;
    const fake = fakeMicromamba(h, { download: () => { cancelled = true; return reply({ code: -1, cancelled: true }); } });
    const res = await runMac(h, fake, { isCancelled: () => cancelled }).result;
    expect(res).toEqual({ cancelled: true });
    expect(metaFiles(h)).toEqual(before);
    expect(exists(h, '.packages_sync.json')).toBe(false);
  });

  it('during the local phase: the marker stays, so the next start finishes it', async () => {
    const h = macHome();
    let cancelled = false;
    const fake = fakeMicromamba(h, { remove: () => { cancelled = true; return reply({ code: -1, cancelled: true }); } });
    const res = await runMac(h, fake, { isCancelled: () => cancelled }).result;
    expect(res).toEqual({ cancelled: true });
    expect(exists(h, '.packages_sync.json')).toBe(true);
    expect(exists(h, '.packages_lock.json')).toBe(false);
  });
});

// --------------------------------------------------------------------------
// the real runner, with the per-call environment micromamba needs
// --------------------------------------------------------------------------

describe('the runner gives a call its own environment', () => {
  it('a per-call env replaces the runner\'s, and without one nothing changes', async () => {
    const runner = sync.createRunner();
    const script = 'process.stdout.write(String(process.env.ORIENTA_SYNC_PROBE))';
    const plain = await runner.run(process.execPath, ['-e', script]);
    expect(plain.stdout).toBe('undefined');
    const scoped = await runner.run(process.execPath, ['-e', script], {
      env: { ...process.env, ORIENTA_SYNC_PROBE: 'scoped' },
    });
    expect(scoped.stdout).toBe('scoped');
    const after = await runner.run(process.execPath, ['-e', script]);
    expect(after.stdout).toBe('undefined');
  });
});

// --------------------------------------------------------------------------
// the source: the properties no behavioural test above can pin
// --------------------------------------------------------------------------

describe('the source', () => {
  it('never runs micromamba clean (the wizard\'s cleanArgs is rejected by micromamba 2.9.0 and clean -a sweeps other caches)', () => {
    const code = MAC_SRC.split('\n').filter((l) => !/^\s*(\*|\/\/)/.test(l)).join('\n');
    expect(code).not.toMatch(/cleanArgs|'clean'/);
  });

  it('never installs a full lock into the existing prefix: no `create`, and the only -f is the delta lock', () => {
    const code = MAC_SRC.split('\n').filter((l) => !/^\s*(\*|\/\/)/.test(l)).join('\n');
    expect(code).not.toMatch(/createArgs|'create'/);
    expect(code).not.toMatch(/lockFile:\s*ctx\.lockPath/);
  });

  it('does not import Electron', () => {
    expect(MAC_SRC).not.toMatch(/require\(['"]electron['"]\)/);
  });
});
