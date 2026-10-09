/**
 * Does an update bring the Python packages to the lock it shipped?
 *
 * Until 0.4.7 nothing did. Installing a newer .exe over an old installation
 * replaced the shell, `bundled_update.js` brought `runtime/` (and the lock
 * files in it) to the new release, and `python/` stayed exactly as the wizard
 * left it: kikuchipy 0.11.3 under a release whose lock names 0.13.1.
 *
 * The orchestration is tested with an injected `run` that plays pip -- the
 * real thing is exercised by `tests/test_package_sync_seam.py`, which needs a
 * network and a throwaway interpreter. What is pinned here is every DECISION
 * (the table), every ARGUMENT list, the classification of pip's own words,
 * the order of the steps and, by reading `main.js`, that the shell calls all
 * of it, in the right place, and ends the children when the window closes.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const requireCjs = createRequire(import.meta.url);
const sync = requireCjs('../../electron/setup/package_sync.js');
const installer = requireCjs('../../electron/setup/installer.js');
const removeData = requireCjs('../../electron/remove_data.js');
const { STRINGS, t } = requireCjs('../../electron/strings.js');
const { createWaitingPage, PHASES } = requireCjs('../../electron/waiting_page.js');

const REPO = path.resolve(import.meta.dirname, '..', '..');
const MAIN = fs.readFileSync(path.join(REPO, 'electron', 'main.js'), 'utf8');
const INSTALLER_SRC = fs.readFileSync(path.join(REPO, 'electron', 'setup', 'installer.js'), 'utf8');
const NSH = fs.readFileSync(path.join(REPO, 'electron', 'nsis', 'uninstall.nsh'), 'utf8');

const REPORT = JSON.parse(fs.readFileSync(
  path.join(import.meta.dirname, 'fixtures', 'pip_report_delta.json'), 'utf8'));

const LOCK = [
  '# GENERATED -- a test lock shaped like requirements-lock-cpu.txt',
  'kikuchipy==0.13.1',
  'orix==0.15.0',
  'pyebsdindex==0.3.10.1',
  'threadpoolctl==3.7.0',
  'numpy==2.3.5',
  '--extra-index-url https://download.pytorch.org/whl/cpu',
  'torch==2.11.0+cpu ; platform_system != "Darwin"',
  'torch==2.11.0 ; platform_system == "Darwin"',
  '',
].join('\n');

// The lock file the INSTALLER would read on this host (macOS reads another one).
const hostLock = (mode) => (process.platform === 'darwin'
  ? 'orienta-macos-lock.yml' : installer.lockFileFor(mode, process.platform));

const OLD = { kikuchipy: '0.11.3', orix: '0.14.1', pyebsdindex: '0.3.9.1', threadpoolctl: '3.7.0' };
const NEW = { kikuchipy: '0.13.1', orix: '0.15.0', pyebsdindex: '0.3.10.1', threadpoolctl: '3.7.0' };

let tmp;
beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-pkgsync-')); });
afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

// --------------------------------------------------------------------------
// a home on disk, and a pip that is a script
// --------------------------------------------------------------------------

function makeHome({ mode = 'cpu', lock = LOCK, files = {} } = {}) {
  const home = path.join(tmp, 'home');
  fs.mkdirSync(path.join(home, 'runtime'), { recursive: true });
  fs.mkdirSync(path.join(home, 'python'), { recursive: true });
  const python = path.join(home, 'python', process.platform === 'win32' ? 'python.exe' : 'python');
  fs.writeFileSync(python, '');
  if (mode !== null) fs.writeFileSync(path.join(home, '.install_mode'), `${mode}\n`);
  fs.writeFileSync(path.join(home, 'runtime', installer.lockFileFor(mode || 'cpu', 'win32')), lock);
  for (const [name, body] of Object.entries(files)) {
    fs.writeFileSync(path.join(home, name), typeof body === 'string' ? body : JSON.stringify(body));
  }
  return { home, python };
}

const readJson = (home, name) => JSON.parse(fs.readFileSync(path.join(home, name), 'utf8'));
const exists = (home, name) => fs.existsSync(path.join(home, name));

const OFFLINE_TEXT = [
  "WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after "
  + "connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPConnection object "
  + "at 0x000001F80909E810>: Failed to establish a new connection: [WinError 10061] Es konnte keine "
  + "Verbindung hergestellt werden, da der Zielcomputer die Verbindung verweigerte')': /simple/kikuchipy/",
  'ERROR: Could not find a version that satisfies the requirement kikuchipy==0.13.1 (from versions: none)',
  'ERROR: No matching distribution found for kikuchipy==0.13.1',
].join('\n');

// What pip prints when a transient connection warning comes first and a file
// cannot be replaced afterwards: it READS as a network failure and IS a failure
// while writing.
const WRITE_PHASE_TEXT = [
  'WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after '
  + "connection broken by 'ReadTimeoutError(\"HTTPSConnectionPool(host='files.pythonhosted.org', port=443): "
  + "Read timed out.\")': /packages/orix-0.15.0-py3-none-any.whl",
  'Installing collected packages: pyebsdindex, orix, kikuchipy',
  '  Attempting uninstall: orix',
  '    Found existing installation: orix 0.14.1',
  '    Uninstalling orix-0.14.1:',
  '      Successfully uninstalled orix-0.14.1',
  "ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: 'C:\\x\\orix'",
].join('\n');

/**
 * A pip that is a function. `installed` is the environment; `lock` what the lock
 * wants; `calls` every invocation in order. Scenario switches break one thing at
 * a time.
 */
function fakePip({ installed = { ...OLD }, want = NEW, scenario = {} } = {}) {
  const calls = [];
  const state = { installed: { ...installed }, scenario: { ...scenario } };
  const reply = (over = {}) => ({
    code: 0, signal: null, timedOut: false, cancelled: false,
    stdout: '', stderr: '', output: '', error: null, ...over,
  });

  async function run(exe, args, opts = {}) {
    const kind = classify(args);
    calls.push({ exe, args, kind, opts });
    const sc = state.scenario;
    if (sc.throwOn === kind) throw new Error(`boom in ${kind}`);
    if (sc.cancelOn === kind) return reply({ code: -1, cancelled: true });
    switch (kind) {
      case 'dry': {
        if (sc.dry === 'offline') return reply({ code: 1, output: OFFLINE_TEXT, stderr: OFFLINE_TEXT });
        if (sc.dry === 'timeout') return reply({ code: -1, timedOut: true });
        if (sc.dry === 'garbage') {
          fs.writeFileSync(args[args.indexOf('--report') + 1], '{"half": ');
          return reply();
        }
        const names = Object.keys(want).filter((n) => state.installed[n] !== want[n]);
        const report = JSON.parse(JSON.stringify(REPORT));
        report.install = REPORT.install.filter((e) => names.includes(e.metadata.name.toLowerCase()));
        if (sc.extraInDelta) report.install.push(sc.extraInDelta);
        fs.writeFileSync(args[args.indexOf('--report') + 1], JSON.stringify(report));
        return reply();
      }
      case 'versions': {
        if (sc.versions === 'broken') return reply({ code: 1, output: 'Traceback' });
        const out = {};
        for (const n of args.slice(2)) out[n] = state.installed[n] ?? null;
        return reply({ stdout: `${JSON.stringify(out)}\n` });
      }
      case 'imports':
        if (sc.imports === 'timeout') return reply({ code: -1, timedOut: true });
        return sc.imports === 'fail'
          ? reply({ code: 1, output: 'ImportError: cannot import name x from kikuchipy' })
          : reply();
      case 'check':
        return reply({ code: sc.check ?? 0, output: 'No broken requirements found.' });
      case 'reinstall': {
        if (sc.reinstall === 'offline') return reply({ code: 1, output: OFFLINE_TEXT, stderr: OFFLINE_TEXT });
        if (sc.reinstall === 'fail') return reply({ code: 1, output: 'ERROR: something else' });
        for (const spec of args.filter((a) => /^[A-Za-z0-9._-]+==/.test(a))) {
          const [n, v] = spec.split('==');
          state.installed[n] = v;
        }
        return reply();
      }
      case 'install': {
        if (sc.install === 'offline') return reply({ code: 1, output: OFFLINE_TEXT, stderr: OFFLINE_TEXT });
        if (sc.install === 'space') {
          state.installed.orix = want.orix;     // half-way: one package replaced
          const text = 'ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device';
          return reply({ code: 1, output: text, stderr: text });
        }
        if (sc.install === 'space-untouched') {
          const text = 'ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device';
          return reply({ code: 1, output: text, stderr: text });
        }
        // pip failed after it had started to write: one package is already new
        if (sc.install === 'write-failure' || sc.install === 'write-failure-nothing-changed') {
          if (sc.install === 'write-failure') state.installed.orix = want.orix;
          return reply({ code: 1, output: WRITE_PHASE_TEXT, stderr: WRITE_PHASE_TEXT });
        }
        // the 10-minute kill, in three different places
        if (sc.install === 'timeout-writing') {
          state.installed.orix = want.orix;
          const text = 'Installing collected packages: pyebsdindex, orix, kikuchipy';
          return reply({ code: -1, timedOut: true, output: text, stdout: text });
        }
        if (sc.install === 'timeout-changed') {   // the text is not what tells here, the environment is
          state.installed.orix = want.orix;
          const text = 'Downloading kikuchipy-0.13.1';
          return reply({ code: -1, timedOut: true, output: text, stdout: text });
        }
        if (sc.install === 'timeout-downloading') {
          const text = 'Collecting orix==0.15.0\n  Downloading orix-0.15.0-py3-none-any.whl';
          return reply({ code: -1, timedOut: true, output: text, stdout: text });
        }
        Object.assign(state.installed, want);
        if (sc.afterInstall) sc.afterInstall(state.installed);
        return reply();
      }
      default:
        throw new Error(`the test pip does not know ${args.join(' ')}`);
    }
  }

  function classify(args) {
    if (args.includes('--dry-run')) return 'dry';
    if (args[0] === '-c' && args[1].includes('metadata.version')) return 'versions';
    if (args[0] === '-c' && args[1].startsWith('import kikuchipy')) return 'imports';
    if (args.includes('check')) return 'check';
    if (args.includes('--force-reinstall')) return 'reinstall';
    if (args.includes('install')) return 'install';
    return 'other';
  }

  return { run, calls, state, kinds: () => calls.map((c) => c.kind) };
}

function makeDeps(pip, over = {}) {
  const clock = { t: Date.parse('2026-10-09T12:00:00Z') };
  const log = [];
  return {
    clock, log,
    notify: vi.fn(async () => {}),
    onPhase: vi.fn(),
    deps: {
      run: pip.run,
      isCancelled: () => false,
      log: (line) => log.push(line),
      freeBytes: () => ({ bytes: 10 * 1024 ** 3, known: true }),
      now: () => clock.t,
      runtimeTag: () => 'v0.4.7',
      ...over,
    },
  };
}

function run(ctx, { home, python }, pip, over = {}) {
  const made = makeDeps(pip, over);
  const deps = { ...made.deps, notify: made.notify, onPhase: made.onPhase, ...over };
  const opts = {
    home, python, decisionMode: 'run', projectRootEnv: undefined,
    platformName: 'win32', env: {}, ...ctx,
  };
  return { made, result: sync.syncPackages(opts, deps), deps };
}

// --------------------------------------------------------------------------
// digest, lock, small helpers
// --------------------------------------------------------------------------

describe('the digest', () => {
  it('is a SHA-256 of the LF-normalised text: a line ending is not another lock', () => {
    const lf = 'a==1\nb==2\n';
    expect(sync.lockDigest(lf)).toMatch(/^[0-9a-f]{64}$/);
    expect(sync.lockDigest(lf.replace(/\n/g, '\r\n'))).toBe(sync.lockDigest(lf));
  });

  it('changes with the content', () => {
    expect(sync.lockDigest('a==1\n')).not.toBe(sync.lockDigest('a==2\n'));
  });
});

describe('what the lock pins', () => {
  it('reads name==version, drops comments, extras and options, canonicalises names', () => {
    const v = sync.parseLockVersions([
      '# comment', '--extra-index-url https://x/', 'uvicorn[standard]==0.42.0',
      'Diffpy.Structure==3.3.1  # pinned', '', 'not a requirement',
    ].join('\n'));
    expect(v.get('uvicorn')).toBe('0.42.0');
    expect(v.get('diffpy-structure')).toBe('3.3.1');
    expect(v.size).toBe(2);
  });

  it('evaluates the platform_system markers the CPU lock uses', () => {
    expect(sync.parseLockVersions(LOCK, 'Windows').get('torch')).toBe('2.11.0+cpu');
    expect(sync.parseLockVersions(LOCK, 'Linux').get('torch')).toBe('2.11.0+cpu');
    expect(sync.parseLockVersions(LOCK, 'Darwin').get('torch')).toBe('2.11.0');
  });
});

describe('the small answers', () => {
  it('the opt-out is a deliberate yes', () => {
    for (const on of ['1', 'true', 'TRUE', 'yes', ' on ']) {
      expect(sync.optedOut({ [sync.SKIP_ENV]: on })).toBe(true);
    }
    for (const off of ['', '0', 'false', 'no', undefined]) {
      expect(sync.optedOut({ [sync.SKIP_ENV]: off })).toBe(false);
    }
    expect(sync.optedOut({})).toBe(false);
  });

  it('mode: gpu is gpu, anything else is cpu, nothing is nothing', () => {
    expect(sync.modeFrom('gpu\n')).toBe('gpu');
    expect(sync.modeFrom('cpu')).toBe('cpu');
    expect(sync.modeFrom('something else')).toBe('cpu');
    expect(sync.modeFrom('')).toBe(null);
    expect(sync.modeFrom('  \n')).toBe(null);
    expect(sync.modeFrom(null)).toBe(null);
  });

  it('a path inside the home is inside; a sibling, a parent and the home itself are not', () => {
    const home = path.join(tmp, 'home');
    expect(sync.isInside(home, path.join(home, 'python', 'python.exe'))).toBe(true);
    expect(sync.isInside(home, path.join(tmp, 'home2', 'python.exe'))).toBe(false);
    expect(sync.isInside(home, tmp)).toBe(false);
    expect(sync.isInside(home, home)).toBe(false);
    expect(sync.isInside(home, '')).toBe(false);
  });

  it('a spec is name==version and nothing pip could read as an option or a URL', () => {
    expect(sync.specFor('kikuchipy', '0.13.1')).toBe('kikuchipy==0.13.1');
    expect(sync.specFor('torch', '2.11.0+cpu')).toBe('torch==2.11.0+cpu');
    expect(sync.specFor('--index-url', '1')).toBe(null);
    expect(sync.specFor('kikuchipy', '1; rm -rf')).toBe(null);
    expect(sync.specFor('a b', '1')).toBe(null);
    expect(sync.specFor('https://evil/x.whl', '1')).toBe(null);
  });
});

// --------------------------------------------------------------------------
// the decision table
// --------------------------------------------------------------------------

describe('the decision', () => {
  const DIGEST = sync.lockDigest(LOCK);
  const NAME = 'requirements-lock-cpu.txt';
  const goodRecord = { sha256: DIGEST, mode: 'cpu', lock: NAME };
  const now = Date.parse('2026-10-09T12:00:00Z');
  const base = { mode: 'cpu', lockDigest: DIGEST, lockName: NAME, now };
  const fail = (over) => ({
    lockSha256: DIGEST, reason: 'network', count: 1, at: new Date(now - 1000).toISOString(), ...over,
  });
  const action = (over) => sync.decide({ ...base, ...over }).action;

  it('opt-out wins over everything, including an interrupted update', () => {
    expect(action({ optedOut: true, record: null, marker: { x: 1 } })).toBe('skip');
  });

  it('without .install_mode it does not guess', () => {
    expect(action({ mode: null })).toBe('skip');
  });

  it('without a lock there is nothing to compare against', () => {
    expect(action({ lockDigest: null })).toBe('skip');
  });

  it('a record for this lock and this flavour is the fast path', () => {
    expect(action({ record: goodRecord })).toBe('fastpath');
  });

  it('a record for ANOTHER lock asks pip', () => {
    expect(action({ record: { ...goodRecord, sha256: 'f'.repeat(64) } })).toBe('dryrun');
  });

  it('a record for another flavour asks pip', () => {
    expect(action({ record: { ...goodRecord, mode: 'gpu' } })).toBe('dryrun');
  });

  it('a record for another lock FILE asks pip, even with the same digest', () => {
    expect(action({ record: { ...goodRecord, lock: 'requirements-lock-gpu.txt' } })).toBe('dryrun');
  });

  it('no record -- every 0.4.6 installation -- asks pip', () => {
    expect(action({ record: null })).toBe('dryrun');
  });

  it('an in-flight marker means repair, even when the record matches', () => {
    expect(action({ record: goodRecord, marker: { from: {}, to: {} } })).toBe('repair');
    expect(action({ record: null, marker: {} })).toBe('repair');
  });

  it('a refused lock is not asked again; a different lock is', () => {
    expect(action({ record: null, failure: fail({ reason: 'guard', count: 1 }) })).toBe('skip');
    expect(action({ record: null, failure: fail({ reason: 'guard', lockSha256: 'e'.repeat(64) }) })).toBe('dryrun');
  });

  it('network failures retry at every start until the third', () => {
    expect(action({ record: null, failure: fail({ count: 1 }) })).toBe('dryrun');
    expect(action({ record: null, failure: fail({ count: 2 }) })).toBe('dryrun');
  });

  it('after three network failures in a row, once per 24 hours', () => {
    expect(action({ record: null, failure: fail({ count: 3 }) })).toBe('skip');
    const justUnder = new Date(now - (24 * 3600 * 1000 - 60000)).toISOString();
    expect(action({ record: null, failure: fail({ count: 7, at: justUnder }) })).toBe('skip');
    const justOver = new Date(now - (24 * 3600 * 1000 + 60000)).toISOString();
    expect(action({ record: null, failure: fail({ count: 7, at: justOver }) })).toBe('dryrun');
  });

  it('the backoff is for network failures of THIS lock only', () => {
    expect(action({ record: null, failure: fail({ count: 9, lockSha256: 'e'.repeat(64) }) })).toBe('dryrun');
    expect(action({ record: null, failure: fail({ count: 9, reason: 'resolver' }) })).toBe('dryrun');
    expect(action({ record: null, failure: fail({ count: 9, reason: 'space' }) })).toBe('dryrun');
  });

  it('a marker is never held back by a backoff: an unverified environment is not left alone', () => {
    expect(action({ record: null, marker: {}, failure: fail({ count: 9 }) })).toBe('repair');
  });

  it('a failure record with an unreadable time does not freeze the update', () => {
    expect(action({ record: null, failure: fail({ count: 9, at: 'yesterday-ish' }) })).toBe('dryrun');
  });
});

describe('the conditions that are about the machine', () => {
  const home = path.join(tmp || os.tmpdir(), 'h');
  const ok = {
    decisionMode: 'run', projectRootEnv: undefined,
    python: path.join(home, 'python', 'python.exe'), home, installMode: 'cpu\n', optedOut: false,
  };
  it('are met by an ordinary installed run', () => {
    expect(sync.preconditions(ok).ok).toBe(true);
  });
  it.each([
    [{ optedOut: true }, 'opt-out'],
    [{ decisionMode: 'dev' }, 'not-run-mode'],
    [{ decisionMode: 'repair' }, 'not-run-mode'],
    [{ projectRootEnv: 'C:/checkout' }, 'project-root-override'],
    [{ python: path.join(os.tmpdir(), 'conda', 'python.exe') }, 'python-outside-home'],
    [{ installMode: null }, 'no-install-mode'],
    [{ installMode: '\n' }, 'no-install-mode'],
  ])('%j is refused as %s', (over, reason) => {
    const verdict = sync.preconditions({ ...ok, ...over });
    expect(verdict.ok).toBe(false);
    expect(verdict.reason).toBe(reason);
  });
});

// --------------------------------------------------------------------------
// the delta, from a real pip report
// --------------------------------------------------------------------------

describe('the delta from pip\'s own report', () => {
  it('names the four packages the 0.4.6 -> 0.4.7 lock moves, with version, URL and hash', () => {
    const delta = sync.deltaFromReport(REPORT);
    expect(delta.map((d) => `${d.name} ${d.version}`)).toEqual([
      'kikuchipy 0.13.1', 'orix 0.15.0', 'pyebsdindex 0.3.10.1', 'threadpoolctl 3.6.0',
    ]);
    for (const d of delta) {
      expect(d.url).toMatch(/^https:\/\/.+\.whl$/);
      expect(d.sha256).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it('an empty install list is an empty delta, not an error', () => {
    expect(sync.deltaFromReport({ version: '1', install: [] })).toEqual([]);
  });

  it.each([
    [null], [{}], [{ install: 'no' }], [{ install: [{}] }], [{ install: [{ metadata: { name: 'x' } }] }],
  ])('a report that is not one (%j) throws, so a torn file is never "nothing to do"', (bad) => {
    expect(() => sync.deltaFromReport(bad)).toThrow();
  });

  it('spells names the way pip canonicalises them', () => {
    const d = sync.deltaFromReport({
      install: [{ metadata: { name: 'Diffpy_Structure', version: '1' }, download_info: { url: 'u' } }],
    });
    expect(d[0].name).toBe('diffpy-structure');
  });
});

describe('the guard', () => {
  const pkg = (name) => ({ name, version: '1', url: null, sha256: null });

  it('lets the real 0.4.7 delta through', () => {
    expect(sync.guardDelta(sync.deltaFromReport(REPORT)).ok).toBe(true);
  });

  it.each([
    'torch', 'torchvision', 'numpy', 'scipy', 'cupy-cuda12x', 'cupy', 'nvidia-cudnn-cu12',
    'nvidia-cublas-cu12', 'triton', 'triton-windows',
  ])('refuses %s by name', (name) => {
    const verdict = sync.guardDelta([pkg('kikuchipy'), pkg(name)]);
    expect(verdict.ok).toBe(false);
    expect(verdict.reason).toBe('guard');
    expect(verdict.guarded).toEqual([name]);
  });

  it('has no size rule at run time: the release\'s size is held by test_lock_delta_budget.py, not by a request at every start', () => {
    // A size is learned from the network, at a start that has to work offline,
    // for a decision that was already made when the release was built.
    expect(sync.guardDelta([pkg('a'), pkg('b')], { a: 10 ** 12, b: 10 ** 12 }).ok).toBe(true);
    expect(sync.MAX_DELTA_BYTES).toBeUndefined();
    const source = fs.readFileSync(path.join(REPO, 'electron', 'setup', 'package_sync.js'), 'utf8');
    expect(source).not.toMatch(/headSize|unknownSize|size check/);
  });

  it('does not mistake a package that merely contains the word', () => {
    expect(sync.guardDelta([pkg('numpydoc'), pkg('scipy-stubs'), pkg('mytorch')]).ok).toBe(true);
  });
});

// --------------------------------------------------------------------------
// the arguments pip is given
// --------------------------------------------------------------------------

describe('the pip arguments', () => {
  const lock = path.join('x', 'runtime', 'requirements-lock-cpu.txt');
  const flag = (args, name) => args[args.indexOf(name) + 1];

  it('the question is a dry run with a report, a hard patience, and the wizard\'s index arguments', () => {
    const args = sync.dryRunArgs({ mode: 'cpu', lockPath: lock, reportPath: 'r.json' });
    expect(args.slice(0, 3)).toEqual(['-m', 'pip', 'install']);
    expect(args).toContain('--dry-run');
    expect(flag(args, '--report')).toBe('r.json');
    expect(flag(args, '--retries')).toBe('1');
    expect(flag(args, '--timeout')).toBe('8');
    expect(args).toContain('--quiet');
    expect(args).toContain('--disable-pip-version-check');
    expect(flag(args, '-r')).toBe(lock);
    // not a copy of the wizard's list but the wizard's list
    const index = installer.pipIndexArgs('cpu');
    const at = args.indexOf(index[0]);
    expect(args.slice(at, at + index.length)).toEqual(index);
  });

  it('the install is the wizard\'s own command: -r <lock> with pipIndexArgs(mode)', () => {
    const args = sync.installArgs({ mode: 'gpu', lockPath: lock });
    expect(args).not.toContain('--dry-run');
    expect(args).not.toContain('--force-reinstall');
    expect(flag(args, '-r')).toBe(lock);
    const index = installer.pipIndexArgs('gpu');
    const at = args.indexOf(index[0]);
    expect(args.slice(at, at + index.length)).toEqual(index);
    expect(args).toContain(index.find((a) => a.includes('cu126')));
  });

  it('the cpu flavour never offers the CUDA index: that is how the 8 GB wheel is fetched', () => {
    for (const args of [
      sync.dryRunArgs({ mode: 'cpu', lockPath: lock, reportPath: 'r' }),
      sync.installArgs({ mode: 'cpu', lockPath: lock }),
      sync.reinstallArgs({ mode: 'cpu', specs: ['a==1'] }),
    ]) {
      expect(args.join(' ')).not.toContain('cu126');
      expect(args).toContain('--only-binary');
    }
  });

  it('a reinstall replaces exactly the named packages and resolves nothing', () => {
    const args = sync.reinstallArgs({ mode: 'cpu', specs: ['orix==0.15.0', 'kikuchipy==0.13.1'] });
    expect(args).toContain('--force-reinstall');
    expect(args).toContain('--no-deps');
    expect(args.slice(-2)).toEqual(['orix==0.15.0', 'kikuchipy==0.13.1']);
    expect(args).not.toContain('-r');
  });

  it('the checks run in the interpreter, not through a shell', () => {
    expect(sync.versionsArgs(['a', 'b']).slice(2)).toEqual(['a', 'b']);
    expect(sync.versionsArgs(['a'])[0]).toBe('-c');
    expect(sync.importArgs()).toEqual(['-c', 'import kikuchipy, orix, pyebsdindex']);
    expect(sync.checkArgs().slice(0, 3)).toEqual(['-m', 'pip', 'check']);
  });
});

describe('what went wrong, from what pip printed', () => {
  const REFUSED = OFFLINE_TEXT;
  const DNS = [
    "WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after "
    + "connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object "
    + "at 0x000001F04613E650>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed')': "
    + '/simple/kikuchipy/',
    'ERROR: Could not find a version that satisfies the requirement kikuchipy==0.13.1 (from versions: none)',
    'ERROR: No matching distribution found for kikuchipy==0.13.1',
  ].join('\n');
  const NO_MATCH = [
    'ERROR: Could not find a version that satisfies the requirement no-such-package (from versions: none)',
    'ERROR: No matching distribution found for no-such-package',
  ].join('\n');
  const CONFLICT = [
    'ERROR: Cannot install requests==2.32.5 and urllib3==1.20 because these package versions have conflicting dependencies.',
    'ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/',
  ].join('\n');

  it.each([
    ['a refused connection (words captured from pip 26 on Windows, in German)', REFUSED, 'network'],
    ['a name that does not resolve', DNS, 'network'],
    ['a timeout of our own making', '', 'network', true],
    ['pip\'s read timeout', 'pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool: Read timed out.', 'network'],
    ['a proxy', 'ProxyError(\'Cannot connect to proxy.\')', 'network'],
    ['a bad certificate', 'SSLError(SSLCertVerificationError)', 'network'],
    ['a package that does not exist', NO_MATCH, 'resolver'],
    ['a conflict', CONFLICT, 'resolver'],
    ['a full disk (POSIX)', 'ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device', 'space'],
    ['a full disk (Windows)', 'OSError: [WinError 112] There is not enough space on the disk', 'space'],
    ['anything else', 'PermissionError: [WinError 5] Access is denied', 'other'],
    ['nothing at all', '', 'other'],
  ])('%s is %s', (_label, text, expected, timedOut = false) => {
    expect(sync.classifyPipFailure({ text, timedOut })).toBe(expected);
  });

  it('"No matching distribution" after an unreachable index is the network, not the resolver', () => {
    // pip ends with the resolver's sentence whether the index was down or the
    // pin was wrong; the lines BEFORE it are what tell them apart.
    expect(sync.classifyPipFailure({ text: REFUSED })).toBe('network');
    expect(sync.classifyPipFailure({ text: NO_MATCH })).toBe('resolver');
  });

  it('a full disk wins over the network words that surround it', () => {
    expect(sync.classifyPipFailure({ text: `${REFUSED}\n[Errno 28] No space left on device` })).toBe('space');
  });
});

// --------------------------------------------------------------------------
// the records
// --------------------------------------------------------------------------

describe('the records', () => {
  it('a first install writes what the next start compares against, and clears what an interrupted sync left', () => {
    const home = path.join(tmp, 'home');
    fs.mkdirSync(home, { recursive: true });
    fs.writeFileSync(path.join(home, sync.MARKER_FILE), '{}');
    fs.writeFileSync(path.join(home, sync.FAILURE_FILE), '{}');
    const rec = sync.recordFirstInstall({
      home, mode: 'gpu', lockName: 'requirements-lock-gpu.txt', lockText: LOCK.replace(/\n/g, '\r\n'),
      runtimeTag: 'v0.4.7', platformName: 'win32', now: Date.parse('2026-10-09T12:00:00Z'),
    });
    expect(readJson(home, sync.RECORD_FILE)).toEqual(rec);
    expect(rec).toMatchObject({
      schema: 1, lock: 'requirements-lock-gpu.txt', sha256: sync.lockDigest(LOCK), mode: 'gpu',
      platform: 'win32', runtimeTag: 'v0.4.7', syncedAt: '2026-10-09T12:00:00.000Z', how: 'install',
    });
    expect(exists(home, sync.MARKER_FILE)).toBe(false);
    expect(exists(home, sync.FAILURE_FILE)).toBe(false);
    expect(fs.readdirSync(home).filter((n) => n.endsWith('.tmp'))).toEqual([]);
  });

  it('reads the three files as the decision wants them: marker present-but-unreadable is still present', () => {
    const home = path.join(tmp, 'home');
    fs.mkdirSync(home, { recursive: true });
    expect(sync.readState(home)).toEqual({ record: null, marker: null, failure: null });
    fs.writeFileSync(path.join(home, sync.MARKER_FILE), '{"half": ');
    fs.writeFileSync(path.join(home, sync.RECORD_FILE), 'not json');
    const state = sync.readState(home);
    expect(state.marker).toEqual({});
    expect(state.record).toBe(null);
  });

  it('a failure counts consecutive failures of the same reason against the same lock', () => {
    const at = Date.parse('2026-10-09T12:00:00Z');
    const first = sync.nextFailure(null, { digest: 'a', reason: 'network', now: at });
    expect(first).toMatchObject({ count: 1, shown: false, reason: 'network', lockSha256: 'a' });
    const second = sync.nextFailure({ ...first, shown: true }, { digest: 'a', reason: 'network', now: at + 1 });
    expect(second).toMatchObject({ count: 2, shown: true });
    expect(sync.nextFailure(second, { digest: 'a', reason: 'space', now: at })).toMatchObject({ count: 1, shown: false });
    expect(sync.nextFailure(second, { digest: 'b', reason: 'network', now: at })).toMatchObject({ count: 1, shown: false });
  });
});

// --------------------------------------------------------------------------
// verifying
// --------------------------------------------------------------------------

describe('verifying the result', () => {
  const python = 'py';
  const say = () => {};
  const expected = { kikuchipy: '0.13.1', orix: '0.15.0' };

  it('passes when versions equal the lock\'s and the libraries import', async () => {
    const pip = fakePip({ installed: { ...NEW } });
    const verdict = await sync.verifyInstall({ run: pip.run }, python, expected, say);
    expect(verdict.ok).toBe(true);
    expect(pip.kinds()).toEqual(['versions', 'imports', 'check']);
  });

  it('fails on a version that is not the lock\'s, and does not bother importing', async () => {
    const pip = fakePip({ installed: { ...NEW, orix: '0.14.1' } });
    const verdict = await sync.verifyInstall({ run: pip.run }, python, expected, say);
    expect(verdict.ok).toBe(false);
    expect(verdict.detail).toMatch(/orix 0\.14\.1 instead of 0\.15\.0/);
    expect(pip.kinds()).toEqual(['versions']);
  });

  it('fails on a package that is not there at all', async () => {
    const pip = fakePip({ installed: { kikuchipy: '0.13.1' } });
    const verdict = await sync.verifyInstall({ run: pip.run }, python, expected, say);
    expect(verdict.ok).toBe(false);
    expect(verdict.detail).toMatch(/orix \(missing\)/);
  });

  it('fails when the libraries do not import, even though the metadata is right', async () => {
    const pip = fakePip({ installed: { ...NEW }, scenario: { imports: 'fail' } });
    const verdict = await sync.verifyInstall({ run: pip.run }, python, expected, say);
    expect(verdict.ok).toBe(false);
    expect(verdict.detail).toMatch(/importing kikuchipy, orix, pyebsdindex failed/);
  });

  it('fails when the versions cannot be read, rather than assuming they are right', async () => {
    const pip = fakePip({ installed: { ...NEW }, scenario: { versions: 'broken' } });
    const verdict = await sync.verifyInstall({ run: pip.run }, python, expected, say);
    expect(verdict.ok).toBe(false);
    expect(verdict.detail).toMatch(/could not be read/);
  });

  it('logs pip check and does not gate on it', async () => {
    const lines = [];
    const pip = fakePip({ installed: { ...NEW }, scenario: { check: 1 } });
    const verdict = await sync.verifyInstall({ run: pip.run }, python, expected, (l) => lines.push(l));
    expect(verdict.ok).toBe(true);
    expect(lines.some((l) => /pip check/.test(l) && /exit 1/.test(l))).toBe(true);
  });
});

// --------------------------------------------------------------------------
// the whole thing, with a pip that is a script
// --------------------------------------------------------------------------

describe('syncing', () => {
  it('opt-out: nothing is read, nothing is run, and the log says so', async () => {
    const h = makeHome();
    const pip = fakePip();
    const { made, result } = run({ env: { [sync.SKIP_ENV]: '1' } }, h, pip);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'skip', reason: 'opt-out' });
    expect(pip.calls).toEqual([]);
    expect(made.log.join('\n')).toContain(sync.SKIP_ENV);
  });

  it('the fast path starts no process at all', async () => {
    const h = makeHome();
    sync.recordFirstInstall({
      home: h.home, mode: 'cpu', lockName: 'requirements-lock-cpu.txt', lockText: LOCK,
      platformName: 'win32',
    });
    const pip = fakePip();
    const res = await run({}, h, pip).result;
    expect(res).toEqual({ ok: true, action: 'fastpath' });
    expect(pip.calls).toEqual([]);
  });

  it('changing the lock by one byte takes the fast path away', async () => {
    const h = makeHome();
    sync.recordFirstInstall({
      home: h.home, mode: 'cpu', lockName: 'requirements-lock-cpu.txt', lockText: LOCK,
      platformName: 'win32',
    });
    fs.appendFileSync(path.join(h.home, 'runtime', 'requirements-lock-cpu.txt'), 'extra==1\n');
    const pip = fakePip({ installed: { ...NEW } });
    const res = await run({}, h, pip).result;
    expect(res.action).toBe('noop');
    expect(pip.kinds()[0]).toBe('dry');
  });

  it('a 0.4.6 installation: asks pip, installs, verifies, records -- in that order', async () => {
    const h = makeHome();
    const pip = fakePip();
    const { made, result } = run({}, h, pip);
    const res = await result;

    expect(res.action).toBe('synced');
    expect(pip.kinds()).toEqual([
      'dry', 'versions', 'install', 'versions', 'imports', 'check',
    ]);
    // the install is the wizard's command against the shipped lock
    const install = pip.calls.find((c) => c.kind === 'install');
    expect(install.args[install.args.indexOf('-r') + 1]).toBe(
      path.join(h.home, 'runtime', 'requirements-lock-cpu.txt'));
    expect(pip.state.installed).toMatchObject({ kikuchipy: '0.13.1', orix: '0.15.0', pyebsdindex: '0.3.10.1' });
    // recorded, marker gone, nothing left over
    expect(readJson(h.home, sync.RECORD_FILE)).toMatchObject({
      sha256: sync.lockDigest(LOCK), mode: 'cpu', lock: 'requirements-lock-cpu.txt',
      runtimeTag: 'v0.4.7', how: 'sync',
    });
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(exists(h.home, sync.FAILURE_FILE)).toBe(false);
    expect(exists(h.home, path.join('setup-tmp', 'package-sync-report.json'))).toBe(false);
    // the waiting page was told, once (and before the question: see below)
    expect(made.onPhase).toHaveBeenCalledTimes(1);
    expect(made.onPhase).toHaveBeenCalledWith('syncing');
    expect(made.notify).not.toHaveBeenCalled();
  });

  it('logs what pip says it did, not the hundred packages it found already in place', async () => {
    const h = makeHome();
    const pip = fakePip();
    const inner = pip.run;
    pip.run = async (exe, args, opts) => {
      if (args.includes('install') && !args.includes('--dry-run') && opts.onLine) {
        opts.onLine('Requirement already satisfied: numpy>=1.23 in c:/x');
        opts.onLine('Collecting kikuchipy==0.13.1');
        opts.onLine('Successfully installed kikuchipy-0.13.1');
      }
      return inner(exe, args, opts);
    };
    const { made, result } = run({}, h, pip);
    await result;
    const text = made.log.join('\n');
    expect(text).toContain('pip: Collecting kikuchipy==0.13.1');
    expect(text).toContain('pip: Successfully installed kikuchipy-0.13.1');
    expect(text).not.toContain('already satisfied');
  });

  it('the marker is on disk while pip writes, and says what it was doing', async () => {
    const h = makeHome();
    const pip = fakePip();
    let seen = null;
    const inner = pip.run;
    pip.run = async (exe, args, opts) => {
      if (args.includes('install') && !args.includes('--dry-run')) {
        seen = readJson(h.home, sync.MARKER_FILE);
      }
      return inner(exe, args, opts);
    };
    await run({}, h, pip).result;
    expect(seen).toMatchObject({
      schema: 1, lockSha256: sync.lockDigest(LOCK), mode: 'cpu',
      from: { kikuchipy: '0.11.3', orix: '0.14.1', pyebsdindex: '0.3.9.1' },
      to: { kikuchipy: '0.13.1', orix: '0.15.0', pyebsdindex: '0.3.10.1' },
    });
  });

  it('the second start is the fast path: no process', async () => {
    const h = makeHome();
    const first = fakePip();
    await run({}, h, first).result;
    const second = fakePip({ installed: first.state.installed });
    const res = await run({}, h, second).result;
    expect(res.action).toBe('fastpath');
    expect(second.calls).toEqual([]);
  });

  it('nothing to change: records that, installs nothing', async () => {
    const h = makeHome();
    const pip = fakePip({ installed: { ...NEW } });
    const { made, result } = run({}, h, pip);
    expect((await result).action).toBe('noop');
    expect(pip.kinds()).toEqual(['dry']);
    expect(readJson(h.home, sync.RECORD_FILE).how).toBe('noop');
    expect(made.onPhase).toHaveBeenCalledTimes(1);   // the question itself takes seconds
  });

  it('the page says "syncing" BEFORE pip is asked, not after: the question takes seconds of its own', async () => {
    const h = makeHome();
    const pip = fakePip();
    const onPhase = vi.fn();
    const seenAtFirstProcess = [];
    const inner = pip.run;
    const spy = async (exe, args, opts) => {
      seenAtFirstProcess.push(onPhase.mock.calls.map((c) => c[0]));
      return inner(exe, args, opts);
    };
    await run({}, h, pip, { run: spy, onPhase }).result;
    expect(pip.calls[0].kind).toBe('dry');
    expect(seenAtFirstProcess[0]).toEqual(['syncing']);
  });

  it('the fast path, a skip and a refusal for room never change the page', async () => {
    const h = makeHome();
    sync.recordFirstInstall({
      home: h.home, mode: 'cpu', lockName: 'requirements-lock-cpu.txt', lockText: LOCK, platformName: 'win32',
    });
    const fast = run({}, h, fakePip());
    await fast.result;
    expect(fast.made.onPhase).not.toHaveBeenCalled();
    const skipped = run({ env: { [sync.SKIP_ENV]: '1' } }, h, fakePip());
    await skipped.result;
    expect(skipped.made.onPhase).not.toHaveBeenCalled();
    const h2 = makeHome({ files: {} });
    fs.rmSync(path.join(h2.home, '.packages_lock.json'), { force: true });
    const tight = run({}, h2, fakePip(), { freeBytes: () => ({ bytes: 1, known: true }) });
    await tight.result;
    expect(tight.made.onPhase).not.toHaveBeenCalled();
  });

  it('never runs in a development checkout, or against a conda environment of the developer\'s', async () => {
    const h = makeHome();
    const pip = fakePip();
    expect((await run({ decisionMode: 'dev' }, h, pip).result).action).toBe('skip');
    expect((await run({ projectRootEnv: 'C:/checkout' }, h, pip).result).action).toBe('skip');
    expect((await run({ python: path.join(os.tmpdir(), 'conda', 'python.exe') }, h, pip).result).action).toBe('skip');
    expect(pip.calls).toEqual([]);
  });

  it('without .install_mode it does not guess the flavour', async () => {
    const h = makeHome({ mode: null });
    const pip = fakePip();
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'skip', reason: 'no-install-mode' });
    expect(pip.calls).toEqual([]);
  });

  it('a lock that redirects pip is not run', async () => {
    const h = makeHome({ lock: `${LOCK}--index-url https://evil.example/simple\n` });
    const pip = fakePip();
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'skip', reason: 'lock-unsafe' });
    expect(pip.calls).toEqual([]);
  });

  it('a runtime without the lock is skipped quietly', async () => {
    const h = makeHome();
    fs.rmSync(path.join(h.home, 'runtime', 'requirements-lock-cpu.txt'));
    const pip = fakePip();
    expect((await run({}, h, pip).result)).toMatchObject({ action: 'skip', reason: 'lock-missing' });
    expect(pip.calls).toEqual([]);
  });

  it('a gpu installation uses the gpu lock', async () => {
    const h = makeHome({ mode: 'gpu' });
    const pip = fakePip();
    await run({}, h, pip).result;
    const dry = pip.calls[0];
    expect(dry.args[dry.args.indexOf('-r') + 1]).toMatch(/requirements-lock-gpu\.txt$/);
    expect(readJson(h.home, sync.RECORD_FILE).mode).toBe('gpu');
  });

  it('too little room: no question is asked, and the user is told once', async () => {
    const h = makeHome();
    const pip = fakePip();
    const over = { freeBytes: () => ({ bytes: 100 * 1024 * 1024, known: true }) };
    const { made, result } = run({}, h, pip, over);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'space' });
    expect(pip.calls).toEqual([]);
    expect(made.notify).toHaveBeenCalledTimes(1);
    expect(readJson(h.home, sync.FAILURE_FILE)).toMatchObject({ reason: 'space', shown: true });
  });

  it('unknown free space is not "full"', async () => {
    const h = makeHome();
    const pip = fakePip();
    const res = await run({}, h, pip, { freeBytes: () => ({ bytes: 0, known: false }) }).result;
    expect(res.action).toBe('synced');
  });

  it('never throws: an exception inside is a log line and the previous packages', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { throwOn: 'dry' } });
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'exception' });
    expect(made.log.join('\n')).toMatch(/failed unexpectedly/);
  });
});

describe('the guard, in the flow', () => {
  const numpyEntry = {
    download_info: { url: 'https://files.example/numpy-9.whl', archive_info: { hashes: { sha256: 'a'.repeat(64) } } },
    metadata: { name: 'numpy', version: '9.0' },
  };

  it('a delta that names numpy is not applied, is recorded, and is shown to the user once', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { extraInDelta: numpyEntry } });
    const first = run({}, h, pip);
    const res = await first.result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'guard' });
    expect(pip.kinds()).toEqual(['dry']);          // nothing installed, nothing marked
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(first.made.notify).toHaveBeenCalledTimes(1);
    expect(readJson(h.home, sync.FAILURE_FILE)).toMatchObject({ reason: 'guard', shown: true, count: 1 });

    // the next start does not even ask pip, and does not show it again
    const pip2 = fakePip({ scenario: { extraInDelta: numpyEntry } });
    const second = run({}, h, pip2);
    expect((await second.result).action).toBe('skip');
    expect(pip2.calls).toEqual([]);
    expect(second.made.notify).not.toHaveBeenCalled();
  });

  it('makes no request of its own to learn a size: the only network use is pip\'s', async () => {
    const h = makeHome();
    const pip = fakePip();
    const headSize = vi.fn(async () => 10 ** 12);
    const res = await run({}, h, pip, { headSize }).result;
    expect(res.action).toBe('synced');
    expect(headSize).not.toHaveBeenCalled();
    expect(MAIN).not.toMatch(/headSize|net\.fetch/);
  });
});

describe('when the network is not there', () => {
  it('leaves the environment alone, records it, tells the user once -- and three failures later, stops asking', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { dry: 'offline' } });
    const shared = makeDeps(pip);
    const go = (extra = {}) => sync.syncPackages(
      { home: h.home, python: h.python, decisionMode: 'run', platformName: 'win32', env: {} },
      { ...shared.deps, notify: shared.notify, onPhase: shared.onPhase, ...extra });

    const one = await go();
    expect(one).toMatchObject({ action: 'failed', reason: 'network' });
    expect(pip.state.installed).toEqual(OLD);                   // byte-identical
    expect(exists(h.home, sync.RECORD_FILE)).toBe(false);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(readJson(h.home, sync.FAILURE_FILE)).toMatchObject({ reason: 'network', count: 1, shown: true });
    expect(shared.notify).toHaveBeenCalledTimes(1);

    shared.clock.t += 60_000;
    await go();
    shared.clock.t += 60_000;
    await go();
    expect(readJson(h.home, sync.FAILURE_FILE).count).toBe(3);
    expect(shared.notify).toHaveBeenCalledTimes(1);             // never again for the same reason

    // now backed off: the fourth start asks nothing
    const before = pip.calls.length;
    shared.clock.t += 60_000;
    const four = await go();
    expect(four.action).toBe('skip');
    expect(pip.calls.length).toBe(before);

    // and after a day, once
    shared.clock.t += 25 * 3600 * 1000;
    await go();
    expect(pip.calls.length).toBe(before + 1);
    expect(readJson(h.home, sync.FAILURE_FILE).count).toBe(4);
  });

  it('a question pip did not answer in time is a network failure, not a hang', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { dry: 'timeout' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'failed', reason: 'network' });
    const dry = pip.calls[0];
    expect(dry.opts.timeoutMs).toBe(20_000);
  });

  it('the connection coming back is a success that wipes the failure', async () => {
    const h = makeHome();
    const down = fakePip({ scenario: { dry: 'offline' } });
    await run({}, h, down).result;
    expect(exists(h.home, sync.FAILURE_FILE)).toBe(true);
    const up = fakePip();
    expect((await run({}, h, up).result).action).toBe('synced');
    expect(exists(h.home, sync.FAILURE_FILE)).toBe(false);
  });

  it('a different reason is allowed its own message', async () => {
    const h = makeHome();
    const shared = makeDeps(fakePip({ scenario: { dry: 'offline' } }));
    const go = (pip, extra = {}) => sync.syncPackages(
      { home: h.home, python: h.python, decisionMode: 'run', platformName: 'win32', env: {} },
      { ...shared.deps, run: pip.run, notify: shared.notify, onPhase: shared.onPhase, ...extra });
    await go(fakePip({ scenario: { dry: 'offline' } }));
    await go(fakePip(), { freeBytes: () => ({ bytes: 1, known: true }) });
    expect(shared.notify).toHaveBeenCalledTimes(2);
  });

  it('a download that fails after the question leaves everything as it was -- and the environment says so', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'offline' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'failed', reason: 'network' });
    expect(pip.state.installed).toEqual(OLD);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);       // the probe found nothing changed
    expect(pip.kinds()).not.toContain('imports');
    expect(pip.kinds()).not.toContain('reinstall');
  });
});

describe('a failed install is never taken on its word that it wrote nothing', () => {
  /** The invariant: after ANY failed install the environment is asked, while the
   *  marker is still on disk, before the marker may go. */
  it.each([
    ['an unreachable index', 'offline'],
    ['a full disk before the first file', 'space-untouched'],
    ['a kill while still downloading', 'timeout-downloading'],
  ])('%s: the versions are read with the marker still in place, then the marker goes', async (_label, install) => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install } });
    const inner = pip.run;
    const markerAtVersions = [];
    pip.run = async (exe, args, opts) => {
      const r = await inner(exe, args, opts);
      if (pip.kinds().includes('install') && pip.calls[pip.calls.length - 1].kind === 'versions') {
        markerAtVersions.push(exists(h.home, sync.MARKER_FILE));
      }
      return r;
    };
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ ok: true, action: 'failed' });
    const kinds = pip.kinds();
    expect(kinds[kinds.indexOf('install') + 1]).toBe('versions');
    expect(markerAtVersions).toEqual([true]);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);       // unchanged: nothing to look after
    expect(pip.state.installed).toEqual(OLD);
  });

  it('a Retrying warning in front of a failure while writing: reads as network, is checked, and is put right', async () => {
    expect(sync.classifyPipFailure({ text: WRITE_PHASE_TEXT })).toBe('network');   // the trap
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'write-failure' } });
    const inner = pip.run;
    let markerAtReinstall = null;
    pip.run = async (exe, args, opts) => {
      if (args.includes('--force-reinstall')) markerAtReinstall = exists(h.home, sync.MARKER_FILE);
      return inner(exe, args, opts);
    };
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'synced', how: 'repair' });
    expect(markerAtReinstall).toBe(true);
    expect(pip.state.installed).toMatchObject(NEW);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(made.notify).not.toHaveBeenCalled();
    expect(made.log.join('\n')).toMatch(/write phase|changed packages/);
  });

  it('...and when nothing can be put back, the repair wizard -- with the marker still there, no dialog about a network', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'write-failure', reinstall: 'offline' } });
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res.repair).toBe(true);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
    expect(exists(h.home, sync.RECORD_FILE)).toBe(false);
    expect(made.notify).not.toHaveBeenCalled();
  });

  it('pip\'s own "Installing collected packages" means verify, even when every version reads as before', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'write-failure-nothing-changed' } });
    const res = await run({}, h, pip).result;
    // The words, not the probe, sent it to the full check: the old versions are
    // not the lock's, so the check fails and the packages are put in place.
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.kinds()).toContain('reinstall');
    expect(pip.state.installed).toMatchObject(NEW);
  });

  it('a kill while pip was replacing files (its words say so): verified, put right', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'timeout-writing' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.kinds()).toContain('reinstall');
    expect(pip.state.installed).toMatchObject(NEW);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
  });

  it('a kill that left the environment changed but printed nothing about it: the probe catches it', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'timeout-changed' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.kinds()).toContain('reinstall');
    expect(pip.state.installed).toMatchObject(NEW);
  });

  it('a kill mid-write that cannot be put right is the repair wizard, with the marker', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'timeout-writing', reinstall: 'offline' } });
    const res = await run({}, h, pip).result;
    expect(res.repair).toBe(true);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
  });
});

describe('an import check that does not answer in time is not a broken environment', () => {
  const python = 'py';
  const expected = { kikuchipy: '0.13.1', orix: '0.15.0' };

  it('verifyInstall calls it unverified; a non-zero exit stays broken', async () => {
    const slow = await sync.verifyInstall(
      { run: fakePip({ installed: { ...NEW }, scenario: { imports: 'timeout' } }).run }, python, expected, () => {});
    expect(slow).toMatchObject({ ok: false, unverified: true });
    expect(slow.detail).toMatch(/did not finish within 120 s/);

    const broken = await sync.verifyInstall(
      { run: fakePip({ installed: { ...NEW }, scenario: { imports: 'fail' } }).run }, python, expected, () => {});
    expect(broken.ok).toBe(false);
    expect(broken.unverified).toBeFalsy();
  });

  it('after an install: start with what is there, keep the marker, tell once, reinstall nothing', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { imports: 'timeout' } });
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', unverified: true });
    expect(res.repair).toBeUndefined();
    expect(pip.kinds()).not.toContain('reinstall');
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);        // the next start verifies again
    expect(exists(h.home, sync.RECORD_FILE)).toBe(false);
    expect(made.notify).toHaveBeenCalledTimes(1);
    expect(made.notify).toHaveBeenCalledWith(expect.objectContaining({ unverified: true }));
    expect(made.log.join('\n')).toMatch(/did not finish within 120 s/);
  });

  it('after a reinstall too', async () => {
    const h = makeHome();
    let corrupted = false;
    const pip = fakePip({ scenario: {
      afterInstall: (installed) => { if (!corrupted) { corrupted = true; installed.orix = '0.14.1'; } },
    } });
    const inner = pip.run;
    pip.run = async (exe, args, opts) => {
      if (args[0] === '-c' && args[1].startsWith('import kikuchipy') && pip.kinds().includes('reinstall')) {
        pip.calls.push({ exe, args, kind: 'imports', opts });
        return { code: -1, timedOut: true, stdout: '', stderr: '', output: '' };
      }
      return inner(exe, args, opts);
    };
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'failed', unverified: true });
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
  });
});

describe('an interrupted update with nothing named in its marker, and no way to ask pip', () => {
  const marker = { [sync.MARKER_FILE]: { schema: 1 } };

  it('libraries that import: start, marker kept, told once as unverified', async () => {
    const h = makeHome({ files: marker });
    const pip = fakePip({ installed: { ...NEW }, scenario: { dry: 'offline' } });
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'network', unverified: true });
    expect(pip.kinds()).toEqual(['dry', 'imports']);            // one look, once
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
    expect(made.notify).toHaveBeenCalledWith(expect.objectContaining({ unverified: true }));
  });

  it('libraries that do not import: the repair wizard, not a backend that dies on its first import', async () => {
    const h = makeHome({ files: marker });
    const pip = fakePip({ installed: { ...NEW }, scenario: { dry: 'offline', imports: 'fail' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ ok: false, repair: true });
    expect(pip.kinds()).toEqual(['dry', 'imports']);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
  });

  it('an import check that does not answer in time is not a broken environment', async () => {
    const h = makeHome({ files: marker });
    const pip = fakePip({ installed: { ...NEW }, scenario: { dry: 'offline', imports: 'timeout' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ ok: true, action: 'failed', unverified: true });
  });

  it('without a marker the same dry-run failure asks nothing else', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { dry: 'offline' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'failed', reason: 'network' });
    expect(res.unverified).toBeUndefined();
    expect(pip.kinds()).toEqual(['dry']);
  });
});

describe('when pip fails while writing', () => {
  it('a disk that fills BEFORE anything changed: skipped, marker removed', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'space-untouched' } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'failed', reason: 'space' });
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(pip.kinds()).not.toContain('reinstall');
  });

  it('a disk that fills AFTER one package was replaced: put back, verified, recorded', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { install: 'space' } });
    const res = await run({}, h, pip).result;
    // the reinstall is the one retry; it succeeds in this scenario
    expect(pip.kinds()).toContain('reinstall');
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.state.installed).toMatchObject(NEW);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
  });
});

describe('when the result is not what the lock asked for', () => {
  it('one retry, and when that fixes it, the result is recorded', async () => {
    const h = makeHome();
    // pip exits 0 but leaves orix at the old version, once
    let corrupted = false;
    const pip = fakePip({ scenario: { afterInstall: (installed) => {
      if (!corrupted) { corrupted = true; installed.orix = '0.14.1'; }
    } } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.kinds()).toEqual([
      'dry', 'versions', 'install', 'versions', 'reinstall', 'versions', 'imports', 'check',
    ]);
    const reinstall = pip.calls.find((c) => c.kind === 'reinstall');
    expect(reinstall.args).toContain('--force-reinstall');
    expect(reinstall.args).toContain('orix==0.15.0');
  });

  it('after the retry it is still wrong: the repair wizard, no backend, the marker stays', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { imports: 'fail' } });
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res.repair).toBe(true);
    expect(res.ok).toBe(false);
    expect(res.detail).toMatch(/importing/);
    expect(pip.kinds().filter((k) => k === 'reinstall')).toHaveLength(1);   // exactly one retry
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
    expect(exists(h.home, sync.RECORD_FILE)).toBe(false);
    expect(made.notify).not.toHaveBeenCalled();                              // the wizard is the message
  });

  it('and the next start does not take the fast path, nor trust the environment: it repairs and verifies', async () => {
    const h = makeHome();
    await run({}, h, fakePip({ scenario: { imports: 'fail' } })).result;
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
    // pip believes everything is in place; only the marker says otherwise
    const pip = fakePip({ installed: { ...NEW } });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.kinds()).toContain('reinstall');
    expect(pip.kinds()).toContain('imports');
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(readJson(h.home, sync.RECORD_FILE).how).toBe('repair');
  });
});

describe('after a start that was killed in the middle', () => {
  it('re-installs what the marker named, whatever pip believes, verifies, and clears the marker', async () => {
    const h = makeHome({
      files: {
        [sync.MARKER_FILE]: {
          schema: 1, lockSha256: sync.lockDigest(LOCK), mode: 'cpu',
          from: OLD, to: { kikuchipy: '0.13.1', orix: '0.15.0', pyebsdindex: '0.3.10.1' },
        },
      },
    });
    // pip thinks everything is fine (the metadata of the new versions is there)
    const pip = fakePip({ installed: { ...NEW } });
    const res = await run({}, h, pip).result;
    expect(res.action).toBe('synced');
    const reinstall = pip.calls.find((c) => c.kind === 'reinstall');
    expect(reinstall).toBeTruthy();
    expect(reinstall.args.filter((a) => a.includes('=='))).toEqual([
      'kikuchipy==0.13.1', 'orix==0.15.0', 'pyebsdindex==0.3.10.1',
    ]);
    expect(pip.kinds()).toContain('imports');
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
    expect(readJson(h.home, sync.RECORD_FILE).sha256).toBe(sync.lockDigest(LOCK));
  });

  it('refuses to hand pip a spec from a marker that is not a plain name==version', async () => {
    const h = makeHome({
      files: { [sync.MARKER_FILE]: { to: { 'kikuchipy; rm': '1', '--index-url': 'x', orix: '0.15.0' } } },
    });
    const pip = fakePip({ installed: { ...NEW } });
    await run({}, h, pip).result;
    const reinstall = pip.calls.find((c) => c.kind === 'reinstall');
    expect(reinstall.args.filter((a) => a.includes('=='))).toEqual(['orix==0.15.0']);
  });

  it('offline, but what the marker named is in place and imports: verified, finished, marker cleared', async () => {
    const h = makeHome({
      files: { [sync.MARKER_FILE]: { to: { kikuchipy: '0.13.1' }, from: OLD } },
    });
    const pip = fakePip({
      installed: { ...OLD, kikuchipy: '0.13.1' },
      scenario: { dry: 'offline', reinstall: 'offline' },
    });
    const res = await run({}, h, pip).result;
    expect(res).toMatchObject({ action: 'synced', how: 'repair' });
    expect(pip.kinds()).toContain('imports');
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
  });

  it('offline, with versions that differ from the lock but libraries that import: starts, marker kept, told once', async () => {
    const h = makeHome({
      files: { [sync.MARKER_FILE]: { to: { kikuchipy: '0.13.1' }, from: OLD } },
    });
    const pip = fakePip({ scenario: { dry: 'offline', reinstall: 'offline' } });
    const { made, result } = run({}, h, pip);
    const res = await result;
    expect(res).toMatchObject({ ok: true, action: 'failed', reason: 'network' });
    expect(res.detail).toMatch(/still unverified/);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);       // the next start tries again
    expect(exists(h.home, sync.RECORD_FILE)).toBe(false);
    expect(made.notify).toHaveBeenCalledTimes(1);
  });

  it('offline, with libraries that no longer import: nothing can fix it from here -- the repair wizard', async () => {
    const h = makeHome({
      files: { [sync.MARKER_FILE]: { to: { kikuchipy: '0.13.1', orix: '0.15.0' }, from: OLD } },
    });
    const pip = fakePip({ scenario: { dry: 'offline', reinstall: 'offline', imports: 'fail' } });
    const res = await run({}, h, pip).result;
    expect(res.repair).toBe(true);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
  });
});

describe('when the app is closed during it', () => {
  it('during the question: nothing was written, nothing is recorded, nobody is told', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { cancelOn: 'dry' } });
    const { made, result } = run({}, h, pip, { isCancelled: () => true });
    expect(await result).toEqual({ cancelled: true });
    expect(made.notify).not.toHaveBeenCalled();
    expect(exists(h.home, sync.FAILURE_FILE)).toBe(false);
    expect(exists(h.home, sync.MARKER_FILE)).toBe(false);
  });

  it('during the install: the marker stays, so the next start verifies; no failure, no dialog', async () => {
    const h = makeHome();
    const pip = fakePip({ scenario: { cancelOn: 'install' } });
    let quitting = false;
    const inner = pip.run;
    pip.run = async (...a) => { const r = await inner(...a); if (r.cancelled) quitting = true; return r; };
    const { made, result } = run({}, h, pip, { isCancelled: () => quitting });
    expect(await result).toEqual({ cancelled: true });
    expect(exists(h.home, sync.MARKER_FILE)).toBe(true);
    expect(exists(h.home, sync.FAILURE_FILE)).toBe(false);
    expect(made.notify).not.toHaveBeenCalled();
  });
});

describe('macOS', () => {
  function macHome() {
    const home = path.join(tmp, 'home');
    fs.mkdirSync(path.join(home, 'runtime'), { recursive: true });
    fs.mkdirSync(path.join(home, 'envs', 'orienta', 'bin'), { recursive: true });
    const python = path.join(home, 'envs', 'orienta', 'bin', 'python');
    fs.writeFileSync(python, '');
    fs.writeFileSync(path.join(home, '.install_mode'), 'cpu\n');
    fs.writeFileSync(path.join(home, 'runtime', 'orienta-macos-lock.yml'), 'version: 1\n');
    return { home, python };
  }

  it('calls the injected hook with everything decided, and starts no pip', async () => {
    const h = macHome();
    const pip = fakePip();
    const syncMacos = vi.fn(async () => ({ ok: true, action: 'noop' }));
    const { made, result } = run({ platformName: 'darwin' }, h, pip, { syncMacos });
    const res = await result;
    expect(res).toEqual({ ok: true, action: 'noop' });
    expect(pip.calls).toEqual([]);
    expect(syncMacos).toHaveBeenCalledTimes(1);
    const ctx = syncMacos.mock.calls[0][0];
    expect(ctx.lockName).toBe('orienta-macos-lock.yml');
    expect(ctx.digest).toBe(sync.lockDigest('version: 1\n'));
    expect(ctx.decision.action).toBe('dryrun');
    expect(ctx.python).toBe(h.python);
    expect(ctx.env).toEqual({});
    expect(made.log.join('\n')).toMatch(/macOS: \{"ok":true,"action":"noop"\}/);
  });

  it('with a record for this lock the hook is not even called: macOS gets the fast path', async () => {
    const h = macHome();
    sync.recordFirstInstall({
      home: h.home, mode: 'cpu', lockName: 'orienta-macos-lock.yml', lockText: 'version: 1\n',
      platformName: 'darwin',
    });
    const syncMacos = vi.fn();
    const res = await run({ platformName: 'darwin' }, h, fakePip(), { syncMacos }).result;
    expect(res).toEqual({ ok: true, action: 'fastpath' });
    expect(syncMacos).not.toHaveBeenCalled();
  });

  it('without a hook the micromamba flow runs -- and a directory that is no conda environment is left alone', async () => {
    const h = macHome();
    const pip = fakePip();
    const { made, result } = run({ platformName: 'darwin' }, h, pip);
    const res = await result;
    expect(res).toEqual({ ok: true, action: 'skip', reason: 'not-a-conda-prefix' });
    expect(pip.calls).toEqual([]);
    expect(made.log.join('\n')).toMatch(/not a conda environment/);
  });

  it('never runs pip on macOS, whatever the flow decides', async () => {
    const h = macHome();
    const pip = fakePip();
    await run({ platformName: 'darwin' }, h, pip).result;
    expect(pip.kinds().filter((k) => k === 'dry' || k === 'install' || k === 'reinstall')).toEqual([]);
  });
});

// --------------------------------------------------------------------------
// the runner: real processes
// --------------------------------------------------------------------------

describe('the runner', () => {
  const node = process.execPath;

  it('returns output, exit code and lines without throwing', async () => {
    const runner = sync.createRunner();
    const lines = [];
    const res = await runner.run(node, ['-e', 'console.log("one"); console.error("two"); process.exit(3)'],
      { onLine: (l) => lines.push(l) });
    expect(res.code).toBe(3);
    expect(res.stdout).toContain('one');
    expect(res.stderr).toContain('two');
    expect(lines.sort()).toEqual(['one', 'two']);
    expect(res.timedOut).toBe(false);
  });

  it('a program that is not there is a result, not an exception', async () => {
    const res = await sync.createRunner().run(path.join(tmp, 'no-such-python'), []);
    expect(res.code).toBe(-1);
    expect(res.error).toBeTruthy();
  });

  it('ends a process that outlives its timeout, and says it did', async () => {
    const runner = sync.createRunner();
    const started = Date.now();
    const res = await runner.run(node, ['-e', 'setTimeout(() => {}, 60000)'], { timeoutMs: 400 });
    expect(res.timedOut).toBe(true);
    expect(Date.now() - started).toBeLessThan(15000);
  });

  it('killAll ends every child, flips cancelled, and later runs start nothing', async () => {
    const runner = sync.createRunner();
    const pending = runner.run(node, ['-e', 'setTimeout(() => {}, 60000)']);
    await new Promise((r) => setTimeout(r, 400));
    expect(runner.size).toBe(1);
    runner.killAll();
    const res = await pending;
    expect(res.cancelled).toBe(true);
    expect(runner.cancelled).toBe(true);
    expect(runner.size).toBe(0);
    const later = await runner.run(node, ['-e', 'console.log("should not run")']);
    expect(later.cancelled).toBe(true);
    expect(later.stdout).toBe('');
  });
});

// --------------------------------------------------------------------------
// the three files are known to everything that lists the home's contents
// --------------------------------------------------------------------------

describe('the three files', () => {
  it('are named, and are three', () => {
    expect(sync.OWN_FILES).toEqual(['.packages_lock.json', '.packages_sync.json', '.packages_sync_failed.json']);
  });

  it('are Orienta\'s to the setup, so a repair is not refused for them', () => {
    expect(installer.foreignEntries(tmp, [...sync.OWN_FILES])).toEqual([]);
  });

  it('are Orienta\'s to "remove all data"', () => {
    const home = path.join(tmp, 'Orienta');
    fs.mkdirSync(path.join(home, 'runtime', 'Database'), { recursive: true });
    fs.writeFileSync(path.join(home, '.orienta-home'), '');
    for (const f of sync.OWN_FILES) fs.writeFileSync(path.join(home, f), '{}');
    const plan = removeData.planRemoval(home, fs, tmp);
    expect(plan.ok).toBe(true);
    for (const f of sync.OWN_FILES) {
      expect(plan.entries).toContain(f);
      expect(plan.foreign).not.toContain(f);
    }
  });

  it('are removed by the Windows uninstaller', () => {
    for (const f of sync.OWN_FILES) {
      expect(NSH).toContain(`!insertmacro orientaRemovePath "$oHome\\${f}"`);
    }
  });

  it('leave nothing behind in the home when a write is killed between the temp file and the rename', () => {
    // A crash there strands the temp file. It must not turn the repair wizard or
    // "remove all data" against the user ("this folder contains files that are not
    // Orienta's"), so it is written where Orienta's own scratch space is.
    const home = path.join(tmp, 'Orienta');
    fs.mkdirSync(path.join(home, 'runtime', 'Database'), { recursive: true });
    fs.writeFileSync(path.join(home, '.orienta-home'), '');
    const dying = { ...fs, renameSync: () => { throw new Error('killed'); } };
    for (const f of sync.OWN_FILES) {
      expect(() => sync.writeJsonAtomic(path.join(home, f), { a: 1 }, dying)).toThrow('killed');
    }
    const stranded = fs.readdirSync(path.join(home, 'setup-tmp'));
    expect(stranded).toHaveLength(3);
    expect(stranded.every((n) => /^\.packages_.*\.json\.\d+\.tmp$/.test(n))).toBe(true);
    expect(fs.readdirSync(home).filter((n) => n.endsWith('.tmp'))).toEqual([]);
    expect(installer.foreignEntries(home)).toEqual([]);
    const plan = removeData.planRemoval(home, fs, tmp);
    expect(plan.ok).toBe(true);
    expect(plan.foreign).toEqual([]);
  });

  it('are still written whole: the target holds the new content and the temp name is gone', () => {
    const home = path.join(tmp, 'Orienta');
    sync.writeJsonAtomic(path.join(home, sync.RECORD_FILE), { a: 1 });
    expect(readJson(home, sync.RECORD_FILE)).toEqual({ a: 1 });
    expect(fs.readdirSync(path.join(home, 'setup-tmp'))).toEqual([]);
  });
});

describe('the first install records the lock', () => {
  it('is called in the installer right after .install_mode is written', () => {
    const mode = INSTALLER_SRC.indexOf("path.join(home, '.install_mode')");
    const record = INSTALLER_SRC.indexOf('recordPackagesLock(home, mode, resolvedTag, macosLockText)');
    expect(mode).toBeGreaterThan(-1);
    expect(record).toBeGreaterThan(mode);
    expect(record - mode).toBeLessThan(200);
  });

  it('writes the record from the lock that was installed, and a macOS one from the text the environment was made from', () => {
    const home = path.join(tmp, 'home');
    fs.mkdirSync(path.join(home, 'runtime'), { recursive: true });
    fs.writeFileSync(path.join(home, 'runtime', hostLock('cpu')), LOCK);
    installer.recordPackagesLock(home, 'cpu', 'v0.4.7');
    const rec = readJson(home, sync.RECORD_FILE);
    expect(rec).toMatchObject({ sha256: sync.lockDigest(LOCK), mode: 'cpu', runtimeTag: 'v0.4.7', how: 'install' });
    expect(rec.lock).toBe(hostLock('cpu'));
  });

  it('never fails an install: no lock on disk is a log line', () => {
    const home = path.join(tmp, 'empty');
    fs.mkdirSync(home, { recursive: true });
    expect(() => installer.recordPackagesLock(home, 'cpu', 'v0.4.7')).not.toThrow();
    expect(exists(home, sync.RECORD_FILE)).toBe(false);
    expect(fs.readFileSync(path.join(home, 'logs', 'orienta-setup.log'), 'utf8'))
      .toMatch(/could not record the package lock/);
  });

  it('a repair install supersedes the marker and the failure an interrupted sync left', () => {
    const home = path.join(tmp, 'home');
    fs.mkdirSync(path.join(home, 'runtime'), { recursive: true });
    fs.writeFileSync(path.join(home, 'runtime', hostLock('cpu')), LOCK);
    fs.writeFileSync(path.join(home, sync.MARKER_FILE), '{}');
    fs.writeFileSync(path.join(home, sync.FAILURE_FILE), '{}');
    installer.recordPackagesLock(home, 'cpu', 'v0.4.7');
    expect(exists(home, sync.MARKER_FILE)).toBe(false);
    expect(exists(home, sync.FAILURE_FILE)).toBe(false);
  });
});

// --------------------------------------------------------------------------
// the shell calls it -- in the right place
// --------------------------------------------------------------------------

describe('the shell does it, after the program files and before the backend', () => {
  // Same reason as bundledUpdate.test.js: a module nobody calls is a green test
  // on a disconnected wire. This reads the wire.
  const at = (needle, from = 0) => MAIN.indexOf(needle, from);

  it('awaits the sync after the runtime update and before the backend is spawned', () => {
    const update = at('await applyBundledUpdate(');
    const syncCall = at('await runPackageSync(');
    const spawn = at('startBackend(plan.python');
    expect(update).toBeGreaterThan(-1);
    expect(syncCall).toBeGreaterThan(update);     // the lock it uses is one of the files just applied
    expect(spawn).toBeGreaterThan(syncCall);      // Windows holds a running interpreter's files open
  });

  it('is skipped when the program files were left half-updated: the quit comes first', () => {
    const quit = MAIN.slice(at('await applyBundledUpdate('), at('await runPackageSync('));
    expect(quit).toMatch(/if\s*\(!update\.ok\)\s*\{[\s\S]{0,200}?app\.quit\(\)[\s\S]{0,40}?return;/);
  });

  it('runs inside the packaged, spawning branch only', () => {
    const branch = MAIN.slice(at('if (plan.spawn && !isDev) {'), at('if (plan.spawn) startBackend('));
    expect(branch).toContain('await applyBundledUpdate(');
    expect(branch).toContain('await runPackageSync(decision, plan)');
  });

  it('returns before the spawn when the environment needs repair, and shows the repair wizard', () => {
    const branch = MAIN.slice(at('await runPackageSync('), at('startBackend(plan.python'));
    expect(branch).toMatch(/if\s*\(sync\.repair\)\s*\{[\s\S]*?showRepairWizard\(t\(shellLocale\(\), 'syncRepairBody'\)\);\s*return;\s*\}/);
  });

  it('never lets a failure to run the sync at all stop the start', () => {
    expect(MAIN).toMatch(/async function runPackageSync\(decision, plan\) \{\s*try \{\s*return await runPackageSyncInner\(decision, plan\);\s*\} catch \(err\) \{(?:(?!throw)[\s\S]){0,300}?return \{ ok: true \};/);
  });

  it('starts nothing when the app is quitting', () => {
    const branch = MAIN.slice(at('await runPackageSync('), at('startBackend(plan.python'));
    expect(branch).toMatch(/if\s*\(sync\.cancelled\)\s*return;/);
  });

  it('hands the sync what its preconditions are about, and the macOS hook', () => {
    const fn = MAIN.slice(at('async function runPackageSync('), at('function showRepairWizard('));
    expect(fn).toContain('decisionMode: decision.mode');
    expect(fn).toContain('projectRootEnv: process.env.ORIENTA_PROJECT_ROOT');
    expect(fn).toContain('python: plan.python');
    // macOS runs package_sync_macos.js by default: main.js passes no stub.
    expect(fn).not.toContain('syncMacos');
    expect(fn).not.toContain('NO_MACOS_YET');
    expect(fn).toContain('installer.measureFree');
  });

  it('shows a failure in a dialog titled and worded for it, with the shell log\'s path', () => {
    const fn = MAIN.slice(at('async function runPackageSync('), at('function showRepairWizard('));
    expect(fn).toMatch(/notify:\s*\(\)\s*=>\s*showUpdateProblem\(\s*t\(lang, 'syncSkippedBody', \{ path: shellLog \}\), lang, t\(lang, 'updateFailedTitle'\)\)/);
    expect(fn).toContain("'orienta-shell.log'");
  });

  it('puts the waiting page into the syncing phase through the sync\'s own callback', () => {
    const fn = MAIN.slice(at('async function runPackageSync('), at('function showRepairWizard('));
    expect(fn).toContain('onPhase: (phase) => waitingPage.setPhase(mainWindow, phase)');
  });

  it('opens the repair wizard in a NEW window: the one made for a run carries no setup channels', () => {
    const fn = MAIN.slice(at('function showRepairWizard('), at('async function clearRendererCache'));
    expect(fn).toContain("createWindow({ mode: 'repair', message })");
    expect(fn).toMatch(/removeAllListeners\('closed'\)[\s\S]*?\.destroy\(\)/);
    expect(fn).not.toContain('loadSetupPlaceholder(');
  });

  it('ends the sync\'s children when the app quits -- in the handler that runs on every quit', () => {
    const handler = MAIN.slice(at("app.on('before-quit', () => {\n  // The package sync's"));
    const body = handler.slice(0, handler.indexOf('\n});'));
    expect(body).toContain('packageSyncRunner.killAll()');
    // before the early return that is only for a running SETUP
    expect(body.indexOf('packageSyncRunner.killAll()')).toBeLessThan(body.indexOf('if (!setupRunning) return;'));
  });

  it('keeps the runner where before-quit can reach it for exactly as long as the sync runs', () => {
    const fn = MAIN.slice(at('async function runPackageSync('), at('function showRepairWizard('));
    expect(fn).toContain('packageSyncRunner = runner;');
    expect(fn).toMatch(/finally\s*\{\s*packageSyncRunner = null;/);
  });

  it('puts the page back to its ordinary text once the sync is done', () => {
    const branch = MAIN.slice(at('await runPackageSync('), at('startBackend(plan.python'));
    expect(branch).toContain("waitingPage.setPhase(mainWindow, 'starting')");
  });
});

// --------------------------------------------------------------------------
// the waiting page keeps the words it was given
// --------------------------------------------------------------------------

describe('the waiting page', () => {
  function fakeWindow() {
    const scripts = [];
    const handlers = {};
    return {
      scripts,
      handlers,
      isDestroyed: () => false,
      once: (ev, fn) => { handlers[ev] = fn; },
      webContents: {
        once: (ev, fn) => { handlers[`wc:${ev}`] = fn; },
        executeJavaScript: (code) => { scripts.push(code); return Promise.resolve(); },
      },
    };
  }
  function makePage(locale = 'en') {
    const ticks = [];
    let clock = 1_000_000;
    const page = createWaitingPage({
      t, shellLanguage: (l) => String(l).slice(0, 2),
      getLocale: () => locale,
      now: () => clock,
      setIntervalFn: (fn) => { ticks.push(fn); return ticks.length; },
      clearIntervalFn: () => {},
    });
    return { page, ticks, advance: (ms) => { clock += ms; } };
  }
  const lastText = (w) => w.scripts[w.scripts.length - 1];

  it('every phase has its own words in all four languages', () => {
    for (const phase of Object.keys(PHASES)) {
      for (const lang of ['en', 'de', 'ja', 'zh']) {
        expect(STRINGS[lang][PHASES[phase].body], `${lang} ${phase} body`).toBeTruthy();
        expect(STRINGS[lang][PHASES[phase].hint], `${lang} ${phase} hint`).toBeTruthy();
      }
    }
  });

  it('the clock\'s next tick keeps the syncing text: it is not overwritten a second later', () => {
    const { page, ticks, advance } = makePage();
    const win = fakeWindow();
    page.startClock(win);
    page.setPhase(win, 'syncing');
    expect(lastText(win)).toContain(STRINGS.en.syncingBody);
    advance(1000);
    ticks[0]();
    advance(1000);
    ticks[0]();
    expect(win.scripts.length).toBeGreaterThanOrEqual(3);
    const tick = lastText(win);
    expect(tick).toContain(STRINGS.en.syncingBody);
    expect(tick).toContain(STRINGS.en.syncingHint);
    expect(tick).not.toContain(STRINGS.en.startingBody);
    expect(tick).toContain('"elapsed":"2 s"');
  });

  it('the update phase survives the clock as well -- the defect found while reading the code', () => {
    const { page, ticks, advance } = makePage();
    const win = fakeWindow();
    page.startClock(win);
    page.setPhase(win, 'updating');
    advance(1000);
    ticks[0]();
    expect(lastText(win)).toContain(STRINGS.en.updatingBody);
    expect(lastText(win)).not.toContain(STRINGS.en.startingBody);
  });

  it('going back to starting restores the ordinary text', () => {
    const { page, ticks, advance } = makePage();
    const win = fakeWindow();
    page.startClock(win);
    page.setPhase(win, 'syncing');
    page.setPhase(win, 'starting');
    advance(1000);
    ticks[0]();
    expect(lastText(win)).toContain(STRINGS.en.startingBody);
    expect(page.phase()).toBe('starting');
  });

  it('setting a phase does not touch the counter', () => {
    const { page } = makePage();
    const win = fakeWindow();
    page.setPhase(win, 'syncing');
    expect(JSON.parse(/const s = (\{.*?\});/s.exec(lastText(win))[1])).not.toHaveProperty('elapsed');
  });

  it('speaks the language it is asked for', () => {
    const { page } = makePage('de');
    const win = fakeWindow();
    page.setPhase(win, 'syncing');
    expect(lastText(win)).toContain(STRINGS.de.syncingBody);
  });

  it('an unknown phase is a programming error, not a blank page', () => {
    const { page } = makePage();
    expect(() => page.setPhase(fakeWindow(), 'syncin')).toThrow(/unknown waiting phase/);
  });

  it('a destroyed window is not painted', () => {
    const { page } = makePage();
    const win = fakeWindow();
    win.isDestroyed = () => true;
    page.setPhase(win, 'syncing');
    expect(win.scripts).toEqual([]);
  });

  it('main.js builds its clock from the phase and no longer writes the strings itself', () => {
    expect(MAIN).toContain("require('./waiting_page')");
    const fn = MAIN.slice(MAIN.indexOf('function startWaitingPageClock('), MAIN.indexOf('async function applyBundledUpdate'));
    expect(fn).toContain('waitingPage.startClock(window)');
    expect(fn).not.toContain("'startingBody'");
    expect(MAIN).toContain("waitingPage.setPhase(mainWindow, 'updating')");
    expect(MAIN).not.toContain('function paintUpdatingPage');
  });
});

describe('the strings', () => {
  const KEYS = ['syncingBody', 'syncingHint', 'syncSkippedBody', 'syncRepairBody'];
  it.each(['en', 'de', 'ja', 'zh'])('%s has the sync texts, and none is the English one', (lang) => {
    for (const key of KEYS) {
      expect(STRINGS[lang][key], `${lang}.${key}`).toBeTruthy();
      if (lang !== 'en') expect(STRINGS[lang][key]).not.toBe(STRINGS.en[key]);
    }
  });

  it.each(['en', 'de', 'ja', 'zh'])('%s names the three libraries and the log path', (lang) => {
    expect(STRINGS[lang].syncingBody).toMatch(/kikuchipy.*orix.*PyEBSDIndex/);
    expect(t(lang, 'syncSkippedBody', { path: 'C:/logs/x.log' })).toContain('C:/logs/x.log');
    expect(t(lang, 'syncSkippedBody', { path: 'P' })).not.toContain('{{');
  });
});
