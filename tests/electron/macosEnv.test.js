/**
 * Building the macOS environment: the decisions, without a Mac.
 *
 * Every flag here is load-bearing and several of them are load-bearing in a
 * way that produces no error when removed — which is why they are pinned
 * individually rather than by comparing one long argv string.
 */
import { describe, it, expect } from 'vitest';
import { createRequire } from 'node:module';
import path from 'node:path';
import fs from 'node:fs';

const requireCjs = createRequire(import.meta.url);
const mac = requireCjs('../../electron/setup/macos_env.js');

const OPTIONS = {
  root: '/data/Orienta/micromamba',
  prefix: '/data/Orienta/envs/orienta',
  lockFile: '/data/Orienta/runtime/orienta-macos-lock.yml',
};

describe('the lock file name', () => {
  it('accepts the names micromamba recognises', () => {
    expect(mac.isLockFileName('orienta-macos-lock.yml')).toBe(true);
    expect(mac.isLockFileName('anything-lock.yaml')).toBe(true);
  });

  it('rejects the name that installed nothing at all', () => {
    // Measured with micromamba 2.9.0: same bytes under this name gave exit 0,
    // no output and an EMPTY environment. Not an error — a silent success.
    expect(mac.isLockFileName('conda-lock-macos.yml')).toBe(false);
    expect(mac.isLockFileName('conda-lock.yml.bak')).toBe(false);
  });

  it('is the name the repository actually ships', () => {
    expect(mac.isLockFileName(mac.MACOS_LOCK_FILE)).toBe(true);
  });

  it('refuses to build a command around an unrecognised name', () => {
    expect(() => mac.createArgs({ ...OPTIONS, lockFile: '/x/conda-lock-macos.yml' }))
      .toThrow(/install NOTHING/);
  });
});

describe('pulling the lock out of the package early', () => {
  it('names the member itself and writes nothing to an archive path', () => {
    // `-p` sends the member to stdout. Nothing is extracted to a path taken
    // from the archive, so the applier's checks are not being worked around —
    // the full unpack still goes through apply_update.py afterwards.
    const args = mac.unzipMemberArgs('/tmp/pkg.zip', mac.MACOS_LOCK_FILE);
    expect(args[0]).toBe('/usr/bin/unzip');
    expect(args).toContain('-p');
    expect(args[args.length - 1]).toBe(mac.MACOS_LOCK_FILE);
    expect(args).not.toContain('-d');       // never "extract into a directory"
  });

  it('can be probed before anything is downloaded', () => {
    expect(mac.unzipProbeArgs()[0]).toBe('/usr/bin/unzip');
  });
});

describe('checking the lock before micromamba sees it', () => {
  // The real file, not a mock: a shape check written against an imagined
  // shape is worth nothing, and this one exists to catch silent failures.
  const real = fs.readFileSync(
    path.join(import.meta.dirname, '..', '..', mac.MACOS_LOCK_FILE), 'utf8');

  it('passes the file the repository actually ships', () => {
    expect(mac.lockFileComplaint(real)).toBeNull();
  });

  it('catches an empty extraction', () => {
    expect(mac.lockFileComplaint('')).toMatch(/empty/);
    expect(mac.lockFileComplaint('   \n  ')).toMatch(/empty/);
  });

  it('catches a truncated one', () => {
    // A partial `unzip -p` writes a prefix and exits non-zero; this does not
    // rely on the caller having noticed the exit code.
    expect(mac.lockFileComplaint(real.slice(0, 500))).toMatch(/truncated/);
  });

  it('catches a lockfile version micromamba cannot read', () => {
    expect(mac.lockFileComplaint(real.replace(/^version: 1$/m, 'version: 2')))
      .toMatch(/version 2/);
  });

  it('catches something that is not a lock file at all', () => {
    const notALock = `name: orienta\ndependencies:\n  - python=3.11\n${'#'.repeat(2000)}`;
    expect(mac.lockFileComplaint(notALock)).toMatch(/not a conda lock file/);
  });

  it('catches a lock for the wrong machine', () => {
    // The dangerous one: micromamba only WARNS about an absent platform and
    // then reports a successful, empty transaction.
    const complaint = mac.lockFileComplaint(real, { platform: 'osx-64' });
    expect(complaint).toMatch(/osx-64/);
  });

  it('catches a lock whose header lists the platform but which has no packages for it', () => {
    const headerOnly = real.replace(/^  platform: osx-arm64$/gm, '  platform: linux-64');
    expect(mac.lockFileComplaint(headerOnly)).toMatch(/unusable environment/);
  });

  it('catches a lock thinned down to a handful of packages', () => {
    // "At least one" was the first version and does not prevent what this
    // check is for: one package installs, micromamba reports success, and the
    // environment is unusable. A reviewer built exactly that file.
    let kept = 0;
    const thinned = real.replace(/^  platform: osx-arm64$/gm,
      (m) => (kept++ < 3 ? m : '  platform: linux-64'));
    expect(mac.lockFileComplaint(thinned)).toMatch(/only 3 package/);
  });

  // The three checks between "is it a lock file" and "is it for us" were
  // unreachable in the tests above: the not-a-lock fixture has no `version:`
  // line, so it never got past the first one.
  it('catches a lock with no content_hash', () => {
    expect(mac.lockFileComplaint(real.replace(/^  content_hash:$/m, '  other:')))
      .toMatch(/content_hash/);
  });

  it('catches a lock with no package list', () => {
    expect(mac.lockFileComplaint(real.replace(/^package:$/m, 'packages_maybe:')))
      .toMatch(/lists no packages/);
  });

  it('catches a lock whose header does not name our platform', () => {
    // Distinct from the record check below it, and previously masked because
    // both messages interpolate the platform name.
    const noHeader = real.replace(/^  - osx-arm64$/m, '  - linux-64');
    expect(mac.lockFileComplaint(noHeader)).toMatch(/does not cover osx-arm64/);
  });
});

describe('the lock file name, more carefully', () => {
  it('needs the hyphen, not just the suffix', () => {
    // micromamba's test is `ends_with("-lock.yml")`.
    expect(mac.isLockFileName('lock.yml')).toBe(false);
    expect(mac.isLockFileName('mylock.yml')).toBe(false);
  });

  it('is case-sensitive, because that ends_with test is', () => {
    expect(mac.isLockFileName('ORIENTA-LOCK.YML')).toBe(false);
  });
});

describe('micromamba create', () => {
  const args = mac.createArgs(OPTIONS);

  it('says yes, because the prompt has no tty detection', () => {
    // Without -y a non-interactive run can block for ever waiting for input
    // nobody will type.
    expect(args).toContain('-y');
  });

  it('ignores every configuration file and environment variable', () => {
    expect(args).toContain('--no-rc');
    expect(args).toContain('--no-env');
  });

  it('sets its own root, not only the prefix', () => {
    // The package cache lives at <root>/pkgs. Without -r that is the USER's
    // root prefix, and we would download gigabytes into their conda install.
    expect(args[args.indexOf('-r') + 1]).toBe(OPTIONS.root);
    expect(args[args.indexOf('-p') + 1]).toBe(OPTIONS.prefix);
  });

  it('states the platform instead of letting the binary decide', () => {
    // micromamba's own platform is compile-time. An x86_64 build under
    // Rosetta would silently produce a complete x86_64 environment.
    expect(args[args.indexOf('--platform') + 1]).toBe('osx-arm64');
  });

  it('does not pass --override-channels, which only constrains a solve', () => {
    // A lockfile install does not solve; every URL comes from the lock.
    expect(args).not.toContain('--override-channels');
  });
});

describe('the Rosetta guard', () => {
  it('asks micromamba what platform it is, without --platform', () => {
    // Measured: `info` rejects --platform ("The following arguments were not
    // expected"), and without it reports the platform the BINARY was built
    // for -- which is the thing we need to know.
    const args = mac.infoArgs(OPTIONS);
    expect(args).not.toContain('--platform');
    expect(args[0]).toBe('info');
  });

  it('reads the platform out of the report', () => {
    // The real shape, copied from a run: aligned, colon-separated.
    const stdout = [
      '       environment : orienta',
      '          platform : osx-arm64',
      '     base env path : /data',
    ].join('\n');
    expect(mac.platformFromInfo(stdout)).toBe('osx-arm64');
  });

  it('reads the wrong answer just as plainly', () => {
    expect(mac.platformFromInfo('          platform : osx-64')).toBe('osx-64');
  });

  it('returns null rather than guessing when there is no platform line', () => {
    expect(mac.platformFromInfo('some unrelated output')).toBeNull();
    expect(mac.platformFromInfo('')).toBeNull();
  });
});

describe('scrubbing the environment', () => {
  it('removes the named variables micromamba reads', () => {
    const scrubbed = mac.scrubbedEnv({
      MAMBA_ROOT_PREFIX: '/user/mamba',
      CONDARC: '/user/.condarc',
      CONDA_PREFIX: '/user/anaconda3',
      XDG_CONFIG_HOME: '/user/.config',
      PATH: '/usr/bin',
    });
    expect(scrubbed).toEqual({ PATH: '/usr/bin' });
  });

  it('removes any MAMBA_ or CONDA_ variable, including ones not on the list', () => {
    // Every config key is also settable as MAMBA_<KEY>, so the list alone
    // cannot be complete and the prefix sweep is what actually protects us.
    const scrubbed = mac.scrubbedEnv({
      MAMBA_SOMETHING_NEW: 'x',
      CONDA_ANYTHING: 'y',
      HOME: '/home/user',
    });
    expect(scrubbed).toEqual({ HOME: '/home/user' });
  });

  it('leaves everything else alone', () => {
    const scrubbed = mac.scrubbedEnv({ PATH: '/usr/bin', HOME: '/h', LANG: 'de_DE' });
    expect(scrubbed).toEqual({ PATH: '/usr/bin', HOME: '/h', LANG: 'de_DE' });
  });

  it('does not mutate the environment it was given', () => {
    const original = { CONDA_PREFIX: '/x', PATH: '/usr/bin' };
    mac.scrubbedEnv(original);
    expect(original.CONDA_PREFIX).toBe('/x');
  });
});

describe('the package cache', () => {
  it('lives under the root we chose, so removing it cannot touch the user', () => {
    expect(mac.packageCacheDir(OPTIONS.root)).toBe(path.join(OPTIONS.root, 'pkgs'));
  });

  it('is cleaned with our own root, never the default one', () => {
    const args = mac.cleanArgs(OPTIONS);
    expect(args[args.indexOf('-r') + 1]).toBe(OPTIONS.root);
    expect(args).toContain('-y');
    expect(args).toContain('--no-rc');
  });
});

describe('building the environment, in order', () => {
  // The order is the part that was decided rather than derived, and every
  // failure it guards against is silent, so it is driven with fakes here.
  function harness(overrides = {}) {
    const calls = [];
    const real = fs.readFileSync(
      path.join(import.meta.dirname, '..', '..', mac.MACOS_LOCK_FILE), 'utf8');
    const capture = overrides.capture || (async (exe, ...args) => {
      calls.push(['capture', exe, ...args]);
      if (exe === '/usr/bin/unzip' && args.includes('-p')) return real;
      if (args[0] === 'info') return '   platform : osx-arm64';
      return '';
    });
    const run = overrides.run || (async (exe, args) => {
      calls.push(['run', exe, ...args]);
    });
    const written = [];
    return {
      calls,
      written,
      options: {
        micromamba: '/data/micromamba/bin/micromamba',
        root: '/data/micromamba',
        prefix: '/data/envs/orienta',
        archive: '/tmp/pkg.zip',
        lockPath: '/tmp/orienta-macos-lock.yml',
        run,
        capture,
        writeFile: async (p, t) => { written.push([p, t.length]); },
        exists: async () => true,
        codesign: async () => {},
        ...overrides.options,
      },
    };
  }

  it('probes unzip and codesign before it reads or runs anything', async () => {
    const seen = [];
    const h = harness({ options: { codesign: async () => { seen.push('codesign'); } } });
    h.options.capture = async (exe, ...args) => {
      seen.push(`${exe} ${args[0] || ''}`.trim());
      return h.options.__capture(exe, ...args);
    };
    h.options.__capture = harness().options.capture;
    h.options.run = async (exe, args) => { seen.push(`run ${args[0]}`); };
    await mac.installMacosEnvironment(h.options);
    const unzipProbe = seen.findIndex((c) => c.startsWith('/usr/bin/unzip -v'));
    const codesign = seen.indexOf('codesign');
    const create = seen.indexOf('run create');
    expect(unzipProbe).toBeGreaterThanOrEqual(0);
    expect(codesign).toBeGreaterThan(unzipProbe);
    expect(create).toBeGreaterThan(codesign);
  });

  it('does not send the user to install developer tools codesign never needed', async () => {
    // The first version of this message said `xcode-select --install`. The
    // research (spec appendix A.1) showed codesign is a base-system binary
    // and ad-hoc signing has needed no Xcode tools since macOS 12 — so that
    // advice cost an afternoon and several gigabytes and fixed nothing.
    const h = harness({ options: { codesign: async () => { throw new Error('boom'); } } });
    await expect(mac.installMacosEnvironment(h.options)).rejects.toThrow(/cannot sign/);
    await expect(mac.installMacosEnvironment(h.options)).rejects.not.toThrow(/xcode-select/);
  });

  it('stops at a missing unzip rather than downloading first', async () => {
    const h = harness({
      capture: async (exe) => { if (exe === '/usr/bin/unzip') throw new Error('nope'); return ''; },
    });
    await expect(mac.installMacosEnvironment(h.options)).rejects.toThrow(/unzip is missing/);
  });

  it('refuses a damaged lock before micromamba is started', async () => {
    const h = harness({
      capture: async (exe, ...args) => {
        if (exe === '/usr/bin/unzip' && args.includes('-p')) return 'not a lock file at all';
        return '';
      },
    });
    await expect(mac.installMacosEnvironment(h.options)).rejects.toThrow(/damaged|not a conda/);
    expect(h.calls.some((c) => c[0] === 'run')).toBe(false);
  });

  it('refuses an environment built for the wrong architecture', async () => {
    // The Rosetta case: micromamba ran, and built x86_64 without complaining.
    const real = fs.readFileSync(
      path.join(import.meta.dirname, '..', '..', mac.MACOS_LOCK_FILE), 'utf8');
    const h = harness({
      capture: async (exe, ...args) => {
        if (exe === '/usr/bin/unzip' && args.includes('-p')) return real;
        if (args[0] === 'info') return '   platform : osx-64';
        return '';
      },
    });
    await expect(mac.installMacosEnvironment(h.options)).rejects.toThrow(/Rosetta/);
  });

  it('does not fail the install when only the cache cleanup fails', async () => {
    // Wasted disk is not a broken environment, and refusing here would strand
    // a user who has a working one.
    const h = harness({
      run: async (exe, args) => {
        if (args[0] === 'clean') throw new Error('cache busy');
      },
    });
    await expect(mac.installMacosEnvironment(h.options)).resolves.toMatchObject({
      platform: 'osx-arm64',
    });
  });

  it('refuses an environment that produced no interpreter', async () => {
    // `micromamba info` prints a platform for an empty prefix too, so the
    // architecture check does not cover this. Without it, the failure is
    // discovered when the unpacker is spawned and reported as "unpacking the
    // program files failed" -- blaming the wrong step entirely.
    const h = harness({ options: { exists: async () => false } });
    await expect(mac.installMacosEnvironment(h.options))
      .rejects.toThrow(/no interpreter at/);
  });

  it('writes the lock where micromamba will accept its name', async () => {
    const h = harness();
    await mac.installMacosEnvironment(h.options);
    expect(h.written).toHaveLength(1);
    expect(mac.isLockFileName(h.written[0][0])).toBe(true);
  });
});

describe('the two allowlists know what a macOS install creates', () => {
  /**
   * Every directory the installer writes has to appear in BOTH lists, and
   * they are separate files that were each written for a Windows layout:
   *
   *  - the setup refuses a data folder containing anything it does not
   *    recognise, so a missing name makes the SECOND run (a retry, the
   *    "install the processor version instead" button, repair mode) refuse
   *    with "this folder contains files that are not Orienta's", naming
   *    files the installer wrote itself;
   *  - the uninstaller removes only what it recognises, so a missing name
   *    leaves it behind and reports it as the user's own.
   *
   * Both happened: the uninstaller carried `mamba-root`, a name from the
   * spec that nothing creates, and neither list had `envs`.
   */
  const P = requireCjs('../../electron/platform.js');
  const installer = requireCjs('../../electron/setup/installer.js');
  const remove = requireCjs('../../electron/remove_data.js');

  // Two components on POSIX as well. `/o` is ONE, and planRemoval refuses
  // anything shallower than two so that no bug can ever aim it at a root or a
  // mount point -- so on Linux this test tripped the very guard it was written
  // to look past, and plan.entries came back empty.
  const home = process.platform === 'win32' ? 'C:\\O' : '/var/o';
  const created = [P.condaRootIn(home), P.condaPrefixIn(home), P.micromambaExeIn(home)];
  // The top-level entry each of those lives under.
  const topLevel = [...new Set(created.map(
    (p) => path.relative(home, p).split(path.sep)[0]))];

  it('creates the directories these names describe', () => {
    expect(topLevel.sort()).toEqual([P.CONDA_ENVS_DIR, P.CONDA_ROOT_DIR].sort());
  });

  it.each(topLevel)('the setup does not call %s a stranger\'s file', (name) => {
    const foreign = installer.foreignEntries(home, [name]);
    expect(foreign).toEqual([]);
  });

  it.each(topLevel)('the uninstaller removes %s instead of keeping it', (name) => {
    // `exists()` in remove_data goes through lstatSync, so THAT is what
    // decides here — a `.git` that appears to exist makes the plan refuse the
    // folder as a checkout before it classifies anything.
    const io = {
      readdirSync: () => [name, '.orienta-home'],
      lstatSync: (p) => {
        if (String(p).endsWith('.git')) throw new Error('ENOENT');
        return { isSymbolicLink: () => false, isDirectory: () => true };
      },
      statSync: () => ({ isDirectory: () => true }),
      existsSync: (p) => !String(p).endsWith('.git'),
      realpathSync: (p) => p,
    };
    const elsewhere = process.platform === 'win32' ? 'C:\\Users\\someone' : '/home/someone';
    const plan = remove.planRemoval(home, io, elsewhere);
    expect(plan.foreign).not.toContain(name);
    expect(plan.entries).toContain(name);
  });
});

describe('where the environment is allowed to live', () => {
  const P = requireCjs('../../electron/platform.js');
  const home = P.defaultDataFolder('darwin', {}, '/Users/prof');
  const prefix = P.condaPrefixIn(home);

  it('is not inside the .app bundle', () => {
    // If it were, building it would modify the signed bundle and break the
    // app's own ad-hoc seal ON FIRST RUN — and a broken seal is refused even
    // without quarantine, with the dialog that has no "open anyway".
    expect(prefix).not.toMatch(/\.app([/\\]|$)/);
  });

  it('is not in a folder iCloud Drive syncs by default', () => {
    // Sync services attach extended attributes that make codesign fail with
    // "resource fork, Finder information, or similar detritus not allowed".
    // macOS puts Desktop and Documents in iCloud Drive by default; Application
    // Support is not synced.
    expect(prefix).not.toMatch(/[/\\](Desktop|Documents)[/\\]/);
    expect(prefix).toMatch(/Application Support/);
  });
});

describe('how much disk a Mac needs', () => {
  const installer = requireCjs('../../electron/setup/installer.js');

  it('asks for more than Windows, because the cache and the environment coexist', () => {
    // The Windows number is a measured pip install. On macOS the package
    // cache is only deleted AFTER the transaction, so both exist at the
    // worst moment — and the spec's own figure for the installed folder
    // (2-5 GB) already exceeds the Windows floor of 2.5 GB. That floor would
    // pass and then run out of disk part way through.
    expect(installer.cpuBytesNeeded('darwin'))
      .toBeGreaterThan(installer.cpuBytesNeeded('win32'));
    expect(installer.cpuBytesNeeded('darwin')).toBeGreaterThan(5 * 1024 ** 3);
  });

  it('leaves Windows and Linux on the number that shipped', () => {
    expect(installer.cpuBytesNeeded('win32')).toBe(installer.CPU_BYTES_NEEDED);
    expect(installer.cpuBytesNeeded('linux')).toBe(installer.CPU_BYTES_NEEDED);
  });

  it('blocks a Mac that would fit the Windows install but not this one', () => {
    const between = (installer.CPU_BYTES_NEEDED + installer.MACOS_CPU_BYTES_NEEDED) / 2;
    const onMac = installer.chooseRecommendation({ present: false }, between, true, 'darwin');
    const onWindows = installer.chooseRecommendation({ present: false }, between, true, 'win32');
    expect(onMac.blocked).toBe(true);
    expect(onWindows.blocked).toBeFalsy();
  });
});

describe('resuming a half-finished install', () => {
  const installer = requireCjs('../../electron/setup/installer.js');
  const resumable = installer.resumeHasSomethingToContinueFrom;

  it('lets Windows and Linux resume as they always did', () => {
    // There the interpreter is unpacked in step 2 and step 4 checks the lock;
    // nothing about this changed for them.
    expect(resumable('/o', 'win32', () => false)).toBe(true);
    expect(resumable('/o', 'linux', () => false)).toBe(true);
  });

  it('refuses a macOS resume with no interpreter', () => {
    // A resume skips the only step that creates one. Without this the run
    // walks to the end and writes `.python_path` for a file that is not
    // there — a finished installation that cannot start.
    expect(resumable('/o', 'darwin', () => false)).toBe(false);
  });

  it('allows a macOS resume once the environment is there', () => {
    expect(resumable('/o', 'darwin', () => true)).toBe(true);
  });

  it('looks for the interpreter where macOS actually puts it', () => {
    const asked = [];
    resumable('/o', 'darwin', (p) => { asked.push(p); return true; });
    expect(asked).toEqual([requireCjs('../../electron/platform.js')
      .interpreterIn('/o', 'darwin')]);
  });
});

describe('how the wizard wires it up', () => {
  /**
   * Source-level, and deliberately so: driving `runSetupInner` would need the
   * network, a package and a Mac. These assert the three things that would
   * otherwise break silently on a platform nobody here can run.
   */
  const lines = fs.readFileSync(
    path.join(import.meta.dirname, '..', '..', 'electron', 'setup', 'installer.js'), 'utf8')
    .split('\n').map((l) => l.trim());

  /**
   * Each guard is pinned as a WHOLE LINE, not as a substring.
   *
   * A reviewer defeated the substring versions of all four of these with
   * mutations that keep the text: `!== 'darwin' || true`, wrapping a call in
   * `if (process.env.NEVER) {...}`, downgrading a refusal to a log line. The
   * first of those is "pip runs on macOS after all", which restores the
   * three-OpenMP-runtime crash this entire branch exists to fix — with the
   * test green and the comment above it still explaining why it must not.
   *
   * Whole-line equality closes that: any of those mutations changes the line.
   * It is still source-level, and still weaker than behaviour. Driving
   * `runSetupInner` needs a network, a package and a Mac, so what a real
   * darwin run DOES is T11's to prove, not this file's.
   */
  const hasLine = (text) => lines.includes(text);

  it('calls the macOS builder, unconditionally', () => {
    expect(hasLine('await macosEnv.installMacosEnvironment({')).toBe(true);
  });

  it('does not run pip on macOS', () => {
    // The environment is already complete there; a pip step would install a
    // second copy of the stack, with the OpenMP runtimes this all exists to
    // avoid.
    expect(hasLine("if (platform.current() !== 'darwin') {")).toBe(true);
    const pip = lines.findIndex((l) => l.includes("'-m', 'pip', 'install'"));
    const guard = lines.findIndex((l) => l === "if (platform.current() !== 'darwin') {");
    expect(pip).toBeGreaterThan(guard);
    expect(guard).toBeGreaterThan(-1);
  });

  it('verifies the micromamba binary against a pinned digest', () => {
    // Not a checksum fetched over the same connection as the download, and
    // never an unverified binary: the refusal returns, it does not log.
    expect(hasLine('if (!runtime.sha256) {')).toBe(true);
    const refusal = lines.indexOf('if (!runtime.sha256) {');
    expect(lines.slice(refusal, refusal + 8).some((l) => l.startsWith('return {'))).toBe(true);
    expect(lines.some((l) => l.startsWith('await downloadPinned(runtime.url, runtime.sha256'))).toBe(true);
  });

  it('actually asks before resuming', () => {
    // The function is tested directly above; this is the other half — that
    // it is CALLED. Without this line the guard is dead code and a resumed
    // macOS run reports a finished install that cannot start.
    expect(hasLine('if (resume && !resumeHasSomethingToContinueFrom(home)) {')).toBe(true);
  });

  it('makes the downloaded binary executable', () => {
    // A downloaded file has no executable bit, and Apple Silicon will not run
    // it without one.
    expect(hasLine('fs.chmodSync(exe, 0o755);')).toBe(true);
  });
});

describe('the codesign probe', () => {
  it('asks the absolute path, not whatever is on PATH', () => {
    // micromamba spawns /usr/bin/codesign by absolute path and THROWS if it
    // fails, mid-transaction. Apple Silicon refuses to load an unsigned arm64
    // binary at all, so this is checked before anything is installed.
    expect(mac.codesignProbeArgs()[0]).toBe('/usr/bin/codesign');
  });
});
