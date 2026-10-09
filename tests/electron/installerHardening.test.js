/**
 * The failure paths.
 *
 * Every test here stands for one way the setup was found to lose a user on a
 * machine where the happy path would have worked: a full disk discovered after
 * the download, a working graphics card reported as absent, a stalled proxy
 * reported as progress, a repair that gutted a live installation. The happy
 * path has one shape; these have twelve, and they are the ones nobody sees
 * until a tester is sitting in front of them.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const installer = await import(
  path.resolve(import.meta.dirname, '..', '..', 'electron', 'setup', 'installer.js')
).then((m) => m.default || m);

const endpointsModule = await import(
  path.resolve(import.meta.dirname, '..', '..', 'electron', 'update_endpoints.js')
).then((m) => m.default || m);

const {
  pipIndexArgs, measureFree, chooseRecommendation, verifyCudaFromOutput,
  classify, CODES, tarExe, GPU_BYTES_NEEDED, CPU_BYTES_NEEDED, redirectVerdict,
  localPackage, compareTagsDesc,
} = installer;

const SOURCE = fs.readFileSync(
  path.resolve(import.meta.dirname, '..', '..', 'electron', 'setup', 'installer.js'),
  'utf8',
);

// Every CUDA call below names a platform. These describe rules that exist on
// Windows and Linux and deliberately do NOT on a Mac, where
// chooseRecommendation answers "Orienta uses the processor on this Mac" and
// driverMeetsCuda12 is false for any driver string. Called bare they asked the
// HOST, so the first macOS run failed five of them for the Mac logic working
// exactly as designed.
const PC = 'win32';

// --------------------------------------------------------------------------
// disk
// --------------------------------------------------------------------------

describe('the volume pip actually writes to', () => {
  it('does not let pip cache a second copy onto the volume we measured', () => {
    // pip's HTTP cache defaults to %LOCALAPPDATA%\pip\Cache, which is on the
    // same drive as the install. Without --no-cache-dir the peak is the
    // download PLUS the tree, so a machine that clears the 9 GB check by two
    // gigabytes still dies part-way through an 8 GB install.
    expect(pipIndexArgs('gpu')).toContain('--no-cache-dir');
    expect(pipIndexArgs('cpu')).toContain('--no-cache-dir');
  });
});

describe('a measurement that failed is not a measurement of zero', () => {
  it('reports whether it could read the drive at all', () => {
    const real = measureFree(process.cwd());
    expect(real.known).toBe(true);
    expect(real.bytes).toBeGreaterThan(0);

    const impossible = measureFree(path.join('Q:', 'no', 'such', 'volume'));
    expect(impossible.known).toBe(false);
    expect(impossible.bytes).toBe(0);
  });

  it('says "could not read" instead of "only 0.0 GB free"', () => {
    // A redirected profile on a UNC share fails statfs. Telling that user
    // their 500 GB disk is full sends them to delete files for an hour.
    const verdict = chooseRecommendation(
      { present: false }, 0, /* freeKnown */ false, PC);
    expect(verdict.blocked).toBe(true);
    expect(verdict.reason).not.toMatch(/0\.0 GB free/);
    expect(verdict.reason).toMatch(/could not be read/i);
  });
});

describe('the disk floor for the version actually chosen', () => {
  it('is checked before anything is downloaded, not discovered inside pip', () => {
    // `blocked` only means "below the CPU floor", and the wizard is allowed to
    // offer GPU against advice — so runSetup has to test the floor for the mode
    // it was given. Pinned at the source, because the alternative is an
    // eight-gigabyte integration test.
    const body = SOURCE.slice(SOURCE.indexOf('async function runSetup'));
    // `cpuBytesNeeded()` rather than the constant since macOS: the conda
    // environment and its package cache exist at the same time, so the floor
    // there is a different number. The check itself is unchanged.
    const floorCheck = body.indexOf('mode === \'gpu\' ? GPU_BYTES_NEEDED : cpuBytesNeeded()');
    const firstDownload = body.indexOf('downloadVerified(');
    expect(floorCheck).toBeGreaterThan(-1);
    expect(floorCheck).toBeLessThan(firstDownload);
  });

  it('offers the smaller install when only the GPU one does not fit', () => {
    const verdict = chooseRecommendation(
      { present: true, cudaCapable: true, driver: 560 },
      CPU_BYTES_NEEDED + 1e9, true, PC);
    expect(verdict.blocked).toBeFalsy();
    expect(verdict.recommendation).toBe('cpu');
  });
});

// --------------------------------------------------------------------------
// the graphics card
// --------------------------------------------------------------------------

describe('a driver that is installed but broken', () => {
  it('is not reported as "no graphics card found"', () => {
    // nvidia-smi present and failing is a stale driver after a Windows update,
    // or a laptop in hybrid mode. Calling that "no card" routes a working
    // RTX 4090 to the CPU and costs the factor of 16 silently.
    const verdict = chooseRecommendation(
      { present: false, driverBroken: true, detail: 'NVML init failed' },
      500e9, true, PC);
    expect(verdict.recommendation).toBe('cpu');
    expect(verdict.reason).toMatch(/not responding|driver/i);
    expect(verdict.reason).not.toMatch(/No NVIDIA graphics card found/);
  });

  it('still says "no card" when nvidia-smi simply is not there', () => {
    const verdict = chooseRecommendation({ present: false }, 500e9, true, PC);
    expect(verdict.reason).toMatch(/No NVIDIA graphics card found/);
  });
});

describe("reading torch's answer", () => {
  it('takes the last line, not the first thing printed to stdout', () => {
    // torch, or something it imports, may warn on stdout first. Parsing that
    // warning as a version reports a working card as broken.
    const noisy = 'UserWarning: TensorFloat32 is available\n2.11.0+cu126 True NVIDIA RTX 4070\n';
    expect(verifyCudaFromOutput(noisy).ok).toBe(true);
  });

  it('still refuses a CPU wheel on the last line', () => {
    expect(verifyCudaFromOutput('some warning\n2.11.0+cpu False \n').ok).toBe(false);
  });
});

// --------------------------------------------------------------------------
// classification — the user can only act on an error that names itself
// --------------------------------------------------------------------------

describe('classifying a failure', () => {
  it('calls a full disk a space problem, not an unexpected one', () => {
    // Running out of disk is the ONE failure a user can fix in a minute.
    expect(classify(Object.assign(new Error('write ENOSPC'), { code: 'ENOSPC' })))
      .toBe(CODES.blocked);
    expect(classify(new Error('There is not enough space on the disk')))
      .toBe(CODES.blocked);
  });

  it('calls a stalled download a timeout, not a mystery', () => {
    expect(classify(Object.assign(new Error('stopped responding'), { code: 'stalled' })))
      .toBe(CODES.timeout);
  });

  it('has a distinct code for each thing the wizard must say differently', () => {
    for (const key of ['blocked', 'network', 'notFound', 'rateLimited', 'checksum',
      'timeout', 'inUse', 'package', 'pip', 'cuda', 'interrupted', 'unexpected']) {
      expect(typeof CODES[key], key).toBe('string');
    }
    const values = Object.values(CODES);
    expect(new Set(values).size).toBe(values.length);
  });
});

// --------------------------------------------------------------------------
// the record of a failure has to outlive the failure
// --------------------------------------------------------------------------

describe('.install_incomplete', () => {
  it('is not deleted in the finally block', () => {
    // Deleting it there erases the record of every graceful failure at the
    // moment it happens — so a setup that dies in step 4 leaves a complete
    // python/ and no marker, and the next launch says "the interpreter is
    // missing", which is false.
    const finallyBlock = SOURCE.slice(SOURCE.lastIndexOf('} finally {'));
    expect(finallyBlock).not.toMatch(/rmSync\(incomplete/);
    expect(SOURCE).toMatch(/rmSync\(incomplete/);   // it IS removed, on success
  });

  it('records which step was running', () => {
    for (const step of ['probe', 'python', 'runtime', 'packages']) {
      expect(SOURCE).toMatch(new RegExp(`writeMarker\\(incomplete, \\{ mode, step: '${step}'`));
    }
  });
});

// --------------------------------------------------------------------------
// never destroy an install in order to repair it
// --------------------------------------------------------------------------

describe('replacing an existing interpreter', () => {
  it('renames the old directory aside instead of deleting it in place', () => {
    // A recursive delete is depth-first: with python.exe locked it removes
    // site-packages and then throws, leaving a gutted tree that still reads as
    // "installed". Renaming fails atomically instead, having changed nothing.
    const step = SOURCE.slice(SOURCE.indexOf("const destination = path.join(home, 'python')"),
      SOURCE.indexOf('fs.renameSync(source, destination)'));
    expect(step).toMatch(/renameSync\(destination, aside\)/);
    expect(step).not.toMatch(/rmrf\(destination\)/);
  });
});

// --------------------------------------------------------------------------
// the GPU→CPU fallback
// --------------------------------------------------------------------------

describe('falling back to the processor version', () => {
  it('removes the packages pip will not remove by itself', () => {
    // pip replaces torch+cu126 with torch+cpu, but leaves cupy and the
    // nvidia-* wheels: ~5 GB for ever, with .install_mode reading `cpu` so
    // nothing downstream knows to clean up. Worse, torch+cpu beside a working
    // cupy is the configuration behind the 2026-08-03 dispatcher bug.
    expect(SOURCE).toMatch(/CUDA_ONLY_PACKAGES/);
    expect(SOURCE).toMatch(/'cupy-cuda12x'/);
    expect(SOURCE).toMatch(/pip', 'uninstall', '-y', \.\.\.CUDA_ONLY_PACKAGES/);
  });
});

// --------------------------------------------------------------------------
// what the earlier version of this suite asserted, and should have
// --------------------------------------------------------------------------

describe('tarExe', () => {
  it('refuses rather than guessing when SystemRoot is unset', () => {
    // The earlier test set SystemRoot and then asserted the result started
    // with the value it had just set — true for any implementation that
    // concatenates. The real contract is that it throws.
    //
    // Asked of `win32` explicitly rather than of the host: off Windows,
    // `tarExe()` answers /usr/bin/tar and does not throw, so the host-bound
    // version would have FAILED — not skipped — the first time this suite ran
    // on the macOS runner, and the port would have been blamed for it. The
    // environment is passed in too, so nothing has to be deleted and restored.
    const platform = createRequire(import.meta.url)('../../electron/platform.js');
    expect(() => platform.tarExe('win32', {})).toThrow(/SystemRoot/);
  });

  it('leaves the environment exactly as it found it', () => {
    // Restoring with `process.env.X = saved` assigns the STRING "undefined"
    // when the variable was unset, which then leaks into every later test.
    expect(process.env.SystemRoot === undefined
      || typeof process.env.SystemRoot === 'string').toBe(true);
    expect(process.env.SystemRoot).not.toBe('undefined');
  });
});

describe('downloads', () => {
  it('gives up on a connection that has gone silent', () => {
    expect(SOURCE).toMatch(/STALL_TIMEOUT_MS/);
    const fetchBody = SOURCE.slice(SOURCE.indexOf('function fetchToFile'));
    expect(fetchBody).toMatch(/setTimeout\(/);
  });

  describe('the redirect policy', () => {
    // Tested through `redirectVerdict`, a pure function, rather than by
    // grepping the source. The previous version of this test asserted that the
    // file CONTAINED "redirect: 'manual'" -- which it did, while the code
    // around it cancelled every download with "Redirect was cancelled",
    // because in Electron 'manual' means the hop is refused unless
    // followRedirect() is called. The test was green and the installer could
    // not install anything.
    const from = 'https://github.com/SeSam-MUL/Orienta/releases/download/v1/x.zip';

    it('follows the cross-host hop GitHub actually sends', () => {
      expect(redirectVerdict(from, 'https://objects.githubusercontent.com/abc', 1).ok)
        .toBe(true);
    });

    it('follows a relative hop', () => {
      expect(redirectVerdict(from, '/other/path', 1).ok).toBe(true);
    });

    it('refuses a drop to plain text', () => {
      // Over http an intercepted checksum document and an intercepted payload
      // agree with each other, so the digest proves nothing about either.
      const verdict = redirectVerdict(from, 'http://mirror.example/x.zip', 1);
      expect(verdict.ok).toBe(false);
      expect(verdict.reason).toMatch(/encrypted/);
    });

    it('allows plain text to loopback, which is how this is tested at all', () => {
      expect(redirectVerdict('http://127.0.0.1:9500/a', 'http://127.0.0.1:9500/b', 1).ok)
        .toBe(true);
    });

    it('stops going round in circles', () => {
      expect(redirectVerdict(from, 'https://github.com/loop', 6).ok).toBe(false);
      expect(redirectVerdict(from, 'https://github.com/loop', 5).ok).toBe(true);
    });

    it('is wired to followRedirect, not to a response handler', () => {
      // The one thing a pure function cannot cover: that the verdict is
      // actually applied to Electron's redirect event.
      const wiring = SOURCE.slice(SOURCE.indexOf('function requestOnce'));
      expect(wiring).toMatch(/request\.on\('redirect'/);
      expect(wiring).toMatch(/request\.followRedirect\(\)/);
      expect(wiring).toMatch(/redirectVerdict\(/);
    });
  });

  it('blames a short body on the connection, not on antivirus', () => {
    const fetchBody = SOURCE.slice(SOURCE.indexOf('function fetchToFile'));
    expect(fetchBody).toMatch(/done !== total/);
  });
});

// --------------------------------------------------------------------------
// the evidence a failure is made of
// --------------------------------------------------------------------------

describe('the setup log', () => {
  it('writes each line as it arrives, not through a buffered stream', () => {
    // The line that matters most is the last one before a crash, and a stream
    // may still be holding it.
    const body = SOURCE.slice(SOURCE.indexOf('function logSetupLine'));
    expect(body).toMatch(/appendFileSync/);
    expect(body.slice(0, 400)).not.toMatch(/createWriteStream/);
  });

  it('actually receives every child process\'s output', () => {
    // Each of the four children used to pass `touch` — a function that updates
    // a timestamp and drops the line. Twenty minutes of pip output went
    // nowhere, so a failed install produced eleven words and no evidence.
    const body = SOURCE.slice(SOURCE.indexOf('async function runSetup'));
    expect(body).not.toMatch(/onLine: touch/);
    expect((body.match(/record\b/g) || []).length).toBeGreaterThanOrEqual(4);
  });

  it('attaches the log path to EVERY failure, not to two of twelve', async () => {
    // Run it for real. `electron` is required lazily, so in a plain Node
    // process the resolve step throws "Cannot find module 'electron'" -- which
    // lands in the outer catch and exercises exactly the path that has to
    // carry the log path out.
    //
    // Set per-return, this reached two of the twelve failure returns, so
    // "Open the log" was hidden for the other ten -- including `package` and
    // `cuda`, the two where the log is the whole story.
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-setup-'));
    try {
      const result = await installer.runSetup(
        { mode: 'cpu', home, releaseTag: 'v0.0.0-test' }, () => {});
      expect(result.ok).toBe(false);
      expect(result.logFile, 'no log path on the failure').toBeTruthy();
      expect(result.logFile).toContain('orienta-setup.log');
      expect(typeof result.log).toBe('string');
      expect(fs.existsSync(result.logFile), 'the log was never written').toBe(true);
    } finally {
      fs.rmSync(home, { recursive: true, force: true });
    }
  });

  it('bounds the tail, because an error message is read by a person', () => {
    expect(SOURCE).toMatch(/tail\.length > 50/);
  });

  it('never fails the install because it could not be written', () => {
    const body = SOURCE.slice(SOURCE.indexOf('function logSetupLine'),
      SOURCE.indexOf('function logSetupLine') + 500);
    expect(body).toMatch(/catch \{/);
  });
});

describe('killing the setup', () => {
  it('kills the process tree, because pip spawns its own workers', () => {
    // child.kill() ends one pid on Windows. This project settled the question
    // for the backend in 2026-08; the setup needs the same answer.
    // The plan for the kill now lives in electron/platform.js, so this asks
    // the resolver rather than grepping for `/T` in this file's source.
    const resolver = createRequire(import.meta.url)('../../electron/platform.js');
    expect(resolver.killTree(4242, 'win32'))
      .toEqual({ kind: 'command', command: 'taskkill', args: ['/PID', '4242', '/T', '/F'] });
    // off Windows the children are a process group, reached by the negative pid
    expect(resolver.killTree(4242, 'linux')).toEqual({ kind: 'signal', target: -4242, signal: 'SIGKILL' });
    const killer = SOURCE.slice(SOURCE.indexOf('runSetup.killChildren'));
    expect(killer).toMatch(/killTree/);
  });
});

// --------------------------------------------------------------------------
// a package that is already on this machine
// --------------------------------------------------------------------------

describe('a runtime package lying beside the installer', () => {
  const NAME = 'orienta-runtime-v9.9.9.zip';
  let dir;

  beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-pkg-')); });
  afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

  it('is not used unless its checksum file is there too', () => {
    // A local file is not automatically a trustworthy one: it arrives on the
    // same USB stick as everything else in a shared lab. Accepting a package
    // with no digest would put the weakest check exactly where there is the
    // least supervision.
    fs.writeFileSync(path.join(dir, NAME), 'x');
    expect(localPackage(dir, 'v9.9.9')).toBe(null);

    fs.writeFileSync(path.join(dir, `${NAME}.sha256`), 'x');
    expect(localPackage(dir, 'v9.9.9')).not.toBe(null);
  });

  it('matches the tag, not just the name', () => {
    fs.writeFileSync(path.join(dir, NAME), 'x');
    fs.writeFileSync(path.join(dir, `${NAME}.sha256`), 'x');
    // A package for a different version is not this version's package, and
    // installing it would produce a tree whose VERSION disagrees with the
    // shell that asked for it.
    expect(localPackage(dir, 'v1.2.3')).toBe(null);
  });

  it('finds nothing in an empty folder', () => {
    expect(localPackage(dir, 'v9.9.9')).toBe(null);
  });

  it('is verified with the same parser and the same refusal as a download', () => {
    // Same digest, same parser, same code on mismatch -- the offline path must
    // not be a second, weaker implementation of the check.
    const body = SOURCE.slice(SOURCE.indexOf('if (beside) {', SOURCE.indexOf("emit('runtime'")));
    expect(body.slice(0, 900)).toMatch(/parseSha256Document\(/);
    expect(body.slice(0, 900)).toMatch(/sha256File\(/);
    expect(body.slice(0, 900)).toMatch(/CODES\.checksum/);
  });
});

// --------------------------------------------------------------------------
// which version gets installed
// --------------------------------------------------------------------------

describe('ordering release tags', () => {
  it('knows 0.10 is newer than 0.4', () => {
    // The one bug every project meets exactly once, at the 0.9 -> 0.10
    // boundary: lexically "v0.4.0" sorts above "v0.10.0", because "4" beats
    // "1" one character in. By the time it shows up it has been wrong for a
    // while.
    expect(['v0.4.0', 'v0.10.0'].sort(compareTagsDesc)).toEqual(['v0.10.0', 'v0.4.0']);
    expect(['v0.9.0', 'v0.10.0'].sort(compareTagsDesc)).toEqual(['v0.10.0', 'v0.9.0']);
    expect(['v1.2.3', 'v1.2.10'].sort(compareTagsDesc)).toEqual(['v1.2.10', 'v1.2.3']);
  });

  it('puts a release candidate below its final release', () => {
    expect(['v1.0.0-rc1', 'v1.0.0'].sort(compareTagsDesc)).toEqual(['v1.0.0', 'v1.0.0-rc1']);
  });

  it('sorts something unparseable last instead of throwing', () => {
    // This reads a folder listing, not a contract.
    expect(['wip-something', 'v2.0.0'].sort(compareTagsDesc)[0]).toBe('v2.0.0');
    expect(() => compareTagsDesc('', null)).not.toThrow();
  });
});

describe('the installer is not pinned to its own build number', () => {
  it('asks the shell for no particular version by default', () => {
    // `v${app.getVersion()}` pins a 100 MB installer to exactly one release,
    // so every copy of it dies the day that release is superseded or removed
    // -- with a 404 naming a tag the user never typed. A bootstrap installer
    // has to stay valid while the thing it installs moves on.
    const main = fs.readFileSync(
      path.resolve(import.meta.dirname, '..', '..', 'electron', 'main.js'), 'utf8');
    expect(main).not.toMatch(/releaseTag:\s*`v\$\{app\.getVersion\(\)\}`/);
    expect(main).not.toMatch(/runtimeTagForThisShell/);
    expect(main).toMatch(/let requestedTag = null/);
  });

  it('falls through to the release list when there is no "latest"', () => {
    // GitHub excludes pre-releases from /releases/latest, so a repository
    // whose only releases are pre-releases has no latest at all. Reporting
    // that as "not published yet" would be wrong twice over.
    const body = SOURCE.slice(SOURCE.indexOf('async function resolveRelease'));
    expect(body.slice(0, 1800)).toMatch(/releaseListUrl\(\)/);
    expect(body.slice(0, 1800)).toMatch(/err\.status !== 404/);
  });
});

describe('a package folder the user pointed at', () => {
  let dir;
  beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-pick-')); });
  afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

  const put = (tag) => {
    fs.writeFileSync(path.join(dir, `orienta-runtime-${tag}.zip`), 'x');
    fs.writeFileSync(path.join(dir, `orienta-runtime-${tag}.zip.sha256`), 'x');
  };

  it('answers "newest" from the files that are there', () => {
    // With no version pinned there is no filename to build, so the tag has to
    // be read off the file rather than demanded in advance — otherwise the
    // offline path only works for someone who already knows the version.
    put('v0.4.0'); put('v0.10.0'); put('v0.9.0');
    expect(localPackage(os.tmpdir(), null, [dir]).tag).toBe('v0.10.0');
  });

  it('is searched before anywhere else', () => {
    put('v0.4.0');
    expect(localPackage(os.tmpdir(), 'v0.4.0', [dir]).dir).toBe(dir);
  });
});

// --------------------------------------------------------------------------
// the copy that ships inside the installer
// --------------------------------------------------------------------------

describe('the runtime that came in the box', () => {
  it('is searched before anywhere else', () => {
    // It is the one location that cannot be wrong: nobody had to put it there,
    // no folder can have been moved out from under it, and it works on a
    // machine that cannot reach GitHub. Everything after it exists only for
    // installing a DIFFERENT version than the one that shipped.
    const body = SOURCE.slice(SOURCE.indexOf('function localPackageSearch'));
    const resources = body.indexOf('process.resourcesPath');
    const extras = body.indexOf('...extraDirs');
    const home = body.indexOf('\n    home,');
    expect(resources).toBeGreaterThan(-1);
    expect(resources).toBeLessThan(extras);
    expect(resources).toBeLessThan(home);
  });

  it('is verified, not trusted for being ours', () => {
    // Same digest, same parser, same refusal as a download. A file that came
    // with the installer is one an antivirus may have quarantined and restored
    // in part, or a half-finished copy off a USB stick.
    const body = SOURCE.slice(SOURCE.indexOf('if (beside) {', SOURCE.indexOf("emit('runtime'")));
    expect(body.slice(0, 900)).toMatch(/parseSha256Document\(/);
    expect(body.slice(0, 900)).toMatch(/CODES\.checksum/);
  });

  it('ships through electron-builder, from a folder the build stages', () => {
    const pkg = JSON.parse(fs.readFileSync(
      path.resolve(import.meta.dirname, '..', '..', 'frontend', 'package.json'), 'utf8'));
    const extra = pkg.build.extraResources || [];
    expect(extra.some((e) => e.from === 'bundled-runtime')).toBe(true);

    // `scripts/build_release.py` is not part of the public repository, so the
    // two assertions about its staging step can only run in the development
    // tree. Skipping beats deleting: in that tree the guard still bites.
    const builderPath = path.resolve(import.meta.dirname, '..', '..', 'scripts', 'build_release.py');
    if (!fs.existsSync(builderPath)) return;
    const builder = fs.readFileSync(builderPath, 'utf8');
    expect(builder).toMatch(/def stage_bundled_runtime/);
    // Cleared first: a package from an earlier version shipping beside the
    // current one would be found by name and could be chosen instead of it.
    expect(builder).toMatch(/for old in staging\.glob/);
  });
});

describe('when nothing can be found at all', () => {
  it('says which folders were searched', () => {
    // Otherwise the screen talks about an unpublished GitHub release to
    // someone whose entire task was to put a file in a folder, and never says
    // which folders it read.
    expect(SOURCE).toMatch(/no Orienta package found in:/);
    expect(SOURCE).toMatch(/searchedDirs/);
  });

  it('resolves the known folders through Windows, not through %USERPROFILE%', () => {
    // path.join(USERPROFILE, 'Downloads') is a guess, and it is wrong on every
    // machine where OneDrive has taken the known folders over — the default on
    // a managed university laptop, which is exactly the machine most likely to
    // be handed a file and told where to put it.
    const main = fs.readFileSync(
      path.resolve(import.meta.dirname, '..', '..', 'electron', 'main.js'), 'utf8');
    expect(main).toMatch(/function knownDownloadFolders/);
    expect(main).toMatch(/app\.getPath\(name\)/);
    expect(main).toMatch(/knownFolders: knownDownloadFolders\(\)/);
  });
});

// --------------------------------------------------------------------------
// the bundled copy must not become the pin it replaced
// --------------------------------------------------------------------------

describe('a local copy is a fallback, not a decision', () => {
  it('still asks what the newest release is when no version was pinned', () => {
    // Taking the bundled package merely because it exists re-creates, by a
    // different route, the fault that was just removed: an .exe kept for a
    // year would install a year-old Orienta for ever and never once ask.
    const body = SOURCE.slice(SOURCE.indexOf('async function runSetupInner'));
    const guard = body.indexOf('if (!resume && beside && !releaseTag)');
    expect(guard, 'the bundled copy is taken without ever asking').toBeGreaterThan(-1);
    expect(body.slice(guard, guard + 900)).toMatch(/resolveRelease\(null\)/);
    expect(body.slice(guard, guard + 900)).toMatch(/compareTagsDesc\(/);
  });

  it('keeps the local copy when the network cannot answer', () => {
    // Which is the whole reason it is there. An unreachable GitHub is not a
    // failure on this path and must not be reported as one.
    const body = SOURCE.slice(SOURCE.indexOf('if (!resume && beside && !releaseTag)'));
    expect(body.slice(0, 1400)).toMatch(/catch \(err\)/);
    expect(body.slice(0, 1400)).toMatch(/could not check for a newer release/);
  });

  it('does not second-guess a version the user pinned', () => {
    // `!releaseTag` — asking "is something newer available?" after someone
    // deliberately chose an older version would be overriding them.
    expect(SOURCE).toMatch(/if \(!resume && beside && !releaseTag\)/);
  });
});

// --------------------------------------------------------------------------
// where the gigabytes go
// --------------------------------------------------------------------------

describe('the data folder', () => {
  const endpoints = fs.readFileSync(
    path.resolve(import.meta.dirname, '..', '..', 'electron', 'update_endpoints.js'), 'utf8');

  it('can be chosen, and the choice outlives an update', () => {
    // Not in the Orienta home (circular) and not in the application directory,
    // which a reinstall deletes. Asked of the running code rather than of the
    // source text: the answer now comes from electron/platform.js, and a test
    // that greps for `process.env.APPDATA` would have gone red on Windows for
    // a change that moved nothing.
    expect(endpoints).toMatch(/function homePointerFile/);
    // Asked of Windows by name, with Windows' own variables supplied. Calling
    // homePointerFile() bare resolved against the HOST: on Linux it answers an
    // XDG path and process.env.APPDATA is undefined, so this asserted
    // `startsWith(undefined)` -- a Windows rule tested only where it is
    // already true.
    const WIN = {
      platformName: 'win32',
      env: { APPDATA: 'C:\\Users\\u\\AppData\\Roaming',
             LOCALAPPDATA: 'C:\\Users\\u\\AppData\\Local' },
    };
    const pointer = endpointsModule.homePointerFile(WIN);
    expect(pointer.startsWith(WIN.env.APPDATA)).toBe(true);
    // Components, not a separator. homePointerFile joins with the HOST's path
    // module, so asked for win32 from Linux it answers
    // `C:\Users\u\AppData\Roaming/Orienta/home.txt` -- mixed. That is not a
    // defect: in production the win32 branch only ever runs on Windows, where
    // the join is right. It only means the separator is the host's to choose,
    // so the contract worth asserting is which FOLDER and which NAME, which is
    // what this now does.
    expect(pointer.split(/[\\/]/).slice(-2)).toEqual(['Orienta', 'home.txt']);
    // the pointer must not live inside the folder it points at
    expect(pointer.startsWith(WIN.env.LOCALAPPDATA)).toBe(false);
  });

  it('is proved writable before it is recorded', () => {
    // Recording a folder that cannot be written hands Chromium an unusable
    // path on the next launch, and this project has already seen what that
    // does: the application does not start at all, with no window and no log.
    const body = endpoints.slice(endpoints.indexOf('function setOrientaHome'));
    const probe = body.indexOf('.writable');
    const record = body.indexOf('writeFileSync(file');
    expect(probe).toBeGreaterThan(-1);
    expect(probe).toBeLessThan(record);
  });

  it('lets the environment variable win', () => {
    // Scripts, CI and the test harness set it, and a recorded preference must
    // not silently override an explicit instruction.
    // `homePointerFile(` without the closing paren: it takes arguments now
    // (the platform and the environment, so the data-folder tests can run on
    // the macOS and Linux runners), and matching the empty call made this
    // assertion silently compare against -1.
    //
    // The behaviour itself is covered directly in dataFolder.test.js ("lets
    // ORIENTA_HOME win over a recorded choice"); this one pins the ORDER in
    // the source, which is what decides it.
    const body = endpoints.slice(endpoints.indexOf('function orientaHome'));
    expect(body.indexOf('homePointerFile(')).toBeGreaterThan(-1);
    expect(body.indexOf('HOME_ENV')).toBeLessThan(body.indexOf('homePointerFile('));
  });

  it('survives an unreadable or relative pointer', () => {
    // This runs at module load. A bad byte in a preference file must not be
    // the reason the application cannot start.
    const body = endpoints.slice(endpoints.indexOf('function orientaHome'),
      endpoints.indexOf('function setOrientaHome'));
    expect(body).toMatch(/catch \{/);
    expect(body).toMatch(/path\.isAbsolute\(recorded\)/);
  });

  it('puts Orienta in its own subfolder of whatever was picked', () => {
    // Someone who chooses D:\ must not get python/, runtime/ and logs/ across
    // the root of their drive — and an uninstall has to know what is ours.
    const main = fs.readFileSync(
      path.resolve(import.meta.dirname, '..', '..', 'electron', 'main.js'), 'utf8');
    const body = main.slice(main.indexOf("handleSetup('setup:pickHome'"));
    expect(body.slice(0, 900)).toMatch(/path\.join\(answer\.filePaths\[0\], 'Orienta'\)/);
  });
});

describe('the application folder', () => {
  it('is NOT a free choice — the data folder is', () => {
    // An unelevated installer with a free directory page ends mid-install in
    // "Error opening file for writing" for anyone who picks Program Files.
    // The "for all users" page is the proper route there; see
    // tests/test_packaging_config.py for the full reasoning.
    const pkg = JSON.parse(fs.readFileSync(
      path.resolve(import.meta.dirname, '..', '..', 'frontend', 'package.json'), 'utf8'));
    expect(pkg.build.nsis.allowToChangeInstallationDirectory).toBe(false);
  });
});

// --------------------------------------------------------------------------
// never set up inside somebody else's folder
// --------------------------------------------------------------------------

describe('a data folder that already holds other files', () => {
  let dir;
  beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-foreign-')); });
  afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

  it('is accepted when everything in it is Orienta\'s own', () => {
    // electron\ and logs\ are written by the shell at startup, BEFORE setup
    // runs, so a fresh home is never quite empty.
    for (const name of ['electron', 'logs', 'python', 'runtime', 'python.old-17']) {
      fs.mkdirSync(path.join(dir, name));
    }
    for (const name of ['.orienta-home', '.python_path', '.install_incomplete']) {
      fs.writeFileSync(path.join(dir, name), 'x');
    }
    expect(installer.foreignEntries(dir)).toEqual([]);
  });

  it('accepts what the package sync leaves behind: a repair after an interrupted update must not be refused', () => {
    // The repair wizard runs in this very folder. Without these names it would
    // answer "this folder contains files that are not Orienta's", naming three
    // files the app wrote itself.
    for (const name of ['.packages_lock.json', '.packages_sync.json', '.packages_sync_failed.json']) {
      fs.writeFileSync(path.join(dir, name), '{}');
    }
    expect(installer.foreignEntries(dir)).toEqual([]);
  });

  it('is named, entry by entry, when it is not', () => {
    // Setting up here would write the marker the uninstaller trusts, and a
    // later uninstall would then remove this user's own python\ and logs\.
    fs.mkdirSync(path.join(dir, 'logs'));
    fs.writeFileSync(path.join(dir, 'thesis.docx'), 'x');
    fs.mkdirSync(path.join(dir, 'measurements'));
    expect(installer.foreignEntries(dir).sort()).toEqual(['measurements', 'thesis.docx']);
  });

  it('stops the setup before the marker is written — run, not read', async () => {
    // An earlier version of this test only checked that the check was CALLED
    // before the marker; with the `if` that acts on it removed, it still
    // passed. This one runs the setup.
    fs.writeFileSync(path.join(dir, 'thesis.docx'), 'x');
    const result = await installer.runSetup(
      { mode: 'cpu', home: dir, releaseTag: 'v0.0.0-test' }, () => {});
    expect(result.ok).toBe(false);
    expect(result.code).toBe(CODES.homeNotEmpty);
    expect(result.error).toMatch(/thesis\.docx/);
    expect(fs.existsSync(path.join(dir, '.orienta-home'))).toBe(false);
    expect(fs.existsSync(path.join(dir, 'thesis.docx'))).toBe(true);
  });

  it('goes ahead in a folder that only holds what the shell wrote at startup', async () => {
    fs.mkdirSync(path.join(dir, 'electron'));
    fs.mkdirSync(path.join(dir, 'logs'));
    const result = await installer.runSetup(
      { mode: 'cpu', home: dir, releaseTag: 'v0.0.0-test' }, () => {});
    // It fails later, for want of a network in this process — but not here.
    expect(result.code).not.toBe(CODES.homeNotEmpty);
    expect(fs.existsSync(path.join(dir, '.orienta-home'))).toBe(true);
  });
});
