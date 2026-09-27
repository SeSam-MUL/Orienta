/**
 * The platform resolver, asked for all three operating systems from one.
 *
 * The point of these tests is that nobody here has a Mac or a Linux box: every
 * function takes the platform, the environment and the home directory, so the
 * macOS and Linux answers can be pinned down on Windows. And the Windows
 * answers are pinned against what v0.4.5 shipped, because this resolver exists
 * to move code, not behaviour.
 */
import { describe, it, expect } from 'vitest';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../electron/platform.js');

const WIN_ENV = { LOCALAPPDATA: 'C:\\Users\\t\\AppData\\Local', APPDATA: 'C:\\Users\\t\\AppData\\Roaming', SystemRoot: 'C:\\WINDOWS' };
const NIX_HOME = '/home/t';
const MAC_HOME = '/Users/t';

describe('what we support', () => {
  it('takes Windows and Linux on 64-bit Intel', () => {
    expect(P.supportStatus('win32', 'x64').supported).toBe(true);
    expect(P.supportStatus('linux', 'x64').supported).toBe(true);
  });

  it('takes Apple Silicon from macOS 14 on', () => {
    expect(P.supportStatus('darwin', 'arm64', '23.0.0').supported).toBe(true);
    expect(P.supportStatus('darwin', 'arm64', '25.1.0').supported).toBe(true);
  });

  it('refuses an Intel Mac, and says why', () => {
    const v = P.supportStatus('darwin', 'x64', '23.0.0');
    expect(v.supported).toBe(false);
    expect(v.reason).toMatch(/Apple Silicon/);
    expect(v.reason).toMatch(/PyTorch/);
  });

  it('refuses macOS 13 and older', () => {
    const v = P.supportStatus('darwin', 'arm64', '22.6.0');
    expect(v.supported).toBe(false);
    expect(v.reason).toMatch(/macOS 14/);
  });

  it('refuses a platform we do not build for', () => {
    expect(P.supportStatus('freebsd', 'x64').supported).toBe(false);
  });
});

describe('folders', () => {
  it('Windows keeps exactly what v0.4.5 shipped', () => {
    expect(P.defaultDataFolder('win32', WIN_ENV, 'C:\\Users\\t'))
      .toBe(path.join('C:\\Users\\t\\AppData\\Local', 'Orienta'));
    expect(P.configFolder('win32', WIN_ENV, 'C:\\Users\\t'))
      .toBe(path.join('C:\\Users\\t\\AppData\\Roaming', 'Orienta'));
  });

  it('macOS uses Application Support', () => {
    expect(P.defaultDataFolder('darwin', {}, MAC_HOME))
      .toBe(path.join(MAC_HOME, 'Library', 'Application Support', 'Orienta'));
  });

  it('macOS keeps the pointer OUTSIDE the data folder', () => {
    // setOrientaHome refuses a data folder that overlaps the pointer's folder,
    // so a nested config directory would lock the user out of the default.
    const data = P.defaultDataFolder('darwin', {}, MAC_HOME);
    const conf = P.configFolder('darwin', {}, MAC_HOME);
    expect(conf.startsWith(data + path.sep)).toBe(false);
    expect(data.startsWith(conf + path.sep)).toBe(false);
    expect(conf).not.toBe(data);
  });

  it('macOS stays out of ~/Library/Preferences', () => {
    // Apple reserves it for preference files its own tools manage.
    expect(P.configFolder('darwin', {}, MAC_HOME)).not.toMatch(/Preferences/);
  });

  it('Linux follows XDG, and falls back when it is unset', () => {
    expect(P.defaultDataFolder('linux', { XDG_DATA_HOME: '/x/data' }, NIX_HOME)).toBe(path.join('/x/data', 'Orienta'));
    expect(P.defaultDataFolder('linux', {}, NIX_HOME)).toBe(path.join(NIX_HOME, '.local', 'share', 'Orienta'));
    expect(P.configFolder('linux', { XDG_CONFIG_HOME: '/x/conf' }, NIX_HOME)).toBe(path.join('/x/conf', 'Orienta'));
    expect(P.configFolder('linux', {}, NIX_HOME)).toBe(path.join(NIX_HOME, '.config', 'Orienta'));
  });

  it('keeps data and pointer apart on every platform', () => {
    for (const [plat, env, home] of [['win32', WIN_ENV, 'C:\\Users\\t'], ['darwin', {}, MAC_HOME], ['linux', {}, NIX_HOME]]) {
      expect(P.configFolder(plat, env, home)).not.toBe(P.defaultDataFolder(plat, env, home));
    }
  });
});

describe('where the runtime comes from', () => {
  it('Windows: the pinned python-build-standalone archive of v0.4.5', () => {
    const s = P.runtimeSource('win32', 'x64', {});
    expect(s.kind).toBe('pbs');
    expect(s.file).toBe('cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz');
    expect(s.url).toBe(
      'https://github.com/astral-sh/python-build-standalone/releases/download/20260901/'
      + 'cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz');
  });

  it('Linux: the same release, its own triple', () => {
    const s = P.runtimeSource('linux', 'x64', {});
    expect(s.file).toBe('cpython-3.11.16+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz');
    expect(s.url).toContain('/20260901/');
  });

  it('macOS: micromamba, pinned with its checksum', () => {
    const s = P.runtimeSource('darwin', 'arm64', {});
    expect(s.kind).toBe('conda');
    expect(s.file).toBe('micromamba-osx-arm64');
    expect(s.version).toBe('2.9.0-0');
    expect(s.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(s.url).toContain('micromamba-releases');
  });

  it('every URL stays overridable, for a lab that needs a mirror', () => {
    expect(P.runtimeSource('win32', 'x64', { ORIENTA_PYTHON_URL: 'https://mirror/py.tar.gz' }).url)
      .toBe('https://mirror/py.tar.gz');
    expect(P.runtimeSource('darwin', 'arm64', { ORIENTA_MICROMAMBA_URL: 'https://mirror/mm' }).url)
      .toBe('https://mirror/mm');
  });

  it('refuses a combination we have no build for, instead of guessing', () => {
    expect(() => P.runtimeSource('linux', 'arm64', {})).toThrow(/no Python build/);
  });
});

describe('the rest of the platform answers', () => {
  it('names the interpreter where each platform puts it', () => {
    expect(P.interpreterIn('C:\\O', 'win32')).toBe(path.join('C:\\O', 'python', 'python.exe'));
    expect(P.interpreterIn('/o', 'linux')).toBe(path.join('/o', 'python', 'bin', 'python3'));
    // macOS is NOT the same answer as Linux. It installs from conda-forge --
    // pip wheels of torch, scikit-learn and faiss each carry an OpenMP runtime
    // and the second to initialise aborts the backend -- so micromamba builds
    // a conda environment and there is no `python/` directory at all. The
    // first version of this file answered `<home>/python/bin/python3` here,
    // a path that would never exist on a Mac.
    expect(P.interpreterIn('/o', 'darwin'))
      .toBe(path.join(P.condaPrefixIn('/o'), 'bin', 'python'));
  });

  it('keeps the conda environment out of the package cache it deletes', () => {
    // The cache is `<root>/pkgs` and gets removed after the install. If the
    // environment lived under the same root, that cleanup could reach it.
    expect(P.condaPrefixIn('/o').startsWith(P.condaRootIn('/o'))).toBe(false);
  });

  it('uses Windows own bsdtar, and /usr/bin/tar elsewhere', () => {
    expect(P.tarExe('win32', WIN_ENV)).toBe(path.join('C:\\WINDOWS', 'System32', 'tar.exe'));
    expect(P.tarExe('darwin', {})).toBe('/usr/bin/tar');
    expect(P.tarExe('linux', {})).toBe('/usr/bin/tar');
  });

  it('still refuses to guess when SystemRoot is missing on Windows', () => {
    expect(() => P.tarExe('win32', {})).toThrow(/SystemRoot/);
  });

  it('keeps the two driver numbering schemes apart', () => {
    // 527.41 is a Windows driver version; the same CUDA release is 525.60.13
    // on Linux. Using the Windows number there rejects working drivers.
    expect(P.minDriverDisplay('win32')).toBe('527.41');
    expect(P.minDriverDisplay('linux')).toBe('525.60.13');
    expect(P.minDriverDisplay('darwin')).toBeNull();
    expect(P.driverMeetsCuda12('527.41', 'win32')).toBe(true);
    expect(P.driverMeetsCuda12('527.40', 'win32')).toBe(false);
    expect(P.driverMeetsCuda12('525.60.13', 'linux')).toBe(true);
    expect(P.driverMeetsCuda12('525.60.12', 'linux')).toBe(false);
  });

  it('compares versions by component, and pads the shorter one', () => {
    expect(P.compareVersions([525, 105], [525, 60])).toBe(1);
    expect(P.compareVersions([527, 41], [527, 41, 0])).toBe(0);
    expect(P.compareVersions([1], [1, 0, 1])).toBe(-1);
    expect(P.parseDriverVersion('525.60.13')).toEqual([525, 60, 13]);
    expect(P.parseDriverVersion('junk')).toBeNull();
  });

  it('refuses an Intel Mac asset instead of inventing one', () => {
    expect(() => P.runtimeSource('darwin', 'x64', {})).toThrow(/no micromamba build/);
  });

  it('pins the folders that Windows parity depends on, fallbacks included', () => {
    expect(P.defaultDataFolder('win32', {}, 'C:\Users\t'))
      .toBe(path.join('C:\Users\t', '.local', 'share', 'Orienta'));
    expect(P.configFolder('win32', {}, 'C:\Users\t'))
      .toBe(path.join('C:\Users\t', '.config', 'Orienta'));
    expect(P.configFolder('darwin', {}, MAC_HOME))
      .toBe(path.join(MAC_HOME, 'Library', 'Application Support', 'Orienta Settings'));
  });

  it('knows there is no CUDA on a Mac', () => {
    expect(P.canUseCuda('darwin')).toBe(false);
    expect(P.canUseCuda('win32')).toBe(true);
    expect(P.canUseCuda('linux')).toBe(true);
  });

  it('kills the whole tree, the way each platform can', () => {
    expect(P.killTree(1234, 'win32')).toEqual({ kind: 'command', command: 'taskkill', args: ['/PID', '1234', '/T', '/F'] });
    expect(P.killTree(1234, 'linux')).toEqual({ kind: 'signal', target: -1234, signal: 'SIGKILL' });
    // a negative pid must stay negative, not become positive again
    expect(P.killTree(-1234, 'darwin').target).toBe(-1234);
  });
});

describe('where the program lives, and how it restarts', () => {
  it('forbids the whole .app bundle on macOS, not just the folder inside it', () => {
    // Orienta.app/Contents/MacOS/Orienta: a data folder chosen inside the
    // bundle would be dragged to the Trash with the app, and nobody would
    // connect the two.
    const roots = P.programRoots('/Applications/Orienta.app/Contents/MacOS/Orienta', 'darwin');
    expect(roots).toContain('/Applications/Orienta.app');
    expect(roots).toContain('/Applications/Orienta.app/Contents/MacOS');
  });

  it('forbids the executable directory on Windows and Linux', () => {
    expect(P.programRoots('/opt/orienta/orienta', 'linux')).toEqual(['/opt/orienta']);
    expect(P.programRoots('C:/Program Files/Orienta/Orienta.exe', 'win32'))
      .toEqual([path.dirname('C:/Program Files/Orienta/Orienta.exe')]);
  });

  it('restarts the AppImage file, not the temporary mount it runs from', () => {
    // The mount disappears with the process, so relaunching execPath starts
    // nothing at all.
    expect(P.relaunchOptions({ APPIMAGE: '/home/u/Orienta.AppImage' }))
      .toEqual({ execPath: '/home/u/Orienta.AppImage', args: [] });
  });

  it('leaves the default alone everywhere else', () => {
    expect(P.relaunchOptions({})).toBeUndefined();
    expect(P.relaunchOptions({ APPIMAGE: '   ' })).toBeUndefined();
  });
});
