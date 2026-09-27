/**
 * Choosing where the gigabytes go — tested by DOING it.
 *
 * The first version of this feature shipped with `fs` never imported in
 * update_endpoints.js. `orientaHome()` reads the pointer inside a try, so the
 * ReferenceError landed in the catch and every launch silently fell back to
 * the default folder, while `setOrientaHome()` threw on every call. The tests
 * written with it searched the source for the right-looking strings and all
 * passed. These run the functions.
 *
 * APPDATA and LOCALAPPDATA are supplied to the module as arguments, pointing
 * into a temporary directory, so nothing here can touch the real preference
 * on this machine -- and nothing depends on the host being Windows.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const MODULE = path.resolve(import.meta.dirname, '..', '..', 'electron', 'update_endpoints.js');

let sandbox;
let saved;

function load() {
  // Fresh each time: the module reads the environment on every call, but a
  // cached copy would hide an import-time failure like the missing `fs`.
  delete require.cache[MODULE];
  return require(MODULE);
}

/**
 * The Windows answers, asked for by name rather than by running on Windows.
 *
 * These tests used to redirect the real APPDATA/LOCALAPPDATA and then read the
 * host's own platform. Off Windows that does not merely fail to apply -- the
 * same functions read ~/Library/Application Support and XDG_DATA_HOME there,
 * so all eleven FAILED. They were briefly skipped instead; passing the
 * platform in is better, because a skipped test is one the macOS and Linux
 * runners do not run at all, and this file protects the code that decides
 * where several gigabytes land.
 *
 * The sandbox paths stay native to whichever host runs the suite, so nothing
 * here depends on Windows path syntax either.
 */
const WIN = () => ({
  platformName: 'win32',
  env: {
    APPDATA: path.join(sandbox, 'Roaming'),
    LOCALAPPDATA: path.join(sandbox, 'Local'),
    ORIENTA_HOME: process.env.ORIENTA_HOME,
  },
});

beforeEach(() => {
  sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-home-'));
  saved = { ORIENTA_HOME: process.env.ORIENTA_HOME };
  delete process.env.ORIENTA_HOME;
});

afterEach(() => {
  for (const [key, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  fs.rmSync(sandbox, { recursive: true, force: true });
});

describe('the data folder', () => {
  it('defaults to %LOCALAPPDATA%\\Orienta', () => {
    expect(load().orientaHome(WIN())).toBe(path.join(sandbox, 'Local', 'Orienta'));
  });

  it('follows a recorded choice', () => {
    const endpoints = load();
    const chosen = path.join(sandbox, 'D-drive', 'Orienta');
    endpoints.setOrientaHome(chosen, WIN());
    expect(endpoints.orientaHome(WIN())).toBe(chosen);
    // and a fresh process reads the same answer back from disk
    expect(load().orientaHome(WIN())).toBe(chosen);
  });

  it('survives a path with characters outside ASCII', () => {
    // The pointer is read by the UNINSTALLER too, and NSIS cannot read UTF-8:
    // a folder under "OneDrive - Montanuniversität Leoben" written as UTF-8
    // would come back mangled there, and the data would be silently left
    // behind. Hence UTF-16LE with a BOM, checked byte for byte.
    const endpoints = load();
    const chosen = path.join(sandbox, 'OneDrive - Montanuniversität Leoben', 'Orienta');
    endpoints.setOrientaHome(chosen, WIN());
    const raw = fs.readFileSync(endpoints.homePointerFile(WIN()));
    expect([raw[0], raw[1]]).toEqual([0xff, 0xfe]);
    expect(load().orientaHome(WIN())).toBe(chosen);
  });

  it('lives outside the data folder and outside the application', () => {
    const endpoints = load();
    const file = endpoints.homePointerFile(WIN());
    expect(file.startsWith(path.join(sandbox, 'Roaming'))).toBe(true);
    expect(file.startsWith(path.join(sandbox, 'Local'))).toBe(false);
  });

  it('refuses a folder it cannot write, and records nothing', () => {
    // Recording an unwritable folder would hand Chromium an unusable path on
    // the next launch — a known way to make this app not start at all.
    const endpoints = load();
    const blocker = path.join(sandbox, 'a-file-not-a-folder');
    fs.writeFileSync(blocker, 'x');
    expect(() => endpoints.setOrientaHome(path.join(blocker, 'Orienta'), WIN())).toThrow();
    expect(fs.existsSync(endpoints.homePointerFile(WIN()))).toBe(false);
    expect(load().orientaHome(WIN())).toBe(path.join(sandbox, 'Local', 'Orienta'));
  });

  it('refuses a relative path', () => {
    expect(() => load().setOrientaHome('relative\\Orienta', WIN())).toThrow(/absolute/);
  });

  it('lets ORIENTA_HOME win over a recorded choice', () => {
    const endpoints = load();
    endpoints.setOrientaHome(path.join(sandbox, 'recorded', 'Orienta'), WIN());
    process.env.ORIENTA_HOME = path.join(sandbox, 'from-env');
    expect(load().orientaHome(WIN())).toBe(path.join(sandbox, 'from-env'));
  });

  it('ignores a corrupt pointer instead of refusing to start', () => {
    // This runs at module load; a bad byte in a preference file must not be
    // the reason the application cannot start.
    const endpoints = load();
    fs.mkdirSync(path.dirname(endpoints.homePointerFile(WIN())), { recursive: true });
    fs.writeFileSync(endpoints.homePointerFile(WIN()), 'not\\an\\absolute\\path');
    expect(load().orientaHome(WIN())).toBe(path.join(sandbox, 'Local', 'Orienta'));
  });

  it('refuses the folder that holds its own pointer', () => {
    // electron-builder's `--delete-app-data` runs `RMDir /r` on exactly that
    // folder after Orienta's careful uninstall step. Picking %APPDATA% in the
    // "change…" dialog was enough to put the library there.
    const endpoints = load();
    const pointerDir = path.dirname(endpoints.homePointerFile(WIN()));
    expect(() => endpoints.setOrientaHome(pointerDir, WIN())).toThrow(/overlaps/);
    expect(() => endpoints.setOrientaHome(path.join(pointerDir, 'data'), WIN())).toThrow(/overlaps/);
    expect(() => endpoints.setOrientaHome(path.dirname(pointerDir), WIN())).toThrow(/overlaps/);
    // …case-insensitively, as Windows compares paths
    expect(() => endpoints.setOrientaHome(pointerDir.toUpperCase(), WIN())).toThrow(/overlaps/);
    expect(fs.existsSync(endpoints.homePointerFile(WIN()))).toBe(false);
  });

  it('refuses a folder inside the application it would be uninstalled with', () => {
    const endpoints = load();
    const installDir = path.join(sandbox, 'Programs', 'Orienta');
    expect(() => endpoints.setOrientaHome(path.join(installDir, 'data'),
      { forbidden: [installDir], ...WIN() })).toThrow(/overlaps/);
    // …but a neighbour of it is fine
    const beside = path.join(sandbox, 'Programs', 'OrientaData', 'Orienta');
    expect(endpoints.setOrientaHome(beside, { forbidden: [installDir], ...WIN() })).toBe(beside);
  });

  it('can be forgotten again', () => {
    const endpoints = load();
    endpoints.setOrientaHome(path.join(sandbox, 'elsewhere', 'Orienta'), WIN());
    endpoints.setOrientaHome(null, WIN());
    expect(load().orientaHome(WIN())).toBe(path.join(sandbox, 'Local', 'Orienta'));
  });
});
