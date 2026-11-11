/**
 * Choosing the data folder, tested by DOING it.
 *
 * Reported against 0.4.6: every click on the wizard's "change…" button added
 * another level (…/Orienta/Orienta/Orienta), and after deleting the folders by
 * hand a reinstall still did not work. The old test for this pinned the line
 * `path.join(answer.filePaths[0], 'Orienta')` in the source -- the line that
 * CAUSED the nesting. These run the pure functions and replay the clicks.
 *
 * Every folder here is inside a temporary directory; the pointer file lives in
 * a redirected APPDATA, so nothing can reach this machine's real preference.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ELECTRON = path.resolve(import.meta.dirname, '..', '..', 'electron');
const choice = require(path.join(ELECTRON, 'home_choice.js'));
const endpoints = require(path.join(ELECTRON, 'update_endpoints.js'));

let sandbox;
const env = () => ({
  APPDATA: path.join(sandbox, 'Roaming'),
  LOCALAPPDATA: path.join(sandbox, 'Local'),
});
// These tests play Windows on whatever machine runs them. The sandbox lies on a
// fixed disk, so that is the drive type they report: on a POSIX runner the
// sandbox has no drive letter, and asking the real probe there would answer
// "unknown type" for a folder that is merely deleted. Tests about the drive
// type pass their own.
const opts = () => ({ platformName: 'win32', env: env(), driveType: () => 'Fixed' });

beforeEach(() => {
  sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-choice-'));
});
afterEach(() => {
  fs.rmSync(sandbox, { recursive: true, force: true });
});

// --------------------------------------------------------------------------
// what picking a folder in the dialog means
// --------------------------------------------------------------------------

describe('homeFromPick', () => {
  it('adds Orienta\'s own subfolder to an ordinary folder', () => {
    // Someone who picks the drive must not get python\, runtime\ and logs\
    // across its root.
    expect(choice.homeFromPick(path.join(sandbox, 'Data'))).toBe(path.join(sandbox, 'Data', 'Orienta'));
  });

  it('does not add it again to a folder that is already named Orienta', () => {
    const home = path.join(sandbox, 'Data', 'Orienta');
    expect(choice.homeFromPick(home)).toBe(home);
    expect(choice.homeFromPick(`${home}${path.sep}`)).toBe(home);
  });

  it('does not take a folder of the user\'s own just because of its name', () => {
    // D:\Work\orienta, full of measurements: the shell would write electron\
    // and logs\ into it. The subfolder is added instead.
    const theirs = path.join(sandbox, 'Work', 'orienta');
    fs.mkdirSync(theirs, { recursive: true });
    fs.writeFileSync(path.join(theirs, 'scan1.h5oina'), 'x');
    expect(choice.homeFromPick(theirs)).toBe(path.join(theirs, 'Orienta'));
    // ... and a second pick of what that gave is stable
    expect(choice.homeFromPick(path.join(theirs, 'Orienta'))).toBe(path.join(theirs, 'Orienta'));
  });

  it('takes a folder named Orienta that is empty, absent, or holds only Orienta\'s names', () => {
    const empty = path.join(sandbox, 'A', 'Orienta');
    fs.mkdirSync(empty, { recursive: true });
    expect(choice.homeFromPick(empty)).toBe(empty);
    const absent = path.join(sandbox, 'B', 'Orienta');
    expect(choice.homeFromPick(absent)).toBe(absent);
    const own = path.join(sandbox, 'C', 'Orienta');
    fs.mkdirSync(path.join(own, 'python'), { recursive: true });
    fs.mkdirSync(path.join(own, 'logs'));
    fs.writeFileSync(path.join(own, '.python_path'), 'x');
    fs.mkdirSync(path.join(own, 'Orienta', 'Orienta'), { recursive: true });   // an empty chain
    expect(choice.homeFromPick(own)).toBe(own);
  });

  it('ignores the letter case of the name, as the file system does', () => {
    const lower = path.join(sandbox, 'Data', 'orienta');
    expect(choice.homeFromPick(lower)).toBe(lower);
  });

  it('does not add it to a folder that carries Orienta\'s own marker', () => {
    // A user may have renamed the folder; setup wrote .orienta-home in it.
    const renamed = path.join(sandbox, 'Library');
    fs.mkdirSync(renamed);
    fs.writeFileSync(path.join(renamed, '.orienta-home'), 'x');
    expect(choice.homeFromPick(renamed)).toBe(renamed);
  });

  it('is idempotent: choosing its own answer changes nothing', () => {
    for (const picked of [path.join(sandbox, 'Data'), path.join(sandbox, 'Data', 'Orienta')]) {
      const once = choice.homeFromPick(picked);
      expect(choice.homeFromPick(once)).toBe(once);
    }
  });
});

describe('the dialog, clicked again and again', () => {
  it('keeps the data folder where it is, whichever way OK is pressed', () => {
    // Replays the report: open the dialog, accept what it offers, repeat.
    // Both ways to accept are covered -- the folder the dialog opened in, and
    // the Orienta folder inside it. Before the fix each round added a level.
    const base = path.join(sandbox, 'Local');
    let home = path.join(base, 'Orienta');
    fs.mkdirSync(home, { recursive: true });
    for (let round = 0; round < 5; round++) {
      const opened = choice.pickerStart(home);
      home = choice.homeFromPick(opened);                       // OK on what it opened in
      expect(home).toBe(path.join(base, 'Orienta'));
      home = choice.homeFromPick(path.join(opened, 'Orienta')); // OK on the Orienta folder
      expect(home).toBe(path.join(base, 'Orienta'));
    }
  });

  it('what the old code did, for the record: each pick of the current home nested one more level', () => {
    // The 0.4.6 rule was join(picked, 'Orienta') unconditionally. Computed here
    // by hand, independent of the module under test.
    let home = path.join(sandbox, 'Local', 'Orienta');
    for (let i = 0; i < 2; i++) home = path.join(home, 'Orienta');
    expect(home).toBe(path.join(sandbox, 'Local', 'Orienta', 'Orienta', 'Orienta'));
  });
});

describe('a drive root', () => {
  // Windows path syntax: on another host 'D:\' is just a file name.
  const win = it.skipIf(process.platform !== 'win32');

  win('keeps its backslash: "D:" would be the current folder of drive D, not the drive', () => {
    expect(choice.pickerStart('D:\\', { exists: () => true })).toBe('D:\\');
    expect(choice.pickerStart('D:\\Orienta', { exists: () => true })).toBe('D:\\');
  });

  win('is a place the pick can be made in: D:\\ gives D:\\Orienta, never the relative D:Orienta', () => {
    expect(choice.homeFromPick('D:\\')).toBe('D:\\Orienta');
    expect(choice.homeFromPick('D:\\Orienta\\')).toBe('D:\\Orienta');
  });
});

describe('pickerStart', () => {
  it('opens in the folder that contains the current home', () => {
    const parent = path.join(sandbox, 'D-drive');
    fs.mkdirSync(path.join(parent, 'Orienta'), { recursive: true });
    expect(choice.pickerStart(path.join(parent, 'Orienta'))).toBe(parent);
  });

  it('climbs out of a chain a 0.4.6 user already built', () => {
    const parent = path.join(sandbox, 'Local');
    const nested = path.join(parent, 'Orienta', 'Orienta', 'Orienta');
    fs.mkdirSync(nested, { recursive: true });
    expect(choice.pickerStart(nested)).toBe(parent);
  });

  it('falls back to the nearest folder that exists', () => {
    const parent = path.join(sandbox, 'Local');
    fs.mkdirSync(parent);
    expect(choice.pickerStart(path.join(parent, 'deleted', 'Orienta'))).toBe(parent);
  });

  it('gives undefined, not a throw, for nothing', () => {
    expect(choice.pickerStart('')).toBeUndefined();
    expect(choice.pickerStart(undefined)).toBeUndefined();
  });
});

// --------------------------------------------------------------------------
// a recorded folder that is no longer there
// --------------------------------------------------------------------------

describe('a pointer to a folder that was deleted', () => {
  function pointAt(dir) {
    endpoints.setOrientaHome(dir, opts());
    return endpoints.homePointerFile(opts());
  }

  it('is reported stale, and the default folder is used again', () => {
    const outer = path.join(sandbox, 'chosen');
    const home = path.join(outer, 'Orienta', 'Orienta');
    const file = pointAt(home);
    fs.rmSync(outer, { recursive: true, force: true });        // "deleting the directory"
    expect(endpoints.orientaHome(opts())).toBe(home);          // the trap: still read

    const settled = choice.reconcileHomePointer(opts());
    expect(settled.action).toBe('fellBack');
    expect(settled.pointer).toBe(home);
    expect(settled.home).toBe(path.join(sandbox, 'Local', 'Orienta'));
    expect(fs.existsSync(file)).toBe(false);
    expect(endpoints.orientaHome(opts())).toBe(path.join(sandbox, 'Local', 'Orienta'));
    // and it must not have re-created the chain it just gave up on
    expect(fs.existsSync(outer)).toBe(false);
  });

  it('keeps the old setting as home.txt.previous instead of deleting it', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    const file = pointAt(home);
    const before = fs.readFileSync(file);
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const settled = choice.reconcileHomePointer(opts());
    expect(settled.previousFile).toBe(`${file}.previous`);
    expect(fs.readFileSync(settled.previousFile).equals(before)).toBe(true);
    // and the old setting can be read back with the module's own reader
    expect(endpoints.readPointer(settled.previousFile)).toBe(home);
  });

  it('does not overwrite an earlier copy when it happens a second time', () => {
    const first = path.join(sandbox, 'one', 'Orienta');
    const second = path.join(sandbox, 'two', 'Orienta');
    const file = pointAt(first);
    fs.rmSync(path.join(sandbox, 'one'), { recursive: true });
    const a = choice.reconcileHomePointer(opts());
    pointAt(second);
    fs.rmSync(path.join(sandbox, 'two'), { recursive: true });
    const b = choice.reconcileHomePointer(opts());
    expect(a.previousFile).toBe(`${file}.previous`);
    expect(b.previousFile).toBe(`${file}.previous.2`);
    expect(endpoints.readPointer(a.previousFile)).toBe(first);     // the first record is intact
    expect(endpoints.readPointer(b.previousFile)).toBe(second);
  });

  it('does its work through the fs it was given, not the global one', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    pointAt(home);
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const calls = [];
    const spy = {
      ...fs,
      renameSync: (...a) => { calls.push(['rename', ...a]); return fs.renameSync(...a); },
      existsSync: (p) => { calls.push(['exists', p]); return fs.existsSync(p); },
    };
    const settled = choice.reconcileHomePointer({ ...opts(), fsImpl: spy });
    expect(settled.action).toBe('fellBack');
    expect(calls.some((c) => c[0] === 'rename')).toBe(true);
    expect(calls.some((c) => c[0] === 'exists' && c[1].endsWith('.previous'))).toBe(true);
  });

  it('when the old setting cannot be kept it is not given up, and nothing may be created at it', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    const file = pointAt(home);
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const failing = { ...fs, renameSync: () => { throw fail('EPERM'); } };
    const settled = choice.reconcileHomePointer({ ...opts(), fsImpl: failing });
    expect(settled.action).toBe('unreachable');
    expect(settled.why).toContain('EPERM');
    expect(fs.existsSync(file)).toBe(true);                         // still there, still pointing
    expect(endpoints.orientaHome(opts())).toBe(home);               // ... at a folder that is gone
    const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => settled });
    expect(choice.mustNotCreateHome(notice)).toBe(true);            // so the shell must not mkdir it
    expect(choice.startupLogLines({ notice, checkError: null, home: 'H' })[0]).toContain('EPERM');
  });

  it('keeps every record over three events: .previous, .previous.2, .previous.3', () => {
    const folders = ['one', 'two', 'three'].map((n) => path.join(sandbox, n, 'Orienta'));
    const file = endpoints.homePointerFile(opts());
    const kept = [];
    for (const folder of folders) {
      pointAt(folder);
      fs.rmSync(path.dirname(folder), { recursive: true });
      kept.push(choice.reconcileHomePointer(opts()).previousFile);
    }
    expect(kept).toEqual([`${file}.previous`, `${file}.previous.2`, `${file}.previous.3`]);
    kept.forEach((name, i) => expect(endpoints.readPointer(name)).toBe(folders[i]));
  });

  it('touches nothing but the pointer file', () => {
    const outer = path.join(sandbox, 'chosen');
    const keep = path.join(outer, 'thesis.docx');
    pointAt(path.join(outer, 'Orienta'));
    fs.rmSync(path.join(outer, 'Orienta'), { recursive: true });
    fs.writeFileSync(keep, 'mine');
    choice.reconcileHomePointer(opts());
    expect(fs.readFileSync(keep, 'utf8')).toBe('mine');
  });

  it('is left alone while the folder exists', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    const file = pointAt(home);
    const settled = choice.reconcileHomePointer(opts());
    expect(settled.action).toBe('none');
    expect(fs.existsSync(file)).toBe(true);
    expect(endpoints.orientaHome(opts())).toBe(home);
  });

  it('is left alone when the environment names the folder anyway', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    const file = pointAt(home);
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const withOverride = { ...opts(), env: { ...env(), ORIENTA_HOME: path.join(sandbox, 'elsewhere') } };
    expect(choice.reconcileHomePointer(withOverride).action).toBe('none');
    expect(fs.existsSync(file)).toBe(true);
  });
});

/** An fs whose answers are scripted per path; everything else is the real one. */
function scripted(overrides) {
  return {
    ...fs,
    statSync: (p, ...rest) => {
      if (overrides.stat && overrides.stat(p)) throw overrides.stat(p);
      return fs.statSync(p, ...rest);
    },
    lstatSync: (p, ...rest) => {
      if (overrides.lstat && overrides.lstat(p)) throw overrides.lstat(p);
      return fs.lstatSync(p, ...rest);
    },
  };
}
const fail = (code) => Object.assign(new Error(code), { code });

describe('a pointer that must NOT be given up', () => {
  function pointAt(dir) {
    endpoints.setOrientaHome(dir, opts());
    return endpoints.homePointerFile(opts());
  }

  it('keeps it when the drive is not there (nothing along the way answers)', () => {
    const home = path.join(sandbox, 'usb', 'Orienta');
    const file = pointAt(home);
    // The medium is absent: the folder, every ancestor up to the root fail with ENOENT.
    const absent = scripted({
      stat: (p) => (p === home ? fail('ENOENT') : null),
      lstat: () => fail('ENOENT'),
    });
    const settled = choice.reconcileHomePointer({ ...opts(), fsImpl: absent });
    expect(settled.action).toBe('unreachable');
    expect(settled.pointer).toBe(home);
    expect(fs.existsSync(file)).toBe(true);
    expect(fs.existsSync(`${file}.previous`)).toBe(false);
    expect(endpoints.orientaHome(opts())).toBe(home);
  });

  it.each(['EACCES', 'EPERM', 'EIO', 'UNKNOWN', 'EBUSY'])(
    'keeps it when looking at the folder fails with %s: it may well be there', (code) => {
      const home = path.join(sandbox, 'chosen', 'Orienta');
      const file = pointAt(home);
      const settled = choice.reconcileHomePointer({
        ...opts(), fsImpl: scripted({ stat: (p) => (p === home ? fail(code) : null) }),
      });
      expect(settled.action).toBe('unreachable');
      expect(settled.why).toContain(code);
      expect(fs.existsSync(file)).toBe(true);
    });

  it('keeps it when the ancestor cannot be read either', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    const file = pointAt(home);
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const settled = choice.reconcileHomePointer({
      ...opts(), fsImpl: scripted({ lstat: (p) => (p === sandbox ? fail('EACCES') : null) }),
    });
    expect(settled.action).toBe('unreachable');
    expect(fs.existsSync(file)).toBe(true);
  });

  it('keeps it when it runs through a link whose target is gone (a mounted stick)', () => {
    const target = path.join(sandbox, 'volume');
    fs.mkdirSync(path.join(target, 'Orienta'), { recursive: true });
    const link = path.join(sandbox, 'usb');
    try {
      fs.symlinkSync(target, link, 'junction');
    } catch { return; }    // no links here: nothing to assert
    const file = pointAt(path.join(link, 'Orienta'));
    expect(choice.pointerState(opts()).state).toBe('present');
    fs.rmSync(target, { recursive: true, force: true });       // the volume is removed
    const state = choice.pointerState(opts());
    expect(state.state).toBe('unreachable');
    expect(state.why).toMatch(/link/);
    expect(choice.reconcileHomePointer(opts()).action).toBe('unreachable');
    expect(fs.existsSync(file)).toBe(true);
    fs.rmSync(link, { force: true });
  });

  it('keeps it when the recorded folder is ITSELF a link whose target is gone', () => {
    const target = path.join(sandbox, 'volume', 'Orienta');
    fs.mkdirSync(target, { recursive: true });
    const link = path.join(sandbox, 'Orienta-link');
    try {
      fs.symlinkSync(target, link, 'junction');
    } catch { return; }    // no links here: nothing to assert
    const file = pointAt(link);
    expect(choice.pointerState(opts()).state).toBe('present');
    fs.rmSync(path.join(sandbox, 'volume'), { recursive: true, force: true });
    const state = choice.pointerState(opts());
    expect(state.state).toBe('unreachable');
    expect(state.why).toMatch(/link/);
    expect(choice.reconcileHomePointer(opts()).action).toBe('unreachable');
    expect(fs.existsSync(file)).toBe(true);
    expect(fs.existsSync(`${file}.previous`)).toBe(false);
    fs.rmSync(link, { force: true });
  });

  it('keeps it when the check of the link itself fails with anything but "not there"', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    const file = pointAt(home);
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const settled = choice.reconcileHomePointer({
      ...opts(), fsImpl: scripted({ lstat: (p) => (p === home ? fail('EIO') : null) }),
    });
    expect(settled.action).toBe('unreachable');
    expect(settled.why).toContain('EIO');
    expect(fs.existsSync(file)).toBe(true);
  });

  it('is told apart from a deleted folder by what the ancestor is, not by a guess', () => {
    // Same pointer, same missing folder; only the answer for the ancestor differs.
    const home = path.join(sandbox, 'usb', 'Orienta');
    pointAt(home);
    fs.rmSync(path.join(sandbox, 'usb'), { recursive: true });
    expect(choice.pointerState(opts()).state).toBe('stale');
    const gone = scripted({ lstat: () => fail('ENOENT') });
    expect(choice.pointerState({ ...opts(), fsImpl: gone }).state).toBe('unreachable');
  });
});

describe('a pointer file that cannot be READ', () => {
  const readerFailing = (code) => ({ ...fs, readFileSync: (p, ...rest) => {
    if (String(p).endsWith('home.txt')) throw fail(code);
    return fs.readFileSync(p, ...rest);
  } });
  function pointAt(dir) {
    endpoints.setOrientaHome(dir, opts());
    return endpoints.homePointerFile(opts());
  }

  it.each(['EBUSY', 'EPERM', 'EACCES', 'EIO'])(
    'with %s it is unreadable, not "no choice": nothing is decided and nothing is created', (code) => {
      const home = path.join(sandbox, 'chosen', 'Orienta');
      const file = pointAt(home);
      const state = choice.pointerState({ ...opts(), fsImpl: readerFailing(code) });
      expect(state.state).toBe('unreachable');
      expect(state.kind).toBe('unreadable');
      expect(state.why).toContain(code);
      const settled = choice.reconcileHomePointer({ ...opts(), fsImpl: readerFailing(code) });
      expect(settled.action).toBe('unreachable');
      expect(settled.kind).toBe('unreadable');
      expect(fs.existsSync(file)).toBe(true);
      const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => settled });
      expect(choice.mustNotCreateHome(notice)).toBe(true);
      expect(choice.unreachableForProbe(notice).kind).toBe('unreadable');
      expect(choice.startupLogLines({ notice, checkError: null, home: 'H' })[0]).toMatch(/could not be read/);
      expect(choice.unusableHomeReason({ home: 'H', error: code, notice })).not.toMatch(/ORIENTA_HOME/);
    });

  it('a missing pointer file is still just "no choice recorded"', () => {
    expect(choice.pointerState(opts()).state).toBe('none');
    expect(choice.pointerState({ ...opts(), fsImpl: readerFailing('ENOENT') }).state).toBe('none');
  });
});

describe('a drive letter that now belongs to another drive', () => {
  // The folder was recorded on one stick, and E: is another one now. Only a FIXED disk answers
  // for its folders.
  const gone = () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    endpoints.setOrientaHome(home, opts());
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    return home;
  };
  // the sandbox's drive is not the "system drive" of this homedir, so the type is asked
  const elsewhereHome = process.platform === 'win32' ? 'Q:\\Users\\x' : '/nonexistent-home';

  it.each(['Removable', 'Network', 'CDRom', 'Ram'])('%s: unreachable, the pointer is kept', (type) => {
    const home = gone();
    const state = choice.pointerState({ ...opts(), homedir: elsewhereHome, driveType: () => type });
    expect(state.state).toBe('unreachable');
    expect(state.why).toContain(type);
    const settled = choice.reconcileHomePointer({ ...opts(), homedir: elsewhereHome, driveType: () => type });
    expect(settled.action).toBe('unreachable');
    expect(fs.existsSync(endpoints.homePointerFile(opts()))).toBe(true);
    expect(endpoints.orientaHome(opts())).toBe(home);
  });

  it('a drive whose type cannot be told is not trusted either', () => {
    gone();
    expect(choice.pointerState({ ...opts(), homedir: elsewhereHome, driveType: () => null }).state)
      .toBe('unreachable');
  });

  it('a fixed disk is still stale (the pointer is retired)', () => {
    gone();
    expect(choice.pointerState({ ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed' }).state)
      .toBe('stale');
  });

  it('the system drive is not asked at all', () => {
    gone();
    let asked = 0;
    const homedir = process.platform === 'win32' ? path.join(path.parse(sandbox).root, 'Users', 'x') : sandbox;
    if (process.platform !== 'win32') return;      // drive letters are a Windows matter
    const state = choice.pointerState({ ...opts(), homedir, driveType: () => { asked++; return 'Removable'; } });
    expect(state.state).toBe('stale');
    expect(asked).toBe(0);
  });

  it('a network path is a network drive without asking anyone', () => {
    expect(choice.windowsDriveType('\\\\server\\share\\')).toBe('Network');
    expect(choice.windowsDriveType('')).toBeNull();
    expect(choice.windowsDriveType('1:\\')).toBeNull();
  });

  it.skipIf(process.platform !== 'win32')('the real answer for the system drive is Fixed', () => {
    expect(choice.windowsDriveType(path.parse(process.cwd()).root)).toBe('Fixed');
  });

  it('is not asked off Windows (mount points are judged by the home volume)', () => {
    let asked = 0;
    choice.pointerState({ platformName: 'linux', env: env(), driveType: () => { asked++; return 'Removable'; } });
    expect(asked).toBe(0);
  });
});

describe('a probe that throws means "cannot tell", never "deleted"', () => {
  const gone = () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    endpoints.setOrientaHome(home, { ...opts(), volumeSerial: () => null });
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    return home;
  };
  const elsewhereHome = process.platform === 'win32' ? 'Q:\\Users\\x' : '/nonexistent-home';

  it('a throwing drive-type probe keeps the pointer (and the shell is told not to create the folder)', () => {
    gone();
    const thrower = () => { throw new Error('powershell is not there'); };
    const state = choice.pointerState({ ...opts(), homedir: elsewhereHome, driveType: thrower });
    expect(state.state).toBe('unreachable');
    const settled = choice.reconcileHomePointer({ ...opts(), homedir: elsewhereHome, driveType: thrower });
    expect(settled.action).toBe('unreachable');
    expect(fs.existsSync(endpoints.homePointerFile(opts()))).toBe(true);
    const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => settled });
    expect(choice.mustNotCreateHome(notice)).toBe(true);
  });

  it('a throwing volume probe keeps the pointer too', () => {
    gone();
    const thrower = () => { throw new Error('vol failed'); };
    const sandboxRoot = path.parse(sandbox).root;
    if (process.platform !== 'win32') return;
    fs.writeFileSync(path.join(path.dirname(endpoints.homePointerFile(opts())), 'home.volume'),
      JSON.stringify({ root: sandboxRoot, serial: 'AAAA-0001', folder: 'x' }));
    const state = choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: thrower,
    });
    expect(state.state).toBe('unreachable');
  });
});

describe('the volume a recorded folder was on (an external disk is also "Fixed")', () => {
  const win = it.skipIf(process.platform !== 'win32');
  const elsewhereHome = 'Q:\\Users\\x';
  const recordFile = () => path.join(path.dirname(endpoints.homePointerFile(opts())), 'home.volume');
  const driveRoot = () => path.parse(sandbox).root;
  function recorded(serial) {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    endpoints.setOrientaHome(home, { ...opts(), volumeSerial: () => serial });
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    return home;
  }

  win('setOrientaHome records the volume beside the pointer', () => {
    recorded('1234-ABCD');
    expect(JSON.parse(fs.readFileSync(recordFile(), 'utf8'))).toMatchObject({
      root: driveRoot(), serial: '1234-ABCD',
    });
  });

  win('... and forgets it with the pointer', () => {
    recorded('1234-ABCD');
    endpoints.setOrientaHome(null, opts());
    expect(fs.existsSync(recordFile())).toBe(false);
  });

  win('the same volume and a fixed disk: the folder is deleted, the pointer is retired', () => {
    recorded('1234-ABCD');
    const state = choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => '1234-abcd',
    });
    expect(state.state).toBe('stale');
  });

  win('ANOTHER volume answers for the letter: unreachable, even though it is Fixed', () => {
    recorded('1234-ABCD');
    const state = choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => '9999-0000',
    });
    expect(state.state).toBe('unreachable');
    expect(state.why).toMatch(/not the one the folder was recorded on/);
    const settled = choice.reconcileHomePointer({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => '9999-0000',
    });
    expect(settled.action).toBe('unreachable');
    expect(fs.existsSync(endpoints.homePointerFile(opts()))).toBe(true);
    expect(fs.existsSync(recordFile())).toBe(true);
  });

  win('a volume that cannot be read now is not trusted', () => {
    recorded('1234-ABCD');
    expect(choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => null,
    }).state).toBe('unreachable');
  });

  win('a pointer with no record (older version) falls back to the type: the documented limit', () => {
    recorded(null);                                    // no serial -> no record is written
    expect(fs.existsSync(recordFile())).toBe(false);
    expect(choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => 'ANY-THING',
    }).state).toBe('stale');
    expect(choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Removable', volumeSerial: () => 'ANY-THING',
    }).state).toBe('unreachable');
  });

  win('a record for another drive letter is not consulted', () => {
    recorded('1234-ABCD');
    const rec = JSON.parse(fs.readFileSync(recordFile(), 'utf8'));
    fs.writeFileSync(recordFile(), JSON.stringify({ ...rec, root: 'Z:\\' }));
    expect(choice.pointerState({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => '9999-0000',
    }).state).toBe('stale');
  });

  win('retiring the pointer retires its record too', () => {
    recorded('1234-ABCD');
    const settled = choice.reconcileHomePointer({
      ...opts(), homedir: elsewhereHome, driveType: () => 'Fixed', volumeSerial: () => '1234-ABCD',
    });
    expect(settled.action).toBe('fellBack');
    expect(fs.existsSync(recordFile())).toBe(false);
  });

  it('the volume record module: bad files are "no record", and a failing serial removes the record', () => {
    const vol = require(path.join(ELECTRON, 'volume_id.js'));
    const pointer = path.join(sandbox, 'Roaming', 'Orienta', 'home.txt');
    fs.mkdirSync(path.dirname(pointer), { recursive: true });
    expect(vol.readVolumeRecord(pointer)).toBeNull();
    fs.writeFileSync(vol.volumeRecordFile(pointer), 'not json');
    expect(vol.readVolumeRecord(pointer)).toBeNull();
    fs.writeFileSync(vol.volumeRecordFile(pointer), JSON.stringify({ root: 'E:\\', serial: '' }));
    expect(vol.readVolumeRecord(pointer)).toBeNull();
    fs.writeFileSync(vol.volumeRecordFile(pointer), JSON.stringify({ root: 'E:\\', serial: 'AAAA-0001' }));
    expect(vol.readVolumeRecord(pointer).serial).toBe('AAAA-0001');
    expect(vol.writeVolumeRecord(pointer, 'E:\\Orienta', { volumeSerial: () => null })).toBeNull();
    expect(fs.existsSync(vol.volumeRecordFile(pointer))).toBe(false);
    expect(vol.writeVolumeRecord(pointer, 'E:\\Orienta', { volumeSerial: () => { throw new Error('x'); } })).toBeNull();
    expect(vol.driveRootOf('\\\\server\\share\\x')).toBeNull();
    expect(vol.driveRootOf('relative')).toBeNull();
  });

  win('the serial of the system drive has the XXXX-XXXX form; a bad letter has none', () => {
    const vol = require(path.join(ELECTRON, 'volume_id.js'));
    expect(vol.windowsVolumeSerial(driveRoot())).toMatch(/^[0-9A-F]{4}-[0-9A-F]{4}$/);
    expect(vol.windowsVolumeSerial('1:\\')).toBeNull();
    expect(vol.windowsVolumeSerial('')).toBeNull();
  });
});

describe('when the old setting cannot be kept, the page says so (and not "drive not connected")', () => {
  it('retire failure: its own kind, fact only in the detail', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    endpoints.setOrientaHome(home, opts());
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const failing = { ...fs, renameSync: () => { throw fail('EPERM'); } };
    const settled = choice.reconcileHomePointer({ ...opts(), fsImpl: failing });
    expect(settled.kind).toBe('retire');
    const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => settled });
    expect(choice.unreachableForProbe(notice).kind).toBe('retire');
    const reason = choice.unusableHomeReason({ home: 'H', error: 'x', notice });
    expect(reason).toMatch(/no longer exists/);
    expect(reason).not.toMatch(/ORIENTA_HOME/);
  });

  it('a pointer to a drive that is not there stays kind "drive"', () => {
    const notice = choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => ({ action: 'unreachable', kind: 'drive', pointer: 'P', pointerFile: 'F' }),
    });
    expect(choice.unreachableForProbe(notice).kind).toBe('drive');
  });
});

describe('off Windows a missing folder is only stale inside the home volume', () => {
  const posix = () => ({ platformName: 'linux', env: { ...env(), XDG_CONFIG_HOME: path.join(sandbox, 'Roaming') } });
  function pointAt(dir) {
    // The pointer is written with the Windows-flavoured options (the writer
    // is shared); only the reading is asked as a Linux host.
    endpoints.setOrientaHome(dir, opts());
  }

  it('is stale below the user\'s home', () => {
    const home = path.join(sandbox, 'home', 'anna', 'data', 'Orienta');
    pointAt(home);
    fs.rmSync(path.join(sandbox, 'home', 'anna', 'data'), { recursive: true });
    const state = choice.pointerState({ ...posix(), homedir: path.join(sandbox, 'home', 'anna') });
    expect(state.state).toBe('stale');
  });

  it('is unreachable under /mnt, /media, a gvfs mount: the mount point may just be unmounted', () => {
    const home = path.join(sandbox, 'mnt', 'nfs', 'Orienta');
    pointAt(home);
    fs.rmSync(home, { recursive: true });        // the EMPTY mount point stays
    fs.mkdirSync(path.join(sandbox, 'home', 'anna'), { recursive: true });
    const state = choice.pointerState({ ...posix(), homedir: path.join(sandbox, 'home', 'anna') });
    expect(state.state).toBe('unreachable');
    expect(state.why).toMatch(/home volume/);
  });

  it('is unreachable inside the home when the ancestor is on another device', () => {
    const home = path.join(sandbox, 'home', 'anna', 'nas', 'Orienta');
    pointAt(home);
    fs.rmSync(home, { recursive: true });
    const real = fs.lstatSync(path.join(sandbox, 'home', 'anna'));
    const otherDevice = scripted({});
    otherDevice.lstatSync = (p, ...r) => {
      const st = fs.lstatSync(p, ...r);
      return path.resolve(p) === path.resolve(sandbox, 'home', 'anna', 'nas')
        ? Object.assign(Object.create(Object.getPrototypeOf(st)), st, { dev: real.dev + 1 }) : st;
    };
    const state = choice.pointerState({
      ...posix(), homedir: path.join(sandbox, 'home', 'anna'), fsImpl: otherDevice,
    });
    expect(state.state).toBe('unreachable');
  });
});


// --------------------------------------------------------------------------
// the wiring: a correct function nobody calls fixes nothing
// --------------------------------------------------------------------------

describe('findExistingHome follows a chain as far as the chooser does', () => {
  const chainHome = (levels) => {
    const base = path.join(sandbox, 'B');
    const home = path.join(base, ...Array(levels).fill('Orienta'));
    fs.mkdirSync(home, { recursive: true });
    fs.writeFileSync(path.join(home, '.orienta-home'), 'x');
    return { base, home };
  };

  it('finds an installation 15 levels below the pick (the 16th folder counted from it)', () => {
    const { base, home } = chainHome(15);
    expect(choice.findExistingHome(base)).toBe(home);
  });

  it('does not find one 16 levels below: the chain stops there', () => {
    const { base } = chainHome(16);
    expect(choice.findExistingHome(base)).toBeNull();
  });
});

describe('sameFolder', () => {
  it('ignores letter case and trailing separators, as Windows does', () => {
    const a = path.join(sandbox, 'Data', 'Orienta');
    expect(choice.sameFolder(a, a.toUpperCase())).toBe(true);
    expect(choice.sameFolder(a, `${a}${path.sep}`)).toBe(true);
    expect(choice.sameFolder(a, path.join(sandbox, 'Data', 'Other'))).toBe(false);
  });
});

describe('a real 0.4.6 installation inside the nested chain', () => {
  // X/Orienta/Orienta, the folder the repeated "change..." clicks ended in, with
  // an installation set up there: marker, python, runtime, the user's library.
  let x; let real;
  beforeEach(() => {
    x = path.join(sandbox, 'X');
    real = path.join(x, 'Orienta', 'Orienta');
    fs.mkdirSync(path.join(real, 'python'), { recursive: true });
    fs.mkdirSync(path.join(real, 'runtime', 'Database', 'CIF_Library'), { recursive: true });
    fs.writeFileSync(path.join(real, 'runtime', 'Database', 'CIF_Library', 'mine.cif'), 'irreplaceable');
    fs.writeFileSync(path.join(real, '.orienta-home'), 'x');
    fs.writeFileSync(path.join(real, 'notes.txt'), 'mine');
  });

  it('is found from the folder the repaired dialog opens in', () => {
    // The dialog opens at X; OK gives X/Orienta, which holds only the next level.
    expect(choice.findExistingHome(path.join(x, 'Orienta'))).toBe(real);
    expect(choice.findExistingHome(x)).toBe(real);
    expect(choice.findExistingHome(path.join(sandbox, 'nothing'))).toBeNull();
  });

  it('is the folder chosen, not refused as "other files", and the library is not in the way', () => {
    const decision = choice.decidePick(x, { current: path.join(sandbox, 'Local', 'Orienta') });
    expect(decision.home).toBe(real);
    expect(decision.adopted).toBe(true);
    expect(decision.refusal).toBeNull();
  });

  it('is "unchanged" when it already is the folder in use: nothing is recorded', () => {
    const decision = choice.decidePick(x, { current: real });
    expect(decision.home).toBe(real);
    expect(decision.unchanged).toBe(true);
    expect(decision.refusal).toBeNull();
  });

  it('is recognised without the marker too, by runtime/VERSION plus python', () => {
    fs.rmSync(path.join(real, '.orienta-home'));
    fs.writeFileSync(path.join(real, 'runtime', 'VERSION'), 'v0.4.6\n');
    fs.writeFileSync(path.join(real, 'python', 'python.exe'), 'exe');
    expect(choice.findExistingHome(x)).toBe(real);
    fs.rmSync(path.join(real, 'python', 'python.exe'));
    expect(choice.findExistingHome(x)).toBeNull();   // one sign alone is not enough
  });

  it('is never reported as something to delete', () => {
    // Picking a place where it would be refused: a sibling folder of the same
    // chain without markers but with the library inside.
    fs.rmSync(path.join(real, '.orienta-home'));
    const refusal = choice.decidePick(x, { current: path.join(sandbox, 'Local', 'Orienta') }).refusal;
    expect(refusal.code).toBe('homeHoldsData');
    expect(refusal.values.entries).toBe('Orienta');
  });
});

describe('a pick that cannot be used', () => {
  it('is refused with the entries, and it is decided before the pointer is written', () => {
    const base = path.join(sandbox, 'Y');
    fs.mkdirSync(path.join(base, 'Orienta'), { recursive: true });
    fs.writeFileSync(path.join(base, 'Orienta', 'thesis.docx'), 'mine');
    const decision = choice.decidePick(base, { current: path.join(sandbox, 'Local', 'Orienta') });
    expect(decision.home).toBe(path.join(base, 'Orienta'));
    expect(decision.refusal.code).toBe('homeNotEmpty');
    expect(decision.refusal.values).toEqual({ dir: path.join(base, 'Orienta'), entries: 'thesis.docx' });
  });

  it('is refused for the folder already in use too, when nothing says it is installed', () => {
    // Back -> change -> OK on the offered folder is the natural click; the page
    // used to answer "keeps using it" and Install then refused the same folder.
    const home = path.join(sandbox, 'Z', 'Orienta');
    fs.mkdirSync(home, { recursive: true });
    fs.writeFileSync(path.join(home, 'thesis.docx'), 'mine');
    const decision = choice.decidePick(path.join(sandbox, 'Z'), { current: home });
    expect(decision.unchanged).toBe(true);
    expect(decision.refusal.code).toBe('homeNotEmpty');
    expect(decision.refusal.values.entries).toBe('thesis.docx');
  });

  it('is not asked about an installed folder, in use or not', () => {
    const home = path.join(sandbox, 'Z2', 'Orienta');
    fs.mkdirSync(home, { recursive: true });
    fs.writeFileSync(path.join(home, '.orienta-home'), 'x');
    fs.writeFileSync(path.join(home, 'notes.txt'), 'mine');
    let asked = 0;
    const decision = choice.decidePick(path.join(sandbox, 'Z2'), {
      current: home, refuse: () => { asked++; return { code: 'x' }; },
    });
    expect(decision.unchanged).toBe(true);
    expect(decision.refusal).toBeNull();
    expect(asked).toBe(0);
  });
});

describe('settleHomeAtStartup', () => {
  const settledStale = {
    action: 'fellBack', pointer: 'P', pointerFile: 'F', previousFile: 'F.previous', home: 'H',
  };

  it('does nothing at all in development: the reconciler is not even asked', () => {
    let asked = 0;
    const notice = choice.settleHomeAtStartup({ isPackaged: false, reconcile: () => { asked++; return settledStale; } });
    expect(notice).toBeNull();
    expect(asked).toBe(0);
  });

  it('in a packaged build asks, and carries the kept file into the notice', () => {
    const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => settledStale });
    expect(notice).toEqual({
      code: 'staleFallback', dir: 'P', pointerFile: 'F', previousFile: 'F.previous',
    });
  });

  it('says "default" when the stale pointer named the default folder itself', () => {
    // It falls back to the very same place; "chosen earlier ... the default
    // instead" would be nonsense.
    const same = { action: 'fellBack', pointer: path.join(sandbox, 'Local', 'Orienta'),
      pointerFile: 'F', previousFile: 'F.previous', home: path.join(sandbox, 'Local', 'Orienta') };
    const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => same });
    expect(notice.code).toBe('staleDefault');
    const other = choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => ({ ...same, pointer: path.join(sandbox, 'D', 'Orienta') }),
    });
    expect(other.code).toBe('staleFallback');
    // spelled differently, still the same folder
    const spelled = choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => ({ ...same, pointer: `${same.pointer}${path.sep}`.toUpperCase() }),
    });
    expect(spelled.code).toBe('staleDefault');
    expect(choice.startupLogLines({ notice, checkError: null, home: 'H' })[0]).toMatch(/is the default folder/);
  });

  it('names an unreachable folder as such', () => {
    const notice = choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => ({ action: 'unreachable', pointer: 'P', pointerFile: 'F' }),
    });
    expect(notice.code).toBe('unreachable');
    expect(choice.unreachableForProbe(notice)).toEqual({ dir: 'P', pointerFile: 'F', kind: 'drive' });
  });

  it('says nothing when there is nothing to say', () => {
    expect(choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => ({ action: 'none', pointer: null }),
    })).toBeNull();
    expect(choice.unreachableForProbe(null)).toBeNull();
    expect(choice.unreachableForProbe({ code: 'staleFallback', dir: 'P' })).toBeNull();
  });

  it('a failing check is reported, never stops the start, and BLOCKS creating the folder', () => {
    // With no notice the shell went on to create the recorded folder -- the
    // chain the check exists to stop. A check that failed is not a check that
    // found nothing.
    const errors = [];
    const notice = choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => { throw new Error('boom'); }, onError: (e) => errors.push(e.message),
    });
    expect(errors).toEqual(['boom']);
    expect(notice.code).toBe('unreachable');
    expect(notice.kind).toBe('check');
    expect(notice.why).toBe('boom');
    expect(choice.mustNotCreateHome(notice)).toBe(true);
    expect(choice.unreachableForProbe(notice)).toMatchObject({ kind: 'check', why: 'boom' });
    expect(choice.unusableHomeReason({ home: 'H', error: 'x', notice })).toMatch(/could not check/);
  });

  it('end to end: a real stale pointer in a packaged build is renamed, in development it is left', () => {
    const home = path.join(sandbox, 'chosen', 'Orienta');
    endpoints.setOrientaHome(home, opts());
    fs.rmSync(path.join(sandbox, 'chosen'), { recursive: true });
    const file = endpoints.homePointerFile(opts());
    expect(choice.settleHomeAtStartup({ isPackaged: false, reconcile: () => choice.reconcileHomePointer(opts()) })).toBeNull();
    expect(fs.existsSync(file)).toBe(true);
    const notice = choice.settleHomeAtStartup({ isPackaged: true, reconcile: () => choice.reconcileHomePointer(opts()) });
    expect(notice.code).toBe('staleFallback');
    expect(fs.existsSync(file)).toBe(false);
    expect(fs.existsSync(notice.previousFile)).toBe(true);
  });
});

describe('unusableHomeReason', () => {
  it('for an unreachable recorded folder states the fact and gives no contradicting advice', () => {
    const text = choice.unusableHomeReason({
      home: 'H', error: 'ENOENT', notice: { code: 'unreachable', dir: 'E:/data/Orienta' },
    });
    expect(text).toContain('E:/data/Orienta');
    expect(text).toContain('ENOENT');
    expect(text).not.toMatch(/ORIENTA_HOME/);
    expect(text).not.toMatch(/delete/i);
  });

  it('for any other unusable folder keeps the general advice', () => {
    const text = choice.unusableHomeReason({ home: 'H', error: 'EROFS', notice: null });
    expect(text).toContain('H');
    expect(text).toMatch(/ORIENTA_HOME/);
  });
});

describe('startupLogLines', () => {
  it('reports a failed check, which nothing else would show in a packaged app', () => {
    const lines = choice.startupLogLines({ notice: null, checkError: new Error('EPERM: nope'), home: 'H' });
    expect(lines).toHaveLength(1);
    expect(lines[0]).toContain('EPERM: nope');
  });

  it('says what happened to a stale pointer, and where the old setting went', () => {
    const [line] = choice.startupLogLines({
      notice: { code: 'staleFallback', dir: 'P', pointerFile: 'F', previousFile: 'F.previous' },
      checkError: null, home: 'H',
    });
    expect(line).toContain('P');
    expect(line).toContain('H');
    expect(line).toContain('F.previous');
  });

  it('says an unreachable folder is kept, and is silent when there is nothing to say', () => {
    const [line] = choice.startupLogLines({
      notice: { code: 'unreachable', dir: 'P', pointerFile: 'F' }, checkError: null, home: 'H',
    });
    expect(line).toMatch(/not reachable/);
    expect(line).toMatch(/kept/);
    expect(choice.startupLogLines({ notice: null, checkError: null, home: 'H' })).toEqual([]);
  });

  it('the failure comes from the real settle call', () => {
    let kept = null;
    choice.settleHomeAtStartup({
      isPackaged: true, reconcile: () => { throw new Error('boom'); }, onError: (e) => { kept = e; },
    });
    expect(choice.startupLogLines({ notice: null, checkError: kept, home: 'H' })[0]).toContain('boom');
  });
});

describe('the shell uses it', () => {
  const MAIN = fs.readFileSync(path.join(ELECTRON, 'main.js'), 'utf8');
  const pickBody = MAIN.slice(MAIN.indexOf("handleSetup('setup:pickHome'"),
    MAIN.indexOf("handleSetup('setup:cancel'"));

  it('opens the dialog where the pick cannot nest, and does not append blindly', () => {
    expect(pickBody).toMatch(/homeChoice\.pickerStart\(/);
    expect(pickBody).toMatch(/homeChoice\.decidePick\(/);
    expect(pickBody).not.toMatch(/path\.join\(answer\.filePaths\[0\], 'Orienta'\)/);
    expect(pickBody).not.toMatch(/defaultPath: orientaHome\(\)/);
  });

  it('reports "kept" only for the folder in use and "adopted" for a different one found below the pick', () => {
    expect(pickBody).toMatch(/kept: true, restartNeeded: false/);
    expect(pickBody).toMatch(/adopted: decision.adopted, restartNeeded/);
  });

  it('decides the pick BEFORE the pointer is written, and answers a refusal without writing', () => {
    const decide = pickBody.indexOf('homeChoice.decidePick(');
    const refuse = pickBody.indexOf('if (decision.refusal)');
    const write = pickBody.indexOf('setOrientaHome(');
    expect(decide).toBeGreaterThan(0);
    expect(refuse).toBeGreaterThan(decide);
    expect(write).toBeGreaterThan(refuse);
    expect(pickBody.slice(refuse, write)).toMatch(/return {/);
  });

  it('does not create the recorded folder while it is not proven gone: the guard precedes the mkdir', () => {
    const start = MAIN.indexOf('function usableCacheDir()');
    const body = MAIN.slice(start, MAIN.indexOf('const cacheDir = usableCacheDir();', start));
    const guard = body.indexOf('homeChoice.mustNotCreateHome(homeNotice)');
    const mkdir = body.indexOf('fs.mkdirSync(dir');
    expect(guard).toBeGreaterThan(0);
    expect(mkdir).toBeGreaterThan(guard);
    expect(body.slice(guard, mkdir)).toMatch(/return null;/);
  });

  it('settles a stale pointer BEFORE anything creates the data folder', () => {
    // usableCacheDir() creates <home>/electron recursively: after it, a stale
    // pointer has silently re-created the whole chain.
    const settle = MAIN.indexOf('homeChoice.settleHomeAtStartup(');
    const create = MAIN.indexOf('const cacheDir = usableCacheDir();');
    expect(settle).toBeGreaterThan(0);
    expect(create).toBeGreaterThan(settle);
  });

  it('tells it whether this is a packaged build, and uses it for the probe', () => {
    // Behaviour is run in 'settleHomeAtStartup' above; these pin the call sites.
    expect(MAIN).toMatch(/settleHomeAtStartup\(\{\s*isPackaged: app\.isPackaged,/);
    expect(MAIN).toMatch(/homeUnreachable: homeChoice\.unreachableForProbe\(homeNotice\)/);
  });

  it('keeps the failure of the check and writes it to the shell log once that is open', () => {
    expect(MAIN).toMatch(/onError: \(err\) => \{ homeCheckError = err;/);
    const open = MAIN.indexOf('openStartupLog(path.join(orientaHome()');
    const log = MAIN.indexOf('homeChoice.startupLogLines(');
    expect(open).toBeGreaterThan(0);
    expect(log).toBeGreaterThan(open);
    expect(MAIN.slice(log, log + 200)).toMatch(/checkError: homeCheckError/);
  });

  it('gives the startup check nothing but isPackaged and onError: no stand-in for the reconciler', () => {
    // `reconcile` exists for tests. A call site that passes one would turn the
    // whole check off in the shipped app and every behavioural test above would
    // stay green, because they call the function themselves.
    const call = MAIN.slice(MAIN.indexOf('homeChoice.settleHomeAtStartup({'));
    const body = call.slice(0, call.indexOf('\n});'));
    const keys = [...body.matchAll(/^ {2}(\w+):/gm)].map((m) => m[1]);
    expect(keys).toEqual(['isPackaged', 'onError']);
  });

  it('builds the probe detail with it', () => {
    expect(MAIN).toMatch(/reason: homeChoice\.unusableHomeReason\(\{[^}]*notice: homeNotice/);
  });

  it('hands the notice to the wizard', () => {
    expect(MAIN).toMatch(/handleSetup\('setup:context', \(\) => \(\{[\s\S]*?homeNotice,/);
  });
});
