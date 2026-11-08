/**
 * "Is this folder only Orienta's leftovers?" -- decided by SHAPE.
 *
 * The rule is not "every name in the folder is on a list"
 * (logs, pending, electron, python, runtime ...): a user's own D:\Work\Orienta
 * holding logs\mydata.csv, pending\report.docx or python\myscript.py would be taken
 * for Orienta's data folder, the marker would be written into it, and the uninstaller
 * would delete those folders by name. The cases live in tests/leftover_shapes.json and
 * are run against the NSIS implementation too (tests/test_uninstall_nsis.py).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ELECTRON = path.resolve(import.meta.dirname, '..', '..', 'electron');
const leftovers = require(path.join(ELECTRON, 'leftovers.js'));
const installer = require(path.join(ELECTRON, 'setup', 'installer.js'));
const choice = require(path.join(ELECTRON, 'home_choice.js'));
const removeData = require(path.join(ELECTRON, 'remove_data.js'));
const TABLE = JSON.parse(fs.readFileSync(
  path.resolve(import.meta.dirname, '..', 'leftover_shapes.json'), 'utf8'));

let sandbox;
beforeEach(() => { sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-leftovers-')); });
afterEach(() => { fs.rmSync(sandbox, { recursive: true, force: true }); });

function build(dir, files, dirs = []) {
  for (const sub of dirs) fs.mkdirSync(path.join(dir, ...sub.split('/')), { recursive: true });
  for (const file of files) {
    const full = path.join(dir, ...file.split('/'));
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, 'x');
  }
}

describe('the shared table of cases', () => {
  it('has cases of both kinds, including the four that lost data', () => {
    const names = TABLE.shapes.map((c) => c.name);
    for (const must of ['user_logs', 'user_pending', 'user_python', 'user_runtime']) {
      expect(names, must).toContain(must);
    }
    expect(TABLE.shapes.some((c) => c.leftover)).toBe(true);
    expect(TABLE.shapes.some((c) => !c.leftover)).toBe(true);
  });

  it.each(TABLE.shapes)('foreignEntries: $name -> leftover=$leftover', (c) => {
    const dir = path.join(sandbox, 'Work', 'Orienta');
    fs.mkdirSync(dir, { recursive: true });
    build(dir, c.files, c.dirs);
    const foreign = installer.foreignEntries(dir);
    expect(foreign.length === 0, `foreign: ${foreign.join(', ')}`).toBe(c.leftover);
  });

  it.each(TABLE.shapes.filter((c) => !c.leftover))(
    'a user folder named Orienta holding "$name" is NOT adopted: the Orienta subfolder is added', (c) => {
      const dir = path.join(sandbox, 'Work', 'orienta');
      fs.mkdirSync(dir, { recursive: true });
      build(dir, c.files, c.dirs);
      expect(choice.homeFromPick(dir)).toBe(path.join(dir, 'Orienta'));
      expect(fs.existsSync(path.join(dir, '.orienta-home'))).toBe(false);
    });

  it.each(TABLE.shapes.filter((c) => c.leftover))(
    'a folder named Orienta holding only "$name" IS adopted (an earlier attempt of the wizard)', (c) => {
      const dir = path.join(sandbox, 'Work', 'Orienta');
      fs.mkdirSync(dir, { recursive: true });
      build(dir, c.files, c.dirs);
      expect(choice.homeFromPick(dir)).toBe(dir);
    });

  it.each(TABLE.recognition)('isOrientaHome: $name -> ours=$ours', (c) => {
    const dir = path.join(sandbox, 'Home');
    fs.mkdirSync(dir, { recursive: true });
    build(dir, c.files, c.dirs);
    expect(leftovers.isOrientaHome(dir)).toBe(c.ours);
  });
});

describe('the in-app removal and the other users of the rule agree', () => {
  // remove_data.js had its own copy that accepted any `python` folder where the
  // uninstaller wanted an interpreter.
  const io = (dir) => ({
    ...fs,
    // never aim at the real home: the cleanup plan refuses a home directory
  });

  it.each(TABLE.recognition)('planRemoval: $name -> recognised=$ours', (c) => {
    const dir = path.join(sandbox, 'a', 'Home');
    fs.mkdirSync(dir, { recursive: true });
    build(dir, c.files, c.dirs);
    const plan = removeData.planRemoval(dir, io(dir), path.join(sandbox, 'elsewhere'));
    expect(plan.ok).toBe(c.ours);
    if (!c.ours) expect(plan.reason).toBe('notOurs');
  });

  it('shares its list of names with the setup: one source', () => {
    expect(removeData.OURS).toBe(leftovers.ALL_NAMES);
  });

  it('the folder chooser and the setup call the same function', () => {
    expect(choice.isOrientaHome).toBe(leftovers.isOrientaHome);
    expect(choice.sameFolder).toBe(leftovers.sameFolder);
  });
});

describe('a recognised data folder accepts every name Orienta creates', () => {
  it.each(['python', 'runtime', 'envs', 'micromamba', 'python.old-17', 'electron', 'logs', 'pending'])(
    'marked folder holding a %s', (name) => {
      const dir = path.join(sandbox, 'Home');
      fs.mkdirSync(path.join(dir, name, 'x'), { recursive: true });
      fs.writeFileSync(path.join(dir, name, 'x', 'f'), 'x');
      fs.writeFileSync(path.join(dir, '.orienta-home'), 'x');
      expect(installer.foreignEntries(dir)).toEqual([]);
    });

  it('and the same names without the marker are somebody\'s when they hold something else', () => {
    const dir = path.join(sandbox, 'Home');
    fs.mkdirSync(path.join(dir, 'runtime'), { recursive: true });
    fs.writeFileSync(path.join(dir, 'runtime', 'x.txt'), 'x');
    expect(installer.foreignEntries(dir)).toEqual(['runtime']);
  });
});

describe('the replay of what 0.4.6 left behind', () => {
  // 0.4.6 told the user after every "change..." to close Orienta and start it
  // again, so EVERY chain level holds electron\ and logs\. After the old
  // uninstaller (yes/yes) the default folder is this tree.
  function replay(levels) {
    const base = path.join(sandbox, 'Local', 'Orienta');
    let dir = base;
    for (let i = 0; i < levels; i++) {
      build(dir, ['electron/userData/prefs', 'electron/sessionData/s', 'logs/orienta-shell.log',
        'logs/orienta-shell.log.1', 'logs/orienta-setup.log']);
      dir = path.join(dir, 'Orienta');
    }
    return base;
  }

  it.each([1, 2, 5, 8])('%i levels: the default folder is not "not empty"', (levels) => {
    const base = replay(levels);
    expect(installer.foreignEntries(base)).toEqual([]);
    expect(installer.homeRefusal(base)).toBeNull();
  });

  it('change... + OK on the offered folder is not refused either', () => {
    const base = replay(5);
    const decision = choice.decidePick(path.dirname(base), { current: base });
    expect(decision.refusal).toBeNull();
  });

  it('but one user file anywhere in the chain keeps the refusal, and it does not advise deleting a library', () => {
    const base = replay(5);
    build(path.join(base, 'Orienta', 'Orienta', 'Orienta'), ['logs/mydata.csv']);
    const refusal = installer.homeRefusal(base);
    expect(refusal).not.toBeNull();
    expect(refusal.values.entries).toBe('Orienta');
  });
});
