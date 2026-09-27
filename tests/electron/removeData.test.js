/**
 * Removing the data folder, tested against real folders on disk.
 *
 * This is the one piece of the application that deletes gigabytes on purpose,
 * and macOS and Linux have no uninstaller to ask the questions Windows asks.
 * A mocked filesystem would prove nothing here: the interesting cases are a
 * symlink out of the folder, a file the user put inside it, and a folder that
 * is not ours at all. So every test builds the real thing in a temp directory.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const { planRemoval, removeData, REASONS } = createRequire(import.meta.url)('../../electron/remove_data.js');

let tmp;
beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-rm-')); });
afterEach(() => { try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* gone */ } });

/** A data folder as the setup leaves it. */
function makeHome(extra = {}) {
  const home = path.join(tmp, 'Orienta');
  fs.mkdirSync(path.join(home, 'runtime', 'Database'), { recursive: true });
  fs.mkdirSync(path.join(home, 'python', 'bin'), { recursive: true });
  fs.mkdirSync(path.join(home, 'logs'), { recursive: true });
  fs.writeFileSync(path.join(home, '.orienta-home'), '');
  fs.writeFileSync(path.join(home, 'runtime', 'VERSION'), 'v0.4.5\n');
  fs.writeFileSync(path.join(home, 'runtime', 'backend.py'), '# code');
  fs.writeFileSync(path.join(home, 'runtime', 'Database', 'Al.cif'), 'data');
  for (const [name, content] of Object.entries(extra)) {
    fs.writeFileSync(path.join(home, name), content);
  }
  return home;
}

describe('what may be removed at all', () => {
  it('accepts a folder the setup made', () => {
    const plan = planRemoval(makeHome(), fs, tmp);
    expect(plan.ok).toBe(true);
    expect(plan.entries).toContain('runtime');
    expect(plan.library).toBe(path.join(tmp, 'Orienta', 'runtime', 'Database'));
  });

  it('accepts an older install without the marker', () => {
    const home = makeHome();
    fs.rmSync(path.join(home, '.orienta-home'));
    fs.writeFileSync(path.join(home, '.python_path'), 'x');
    expect(planRemoval(home, fs, tmp).ok).toBe(true);
  });

  it('refuses a folder that is not ours, however full it looks', () => {
    const other = path.join(tmp, 'Documents');
    fs.mkdirSync(path.join(other, 'runtime'), { recursive: true });
    fs.writeFileSync(path.join(other, 'thesis.docx'), 'x');
    const plan = planRemoval(other, fs, tmp);
    expect(plan.ok).toBe(false);
    expect(plan.reason).toBe(REASONS.notOurs);
    expect(plan.entries).toEqual([]);   // a refusal never carries a list
  });

  it('refuses a checkout, however many markers it carries', () => {
    const home = makeHome();
    fs.mkdirSync(path.join(home, '.git'));
    expect(planRemoval(home, fs, tmp).reason).toBe(REASONS.isRepo);
  });

  it('refuses the home directory itself', () => {
    const home = makeHome();
    expect(planRemoval(home, fs, home).reason).toBe(REASONS.isHome);
  });

  it('refuses a root, for being a root', () => {
    // Asserting only `ok === false` proves nothing here: a root is also
    // refused by the marker check three lines later, so the rule this test
    // names would never be exercised.
    expect(planRemoval(path.parse(tmp).root, fs, tmp).reason).toBe(REASONS.tooShallow);
    expect(planRemoval('/Volumes/Backup', fs, tmp).reason).toBe(REASONS.tooShallow);
    expect(planRemoval('/mnt/data', fs, tmp).reason).toBe(REASONS.tooShallow);
  });

  it('refuses a symlink instead of following it', (ctx) => {
    const home = makeHome();
    const link = path.join(tmp, 'link-to-home');
    // An unprivileged Windows account cannot make links. Say so out loud:
    // a silent `return` here would leave a green test that proves nothing.
    try { fs.symlinkSync(home, link, 'junction'); } catch { return ctx.skip(); }
    const plan = planRemoval(link, fs, tmp);
    expect(plan.ok).toBe(false);
    expect(plan.reason).toBe(REASONS.isLink);
    expect(fs.existsSync(path.join(home, 'runtime'))).toBe(true);
  });

  it('refuses a folder that is not there', () => {
    expect(planRemoval(path.join(tmp, 'nope'), fs, tmp).reason).toBe(REASONS.noFolder);
  });
});

describe('what actually happens', () => {
  it('keeps the crystal library unless it is asked for separately', () => {
    const home = makeHome();
    const result = removeData(home, { removeLibrary: false, io: fs, homedir: tmp });
    expect(result.ok).toBe(true);
    expect(result.libraryRemoved).toBe(false);
    expect(fs.existsSync(path.join(home, 'runtime', 'Database', 'Al.cif'))).toBe(true);
    // ...while the program itself is gone
    expect(fs.existsSync(path.join(home, 'runtime', 'backend.py'))).toBe(false);
    expect(fs.existsSync(path.join(home, 'python'))).toBe(false);
  });

  it('removes the library only on the second yes', () => {
    const home = makeHome();
    const result = removeData(home, { removeLibrary: true, io: fs, homedir: tmp });
    expect(result.libraryRemoved).toBe(true);
    expect(fs.existsSync(path.join(home, 'runtime'))).toBe(false);
  });

  it('leaves a file the user put in the folder', () => {
    const home = makeHome({ 'my-notes.txt': 'do not delete me' });
    const result = removeData(home, { removeLibrary: true, io: fs, homedir: tmp });
    expect(fs.readFileSync(path.join(home, 'my-notes.txt'), 'utf8')).toBe('do not delete me');
    expect(result.kept).toContain('my-notes.txt');
  });

  it('unlinks a symlink inside the folder instead of deleting its target', (ctx) => {
    const home = makeHome();
    const outside = path.join(tmp, 'precious');
    fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, 'data.h5'), 'measurements');
    try { fs.symlinkSync(outside, path.join(home, 'logs', 'link'), 'junction'); } catch { return ctx.skip(); }
    removeData(home, { removeLibrary: true, io: fs, homedir: tmp });
    expect(fs.existsSync(path.join(outside, 'data.h5'))).toBe(true);
  });

  it('removes the pointer marker last, so an interrupted run can resume', () => {
    const home = makeHome();
    const result = removeData(home, { removeLibrary: true, io: fs, homedir: tmp });
    const marker = path.join(home, '.orienta-home');
    expect(result.removed[result.removed.length - 1]).toBe(marker);
    expect(fs.existsSync(marker)).toBe(false);
  });

  it('does nothing at all when the plan refuses', () => {
    const other = path.join(tmp, 'Documents');
    fs.mkdirSync(other, { recursive: true });
    fs.writeFileSync(path.join(other, 'thesis.docx'), 'x');
    const result = removeData(other, { removeLibrary: true, io: fs, homedir: tmp });
    expect(result.ok).toBe(false);
    expect(result.removed).toEqual([]);
    expect(fs.existsSync(path.join(other, 'thesis.docx'))).toBe(true);
  });
});

describe('the two ways this could have deleted the wrong thing', () => {
  it('never reads through a linked runtime, even when keeping the library', (ctx) => {
    // Measured before the fix: the checkout's files were deleted one by one.
    // The trigger is the SAFE-looking answer -- "keep my library" is what
    // makes the code enumerate the link's children.
    const home = path.join(tmp, 'Orienta');
    const checkout = path.join(tmp, 'my-checkout');
    fs.mkdirSync(path.join(checkout, 'src'), { recursive: true });
    fs.writeFileSync(path.join(checkout, 'src', 'thesis.py'), 'years of work');
    fs.mkdirSync(path.join(checkout, 'Database'), { recursive: true });
    fs.writeFileSync(path.join(checkout, 'Database', 'Al.cif'), 'x');
    fs.mkdirSync(home, { recursive: true });
    fs.writeFileSync(path.join(home, '.orienta-home'), '');
    try { fs.symlinkSync(checkout, path.join(home, 'runtime'), 'junction'); } catch { return ctx.skip(); }

    const result = removeData(home, { removeLibrary: false, io: fs, homedir: tmp });

    expect(fs.readFileSync(path.join(checkout, 'src', 'thesis.py'), 'utf8')).toBe('years of work');
    expect(fs.existsSync(path.join(checkout, 'Database', 'Al.cif'))).toBe(true);
    expect(fs.existsSync(path.join(home, 'runtime'))).toBe(false);   // the link itself goes
    expect(result.ok).toBe(true);
  });

  it('keeps the library whatever case the filesystem spells it in', () => {
    // APFS and NTFS are case-insensitive, so the library is FOUND as
    // "Database" and then skipped by an exact string compare that misses.
    // The user answers "keep it" and it is deleted, and the report says kept.
    const home = path.join(tmp, 'Orienta');
    fs.mkdirSync(path.join(home, 'runtime', 'database'), { recursive: true });
    fs.writeFileSync(path.join(home, '.orienta-home'), '');
    fs.writeFileSync(path.join(home, 'runtime', 'database', 'Al.cif'), 'data');
    fs.writeFileSync(path.join(home, 'runtime', 'backend.py'), 'code');

    const plan = planRemoval(home, fs, tmp);
    expect(plan.library).toBeTruthy();          // it is asked about
    removeData(home, { removeLibrary: false, io: fs, homedir: tmp });

    expect(fs.existsSync(path.join(home, 'runtime', 'database', 'Al.cif'))).toBe(true);
    expect(fs.existsSync(path.join(home, 'runtime', 'backend.py'))).toBe(false);
  });
});

describe('what it says when it did not finish', () => {
  it('keeps the marker when something of ours is left, so a retry still works', () => {
    const home = makeHome();
    const first = removeData(home, { removeLibrary: false, io: fs, homedir: tmp });
    expect(first.finished).toBe(false);
    expect(fs.existsSync(path.join(home, '.orienta-home'))).toBe(true);
    // ...and the second attempt is not told "this is not Orienta's folder"
    expect(planRemoval(home, fs, tmp).ok).toBe(true);
  });

  it('reports entries it could not remove instead of a green "done"', () => {
    const home = makeHome();
    const io = {
      ...fs,
      rmSync: (target, options) => {
        if (String(target).endsWith('python')) { const e = new Error('EBUSY'); throw e; }
        return fs.rmSync(target, options);
      },
    };
    const result = removeData(home, { removeLibrary: true, io, homedir: tmp });
    expect(result.failed.length).toBe(1);
    expect(result.finished).toBe(false);
    expect(fs.existsSync(path.join(home, '.orienta-home'))).toBe(true);
  });

  it('does not claim to have deleted a library that was a link', (ctx) => {
    const home = makeHome();
    const outside = path.join(tmp, 'nas-library');
    fs.mkdirSync(outside, { recursive: true });
    fs.writeFileSync(path.join(outside, 'Al.cif'), 'shared');
    fs.rmSync(path.join(home, 'runtime', 'Database'), { recursive: true, force: true });
    try { fs.symlinkSync(outside, path.join(home, 'runtime', 'Database'), 'junction'); } catch { return ctx.skip(); }

    const result = removeData(home, { removeLibrary: true, io: fs, homedir: tmp });

    expect(fs.existsSync(path.join(outside, 'Al.cif'))).toBe(true);   // it survives
    expect(result.libraryRemoved).toBe(false);                         // and we say so
    expect(result.libraryWasLink).toBe(true);
  });

  it('says it finished only when the folder really is ours-free', () => {
    const home = makeHome();
    const result = removeData(home, { removeLibrary: true, io: fs, homedir: tmp });
    expect(result.finished).toBe(true);
    expect(fs.existsSync(path.join(home, '.orienta-home'))).toBe(false);
  });
});
