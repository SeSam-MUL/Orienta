/**
 * What this launch should do, decided ONCE.
 *
 * Earlier drafts let createWindow() ask "does .python_path exist" while
 * whenReady() asked "does findPython() return something usable". The two
 * disagree for a user whose interpreter was deleted, moved or quarantined:
 * no wizard is shown AND nothing is started, so the window sits on Chromium's
 * ERR_CONNECTION_REFUSED page forever, with no dialog and no log. Six separate
 * review findings had that one root cause.
 *
 * `startupDecision()` is the single answer both sides read, and it never
 * throws — it is called from app.whenReady() and, through startBackend(), from
 * a setTimeout, where a throw is an unhandled rejection and the window dies
 * with no message at all.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  orientaHome,
  statOrThrow,
  installedInterpreter,
  installedRuntime,
  startupDecision,
  pendingUpdate,
  resolveProjectRoot,
  openStartupLog,
  closeStartupLog,
  logShellLine,
} from '../../electron/project_root.js';

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..');

let home;

beforeEach(() => {
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-'));
  process.env.ORIENTA_HOME = home;
  delete process.env.ORIENTA_PROJECT_ROOT;
});

afterEach(() => {
  closeStartupLog();
  delete process.env.ORIENTA_HOME;
  delete process.env.ORIENTA_PROJECT_ROOT;
  fs.rmSync(home, { recursive: true, force: true });
});

const at = (...p) => path.join(home, ...p);

function seedRuntime(version = 'v0.3.0') {
  fs.mkdirSync(at('runtime'), { recursive: true });
  fs.writeFileSync(at('runtime', 'VERSION'), `${version}\n`);
}

function seedInterpreter() {
  const exe = at('python', 'python.exe');
  fs.mkdirSync(path.dirname(exe), { recursive: true });
  fs.writeFileSync(exe, '');
  fs.writeFileSync(at('.python_path'), `${exe}\n`);
  return exe;
}

// --------------------------------------------------------------------------
// orientaHome — one definition, shared with the setup and with Python
// --------------------------------------------------------------------------

describe('orientaHome', () => {
  it('honours ORIENTA_HOME', () => {
    expect(orientaHome()).toBe(home);
  });

  it('refuses a relative override', () => {
    // This path is the parent of a directory the updater deletes and of the
    // user's crystal library; resolving it against an arbitrary cwd is not an
    // acceptable outcome.
    process.env.ORIENTA_HOME = 'some/relative/dir';
    expect(() => orientaHome()).toThrow(/absolute/i);
  });
});

// --------------------------------------------------------------------------
// statOrThrow — "absent" and "cannot tell" are different answers
// --------------------------------------------------------------------------

describe('statOrThrow', () => {
  it('returns null for a missing path', () => {
    expect(statOrThrow(at('nope'))).toBeNull();
  });

  it('returns a stat for an existing one', () => {
    seedRuntime();
    expect(statOrThrow(at('runtime')).isDirectory()).toBe(true);
  });
});

// --------------------------------------------------------------------------
// installedInterpreter
// --------------------------------------------------------------------------

describe('installedInterpreter', () => {
  it('is null when nothing is installed', () => {
    expect(installedInterpreter()).toBeNull();
  });

  it('returns the recorded interpreter when it is there', () => {
    const exe = seedInterpreter();
    expect(installedInterpreter().path).toBe(exe);
  });

  it('reports a recorded interpreter that has gone missing, rather than null', () => {
    // A deleted or quarantined python/ must be distinguishable from "never
    // installed": one needs a repair, the other needs the wizard.
    seedInterpreter();
    fs.rmSync(at('python'), { recursive: true, force: true });
    const found = installedInterpreter();
    expect(found).not.toBeNull();
    expect(found.path).toBeNull();
    expect(found.error).toMatch(/not there|missing/i);
  });

  it('reports an empty .python_path as an error, not as absent', () => {
    fs.writeFileSync(at('.python_path'), '   \n');
    const found = installedInterpreter();
    expect(found.path).toBeNull();
    expect(found.error).toBeTruthy();
  });

  it('never consults conda or the system PATH', () => {
    // The whole reason this exists: findPython() ends with
    // `return isWin ? 'python' : 'python3'`, and on a clean Windows machine
    // that is an App Execution Alias which opens the Microsoft Store.
    expect(installedInterpreter()).toBeNull();
  });
});

// --------------------------------------------------------------------------
// startupDecision — the single answer
// --------------------------------------------------------------------------

describe('startupDecision', () => {
  it('is dev when a checkout is named explicitly', () => {
    process.env.ORIENTA_PROJECT_ROOT = home;
    expect(startupDecision().mode).toBe('dev');
  });

  it('is setup when nothing is installed', () => {
    expect(startupDecision().mode).toBe('setup');
  });

  it('is run when an interpreter and a runtime are both there', () => {
    seedRuntime();
    const exe = seedInterpreter();
    const decision = startupDecision();
    expect(decision.mode).toBe('run');
    expect(decision.python).toBe(exe);
    expect(decision.root).toBe(at('runtime'));
  });

  it('is repair — never run — when the interpreter is recorded but gone', () => {
    seedRuntime();
    seedInterpreter();
    fs.rmSync(at('python'), { recursive: true, force: true });
    const decision = startupDecision();
    expect(decision.mode).toBe('repair');
    expect(decision.message).toBeTruthy();
  });

  it('is repair when the interpreter is there but the runtime is not', () => {
    seedInterpreter();
    const decision = startupDecision();
    expect(decision.mode).toBe('repair');
    // Assert the PROMISE, not the wording. The message deliberately says
    // "program files" rather than "runtime directory", because the person
    // reading it is a materials scientist, and the load-bearing part is the
    // reassurance that their crystal library survives a reinstall.
    expect(decision.message).toMatch(/not touched/i);
    expect(decision.message.length).toBeGreaterThan(20);
  });

  it('is repair when the runtime is there but the interpreter is not', () => {
    seedRuntime();
    expect(startupDecision().mode).toBe('repair');
  });

  it('never throws, whatever it finds', () => {
    fs.writeFileSync(at('.python_path'), '');
    expect(() => startupDecision()).not.toThrow();
    fs.writeFileSync(at('.python_path'), 'not a path at all');
    expect(() => startupDecision()).not.toThrow();
  });

  it('never returns run without both a python and a root', () => {
    for (const seed of [() => {}, seedRuntime, seedInterpreter]) {
      fs.rmSync(home, { recursive: true, force: true });
      fs.mkdirSync(home, { recursive: true });
      seed();
      const d = startupDecision();
      if (d.mode === 'run') {
        expect(d.python).toBeTruthy();
        expect(d.root).toBeTruthy();
      }
    }
  });
});

// --------------------------------------------------------------------------
// pendingUpdate
// --------------------------------------------------------------------------

describe('pendingUpdate', () => {
  it('is null when nothing is parked', () => {
    seedRuntime();
    expect(pendingUpdate()).toBeNull();
  });

  it('reports the tag when a package is parked', () => {
    seedRuntime();
    fs.writeFileSync(at('pending.json'), JSON.stringify({ tag: 'v0.4.1', file: 'x.zip' }));
    expect(pendingUpdate().tag).toBe('v0.4.1');
  });

  it('reports an unreadable record rather than pretending there is none', () => {
    seedRuntime();
    fs.writeFileSync(at('pending.json'), '{ this is not json');
    expect(pendingUpdate().error).toMatch(/record/i);
  });
});

// --------------------------------------------------------------------------
// resolveProjectRoot
// --------------------------------------------------------------------------

describe('resolveProjectRoot', () => {
  it('prefers an explicit ORIENTA_PROJECT_ROOT', () => {
    process.env.ORIENTA_PROJECT_ROOT = home;
    expect(resolveProjectRoot()).toBe(home);
  });

  it('uses the installed runtime when one exists', () => {
    seedRuntime();
    expect(resolveProjectRoot()).toBe(at('runtime'));
  });

  it('falls back to the checkout when nothing is installed', () => {
    expect(resolveProjectRoot()).toBe(REPO_ROOT);
  });
});

// --------------------------------------------------------------------------
// the shell's own log
// --------------------------------------------------------------------------

describe('openStartupLog / logShellLine', () => {
  it('writes where the diagnostics collector can find it', () => {
    const dir = at('logs');
    openStartupLog(dir);
    logShellLine('hello from the shell');
    const text = fs.readFileSync(path.join(dir, 'orienta-shell.log'), 'utf8');
    expect(text).toContain('hello from the shell');
  });

  it('does not throw when the directory cannot be created', () => {
    // It guards every other guard: openBackendLog() is opened only inside
    // startBackend(), so in every state where startup fails before the backend
    // there is no stream at all and a packaged app has no console either.
    const blocked = path.join(at('a-file'));
    fs.writeFileSync(blocked, 'not a directory');
    expect(() => openStartupLog(path.join(blocked, 'logs'))).not.toThrow();
    expect(() => logShellLine('still fine')).not.toThrow();
  });

  it('is idempotent', () => {
    const dir = at('logs');
    openStartupLog(dir);
    openStartupLog(dir);
    logShellLine('once');
    const text = fs.readFileSync(path.join(dir, 'orienta-shell.log'), 'utf8');
    expect(text.match(/once/g)).toHaveLength(1);
  });
});
