/**
 * What the shell writes and shows when the backend process ends (electron/backend_exit.js).
 *
 * Before 0.4.8 an end left "Backend exited with code N" at best, and nothing when
 * the app itself killed the backend, so an unexpected end could not be told from
 * a normal close, and the page was never told at all.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { describeExit, readBackendState, backendEndLine, backendEndNotice } =
  require('../../electron/backend_exit.js');

const here = path.dirname(fileURLToPath(import.meta.url));
const MAIN = fs.readFileSync(path.join(here, '../../electron/main.js'), 'utf8');
const PRELOAD = fs.readFileSync(path.join(here, '../../electron/preload.js'), 'utf8');

const STATE = {
  pid: 4242, uptime_s: 81.5, written: '2026-11-08 18:30:01',
  last_request: { method: 'POST', path: '/api/phase-map/apply-to-grain', at: '18:29:40', running_s: 21.0 },
  in_flight: [{ method: 'POST', path: '/api/phase-map/apply-to-grain', at: '18:29:40', running_s: 21.0 }],
  memory_mib: { commit: 41250, rss: 38012 },
};

describe('describeExit', () => {
  it('names a native crash on Windows from its NTSTATUS', () => {
    expect(describeExit(3221225477, null, 'win32')).toBe(
      'exit code 0xC0000005 (access violation: a native crash in Python or a library)');
    expect(describeExit(-1073741819, null, 'win32')).toContain('0xC0000005');
    expect(describeExit(0xC000012D, null, 'win32')).toContain('commit limit');
  });

  it('says what 0 and 1 mean for this backend', () => {
    expect(describeExit(0, null, 'win32')).toContain('ended itself');
    expect(describeExit(1, null, 'linux')).toContain('Python error');
  });

  it('names a signal and what SIGKILL usually is', () => {
    expect(describeExit(null, 'SIGKILL', 'linux')).toContain('killed from outside');
    expect(describeExit(null, 'SIGTERM', 'darwin')).toBe('signal SIGTERM');
  });

  it('an unknown Windows error status is still shown in hex', () => {
    expect(describeExit(0xC0001234, null, 'win32')).toBe('exit code 0xC0001234 (a Windows error status)');
    expect(describeExit(42, null, 'win32')).toBe('exit code 42');
  });
});

describe('readBackendState', () => {
  it('reads only the state file of the process that ended', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-state-'));
    const file = path.join(dir, 'backend-state.json');
    fs.writeFileSync(file, JSON.stringify(STATE));
    expect(readBackendState(file, 4242).last_request.path).toBe('/api/phase-map/apply-to-grain');
    // a file an earlier session left is not this process's last state
    expect(readBackendState(file, 999)).toBeNull();
    expect(readBackendState(path.join(dir, 'missing.json'), 4242)).toBeNull();
    fs.writeFileSync(file, '{half a file');
    expect(readBackendState(file, 4242)).toBeNull();
  });
});

describe('backendEndLine', () => {
  it('an unexpected end carries the last state', () => {
    const line = backendEndLine({ code: 3221225477, signal: null, stopReason: null, state: STATE, platform: 'win32' });
    expect(line).toContain('Backend ended: exit code 0xC0000005');
    expect(line).toContain('unexpected end');
    expect(line).toContain('still running: POST /api/phase-map/apply-to-grain (21 s)');
    expect(line).toContain('commit 41250 MiB');
  });

  it('an end the app caused says why', () => {
    const line = backendEndLine({ code: 1, signal: null, stopReason: 'the app is quitting', state: null, platform: 'win32' });
    expect(line).toContain('Stopped by the app: the app is quitting.');
    expect(line).toContain('No state file from this backend process.');
  });

  it('quotes the reason the backend gave itself', () => {
    const line = backendEndLine({ code: 0, signal: null, stopReason: null,
      state: { ...STATE, ended: 'parent watchdog: the launching process is gone' }, platform: 'win32' });
    expect(line).toContain('it said it was ending: parent watchdog');
  });
});

describe('backendEndNotice', () => {
  it('is plain data for the page', () => {
    const n = backendEndNotice({ code: 3221225477, signal: null, state: STATE, restarting: true, platform: 'win32' });
    expect(n).toMatchObject({ lastRequest: 'POST /api/phase-map/apply-to-grain', commitMiB: 41250, restarting: true });
    expect(n.exit).toContain('0xC0000005');
    expect(JSON.parse(JSON.stringify(n))).toEqual(n);
  });
});

describe('main.js wiring', () => {
  it('logs every end with the line, and only an unexpected one restarts', () => {
    const handler = MAIN.slice(MAIN.indexOf("backendProcess.on('exit'"), MAIN.indexOf('function notifyBackendEnded'));
    expect(handler).toContain('backendExit.backendEndLine(');
    expect(handler).toContain('readBackendState(stateFile, proc.pid)');
    expect(handler.indexOf('if (stopReason) return;')).toBeGreaterThan(-1);
    expect(handler.indexOf('if (stopReason) return;')).toBeLessThan(handler.indexOf('maybeRestartBackend('));
    // the exit of a replaced process must not forget the current one
    expect(handler).toContain('if (backendProcess === proc) backendProcess = null;');
  });

  it('every kill by the app states its reason', () => {
    expect(MAIN).toMatch(/function killBackend\(reason = /);
    // call sites: code lines, not the comments that mention the function
    const calls = MAIN.split(/\r?\n/)
      .filter((l) => !/^\s*(\/\/|\*)/.test(l))
      .flatMap((l) => l.match(/killBackend\([^)]*\)/g) || [])
      .filter((c) => !c.startsWith('killBackend(reason'));
    expect(calls.length).toBeGreaterThanOrEqual(2);
    for (const c of calls) expect(c, c).toMatch(/killBackend\('[^']+'\)/);
  });

  it('gives the backend the state file and tells the page about an unexpected end', () => {
    expect(MAIN).toContain('ORIENTA_BACKEND_STATE_FILE: stateFile');
    expect(MAIN).toContain("mainWindow.webContents.send('backend:ended', notice)");
    expect(MAIN).toContain("ipcMain.handle('backend:lastEnd'");
    const restart = MAIN.slice(MAIN.indexOf('async function maybeRestartBackend'));
    expect(restart.slice(0, 400)).toContain('notifyBackendEnded(');
    expect(PRELOAD).toContain("ipcRenderer.on('backend:ended'");
    expect(PRELOAD).toContain("ipcRenderer.invoke('backend:lastEnd')");
  });

  it('says "restarting" only on the paths that start the backend again', () => {
    const restart = MAIN.slice(MAIN.indexOf('async function maybeRestartBackend'),
      MAIN.indexOf('// 3 minutes of wall-clock time'));
    // the port is busy: the answer of the busy-port dialog decides
    expect(restart).toMatch(/const freed = await resolveBusyPort\(\);\s*tell\(freed\);/);
    // the restart limit: never "restarting"
    expect(restart.indexOf('tell(false)')).toBeLessThan(restart.indexOf("'Backend keeps stopping'"));
    // the timer path: "restarting", told before the start is scheduled
    expect(restart.indexOf('tell(true)')).toBeGreaterThan(restart.indexOf('backendRestarts += 1'));
    expect(restart.indexOf('tell(true)')).toBeLessThan(restart.indexOf('setTimeout('));
  });

  it('keeps the stop reason on the process it stopped', () => {
    expect(MAIN).not.toContain('backendStopReason');
    expect(MAIN).toContain('backendProcess.orientaStopReason = reason;');
    expect(MAIN).toContain('const stopReason = proc.orientaStopReason ||');
  });
});
