/**
 * "Show log files" in Settings opens the folder the backend writes its logs to.
 *
 * Two decisions are pinned here. (1) The folder is worked out in the main
 * process, from the same project root openBackendLog() writes
 * backend-console.log under; the renderer sends no path, so the channel cannot
 * be used to open an arbitrary folder. (2) The handler is wired in main.js and
 * exposed through the preload bridge, which are read as source because main.js
 * spawns the backend when loaded.
 */
import { describe, it, expect, vi } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { openLogFolder } = require('../../electron/log_folder.js');

const here = path.dirname(fileURLToPath(import.meta.url));
const MAIN = fs.readFileSync(path.join(here, '../../electron/main.js'), 'utf8');
const PRELOAD = fs.readFileSync(path.join(here, '../../electron/preload.js'), 'utf8');

describe('openLogFolder', () => {
  it('opens <project root>/logs, creating it when absent, and returns that path', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-logs-'));
    const shell = { openPath: vi.fn().mockResolvedValue('') };
    const result = await openLogFolder({ shell, projectRoot: root });
    const expected = path.join(root, 'logs');
    expect(shell.openPath).toHaveBeenCalledWith(expected);
    expect(result).toEqual({ ok: true, path: expected });
    expect(fs.statSync(expected).isDirectory()).toBe(true);
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('reports the failure instead of throwing when the folder cannot be opened', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-logs-'));
    const shell = { openPath: vi.fn().mockResolvedValue('no file manager') };
    const result = await openLogFolder({ shell, projectRoot: root });
    expect(result.ok).toBe(false);
    expect(result.error).toBe('no file manager');
    expect(result.path).toBe(path.join(root, 'logs'));
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('survives shell.openPath throwing', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-logs-'));
    const shell = { openPath: vi.fn().mockRejectedValue(new Error('boom')) };
    const result = await openLogFolder({ shell, projectRoot: root });
    expect(result).toMatchObject({ ok: false, error: 'boom' });
    fs.rmSync(root, { recursive: true, force: true });
  });
});

describe('wiring', () => {
  it('main.js answers app:openLogFolder from its own project root and takes no path', () => {
    const at = MAIN.indexOf("ipcMain.handle('app:openLogFolder'");
    expect(at).toBeGreaterThan(-1);
    const handler = MAIN.slice(at, at + 200);
    // No parameters at all: nothing the page sends can reach the handler.
    expect(handler).toMatch(/^ipcMain\.handle\('app:openLogFolder',\s*\(\)\s*=>/);
    expect(handler).toMatch(/lastProjectRoot \|\| resolveProjectRoot\(\)/);
    expect(handler).toMatch(/openLogFolder\(/);
  });

  it('the preload bridge exposes it on the application window object', () => {
    expect(PRELOAD).toMatch(/openLogFolder:\s*\(\)\s*=>\s*ipcRenderer\.invoke\('app:openLogFolder'\)/);
    // ... and not on the setup-only object, which is a different boundary.
    const setupAt = PRELOAD.indexOf("exposeInMainWorld('setupAPI'");
    expect(PRELOAD.indexOf('openLogFolder')).toBeLessThan(setupAt);
  });
});
