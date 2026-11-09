/**
 * Two start-up and file-open problems of the desktop app, pinned in main.js.
 *
 * main.js cannot be imported in a test (it spawns the backend on load), so its
 * wiring is read as text, as in backendWait.test.js and appName.test.js.
 *
 * 1. A dev start loaded the dev server while the backend was still starting:
 *    the page's first requests failed and were never repeated, and with the
 *    dev port on the backend the window stayed blank until Ctrl+R.
 * 2. Every open dialog started in Downloads.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
// Sources are read with CRLF turned into LF: a Windows checkout with core.autocrlf=true
// gives them CRLF line endings, and the searches below are written with LF.
const MAIN = fs.readFileSync(path.join(here, '../../electron/main.js'), 'utf8').replace(/\r\n/g, '\n');

function body(startPattern, endPattern) {
  const start = MAIN.search(startPattern);
  expect(start).toBeGreaterThan(-1);
  const rest = MAIN.slice(start);
  const end = rest.search(endPattern);
  expect(end).toBeGreaterThan(0);
  return rest.slice(0, end);
}

const DEV_LOAD = /loadURL\(`http:\/\/localhost:\$\{FRONTEND_DEV_PORT\}`\)/;

describe('dev start', () => {
  it('does not load the dev server when the window is created', () => {
    const createWindow = body(/function createWindow\(/, /\n}\n/);
    expect(createWindow).not.toMatch(DEV_LOAD);
    expect(createWindow).toMatch(/loadFile\(path\.join\(__dirname, 'setup', 'waiting\.html'\)\)/);
  });

  it('loads the dev server only after the backend answered', () => {
    const waited = MAIN.search(/await waitForBackend\(\);/);
    const devLoad = MAIN.search(DEV_LOAD);
    expect(waited).toBeGreaterThan(-1);
    expect(devLoad).toBeGreaterThan(waited);
    expect(MAIN.slice(waited, devLoad)).toMatch(/plan\.thenLoad === 'devServer'/);
  });

  it('says so when the dev server does not answer, instead of a frozen waiting page', () => {
    const devLoad = MAIN.search(DEV_LOAD);
    const after = MAIN.slice(devLoad, devLoad + 800);
    expect(after).toMatch(/\.catch\(\(err\) => \{[\s\S]*dialog\.showErrorBox\('Dev server not reachable'/);
  });
});

describe('dialog folder memory', () => {
  it('survives a home that cannot be resolved: the dialog opens without the memory', () => {
    const helper = body(/function dialogHome\(\)/, /\n}\n/);
    expect(helper).toMatch(/try \{\s*return orientaHome\(\);\s*\} catch/);
    expect(helper).toMatch(/return null;/);
    // no dialog handler reaches orientaHome() directly any more
    for (const channel of ['dialog:openFile', 'dialog:openFolder', 'dialog:saveFile', 'dialog:saveImage']) {
      const handler = body(new RegExp(`ipcMain\\.handle\\('${channel}'`), /\n}\);\n/);
      expect(handler, channel).not.toMatch(/orientaHome\(\)/);
    }
  });
});

describe('open dialogs', () => {
  it('open a file where the last file was opened, and remember it', () => {
    const handler = body(/ipcMain\.handle\('dialog:openFile'/, /\n}\);\n/);
    expect(handler).toMatch(/readLastDir\(dialogHome\(\), 'open'\)/);
    expect(handler).toMatch(/requested: options\?\.defaultPath/);
    expect(handler).toMatch(/rememberDir\(dialogHome\(\), result\.filePaths\[0\], \{ kind: 'open' \}\)/);
  });

  it('pick a folder to read in the open memory, an export folder in the save memory', () => {
    const handler = body(/ipcMain\.handle\('dialog:openFolder'/, /\n}\);\n/);
    expect(handler).toMatch(/const kind = remember \? 'save' : 'open';/);
    expect(handler).toMatch(/readLastDir\(dialogHome\(\), kind\)/);
    expect(handler).toMatch(/rememberDir\(dialogHome\(\), result\.filePaths\[0\], \{ isDirectory: true, kind \}\)/);
  });
});
