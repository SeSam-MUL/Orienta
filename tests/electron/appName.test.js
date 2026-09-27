/**
 * The macOS menu bar must say Orienta.
 *
 * M5 tester, 2026-09-25 (tasks/mac-test-m5-2026-09-25/bericht.md, point 4):
 * the menu bar of the signed dmg read "About kikuchipy-gui" and
 * "Quit kikuchipy-gui". macOS builds those labels from `app.name`, and Electron
 * reads that from package.json `name`; `productName` lives under `build`, where
 * only electron-builder looks.
 *
 * This corrects an earlier audit of mine (tasks/mac-tester/naming-audit.md)
 * which reasoned that `name` was "only visible in a dev run". It is visible in
 * the menu bar of the packaged app, and the tester's report is the measurement
 * that says so.
 *
 * main.js cannot be imported in a test — it spawns the backend on load — so
 * these read the source. What is under test is a decision, not a computation.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const MAIN = fs.readFileSync(path.join(here, '../../electron/main.js'), 'utf8');
const PKG = JSON.parse(
  fs.readFileSync(path.join(here, '../../frontend/package.json'), 'utf8'));

describe('the application name', () => {
  it('is set explicitly, so the menu bar cannot inherit the package name', () => {
    expect(MAIN).toMatch(/app\.setName\('Orienta'\)/);
  });

  it('is set before the app is ready', () => {
    // The default menu is built at ready; naming it afterwards is too late.
    const setName = MAIN.indexOf("app.setName('Orienta')");
    // The literal call, not the word: a comment near the top of the file
    // mentions whenReady and made the first version of this test fail on the
    // comment rather than on the code.
    const ready = MAIN.indexOf('app.whenReady(');
    expect(setName).toBeGreaterThan(-1);
    expect(ready).toBeGreaterThan(-1);
    expect(setName).toBeLessThan(ready);
  });

  it('does not move the userData directory', () => {
    // That directory holds the renderer's localStorage: saved phase colours,
    // the dashboard background, preferences. setName changes its default, so
    // the rename is paired with a guard that puts it back.
    const rename = MAIN.indexOf("app.setName('Orienta')");
    const after = MAIN.slice(rename, rename + 400);
    expect(after).toMatch(/userDataBeforeRename/);
    expect(after).toMatch(/app\.setPath\('userData', userDataBeforeRename\)/);
    // and the value is captured BEFORE the rename
    expect(MAIN.slice(0, rename)).toMatch(/const userDataBeforeRename = app\.getPath\('userData'\)/);
  });

  it('keeps productName out of the top level of package.json', () => {
    // A top-level productName would change app.name before main.js runs, and
    // the guard above would then capture the already-moved path. The build
    // config keeps its own productName; that one is correct and stays.
    expect(PKG.productName).toBeUndefined();
    expect(PKG.build.productName).toBe('Orienta');
  });
});
