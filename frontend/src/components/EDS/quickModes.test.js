// @vitest-environment node
/**
 * Every quick-mode button the panel draws must do something.
 *
 * `LayerStackPanel` renders six (phase, ipf-z, ipf-x, ipf-y, bc, ci) and the
 * EDS page mounts it unfiltered — but `quickModeToLayer` in EDSPage.jsx knew
 * only four and returned null for the rest, and the click handler bails on
 * null without a word. So "IPF-X" and "IPF-Y" sat there, enabled, and did
 * nothing when pressed. Reported by a documentation audit, 2026-08-27.
 *
 * A button that no-ops silently is read as broken DATA, not as an unwired
 * control — the user goes looking at their scan.
 *
 * Source-level, because the two files are what disagree and neither is wrong
 * on its own: the panel is entitled to offer six, the page is entitled to
 * translate the ones it supports. Only their intersection is a fact, and only
 * reading both can see it. Same reasoning as
 * `DatabaseBrowser/syncResultContract.test.js`.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const panel = readFileSync(join(HERE, '../PhaseMap/LayerStackPanel.jsx'), 'utf8');
const page = readFileSync(join(HERE, 'EDSPage.jsx'), 'utf8');
const stack = readFileSync(join(HERE, 'hooks/useEdsLayerStack.js'), 'utf8');

/** Quick-mode ids the panel puts on screen. */
function offered() {
  const m = /const QUICK_MODES = \[([\s\S]*?)\];/.exec(panel);
  expect(m, 'LayerStackPanel must declare QUICK_MODES').toBeTruthy();
  return [...m[1].matchAll(/id:\s*'([^']+)'/g)].map((x) => x[1]);
}

/** Quick-mode ids `quickModeToLayer` can turn into a layer. */
function handled() {
  const m = /function quickModeToLayer\(id\) \{([\s\S]*?)\n\}/.exec(page);
  expect(m, 'EDSPage must define quickModeToLayer').toBeTruthy();
  return [...m[1].matchAll(/id === '([^']+)'/g)].map((x) => x[1]);
}

describe('EDS quick-mode buttons', () => {
  it('every button the panel offers is handled by the page', () => {
    const dead = offered().filter((id) => !handled().includes(id));
    expect(dead, `LayerStackPanel offers ${dead.join(', ')} on the EDS page, `
      + 'but quickModeToLayer returns null for them and the click is dropped '
      + 'without a message').toEqual([]);
  });

  it('names IPF-X and IPF-Y specifically — they were the two that were dead', () => {
    expect(handled()).toContain('ipf-x');
    expect(handled()).toContain('ipf-y');
  });

  it('the EDS layer stack can actually fetch them', () => {
    // Not a hypothetical: the stack already routes anything whose kind starts
    // with `ipf` to the phase-map layer endpoint, and the backend lists
    // ipf-x / ipf-y among its valid kinds. Hiding the buttons instead of
    // wiring them would have hidden a capability that was already there.
    expect(stack).toMatch(/kind\.startsWith\('ipf'\)/);
    expect(stack).toMatch(/phaseMapApi\.layer\(layer\.id/);
  });

  it('each handled mode also gets a label, so none falls back to its raw id', () => {
    const m = /const quickLabels = \{([\s\S]*?)\};/.exec(page);
    expect(m, 'EDSPage must map quick-mode ids to labels').toBeTruthy();
    const labelled = [...m[1].matchAll(/'?([\w-]+)'?\s*:/g)].map((x) => x[1]);
    const unlabelled = handled().filter((id) => !labelled.includes(id));
    expect(unlabelled, `no label for ${unlabelled.join(', ')}`).toEqual([]);
  });
});
