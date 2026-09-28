// @vitest-environment node
//
// A source-level guard, deliberately, and only for wiring.
//
// A review measured the hole: delete `loadState={discoverState}` from
// IndexingPage's `<PhaseDropdown>` and all 406 tests stay green while the
// dropdown falls back to its `'loaded'` default and says "No phases found" on
// a failed first fetch — the exact sentence this change exists to remove. The
// same is true of `onRetry` and of `!phaseListBlocked` in `canStart`. The
// defect's home is the wiring, and nothing guarded it.
//
// A mount test would be better and is not available: IndexingPage is ~4000
// lines behind dozens of API calls, and no test in this repo mounts it — the
// nearest suite, `indexingCollection.test.jsx`, says so in its own comment and
// tests the pure functions instead. This file therefore checks only that three
// specific props/terms are present, never what they do; what they do is tested
// in `phaseDiscovery.test.js` and `phaseDropdownLoadState.test.jsx` against the
// real functions and the real component.
//
// DELETE THIS FILE the day IndexingPage can be mounted in a test. A source
// check is a placeholder for a real one, not a substitute, and this repo has
// rejected source checks before — correctly — where they stood in for logic.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(
  fileURLToPath(new URL('./IndexingPage.jsx', import.meta.url)), 'utf8');

/** The JSX block that mounts the dropdown. */
function dropdownJsx() {
  const i = SRC.indexOf('<PhaseDropdown');
  expect(i).toBeGreaterThan(-1);
  return SRC.slice(i, SRC.indexOf('/>', i));
}

describe('the phase dropdown is wired to the load state', () => {
  it('receives the load state, or it silently claims "no phases found"', () => {
    expect(dropdownJsx()).toMatch(/loadState=\{discoverState\}/);
  });

  it('receives the retry, or the "try again" button never renders', () => {
    // PhaseDropdown guards the button with `onRetry && (...)`.
    expect(dropdownJsx()).toMatch(/onRetry=\{retryDiscover\}/);
  });

  /**
   * The whole STATEMENT, not its first line.
   *
   * This read one line, and broke the day a third condition made the
   * expression wrap -- while the gate it checks was still there and still
   * correct. A source-reading test should be exactly as brittle as the
   * thing it protects, and no more.
   */
  const canStartExpr = () => {
    const i = SRC.indexOf('const canStart');
    expect(i).toBeGreaterThan(-1);
    return SRC.slice(i, SRC.indexOf(';', i));
  };

  it('Start is gated on the phase list', () => {
    expect(canStartExpr()).toMatch(/!phaseListBlocked/);
  });

  it('and on the active group still existing', () => {
    // An active group that resolves to nothing used to widen the run to the
    // whole library in silence. It offers nothing now, so starting has to be
    // blocked too -- otherwise the button is live over an empty selection
    // and the reason is nowhere on screen.
    expect(canStartExpr()).toMatch(/!groupMissing/);
  });

  it('...and the gate leaves a way out when the user picked a phase by hand', () => {
    // Without this clause a permanently failing discovery endpoint disables
    // Start even for someone who browsed to a CIF themselves.
    const i = SRC.indexOf('const phaseListBlocked');
    expect(i).toBeGreaterThan(-1);
    expect(SRC.slice(i, i + 200)).toMatch(/effectivePhasePaths\.length === 0/);
  });

  it('the blocked Start says why', () => {
    expect(SRC).toMatch(/phaseListBlocked \? t\('phaseDropdown\.startBlockedTip'\)/);
  });

  it('the PC is read through a ref, not captured at first render', () => {
    // The factory is built once; reading `pcValues` directly would freeze the
    // placeholder it holds before the detector info arrives, and empty
    // `current_pc` on every discovery call for the life of the window.
    const i = SRC.indexOf('function currentPc()');
    expect(i).toBeGreaterThan(-1);
    const body = SRC.slice(i, i + 260);
    expect(body).toMatch(/pcValuesRef\.current/);
    expect(body).not.toMatch(/\bpcValues\.match\b/);
  });
});
