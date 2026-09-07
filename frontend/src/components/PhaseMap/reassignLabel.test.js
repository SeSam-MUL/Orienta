/**
 * The Reassign button must describe everything it is about to do.
 *
 * "Reassign 3 grains" was accurate about stage 1 and silent about stage 2 —
 * and stage 2 is the repair the user actually asked for (19 wrong pixels
 * inside an Al7FeCu2 particle, measured 2026-09-07). A button that under-
 * reports its own effect is how a user presses it and concludes "nothing
 * happened", which is exactly the bug report this came from.
 */
import { describe, it, expect } from 'vitest';
import { reassignWork, reassignLabelKey } from './reassignLabel';

describe('reassignWork', () => {
  it('is enabled when EITHER stage has something to do', () => {
    expect(reassignWork({ n_reassign: 3, n_islands_reassign: 0 }).has).toBe(true);
    // The regression: stage 2 alone used to leave the button dead.
    expect(reassignWork({ n_reassign: 0, n_islands_reassign: 19 }).has).toBe(true);
    expect(reassignWork({ n_reassign: 0, n_islands_reassign: 0 }).has).toBe(false);
  });

  it('treats a missing summary as nothing to do, not as a crash', () => {
    expect(reassignWork(null).has).toBe(false);
    expect(reassignWork(undefined)).toEqual({ grains: 0, islands: 0, has: false });
    expect(reassignWork({}).has).toBe(false);
  });
});

describe('reassignLabelKey', () => {
  it('names both repairs when both are pending', () => {
    expect(reassignLabelKey({ n_reassign: 3, n_islands_reassign: 19 })).toEqual({
      key: 'phasemap:phaseCheck.reassignButtonBoth',
      params: { n: 3, islands: 19 },
    });
  });

  it('says pixels, not grains, when only islands are pending', () => {
    const r = reassignLabelKey({ n_reassign: 0, n_islands_reassign: 19 });
    expect(r.key).toBe('phasemap:phaseCheck.reassignButtonIslands');
    expect(r.params).toEqual({ islands: 19 });
  });

  it('keeps the plain grain label when there are no islands', () => {
    expect(reassignLabelKey({ n_reassign: 3, n_islands_reassign: 0 })).toEqual({
      key: 'phasemap:phaseCheck.reassignButton',
      params: { n: 3 },
    });
    // Disabled state still renders a label; it must be the ordinary one.
    expect(reassignLabelKey(null).key).toBe('phasemap:phaseCheck.reassignButton');
  });
});

describe('the label keys exist in every language', () => {
  const langs = ['en', 'de', 'ja', 'zh'];
  const keys = [
    'reassignButton', 'reassignButtonBoth', 'reassignButtonIslands',
    'checkDoneIslands', 'reassignDoneIslands', 'islandsFailed',
  ];
  langs.forEach((l) => {
    it(`${l} has all of them`, async () => {
      const m = await import(`../../locales/${l}/phasemap.json`);
      const pc = (m.default || m).phaseCheck;
      keys.forEach((k) => expect(pc[k], `${l}.phaseCheck.${k}`).toBeTruthy());
    });
  });
});
