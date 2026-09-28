/**
 * The two reasons a phase is not in the phase tester's table.
 *
 * Merging them is the defect this guards: the dialog read `excluded` and
 * not `unsupported`, so a phase that could not be tested at all -- no
 * simulated master -- vanished from its own results with no line about
 * it, looking exactly like a phase the chemistry filter had rejected.
 */
import { describe, it, expect } from 'vitest';
import { omissions, anyOmitted, KNOWN_REASONS } from './phaseTestOmissions';

const RESULT = {
  excluded: [
    { phase_key: 'Si', display_formula: 'Si', chemistry_fit: 0.12 },
    { phase_key: 'WC', formula: 'WC', chemistry_fit: 0 },
  ],
  unsupported: [{ key: 'Al', reason: 'no_sht' }],
};

describe('the two lists stay apart', () => {
  it('a phase the chemistry rejected is excluded, not unsupported', () => {
    const { excluded, unsupported } = omissions(RESULT);
    expect(excluded.map((e) => e.key)).toEqual(['Si', 'WC']);
    expect(unsupported.map((u) => u.key)).toEqual(['Al']);
  });

  it('and a phase with no master is unsupported, not excluded', () => {
    // `Al` is in the library and was deliberately ticked. If it appeared
    // under "excluded by EDS" the user would read a verdict where there
    // was no test; if it appeared nowhere -- which is what shipped --
    // they would not know it had been dropped.
    const { excluded } = omissions(RESULT);
    expect(excluded.map((e) => e.key)).not.toContain('Al');
  });

  it('a fit of zero is a measurement, not a missing one', () => {
    // `chemistry_fit: 0` is a real answer and `|| null` would erase it.
    expect(omissions(RESULT).excluded[1].fit).toBe(0);
  });

  it('a missing fit is null rather than a made-up number', () => {
    expect(omissions({ excluded: [{ phase_key: 'X' }] }).excluded[0].fit)
      .toBe(null);
  });
});

describe('what the screen can put into words', () => {
  it('names `no_sht` as one it can explain', () => {
    expect(omissions(RESULT).unsupported[0].known).toBe(true);
    expect(KNOWN_REASONS).toContain('no_sht');
  });

  it('a reason nobody has written a sentence for is still shown', () => {
    // Quoting a reason we cannot phrase beats inventing a sentence for
    // it, and both beat dropping the row -- which is the whole defect.
    const { unsupported } = omissions(
      { unsupported: [{ key: 'Q', reason: 'wombat' }] });
    expect(unsupported[0]).toEqual({ key: 'Q', reason: 'wombat', known: false });
  });

  it('and a reason the server left out does not become a blank line', () => {
    expect(omissions({ unsupported: [{ key: 'Q' }] }).unsupported[0].reason)
      .toBe('unknown');
  });
});

describe('an answer with nothing left out', () => {
  it('still gives both lists, so "none" cannot be mistaken for "not asked"', () => {
    expect(omissions({})).toEqual({ excluded: [], unsupported: [] });
    expect(omissions(null)).toEqual({ excluded: [], unsupported: [] });
  });

  it('anyOmitted is true for either reason on its own', () => {
    expect(anyOmitted({})).toBe(false);
    expect(anyOmitted({ excluded: [{ phase_key: 'Si' }] })).toBe(true);
    expect(anyOmitted({ unsupported: [{ key: 'Al', reason: 'no_sht' }] }))
      .toBe(true);
  });
});
