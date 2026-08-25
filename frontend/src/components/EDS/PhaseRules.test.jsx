// @vitest-environment jsdom
/**
 * The rule editor.
 *
 * The thing worth pinning is the seeding: "Rule from this structure" is the
 * primary path, and a seeded rule that fails on the very structure it came
 * from would be worse than no seeding at all. The rest is plumbing, but the
 * plumbing must not silently drop a clause.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import PhaseRules, {
  countMatching, renormalise, seedRuleFromStructure,
} from './PhaseRules';

/** What the inspector hands over for one structure. */
const DETAIL = {
  cif_filename: 'Si.cif',
  formula: 'Si',
  elements: [
    { element: 'Si', at_pct: 56.0, spread: 6.4, enrichment: 7.69 },
    { element: 'Al', at_pct: 42.6, spread: 6.4, enrichment: 0.50 },
    { element: 'Fe', at_pct: 1.2, spread: 0.3, enrichment: 0.06 },
  ],
};

const STRUCTURES = [
  { structure_id: 0, mean_at_pct: { Al: 97.0, Si: 1.2 } },
  { structure_id: 1, mean_at_pct: { Al: 60.0, Si: 38.0 } },
  { structure_id: 2, mean_at_pct: { Al: 43.0, Si: 56.0 } },
];

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('renormalise', () => {
  it('scales to 100% over the scored elements', () => {
    const out = renormalise({ Al: 40, Si: 10 });
    expect(out.Al).toBeCloseTo(80);
    expect(out.Si).toBeCloseTo(20);
  });

  it('leaves out C and O, as the clustering does', () => {
    const out = renormalise({ Al: 50, Si: 50, C: 100, O: 100 });
    expect(Object.keys(out).sort()).toEqual(['Al', 'Si']);
    expect(out.Al).toBeCloseTo(50);
  });

  it('survives an empty composition', () => {
    expect(renormalise({})).toEqual({});
    expect(renormalise(undefined)).toEqual({});
  });
});

describe('seedRuleFromStructure', () => {
  it('only writes rows for elements that are genuinely concentrated', () => {
    // A band on an element sitting at background constrains nothing and only
    // makes the rule brittle. 1.3x is the bar the classifier itself gates on.
    const seeded = seedRuleFromStructure(DETAIL);
    expect(seeded.elements.map((e) => e.element)).toEqual(['Si']);
  });

  it('brackets the measured value, so it cannot fail on its own structure', () => {
    const seeded = seedRuleFromStructure(DETAIL);
    const si = seeded.elements[0];
    expect(si.min_at_pct).toBeLessThanOrEqual(56.0);
    expect(si.max_at_pct).toBeGreaterThanOrEqual(56.0);
  });

  it('widens by the spread, so the dilution gradient survives', () => {
    // Si 56.0 +/- 2 * 6.4 -> 43.2 .. 68.8. The particle's other rings sit at
    // 38 and 25 at%, so the user still has to widen it - which is the point of
    // showing the numbers rather than guessing them.
    const si = seedRuleFromStructure(DETAIL).elements[0];
    expect(si.min_at_pct).toBeCloseTo(43.2, 1);
    expect(si.max_at_pct).toBeCloseTo(68.8, 1);
  });

  it('falls back to the strongest elements when nothing is enriched', () => {
    const flat = { cif_filename: 'X', elements: [
      { element: 'Al', at_pct: 90, spread: 1, enrichment: 1.0 },
      { element: 'Si', at_pct: 10, spread: 1, enrichment: 1.0 },
    ] };
    expect(seedRuleFromStructure(flat).elements).toHaveLength(2);
  });

  it('keys the rule on the phase name, not on a position', () => {
    expect(seedRuleFromStructure(DETAIL).phase_key).toBe('Si.cif');
  });

  it('survives a structure with no chemistry', () => {
    const seeded = seedRuleFromStructure({ cif_filename: 'X', elements: [] });
    expect(seeded.elements).toEqual([]);
  });
});

describe('countMatching', () => {
  it('counts the structures a rule would admit', () => {
    const rule = { elements: [{ element: 'Si', min_at_pct: 30, max_at_pct: null }] };
    expect(countMatching(rule, STRUCTURES)).toBe(2);
  });

  it('counts zero when a rule matches nothing — the case that must be visible', () => {
    const rule = { elements: [{ element: 'Si', min_at_pct: 90, max_at_pct: null }] };
    expect(countMatching(rule, STRUCTURES)).toBe(0);
  });

  it('does not count a structure whose element the rule names but the data lacks', () => {
    const rule = { elements: [{ element: 'W', min_at_pct: 1 }] };
    expect(countMatching(rule, STRUCTURES)).toBe(0);
  });
});

describe('the editor', () => {
  const base = {
    rules: { rules: [] },
    setRules: vi.fn(),
    structures: STRUCTURES,
    allPhases: [],
    inspectorDetail: DETAIL,
    elements: ['Al', 'Si', 'Fe'],
    onReclassify: vi.fn(),
    busy: false,
  };

  it('says plainly when there are no rules yet', () => {
    render(<PhaseRules {...base} />);
    expect(screen.getByText('rules.noneYet')).toBeTruthy();
  });

  it('seeds a rule from the inspected structure', () => {
    const setRules = vi.fn();
    render(<PhaseRules {...base} setRules={setRules} />);
    fireEvent.click(screen.getByText('rules.seed'));
    expect(setRules).toHaveBeenCalled();
    const produced = setRules.mock.calls[0][0]({ rules: [] });
    expect(produced.rules[0].phase_key).toBe('Si.cif');
  });

  it('cannot seed without a structure selected', () => {
    render(<PhaseRules {...base} inspectorDetail={null} />);
    expect(screen.getByText('rules.seed').disabled).toBe(true);
  });

  it('marks a phase whose rules match nothing', () => {
    const rules = { rules: [{ phase_key: 'Si.cif',
                              elements: [{ element: 'Si', min_at_pct: 90 }] }] };
    render(<PhaseRules {...base} rules={rules} />);
    expect(screen.getByText('rules.matchesNone')).toBeTruthy();
  });

  it('shows how many structures a rule admits', () => {
    const rules = { rules: [{ phase_key: 'Si.cif',
                              elements: [{ element: 'Si', min_at_pct: 30 }] }] };
    render(<PhaseRules {...base} rules={rules} />);
    expect(screen.getByText('rules.matchesN:{"count":2}')).toBeTruthy();
  });

  it('edits a bound through the number field', () => {
    const setRules = vi.fn();
    const rules = { rules: [{ phase_key: 'Si.cif',
                              elements: [{ element: 'Si', min_at_pct: 30, max_at_pct: null }] }] };
    render(<PhaseRules {...base} rules={rules} setRules={setRules} />);
    fireEvent.click(screen.getByText('Si.cif'));
    fireEvent.change(screen.getByLabelText('rules.minFor:{"element":"Si"}'),
                     { target: { value: '45' } });
    const produced = setRules.mock.calls.at(-1)[0](rules);
    expect(produced.rules[0].elements[0].min_at_pct).toBe(45);
  });

  it('removes a rule entirely', () => {
    const setRules = vi.fn();
    const rules = { rules: [{ phase_key: 'Si.cif', elements: [] }] };
    render(<PhaseRules {...base} rules={rules} setRules={setRules} />);
    fireEvent.click(screen.getByTitle('rules.removeTooltip:{"name":"Si.cif"}'));
    const produced = setRules.mock.calls.at(-1)[0](rules);
    expect(produced.rules).toEqual([]);
  });

  it('asks for a re-classify rather than applying silently', () => {
    // Rules change the grouping's outcome, so they cannot take effect under an
    // existing map without the picture and its explanation disagreeing.
    const onReclassify = vi.fn();
    render(<PhaseRules {...base} onReclassify={onReclassify} />);
    fireEvent.click(screen.getByText('rules.apply'));
    expect(onReclassify).toHaveBeenCalled();
  });

  it('disables the actions while a classification is running', () => {
    render(<PhaseRules {...base} busy />);
    expect(screen.getByText('rules.apply').disabled).toBe(true);
    expect(screen.getByText('rules.seed').disabled).toBe(true);
  });
});
