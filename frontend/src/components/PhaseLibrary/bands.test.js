// @vitest-environment node
import { describe, it, expect } from 'vitest';
import fixture from './__fixtures__/library.json';
import { systemBands, phasesWithoutBand } from './bands';
import { applyElementFacets } from './facets';

const ALL = fixture.phases;

describe('the bands this library has', () => {
  it('is nine of them, in the measured order and sizes', () => {
    const { bands } = systemBands(ALL);
    expect(bands.map((b) => [b.element, b.phases.length])).toEqual([
      ['Al', 28], ['Fe', 19], ['Si', 18], ['Mg', 9], ['Mn', 8],
      ['Zn', 5], ['Cu', 4], ['Ni', 1], ['O', 1],
    ]);
  });

  it('counts 36 phases in 93 placements, and both numbers are the point', () => {
    const { phaseCount, placementCount } = systemBands(ALL);
    expect(phaseCount).toBe(36);
    expect(placementCount).toBe(93);
    // 93 alone reads as a library three times this size; 36 alone leaves the
    // reader counting one card three times.
    expect(placementCount).toBeGreaterThan(phaseCount);
  });

  it('puts a phase in every system it belongs to', () => {
    const { bands } = systemBands(ALL);
    const inBand = (e) => bands.find((b) => b.element === e).phases.map((p) => p.key);
    const alfesi = ALL.find((p) => p.key === 'Al3Fe2Si_mp-1190708_symmetrized');
    expect(alfesi).toBeTruthy();
    for (const e of ['Al', 'Fe', 'Si']) expect(inBand(e)).toContain(alfesi.key);
  });

  it('ties break alphabetically — Ni before O, both with one card', () => {
    const { bands } = systemBands(ALL);
    const one = bands.filter((b) => b.phases.length === 1).map((b) => b.element);
    expect(one).toEqual(['Ni', 'O']);
  });

  it('a band of one is still a band', () => {
    // Hiding a small band would hide the only phase in it.
    const { bands } = systemBands(ALL);
    expect(bands.find((b) => b.element === 'Ni').phases).toHaveLength(1);
  });
});

describe('bands follow the structure, never the label', () => {
  it('the Si band leaves out the phases merely NAMED Si', () => {
    // A band has nowhere to put the search's "nur laut Etikett" caveat, so a
    // Si band containing `beta-AlFeSi` would drop it silently.
    const { bands } = systemBands(ALL);
    const si = bands.find((b) => b.element === 'Si').phases.map((p) => p.key);
    expect(si).not.toContain('beta-AlFeSi');
    expect(si).not.toContain('sd_1401510');
    expect(si).toHaveLength(18);
  });

  it('no band exists for an element only a label claims', () => {
    const { bands } = systemBands(ALL);
    const elements = bands.map((b) => b.element);
    // M and P were invented by reading a Materials Project tag as chemistry.
    expect(elements).not.toContain('M');
    expect(elements).not.toContain('P');
    expect(elements).toHaveLength(9);
  });
});

describe('bands are built from what is on screen', () => {
  it('a filtered library produces fewer, smaller bands', () => {
    const shown = applyElementFacets(ALL, ['Mg']);
    const { bands, phaseCount, placementCount } = systemBands(shown);
    expect(phaseCount).toBe(9);
    expect(bands.find((b) => b.element === 'Mg').phases).toHaveLength(9);
    expect(bands.length).toBeLessThan(9);
    expect(placementCount).toBeLessThan(93);
    // Every band still holds only phases that are shown.
    const keys = new Set(shown.map((p) => p.key));
    for (const b of bands) for (const p of b.phases) expect(keys.has(p.key)).toBe(true);
  });

  it('an empty list produces no bands and no placements', () => {
    expect(systemBands([])).toEqual({ bands: [], phaseCount: 0, placementCount: 0 });
  });
});

describe('a phase whose structure names no element', () => {
  it('is reported rather than dropped off the screen', () => {
    // None in this library today, which is why the case is written down now
    // instead of found later: a band view silently loses such a phase.
    expect(phasesWithoutBand(ALL)).toEqual([]);
    const orphan = { key: 'mystery', elements_structure: [] };
    expect(phasesWithoutBand([...ALL, orphan]).map((p) => p.key)).toEqual(['mystery']);
    expect(systemBands([orphan]).placementCount).toBe(0);
  });
});
