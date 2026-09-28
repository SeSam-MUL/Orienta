import { describe, it, expect } from 'vitest';
import fixture from './__fixtures__/library.json';
import { siblingsOf, differencesBetween } from './siblings';

const byKey = (k) => fixture.phases.find((p) => p.key === k);

describe('on the real library', () => {
  it('finds two pairs on 36 phases, and no more', () => {
    // A warning that fires on half the library is a warning nobody reads.
    // Same space group AND same compound is narrow enough to leave two --
    // and I expected ONE when I wrote this. The second is real and I did
    // not know about it: two Mg17Al12 entries from the same Materials
    // Project id, `I-43m` both, lattice parameters 10.436 and 10.533 A.
    // The rule found a near-duplicate nobody had noticed, which is what it
    // is for.
    const pairs = fixture.phases
      .map((p) => [p.key, siblingsOf(fixture.phases, p.key).map((s) => s.key)])
      .filter(([, s]) => s.length);
    expect(Object.fromEntries(pairs)).toEqual({
      sd_0302719: ['sd_1401510'],
      sd_1401510: ['sd_0302719'],
      'Mg17Al12_MP-mp-2151': ['Mg17Al12_mp-2151_conventional_standard'],
      'Mg17Al12_mp-2151_conventional_standard': ['Mg17Al12_MP-mp-2151'],
    });
  });

  it('names the difference that decides the run, first', () => {
    // Spherical alone, and that surprised me too: `sd_1401510` has no
    // `.sht` and no pre-built dictionary, but it HAS a master pattern, and
    // a master is what Dictionary indexing needs. The card's file block
    // shows an empty Dictionary slot, which is why a reader concluded
    // Dictionary was off the table for it -- the slot and the capability
    // are both right and answer different questions.
    const [other] = siblingsOf(fixture.phases, 'sd_1401510');
    expect(other.differences[0]).toEqual(
      { kind: 'alsoIndexableWith', methods: ['spherical'] });
  });

  it('and the composition the label hides', () => {
    // Cooper's structure carries no silicon site while its label reads
    // `…Si0.68`; Barlock's does. That is the crystallography of the
    // choice, and the list cannot show it.
    const [other] = siblingsOf(fixture.phases, 'sd_1401510');
    expect(other.differences.map((d) => d.kind)).toContain('composition');
    expect(other.differences.find((d) => d.kind === 'composition').theirs)
      .toBe('Al Fe Mn Si');
  });

  it('and the paper, and the lattice parameter', () => {
    const [other] = siblingsOf(fixture.phases, 'sd_1401510');
    const kinds = other.differences.map((d) => d.kind);
    expect(kinds).toContain('reference');
    expect(kinds).toContain('cellA');
    expect(other.differences.find((d) => d.kind === 'reference').theirs)
      .toMatch(/Barlock/);
  });

  it('is symmetric: whichever one you open, it names the other', () => {
    // A reader who types `Al Fe Si` gets one of them first and the other
    // fifteenth. Either card has to mention the one they did not open.
    const a = siblingsOf(fixture.phases, 'sd_0302719');
    expect(a.map((s) => s.key)).toEqual(['sd_1401510']);
    expect(a[0].differences.map((d) => d.kind))
      .toContain('notIndexableWith');
  });

  it('a phase with no twin gets no line', () => {
    expect(siblingsOf(fixture.phases, 'Al')).toEqual([]);
  });
});

describe('what counts as a sibling', () => {
  const p = (over) => ({
    key: 'x', space_group_hm: 'Im-3', formula_label: 'Al5Fe',
    elements_structure: ['Al', 'Fe'], capabilities: {}, cell: {}, ...over });

  it('the same compound in a DIFFERENT space group is not one', () => {
    // The same compound in two structure types is a genuinely different
    // entry, and the space group is what says so.
    const list = [p({}), p({ key: 'y', space_group_hm: 'Fm-3m' })];
    expect(siblingsOf(list, 'x')).toEqual([]);
  });

  it('the same space group with a different compound is not one either', () => {
    const list = [p({}), p({ key: 'y', formula_label: 'Mg2Si' })];
    expect(siblingsOf(list, 'x')).toEqual([]);
  });

  it('a proportional formula IS the same compound', () => {
    // `Al2CuMg` and `Al4Cu2Mg2` are one compound written twice.
    const list = [p({ formula_label: 'Al2CuMg' }),
      p({ key: 'y', formula_label: 'Al4Cu2Mg2' })];
    expect(siblingsOf(list, 'x').map((s) => s.key)).toEqual(['y']);
  });

  it('a row with no space group says nothing rather than guessing', () => {
    const list = [p({ space_group_hm: null }), p({ key: 'y' })];
    expect(siblingsOf(list, 'x')).toEqual([]);
  });

  it('a row with no readable formula says nothing either', () => {
    const list = [p({ formula_label: null, formula_structure: null }),
      p({ key: 'y' })];
    expect(siblingsOf(list, 'x')).toEqual([]);
  });

  it('a key that is not in the library is not an error', () => {
    expect(siblingsOf([p({})], 'nope')).toEqual([]);
    expect(siblingsOf(null, 'x')).toEqual([]);
  });
});

describe('the differences, on their own', () => {
  const base = { capabilities: { hough: true }, elements_structure: ['Al'],
    reference: 'R1', cell: { a: 1 } };

  it('says nothing about two rows that agree', () => {
    expect(differencesBetween(base, { ...base })).toEqual([]);
  });

  it('puts what you can index with first', () => {
    const other = { ...base, capabilities: { hough: true, spherical: true } };
    expect(differencesBetween(base, other)[0].kind).toBe('alsoIndexableWith');
  });

  it('reports a capability the OTHER one lacks, too', () => {
    const other = { ...base, capabilities: {} };
    expect(differencesBetween(base, other)[0])
      .toEqual({ kind: 'notIndexableWith', methods: ['hough'] });
  });

  it('does not offer a citation the other one does not have', () => {
    const other = { ...base, reference: null };
    expect(differencesBetween(base, other).map((d) => d.kind))
      .not.toContain('reference');
  });

  it('needs both lattice parameters to compare them', () => {
    expect(differencesBetween(base, { ...base, cell: {} })
      .map((d) => d.kind)).not.toContain('cellA');
  });
});

describe('the name of the twin, which is the only part a reader can act on', () => {
  // `display` was never asserted: `display: p.key` passed the whole suite,
  // and the card would then have pointed at "sd_1401510" instead of
  // "alpha-Al(Fe,Mn)" -- a note saying "there is another one" while
  // naming it in the one spelling the reader cannot recognise.
  const phases = [
    { key: 'a1', display_name: 'alpha-Al(Fe,Mn)Si',
      formula_label: 'Mn0.5Fe0.5Al5Si0.68', space_group_hm: 'Im-3',
      elements_structure: ['Al', 'Fe', 'Mn', 'Si'], capabilities: {} },
    { key: 'a2', display_name: 'alpha-Al(Fe,Mn)',
      formula_label: 'Mn0.5Fe0.5Al5Si0.68', space_group_hm: 'Im-3',
      elements_structure: ['Al', 'Fe', 'Mn'], capabilities: {} },
  ];

  it('is the readable name, not the key', () => {
    const [twin] = siblingsOf(phases, 'a1');
    expect(twin.display).toBe('alpha-Al(Fe,Mn)');
    expect(twin.key).toBe('a2');
  });

  it('falls back to the label, then the key, rather than to nothing', () => {
    const noName = phases.map((p) => ({ ...p, display_name: null }));
    expect(siblingsOf(noName, 'a1')[0].display).toBe('Mn0.5Fe0.5Al5Si0.68');
    const bare = noName.map((p) => ({ ...p, formula_label: null,
      formula_structure: 'Mn0.5Fe0.5Al5Si0.68' }));
    expect(siblingsOf(bare, 'a1')[0].display).toBe('a2');
  });
});
