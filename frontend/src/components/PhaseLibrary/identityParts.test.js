// @vitest-environment node
import { describe, it, expect } from 'vitest';
import fixture from './__fixtures__/library.json';
import {
  identityParts, pearsonParts, displaySpaceGroup, nameProvenance,
} from './identityParts';

const ALL = fixture.phases;
const byKey = (k) => ALL.find((p) => p.key === k);
const kinds = (parts) => parts.map((p) => p.kind);

describe('what is on the line, and in which order', () => {
  it('formula, Pearson, space group, source -- in that order', () => {
    const parts = identityParts(byKey('Al'));
    expect(kinds(parts)).toEqual(['formula', 'pearson', 'spaceGroup', 'reference']);
  });

  it('a synonym goes first when there is one', () => {
    const parts = identityParts(byKey('sd_0302719'), { synonym: 'alpha cubic' });
    expect(parts[0]).toEqual({ kind: 'synonym', text: 'alpha cubic' });
  });

  it('the formula is never dropped, because Pearson does not separate these two', () => {
    // Both cP138, both Pm-3. On a line, only the iron tells them apart --
    // which is why Fassung 1's "Pearson is the distinguishing field" was
    // wrong and the formula is not optional.
    const a = byKey('alpha-AlFeMnSi_ICSD-52623');
    const b = byKey('α-(AlMnSi)');
    expect(a.pearson).toBe(b.pearson);
    expect(displaySpaceGroup(a.space_group_hm))
      .toBe(displaySpaceGroup(b.space_group_hm));
    const fa = identityParts(a).find((x) => x.kind === 'formula').text;
    const fb = identityParts(b).find((x) => x.kind === 'formula').text;
    expect(fa).not.toBe(fb);           // the formula is what separates them
    expect(fa + fb).toContain('Fe');
  });

  it('there is always a formula part, whatever the name is made of', () => {
    for (const p of ALL) {
      expect(kinds(identityParts(p)), p.key).toContain('formula');
    }
  });
});

describe('where the name came from is visible', () => {
  // MEASURED FIRST, and it overturned the obvious implementation: all
  // FIFTEEN labels in this library come from `_sm_phase_labels`, and the
  // three phases whose element source is the filename have no label at all.
  // So a provenance keyed on `elements_label_source === 'filename'` could
  // never fire, and §2.2's requirement would have looked implemented while
  // doing nothing. The guessed name a reader actually sees is the STEM.
  it('every label in this library is a recorded one', () => {
    const labelled = ALL.filter((p) => p.formula_label);
    expect(labelled).toHaveLength(15);
    for (const p of labelled) {
      expect(p.elements_label_source, p.key).toBe('sm_phase_labels');
      expect(nameProvenance(p), p.key).toBe('label');
    }
  });

  it('without a label the structure formula stands in, and is marked as that', () => {
    const p = ALL.find((x) => !x.formula_label && x.formula_structure);
    expect(p).toBeTruthy();
    expect(nameProvenance(p)).toBe('structure');
    expect(identityParts(p).find((x) => x.kind === 'formula').provenance)
      .toBe('structure');
  });

  it('with neither, the row shows the file stem and calls it a guess', () => {
    const p = { key: 'sd_9999999' };
    expect(nameProvenance(p)).toBe('stem');
    const part = identityParts(p).find((x) => x.kind === 'formula');
    expect(part.text).toBe('sd_9999999');
    expect(part.provenance).toBe('stem');
  });

  it('the three kinds reach the line, so they cannot look the same', () => {
    const seen = new Set(ALL.map((p) =>
      identityParts(p).find((x) => x.kind === 'formula').provenance));
    expect(seen.has('label')).toBe(true);
    expect(seen.has('structure')).toBe(true);
  });
});

describe('a source is shown or nothing is', () => {
  it('a phase with a checked citation shows it', () => {
    const withRef = ALL.find((p) => p.reference);
    expect(kinds(identityParts(withRef))).toContain('reference');
  });

  it('a phase with only a DOI shows the DOI', () => {
    const doiOnly = ALL.find((p) => p.doi && !p.reference);
    expect(doiOnly, 'six phases are in this state').toBeTruthy();
    expect(kinds(identityParts(doiOnly))).toContain('doi');
  });

  it('a phase with BOTH shows both, not one instead of the other', () => {
    // This was `else if`, so the DOI was shown only where there was no
    // citation -- suppressing the link to the paper exactly where a paper
    // exists. Four of the 36 phases in this library carry both, and for
    // all four a reader wanting the source concluded the file had none.
    const both = ALL.filter((p) => p.reference && p.doi);
    expect(both.length, 'four phases are in this state').toBe(4);
    for (const p of both) {
      const k = kinds(identityParts(p));
      expect(k).toContain('reference');
      expect(k).toContain('doi');
    }
  });

  it('a phase with neither shows NOTHING -- no placeholder, no "unknown"', () => {
    const none = ALL.find((p) => !p.reference && !p.doi);
    expect(none, 'three phases are in this state').toBeTruthy();
    const k = kinds(identityParts(none));
    expect(k).not.toContain('reference');
    expect(k).not.toContain('doi');
  });
});

describe('Pearson, decoded', () => {
  it('reads the system, the centring and the atom count', () => {
    expect(pearsonParts('cF4')).toEqual(
      { system: 'cubic', centring: 'faceCentred', atoms: 4 });
    expect(pearsonParts('hP12')).toEqual(
      { system: 'hexagonal', centring: 'primitive', atoms: 12 });
    expect(pearsonParts('cI168')).toEqual(
      { system: 'cubic', centring: 'bodyCentred', atoms: 168 });
    expect(pearsonParts('hR123')).toEqual(
      { system: 'hexagonal', centring: 'rhombohedral', atoms: 123 });
  });

  it('agrees with the crystal system the backend reports, on every phase', () => {
    // A real cross-check: `crystal_system` comes from the structure reader,
    // the Pearson symbol from a CIF field. Two sources, one answer -- and
    // trigonal is the known exception, since Pearson has no letter for it
    // and files them under h.
    for (const p of ALL) {
      const { system } = pearsonParts(p.pearson);
      if (!system || !p.crystal_system) continue;
      const expected = p.crystal_system === 'trigonal' ? 'hexagonal' : p.crystal_system;
      expect(system, `${p.key} (${p.pearson})`).toBe(expected);
    }
  });

  it('says nothing rather than guessing at something it cannot read', () => {
    expect(pearsonParts('nonsense')).toEqual(
      { system: null, centring: null, atoms: null });
    expect(pearsonParts(null).system).toBeNull();
  });
});

describe('space groups, as they should be read', () => {
  it('folds the four spellings this library uses', () => {
    expect(displaySpaceGroup('Fm-3m')).toBe('Fm-3m');
    expect(displaySpaceGroup('P m -3')).toBe('Pm-3');
    expect(displaySpaceGroup('C 1 2/c 1')).toBe('C2/c');
    expect(displaySpaceGroup('A 1 2/a 1')).toBe('A2/a');
  });

  it('leaves every real space group in this library readable', () => {
    for (const p of ALL) {
      if (!p.space_group_hm) continue;
      const shown = displaySpaceGroup(p.space_group_hm);
      expect(shown, p.key).toBeTruthy();
      expect(shown, p.key).not.toMatch(/\s/);        // no stray spacing
      expect(shown[0], p.key).toMatch(/[A-Z]/);      // still starts with the lattice letter
    }
  });

  it('nothing in, nothing out', () => {
    expect(displaySpaceGroup(null)).toBeNull();
    expect(displaySpaceGroup('  ')).toBeNull();
  });
});
