// @vitest-environment node
/**
 * The one property that makes a facet worth having: the number on the chip
 * is what you get when you click it. Everything else here follows from that.
 */
import { describe, it, expect } from 'vitest';
import fixture from './__fixtures__/library.json';
import {
  elementFacets, applyElementFacets, toggleElement, filterChips,
  CHIP_SEARCH_THRESHOLD,
  capabilityFacets, applyCapabilityFacets, toggleCapability, CAPABILITIES,
} from './facets';
import { ELEMENT_NAMES } from './elementNames';

const ALL = fixture.phases;
const row = (rows, sym) => rows.find((r) => r.symbol === sym);
const keys = (phases) => phases.map((p) => p.key).sort();

describe('the count on a chip is what clicking it gives you', () => {
  // Stated as a property over EVERY chip and every starting point, rather
  // than as a handful of numbers. A count is passable by accident; this is
  // not.
  it('holds for every element, from no selection', () => {
    const rows = elementFacets(ALL, ALL, []);
    for (const r of rows) {
      expect(applyElementFacets(ALL, [r.symbol]), r.symbol).toHaveLength(r.count);
    }
  });

  it('still holds after two elements are already chosen', () => {
    for (const start of [['Al'], ['Al', 'Fe'], ['Fe', 'Si'], ['Mg']]) {
      const shown = applyElementFacets(ALL, start);
      const rows = elementFacets(ALL, shown, start);
      for (const r of rows) {
        if (r.selected) continue;                 // clicking removes, not adds
        const after = applyElementFacets(ALL, [...start, r.symbol]);
        expect(after.length, `${start.join('+')} then ${r.symbol}`).toBe(r.count);
      }
    }
  });

  it('a chosen chip reports what is on screen, because that is what it did', () => {
    const shown = applyElementFacets(ALL, ['Al', 'Fe']);
    const rows = elementFacets(ALL, shown, ['Al', 'Fe']);
    expect(row(rows, 'Al').count).toBe(shown.length);
    expect(row(rows, 'Fe').count).toBe(shown.length);
  });
});

describe('the measured numbers of this library', () => {
  // From the real 36, named so a change in the library is visible as a
  // change here rather than as a silently different screen.
  it('narrows the way a person expects', () => {
    const n = (sel) => applyElementFacets(ALL, sel).length;
    expect(n([])).toBe(36);
    expect(n(['Al'])).toBe(28);
    expect(n(['Al', 'Fe'])).toBe(18);
    expect(n(['Al', 'Fe', 'Si'])).toBe(13);
    expect(n(['Al', 'Fe', 'Si', 'Mn'])).toBe(4);
  });

  it('offers the nine elements the structures contain, commonest first', () => {
    const rows = elementFacets(ALL, ALL, []);
    expect(rows.map((r) => r.symbol)).toEqual(
      ['Al', 'Fe', 'Si', 'Mg', 'Mn', 'Zn', 'Cu', 'Ni', 'O']);
    // And NOT the two that a file stem once invented: `Al2Zn_MP-mp-...` read
    // the Materials Project tag as manganese and phosphorus.
    expect(rows.map((r) => r.symbol)).not.toContain('P');
  });
});

describe('the facets count structures, and say nothing about labels', () => {
  it('silicon does not return the phase that is merely named Si', () => {
    // `beta-AlFeSi` is called Si and its structure has none. The search says
    // so in words; a facet has no room for a caveat, and a silicon filter
    // that yields a phase without silicon is wrong rather than nuanced.
    const got = keys(applyElementFacets(ALL, ['Si']));
    expect(got).not.toContain('beta-AlFeSi');
    expect(got).not.toContain('sd_1401510');
    expect(row(elementFacets(ALL, ALL, []), 'Si').count).toBe(got.length);
  });
});

describe('a chip that would empty the list', () => {
  it('is shown and disabled, not hidden', () => {
    // Nickel is in exactly one phase and aluminium is not in it.
    const shown = applyElementFacets(ALL, ['Ni']);
    const rows = elementFacets(ALL, shown, ['Ni']);
    const dead = rows.filter((r) => r.count === 0);
    expect(dead.length).toBeGreaterThan(0);
    for (const r of dead) expect(r.disabled).toBe(true);
    // Hiding them would drop the information that the element exists here.
    expect(rows).toHaveLength(9);
  });
});

describe('toggling', () => {
  it('adds, removes, and keeps the order it was given', () => {
    expect(toggleElement([], 'Al')).toEqual(['Al']);
    expect(toggleElement(['Al'], 'Fe')).toEqual(['Al', 'Fe']);
    expect(toggleElement(['Al', 'Fe'], 'Al')).toEqual(['Fe']);
  });
});

describe('the chip search, above fifteen chips', () => {
  // This library has nine, so the box never appears on it. The threshold is
  // for the libraries this app will meet; the case is constructed rather
  // than pretended.
  const many = Object.keys(ELEMENT_NAMES).slice(0, 40)
    .map((symbol) => ({ symbol, count: 1, selected: false, disabled: false }));
  const names = (sym) => ELEMENT_NAMES[sym] || [];

  it('needs more chips than this library has', () => {
    expect(elementFacets(ALL, ALL, []).length).toBeLessThan(CHIP_SEARCH_THRESHOLD);
    expect(many.length).toBeGreaterThan(CHIP_SEARCH_THRESHOLD);
  });

  it('matches a symbol and a name, in more than one language', () => {
    expect(filterChips(many, 'fe', names).map((r) => r.symbol)).toContain('Fe');
    expect(filterChips(many, 'eisen', names).map((r) => r.symbol)).toEqual(['Fe']);
    expect(filterChips(many, 'iron', names).map((r) => r.symbol)).toEqual(['Fe']);
  });

  it('never filters away an element you have chosen', () => {
    // Hiding it would hide the only control that undoes it -- the §2.8 trap
    // in miniature.
    const withChosen = many.map((r) =>
      (r.symbol === 'Al' ? { ...r, selected: true } : r));
    expect(filterChips(withChosen, 'zzz', names).map((r) => r.symbol)).toEqual(['Al']);
  });

  it('an empty query changes nothing', () => {
    expect(filterChips(many, '  ', names)).toBe(many);
  });
});


describe('the three capability facets (§2.7)', () => {
  it('answer the three numbers that were measured', () => {
    const rows = capabilityFacets(ALL, ALL, []);
    const n = (c) => rows.find((r) => r.capability === c).count;
    // 35 and not 36: `sd_1816951` has a CIF, and it parses to two different
    // compositions. c1 measured what happens if the tick is green anyway --
    // the reader Hough uses (orix/diffpy, not pymatgen) SUCCEEDS and returns
    // Mg 80 / Cu 20 at% where the truth is Cu 66.7 / Mg 33.3 -- so a green
    // tick the indexer then refuses is worse than an honest cross. The
    // spec's §2.7 table says 36/36 and is wrong by this one.
    expect(n('hough')).toBe(35);
    expect(n('spherical')).toBe(29);      // .sht
    expect(n('dictionary')).toBe(16);     // master OR pre-built dictionary
    // FOUR pre-built dictionaries, not the three §2.7 reports: the fourth
    // sits in a folder called `sd` rather than under its phase stem, so a
    // folder rule missed it. Not 16, which is what counting masters gives,
    // and not 2, which is what the shipped app's name matcher answers.
    expect(ALL.filter((p) => p.files.dictionary)).toHaveLength(4);
  });

  it('the count on a capability is also what clicking it gives you', () => {
    for (const start of [[], ['spherical'], ['dictionary'], ['hough', 'spherical']]) {
      const shown = applyCapabilityFacets(ALL, start);
      for (const r of capabilityFacets(ALL, shown, start)) {
        if (r.selected) continue;
        expect(applyCapabilityFacets(ALL, [...start, r.capability]).length,
               `${start.join('+')} then ${r.capability}`).toBe(r.count);
      }
    }
  });

  it('dictionary counts a master OR a pre-built dictionary, never the name', () => {
    // The file that made this measurable: `pi-Al8FeMg3Si6` is a real master
    // called `..._E20kV_sig70_n501_o0.h5`, with no "master" in its name.
    const pi = ALL.find((p) => p.key.startsWith('pi-Al8FeMg3Si6'));
    expect(pi, 'the phase that breaks name matching is in the library').toBeTruthy();
    expect(pi.capabilities.dictionary).toBe(true);
    expect(pi.files.master.name).not.toContain('master');
  });

  it('capabilities and elements narrow together', () => {
    const both = applyCapabilityFacets(applyElementFacets(ALL, ['Fe']), ['spherical']);
    expect(both.length).toBeLessThan(applyElementFacets(ALL, ['Fe']).length);
    for (const p of both) {
      expect(p.elements_structure).toContain('Fe');
      expect(p.capabilities.spherical).toBe(true);
    }
  });

  it('toggling works the same way as for elements', () => {
    expect(toggleCapability([], 'hough')).toEqual(['hough']);
    expect(toggleCapability(['hough'], 'hough')).toEqual([]);
    expect(CAPABILITIES).toEqual(['hough', 'spherical', 'dictionary']);
  });
});

describe('masters that belong to no phase (§2.7)', () => {
  it('are in the contract, not swallowed', () => {
    // Four of them, and the second is the S-phase -- the same phase §2.1
    // uses as an acceptance case. Its master sits under a stem the library
    // does not map, so without this the card says "no master" about a phase
    // whose master is right there on the disk.
    const folders = (fixture.unassigned_masters || []).map((m) => m.folder);
    expect(folders).toEqual(['deposit_Al', 'deposit_Al2CuMg',
                             'deposit_Al7Cu2Fe', 'deposit_alpha-AlMnSi']);
    for (const m of fixture.unassigned_masters) {
      expect(m.name, m.folder).toBeTruthy();
      expect(m.rel, m.folder).toContain('EBSD_H5_Cache');
    }
  });

  it('are not counted as any phase capability', () => {
    // 16, not 20: four of the twenty masters on disk have no phase.
    expect(ALL.filter((p) => p.capabilities.dictionary)).toHaveLength(16);
    // And the one phase whose dictionary is filed under a different folder
    // is still counted -- it has a master too, so this asserts the file,
    // not the capability.
    expect(ALL.find((p) => p.key === 'sd_0302719').files.dictionary).toBeTruthy();
  });
});
