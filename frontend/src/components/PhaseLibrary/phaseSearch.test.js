// @vitest-environment node
//
// The acceptance criteria from spec §2.1, against the REAL library.
//
// Every one names KEYS, never a count. A review showed why: `s phase -> 1`
// was satisfiable, and satisfied by the wrong phase (the T-phase, the only
// stem containing the word "phase"). A count is passable by accident.
import { describe, it, expect } from 'vitest';
import fixture from './__fixtures__/library.json';
import { buildIndex, search, parseFormula, normaliseSpaceGroup, queryElements } from './phaseSearch';

const index = buildIndex(fixture.phases);

/**
 * The fixture with a name written onto one of its phases.
 *
 * These three tests used to inject names through a second argument to
 * `buildIndex` that the program never passed -- so they were green over a
 * feature a running app could not reach. Names live ON THE ROW now, the way
 * the endpoint sends them, and these build the row the endpoint would.
 */
function named(key, names) {
  return buildIndex(fixture.phases.map(
    (p) => (p.key === key ? { ...p, ...names } : p)));
}

const keys = (q) => search(index, q).identity.map((e) => e.key);
const textKeys = (q) => search(index, q).text.map((e) => e.key);
const why = (q, key) => search(index, q).identity.find((e) => e.key === key)?.why;

// Everything the structure says contains all three.
const AL_FE_SI = fixture.phases
  .filter((p) => ['Al', 'Fe', 'Si'].every((e) => p.elements_structure.includes(e)))
  .map((p) => p.key);

describe('the library the tests run against', () => {
  it('is the real one', () => {
    expect(fixture.phases).toHaveLength(36);
    // The seven cases the spec's approval requires to be present.
    for (const k of ['sd_0302719', 'sd_1401510', 'alpha-AlFeMnSi_ICSD-52623',
                     'α-(AlMnSi)', 'Al', 'sd_1816951', 'beta-AlFeSi']) {
      expect(fixture.phases.some((p) => p.key === k)).toBe(true);
    }
  });
});

describe('§2.1 acceptance — the queries that were measured to fail', () => {
  // Measured before: 36 of 36, because "al" is inside "Journal" and
  // "Metallkunde". The citations are prose and prose contains every short
  // string; that is why they are a separate hit class.
  it('`Al` returns the phases that contain aluminium, and pure Al among them', () => {
    const expected = fixture.phases
      .filter((p) => p.elements_structure.includes('Al')).map((p) => p.key);
    expect(new Set(keys('Al'))).toEqual(new Set(expected));
    expect(keys('Al')).toContain('Al');
    expect(keys('Al').length).toBeLessThan(36);
  });

  // The Erstnutzer's worst finding: his supervisor said "use the Al phases",
  // and no search term found pure aluminium. `Al.cif` contains the string
  // "alumin" nowhere.
  it('`aluminium` finds pure Al — the whole reason the name table exists', () => {
    expect(keys('aluminium')).toContain('Al');
    expect(keys('aluminum')).toContain('Al');   // US spelling
  });

  it('German element names work too', () => {
    expect(keys('eisen')).toEqual(expect.arrayContaining(
      fixture.phases.filter((p) => p.elements_structure.includes('Fe'))
        .map((p) => p.key).slice(0, 1)));
    expect(keys('silizium')).toContain('Si');
  });

  // Measured before: 13 hits for a library holding exactly one Ni phase.
  it('`Ni` returns the nickel phases and nothing else', () => {
    const expected = fixture.phases
      .filter((p) => p.elements_structure.includes('Ni')).map((p) => p.key);
    expect(new Set(keys('Ni'))).toEqual(new Set(expected));
  });

  // Measured before: 1, where 13 phases contain all three. The system name
  // Sebastian himself uses.
  it('`Al-Fe-Si` is an element combination, not a string', () => {
    const got = keys('Al-Fe-Si');
    expect(new Set(got.filter((k) => AL_FE_SI.includes(k)))).toEqual(new Set(AL_FE_SI));
    expect(AL_FE_SI.length).toBeGreaterThan(10);
    for (const spelling of ['Al Fe Si', 'AlFeSi', 'al-fe-si']) {
      expect(new Set(keys(spelling).filter((k) => AL_FE_SI.includes(k))))
        .toEqual(new Set(AL_FE_SI));
    }
  });

  // `beta-AlFeSi` is CALLED Si and its structure has none. It must be found
  // and it must be marked, because a user who types Al-Fe-Si and does not see
  // a phase called beta-AlFeSi concludes the search is broken.
  it('label-only members are found, and say so', () => {
    expect(keys('Al-Fe-Si')).toContain('beta-AlFeSi');
    expect(why('Al-Fe-Si', 'beta-AlFeSi')).toContain('labelOnly');
    expect(why('Al-Fe-Si', 'Al3Fe2Si_mp-1190708_symmetrized')).not.toContain('labelOnly');
  });

  // Measured before: 0. The file writes `MgCuAl2`; element order in a formula
  // is not information.
  it('`Al2CuMg` finds the S-phase however the file spells it', () => {
    for (const spelling of ['Al2CuMg', 'MgCuAl2', 'CuAl2Mg', 'Al₂CuMg']) {
      expect(keys(spelling)).toContain('sd_1814127');
    }
  });

  // Measured before: 0, because the file writes `Im-3`.
  it('space groups fold across the four spellings in this library', () => {
    const im3 = fixture.phases
      .filter((p) => normaliseSpaceGroup(p.space_group_hm).core === 'im3').map((p) => p.key);
    expect(im3.length).toBeGreaterThan(1);
    for (const spelling of ['Im3', 'Im-3', 'I m -3', 'im 3']) {
      expect(new Set(keys(spelling).filter((k) => im3.includes(k)))).toEqual(new Set(im3));
    }
  });

  it('`P63/mmc` finds the hexagonal phases whichever way they are written', () => {
    const p63 = fixture.phases
      .filter((p) => normaliseSpaceGroup(p.space_group_hm).core === 'p63/mmc').map((p) => p.key);
    expect(p63.length).toBeGreaterThan(1);
    for (const spelling of ['P63/mmc', 'P6_3/mmc', 'p 63 / m m c']) {
      expect(new Set(keys(spelling).filter((k) => p63.includes(k)))).toEqual(new Set(p63));
    }
  });

  // Three files, not two: two named with the Greek letter and one spelled out.
  it('`alpha` finds all three, Greek or spelled', () => {
    for (const k of ['α-(AlMnSi)', 'α-(AlFeSi) (Fe23Al81Si15)',
                     'alpha-AlFeMnSi_ICSD-52623']) {
      expect(keys('alpha')).toContain(k);
      expect(keys('α')).toContain(k);
    }
  });

  // The structure type of pure Al and pure Ni is literally `Cu`. Indexing it
  // as identity makes `Cu` return 6 where the facet says 4 — the same
  // search-versus-facet contradiction this rewrite removes.
  it('`Cu` returns the copper phases, not the ones whose prototype is Cu', () => {
    const copper = fixture.phases
      .filter((p) => p.elements_structure.includes('Cu')).map((p) => p.key);
    // Exactly the four, and the count a user compares against the facet.
    expect(new Set(keys('Cu'))).toEqual(new Set(copper));
    expect(keys('Cu')).not.toContain('Al');
    expect(keys('Cu')).not.toContain('Ni');
    // Pure Al and pure Ni ARE reachable -- their structure type is Cu -- but
    // in their own group, because 6-against-4 is the same search-versus-facet
    // contradiction as 10-against-4, only smaller.
    const proto = search(index, 'Cu').prototype.map((e) => e.key);
    expect(proto).toEqual(expect.arrayContaining(['Al', 'Ni']));
  });

  // Measured before: 10, because "s" is inside almost every word.
  it('`s phase` does not match on a single letter', () => {
    expect(keys('s phase').length).toBeLessThan(3);
    expect(keys('s phase')).not.toContain('Si');
  });

  it('a synonym makes the S-phase findable by the name the field uses', () => {
    const withSyn = named('sd_1814127',
      { display_name: 'S-Phase', search_terms: ['s phase'] });
    expect(search(withSyn, 'S-Phase').identity.map((e) => e.key)).toContain('sd_1814127');
    expect(search(withSyn, 'S-Phase').identity[0].key).toBe('sd_1814127');
  });
});

describe('a space group by its NUMBER, which the glossary offers as an example', () => {
  // Measured before the fix, on this very fixture: `225` -> 0 hits, `63`
  // -> 0, while `Fm-3m` -> 6. The International Tables number was in the
  // payload and displayed on the card, and indexed nowhere. The glossary
  // and the manual both give `225` as a worked example, so the one query
  // a reader is invited to try was the one that returned nothing.

  it('finds by IT number what the symbol finds', () => {
    const byNumber = search(index, '225').identity.map((h) => h.key).sort();
    const bySymbol = search(index, 'Fm-3m').identity.map((h) => h.key).sort();
    expect(byNumber.length).toBe(6);
    expect(byNumber).toEqual(bySymbol);
  });

  it('and says space group was the reason', () => {
    expect(search(index, '225').identity[0].why).toContain('spaceGroup');
  });

  it('a number no phase has finds nothing, rather than everything', () => {
    expect(search(index, '229').identity).toEqual([]);
  });
});

describe('the two hit classes stay apart', () => {
  it('a citation match is never mixed into the identity hits', () => {
    // Barlock wrote the paper behind sd_0302719 and is in no phase name.
    expect(keys('Barlock')).toEqual([]);
    expect(textKeys('Barlock')).toContain('sd_0302719');
  });

  it('a short token cannot reach the citations at all', () => {
    expect(textKeys('al')).toEqual([]);
  });
});

// The four below exist because a mutation survived: the rule they cover was
// in the code, described in a comment, and provable by nothing. Each one was
// re-run against the mutation it names and fails there.

describe('rules no query in this library happens to exercise', () => {
  // MUTATION: `const whole = token.length <= 2` -> `false`. Survived: on the
  // 36 phases we ship, no realistic two-letter query changes its answer, so
  // the guard that fixed `Al -> 36 of 36` was resting on the citation class
  // alone. A SYNONYM is user-typed text and is where a fragment match bites
  // first -- the day someone names a phase "Almandin-Typ", `Al` returns a
  // phase with no aluminium in it.
  it('a one- or two-letter query matches a whole word, never a fragment', () => {
    const withSyn = named('MgZn2_sd_0261233', { display_name: 'Almandin-Typ' });
    const k = (q) => search(withSyn, q).identity.map((e) => e.key);
    expect(k('Al')).not.toContain('MgZn2_sd_0261233');   // no aluminium in it
    expect(k('Almandin')).toContain('MgZn2_sd_0261233'); // the long token still works
  });

  // MUTATION: proportional equality -> exact equality. Survived, and finding
  // out why turned up a defect: `parseFormula` was replacing `.` with a space
  // before reading numbers, so `Al162.04Fe46Si30` parsed as 162 and the one
  // proportional pair in the library came out exact BY ACCIDENT.
  it('a decimal count survives parsing', () => {
    expect(parseFormula('Al162.04Fe46Si30')).toEqual({ al: 162.04, fe: 46, si: 30 });
    expect(parseFormula('Al0.5Fe0.5')).toEqual({ al: 0.5, fe: 0.5 });
  });

  // The same compound at twice the counts. This library writes BOTH for the
  // alpha phase -- label `Fe23Al81Si15`, structure `Al162.04Fe46Si30` -- and
  // the profile card shows them on one page, so the search must not call them
  // two phases.
  it('a formula and its doubled form are one phase', () => {
    const k = 'α-(AlFeSi) (Fe23Al81Si15)';
    expect(keys('Fe23Al81Si15')).toContain(k);
    expect(keys('Al162.04Fe46Si30')).toContain(k);

    // The two above prove less than they look like they do: both strings sit
    // verbatim in an indexed field, so free text alone finds them. Tightening
    // the tolerance to 1e-12 left this test green. THIS spelling is reachable
    // only through the ratio -- reordered, so it is in no field, and carrying
    // the .04, so the ratio is 2.00049 and not a clean 2.
    expect(keys('Fe46Al162.04Si30')).toContain(k);
    expect(keys('Al3Fe1Si1')).not.toContain(k);       // a different ratio is a different phase
  });
});

describe('ranking', () => {
  // MUTATION: `sort((a,b) => a.weight - b.weight || byName)` -> `sort(byName)`.
  // Survived, because on this library no query ranks differently from
  // alphabetical -- every `alpha` hit is weight 3. The earlier version of this
  // test used the name "alpha cubic", which sorts first anyway, so it proved
  // nothing. This one names the phase so that the two orders DISAGREE.
  it('what a phase is called outranks what its file is called', () => {
    const withSyn = named('sd_0302719', { display_name: 'Zzz-alpha (Bergman)' });
    const ranked = search(withSyn, 'alpha').identity;
    expect(ranked[0].key).toBe('sd_0302719');          // weight 0, sorts last by name
    expect(ranked[0].why).toContain('name');
    expect(ranked.at(-1).why).toEqual(['filename']);  // the filename hits come last
    // Without the name it is just another filename hit, and then it is NOT first.
    expect(search(buildIndex(fixture.phases), 'alpha').identity[0].key).not.toBe('sd_0302719');
  });
});

describe('the pieces, separately', () => {
  it('parseFormula ignores element order and proportional counts', () => {
    expect(parseFormula('Al2CuMg')).toEqual(parseFormula('MgCuAl2'));
    expect(parseFormula('not a formula at all')).toBeNull();
    expect(parseFormula('Xx2Y')).toBeNull();
  });

  it('normaliseSpaceGroup splits the origin choice off instead of dropping it', () => {
    expect(normaliseSpaceGroup('Fd-3m O1')).toEqual({ core: 'fd3m', suffix: 'O1' });
    expect(normaliseSpaceGroup('R -3 :H')).toEqual({ core: 'r3', suffix: 'H' });
    expect(normaliseSpaceGroup('C 1 2/c 1').core).toBe(normaliseSpaceGroup('C2/c').core);
  });

  it('queryElements reads symbols, words and run-together spellings', () => {
    expect(queryElements('Al-Fe-Si')).toEqual(['Al', 'Fe', 'Si']);
    expect(queryElements('AlFeSi')).toEqual(['Al', 'Fe', 'Si']);
    expect(queryElements('eisen')).toEqual(['Fe']);
    expect(queryElements('Barlock')).toBeNull();
  });
});
