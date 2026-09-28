// @vitest-environment node
/**
 * The card, against real answers from the real endpoint.
 *
 * The fixture is eight cards captured from `build_phase_detail` on the real
 * library -- with the absolute paths removed, which is its own test below.
 */
import { describe, it, expect } from 'vitest';
import cards from './__fixtures__/cards.json';
import {
  cellRows, compositionRows, fileRows, FILE_SLOTS,
  masterParameters, shtParameters, shtProvenance, notices,
} from './phaseCardRows';

const card = (key) => cards[key];

describe('this folder has no two names that differ only in case', () => {
  // Three times in one day: `ElementFacets.jsx` / `elementFacets.js`,
  // `IdentityLine.jsx` / `identityLine.js`, `PhaseCard.jsx` /
  // `phaseCard.js`. Windows resolves both spellings to one file, the import
  // hands back the wrong module, and every test that renders the component
  // dies. `src/caseCollisions.test.js` walks the whole tree -- but I kept
  // running only this folder's suite, where the collision was invisible.
  it('so a run of this folder alone still catches one', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const url = await import('node:url');
    const here = path.dirname(url.fileURLToPath(import.meta.url));
    const seen = new Map();
    const clashes = [];
    for (const name of fs.readdirSync(here)) {
      const lower = name.toLowerCase();
      if (seen.has(lower)) clashes.push(`${seen.get(lower)} vs ${name}`);
      else seen.set(lower, name);
    }
    expect(clashes).toEqual([]);
  });
});

describe('no absolute path reaches the card', () => {
  it('and none is in the fixture either', () => {
    // The endpoint sends `simulation.sht.provenance.source_xtal.path`, and
    // on the machine this was captured on it was an absolute Windows path
    // through a home directory. This project has already put a local path
    // into a manuscript.
    //
    // AGAINST THE PATTERN, NOT AGAINST A NAME. This test used to end with
    // `not.toContain('<the account name on this machine>')` -- a test
    // against leaking account names that shipped one, in a repo whose
    // public half is curated by hand (found by c1). Matching the shape
    // instead keeps the name out AND catches every other user's, on either
    // kind of machine.
    //
    // A DRIVE LETTER AT THE START OF A VALUE, not `letter colon slash`
    // anywhere -- the first version of that matched `https://` in every DOI
    // and failed on a fixture that was already clean.
    const blob = JSON.stringify(cards);
    expect(blob).not.toMatch(/"[A-Za-z]:[\\/]/);
    expect(blob).not.toMatch(/[A-Za-z]:[\\/]+users[\\/]/i);
    expect(blob).not.toMatch(/\/(?:home|users|media|run\/media)\/[^/"]+\//i);
  });

  it('provenance keeps the name and whether the file is still there', () => {
    const p = shtProvenance(card('Al'));
    expect(p.sources.map((s) => s.slot)).toEqual(['xtal', 'cif']);
    expect(p.sources[0].name).toBe('Al.xtal');
    expect(p.sources[0].found).toBe(true);
    for (const s of p.sources) expect(JSON.stringify(s)).not.toMatch(/[A-Za-z]:[\\/]/);
  });

  it('a file row offers a repo-relative path and nothing else', () => {
    for (const row of fileRows(card('Al'))) {
      if (!row.rel) continue;
      expect(row.rel.startsWith('Database/'), row.slot).toBe(true);
    }
  });
});

describe('the cell', () => {
  it('shows every constant, at a sensible precision', () => {
    const rows = cellRows(card('Al').cell);
    expect(rows.map((r) => r.key)).toEqual(['a', 'b', 'c', 'alpha', 'beta', 'gamma']);
    expect(rows[0].text).toBe('4.049 Å');
    expect(rows[3].text).toBe('90°');
  });

  it('separates the two phases whose cells are all that differ', () => {
    // 12.5 against 12.56: the only thing that tells the twins apart.
    expect(cellRows(card('sd_0302719').cell)[0].text).toBe('12.5 Å');
    expect(cellRows(card('sd_1401510').cell)[0].text).toBe('12.56 Å');
  });

  it('says nothing when there is no cell', () => {
    expect(cellRows(null)).toEqual([]);
    expect(cellRows({})).toEqual([]);
  });
});

describe('the composition, twice', () => {
  it('computes the difference rather than looking it up', () => {
    // `beta-AlFeSi` is named for silicon and its structure has none. A
    // curated list of known discrepancies would go stale the day someone
    // adds a CIF; this cannot.
    const rows = compositionRows(card('beta-AlFeSi'));
    const si = rows.find((r) => r.element === 'Si');
    expect(si).toBeTruthy();
    expect(si.inLabel).toBe(true);
    expect(si.inStructure).toBe(false);
    expect(si.disagrees).toBe(true);
    const al = rows.find((r) => r.element === 'Al');
    expect(al.disagrees).toBe(false);
  });

  it('agrees with itself when the two sides agree', () => {
    for (const r of compositionRows(card('Al'))) expect(r.disagrees).toBe(false);
  });
});

describe('the files', () => {
  it('are five slots, present or not, never just a colour', () => {
    const rows = fileRows(card('beta-AlFeSi'));
    expect(rows.map((r) => r.slot)).toEqual(FILE_SLOTS);
    const present = rows.filter((r) => r.present).map((r) => r.slot);
    // This phase has only a CIF -- which is exactly why it cannot be
    // indexed spherically and why its row must say so rather than show a
    // red mark a reader reads as an error.
    expect(present).toEqual(['cif']);
    for (const r of rows) {
      if (r.present) expect(r.name).toBeTruthy();
      else expect(r.name).toBeNull();
    }
  });

  it('carry a name and a date for what is there', () => {
    const cif = fileRows(card('Al')).find((r) => r.slot === 'cif');
    expect(cif.name).toBe('Al.cif');
    expect(cif.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(cif.category).toBe('cif_library');
  });
});

describe('the simulation parameters behind master and SHT', () => {
  it('are chosen, not dumped: the ones that decide whether it is usable', () => {
    const m = masterParameters(card('Al'));
    const byKey = Object.fromEntries(m.map((x) => [x.key, x]));
    expect(byKey.energy.value).toBe(20);
    expect(byKey.energy.unit).toBe('kV');
    expect(byKey.dmin.value).toBeCloseTo(0.06, 4);
    expect(byKey.npx.value).toBe(500);
    expect(byKey.tilt.value).toBe(70);
    // Fifty namelist entries; six on the card.
    expect(m.length).toBeLessThan(10);
  });

  it('read the .sht from its own sidecar', () => {
    const s = Object.fromEntries(shtParameters(card('Al')).map((x) => [x.key, x.value]));
    expect(s.energy).toBe(20);
    expect(s.bandwidth).toBe(384);
    expect(s.engine).toBe('ours');
  });

  it('and say nothing at all for a phase that was never simulated', () => {
    expect(masterParameters(card('beta-AlFeSi'))).toEqual([]);
    expect(shtParameters(card('beta-AlFeSi'))).toEqual([]);
    expect(shtProvenance(card('beta-AlFeSi')).sources).toEqual([]);
  });

  it('show the master alone when there is no .sht', () => {
    // `sd_1401510` is the twin that cannot be indexed spherically.
    expect(masterParameters(card('sd_1401510')).length).toBeGreaterThan(0);
    expect(shtParameters(card('sd_1401510'))).toEqual([]);
  });
});

describe('what the card has to SAY', () => {
  it('an unreadable CIF is stated, and the card is not empty because of it', () => {
    const broken = card('sd_1816951');
    const kinds = notices(broken).map((n) => n.kind);
    expect(kinds).toContain('parseError');
    expect(notices(broken).find((n) => n.kind === 'parseError').detail)
      .toMatch(/2 different compositions/);
    // And it still has something to show: an .xtal, an .sht, a master.
    expect(fileRows(broken).filter((r) => r.present).map((r) => r.slot))
      .toEqual(expect.arrayContaining(['xtal', 'master', 'sht']));
    expect(cellRows(broken.cell).length).toBe(6);
  });

  it('names the disagreement between label and structure', () => {
    expect(notices(card('beta-AlFeSi')).map((n) => n.kind))
      .toContain('elementsDisagree');
    expect(notices(card('Al')).map((n) => n.kind))
      .not.toContain('elementsDisagree');
  });

  it('says which cell setting is shown when the file declares several', () => {
    // `Si.cif` reports Fd-3m from one block and Fd-3m O1 from another --
    // the origin choice, which §2.3 is about and which no number fixes.
    const n = notices(card('Si')).find((x) => x.kind === 'severalSettings');
    expect(n, 'Si declares more than one setting').toBeTruthy();
    expect(n.detail).toBeTruthy();
  });

  it('says when a source was refused, rather than showing nothing', () => {
    const withRejected = Object.values(cards).find((c) => c.reference_rejected);
    if (!withRejected) return;                 // none among these eight
    expect(notices(withRejected).map((n) => n.kind)).toContain('referenceRejected');
  });

  it('a clean phase has nothing to warn about', () => {
    expect(notices(card('Al')).filter((n) => n.kind === 'parseError')).toEqual([]);
  });
});

describe('a master built against a library that is not this one', () => {
  // The server throws the recorded path away on purpose -- it names an
  // account, and this project has already put a local path into a
  // manuscript -- and keeps one boolean, `recorded_elsewhere`, because
  // that is the part a reader needed. Nothing in the frontend read it, so
  // the card said only "✗", which reads as "the file was deleted" rather
  // than "this master's provenance belongs to another machine". The two
  // have very different consequences for reproducing a result.
  //
  // NOT IN THE FIXTURE: no card in this library is in that state, so this
  // is built by hand rather than measured. Said plainly, because a test
  // over invented data proves the mapping and not the situation.
  const withSource = (source) => ({
    key: 'x', simulation: { sht: { provenance: { source_xtal: source } } },
  });

  it('is carried through to the card', () => {
    expect(shtProvenance(withSource(
      { name: 'Al.xtal', found: false, recorded_elsewhere: true })).sources[0])
      .toEqual({ slot: 'xtal', name: 'Al.xtal', found: false, elsewhere: true });
  });

  it('and a source that is simply missing is not accused of it', () => {
    expect(shtProvenance(withSource(
      { name: 'Al.xtal', found: false })).sources[0].elsewhere).toBe(false);
  });

  it('nor is one that is right here', () => {
    expect(shtProvenance(withSource(
      { name: 'Al.xtal', found: true })).sources[0].elsewhere).toBe(false);
  });
});
