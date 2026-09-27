// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { keyForPath, activeKeySet } from '../PhaseCollections/collectionFilter';
import { planCollectionAdoption, formatMissingLogMessage, resolveAllowedPaths } from '../PhaseCollections/collectionAdopt';
import PhaseDropdown from './PhaseDropdown';

const FILES = [
  { path: 'D:/lib/CIF_Library/Al.cif', filename: 'Al.cif', element_group: 'Reine Elemente' },
  { path: 'D:/lib/CIF_Library/Al4Fe1.7Si (τ11).cif', filename: 'Al4Fe1.7Si (τ11).cif', element_group: 'Al-Fe-Si' },
  { path: 'D:/lib/CIF_Library/WC.cif', filename: 'WC.cif', element_group: 'C-W' },
];
const COLS = [{
  name: 'Intermetallics in Al', parent: null,
  members: [{ key: 'Al' }, { key: 'Al4Fe1.7Si (τ11)' }],
}];

/**
 * Hough files ARE named by the stem, so the stem is the key. Spherical and
 * Dictionary files are not: an .sht is `Formula (CIF_stem) [Pearson] {kV}.sht`
 * (spec §2.6) and a master is `..._master_...h5`. The first draft filtered all
 * three on `keyForPath(f.path)` and its test used only CIF paths, where stem
 * == key — so it stayed green while Spherical and Dictionary would have been
 * filtered down to nothing. Those two go through the server's `resolve`, which
 * reads `LocalEntry.sht_path` and `_master_h5_for_phase`, and match on PATH.
 */
function visibleHough(files, collectionKeys) {
  if (!collectionKeys) return files;
  return files.filter((f) => collectionKeys.has(keyForPath(f.path)));
}

function visibleResolved(files, resolvedPaths) {
  if (!resolvedPaths) return files;
  const allow = new Set(resolvedPaths);
  return files.filter((f) => allow.has(f.path));
}

const SHT_FILES = [
  { path: 'D:/lib/EBSD_SHT_Database/Al/Al (Al) [cF4] {20kV}.sht' },
  { path: 'D:/lib/EBSD_SHT_Database/x/Fe1.8Al4.4Si0.6ht (Al4Fe1.7Si (τ11)) [hP28] {20kV}.sht' },
  { path: 'D:/lib/EBSD_SHT_Database/WC/WC (WC) [hP2] {20kV}.sht' },
];

describe('indexing phase dropdown under a collection', () => {
  it('Hough: offers only the collection, matching on the stem including dots', () => {
    const v = visibleHough(FILES, activeKeySet(COLS, 'Intermetallics in Al'));
    expect(v.map((f) => f.filename)).toEqual(['Al.cif', 'Al4Fe1.7Si (τ11).cif']);
  });

  it('offers everything when no collection is active', () => {
    expect(visibleHough(FILES, activeKeySet(COLS, null))).toHaveLength(3);
  });

  it('REGRESSION: filtering SHT files by stem would offer nothing at all', () => {
    const keys = activeKeySet(COLS, 'Intermetallics in Al');
    expect(visibleHough(SHT_FILES, keys)).toEqual([]);   // the old, wrong way
  });

  it('Spherical: the resolved paths are what decides', () => {
    // What GET /resolve?method=spherical returns for this collection.
    const resolved = [SHT_FILES[0].path, SHT_FILES[1].path];
    const v = visibleResolved(SHT_FILES, resolved);
    expect(v).toHaveLength(2);
    expect(v.some((f) => f.path.includes('WC'))).toBe(false);
  });

  it('a collection whose phases are not among the discovered files offers nothing', () => {
    const cols = [{ name: 'Hartmetalle', parent: null, members: [{ key: 'TiC' }] }];
    expect(visibleHough(FILES, activeKeySet(cols, 'Hartmetalle'))).toEqual([]);
    expect(visibleResolved(SHT_FILES, [])).toEqual([]);
  });
});

/**
 * `planCollectionAdoption`/`formatMissingLogMessage`/`resolveAllowedPaths`
 * are IMPORTED from `collectionAdopt.js` — the module `adoptCollection` and
 * the `allowedPaths` effect in IndexingPage.jsx actually call — not
 * reimplemented here. A round of review on this task correctly rejected an
 * earlier version of this file that duplicated `adoptCollection`'s mapping
 * logic in a local function and tested THAT: it proved the maths but nothing
 * about the shipped code, and the wiring it stood in for was checked only by
 * source-text regexes matching things like `data.missing.length` — which
 * pass whether that code is reachable or not. Extracting the logic into an
 * importable module means these tests exercise the exact code path
 * IndexingPage runs.
 */
describe('planCollectionAdoption (real, imported from collectionAdopt.js)', () => {
  const discovered = [
    { path: 'D:/lib/CIF_Library/Al.cif', filename: 'Al.cif', formula: 'Al', display_label: 'Al (cF4)' },
  ];

  it('prefers the discovered object, which carries display_label/formula', () => {
    const plan = planCollectionAdoption({ paths: ['D:/lib/CIF_Library/Al.cif'], missing: [] }, discovered);
    expect(plan.files).toEqual(discovered);
    expect(plan.phaseFiles).toEqual(['D:/lib/CIF_Library/Al.cif']);
  });

  it('falls back to a derived object when discovery has not seen the path', () => {
    const plan = planCollectionAdoption({ paths: ['D:/lib/CIF_Library/Ni.cif'], missing: [] }, discovered);
    expect(plan.files).toEqual([
      { path: 'D:/lib/CIF_Library/Ni.cif', filename: 'Ni.cif', formula: 'Ni' },
    ]);
  });

  it('keeps files/phaseFiles index-aligned: both derive from one paths.map, in resolve order', () => {
    const resolved = ['D:/lib/CIF_Library/Ni.cif', 'D:/lib/CIF_Library/Al.cif'];
    const plan = planCollectionAdoption({ paths: resolved, missing: [] }, discovered);
    expect(plan.phaseFiles).toEqual(resolved);
    expect(plan.files.map((f) => f.path)).toEqual(resolved);
  });

  it('carries `missing` through unchanged — the adopt button must not drop it', () => {
    const missing = [{ key: 'TiC', reason: 'not_in_library' }];
    const plan = planCollectionAdoption({ paths: [], missing }, discovered);
    expect(plan.missing).toBe(missing);
  });

  it('tolerates a resolve response with no paths/missing field', () => {
    expect(planCollectionAdoption({}, discovered)).toEqual({ files: [], phaseFiles: [], missing: [] });
  });
});

describe('formatMissingLogMessage (real, imported)', () => {
  it('is null when nothing is missing — the caller must not log an empty line', () => {
    expect(formatMissingLogMessage(() => 'x', [])).toBeNull();
    expect(formatMissingLogMessage(() => 'x', null)).toBeNull();
  });

  it('reports the count and joins every missing key — dropping either fails this', () => {
    const calls = [];
    const fakeT = (key, opts) => { calls.push([key, opts]); return opts ? `MSG(${opts.count})` : key; };
    const msg = formatMissingLogMessage(fakeT, [{ key: 'TiC', reason: 'not_in_library' }, { key: 'WC', reason: 'not_in_library' }]);
    expect(msg).toBe('MSG(2): collections:counts.reasonNotInLibrary: TiC, WC');
    expect(calls).toEqual([
      ['collections:counts.notUsableHere', { count: 2 }],
      ['collections:counts.reasonNotInLibrary', undefined],
    ]);
  });

  /**
   * `GET /resolve` reports one of four reasons per missing member
   * (`backend/api/routes/phase_collections.py#resolve`): `no_cif` for
   * Hough, `no_sht` for Spherical, `no_master` for Dictionary, and
   * `not_in_library` when the key has no library entry at all. Those are
   * four different problems with four different fixes — "run a simulation"
   * is not "this phase isn't in your library" — so the log line has to name
   * WHICH one, not just list the keys. This is the change requested by the
   * task-11 review: the manual originally claimed the log already said
   * "and why", which was false until this test (and the implementation it
   * pins) landed.
   */
  it('names the reason, not just the count — a flat key list said nothing about WHICH problem each phase has', () => {
    const fakeT = (key, opts) => (opts ? `${key}(${JSON.stringify(opts)})` : key);
    const msg = formatMissingLogMessage(fakeT, [
      { key: 'Al2O3', reason: 'no_cif' },
      { key: 'TiC', reason: 'not_in_library' },
    ]);
    expect(msg).toBe(
      'collections:counts.notUsableHere({"count":2}): '
      + 'collections:counts.reasonNoCif: Al2O3 · collections:counts.reasonNotInLibrary: TiC'
    );
  });

  it('groups every key under its own reason, in first-seen reason order — no key dropped, none merged into the wrong group', () => {
    const fakeT = (key, opts) => (opts ? `${key}(${opts.count})` : key);
    const msg = formatMissingLogMessage(fakeT, [
      { key: 'WC', reason: 'no_sht' },
      { key: 'Al', reason: 'no_cif' },
      { key: 'Fe3C', reason: 'no_sht' },
      { key: 'TiN', reason: 'no_master' },
    ]);
    expect(msg).toBe(
      'collections:counts.notUsableHere(4): '
      + 'collections:counts.reasonNoSht: WC, Fe3C · '
      + 'collections:counts.reasonNoCif: Al · '
      + 'collections:counts.reasonNoMaster: TiN'
    );
  });

  it('an unrecognised or missing reason falls back to the "not in library" label rather than crashing or dropping the key', () => {
    const fakeT = (key, opts) => (opts ? `${key}(${opts.count})` : key);
    expect(formatMissingLogMessage(fakeT, [{ key: 'Odd', reason: 'something_new' }]))
      .toBe('collections:counts.notUsableHere(1): collections:counts.reasonNotInLibrary: Odd');
    expect(formatMissingLogMessage(fakeT, [{ key: 'NoReasonField' }]))
      .toBe('collections:counts.notUsableHere(1): collections:counts.reasonNotInLibrary: NoReasonField');
  });
});

describe('resolveAllowedPaths (real, imported)', () => {
  it('is null with no active collection', async () => {
    const resolveFn = () => Promise.resolve({ data: { paths: ['x'], missing: [] } });
    expect(await resolveAllowedPaths(null, 'spherical', resolveFn)).toBeNull();
  });

  it('is null for Hough — Hough filters by collectionKeys/stem, never this', async () => {
    const resolveFn = () => Promise.resolve({ data: { paths: ['x'], missing: [] } });
    expect(await resolveAllowedPaths('Matrix', 'hough', resolveFn)).toBeNull();
  });

  it('resolves to the Set of paths on success', async () => {
    const resolveFn = () => Promise.resolve({ data: { paths: ['a', 'b'], missing: [] } });
    const result = await resolveAllowedPaths('Matrix', 'spherical', resolveFn);
    expect(result).toEqual(new Set(['a', 'b']));
  });

  it('REGRESSION: a rejecting resolve produces an empty Set, never null', async () => {
    // A real rejection, not a stand-in: resolveFn's promise actually rejects,
    // and the assertion observes what resolveAllowedPaths returns from
    // inside its own catch — not a duplicate of that catch.
    const resolveFn = () => Promise.reject(new Error('network down'));
    const result = await resolveAllowedPaths('Matrix', 'dictionary', resolveFn);
    expect(result).toBeInstanceOf(Set);
    expect(result.size).toBe(0);
  });
});

/**
 * The pure-function tests above prove the maths. This suite mounts the REAL
 * PhaseDropdown and proves it is actually wired to it — in particular that a
 * Spherical/Dictionary mount ignores `collectionKeys` (which would filter an
 * .sht list to nothing, per the REGRESSION case above) and filters on
 * `allowedPaths` instead, exactly as the interface in the task brief
 * specifies. This is the render-level guard Tasks 7/8 added for their own
 * consumers; a full IndexingPage mount was not attempted here — the page is
 * ~3900 lines with dozens of API dependencies gated behind many effects, and
 * standing up that harness is out of proportion to what this suite needs to
 * prove. What IndexingPage's own handlers do with `planCollectionAdoption`/
 * `formatMissingLogMessage`/`resolveAllowedPaths` is proven directly by the
 * three `describe` blocks above, against the real, imported functions — not
 * by a source-text check of IndexingPage.jsx, which a prior round of this
 * file used and which a review correctly rejected (see the comment above
 * `planCollectionAdoption`'s describe block).
 */
const HOUGH_FILES = [
  { path: 'D:/lib/CIF_Library/Al.cif', filename: 'Al.cif', formula: 'Al', element_group: 'Al-Fe-Si' },
  { path: 'D:/lib/CIF_Library/Al4Fe1.7Si (τ11).cif', filename: 'Al4Fe1.7Si (τ11).cif', formula: 'Al4Fe1.7Si (τ11)', element_group: 'Al-Fe-Si' },
  { path: 'D:/lib/CIF_Library/WC.cif', filename: 'WC.cif', formula: 'WC', element_group: 'C-W' },
];

const SPHERICAL_FILES = [
  { path: SHT_FILES[0].path, filename: 'Al (Al) [cF4] {20kV}.sht', formula: 'Al', element_group: 'Al-Fe-Si' },
  { path: SHT_FILES[1].path, filename: 'Fe1.8Al4.4Si0.6ht (Al4Fe1.7Si (τ11)) [hP28] {20kV}.sht', formula: 'Al4Fe1.7Si (τ11)', element_group: 'Al-Fe-Si' },
  { path: SHT_FILES[2].path, filename: 'WC (WC) [hP2] {20kV}.sht', formula: 'WC', element_group: 'C-W' },
];

function renderDropdown(props) {
  return render(
    <PhaseDropdown
      discoveredFiles={props.discoveredFiles}
      groups={['Al-Fe-Si', 'C-W']}
      selectedPaths={props.selectedPaths || []}
      onTogglePath={() => {}}
      onSetAll={() => {}}
      method={props.method}
      open
      onClose={() => {}}
      collectionKeys={props.collectionKeys ?? null}
      allowedPaths={props.allowedPaths ?? null}
      onShowAll={props.onShowAll}
      overriddenCollectionName={props.overriddenCollectionName ?? null}
      onReapplyCollection={props.onReapplyCollection}
    />
  );
}

describe('PhaseDropdown mounted for real, under a collection', () => {
  afterEach(() => cleanup());

  it('Hough: renders only the collection members, by stem', () => {
    renderDropdown({
      discoveredFiles: HOUGH_FILES,
      method: 'hough',
      collectionKeys: activeKeySet(COLS, 'Intermetallics in Al'),
    });
    screen.getByText('Al');
    screen.getByText('Al4Fe1.7Si (τ11)');
    expect(screen.queryByText('WC')).toBeNull();
  });

  it('Spherical: ignores collectionKeys (would zero the list) and filters on allowedPaths', () => {
    // The collection's stems ('Al', 'Al4Fe1.7Si (τ11)') do not match ANY .sht
    // filename stem — passing this through unchanged is exactly the
    // REGRESSION case above. A component that (re-)introduced stem
    // filtering for Spherical would render nothing at all here.
    renderDropdown({
      discoveredFiles: SPHERICAL_FILES,
      method: 'spherical',
      collectionKeys: activeKeySet(COLS, 'Intermetallics in Al'),
      allowedPaths: new Set([SHT_FILES[0].path, SHT_FILES[1].path]),
    });
    screen.getByText('Al');
    screen.getByText('Al4Fe1.7Si (τ11)');
    expect(screen.queryByText('WC')).toBeNull();
  });

  it('Spherical: a resolve that found nothing offers nothing (not the whole library)', () => {
    renderDropdown({
      discoveredFiles: SPHERICAL_FILES,
      method: 'spherical',
      collectionKeys: activeKeySet(COLS, 'Intermetallics in Al'),
      allowedPaths: new Set(),   // failed / no-match resolve — never null here
    });
    expect(screen.queryByText('Al')).toBeNull();
    expect(screen.queryByText('Al4Fe1.7Si (τ11)')).toBeNull();
    expect(screen.queryByText('WC')).toBeNull();
  });

  it('shows "Show all phases" while a collection filters the list, and it fires onShowAll', () => {
    const onShowAll = () => { onShowAll.called = true; };
    renderDropdown({
      discoveredFiles: HOUGH_FILES,
      method: 'hough',
      collectionKeys: activeKeySet(COLS, 'Intermetallics in Al'),
      onShowAll,
    });
    const btn = screen.getByText('Show all phases');
    fireEvent.click(btn);
    expect(onShowAll.called).toBe(true);
  });

  it('no "Show all phases" escape when nothing is filtering (collectionKeys null)', () => {
    renderDropdown({ discoveredFiles: HOUGH_FILES, method: 'hough', collectionKeys: null });
    expect(screen.queryByText('Show all phases')).toBeNull();
    // And the full, unfiltered library is offered.
    screen.getByText('Al');
    screen.getByText('Al4Fe1.7Si (τ11)');
    screen.getByText('WC');
  });
});

/**
 * SMALLER finding (whole-branch review): after "Show all phases" is used,
 * `collectionKeys`/`allowedPaths` both go to `null` for THIS picker and the
 * escape row above disappears with them (it is a filter escape, only shown
 * `{collectionKeys && onShowAll}` — by design, since a null `collectionKeys`
 * means there is nothing left to escape FROM). But `null` here does not mean
 * "no collection is active" — the toolbar can still read "Collection:
 * Matrix" while this picker quietly ignores it, and until now nothing on the
 * picker said so. `overriddenCollectionName` (set by IndexingPage.jsx only
 * while its `overrideAll` local override is on) drives a SEPARATE,
 * persistent notice for exactly this case.
 */
describe('PhaseDropdown names an overridden collection instead of going silent', () => {
  afterEach(() => cleanup());

  it('shows a persistent notice naming the overridden collection, with a way back', () => {
    const onReapplyCollection = () => { onReapplyCollection.called = true; };
    renderDropdown({
      discoveredFiles: HOUGH_FILES, method: 'hough',
      collectionKeys: null,   // overridden, not "no collection active"
      overriddenCollectionName: 'Intermetallics in Al',
      onReapplyCollection,
    });
    screen.getByText(/Intermetallics in Al/);
    // The escape-to-show-all row is gone (collectionKeys is null) — this is
    // the DIFFERENT, persistent row that must be there instead.
    expect(screen.queryByText('Show all phases')).toBeNull();
    fireEvent.click(screen.getByText('Re-apply'));
    expect(onReapplyCollection.called).toBe(true);
  });

  it('shows nothing when no collection is overridden (the ordinary unfiltered state)', () => {
    renderDropdown({
      discoveredFiles: HOUGH_FILES, method: 'hough',
      collectionKeys: null, overriddenCollectionName: null,
    });
    expect(screen.queryByText('Re-apply')).toBeNull();
  });

  it('shows nothing while a collection is actively filtering (the escape row covers that case)', () => {
    renderDropdown({
      discoveredFiles: HOUGH_FILES, method: 'hough',
      collectionKeys: activeKeySet(COLS, 'Intermetallics in Al'),
      overriddenCollectionName: null,
      onShowAll: () => {},
    });
    screen.getByText('Show all phases');
    expect(screen.queryByText('Re-apply')).toBeNull();
  });
});
