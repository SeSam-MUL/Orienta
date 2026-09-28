// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';

/**
 * Whole-branch review (2026-09-26), "tests that do not bite": this file used
 * to open with a hand-rolled `mergeUnderCollection(prev, phases, appeared,
 * collectionKeys)` reconstruction of the refresh-on-open merge rule
 * (golive 2026-09-12), plus four tests against it. It was a WRONG
 * reconstruction: for a phase already in `prev`, it re-checked `isIn`
 * (`if (prev.has(p.key)) { if (isIn) next.add(p.key); continue; }`) and
 * DROPPED it when the phase was outside the active collection. The real
 * merge in `PhaseMapPanel.jsx`'s `refreshCifPhases` does not do that —
 * `if (prev.has(p.key) || appearedIn.has(p.key)) next.add(p.key)` keeps ANY
 * already-selected phase that is still present in the library, regardless of
 * collection membership; the collection only gates phases that are NEWLY
 * ARRIVING. A reader taking the deleted helper at face value would believe
 * the opposite of what ships. It has been removed rather than "corrected" in
 * place — this codebase's own standing rule is to test the real imported
 * function, not a reimplementation of it (see this file's own comment on the
 * render-level suite below, and `indexingCollection.test.jsx`'s identical
 * rejection of a local stand-in for `planCollectionAdoption`).
 *
 * The render-level suite below ("PhaseMapControls: refreshCifPhases gates
 * newly-arrived phases by the collection") is what actually guards this
 * behaviour, against the real `usePhaseMap`/`PhaseMapControls` — including a
 * dedicated regression for exactly the property the deleted helper got
 * backwards: a previously-ticked, now out-of-collection phase must survive a
 * refresh untouched.
 */

/**
 * A pure-function reimplementation of the merge rule would still pass if
 * PhaseMapPanel.jsx never called `keyForPath`, never read the collection
 * live, or never wired the count line to a real state — Task 7 hit exactly
 * this gap (`phaseTestCollection.test.jsx`), which is also why the wrong
 * reconstruction above was deleted rather than fixed in place. This suite
 * renders the real `usePhaseMap` + `PhaseMapControls` and drives real state
 * through them, real translations included (no react-i18next mock —
 * `src/test/i18n-setup.js` loads the English bundles, so
 * `collections:counts.usableHere` renders as "2 of 3 phases · Matrix", not a
 * raw key).
 */
vi.mock('../../services/api', () => ({
  edsApi: {
    getPhaseMap: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(),
    elements: vi.fn(() => Promise.resolve({ data: { elements: [] } })),
    autoClassify: vi.fn(),
    clearPhaseMap: vi.fn(),
  },
  edsExportApi: {
    listPresets: vi.fn(() => Promise.resolve({ data: { presets: [] } })),
    getPreset: vi.fn(), checkPreset: vi.fn(), savePreset: vi.fn(),
    deletePreset: vi.fn(), importPreset: vi.fn(), exportPresetFile: vi.fn(),
    exportPreview: vi.fn(), runExport: vi.fn(),
  },
}));

const { edsApi } = await import('../../services/api');
const { default: useDataStore } = await import('../../stores/useDataStore');
const { default: useCollectionStore } = await import('../../stores/useCollectionStore');
const { usePhaseMap, PhaseMapControls } = await import('./PhaseMapPanel');

// Al + Si make up "Matrix". WC is a collection member with no matching CIF in
// the library at all (edsApi.cifPhases never returns it) — the EDS analogue
// of Task 7's "no .sht" case: it must count as "not usable", here spelled
// "not in the library". "Carbides" (Ni + WC, only Ni present) exists purely
// so the follow-effect tests below have a second, DIFFERENT collection to
// switch to.
const COLLECTIONS = [
  { name: 'Matrix', parent: null, members: [{ key: 'Al' }, { key: 'Si' }, { key: 'WC' }] },
  { name: 'Carbides', parent: null, members: [{ key: 'Ni' }, { key: 'WC' }] },
];

const THREE_PHASES = {
  data: {
    phases: [
      { key: 'Al.cif', cif_filename: 'Al.cif', formula: 'Al' },
      { key: 'Si.cif', cif_filename: 'Si.cif', formula: 'Si' },
      { key: 'Ni.cif', cif_filename: 'Ni.cif', formula: 'Ni' },
    ],
  },
};

function setActiveCollection(name) {
  useCollectionStore.setState({
    data: { collections: COLLECTIONS, unassigned: [], problems: [], state: { active: name, hidden: [] } },
  });
}

/**
 * Change a collection's OWN members without touching `state.active` — the
 * scenario Task 10 makes reachable (the database browser's "move to
 * collection", or the Collection Manager, editing the collection this panel
 * is already showing). A fresh `collections` array and a fresh `data`
 * object, matching what a real `useCollectionStore.load()` reply looks like
 * (brand new every time) — the same "reference changed but not the content"
 * question `activeKeySignature` exists to answer.
 */
function updateCollectionMembers(name, members) {
  useCollectionStore.setState((s) => ({
    data: {
      ...s.data,
      collections: s.data.collections.map((c) => (c.name === name ? { ...c, members } : c)),
    },
  }));
}

function Harness() {
  const handle = usePhaseMap({});
  return <PhaseMapControls handle={handle} />;
}

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  useDataStore.setState({ filePath: 'D:/scans/one.h5oina' });
  useCollectionStore.setState({
    data: { collections: [], unassigned: [], problems: [], state: {} },
  });
  edsApi.cifPhases.mockResolvedValue(THREE_PHASES);
});
afterEach(() => cleanup());

async function renderPanel() {
  render(<Harness />);
  await waitFor(() => expect(edsApi.cifPhases).toHaveBeenCalled());
  // Let the fetch's .then() resolve and commit its setState calls.
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
}

function checkbox(filename) {
  return screen.getByText(filename).closest('label').querySelector('input');
}

describe('PhaseMapControls under an active collection', () => {
  it('seeds the selection to the collection, keeping every phase visible', async () => {
    setActiveCollection('Matrix');
    await renderPanel();

    // All three library phases are still OFFERED (`cifPhases` stays full) —
    // getByText throws if any is missing, which is assertion enough.
    screen.getByText('Al.cif');
    screen.getByText('Si.cif');
    screen.getByText('Ni.cif');

    // ...but only Al and Si (the collection's members present in the
    // library) are checked.
    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(true);
    expect(checkbox('Ni.cif').checked).toBe(false);

    screen.getByText(/Phases taking part \(2\/3\)/);
  });

  it('shows the usable/not-usable counts and the collection name', async () => {
    setActiveCollection('Matrix');
    await renderPanel();

    // 2 usable (Al, Si) of 3 in-collection (Al, Si, WC); WC has no matching
    // CIF in the library.
    screen.getByText(/2 of 3 phases/);
    screen.getByText(/Matrix/);
    screen.getByText(/1 phase is not in the library/);
  });

  it('"Show all phases" restores every phase with one click', async () => {
    setActiveCollection('Matrix');
    await renderPanel();

    expect(checkbox('Ni.cif').checked).toBe(false);
    fireEvent.click(screen.getByText('Show all phases'));
    expect(checkbox('Ni.cif').checked).toBe(true);
    screen.getByText(/Phases taking part \(3\/3\)/);
  });

  it('with no collection active, every phase is selected and the header is absent', async () => {
    await renderPanel();   // no setActiveCollection() call: state.active stays unset

    expect(checkbox('Ni.cif').checked).toBe(true);
    expect(screen.queryByText('Show all phases')).toBeNull();
    screen.getByText(/Phases taking part \(3\/3\)/);
  });
});

/**
 * App.jsx keeps every page mounted and only hides inactive ones with
 * `display: none` — the EDS page (and this hook's state with it) can sit
 * hidden while the user flips the toolbar's active collection elsewhere and
 * comes back. These tests drive that scenario directly: change
 * `useCollectionStore`'s `state.active` WITHOUT remounting the panel, exactly
 * as the zustand subscription would when a hidden page's store dependency
 * changes. Same pattern as SinglePixelPhaseTestDialog's
 * "follows a LIVE collection switch" suite.
 */
describe('PhaseMapControls follows a LIVE collection switch', () => {
  it('an untouched selection follows the collection when it changes underneath it', async () => {
    setActiveCollection('Matrix');
    await renderPanel();

    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(true);
    expect(checkbox('Ni.cif').checked).toBe(false);

    // Switch the ACTIVE collection without touching the panel at all — this
    // is the toolbar click that happens on a different, visible page.
    act(() => { setActiveCollection('Carbides'); });

    // The untouched selection follows: now only Ni (Carbides' member present
    // in the library).
    expect(checkbox('Al.cif').checked).toBe(false);
    expect(checkbox('Si.cif').checked).toBe(false);
    expect(checkbox('Ni.cif').checked).toBe(true);
    screen.getByText(/Phases taking part \(1\/3\)/);
    screen.getByText(/Carbides/);
  });

  it('REGRESSION: a hand-edited selection does NOT follow a later collection switch', async () => {
    setActiveCollection('Matrix');
    await renderPanel();

    // Hand-edit: check Ni too (Matrix's seed was Al+Si only).
    fireEvent.click(checkbox('Ni.cif'));
    expect(checkbox('Ni.cif').checked).toBe(true);

    // Now switch collections underneath the panel, same as the test above.
    act(() => { setActiveCollection('Carbides'); });

    // The hand-edited selection (Al, Si, Ni) must survive UNCHANGED — it
    // must NOT collapse to Carbides' usable members (Ni only).
    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(true);
    expect(checkbox('Ni.cif').checked).toBe(true);
    screen.getByText(/Phases taking part \(3\/3\)/);
  });

  it('RACE: seeds from the collection active when cifPhases RESOLVES, not the one active when it was CALLED', async () => {
    // Same failure family as the two tests above, reached through a
    // different door: `loadCifPhases`'s `.then()` must not close over
    // `collections`/`activeName` at the moment the effect is scheduled. If
    // the active collection changes WHILE the request is still in flight, a
    // `.then()` that trusts a stale closure would seed from the OLD
    // (request-time) collection while the header already names the NEW one.
    setActiveCollection('Matrix');

    // Hold the fetch open deliberately — the default mock resolves on the
    // next microtask, which would make this test pass even with the bug,
    // proving nothing.
    let resolveFetch;
    edsApi.cifPhases.mockImplementationOnce(
      () => new Promise((resolve) => { resolveFetch = resolve; }));

    render(<Harness />);
    await waitFor(() => expect(edsApi.cifPhases).toHaveBeenCalled());

    // Switch collections WHILE the request is still pending.
    act(() => { setActiveCollection('Carbides'); });

    // NOW let the fetch resolve, with Carbides active.
    await act(async () => {
      resolveFetch(THREE_PHASES);
      await Promise.resolve(); await Promise.resolve();
    });

    // The seed must match CARBIDES (Ni only) — the collection active when
    // the phases ARRIVED — not Matrix (Al+Si), which was only active when
    // the request was SENT.
    expect(checkbox('Al.cif').checked).toBe(false);
    expect(checkbox('Si.cif').checked).toBe(false);
    expect(checkbox('Ni.cif').checked).toBe(true);
    screen.getByText(/Phases taking part \(1\/3\)/);
    screen.getByText(/Carbides/);
  });
});

/**
 * Task 10's inherited debt: Tasks 7/8 only re-seeded on a NAME change
 * (`[activeName]`). Nothing could edit a collection's membership while a
 * picker was open until this task's database browser / Collection Manager
 * existed — so a user adding a phase to the STILL-active collection and
 * flipping back to this panel is now reachable, and the effect must widen to
 * catch it (`[activeName, membersSignature]`, see PhaseMapPanel.jsx's own
 * comment on `activeKeySignature`).
 */
describe('PhaseMapControls follows a membership change to the ACTIVE collection', () => {
  it('an untouched selection re-seeds when the active collection gains/loses a member, name unchanged', async () => {
    setActiveCollection('Matrix');   // Al, Si, WC — Al/Si present in the library
    await renderPanel();

    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(true);
    expect(checkbox('Ni.cif').checked).toBe(false);

    // "Matrix" stays active BY NAME, but Si is swapped for Ni — exactly what
    // moving a phase in/out via the database browser's "move to collection"
    // does. No `setActiveCollection` call: only membership moved.
    act(() => { updateCollectionMembers('Matrix', [{ key: 'Al' }, { key: 'Ni' }, { key: 'WC' }]); });

    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(false);   // left the collection
    expect(checkbox('Ni.cif').checked).toBe(true);    // joined it
    screen.getByText(/Phases taking part \(2\/3\)/);
  });

  it('REGRESSION: a hand-edited selection does NOT get overwritten by a later membership change', async () => {
    setActiveCollection('Matrix');
    await renderPanel();

    // Hand-edit: add Ni to Matrix's seed (Al, Si).
    fireEvent.click(checkbox('Ni.cif'));
    expect(checkbox('Ni.cif').checked).toBe(true);

    // Now grow "Matrix" itself — same scenario as the test above, but the
    // selection was already hand-touched and must survive as-is.
    act(() => { updateCollectionMembers('Matrix', [{ key: 'Al' }, { key: 'Si' }, { key: 'Ni' }, { key: 'WC' }]); });

    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(true);
    expect(checkbox('Ni.cif').checked).toBe(true);
    screen.getByText(/Phases taking part \(3\/3\)/);   // the hand edit, not a reseed
  });
});

/**
 * Proves the OTHER half of the wiring: `refreshCifPhases` (triggered by
 * opening the `<details>`) gates a newly-arrived CIF by the active
 * collection, using the REAL component rather than the hand-rolled
 * `mergeUnderCollection` at the top of this file. This is the render-level
 * counterpart `phaseSelectionRefreshWire.test.jsx` is for the plain
 * (no-collection) refresh contract.
 */
describe('PhaseMapControls: refreshCifPhases gates newly-arrived phases by the collection', () => {
  async function openPicker() {
    const d = screen.getByTestId('phase-selection');
    await act(async () => {
      d.open = true;
      await new Promise((resolve) => { setTimeout(resolve, 0); });
    });
  }

  it('a CIF that appears mid-session outside the collection is not ticked, and is named', async () => {
    // Matrix's members are Al, Si, WC — so the newly-arrived phase here must
    // be something Matrix does NOT claim (WC would land in the "inside" case).
    setActiveCollection('Matrix');
    await renderPanel();
    expect(screen.queryByText('Mg2Si.cif')).toBeNull();

    edsApi.cifPhases.mockResolvedValue({
      data: { phases: [...THREE_PHASES.data.phases,
                        { key: 'Mg2Si.cif', cif_filename: 'Mg2Si.cif', formula: 'Mg2Si' }] },
    });
    await openPicker();

    screen.getByText('Mg2Si.cif');                       // offered...
    expect(checkbox('Mg2Si.cif').checked).toBe(false);    // ...but not ticked
    // "group", not "collection": the word changed in the interface when
    // the manager dialog went. The key and the file on disk did not.
    screen.getByText(/1 new phase is outside this group/);
  });

  it('a CIF that appears mid-session inside the collection is ticked', async () => {
    setActiveCollection('Carbides');   // members: Ni, WC
    await renderPanel();
    // Ni is already in THREE_PHASES and already ticked; WC is the new arrival.
    expect(checkbox('Ni.cif').checked).toBe(true);

    edsApi.cifPhases.mockResolvedValue({
      data: { phases: [...THREE_PHASES.data.phases,
                        { key: 'WC.cif', cif_filename: 'WC.cif', formula: 'WC' }] },
    });
    await openPicker();

    expect(checkbox('WC.cif').checked).toBe(true);
    expect(screen.queryByText(/is outside this group/)).toBeNull();
  });

  it('a previously-ticked phase that is now outside the collection SURVIVES a refresh — the property the deleted mergeUnderCollection got backwards', async () => {
    // Matrix's members are Al, Si, WC; Ni is NOT in Matrix. Ni starts
    // unticked (Matrix seeds Al+Si only) — hand-tick it, which freezes the
    // auto-seed (isAutoSeedRef -> false) so the SEPARATE "follow the active
    // collection" effect stops overwriting the selection, isolating this
    // test to `refreshCifPhases`'s OWN merge alone.
    setActiveCollection('Matrix');
    await renderPanel();
    expect(checkbox('Ni.cif').checked).toBe(false);
    fireEvent.click(checkbox('Ni.cif'));
    expect(checkbox('Ni.cif').checked).toBe(true);

    // Refresh with the SAME three phases — nothing new arrives, so nothing
    // is gated by `appearedIn`/`appearedOutside` at all. The only question
    // is whether a phase already in `prev` (Ni.cif) survives being outside
    // the collection. The deleted `mergeUnderCollection` re-checked
    // collection membership for already-selected phases and would have
    // DROPPED Ni.cif here; the real `refreshCifPhases` does not.
    edsApi.cifPhases.mockResolvedValue(THREE_PHASES);
    await openPicker();

    expect(checkbox('Al.cif').checked).toBe(true);
    expect(checkbox('Si.cif').checked).toBe(true);
    expect(checkbox('Ni.cif').checked).toBe(true);   // <- survives, not dropped
    screen.getByText(/Phases taking part \(3\/3\)/);
  });
});
