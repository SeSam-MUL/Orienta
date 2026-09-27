// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';
import { activeKeySet, narrowSelection } from '../PhaseCollections/collectionFilter';

/**
 * The trap this file exists for.
 *
 * SinglePixelPhaseTestDialog decides whether to send `phase_keys` like this:
 *
 *   allSelected   = phases.length === 0 || selectedKeys.size === phases.length
 *   subsetSelected= 0 < selectedKeys.size < phases.length
 *   if (subsetSelected) params.phase_keys = [...selectedKeys]
 *
 * So if a collection filter shrinks the RENDERED list instead of the
 * SELECTION, `selectedKeys.size === phases.length` becomes true again, no
 * `phase_keys` is sent, and the backend tests the whole library — the exact
 * opposite of what the toolbar says. `phases` must stay the full list.
 */
function decide(phases, selectedKeys) {
  const allSelected = phases.length === 0 || selectedKeys.size === phases.length;
  const subset = phases.length > 0 && selectedKeys.size > 0
              && selectedKeys.size < phases.length;
  return { allSelected, sends: subset ? [...selectedKeys].sort() : null };
}

const PHASES = [{ key: 'Al' }, { key: 'Si' }, { key: 'Ni' }, { key: 'WC' }];
const COLS = [{ name: 'Matrix', parent: null, members: [{ key: 'Al' }, { key: 'Si' }] }];

describe('phase tester under an active collection', () => {
  it('sends exactly the collection, because the full list is kept', () => {
    const keys = activeKeySet(COLS, 'Matrix');
    const { keys: sel } = narrowSelection(
      PHASES.map((p) => p.key), keys, PHASES.map((p) => p.key));
    expect(decide(PHASES, new Set(sel)).sends).toEqual(['Al', 'Si']);
  });

  it('REGRESSION: shrinking the rendered list would send nothing at all', () => {
    const keys = activeKeySet(COLS, 'Matrix');
    const shrunk = PHASES.filter((p) => keys.has(p.key));
    const bad = decide(shrunk, new Set(shrunk.map((p) => p.key)));
    expect(bad.allSelected).toBe(true);
    expect(bad.sends).toBeNull();          // the whole library would be tested
  });

  it('"show all phases" restores the full selection', () => {
    const { keys: sel } = narrowSelection(
      PHASES.map((p) => p.key), null, PHASES.map((p) => p.key));
    expect(decide(PHASES, new Set(sel)).sends).toBeNull();   // = all, correctly
    expect(sel).toHaveLength(4);
  });

  it('a collection with nothing testable here selects nothing, and says so', () => {
    const cols = [{ name: 'Hartmetalle', parent: null, members: [{ key: 'WC' }] }];
    const r = narrowSelection(['Al', 'Si'], activeKeySet(cols, 'Hartmetalle'), ['Al', 'Si']);
    expect(r.keys).toEqual([]);
    expect(r.inCollection).toBe(1);
    expect(r.usableHere).toBe(0);

    // CRITICAL (C1): this is the exact trap in the review report — feed the
    // empty selection above through the dialog's own `decide()` and the
    // library-wide run reappears. `allSelected` is false (0 !== 2 phases),
    // so the label would read "Run 0 selected" — but `subsetSelected` is
    // ALSO false (0 is not > 0), so `phase_keys` is omitted from the
    // request. `backend/api/routes/indexing.py`'s `if phase_keys:` reads an
    // absent list as "test every phase". Zero on screen, everything on the
    // wire. The dialog itself must refuse this case outright (see the
    // render-level "refuses to run" tests below) rather than ever reaching
    // `decide()` with an empty selection.
    const localPhases = ['Al', 'Si'].map((key) => ({ key }));
    const trapped = decide(localPhases, new Set(r.keys));
    expect(trapped.allSelected).toBe(false);
    expect(trapped.sends).toBeNull();
  });
});

/**
 * The pure-function tests above prove the maths. This suite proves
 * SinglePixelPhaseTestDialog is actually WIRED to them: that the fetch
 * effect seeds `selectedKeys` from the active collection while `phases`
 * stays full, that the new count row is correct, and that "Show all
 * phases" really is a one-click way back to the whole library.
 */
vi.mock('../../services/api', () => ({
  indexApi: {
    phaseTestPhases: vi.fn(() => Promise.resolve({
      data: [
        { key: 'Al', label: 'Al (Al)' },
        { key: 'Si', label: 'Si (Si)' },
        { key: 'Ni', label: 'Ni (Ni)' },
      ],
    })),
    phaseTestStart: vi.fn(() => Promise.resolve({ data: { job_id: 'job-1', total: 1 } })),
    phaseTestPixelChemistry: vi.fn(() => Promise.resolve({ data: { pixel_at_pct: null } })),
  },
  ebsdApi: {
    overview: vi.fn(() => Promise.resolve({ data: { image: null } })),
    getPattern: vi.fn(() => Promise.resolve({ data: { image: null } })),
  },
  edsApi: {
    elements: vi.fn(() => Promise.resolve({ data: { elements: [] } })),
    getMap: vi.fn(() => Promise.resolve({ data: { image: null } })),
  },
}));

const { indexApi } = await import('../../services/api');
const { default: useDataStore } = await import('../../stores/useDataStore');
const { default: useCollectionStore } = await import('../../stores/useCollectionStore');
const { default: SinglePixelPhaseTestDialog } = await import('./SinglePixelPhaseTestDialog');

// Al + Si make up "Matrix". WC is a collection member but has no .sht here
// (phaseTestPhases never returns it), so it must count as "not usable".
// "Carbides" (Ni + WC, only Ni usable) exists purely so the follow-effect
// tests below have a second, DIFFERENT collection to switch to.
const COLLECTIONS = [
  { name: 'Matrix', parent: null, members: [{ key: 'Al' }, { key: 'Si' }, { key: 'WC' }] },
  { name: 'Carbides', parent: null, members: [{ key: 'Ni' }, { key: 'WC' }] },
];

function setActiveCollection(name) {
  useCollectionStore.setState({
    data: { collections: COLLECTIONS, unassigned: [], problems: [], state: { active: name, hidden: [] } },
  });
}

/**
 * Change a collection's OWN members without touching `state.active` — the
 * scenario Task 10 makes reachable (the database browser's "move to
 * collection", or the Collection Manager, editing the collection a dialog
 * elsewhere is already showing). Builds a fresh `collections` array and a
 * fresh `data` object, matching what a real `useCollectionStore.load()`
 * reply looks like (a brand-new object every time), so this exercises the
 * same "did the reference change but not the content" question
 * `activeKeySignature` exists to answer.
 */
function updateCollectionMembers(name, members) {
  useCollectionStore.setState((s) => ({
    data: {
      ...s.data,
      collections: s.data.collections.map((c) => (c.name === name ? { ...c, members } : c)),
    },
  }));
}

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  useDataStore.setState({
    isFileOpen: true,
    ebsdLoaded: true,
    filePath: 'D:/scans/one.h5oina',
    gridShape: [4, 4],
    currentIndex: 0,
  });
  useCollectionStore.setState({
    data: { collections: [], unassigned: [], problems: [], state: {} },
  });
});

afterEach(() => { cleanup(); });

async function openDialog() {
  render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
  await waitFor(() => expect(indexApi.phaseTestPhases).toHaveBeenCalled());
  // Let the fetch's .then() resolve and commit its setState calls.
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
}

describe('SinglePixelPhaseTestDialog under an active collection', () => {
  it('seeds the selection to the collection, keeping every phase visible', async () => {
    setActiveCollection('Matrix');
    await openDialog();

    // Expand the phase picker.
    fireEvent.click(screen.getByText(/Phases \(/));

    // All three testable phases are still OFFERED (phases stays full) —
    // getByText throws if any is missing, which is assertion enough.
    screen.getByText('Al (Al)');
    screen.getByText('Si (Si)');
    screen.getByText('Ni (Ni)');

    // ...but only Al and Si (the collection's usable members) are checked.
    const alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    const siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    const niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(true);
    expect(siCheckbox.checked).toBe(true);
    expect(niCheckbox.checked).toBe(false);

    // The run button reflects a 2-of-3 subset, not "auto-run all".
    screen.getByText(/Run 2 selected/);
  });

  it('shows the usable/not-usable counts and the collection name', async () => {
    setActiveCollection('Matrix');
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    // 2 usable (Al, Si) of 3 in-collection (Al, Si, WC); WC has no .sht here.
    screen.getByText(/2 of 3 phases/);
    screen.getByText(/Matrix/);
    screen.getByText(/1 cannot be used here/);
  });

  it('"Show all phases" restores every phase with one click', async () => {
    setActiveCollection('Matrix');
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    screen.getByText(/Run 2 selected/);

    fireEvent.click(screen.getByText('Show all phases'));

    const niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(niCheckbox.checked).toBe(true);
    screen.getByText(/Auto-Run all phases/);
  });

  it('with no collection active, every phase is selected and the header is absent', async () => {
    await openDialog();  // no setActiveCollection() call: state.active stays unset
    fireEvent.click(screen.getByText(/Phases \(/));

    const niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(niCheckbox.checked).toBe(true);
    expect(screen.queryByText('Show all phases')).toBeNull();
    screen.getByText(/Auto-Run all phases/);
  });
});

/**
 * App.jsx keeps every page mounted and only hides inactive ones with
 * `display: none` — `showPhaseTestDialog` is untouched by the `isActive`
 * prop, so this dialog can sit open, hidden, while the user navigates away
 * and flips the toolbar's active collection elsewhere, then navigates back.
 * These tests drive that scenario directly: change `useCollectionStore`'s
 * state.active WITHOUT closing/reopening the dialog, exactly as the zustand
 * subscription would when a hidden page's store dependency changes.
 */
describe('SinglePixelPhaseTestDialog follows a LIVE collection switch', () => {
  it('an untouched selection follows the collection when it changes underneath it', async () => {
    setActiveCollection('Matrix');
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    // Seeded to Matrix's usable members (Al, Si).
    let alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    let siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    let niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(true);
    expect(siCheckbox.checked).toBe(true);
    expect(niCheckbox.checked).toBe(false);

    // Switch the ACTIVE collection without touching the dialog at all —
    // this is the toolbar click that happens on a different, visible page.
    act(() => { setActiveCollection('Carbides'); });

    // The untouched selection follows: now only Ni (Carbides' usable member).
    alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(false);
    expect(siCheckbox.checked).toBe(false);
    expect(niCheckbox.checked).toBe(true);
    screen.getByText(/Run 1 selected/);
    screen.getByText(/Carbides/);
  });

  it('REGRESSION: a hand-edited selection does NOT follow a later collection switch', async () => {
    setActiveCollection('Matrix');
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    // Hand-edit: check Ni too (Matrix's seed was Al+Si only).
    const niLabel = screen.getByText('Ni (Ni)').closest('label');
    fireEvent.click(niLabel.querySelector('input'));
    expect(niLabel.querySelector('input').checked).toBe(true);

    // Now switch collections underneath the dialog, same as the test above.
    act(() => { setActiveCollection('Carbides'); });

    // The hand-edited selection (Al, Si, Ni) must survive UNCHANGED — it must
    // NOT collapse to Carbides' usable members (Ni only).
    const alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    const siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    const niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(true);
    expect(siCheckbox.checked).toBe(true);
    expect(niCheckbox.checked).toBe(true);
    screen.getByText(/Auto-Run all phases/);   // all 3 of 3 selected, not "1 selected"
  });

  it('RACE: seeds from the collection active when phaseTestPhases RESOLVES, not the one active when it was CALLED', async () => {
    // Same failure family as the two tests above, reached through a
    // different door: the fetch effect's `.then()` closes over
    // `collections`/`activeName` at the moment the effect is scheduled. If
    // the active collection changes WHILE the request is still in flight,
    // the `[activeName]`-keyed follow effect fires immediately for the new
    // collection — but `phases` is still empty at that point, so it's a
    // no-op — and it will not fire again once the fetch finally resolves,
    // because `activeName` itself does not change a second time. A `.then()`
    // that trusts its closure would seed from the OLD (request-time)
    // collection while the header already names the NEW one.
    setActiveCollection('Matrix');

    // Hold the fetch open deliberately — the default mock (see the `vi.mock`
    // factory above) resolves on the next microtask, which would make this
    // test pass even with the bug, proving nothing.
    let resolveFetch;
    indexApi.phaseTestPhases.mockImplementationOnce(
      () => new Promise((resolve) => { resolveFetch = resolve; }));

    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() => expect(indexApi.phaseTestPhases).toHaveBeenCalled());

    // Switch collections WHILE the request is still pending.
    act(() => { setActiveCollection('Carbides'); });

    // NOW let the fetch resolve, with Carbides active.
    await act(async () => {
      resolveFetch({
        data: [
          { key: 'Al', label: 'Al (Al)' },
          { key: 'Si', label: 'Si (Si)' },
          { key: 'Ni', label: 'Ni (Ni)' },
        ],
      });
      await Promise.resolve(); await Promise.resolve();
    });

    fireEvent.click(screen.getByText(/Phases \(/));

    // The seed must match CARBIDES (Ni only) — the collection active when
    // the phases ARRIVED — not Matrix (Al+Si), which was only active when
    // the request was SENT.
    const alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    const siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    const niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(false);
    expect(siCheckbox.checked).toBe(false);
    expect(niCheckbox.checked).toBe(true);
    screen.getByText(/Run 1 selected/);
    screen.getByText(/Carbides/);
  });
});

/**
 * Task 10's inherited debt: Tasks 7/8 only re-seeded on a NAME change
 * (`[activeName]`). Nothing could edit a collection's membership while a
 * picker was open until this task's database browser / Collection Manager
 * existed — so a user adding a phase to the STILL-active collection and
 * flipping back to this dialog is now reachable, and the effect must widen
 * to catch it (`[activeName, membersSignature]`, see the dialog's own
 * comment on `activeKeySignature`).
 */
describe('SinglePixelPhaseTestDialog follows a membership change to the ACTIVE collection', () => {
  it('an untouched selection re-seeds when the active collection gains/loses a member, name unchanged', async () => {
    setActiveCollection('Matrix');   // Al, Si, WC — Al/Si usable here
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    let alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    let siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    let niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(true);
    expect(siCheckbox.checked).toBe(true);
    expect(niCheckbox.checked).toBe(false);

    // "Matrix" stays active BY NAME, but Si is swapped for Ni — exactly what
    // moving a phase in/out via the database browser's "move to collection"
    // does. No `setActiveCollection` call: only membership moved.
    act(() => { updateCollectionMembers('Matrix', [{ key: 'Al' }, { key: 'Ni' }, { key: 'WC' }]); });

    alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(true);
    expect(siCheckbox.checked).toBe(false);   // left the collection
    expect(niCheckbox.checked).toBe(true);    // joined it
    screen.getByText(/Run 2 selected/);
  });

  it('REGRESSION: a hand-edited selection does NOT get overwritten by a later membership change', async () => {
    setActiveCollection('Matrix');
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    // Hand-edit: add Ni to Matrix's seed (Al, Si).
    const niLabel = screen.getByText('Ni (Ni)').closest('label');
    fireEvent.click(niLabel.querySelector('input'));
    expect(niLabel.querySelector('input').checked).toBe(true);

    // Now grow "Matrix" itself — same scenario as the test above, but this
    // time the selection was already hand-touched and must survive as-is.
    act(() => { updateCollectionMembers('Matrix', [{ key: 'Al' }, { key: 'Si' }, { key: 'Ni' }, { key: 'WC' }]); });

    const alCheckbox = screen.getByText('Al (Al)').closest('label').querySelector('input');
    const siCheckbox = screen.getByText('Si (Si)').closest('label').querySelector('input');
    const niCheckbox = screen.getByText('Ni (Ni)').closest('label').querySelector('input');
    expect(alCheckbox.checked).toBe(true);
    expect(siCheckbox.checked).toBe(true);
    expect(niCheckbox.checked).toBe(true);
    screen.getByText(/Auto-Run all phases/);   // all 3 of 3 — the hand edit, not a reseed
  });
});

/**
 * CRITICAL finding C1. A collection whose members are all untestable here
 * (none has an .sht `phaseTestPhases` returns) narrows `selectedKeys` to the
 * empty set while `phases` stays full — `0 / 3`, "Run 0 selected". Before
 * the fix, clicking Run in that state omitted `phase_keys` from the request
 * (`subsetSelected` requires `size > 0`), and the backend
 * (`routes/indexing.py`: `if phase_keys:`) reads an absent list as "test
 * every phase in the library". These tests prove the dialog now refuses
 * instead, and — the part a button-label or a `sends: null` check cannot
 * prove — that `indexApi.phaseTestStart` is never even called, so no
 * request bearing that omission can reach the wire.
 */
describe('SinglePixelPhaseTestDialog refuses to run when the collection selects nothing (C1)', () => {
  const HARTMETALLE_ONLY = [
    // WC is a real collection member, but `phaseTestPhases` below (Al, Si,
    // Ni) never returns it — exactly the "6 of 36 library phases have no
    // .sht" case from the review report.
    { name: 'Hartmetalle', parent: null, members: [{ key: 'WC' }] },
  ];

  function setHartmetalleActive() {
    useCollectionStore.setState({
      data: {
        collections: HARTMETALLE_ONLY, unassigned: [], problems: [],
        state: { active: 'Hartmetalle', hidden: [] },
      },
    });
  }

  it('shows 0 selected, disables Run, and names the reason', async () => {
    setHartmetalleActive();
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    screen.getByText(/Phases \(0\/3\)/);
    const runButton = screen.getByText(/Run 0 selected/).closest('button');
    expect(runButton.disabled).toBe(true);
    screen.getByText('Select at least one phase.');
  });

  it('clicking Run sends nothing — phaseTestStart is never called with an omitted phase_keys', async () => {
    setHartmetalleActive();
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    const runButton = screen.getByText(/Run 0 selected/).closest('button');
    fireEvent.click(runButton);
    await act(async () => { await Promise.resolve(); });

    // The exact payload assertion the review asked for: not "the button
    // says 0", but that the call which would have silently run the whole
    // library (an object with no `phase_keys` key at all) never happens.
    expect(indexApi.phaseTestStart).not.toHaveBeenCalled();
    screen.getByText('Select at least one phase.');
  });

  it('the guard inside runAuto refuses independently of the disabled attribute', async () => {
    // Belt-and-braces: force the click through even if the `disabled`
    // attribute were ever removed by a future change, so this test still
    // catches a regression that only deletes the internal `if (noneSelected)`
    // guard in `runAuto` and leaves the button's `disabled` prop alone (or
    // vice versa) — either half regressing alone must still be caught.
    setHartmetalleActive();
    await openDialog();
    fireEvent.click(screen.getByText(/Phases \(/));

    const runButton = screen.getByText(/Run 0 selected/).closest('button');
    runButton.removeAttribute('disabled');
    fireEvent.click(runButton);
    await act(async () => { await Promise.resolve(); });

    expect(indexApi.phaseTestStart).not.toHaveBeenCalled();
  });
});

/**
 * The SAME trap (C1) through a DIFFERENT door. `noneSelected` only fires
 * once `phaseTestPhases` has resolved — it requires `phases.length > 0`.
 * While a collection is active and that fetch is still in flight, or has
 * failed outright, `phases` is `[]`: `noneSelected` is false AND
 * `allSelected` is true (its own `phases.length === 0` clause), so the
 * button reads "Auto-Run all phases" and, before this fix, a click reached
 * `phaseTestStart` with `phase_keys` omitted — the whole library, silently,
 * while the toolbar names a specific collection. These tests assert the
 * wire (`phaseTestStart` not called), the same way the C1 tests above do,
 * because the button's label branches on `allSelected`, a different
 * expression than the guard — asserting the label proves nothing here.
 */
describe('SinglePixelPhaseTestDialog refuses to run while the collection cannot be honoured yet (residual C1)', () => {
  it('a collection is active, the phases request is still in flight, click Run → no request', async () => {
    setActiveCollection('Matrix');
    let resolveFetch;
    indexApi.phaseTestPhases.mockImplementationOnce(
      () => new Promise((resolve) => { resolveFetch = resolve; }));

    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() => expect(indexApi.phaseTestPhases).toHaveBeenCalled());

    // The phase picker itself has nothing to show yet (phases is still
    // empty), so this is the button in its default, pre-collection state —
    // and it must already be disabled and say why.
    const runButton = screen.getByText(/Auto-Run all phases/).closest('button');
    expect(runButton.disabled).toBe(true);
    screen.getByText(/Loading the phase list/);

    fireEvent.click(runButton);
    await act(async () => { await Promise.resolve(); });
    expect(indexApi.phaseTestStart).not.toHaveBeenCalled();

    // Let the pending request resolve so it can't leak into another test.
    await act(async () => { resolveFetch({ data: [] }); await Promise.resolve(); });
  });

  it('a collection is active, the phases request rejected → no request', async () => {
    setActiveCollection('Matrix');
    indexApi.phaseTestPhases.mockImplementationOnce(() => Promise.reject(new Error('network down')));

    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() => expect(indexApi.phaseTestPhases).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });

    const runButton = screen.getByText(/Auto-Run all phases/).closest('button');
    expect(runButton.disabled).toBe(true);
    screen.getByText(/Could not load the phase list/);
    screen.getByText('Retry');   // a cheap retry is offered, not just a dead end

    fireEvent.click(runButton);
    await act(async () => { await Promise.resolve(); });
    expect(indexApi.phaseTestStart).not.toHaveBeenCalled();
  });

  it('no collection active, phases still empty → the run proceeds exactly as before', async () => {
    // No setActiveCollection() call: state.active stays unset, matching the
    // beforeEach default. An empty `phases` list here legitimately means
    // "the backend will test everything" — this path must be untouched.
    let resolveFetch;
    indexApi.phaseTestPhases.mockImplementationOnce(
      () => new Promise((resolve) => { resolveFetch = resolve; }));

    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() => expect(indexApi.phaseTestPhases).toHaveBeenCalled());

    const runButton = screen.getByText(/Auto-Run all phases/).closest('button');
    expect(runButton.disabled).toBe(false);

    fireEvent.click(runButton);
    await act(async () => { await Promise.resolve(); });
    expect(indexApi.phaseTestStart).toHaveBeenCalled();

    await act(async () => { resolveFetch({ data: [] }); await Promise.resolve(); });
  });
});
