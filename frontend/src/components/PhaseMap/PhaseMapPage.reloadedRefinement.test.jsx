// @vitest-environment jsdom
/**
 * The refinement panel and the Original/Refined switch for a result THIS
 * browser session did not index -- after a reload, after loading a saved
 * result, after indexing in another tab -- and the one precedence the page
 * uses for "which result is on screen".
 *
 * The page took "the original result" from the result store, which only knows
 * runs made in this session; the panel was disabled and the switch (driven by
 * a store entry only the panel writes, in memory, after a compute here) never
 * appeared, although `GET /api/indexing/results` lists the refined entry with
 * `is_refined: true, refinement_of: <original>`. Here the result store and the
 * refinement store are EMPTY unless a test says otherwise, and the fake backend
 * is the only source of truth.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';

const PARENT_ID = 'spherical_123';
const REFINED_ID = 'refinement_456';
const REFINED_ID_2 = 'refinement_789';

// The fake backend: an ordered registry (originals, each followed by its
// refined result if it has one) and exactly one active result.
let activeId;
let originals;
let refinedOf;
let sourceFileOf = {};
let loadedFile = null;
const registry = () => originals.flatMap((o) => (refinedOf[o] ? [o, refinedOf[o]] : [o]));
const entryOf = (id) => {
  const original = Object.keys(refinedOf).find((o) => refinedOf[o] === id) ?? null;
  return {
    id, method: 'spherical', n_indexed: 100, phases: [{ id: 1, name: 'Al' }],
    mean_ci: 0.4, is_active: activeId === id, is_refined: original !== null,
    refinement_of: original, source_file: sourceFileOf[id] ?? null, roi_bbox: null,
  };
};
const activateResult = vi.fn((id) => {
  activeId = id;
  return Promise.resolve({ data: { status: 'ok', active_id: id } });
});
const deactivateResult = vi.fn(() => {
  activeId = null;
  return Promise.resolve({ data: { status: 'ok' } });
});
const listResults = vi.fn(() => Promise.resolve({
  data: { active_id: activeId, results: registry().map(entryOf) },
}));
const compute = vi.fn(() => Promise.resolve({ job_id: 'job1' }));
const refinementSummary = vi.fn(() => Promise.reject(new Error('404')));
const refinementProgress = vi.fn(() => Promise.resolve({ state: 'running', progress: 0.5 }));
const diagnosticsSummary = vi.fn(() => Promise.reject(new Error('404')));
const diagnosticsCompute = vi.fn(() => Promise.resolve({ job_id: 'djob' }));
const citationsForResult = vi.fn(() => Promise.resolve({ data: { steps: [], bibtex: '' } }));

vi.mock('../../services/api', () => {
  const ok = (data) => Promise.resolve({ data });
  const anyApi = (overrides = {}) => new Proxy({ ...overrides }, {
    get: (target, prop) => {
      if (prop in target) return target[prop];
      if (typeof prop !== 'string') return undefined;
      target[prop] = vi.fn(() => ok({}));
      return target[prop];
    },
  });
  return {
    default: anyApi(),
    indexApi: anyApi({
      listResults: (...a) => listResults(...a),
      activateResult: (...a) => activateResult(...a),
      deactivateResult: (...a) => deactivateResult(...a),
    }),
    phaseMapApi: anyApi({ layer: vi.fn(() => ok({ image: '', scale: null })) }),
    analysisApi: anyApi(),
    ebsdApi: anyApi({ info: () => Promise.resolve({ data: { file_path: loadedFile } }) }),
    h5Api: anyApi(),
    pcApi: anyApi(),
    forwardDiagApi: anyApi({
      summary: (...a) => diagnosticsSummary(...a),
      compute: (...a) => diagnosticsCompute(...a),
    }),
    refinementApi: anyApi({
      summary: (...a) => refinementSummary(...a),
      compute: (...a) => compute(...a),
      progress: (...a) => refinementProgress(...a),
    }),
    citationsApi: anyApi({ forResult: (...a) => citationsForResult(...a) }),
  };
});

vi.mock('../../services/addonsApi', () => ({
  addonsApi: { mapImage: vi.fn(() => Promise.resolve({ data: { image: '', scale: null } })) },
}));

vi.mock('./LayeredCanvas', () => ({ default: () => <div data-testid="canvas" /> }));
vi.mock('../EDS/LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../EDS/MagnifierLens', () => ({ default: () => null }));
vi.mock('../PatternMatch/LinkedPatternImage', () => ({ default: () => null }));
vi.mock('../PoleFigure/openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }));

import useResultStore from '../../stores/useResultStore';
import useDataStore from '../../stores/useDataStore';
import PhaseMapPage from './PhaseMapPage';

beforeEach(() => {
  vi.clearAllMocks();
  activeId = PARENT_ID;
  originals = [PARENT_ID];
  refinedOf = { [PARENT_ID]: REFINED_ID };
  sourceFileOf = {};
  loadedFile = null;
  refinementSummary.mockImplementation(() => Promise.reject(new Error('404')));
  refinementProgress.mockImplementation(() => Promise.resolve({ state: 'running', progress: 0.5 }));
  diagnosticsSummary.mockImplementation(() => Promise.reject(new Error('404')));
  // A page that was just (re)loaded: nothing was indexed or refined here.
  useResultStore.setState({ indexingResult: null, indexingMethod: null, resultsList: [] });
  useDataStore.setState({ refinementComputed: {}, diagnosticsComputed: {}, refinementJob: null });
});

afterEach(() => {
  cleanup();
  useResultStore.setState({ indexingResult: null, indexingMethod: null, resultsList: [] });
  useDataStore.setState({ refinementComputed: {}, diagnosticsComputed: {}, refinementJob: null });
});

async function openAdvanced() {
  fireEvent.click(await screen.findByText(/Advanced tools/));
}

function startRefinementJobFor(resultId) {
  act(() => {
    useDataStore.getState().setRefinementJob({
      result_id: resultId, job_id: 'job1', state: 'running', progress: 0.9,
      current_stage: 'stage_3', outer_iter: 1,
    });
  });
}

describe('a result this session did not index', () => {
  it('offers the Original/Refined switch from the backend list and activates the refined result', async () => {
    render(<PhaseMapPage isActive />);

    const refinedBtn = await screen.findByRole('button', { name: 'Refined' });
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(true));
    activateResult.mockClear();

    fireEvent.click(refinedBtn);

    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(REFINED_ID));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(false);
  });

  it('offers no switch while the original has no refined result', async () => {
    refinedOf = {};
    render(<PhaseMapPage isActive />);

    await waitFor(() => expect(listResults).toHaveBeenCalled());
    await screen.findByRole('tab', { name: /^spherical/ });
    expect(screen.queryByRole('button', { name: 'Refined' })).toBeNull();
  });

  it('enables the refinement panel for the shown original', async () => {
    refinedOf = {};
    render(<PhaseMapPage isActive />);
    await openAdvanced();

    const refine = await screen.findByRole('button', { name: 'Refine R + PC' });
    await waitFor(() => expect(refine.disabled).toBe(false));
    fireEvent.click(refine);

    await waitFor(() => expect(compute).toHaveBeenCalled());
    expect(compute.mock.calls[0][0]).toBe(PARENT_ID);
  });

  it('in the refined view the panel and the switch still belong to the ORIGINAL', async () => {
    // Reloaded while looking at the refined result: the backend's active result
    // is the refined one, the original is its `refinement_of`.
    activeId = REFINED_ID;
    render(<PhaseMapPage isActive />);

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(false);
    await openAdvanced();
    const reRefine = await screen.findByRole('button', { name: 'Re-Refine' });
    await waitFor(() => expect(reRefine.disabled).toBe(false));
    fireEvent.click(reRefine);

    await waitFor(() => expect(compute).toHaveBeenCalled());
    expect(compute.mock.calls[0][0]).toBe(PARENT_ID);          // not the refined result's id

    // ... and the switch goes back to the original
    activateResult.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Original' }));
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(PARENT_ID));
  });

  it('asks whether the diagnostics exist on the ORIGINAL when the refined view is shown', async () => {
    // Reloaded in the refined view; only the original has diagnostics.
    activeId = REFINED_ID;
    diagnosticsSummary.mockImplementation((id) =>
      (id === PARENT_ID ? Promise.resolve({ computed: true }) : Promise.reject(new Error('404'))));
    render(<PhaseMapPage isActive />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));

    fireEvent.click(screen.getByText('+ Add Layer'));
    fireEvent.click(screen.getByLabelText(/Show unavailable/));

    const carries = (els, text) => els.some((el) => el.textContent.includes(text));
    await waitFor(() => {
      const forwardNcc = screen.getAllByText(/Forward NCC/);
      expect(carries(forwardNcc, 'Computed on the original result')).toBe(true);
    });
    expect(diagnosticsSummary).toHaveBeenCalledWith(PARENT_ID);
  });

  it('opens the refinement layers from the listed pair, without a store entry', async () => {
    // The refinement summary request fails here (the default mock), so only the
    // /results pair can tell the page that the original has been refined.
    render(<PhaseMapPage isActive />);
    await screen.findByRole('button', { name: 'Refined' });

    fireEvent.click(screen.getByText('+ Add Layer'));
    fireEvent.click(screen.getByLabelText(/Show unavailable/));

    expect(await screen.findAllByText(/Convergence/)).not.toHaveLength(0);
    expect(screen.queryAllByText(/Run 'Refine R \+ PC' first/)).toHaveLength(0);
  });

  it('holds the refinement layers back when there is no refined result (control)', async () => {
    refinedOf = {};
    render(<PhaseMapPage isActive />);
    await screen.findByRole('tab', { name: /^spherical/ });

    fireEvent.click(screen.getByText('+ Add Layer'));
    fireEvent.click(screen.getByLabelText(/Show unavailable/));

    await waitFor(() =>
      expect(screen.queryAllByText(/Run 'Refine R \+ PC' first/).length).toBeGreaterThan(0));
  });

  it('learns of a refined result made elsewhere the next time the page reads the list', async () => {
    // Another tab refines the shown result; clicking the gallery chip re-reads
    // the backend's list, which now has the pair.
    refinedOf = {};
    render(<PhaseMapPage isActive />);
    const chip = await screen.findByRole('tab', { name: /^spherical/ });
    expect(screen.queryByRole('button', { name: 'Refined' })).toBeNull();

    refinedOf = { [PARENT_ID]: REFINED_ID };
    fireEvent.click(chip);

    expect(await screen.findByRole('button', { name: 'Refined' })).toBeTruthy();
  });

  it('picks up a refined result that a compute in this session has just made', async () => {
    // First a plain original; the refinement then finishes (the panel records
    // it in the store) and the backend lists the new entry.
    refinedOf = {};
    useResultStore.setState({
      indexingResult: { result_id: PARENT_ID, phases: ['Al'], n_indexed: 100, mean_ci: 0.4 },
      indexingMethod: 'spherical',
    });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    expect(screen.queryByRole('button', { name: 'Refined' })).toBeNull();

    refinedOf = { [PARENT_ID]: REFINED_ID };
    useDataStore.getState().setRefinementComputed(PARENT_ID, {
      lambda: 0.01, computed_at: 'now', refined_result_id: REFINED_ID,
    });

    expect(await screen.findByRole('button', { name: 'Refined' })).toBeTruthy();
  });

  it('does not trust a store entry for a refined result the backend does not list', async () => {
    // The list has answered and has no pair: the store's remembered id (a
    // result that has since gone, say) does not make a switch.
    refinedOf = {};
    useResultStore.setState({
      indexingResult: { result_id: PARENT_ID, phases: ['Al'], n_indexed: 100, mean_ci: 0.4 },
      indexingMethod: 'spherical',
    });
    useDataStore.setState({ refinementComputed: {
      [PARENT_ID]: { lambda: 0.01, computed_at: 'then', refined_result_id: 'refinement_gone' },
    } });
    render(<PhaseMapPage isActive />);

    await screen.findByRole('tab', { name: /^spherical/ });
    await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(1));
    expect(screen.queryByRole('button', { name: 'Refined' })).toBeNull();
  });
});

describe('a refinement job that leaves "running"', () => {
  it('makes the page re-read the list, so a refined result appears even if the summary cannot be read', async () => {
    refinedOf = {};
    render(<PhaseMapPage isActive />);
    await openAdvanced();
    await screen.findByRole('button', { name: 'Refine R + PC' });
    expect(screen.queryByRole('button', { name: 'Refined' })).toBeNull();

    // The refinement finishes in the backend (the refined result is registered);
    // the panel sees "completed" and cannot read the summary.
    refinedOf = { [PARENT_ID]: REFINED_ID };
    refinementProgress.mockImplementation(() => Promise.resolve({ state: 'completed', progress: 1 }));
    startRefinementJobFor(PARENT_ID);

    expect(await screen.findByRole('button', { name: 'Refined' }, { timeout: 3000 })).toBeTruthy();
    expect(useDataStore.getState().refinementComputed[PARENT_ID]).toBeUndefined();
  });

  it('also does so when the job was cancelled after the refined result was registered', async () => {
    refinedOf = {};
    render(<PhaseMapPage isActive />);
    await openAdvanced();
    await screen.findByRole('button', { name: 'Refine R + PC' });

    refinedOf = { [PARENT_ID]: REFINED_ID };
    refinementProgress.mockImplementation(() => Promise.resolve({ state: 'cancelled' }));
    startRefinementJobFor(PARENT_ID);

    expect(await screen.findByRole('button', { name: 'Refined' }, { timeout: 3000 })).toBeTruthy();
  });

  it('follows a re-refine that replaces the refined result: the switch activates the NEW id', async () => {
    // Looking at the refined result when the new refinement finishes: the
    // backend replaces it under a new id and moves the active id along.
    activeId = REFINED_ID;
    render(<PhaseMapPage isActive />);
    await openAdvanced();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));

    refinementProgress.mockImplementation(() => {
      refinedOf = { [PARENT_ID]: REFINED_ID_2 };
      activeId = REFINED_ID_2;
      return Promise.resolve({ state: 'completed', progress: 1 });
    });
    const readsBeforeJob = listResults.mock.calls.length;
    startRefinementJobFor(PARENT_ID);

    // the job is over (the panel clears it) and the page has re-read the list
    await waitFor(() => expect(useDataStore.getState().refinementJob).toBeNull(),
      { timeout: 3000 });
    await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(readsBeforeJob));
    // still the refined view, now of the new result
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(false);
    // ... and going Original -> Refined lands on the new id, not the removed one
    fireEvent.click(screen.getByRole('button', { name: 'Original' }));
    await waitFor(() => expect(activateResult).toHaveBeenLastCalledWith(PARENT_ID));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Refined' }));
    await waitFor(() => expect(activateResult).toHaveBeenLastCalledWith(REFINED_ID_2));
  });
});

describe('one precedence for "which result is on screen"', () => {
  it('follows the backend when the result store holds an older run', async () => {
    // This tab indexed OLD; the backend now shows PARENT (brought on screen
    // elsewhere). Everything that names the shown result follows the backend.
    const OLD_ID = 'spherical_old';
    originals = [OLD_ID, PARENT_ID];
    refinedOf = { [PARENT_ID]: REFINED_ID };
    activeId = OLD_ID;
    useResultStore.setState({
      indexingResult: { result_id: OLD_ID, phases: ['Al'], n_indexed: 100, mean_ci: 0.4 },
      indexingMethod: 'spherical',
    });
    const view = render(<PhaseMapPage isActive={false} />);
    view.rerender(<PhaseMapPage isActive />);
    await waitFor(() => expect(citationsForResult).toHaveBeenCalledWith(OLD_ID));

    // another tab brings PARENT on screen; this page re-reads on its next visit
    activeId = PARENT_ID;
    citationsForResult.mockClear();
    view.rerender(<PhaseMapPage isActive={false} />);
    view.rerender(<PhaseMapPage isActive />);

    await waitFor(() => expect(citationsForResult).toHaveBeenCalledWith(PARENT_ID));
    expect(citationsForResult).not.toHaveBeenCalledWith(OLD_ID);
    expect(await screen.findByRole('button', { name: 'Refined' })).toBeTruthy();
    await openAdvanced();
    fireEvent.click(await screen.findByRole('button', { name: 'Re-Refine' }));
    await waitFor(() => expect(compute).toHaveBeenCalled());
    expect(compute.mock.calls[0][0]).toBe(PARENT_ID);
  });

  it('switching between two originals through the gallery follows the pair of each', async () => {
    const OTHER_ID = 'spherical_other';
    const OTHER_REFINED = 'refinement_other';
    originals = [PARENT_ID, OTHER_ID];
    refinedOf = { [PARENT_ID]: REFINED_ID, [OTHER_ID]: OTHER_REFINED };
    activeId = PARENT_ID;
    useResultStore.setState({ resultsList: [
      { id: 1, result_id: PARENT_ID, label: 'run A', phases: ['Al'], mean_ci: 0.4, n_indexed: 100 },
      { id: 2, result_id: OTHER_ID, label: 'run B', phases: ['Al'], mean_ci: 0.4, n_indexed: 100 },
    ] });
    render(<PhaseMapPage isActive />);

    const chipB = await screen.findByRole('tab', { name: 'run B' });
    await screen.findByRole('button', { name: 'Refined' });
    fireEvent.click(chipB);
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(OTHER_ID));
    await waitFor(() => expect(activeId).toBe(OTHER_ID));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(true));

    activateResult.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Refined' }));
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(OTHER_REFINED));

    // and back to A: its own pair
    activateResult.mockClear();
    fireEvent.click(screen.getByRole('tab', { name: 'run A' }));
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(PARENT_ID));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(true));
    activateResult.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Refined' }));
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(REFINED_ID));
  });

  it('after the switch the selected chip is the result that is shown', async () => {
    useResultStore.setState({
      indexingResult: { result_id: PARENT_ID, phases: ['Al'], n_indexed: 100, mean_ci: 0.4 },
      indexingMethod: 'spherical',
    });
    render(<PhaseMapPage isActive />);
    fireEvent.click(await screen.findByRole('button', { name: 'Refined' }));

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    await waitFor(() => {
      const selected = screen.getAllByRole('tab').filter(
        (el) => el.getAttribute('aria-selected') === 'true');
      expect(selected).toHaveLength(1);
      expect(selected[0].textContent).toMatch(/^Refined/);
    });
  });

  it('claims no result is shown when the backend has none active (it does not name the newest)', async () => {
    // The backend draws the ACTIVE result; with none active the renderer draws
    // the analysis dataset (if any), never "the newest stored run".
    useResultStore.setState({
      indexingResult: { result_id: PARENT_ID, phases: ['Al'], n_indexed: 100, mean_ci: 0.4 },
      indexingMethod: 'spherical',
    });
    refinedOf = {};
    render(<PhaseMapPage isActive />);
    await screen.findByRole('tab', { name: /^spherical/ });

    refinedOf = { [PARENT_ID]: REFINED_ID };
    activeId = null;                       // nothing active; the newest stored is REFINED_ID
    citationsForResult.mockClear();
    act(() => {
      useDataStore.getState().setRefinementComputed(PARENT_ID, {
        lambda: 0.01, computed_at: 'now', refined_result_id: REFINED_ID,
      });
    });

    // no result is shown, so there is no original/refined pair to switch
    // between (the view is not claimed to be the refined one either)
    await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(2));
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByRole('button', { name: 'Refined' })).toBeNull();
    // and nothing that names "the shown result" was pointed at the newest one
    expect(citationsForResult).not.toHaveBeenCalledWith(REFINED_ID);
  });
});

describe('a gallery chip that is not a stored result', () => {
  it('does not pull the newest stored run on screen when nothing is active', async () => {
    // A file entry (.ang/.ctf "Add file...", batch hand-off): no result_id; the
    // backend deactivates its results and the renderer draws the analysis
    // dataset. The registry still holds stored runs; none of them is shown.
    activeId = PARENT_ID;
    useResultStore.setState({
      indexingResult: { phases: ['Al'], n_indexed: 100, mean_ci: 0.4,
        source_path: '/scan/file.ang' },
      indexingMethod: 'Imported file',
    });
    render(<PhaseMapPage isActive={false} />);

    await waitFor(() => expect(deactivateResult).toHaveBeenCalled());
    await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(1));
    await new Promise((r) => setTimeout(r, 50));       // let the follow-up renders settle

    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(1);                       // the file chip, nothing else
    expect(screen.queryByRole('tab', { name: /spherical|Refined/ })).toBeNull();
    expect(citationsForResult).not.toHaveBeenCalled();  // no stored result is "the shown one"
  });
});

describe('a page visit with nothing active', () => {
  it('adopts the newest stored run (and gives it a chip) when that cannot switch the loaded file', async () => {
    activeId = null;
    sourceFileOf = { [REFINED_ID]: '/runs/a.h5oina', [PARENT_ID]: '/runs/a.h5oina' };
    loadedFile = '/runs/a.h5oina';
    render(<PhaseMapPage isActive />);

    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(REFINED_ID));
    expect(await screen.findByRole('tab', { name: /^Refined/ })).toBeTruthy();
  });

  it('adopts nothing and gives no chip when that would replace the loaded file', async () => {
    activeId = null;
    sourceFileOf = { [REFINED_ID]: '/runs/a.h5oina', [PARENT_ID]: '/runs/a.h5oina' };
    loadedFile = '/other/b.h5oina';            // the user loaded a different file
    render(<PhaseMapPage isActive />);

    await waitFor(() => expect(listResults).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 100));
    expect(activateResult).not.toHaveBeenCalled();
    expect(screen.queryAllByRole('tab')).toHaveLength(0);
  });
});

describe('the gallery after a refined result is replaced', () => {
  it('has one Refined chip, with the new id, and it is the selected one', async () => {
    render(<PhaseMapPage isActive />);
    await openAdvanced();
    fireEvent.click(await screen.findByRole('button', { name: 'Refined' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    await waitFor(() => expect(screen.getAllByRole('tab')).toHaveLength(2));   // original + refined

    // Re-Refine: the backend replaces the refined result under a new id and
    // keeps the refined view on screen.
    refinementProgress.mockImplementation(() => {
      refinedOf = { [PARENT_ID]: REFINED_ID_2 };
      activeId = REFINED_ID_2;
      return Promise.resolve({ state: 'completed', progress: 1 });
    });
    startRefinementJobFor(PARENT_ID);
    await waitFor(() => expect(useDataStore.getState().refinementJob).toBeNull(),
      { timeout: 3000 });

    await waitFor(() => {
      const refinedChips = screen.getAllByRole('tab').filter((el) => /^Refined/.test(el.textContent));
      expect(refinedChips).toHaveLength(1);
      expect(refinedChips[0].getAttribute('aria-selected')).toBe('true');
    });
    activateResult.mockClear();
    fireEvent.click(screen.getAllByRole('tab').find((el) => /^Refined/.test(el.textContent)));
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(REFINED_ID_2));
    expect(activateResult).not.toHaveBeenCalledWith(REFINED_ID);   // the dead id is never used
  });

  it('drops a Refined chip whose result is gone and selects what is shown', async () => {
    render(<PhaseMapPage isActive />);
    fireEvent.click(await screen.findByRole('button', { name: 'Refined' }));
    await waitFor(() => expect(screen.getAllByRole('tab')).toHaveLength(2));

    // the refined result disappears from the backend; the original is active again
    refinedOf = {};
    activeId = PARENT_ID;
    fireEvent.click(screen.getAllByRole('tab').find((el) => !/^Refined/.test(el.textContent)));

    await waitFor(() => expect(screen.getAllByRole('tab')).toHaveLength(1));
    const only = screen.getAllByRole('tab')[0];
    expect(only.textContent).not.toMatch(/^Refined/);
    expect(only.getAttribute('aria-selected')).toBe('true');
  });
});

describe('the forward-model diagnostics in the refined view', () => {
  it('are held back, and nothing is asked of the refined result', async () => {
    activeId = REFINED_ID;
    render(<PhaseMapPage isActive />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    await openAdvanced();

    const computeBtn = await screen.findByRole('button', { name: /Compute Diagnostics|Re-compute/ });
    expect(computeBtn.disabled).toBe(true);
    // never computed (the summary request fails): the reason says so, it does
    // not send the user to the original result to find them
    expect(computeBtn.getAttribute('title')).toMatch(/Run 'Compute Diagnostics' first/);
    expect(screen.getByRole('button', { name: /Open Anomaly Browser/ }).disabled).toBe(true);
    fireEvent.click(computeBtn);

    expect(diagnosticsCompute).not.toHaveBeenCalled();
    const asked = [...diagnosticsSummary.mock.calls, ...diagnosticsCompute.mock.calls].flat();
    expect(asked).not.toContain(REFINED_ID);
  });

  it('say "computed on the original" only when they are', async () => {
    activeId = REFINED_ID;
    diagnosticsSummary.mockImplementation((id) =>
      (id === PARENT_ID ? Promise.resolve({ computed: true }) : Promise.reject(new Error('404'))));
    render(<PhaseMapPage isActive />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    await openAdvanced();

    await waitFor(() => {
      const btn = screen.getByRole('button', { name: /Compute Diagnostics|Re-compute/ });
      expect(btn.getAttribute('title')).toMatch(/Computed on the original result/);
    });
  });

  it('are computed on the original while the original is shown', async () => {
    render(<PhaseMapPage isActive />);
    await openAdvanced();

    const computeBtn = await screen.findByRole('button', { name: /Compute Diagnostics/ });
    await waitFor(() => expect(computeBtn.disabled).toBe(false));
    fireEvent.click(computeBtn);

    await waitFor(() => expect(diagnosticsCompute).toHaveBeenCalled());
    expect(diagnosticsCompute.mock.calls[0][0]).toBe(PARENT_ID);
  });
});
