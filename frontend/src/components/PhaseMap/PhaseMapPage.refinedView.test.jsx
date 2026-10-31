// @vitest-environment jsdom
/**
 * The Original/Refined toggle switches the ACTIVE result -- measured through
 * the PAGE.
 *
 * A joint R+PC refinement stores a second result next to the one it refined
 * and leaves the original active. The toggle used to
 * flip a state variable nothing read, so "Refined" showed the original and
 * "Original" showed whatever the backend had active. The page is rendered
 * here with the real stores and the real hook; only the network is mocked, by
 * a small fake backend that really has one active result at a time.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';

const PARENT_ID = 'spherical_123';
const REFINED_ID = 'refinement_456';

// The fake backend: exactly one active result, switched by activateResult.
let activeId;
// The refined result's id as the backend currently has it (a re-smooth gives
// the replacement a new one).
let refinedId = REFINED_ID;
const activateResult = vi.fn((id) => {
  activeId = id;
  return Promise.resolve({ data: { status: 'ok', active_id: id } });
});
const listResults = vi.fn(() => Promise.resolve({
  data: {
    active_id: activeId,
    results: [
      { id: PARENT_ID, method: 'spherical', n_indexed: 100, phases: [{ id: 1, name: 'Al' }],
        mean_ci: 0.4, is_active: activeId === PARENT_ID, is_refined: false,
        refinement_of: null, source_file: null, roi_bbox: null },
      { id: refinedId, method: 'spherical', n_indexed: 100, phases: [{ id: 1, name: 'Al' }],
        mean_ci: 0.4, is_active: activeId === refinedId, is_refined: true,
        refinement_of: PARENT_ID, source_file: null, roi_bbox: null },
    ],
  },
}));

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
    }),
    phaseMapApi: anyApi({ layer: vi.fn(() => ok({ image: '', scale: null })) }),
    analysisApi: anyApi(),
    ebsdApi: anyApi(),
    h5Api: anyApi(),
    pcApi: anyApi(),
    forwardDiagApi: anyApi({ summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    refinementApi: anyApi({ summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    citationsApi: anyApi({ forResult: vi.fn(() => ok({ steps: [], bibtex: '' })) }),
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
  refinedId = REFINED_ID;
  // A run made in this session (so the page knows its id from the store) and
  // the refinement that was computed on it.
  useResultStore.setState({
    indexingResult: { result_id: PARENT_ID, phases: ['Al'], n_indexed: 100, mean_ci: 0.4 },
    indexingMethod: 'spherical',
  });
  useDataStore.setState({
    refinementComputed: {
      [PARENT_ID]: { lambda: 0.01, computed_at: 'now', refined_result_id: REFINED_ID },
    },
    // ... and forward-model diagnostics, so that both kinds of gate are open on it
    diagnosticsComputed: { [PARENT_ID]: { bandwidth: 128, computed_at: 'now' } },
  });
});

afterEach(() => {
  cleanup();
  useResultStore.setState({ indexingResult: null, indexingMethod: null });
  useDataStore.setState({ refinementComputed: {}, diagnosticsComputed: {} });
});

// The picker remembers the state of its "Show unavailable" box across opening
// and closing, so set it rather than toggle it.
function showUnavailable() {
  const box = screen.getByLabelText(/Show unavailable/);
  if (!box.checked) fireEvent.click(box);
}

describe('the gallery card of a refined result', () => {
  it('says that it is the refined one', async () => {
    // The refined result is the active one (the user switched to it, then came
    // back to the page); no session run is in the store, so the page adopts
    // what the backend reports.
    useResultStore.setState({ indexingResult: null, indexingMethod: null });
    useDataStore.setState({ refinementComputed: {} });
    activeId = REFINED_ID;

    render(<PhaseMapPage isActive />);

    const chip = await screen.findByRole('tab', { name: /^Refined — spherical/ });
    expect(chip).toBeTruthy();
  });

  it('does not label the original card that way', async () => {
    useResultStore.setState({ indexingResult: null, indexingMethod: null });
    useDataStore.setState({ refinementComputed: {} });
    activeId = PARENT_ID;

    render(<PhaseMapPage isActive />);

    const chip = await screen.findByRole('tab', { name: /^spherical — Al/ });
    expect(chip.textContent).not.toMatch(/Refined/);
  });
});

describe('the Original/Refined toggle', () => {
  it('activates the refined result on "Refined" and the original on "Original"', async () => {
    render(<PhaseMapPage isActive />);

    const refinedBtn = await screen.findByRole('button', { name: 'Refined' });
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    // The original is the one on screen after a refinement.
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(true));
    activateResult.mockClear();
    // ... and there the refinement layers are on offer (the store says the
    // refinement was computed on it): the held-back reason is not shown.
    fireEvent.click(screen.getByText('+ Add Layer'));
    showUnavailable();
    expect(screen.queryAllByText(/Computed on the original result/)).toHaveLength(0);
    fireEvent.click(screen.getByText('+ Add Layer'));

    fireEvent.click(refinedBtn);

    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(REFINED_ID));
    // ... and the toggle now shows the refined view (read off the backend's answer)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(false);

    // In the refined view the refinement layers are held back in the layer
    // picker, with the reason (they belong to the original result).
    fireEvent.click(screen.getByText('+ Add Layer'));
    showUnavailable();
    // One layer of each kind of gate: a refinement layer and a forward-model diagnostics one.
    const REASON = 'Computed on the original result';
    // (findAll: the same names also label the compare pickers.)
    const carriesReason = (els) => els.some((el) => el.textContent.includes(REASON));
    expect(carriesReason(await screen.findAllByText(/Convergence/))).toBe(true);
    expect(carriesReason(screen.getAllByText(/Forward NCC/))).toBe(true);

    activateResult.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Original' }));

    await waitFor(() => expect(activateResult).toHaveBeenCalledWith(PARENT_ID));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(true));
  });

  it('in the refined view only what IS computed on the original points there', async () => {
    // The forward-model diagnostics were never computed (the refinement was). Both kinds
    // of layer are held back in the refined view, but "computed on the original
    // result; switch to Original" is only true of the refinement layers.
    useDataStore.setState({ diagnosticsComputed: {} });
    render(<PhaseMapPage isActive />);
    fireEvent.click(await screen.findByRole('button', { name: 'Refined' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));

    fireEvent.click(screen.getByText('+ Add Layer'));
    showUnavailable();

    const carries = (els, text) => els.some((el) => el.textContent.includes(text));
    const convergence = await screen.findAllByText(/Convergence/);
    const forwardNcc = screen.getAllByText(/Forward NCC/);
    expect(carries(convergence, 'Computed on the original result')).toBe(true);
    expect(carries(forwardNcc, "Run 'Compute Diagnostics' first")).toBe(true);
    expect(carries(forwardNcc, 'Computed on the original result')).toBe(false);

    // A held-back layer stays addable ("add it, then compute"). Its row gets
    // the same reasons: the diagnostic nobody computed says to compute it ...
    fireEvent.click(forwardNcc.find((el) => el.textContent.includes('Run \'Compute')));
    expect(await screen.findByTitle("Forward NCC — Run 'Compute Diagnostics' first")).toBeTruthy();
    // ... and a refinement layer, computed on the original, points there.
    fireEvent.click(screen.getByText('+ Add Layer'));
    fireEvent.click((await screen.findAllByText(/Convergence/))
      .find((el) => el.textContent.includes('Computed on the original result')));
    expect(await screen.findByTitle(
      /Convergence.* — Computed on the original result; switch the view to Original/)).toBeTruthy();
  });

  it('stays on the refined view when a re-smooth replaces the refined result', async () => {
    // The backend gives the replacement a new id, removes the old result and
    // keeps the user on a refined one; the page has to follow, or it would
    // read "Original" while a refined result is on screen.
    render(<PhaseMapPage isActive />);
    fireEvent.click(await screen.findByRole('button', { name: 'Refined' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));

    refinedId = 'refinement_789';
    activeId = refinedId;
    // (count the reads made AFTER the action: a count taken from zero is
    // already satisfied by the page's own reads on the way here)
    const readsBefore = listResults.mock.calls.length;
    act(() => {
      useDataStore.getState().setRefinementComputed(PARENT_ID, {
        lambda: 0.1, computed_at: 'later', refined_result_id: refinedId,
      });
    });

    await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(readsBefore));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(true));
    expect(screen.getByRole('button', { name: 'Original' }).disabled).toBe(false);
    // and the switch now leads to the NEW result: Original, then Refined
    fireEvent.click(screen.getByRole('button', { name: 'Original' }));
    await waitFor(() => expect(activateResult).toHaveBeenLastCalledWith(PARENT_ID));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refined' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Refined' }));
    await waitFor(() => expect(activateResult).toHaveBeenLastCalledWith('refinement_789'));
  });
});
