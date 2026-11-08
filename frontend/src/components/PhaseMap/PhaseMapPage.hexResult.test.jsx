// @vitest-environment jsdom
/**
 * What the Phase Map page does with the geometry of the result on screen -- measured through the
 * PAGE, with the real store, hook and reducer (only the network and the canvas leaves are mocked,
 * as in PhaseMapPage.addonLayer.test.jsx).
 *
 * The backend computes the cleanup (min cluster, modal smoothing, fill unindexed), the
 * grain-stabilised IPF, the grain boundaries and the local anomaly on the six neighbours of a
 * hexagonal scan, so a hexagonal result takes the same controls and sends the same requests as a
 * square one: what the user set is what goes out, in every layer request, the single-image
 * preview, the export and the apply. Only the Analysis page's KAM and GOS pair is held back on a
 * hexagonal result (the picker's `squareGrid` gate); the backend refuses every analysis map there.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';

const listResults = vi.fn();
const applyCleanup = vi.fn();

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
  const layer = () => ok({ image: '', scale: null });
  return {
    default: anyApi(),
    indexApi: anyApi({ listResults: (...a) => listResults(...a) }),
    phaseMapApi: anyApi({
      layer: vi.fn(layer),
      applyCleanup: (...a) => applyCleanup(...a),
    }),
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
// The drawer is not told the geometry any more: it behaves the same for every result.
vi.mock('./AnomalyBrowser/AnomalyBrowserDrawer', () => ({ default: () => <div data-testid="anomaly-drawer" /> }));

import { phaseMapApi, analysisApi } from '../../services/api';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';
import useDataStore from '../../stores/useDataStore';
import useResultStore from '../../stores/useResultStore';
import PhaseMapPage from './PhaseMapPage';

const RESULT_ID = 'spherical_123';
const SQUARE = { kind: 'square', n_rows: 10, n_cols: 10, step_x: 0.5, step_y: 0.5 };
const HEX = { kind: 'hex', n_rows: 10, n_cols: 10, step_x: 0.5, step_y: 0.5, odd_row_shift: 0.5 };
const entry = (scan_geometry) => ({
  id: RESULT_ID, result_id: RESULT_ID, label: 'spherical - Al', is_active: true,
  source_file: 'SampleB.h5oina', phases: ['Al'], shape: [10, 10],
  n_pixels: 100, ci_mean: 0.3, ...(scan_geometry ? { scan_geometry } : {}),
});
const showing = (scan_geometry) => listResults.mockResolvedValue({
  data: { results: [entry(scan_geometry)] } });

const GATE_REASON = 'Not available for a hexagonal scan yet';   // locales/en/phasemap.json layers.gateHexGrid

const layerCalls = () => phaseMapApi.layer.mock.calls;
const lastParamsOf = (kind) => {
  const mine = layerCalls().filter(([k]) => k === kind);
  return mine.length ? mine[mine.length - 1][1] : undefined;
};

// the only range input with a maximum of 50 is the min-cluster slider
const minCluster = () => document.querySelector('input[type="range"][max="50"]');
const modalRadios = () => screen.getAllByRole('radio', { name: /^(Off|3×3|5×5)$/ });
const fillBox = () => screen.getByLabelText(/Fill unindexed pixels/);
const applyBtn = () => screen.getByRole('button', { name: /Apply cleanup/ });
const quick = (label) => screen.getAllByRole('button').find(
  (b) => b.textContent === label && /quick|Show|Add|Replace|Stack/i.test(b.title || ''))
  ?? screen.getAllByRole('button').find((b) => b.textContent === label);
// Let an in-flight layer fetch finish: the mocked image is empty, so the fetch ends in an error,
// and a flush that arrives while it is still in flight is (rightly) not retried until it ends.
const settle = () => new Promise((r) => setTimeout(r, 60));
const chip = () => screen.getByRole('tab', { name: /Indexing/ });

const mount = async () => {
  const utils = render(<PhaseMapPage isActive />);
  await waitFor(() => expect(listResults).toHaveBeenCalled());
  await waitFor(() => expect(layerCalls().some(([k]) => k === 'phase')).toBe(true));
  return utils;
};
/** Make the page read /results again (a click on the chip does that) and wait for it. */
const reread = async () => {
  const before = listResults.mock.calls.length;
  fireEvent.click(chip());
  await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(before));
};

beforeEach(() => {
  vi.clearAllMocks();
  showing(SQUARE);
  useAddonLayerRequests.setState({ requests: [], superseded: [], citationTick: 0 });
});
afterEach(cleanup);

describe('the cleanup controls and the layer requests on a hexagonal result', () => {
  it('a square result behaves as before: controls live, the slider value is sent', async () => {
    await mount();
    expect(minCluster().disabled).toBe(false);
    expect(modalRadios().every((b) => !b.disabled)).toBe(true);
    expect(fillBox().disabled).toBe(false);
    fireEvent.change(minCluster(), { target: { value: '7' } });
    await waitFor(() => expect(lastParamsOf('phase').min_cluster_size).toBe(7));
    fireEvent.click(modalRadios()[1]);
    await waitFor(() => expect(lastParamsOf('phase').modal_filter_size).toBe(3));
    fireEvent.click(fillBox());
    await waitFor(() => expect(lastParamsOf('phase').fill_unindexed).toBe(true));
    expect(applyBtn().disabled).toBe(false);
  });

  it('an unknown geometry (no scan_geometry in /results) behaves exactly like a square one', async () => {
    showing(null);
    await mount();
    fireEvent.change(minCluster(), { target: { value: '7' } });
    await waitFor(() => expect(lastParamsOf('phase').min_cluster_size).toBe(7));
    expect(minCluster().disabled).toBe(false);
    expect(applyBtn().disabled).toBe(false);
  });

  it('a hexagonal result has the same live controls, and what the user sets is sent', async () => {
    showing(HEX);
    await mount();
    expect(minCluster().disabled).toBe(false);
    expect(minCluster().title).not.toBe(GATE_REASON);
    expect(modalRadios().every((b) => !b.disabled)).toBe(true);
    expect(fillBox().disabled).toBe(false);
    fireEvent.change(minCluster(), { target: { value: '7' } });
    await waitFor(() => expect(lastParamsOf('phase').min_cluster_size).toBe(7));
    fireEvent.click(modalRadios()[2]);
    await waitFor(() => expect(lastParamsOf('phase').modal_filter_size).toBe(5));
    fireEvent.click(fillBox());
    await waitFor(() => expect(lastParamsOf('phase').fill_unindexed).toBe(true));
    expect(applyBtn().disabled).toBe(false);
    expect(applyBtn().title).not.toBe(GATE_REASON);
  });

  it('keeps what the user set when the page moves from a square to a hexagonal result and back', async () => {
    await mount();
    fireEvent.change(minCluster(), { target: { value: '7' } });
    await waitFor(() => expect(lastParamsOf('phase').min_cluster_size).toBe(7));
    showing(HEX);
    await reread();
    await waitFor(() => expect(lastParamsOf('phase').min_cluster_size).toBe(7));
    expect(minCluster().value).toBe('7');
    expect(minCluster().disabled).toBe(false);
    showing(SQUARE);
    await reread();
    expect(minCluster().value).toBe('7');
  });

  it('sends the cleanup with every layer kind of a hexagonal result, not only the phase map', async () => {
    showing(HEX);
    await mount();
    fireEvent.change(minCluster(), { target: { value: '7' } });
    await waitFor(() => expect(lastParamsOf('phase').min_cluster_size).toBe(7));
    for (const [label, kind] of [['IPF-Z', 'ipf-z'], ['BC', 'bc'], ['CI', 'ci'], ['Phase', 'phase']]) {
      const before = layerCalls().length;
      fireEvent.click(quick(label));
      await waitFor(() => expect(layerCalls().slice(before).some(([k]) => k === kind)).toBe(true));
      for (const [, params] of layerCalls().slice(before).filter(([k]) => k === kind)) {
        expect(params.min_cluster_size).toBe(7);
      }
      await settle();
    }
  });

  it('the grain-stabilised IPF box is live on a hexagonal result and its flag is sent', async () => {
    showing(HEX);
    await mount();
    fireEvent.click(quick('IPF-Z'));
    await waitFor(() => expect(lastParamsOf('ipf-z')).toBeTruthy());
    await settle();    // the first fetch has failed on the mock's empty image; a flush may now clear that
    const box = await screen.findByLabelText(/Stabilize colors per grain/);
    expect(box.disabled).toBe(false);
    expect(box.closest('label').title).not.toBe(GATE_REASON);
    fireEvent.click(box);
    await waitFor(() => expect(lastParamsOf('ipf-z')?.grain_stabilized).toBe(true));
  });
});

describe('the preview, the export and the apply of a hexagonal result', () => {
  afterEach(() => {
    phaseMapApi.render.mockImplementation(() => Promise.resolve({ data: {} }));
    delete window.electronAPI;
  });

  it('send the cleanup the user set, as they do for a square result', async () => {
    phaseMapApi.render.mockImplementation(() => Promise.resolve({ data: { image: 'IMG' } }));
    window.electronAPI = { saveFile: vi.fn(() => Promise.resolve('out.png')) };
    applyCleanup.mockResolvedValue({ data: { target: 'checkpoint', moved: 1, after_unindexed: 0 } });
    showing(HEX);
    await mount();
    fireEvent.change(minCluster(), { target: { value: '7' } });
    fireEvent.click(modalRadios()[1]);
    fireEvent.click(fillBox());

    // the single-image preview
    await waitFor(() => expect(phaseMapApi.render.mock.calls.some(([p]) => (
      p.min_cluster_size === 7 && p.modal_filter_size === 3 && p.fill_unindexed === true))).toBe(true));

    // the export of the picture
    await waitFor(() => expect(screen.getByRole('button', { name: /^PNG/ }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: /^PNG/ }));
    await waitFor(() => expect(phaseMapApi.export).toHaveBeenCalled());
    const exported = phaseMapApi.export.mock.calls[0][3];
    expect([exported.min_cluster_size, exported.modal_filter_size, exported.fill_unindexed]).toEqual([7, 3, true]);

    // the apply
    await waitFor(() => expect(applyBtn().disabled).toBe(false));
    fireEvent.click(applyBtn());
    await waitFor(() => expect(applyCleanup).toHaveBeenCalled());
    const applied = applyCleanup.mock.calls[0][0];
    expect([applied.minClusterSize, applied.modalFilterSize, applied.fillUnindexed]).toEqual([7, 3, true]);
  });
});

// Used by the two describes that look at the picker's entries for KAM and GOS: those come from the
// Analysis page's dataset, so the page has to be told one is loaded, or the picker would name that
// as the reason instead of the geometry.
const withAnalysisLoaded = () => {
  beforeEach(() => {
    analysisApi.status.mockImplementation(() => Promise.resolve({ data: { loaded: true } }));
  });
  afterEach(() => {
    analysisApi.status.mockImplementation(() => Promise.resolve({ data: {} }));
  });
};
const boundaries = () => screen.queryByRole('button', { name: /Grain Boundaries/ });
const kam = () => screen.queryByRole('button', { name: /KAM/ });
const gos = () => screen.queryByRole('button', { name: /GOS/ });
const openPickerWithUnavailable = async (entryOf = boundaries) => {
  fireEvent.click(screen.getByRole('button', { name: /\+ Add Layer/ }));
  fireEvent.click(await screen.findByLabelText(/Show unavailable/));
  await waitFor(() => expect(entryOf()).toBeTruthy());
};

describe('the layer picker on a hexagonal result', () => {
  withAnalysisLoaded();

  it('offers the grain boundaries for a hexagonal result as for a square one', async () => {
    showing(HEX);
    await mount();
    await openPickerWithUnavailable();
    expect(boundaries().title).not.toBe(GATE_REASON);
    showing(SQUARE);
    await reread();
    await waitFor(() => expect(boundaries().title).not.toBe(GATE_REASON));
  });

  it('holds KAM and GOS back for a hexagonal result and offers them again for a square one', async () => {
    showing(HEX);
    await mount();
    await openPickerWithUnavailable(kam);
    expect(kam().title).toBe(GATE_REASON);
    expect(gos().title).toBe(GATE_REASON);

    // The same result id, now square: nothing but the geometry changes, so a picker whose list was
    // computed from stale inputs would still hold the layers back.
    showing(SQUARE);
    await reread();
    await waitFor(() => expect(kam().title).not.toBe(GATE_REASON));
    expect(gos().title).not.toBe(GATE_REASON);

    showing(HEX);
    await reread();
    await waitFor(() => expect(kam().title).toBe(GATE_REASON));
  });

  it('does not hold anything back when the backend does not know the geometry', async () => {
    showing(null);
    await mount();
    await openPickerWithUnavailable(kam);
    expect(kam().title).not.toBe(GATE_REASON);
    expect(gos().title).not.toBe(GATE_REASON);
  });

  it('forgets the geometry together with the result when nothing is stored any more', async () => {
    showing(HEX);
    const { rerender } = await mount();
    await openPickerWithUnavailable(kam);
    expect(kam().title).toBe(GATE_REASON);

    listResults.mockResolvedValue({ data: { results: [] } });
    rerender(<PhaseMapPage isActive={false} />);
    rerender(<PhaseMapPage isActive />);
    await waitFor(() => expect(kam().title).not.toBe(GATE_REASON));
  });

  it('forgets a hexagonal geometry when the results cannot be read again, rather than guessing', async () => {
    showing(HEX);
    await mount();
    await openPickerWithUnavailable(kam);
    expect(kam().title).toBe(GATE_REASON);

    listResults.mockRejectedValue(new Error('backend gone'));
    fireEvent.click(chip());
    await waitFor(() => expect(kam().title).not.toBe(GATE_REASON));
  });

  afterEach(() => {
    phaseMapApi.availableMaps.mockImplementation(() => Promise.resolve({ data: {} }));   // the mock's default
  });

  // With no result on screen the routes draw the Analysis page's dataset, and the backend refuses KAM
  // and GOS for a hexagonal one: the picker follows the geometry that is drawn, not only the result's.
  it('holds KAM and GOS back for a hexagonal analysis dataset when no result is on screen', async () => {
    listResults.mockResolvedValue({ data: { results: [] } });
    phaseMapApi.availableMaps.mockResolvedValue({ data: { maps: [], dataset_scan_geometry: HEX } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(phaseMapApi.availableMaps).toHaveBeenCalled());
    await openPickerWithUnavailable(kam);
    await waitFor(() => expect(kam().title).toBe(GATE_REASON));
    expect(gos().title).toBe(GATE_REASON);
  });

  it('offers KAM and GOS for a square analysis dataset when no result is on screen', async () => {
    listResults.mockResolvedValue({ data: { results: [] } });
    phaseMapApi.availableMaps.mockResolvedValue({ data: { maps: [], dataset_scan_geometry: SQUARE } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(phaseMapApi.availableMaps).toHaveBeenCalled());
    await openPickerWithUnavailable(kam);
    expect(kam().title).not.toBe(GATE_REASON);
    expect(gos().title).not.toBe(GATE_REASON);
  });
});

describe('an answer to a superseded preview request', () => {
  // The request for the single-image preview (/render) is sent from several places at once; when
  // their answers arrive out of order, only the newest one may decide whether the map or an error
  // is on screen.
  const deferred = () => {
    let resolve; let reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
  };
  const failure = (status, code, detail) => Object.assign(new Error(`status ${status}`), {
    response: { status, headers: code ? { 'x-orienta-code': code } : {}, data: { detail } },
  });
  const REFUSAL_TEXT = /compares each point with its four neighbours/;   // locales/en/common.json
  const renderCalls = () => phaseMapApi.render.mock.calls;

  afterEach(() => {
    phaseMapApi.render.mockImplementation(() => Promise.resolve({ data: {} }));
    analysisApi.load.mockImplementation(() => Promise.resolve({ data: {} }));
    try { sessionStorage.removeItem('batch_handoff_path'); } catch { /* ignore */ }
  });

  it('cannot leave the map hidden behind an error when it arrives after the newer request succeeded', async () => {
    // Two cleanup values are asked for in turn: the first request (7) is held back, the second (8)
    // succeeds. The first one then fails, late; only the newest answer decides what is on screen.
    const stale = deferred();
    let armed = false;
    phaseMapApi.render.mockImplementation((p) => (
      armed && p.min_cluster_size === 7
        ? stale.promise
        : Promise.resolve({ data: { image: 'IMG' } })));
    await mount();
    await waitFor(() => expect(screen.getByTestId('canvas')).toBeTruthy());

    armed = true;
    fireEvent.change(minCluster(), { target: { value: '7' } });
    await waitFor(() => expect(renderCalls().some(([p]) => p.min_cluster_size === 7)).toBe(true));
    fireEvent.change(minCluster(), { target: { value: '8' } });
    await waitFor(() => expect(renderCalls().some(([p]) => p.min_cluster_size === 8)).toBe(true));
    await settle();
    expect(screen.queryByText(REFUSAL_TEXT)).toBeNull();

    // only now does the answer to the first one arrive
    stale.reject(failure(400, 'hex_neighbourhood_unsupported', 'ENGLISH PROSE FROM THE BACKEND'));
    await settle();
    expect(screen.queryByText(REFUSAL_TEXT)).toBeNull();
    expect(screen.queryByText(/ENGLISH PROSE/)).toBeNull();
    expect(screen.getByTestId('canvas')).toBeTruthy();
  });

  it('cannot leave an error over a preview that then succeeded', async () => {
    // Another path (the batch hand-off) reports a failure while the preview request is in flight;
    // the preview then succeeds, so nothing may stay on top of the map.
    sessionStorage.setItem('batch_handoff_path', 'batch_result.ang');
    const handoff = deferred();
    const preview = deferred();
    analysisApi.load.mockImplementation(() => handoff.promise);
    phaseMapApi.render.mockImplementation(() => preview.promise);
    await mount();
    await waitFor(() => expect(renderCalls().length).toBeGreaterThan(0));
    await waitFor(() => expect(analysisApi.load).toHaveBeenCalled());

    handoff.reject(failure(500, null, 'handoff went wrong'));
    await settle();
    preview.resolve({ data: { image: 'IMG' } });
    await settle();
    expect(screen.queryByText(/handoff went wrong/)).toBeNull();
    expect(screen.getByTestId('canvas')).toBeTruthy();
  });

  it('still shows the words of a preview request that fails and has not been replaced', async () => {
    phaseMapApi.render.mockImplementation(() => Promise.reject(
      failure(500, null, 'disk is full')));
    await mount();
    expect(await screen.findByText('disk is full')).toBeTruthy();
    expect(screen.queryByTestId('canvas')).toBeNull();
  });
});

describe('Apply cleanup', () => {
  const refused = () => Object.assign(new Error('Request failed with status code 400'), {
    response: {
      status: 400,
      headers: { 'x-orienta-code': 'hex_neighbourhood_unsupported' },
      data: { detail: 'ENGLISH PROSE FROM THE BACKEND' },
    },
  });

  it('shows a coded refusal in the language of the page, on a square result too', async () => {
    applyCleanup.mockRejectedValue(refused());
    await mount();
    fireEvent.change(minCluster(), { target: { value: '3' } });
    await waitFor(() => expect(applyBtn().disabled).toBe(false));
    fireEvent.click(applyBtn());
    expect(await screen.findByText(/compares each point with its four neighbours/)).toBeTruthy();
    expect(screen.queryByText(/ENGLISH PROSE/)).toBeNull();
  });

  it('still shows the words of an uncoded failure', async () => {
    applyCleanup.mockRejectedValue(Object.assign(new Error('boom'), {
      response: { status: 500, headers: {}, data: { detail: 'disk is full' } } }));
    await mount();
    fireEvent.change(minCluster(), { target: { value: '3' } });
    await waitFor(() => expect(applyBtn().disabled).toBe(false));
    fireEvent.click(applyBtn());
    expect(await screen.findByText('disk is full')).toBeTruthy();
  });
});

describe('the geometry follows the result when the page re-reads the list for another reason', () => {
  // Three effects re-read /results and used to take only the id of the shown result. After a run on
  // another grid the geometry (the picker's gate for KAM and GOS) stayed the previous result's.
  // The only thing left that reads the geometry is that gate, so it is what these tests look at.
  withAnalysisLoaded();

  const flip = async (change) => {
    await mount();
    await openPickerWithUnavailable(kam);
    await waitFor(() => expect(kam().title).not.toBe(GATE_REASON));   // square
    showing(HEX);                                   // the next run was on a hexagonal grid
    await act(async () => { change(); });
    await waitFor(() => expect(kam().title).toBe(GATE_REASON));
  };

  it('after a result of a new run is announced to the result store', async () => {
    await flip(() => useResultStore.setState({ indexingResult: { result_id: 'new_run_1' } }));
  });

  it('after a refined result appears for the shown one', async () => {
    await flip(() => useDataStore.setState({
      refinementComputed: { [RESULT_ID]: { refined_result_id: 'refined_9' } } }));
  });

  it('after a refinement job leaves "running"', async () => {
    await mount();
    await openPickerWithUnavailable(kam);
    await waitFor(() => expect(kam().title).not.toBe(GATE_REASON));
    await act(async () => { useDataStore.setState({ refinementJob: { state: 'running' } }); });
    showing(HEX);
    await act(async () => { useDataStore.setState({ refinementJob: { state: 'done' } }); });
    await waitFor(() => expect(kam().title).toBe(GATE_REASON));
  });
});
