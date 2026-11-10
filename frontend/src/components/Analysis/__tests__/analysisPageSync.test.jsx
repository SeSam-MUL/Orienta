// @vitest-environment jsdom
//
// The page follows the backend. AnalysisPage asks GET /status whenever it becomes active; the answer carries
// what the backend holds (grains, the statistics settings, the reconstruction parameters, the phases of the
// map). Two ways the page and the backend drift apart are tested end to end: a reconstruction that finished
// while the Grain tab was away (a tab that is left stops polling), and a map loaded or switched on another
// page. Plus: a map loaded on the Data tab refetches the layers that need no grains, "Run Complete Analysis"
// waits for the Grain tab's own reconstruction and sends the phase.
// Real components, only the HTTP layer mocked, with the shapes of the real answers.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  // the Grain tab's citation panel asks for the grains' citations; left pending, it shows its loading line
  citationsApi: { forAnalysis: vi.fn(() => new Promise(() => {})) },
  analysisApi: {
    status: vi.fn(),
    load: vi.fn(),
    reconstructGrains: vi.fn(),
    getTaskStatus: vi.fn(),
    getMap: vi.fn(),
    grainSize: vi.fn(),
    batch: vi.fn(),
    qualityStats: vi.fn(),
    deformation: vi.fn(),
    texture: vi.fn(),
    rxClassify: vi.fn(),
    exportExcel: vi.fn(),
  },
  indexApi: { getLastResult: vi.fn() },
  edsApi: {},
}));

import { analysisApi, indexApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import AnalysisPage from '../AnalysisPage';

const AL_SI = [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }];

/** GET /api/analysis/status, as routes/analysis.py writes it. */
const status = (extra = {}) => ({
  data: {
    loaded: true, shape: [40, 60], has_grains: false, phases: ['Aluminium (Fm-3m)', 'Si (Fd-3m)'],
    phase_options: AL_SI, step_size: 0.5,
    edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, recon_params: null,
    dataset_token: 'a1', reconstruct: null,
    ...extra,
  },
});
const grainStatus = (extra = {}) => status({
  has_grains: true, edge_rule: 'outer_row', ecd_basis: 'measured_pixels', stats_phase: 2,
  recon_params: { threshold: 6.5, min_pixels: 5, min_intercept: 2, alpha: 1.5, fill: false },
  ...extra,
});

function setUpBackend() {
  indexApi.getLastResult.mockRejectedValue(new Error('none'));
  analysisApi.qualityStats.mockRejectedValue(new Error('none'));
  analysisApi.status.mockResolvedValue(status());
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
  analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.2 } });
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  useResultStore.getState().clearResults();
  try { sessionStorage.clear(); } catch { /* ignore */ }
});

const mapCalls = (type) => analysisApi.getMap.mock.calls.filter((c) => c[0] === type);
const state = () => useResultStore.getState();
const page = (isActive) => <AnalysisPage isActive={isActive} onNavigate={() => {}} />;

/** The page is left (another page shown) and entered again: the status effect runs once more. */
function leaveAndReturn(view) {
  view.rerender(page(false));
  view.rerender(page(true));
}

async function openPage() {
  const view = render(page(true));
  await waitFor(() => expect(state().analysisLoaded).toBe(true));
  return view;
}

const RUNNING = { task_id: 't1', status: 'running', progress: 0.4 };
const doneTask = {
  data: {
    status: 'completed', progress: 1, grain_count: 98,
    result: {
      n_grains: 98, grain_count: 98, mean_ecd: 12.34, median_ecd: 11, std_ecd: 3.2, mean_aspect_ratio: 1.6,
      edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, stats_phase_name: null,
      n_grains_total: 98, n_edge_grains: 34, n_grains_used: 64, edge_safeguard_applied: false,
    },
  },
};
// (by text: getByRole computes the accessible name of a disabled Button, and jsdom cannot clone its style)
const buttonByText = (re) => Array.from(document.querySelectorAll('button')).find((b) => re.test(b.textContent));
const reconstructButton = () => buttonByText(/^Reconstruct Grains|^Reconstructing/);
const runButton = () => buttonByText(/^Run Complete Analysis/);

describe('AnalysisPage: a reconstruction and a switch of tabs (the page is not left)', () => {
  it('the Grain tab picks the running reconstruction up again: busy, polled, and its result lands', async () => {
    setUpBackend();
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));

    // the tab is left; the backend still runs the task
    analysisApi.status.mockResolvedValue(status({ reconstruct: RUNNING }));
    fireEvent.click(screen.getByText('Deformation'));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(3));
    expect(state().grainReconBusy).toBe(true);
    expect(runButton().disabled).toBe(true);

    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(reconstructButton().disabled).toBe(true);
    expect(runButton().disabled).toBe(true);
    analysisApi.getTaskStatus.mockClear();
    analysisApi.getTaskStatus.mockResolvedValue(doneTask);
    await waitFor(() => expect(analysisApi.getTaskStatus).toHaveBeenCalledWith('t1'), { timeout: 4000 });
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy(), { timeout: 4000 });
    expect(state()).toMatchObject({ hasGrains: true, reconTaskId: null, grainReconBusy: false });
    expect(reconstructButton().disabled).toBe(false);
    expect(runButton().disabled).toBe(false);
  }, 20000);

  it('one that finished while the tab was away is found by the next tab click: grains, settings, free buttons', async () => {
    setUpBackend();
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));
    fireEvent.click(screen.getByText('Deformation'));
    await waitFor(() => expect(state().grainReconBusy).toBe(true));      // the task runs on: the flag stays

    // the backend finished meanwhile
    analysisApi.status.mockResolvedValue(grainStatus());
    analysisApi.grainSize.mockResolvedValue({
      data: {
        success: true, edge_rule: 'outer_row', ecd_basis: 'measured_pixels', stats_phase: 2, stats_phase_name: 'Si',
        n_grains_total: 12, n_edge_grains: 5, n_grains_used: 7, edge_safeguard_applied: false,
        mean_ecd: 3.5, median_ecd: 3.1, std_ecd: 0.9, mean_aspect_ratio: 1.4,
      },
    });
    const versionBefore = state().grainsVersion;
    fireEvent.click(screen.getByText('Texture'));
    await waitFor(() => expect(state().hasGrains).toBe(true));
    expect(state()).toMatchObject({ reconTaskId: null, grainReconBusy: false, grainsVersion: versionBefore + 1 });
    expect(state().grainStatsSettings).toMatchObject({
      edgeRule: 'outer_row', ecdBasis: 'measured_pixels', statsPhase: 2,
      gbThreshold: '6.5', minPixels: '5', minIntercept: '2', alpha: '1.5', fill: false,
    });
    expect(runButton().disabled).toBe(false);

    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(screen.getByLabelText('Edge grains').value).toBe('outer_row');
    expect(screen.getByLabelText('Phase').value).toBe('2');
    expect(screen.getByDisplayValue('6.5')).toBeTruthy();
    expect(reconstructButton().disabled).toBe(false);
    await waitFor(() => expect(mapCalls('ecd').length).toBeGreaterThan(0));
    // the grains came without statistics: asked for once, with the settings the backend reported
    await waitFor(() => expect(screen.getByText(/Grains of Si:\s+12/)).toBeTruthy());
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);
    expect(analysisApi.grainSize).toHaveBeenCalledWith('outer_row', 'measured_pixels', 2);
    // and not again at the next tab click: nothing changed
    const asked = analysisApi.status.mock.calls.length;
    fireEvent.click(screen.getByText('Deformation'));
    fireEvent.click(screen.getByText('Grain Analysis'));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(asked + 2));
    await new Promise((r) => setTimeout(r, 50));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);
  }, 20000);

  it('a reconstruction started elsewhere (409): the detail is shown and the running one is polled', async () => {
    setUpBackend();
    const detail = 'A grain reconstruction is already running; wait for it to finish.';
    analysisApi.reconstructGrains.mockRejectedValue({ response: { status: 409, data: { detail } } });
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    analysisApi.status.mockResolvedValue(status({ reconstruct: { task_id: 't9', status: 'running', progress: 0.1 } }));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(screen.getByText(detail)).toBeTruthy());
    await waitFor(() => expect(state().reconTaskId).toBe('t9'));
    await waitFor(() => expect(analysisApi.getTaskStatus).toHaveBeenCalledWith('t9'), { timeout: 4000 });
    expect(reconstructButton().disabled).toBe(true);
    expect(runButton().disabled).toBe(true);
    expect(screen.getByText(detail)).toBeTruthy();
  }, 15000);
});

describe('AnalysisPage: the page watches a running reconstruction, whichever tab is open', () => {
  it('on a tab that does not poll, the end of the task is seen without a click: grains, statistics, a free Run Complete', async () => {
    setUpBackend();
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));

    analysisApi.status.mockResolvedValue(status({ reconstruct: RUNNING }));
    fireEvent.click(screen.getByText('Texture'));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(3));
    expect(runButton().disabled).toBe(true);
    expect(state().hasGrains).toBe(false);

    // the task ends; the user stays where they are and clicks nothing
    analysisApi.getTaskStatus.mockResolvedValue(doneTask);
    await waitFor(() => expect(state().hasGrains).toBe(true), { timeout: 5000 });
    expect(state()).toMatchObject({ reconTaskId: null, grainReconBusy: false });
    expect(state().grainStats).toMatchObject({ count: 98, meanEcd: 12.34 });
    expect(runButton().disabled).toBe(false);

    // the Grain tab shows the result and the end of the task
    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy();
    expect(screen.getByText('Reconstructed 98 grains')).toBeTruthy();
    expect(reconstructButton().disabled).toBe(false);
  }, 20000);

  it('a task that fails is seen too: the busy state ends and the Grain tab says why', async () => {
    setUpBackend();
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));
    analysisApi.status.mockResolvedValue(status({ reconstruct: RUNNING }));
    fireEvent.click(screen.getByText('Texture'));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(3));
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1, error: 'no grains today' } });
    await waitFor(() => expect(state().reconTaskId).toBeNull(), { timeout: 5000 });
    expect(state()).toMatchObject({ grainReconBusy: false, hasGrains: false });
    expect(runButton().disabled).toBe(false);
    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(screen.getByText('no grains today')).toBeTruthy();
  }, 20000);

  it('Run Complete with a tab click during the chain: the chain\'s task is polled once, and nothing stays busy at the end', async () => {
    setUpBackend();
    analysisApi.deformation.mockResolvedValue({ data: {} });
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.getMap.mockResolvedValue({ data: { image: 'AAAA' } });
    analysisApi.rxClassify.mockResolvedValue({ data: { success: true } });   // the RX step
    await openPage();
    const versionBefore = state().grainsVersion;
    fireEvent.click(runButton());
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));

    // a tab click asks /status, which lists the chain's own task
    analysisApi.status.mockResolvedValue(status({ reconstruct: RUNNING }));
    fireEvent.click(screen.getByText('Deformation'));
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));

    // while the task runs, one poll asks for it: the chain's (every second); the page's watch (every 1.5 s) stays out
    analysisApi.getTaskStatus.mockClear();
    await new Promise((r) => setTimeout(r, 3300));
    expect(analysisApi.getTaskStatus.mock.calls.length).toBeLessThanOrEqual(4);

    analysisApi.getTaskStatus.mockResolvedValue(doneTask);
    await waitFor(() => expect(screen.getByText(/Complete analysis finished/)).toBeTruthy(), { timeout: 8000 });
    expect(state()).toMatchObject({ reconTaskId: null, grainReconBusy: false, runAllBusy: false, hasGrains: true });
    expect(runButton().disabled).toBe(false);
    // one finished reconstruction is one new version (a second poll of the same task would make two), and the
    // deformation, texture and RX steps of the chain each draw the visible map again (parameters, preset and tolerance,
    // criteria): three more
    expect(state().grainsVersion).toBe(versionBefore + 4);
  }, 25000);

  it('Run Complete that fails after the reconstruction leaves no task id behind', async () => {
    setUpBackend();
    analysisApi.deformation.mockRejectedValue({ response: { status: 400, data: { detail: 'Reconstruct grains first' } } });
    analysisApi.getTaskStatus.mockResolvedValue(doneTask);
    await openPage();
    fireEvent.click(runButton());
    await waitFor(() => expect(screen.getByText(/Failed at step 2\/4.*Reconstruct grains first/)).toBeTruthy(), { timeout: 8000 });
    expect(state()).toMatchObject({ reconTaskId: null, grainReconBusy: false, runAllBusy: false });
    expect(runButton().disabled).toBe(false);
  }, 20000);
});

describe('AnalysisPage: races around the task id', () => {
  it('the reconstruction progress is shown in percent: 0.6 from the backend is 60 %', async () => {
    setUpBackend();
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.6 } });
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(screen.getByText('60%')).toBeTruthy(), { timeout: 5000 });
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('60');
  }, 15000);

  it('a /status answer asked before the task id was stored cannot take the id away', async () => {
    setUpBackend();
    const view = await openPage();
    let answer;
    analysisApi.status.mockImplementationOnce(() => new Promise((r) => { answer = r; }));
    leaveAndReturn(view);
    await waitFor(() => expect(answer).toBeDefined());
    state().setReconTaskId('tNew');                      // the POST of a reconstruction came back meanwhile
    answer(status({ reconstruct: null }));               // the backend had not heard of it yet
    await new Promise((r) => setTimeout(r, 100));
    expect(state()).toMatchObject({ reconTaskId: 'tNew', grainReconBusy: true });
  });

  it('a "completed" answer for an epoch that has moved on is dropped; the next poll applies it', async () => {
    setUpBackend();
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));
    let answerTask;
    analysisApi.getTaskStatus.mockImplementationOnce(() => new Promise((r) => { answerTask = r; }));
    await waitFor(() => expect(answerTask).toBeDefined(), { timeout: 4000 });
    useResultStore.setState({ grainSizeEpoch: state().grainSizeEpoch + 1 });       // something replaced the grains' basis
    analysisApi.getTaskStatus.mockResolvedValue(doneTask);
    answerTask(doneTask);
    await new Promise((r) => setTimeout(r, 200));
    expect(state().hasGrains).toBe(false);                                      // that answer was dropped
    await waitFor(() => expect(state().hasGrains).toBe(true), { timeout: 4000 });  // the next poll's is taken
  }, 20000);

  it('a "completed" answer that overtook a load of another map is not applied to it', async () => {
    setUpBackend();
    analysisApi.load.mockResolvedValue({ data: { loaded: true, shape: [20, 30], has_grains: false, phases: ['Ferrite'], phase_options: [{ id: 5, name: 'Ferrite' }] } });
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(reconstructButton());
    await waitFor(() => expect(state().reconTaskId).toBe('t1'));
    let answerTask;
    analysisApi.getTaskStatus.mockImplementationOnce(() => new Promise((r) => { answerTask = r; }));
    await waitFor(() => expect(answerTask).toBeDefined(), { timeout: 4000 });

    // a map is loaded on the Data tab while the poll's answer is on its way
    fireEvent.click(screen.getByText('Data & Preprocessing'));
    fireEvent.change(screen.getByPlaceholderText('/path/to/result.h5'), { target: { value: '/x/new.ang' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    await waitFor(() => expect(analysisApi.load).toHaveBeenCalledWith('/x/new.ang'));
    // the load took the task id of the replaced map with it
    await waitFor(() => expect(state().reconTaskId).toBeNull());
    expect(state().grainReconBusy).toBe(false);
    answerTask(doneTask);
    await new Promise((r) => setTimeout(r, 100));
    expect(state().hasGrains).toBe(false);
    expect(state().grainStats).toBeNull();
  }, 20000);
});

describe('AnalysisPage: the page is left and entered again', () => {
  it('a reconstruction that finished meanwhile: flag, parameters, settings, one new version, grain layers requested', async () => {
    setUpBackend();
    const view = await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    analysisApi.status.mockResolvedValue(grainStatus());
    const versionBefore = state().grainsVersion;
    leaveAndReturn(view);
    await waitFor(() => expect(state().hasGrains).toBe(true));
    expect(state().grainsVersion).toBe(versionBefore + 1);
    expect(screen.getByLabelText('Phase').value).toBe('2');
    await waitFor(() => expect(mapCalls('ecd').length).toBeGreaterThan(0));
  }, 15000);

  it('statistics that could not be had are not asked for at every activation: only when the grains appeared', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    analysisApi.grainSize.mockRejectedValue({ response: { status: 500, data: { detail: 'broke' } } });
    const view = await openPage();
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    expect(state().grainStats).toBeNull();
    leaveAndReturn(view);
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(2));
    await new Promise((r) => setTimeout(r, 100));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);
  });

  it('a status answer that is older than a reconstruction finished here does not take the grains away', async () => {
    setUpBackend();
    const view = await openPage();
    let answerStatus;
    analysisApi.status.mockImplementationOnce(() => new Promise((r) => { answerStatus = r; }));
    leaveAndReturn(view);
    await waitFor(() => expect(answerStatus).toBeDefined());

    state().markGrainsReconstructed();          // the chain or the tab finished while the question was on its way
    const version = state().grainsVersion;
    answerStatus(status());                      // asked before: no grains
    await new Promise((r) => setTimeout(r, 100));
    expect(state().hasGrains).toBe(true);
    expect(state().grainsVersion).toBe(version);
  });
});

describe('AnalysisPage: a map loaded or switched on another page', () => {
  it('no grains in the new map: the statistics go, the phase list is the new map\'s, the old pick of a phase is dropped', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    const view = await openPage();
    await waitFor(() => expect(state().hasGrains).toBe(true));
    state().setGrainStats({ count: 98, meanEcd: 12.34, nGrainsTotal: 12, statsPhase: 2, statsPhaseName: 'Si' });
    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(screen.getByText('Grain Statistics')).toBeTruthy();
    expect(screen.getByLabelText('Phase').value).toBe('2');

    // another page loaded another map: no grains, one phase that Si is not
    analysisApi.status.mockResolvedValue(status({ shape: [20, 30], phase_options: [{ id: 5, name: 'Ferrite' }] }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().hasGrains).toBe(false));
    expect(state().grainStats).toBeNull();
    expect(state().phaseOptions).toEqual([{ id: 5, name: 'Ferrite' }]);
    expect(state().grainStatsSettings.statsPhase).toBeNull();
    expect(screen.queryByText('Grain Statistics')).toBeNull();
    expect(Array.from(screen.getByLabelText('Phase').options).map((o) => o.textContent)).toEqual(['All phases', 'Ferrite']);
  }, 15000);

  it('another map with grains: the maps are fetched again and the statistics of the old map go', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    const view = await openPage();
    await waitFor(() => expect(state().hasGrains).toBe(true));
    state().setGrainStats({ count: 98, meanEcd: 12.34 });
    const version = state().grainsVersion;
    analysisApi.status.mockResolvedValue(grainStatus({ shape: [10, 10] }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().grainsVersion).toBe(version + 1));
    expect(state().grainStats).toBeNull();
  });

  it('nothing changed behind the page\'s back: the activation leaves version and statistics alone', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    const view = await openPage();
    await waitFor(() => expect(state().hasGrains).toBe(true));
    state().setGrainStats({ count: 98, meanEcd: 12.34 });
    const { grainsVersion, grainSizeEpoch } = state();
    leaveAndReturn(view);
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(2));
    await new Promise((r) => setTimeout(r, 50));
    expect(state()).toMatchObject({ grainsVersion, grainSizeEpoch, grainStats: { count: 98, meanEcd: 12.34 } });
  });
});

describe('AnalysisPage: a map loaded on the Data tab', () => {
  it('the layers that need no grains are fetched again, and the phase list is the new map\'s', async () => {
    setUpBackend();
    analysisApi.load.mockResolvedValue({ data: { loaded: true, shape: [20, 30], has_grains: false, phases: ['Ferrite'], phase_options: [{ id: 5, name: 'Ferrite' }] } });
    await openPage();
    await waitFor(() => expect(mapCalls('bc').length).toBeGreaterThan(0));
    const before = mapCalls('bc').length;

    fireEvent.change(screen.getByPlaceholderText('/path/to/result.h5'), { target: { value: '/x/new.ang' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    await waitFor(() => expect(analysisApi.load).toHaveBeenCalledWith('/x/new.ang'));
    await waitFor(() => expect(mapCalls('bc').length).toBe(before + 1));
    expect(state().phaseOptions).toEqual([{ id: 5, name: 'Ferrite' }]);
  });
});

describe('AnalysisPage: Run Complete Analysis and the Grain tab\'s own reconstruction', () => {
  const runButton = () => screen.getByRole('button', { name: /run complete analysis/i });

  it('waits while the tab reconstructs and is free again when that is over', async () => {
    setUpBackend();
    await openPage();
    expect(runButton().disabled).toBe(false);
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(runButton().disabled).toBe(true));

    analysisApi.getTaskStatus.mockResolvedValue({
      data: { status: 'completed', progress: 1, grain_count: 5, result: { grain_count: 5, n_grains: 5, mean_ecd: 1, median_ecd: 1, std_ecd: 1, mean_aspect_ratio: 1 } },
    });
    await waitFor(() => expect(runButton().disabled).toBe(false), { timeout: 5000 });
  }, 15000);

  it('is free again when the reconstruction failed', async () => {
    setUpBackend();
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1, error: 'no grains today' } });
    await openPage();
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(screen.getByText('no grains today')).toBeTruthy(), { timeout: 5000 });
    expect(runButton().disabled).toBe(false);
  }, 15000);

  it('the chain sends the chosen phase and its statistics name it', async () => {
    setUpBackend();
    analysisApi.getTaskStatus.mockResolvedValue({
      data: {
        status: 'completed', progress: 1, grain_count: 98,
        result: {
          n_grains: 98, grain_count: 98, mean_ecd: 3.5, median_ecd: 3.1, std_ecd: 0.9, mean_aspect_ratio: 1.4,
          edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: 2, stats_phase_name: 'Si',
          n_grains_total: 12, n_edge_grains: 5, n_grains_used: 7, edge_safeguard_applied: false,
        },
      },
    });
    analysisApi.deformation.mockResolvedValue({ data: {} });
    analysisApi.texture.mockResolvedValue({ data: {} });
    await openPage();
    state().setGrainStatsSettings({ statsPhase: 2 });
    fireEvent.click(screen.getByText('Grain Analysis'));
    fireEvent.click(runButton());
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));
    expect(analysisApi.reconstructGrains).toHaveBeenCalledWith(4, 3, 1.5, 3.1, true, 'matlab_script', 'cell_area', 2);
    await waitFor(() => expect(screen.getByText(/Grains of Si:\s+12 \(all phases: 98\)/)).toBeTruthy(), { timeout: 5000 });
  }, 15000);
});


describe('AnalysisPage: the backend lost its map', () => {
  it('"loaded: false": the page forgets the map, the statistics, the phases and the export; a batch stays', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    const view = await openPage();
    await waitFor(() => expect(state().hasGrains).toBe(true));
    state().setGrainStats({ count: 98, meanEcd: 12.34 });
    state().setBatchRun({ msg: 'Processed 2 of 2 files.', result: { processed: ['/d/a.h5'] } });
    state().setExportRun({ msg: 'Exported to C:/out/Doc.xlsx', warnings: ['rx: x'] });
    const { grainsVersion, grainSizeEpoch } = state();

    analysisApi.status.mockResolvedValue({ data: { loaded: false } });          // the backend was restarted
    leaveAndReturn(view);
    await waitFor(() => expect(state().analysisLoaded).toBe(false));
    expect(state()).toMatchObject({
      hasGrains: false, grainStats: null, phaseOptions: [], reconTaskId: null, grainReconBusy: false,
      grainsVersion: grainsVersion + 1, grainSizeEpoch: grainSizeEpoch + 1,
    });
    expect(state().batchRun.result).toEqual({ processed: ['/d/a.h5'] });
    expect(state().exportRun.warnings).toEqual([]);
    // the page says so: the empty-state guidance is back and the buttons need a map again
    expect(screen.getByText('No indexed data loaded.')).toBeTruthy();
    expect(runButton().disabled).toBe(true);
    expect(screen.queryByText(/pixels loaded/)).toBeNull();
  }, 15000);

  it('a batch that runs across a "loaded: false" is polled to its end and shows its result', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    const view = await openPage();
    fireEvent.click(screen.getByText('Batch & Export'));
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(state().batchRun.taskId).toBe('b1'));

    // the file was switched in the EBSD viewer: the backend has no analysis map for the new file
    analysisApi.status.mockResolvedValue({ data: { loaded: false } });
    leaveAndReturn(view);
    await waitFor(() => expect(state().analysisLoaded).toBe(false));
    expect(state().batchRun).toMatchObject({ taskId: 'b1', running: true });

    analysisApi.getTaskStatus.mockResolvedValue({
      data: {
        status: 'completed', progress: 1, message: 'Processed 1 of 1 files.',
        result: { n_files: 1, n_processed: 1, processed: ['/data/a.h5'], failed: [], skipped: [], warnings: [] },
      },
    });
    await waitFor(() => expect(screen.getByText('Processed files (1)')).toBeTruthy(), { timeout: 5000 });
  }, 20000);

  it('"loaded: false" on a page that never had a map changes nothing (a batch that is running is kept)', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue({ data: { loaded: false } });
    state().setBatchRun({ taskId: 'b1', running: true });
    render(page(true));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 50));
    expect(state().batchRun).toMatchObject({ taskId: 'b1', running: true });
    expect(state().grainsVersion).toBe(0);
  });
});

describe('AnalysisPage: a map swapped for another one of the same shape', () => {
  it('another dataset_token: maps are fetched again, statistics go, the pick of a phase and the settings are reset', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    const view = await openPage();
    await waitFor(() => expect(state().hasGrains).toBe(true));
    state().setGrainStats({ count: 98, meanEcd: 12.34 });
    expect(state().grainStatsSettings).toMatchObject({ statsPhase: 2, edgeRule: 'outer_row' });
    const { grainsVersion, grainSizeEpoch } = state();

    // the same size, the same phase ids and names, no grains: only the token says it is another map
    analysisApi.status.mockResolvedValue(status({ dataset_token: 'b2' }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().grainsVersion).toBe(grainsVersion + 1));
    expect(state().grainSizeEpoch).toBe(grainSizeEpoch + 1);
    expect(state()).toMatchObject({ hasGrains: false, grainStats: null, datasetToken: 'b2' });
    expect(state().grainStatsSettings).toMatchObject({ statsPhase: null, edgeRule: 'matlab_script', ecdBasis: 'cell_area' });
  }, 15000);

  it('the same token and shape: nothing is reset', async () => {
    setUpBackend();
    analysisApi.status.mockResolvedValue(grainStatus());
    const view = await openPage();
    await waitFor(() => expect(state().hasGrains).toBe(true));
    state().setGrainStats({ count: 98, meanEcd: 12.34 });
    const { grainsVersion } = state();
    leaveAndReturn(view);
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(2));
    await new Promise((r) => setTimeout(r, 50));
    expect(state()).toMatchObject({ grainsVersion, grainStats: { count: 98, meanEcd: 12.34 } });
    expect(state().grainStatsSettings.statsPhase).toBe(2);
  });

  it('the same ids with other names (no token yet): the pick of phase 2 does not stay with another phase', async () => {
    setUpBackend();
    const view = await openPage();
    state().setGrainStatsSettings({ statsPhase: 2 });
    analysisApi.status.mockResolvedValue(status({
      dataset_token: undefined, phase_options: [{ id: 1, name: 'Ferrite' }, { id: 2, name: 'Cementite' }],
    }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().phaseOptions[1].name).toBe('Cementite'));
    expect(state().grainStatsSettings.statsPhase).toBeNull();
    expect(screen.queryByText(/pixels loaded/)).not.toBeNull();
  }, 15000);
});

describe('AnalysisPage: a status answer that was overtaken changes nothing on the page', () => {
  it('neither the header, nor the tick of the grain step, nor the status line', async () => {
    setUpBackend();
    const view = await openPage();
    await waitFor(() => expect(screen.getByText(/40 × 60 pixels loaded/)).toBeTruthy());
    let answer;
    analysisApi.status.mockImplementationOnce(() => new Promise((r) => { answer = r; }));
    leaveAndReturn(view);
    await waitFor(() => expect(answer).toBeDefined());
    state().markGrainsReconstructed();                       // a reconstruction finished here meanwhile
    answer(status({ shape: [99, 99], has_grains: false }));
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByText(/99 × 99/)).toBeNull();
    expect(screen.getByText(/40 × 60 pixels loaded/)).toBeTruthy();
    expect(state().hasGrains).toBe(true);
  });
});

describe('AnalysisPage: the Batch tab survives a switch of tabs', () => {
  const doneBatch = {
    data: {
      status: 'completed', progress: 1, message: 'Processed 1 of 1 files.',
      result: { n_files: 1, n_processed: 1, processed: ['/data/a.h5'], failed: [], skipped: [], warnings: [] },
    },
  };

  it('a batch that finishes while the tab is away shows its result when the tab is opened again', async () => {
    setUpBackend();
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    await openPage();
    fireEvent.click(screen.getByText('Batch & Export'));
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(state().batchRun.taskId).toBe('b1'));

    fireEvent.click(screen.getByText('Data & Preprocessing'));
    analysisApi.getTaskStatus.mockResolvedValue(doneBatch);       // done while the tab is away
    fireEvent.click(screen.getByText('Batch & Export'));
    // running until the poll (which starts again with the tab) has seen it
    await waitFor(() => expect(screen.getByText('Processed files (1)')).toBeTruthy(), { timeout: 5000 });
    expect(screen.getByText('a.h5')).toBeTruthy();
    expect(state().batchRun).toMatchObject({ taskId: null, running: false });
  }, 20000);

  it('the result is shown with the settings that produced it, and the selects keep what was chosen', async () => {
    setUpBackend();
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue(doneBatch);
    await openPage();
    fireEvent.click(screen.getByText('Batch & Export'));
    fireEvent.change(screen.getByLabelText('Phase'), { target: { value: 'most_measured' } });
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'keep_all' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(screen.getByTestId('batch-ran-with')).toBeTruthy(), { timeout: 5000 });
    const expected = 'Run with: edge grains: all kept · ECD from: cell area · Phase with the most measured pixels'
      + ' · reconstruction 4°, 3 px, alpha 3.1, fill on'
      + ' · RX: GOS < 1.25°, gBC > 0.7, gKAM < 0.55°, radius > 0.4 µm'
      + ' · Deformation: KAM order 2, KAM threshold 7°, HAGB from 15°';
    expect(screen.getByTestId('batch-ran-with').closest('[aria-live="polite"]')).toBeTruthy();
    expect(screen.getByTestId('batch-ran-with').textContent).toBe(expected);

    // the tab is left; the batch's own settings stay (they live in the store), and so does the result
    fireEvent.click(screen.getByText('Deformation'));
    fireEvent.click(screen.getByText('Batch & Export'));
    expect(screen.getByLabelText('Phase').value).toBe('most_measured');
    expect(screen.getByLabelText('Edge grains').value).toBe('keep_all');
    expect(state().grainStatsSettings.edgeRule).toBe('matlab_script');        // the Grain tab's settings did not move
    expect(screen.getByTestId('batch-ran-with').textContent).toBe(expected);
  }, 20000);

  it('the result stays when the tab is left and opened again', async () => {
    setUpBackend();
    state().setBatchRun({
      msg: 'Processed 1 of 1 files.',
      result: { n_files: 1, n_processed: 1, processed: ['/data/a.h5'], failed: [], skipped: [], warnings: [] },
    });
    await openPage();
    fireEvent.click(screen.getByText('Batch & Export'));
    expect(screen.getByText('Processed files (1)')).toBeTruthy();
    fireEvent.click(screen.getByText('Deformation'));
    fireEvent.click(screen.getByText('Batch & Export'));
    expect(screen.getByText('Processed files (1)')).toBeTruthy();
  });

  it('an export that finishes while the tab is away still tells the user what is missing in the workbook', async () => {
    setUpBackend();
    window.electronAPI = { saveFile: vi.fn() };
    try {
      analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
      await openPage();
      fireEvent.click(screen.getByText('Batch & Export'));
      fireEvent.change(screen.getByPlaceholderText('Documentation.xlsx'), { target: { value: 'C:/out/Doc.xlsx' } });
      fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
      await waitFor(() => expect(state().exportRun.taskId).toBe('e1'));

      fireEvent.click(screen.getByText('Deformation'));
      analysisApi.getTaskStatus.mockResolvedValue({
        data: {
          status: 'completed', progress: 1,
          result: { path: 'C:/out/Doc.xlsx', requested_path: 'C:/out/Doc.xlsx', warnings: ['texture: ValueError: Größe µm'] },
        },
      });
      fireEvent.click(screen.getByText('Batch & Export'));
      await waitFor(() => expect(screen.getByText('texture: ValueError: Größe µm')).toBeTruthy(), { timeout: 5000 });
      expect(screen.getByText('Exported to C:/out/Doc.xlsx')).toBeTruthy();
    } finally {
      delete window.electronAPI;
    }
  }, 20000);
});
