// @vitest-environment jsdom
//
// The whole page, not only the Grain tab: AnalysisPage raises `grainsVersion` when a
// reconstruction finishes and hands it, with hasGrains, to the map viewer. Without that the
// ECD map of the Grain tab, left behind as a hint, would not be fetched after the run.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
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
  },
  indexApi: { getLastResult: vi.fn() },
  edsApi: {},
}));

import { analysisApi, indexApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import AnalysisPage from '../AnalysisPage';

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  useResultStore.getState().clearResults();
  try { sessionStorage.clear(); } catch { /* ignore */ }
});

function setUpBackend() {
  indexApi.getLastResult.mockRejectedValue(new Error('none'));
  analysisApi.qualityStats.mockRejectedValue(new Error('none'));
  analysisApi.status.mockResolvedValue({ data: { loaded: true, shape: [10, 10], has_grains: false } });
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
  analysisApi.getTaskStatus.mockResolvedValue({
    data: {
      status: 'completed', progress: 1, grain_count: 5,
      result: { grain_count: 5, n_grains: 5, mean_ecd: 1, median_ecd: 1, std_ecd: 1, mean_aspect_ratio: 1 },
    },
  });
}

const mapCalls = (type) => analysisApi.getMap.mock.calls.filter((c) => c[0] === type);

async function openGrainTab() {
  render(<AnalysisPage isActive onNavigate={() => {}} />);
  await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
  fireEvent.click(screen.getByText('Grain Analysis'));
}

describe('AnalysisPage: the Grain tab map after a reconstruction', () => {
  it('shows the hint first and requests the ECD map once the grains exist', async () => {
    setUpBackend();
    await openGrainTab();
    await waitFor(() => expect(screen.getAllByText('This layer needs grains. Reconstruct grains first.').length).toBeGreaterThan(0));
    expect(mapCalls('ecd')).toHaveLength(0);

    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(1), { timeout: 5000 });
  });

  it('a layer that needs no grains is fetched again when the reconstruction finishes', async () => {
    setUpBackend();
    await openGrainTab();
    const layer = Array.from(document.querySelectorAll('select')).find((el) => el.value === 'ecd');
    const onDataTab = mapCalls('bc').length;           // the Data tab asked for it before
    fireEvent.change(layer, { target: { value: 'bc' } });
    await waitFor(() => expect(mapCalls('bc')).toHaveLength(onDataTab + 1));

    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(mapCalls('bc')).toHaveLength(onDataTab + 2), { timeout: 5000 });
  });
});

describe('AnalysisPage: the grain-size settings follow the backend', () => {
  const chosen = { edgeRule: 'outer_row', ecdBasis: 'measured_pixels' };

  it('the selects keep their value when the tab is left and opened again', async () => {
    setUpBackend();
    await openGrainTab();
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    fireEvent.click(screen.getByText('Deformation'));
    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(screen.getByLabelText('ECD from').value).toBe('measured_pixels');
    expect(screen.getByLabelText('Edge grains').value).toBe('outer_row');
  });

  it('loading a map puts them back to the defaults, as the backend does', async () => {
    setUpBackend();
    analysisApi.load.mockResolvedValue({ data: { loaded: true, shape: [10, 10], has_grains: false } });
    await openGrainTab();
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    expect(useResultStore.getState().grainStatsSettings.ecdBasis).toBe('measured_pixels');

    fireEvent.click(screen.getByText('Data & Preprocessing'));
    fireEvent.change(screen.getByPlaceholderText('/path/to/result.h5'), { target: { value: '/x/new.ang' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    await waitFor(() => expect(analysisApi.load).toHaveBeenCalledWith('/x/new.ang'));
    await waitFor(() => expect(useResultStore.getState().grainStatsSettings.ecdBasis).toBe('cell_area'));
    fireEvent.click(screen.getByText('Grain Analysis'));
    expect(screen.getByLabelText('ECD from').value).toBe('cell_area');
  });

  it('a map handed over from a batch run resets them too', async () => {
    setUpBackend();
    useResultStore.setState({ grainStatsSettings: chosen });
    analysisApi.load.mockResolvedValue({ data: { loaded: true, shape: [10, 10], has_grains: false } });
    sessionStorage.setItem('batch_handoff_path', '/x/batch.ang');
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(analysisApi.load).toHaveBeenCalledWith('/x/batch.ang'));
    await waitFor(() => expect(useResultStore.getState().grainStatsSettings).toEqual({
      edgeRule: 'matlab_script', ecdBasis: 'cell_area', statsPhase: null,
    }));
  });
});

describe('AnalysisPage: a grain layer after a reconstruction', () => {
  // The cause (hasGrains and the version in one update) is pinned in the store test; this is the symptom.
  it('is requested once', async () => {
    setUpBackend();
    await openGrainTab();
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(1), { timeout: 5000 });
    await new Promise((r) => setTimeout(r, 200));
    expect(mapCalls('ecd')).toHaveLength(1);
  });
});

describe('AnalysisPage: Run Complete Analysis', () => {
  const runButton = () => screen.getByRole('button', { name: /run complete analysis/i });

  function setUpChain(taskPayload) {
    setUpBackend();
    analysisApi.getTaskStatus.mockResolvedValue({ data: taskPayload });
    analysisApi.deformation.mockResolvedValue({ data: {} });
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.getMap.mockResolvedValue({ data: { image: 'AAAA' } });   // the RX step asks for a map
    analysisApi.rxClassify.mockResolvedValue({ data: { success: true } });   // the RX step
  }

  it('sends the stored parameters and statistics settings, and goes on when the backend says completed', async () => {
    setUpChain({ status: 'completed', progress: 1, grain_count: 5, result: { grain_count: 5 } });
    useResultStore.getState().setGrainStatsSettings({
      gbThreshold: '6.5', minPixels: '5', edgeRule: 'outer_row', ecdBasis: 'measured_pixels',
    });
    useResultStore.getState().setRxCriteria({ gosThreshold: '2', minRadius: '1.5' });      // typed in the RX tab, never sent
    useResultStore.getState().setTexturePreset('BCC_Rolling');                             // chosen in the Texture tab, not run
    useResultStore.getState().setTextureTolerance('4');
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    fireEvent.click(runButton());
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));
    expect(analysisApi.reconstructGrains).toHaveBeenCalledWith(6.5, 5, 1.5, 3.1, true, 'outer_row', 'measured_pixels', null);
    // 'completed' is what the backend writes; the chain used to wait for 'done' and timed out after 5 minutes
    await waitFor(() => expect(analysisApi.deformation).toHaveBeenCalledTimes(1), { timeout: 5000 });
    await waitFor(() => expect(screen.getByText(/Complete analysis finished/)).toBeTruthy(), { timeout: 5000 });
    expect(analysisApi.texture).toHaveBeenCalledTimes(1);
    expect(analysisApi.texture).toHaveBeenCalledWith('most_measured', 'BCC_Rolling', 4);       // the tab's choice, as it is now
    expect(useResultStore.getState().textureEdited).toBe(false);                               // and the run applied it
    expect(useResultStore.getState().hasGrains).toBe(true);
    // the RX step classifies with the criteria of the RX tab, not with whatever the backend held before
    expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1);
    expect(analysisApi.rxClassify).toHaveBeenCalledWith({ gosThreshold: 2, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 1.5 }, 'most_measured');
  }, 15000);

  it('checks the RX criteria before the first step: nothing runs with a criterion that cannot be sent', async () => {
    setUpChain({ status: 'completed', progress: 1, grain_count: 5, result: { grain_count: 5 } });
    useResultStore.getState().setRxCriteria({ gkamThreshold: '' });
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    fireEvent.click(runButton());
    await waitFor(() => expect(screen.getByText(/Enter numbers inside the ranges/)).toBeTruthy());
    expect(analysisApi.reconstructGrains).not.toHaveBeenCalled();
    expect(analysisApi.deformation).not.toHaveBeenCalled();
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
    expect(useResultStore.getState().runAllBusy).toBe(false);
  }, 15000);

  it('refuses an unusable texture tolerance before anything runs, with the hint of the Texture tab, and keeps the pick pending', async () => {
    setUpChain({ status: 'completed', progress: 1, grain_count: 5, result: { grain_count: 5 } });
    useResultStore.getState().setTextureTolerance('');
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    fireEvent.click(runButton());
    await waitFor(() => expect(screen.getByText('Tolerance angle: more than 0° and at most 30°.')).toBeTruthy());
    expect(analysisApi.reconstructGrains).not.toHaveBeenCalled();
    expect(analysisApi.texture).not.toHaveBeenCalled();
    expect(useResultStore.getState().runAllBusy).toBe(false);
    expect(useResultStore.getState().textureEdited).toBe(true);            // nothing was applied
    expect(useResultStore.getState().textureTolerance).toBe('');
  }, 15000);

  it('fetches the visible map layer again after the texture step (it was drawn with the earlier choice)', async () => {
    setUpChain({ status: 'completed', progress: 1, grain_count: 5, result: { grain_count: 5 } });
    let versionAtTexture = null;
    analysisApi.texture.mockImplementation(async () => {
      versionAtTexture = useResultStore.getState().grainsVersion;
      return { data: {} };
    });
    analysisApi.rxClassify.mockImplementation(() => new Promise(() => {}));   // the chain waits at the RX step
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    fireEvent.click(runButton());
    await waitFor(() => expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1), { timeout: 5000 });
    expect(versionAtTexture).not.toBeNull();
    expect(useResultStore.getState().grainsVersion).toBeGreaterThan(versionAtTexture);
  }, 15000);

  it('stops with the backend message when the reconstruction failed', async () => {
    setUpChain({ status: 'failed', progress: 1, error: 'no grains today' });
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    fireEvent.click(runButton());
    await waitFor(() => expect(screen.getByText(/Failed at step 1\/4.*no grains today/)).toBeTruthy(), { timeout: 5000 });
    expect(analysisApi.deformation).not.toHaveBeenCalled();
  }, 15000);

  it('stops when the backend no longer knows the task', async () => {
    setUpChain({});
    analysisApi.getTaskStatus.mockRejectedValue({ response: { status: 404 } });
    render(<AnalysisPage isActive onNavigate={() => {}} />);
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    fireEvent.click(runButton());
    await waitFor(() => expect(screen.getByText(/Failed at step 1\/4.*Reconstruction failed/)).toBeTruthy(), { timeout: 5000 });
    expect(analysisApi.deformation).not.toHaveBeenCalled();
  }, 15000);
});

describe('AnalysisPage: Run Complete Analysis with the Grain tab open', () => {
  const chainResult = {
    status: 'completed', progress: 1, grain_count: 98,
    result: {
      n_grains: 98, grain_count: 98, mean_ecd: 12.34, median_ecd: 11, std_ecd: 3.21, mean_aspect_ratio: 1.678,
      edge_rule: 'matlab_script', ecd_basis: 'cell_area',
      n_grains_total: 98, n_edge_grains: 34, n_grains_used: 64, edge_safeguard_applied: false,
    },
  };

  async function runChainOnGrainTab() {
    setUpBackend();
    let finishTask;
    let finishDeformation;
    analysisApi.getTaskStatus.mockImplementation(() => new Promise((r) => { finishTask = r; }));
    analysisApi.deformation.mockImplementation(() => new Promise((r) => { finishDeformation = r; }));
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.getMap.mockResolvedValue({ data: { image: 'AAAA' } });
    analysisApi.rxClassify.mockResolvedValue({ data: { success: true } });   // the RX step
    await openGrainTab();
    fireEvent.click(screen.getByRole('button', { name: /run complete analysis/i }));
    return {
      finishTask: async (payload) => {
        await waitFor(() => expect(finishTask).toBeDefined(), { timeout: 4000 });
        finishTask({ data: payload });
      },
      finishDeformation: async () => {
        await waitFor(() => expect(finishDeformation).toBeDefined(), { timeout: 4000 });
        finishDeformation({ data: {} });
      },
    };
  }

  /** Grains exist and a /grain-size answer is on its way when the chain is started. */
  async function runChainOnGrainTabWithGrains() {
    setUpBackend();
    analysisApi.status.mockResolvedValue({ data: { loaded: true, shape: [10, 10], has_grains: true } });
    let resolveOldAnswer;
    analysisApi.grainSize.mockImplementationOnce(() => new Promise((r) => { resolveOldAnswer = r; }));
    let finishTask;
    let finishDeformation;
    analysisApi.getTaskStatus.mockImplementation(() => new Promise((r) => { finishTask = r; }));
    analysisApi.deformation.mockImplementation(() => new Promise((r) => { finishDeformation = r; }));
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.getMap.mockResolvedValue({ data: { image: 'AAAA' } });
    analysisApi.rxClassify.mockResolvedValue({ data: { success: true } });   // the RX step
    await openGrainTab();
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole('button', { name: /run complete analysis/i }));
    return {
      resolveOldAnswer: (answer) => resolveOldAnswer(answer),
      finishTask: async (payload) => {
        await waitFor(() => expect(finishTask).toBeDefined(), { timeout: 4000 });
        finishTask({ data: payload });
      },
      finishDeformation: () => finishDeformation?.({ data: {} }),
    };
  }

  it('the tab stands still during the chain and shows its result as soon as the grains exist', async () => {
    const chain = await runChainOnGrainTab();
    await waitFor(() => expect(screen.getByLabelText('Edge grains').disabled).toBe(true));
    expect(screen.getByLabelText('ECD from').disabled).toBe(true);
    expect(screen.getByRole('button', { name: /reconstruct grains/i }).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await new Promise((r) => setTimeout(r, 400));
    expect(analysisApi.grainSize).not.toHaveBeenCalled();

    await chain.finishTask(chainResult);
    // right after the reconstruction, not at the end of the chain: grains flag and the chain's statistics
    await waitFor(() => expect(analysisApi.deformation).toHaveBeenCalledTimes(1), { timeout: 4000 });
    expect(useResultStore.getState().hasGrains).toBe(true);
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy());
    expect(screen.getByLabelText('Edge grains').disabled).toBe(true);        // the chain is still running

    await chain.finishDeformation();
    await waitFor(() => expect(screen.getByText(/Complete analysis finished/)).toBeTruthy(), { timeout: 5000 });
    expect(screen.getByLabelText('Edge grains').disabled).toBe(false);
    expect(screen.getByRole('button', { name: /reconstruct grains/i }).disabled).toBe(false);
  }, 20000);

  it('the block of the chain survives a switch of tabs', async () => {
    const chain = await runChainOnGrainTab();
    await chain.finishTask(chainResult);
    await chain.finishDeformation();
    await waitFor(() => expect(screen.getByText(/Complete analysis finished/)).toBeTruthy(), { timeout: 5000 });
    // the backend has the grains now; every tab click asks it (GET /status) and the answer must agree
    analysisApi.status.mockResolvedValue({ data: { loaded: true, shape: [10, 10], has_grains: true } });
    fireEvent.click(screen.getByText('Deformation'));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(3));
    fireEvent.click(screen.getByText('Grain Analysis'));
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(4));
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy();
  }, 20000);

  it('a /grain-size answer asked for before the chain began does not land on the chain\'s grains', async () => {
    const chain = await runChainOnGrainTabWithGrains();
    await chain.finishTask(chainResult);
    await waitFor(() => expect(analysisApi.deformation).toHaveBeenCalledTimes(1), { timeout: 4000 });
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy());
    chain.resolveOldAnswer({
      data: { success: true, ...chainResult.result, mean_ecd: 99, median_ecd: 98, std_ecd: 1, mean_aspect_ratio: 1, ecd_basis: 'measured_pixels' },
    });
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy();
    expect(screen.queryByText(/99\.00/)).toBeNull();
    chain.finishDeformation();
  }, 20000);
});
