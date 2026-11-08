// @vitest-environment jsdom
//
// The whole page: (1) a reconstruction parameter typed on the Grain tab is not taken back by the status answer that
// every tab change asks for; (2) the results the Deformation, Texture and RX tabs keep (and the ticks of the tab bar)
// go when the map is replaced or lost; (3) "Run Complete Analysis" sends the Deformation parameters and the RX phase that
// are set now. Real components, only the HTTP layer mocked, with the shapes of the real answers.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    status: vi.fn(), load: vi.fn(), reconstructGrains: vi.fn(), getTaskStatus: vi.fn(), getMap: vi.fn(), grainSize: vi.fn(),
    batch: vi.fn(), qualityStats: vi.fn(), deformation: vi.fn(), texture: vi.fn(), rxClassify: vi.fn(), exportExcel: vi.fn(),
    textureComponents: vi.fn(), applyQualityFilter: vi.fn(),
  },
  indexApi: { getLastResult: vi.fn() },
  edsApi: {},
}));

import { analysisApi, indexApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import AnalysisPage from '../AnalysisPage';

const PHASES = [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }];
const status = (extra = {}) => ({
  data: {
    loaded: true, shape: [40, 60], has_grains: true, phases: ['Aluminium', 'Si'], phase_options: PHASES, step_size: 0.5,
    edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null,
    recon_params: { threshold: 5, min_pixels: 3, min_intercept: null, alpha: 3.1, fill: true },
    dataset_token: 'a1', reconstruct: null, has_native_bc: true,
    ...extra,
  },
});

function setUpBackend() {
  indexApi.getLastResult.mockRejectedValue(new Error('none'));
  analysisApi.qualityStats.mockRejectedValue(new Error('none'));
  analysisApi.status.mockResolvedValue(status());
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  analysisApi.grainSize.mockRejectedValue(new Error('none'));
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  useResultStore.getState().clearResults();
  try { sessionStorage.clear(); } catch { /* ignore */ }
});

const state = () => useResultStore.getState();
async function openPage() {
  const view = render(<AnalysisPage isActive onNavigate={() => {}} />);
  await waitFor(() => expect(state().analysisLoaded).toBe(true));
  return view;
}
const tab = (label) => fireEvent.click(screen.getByText(label));

describe('a reconstruction parameter typed on the Grain tab', () => {
  it('is still there after a visit to another tab (the status answer carries the old grains\' 4 degrees)', async () => {
    setUpBackend();
    await openPage();
    tab('Grain Analysis');
    await waitFor(() => expect(screen.getByLabelText('GB Threshold:').value).toBe('5'));
    fireEvent.change(screen.getByLabelText('GB Threshold:'), { target: { value: '6' } });
    tab('Batch & Export');
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(3));
    expect(state().grainStatsSettings.gbThreshold).toBe('6');
    expect(screen.getByTestId('batch-recon-line').textContent).toMatch(/as set in Grain Analysis: 6°, 3 px/);
    tab('Grain Analysis');
    expect(screen.getByLabelText('GB Threshold:').value).toBe('6');
  });
});

describe('results of the old map', () => {
  // The page stays mounted when another page is shown, and its tab stays selected: the status answer on re-entering is
  // what tells it that the map is another one (switching tabs would remount the tab and hide the problem).
  const page = (isActive) => <AnalysisPage isActive={isActive} onNavigate={() => {}} />;
  const leaveAndReturn = (view) => { view.rerender(page(false)); view.rerender(page(true)); };

  it('go when the map is swapped: the RX numbers and the tick of the tab', async () => {
    setUpBackend();
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_fraction: 12.3, rx_grain_count: 4, total_grains: 30 } });
    const view = await openPage();
    tab('Recrystallization');
    fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));
    await waitFor(() => expect(screen.getByText(/RX Fraction:\s+12\.3%/)).toBeTruthy());
    expect(screen.queryAllByTitle(/^Recrystallization.*completed$/)).toHaveLength(1);       // the tick is there before

    analysisApi.status.mockResolvedValue(status({ dataset_token: 'b2', has_grains: false, recon_params: null }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().datasetToken).toBe('b2'));
    expect(screen.queryByText(/RX Fraction:\s+12\.3%/)).toBeNull();
    expect(screen.queryAllByTitle(/^Recrystallization.*completed$/)).toHaveLength(0);
  }, 20000);

  it('go when the map is swapped: the deformation numbers', async () => {
    setUpBackend();
    analysisApi.deformation.mockResolvedValue({ data: { mean_kam: 0.5 } });
    const view = await openPage();
    tab('Deformation');
    fireEvent.click(screen.getByRole('button', { name: /analyze deformation/i }));
    await waitFor(() => expect(screen.getByText(/Mean KAM:\s+0\.500/)).toBeTruthy());
    analysisApi.status.mockResolvedValue(status({ dataset_token: 'b2', has_grains: false, recon_params: null }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().datasetToken).toBe('b2'));
    expect(screen.queryByText(/Mean KAM:/)).toBeNull();
  }, 20000);

  it('go when the map is swapped: the texture numbers', async () => {
    setUpBackend();
    analysisApi.texture.mockResolvedValue({
      data: { texture_index: 2.5, texture_entropy: 1.1, texture_phase: { id: 1, name: 'Aluminium' }, components: [], surface_planes: null },
    });
    const view = await openPage();
    tab('Texture');
    fireEvent.click(screen.getByRole('button', { name: /analyze texture/i }));
    await waitFor(() => expect(screen.getByTestId('texture-phase')).toBeTruthy());
    analysisApi.status.mockResolvedValue(status({ dataset_token: 'b2', has_grains: false, recon_params: null }));
    leaveAndReturn(view);
    await waitFor(() => expect(state().datasetToken).toBe('b2'));
    expect(screen.queryByTestId('texture-phase')).toBeNull();
  }, 20000);

  it('go when the backend lost its map', async () => {
    setUpBackend();
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_fraction: 12.3 } });
    const view = await openPage();
    tab('Recrystallization');
    fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));
    await waitFor(() => expect(screen.getByText(/RX Fraction:\s+12\.3%/)).toBeTruthy());
    analysisApi.status.mockResolvedValue({ data: { loaded: false } });
    leaveAndReturn(view);
    await waitFor(() => expect(state().analysisLoaded).toBe(false));
    expect(screen.queryByText(/RX Fraction:\s+12\.3%/)).toBeNull();
  }, 20000);

  it('stay while the map is the same one (a status answer of the same map takes nothing away)', async () => {
    setUpBackend();
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_fraction: 12.3 } });
    const view = await openPage();
    tab('Recrystallization');
    fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));
    await waitFor(() => expect(screen.getByText(/RX Fraction:\s+12\.3%/)).toBeTruthy());
    leaveAndReturn(view);
    await waitFor(() => expect(analysisApi.status.mock.calls.length).toBeGreaterThanOrEqual(3));
    expect(screen.getByText(/RX Fraction:\s+12\.3%/)).toBeTruthy();
  }, 20000);
});

describe('Run Complete Analysis', () => {
  it('sends the Deformation parameters and the RX phase that are set now', async () => {
    setUpBackend();
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'completed', progress: 1, grain_count: 5, result: { grain_count: 5 } } });
    analysisApi.deformation.mockResolvedValue({ data: {} });
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.rxClassify.mockResolvedValue({ data: {} });
    analysisApi.getMap.mockResolvedValue({ data: { image: 'AAAA' } });
    state().setDeformationParams({ kamOrder: '4', hagbMin: '12' });
    state().setRxPhase(2);
    await openPage();
    fireEvent.click(screen.getByRole('button', { name: /run complete analysis/i }));
    await waitFor(() => expect(screen.getByText(/Complete analysis finished/)).toBeTruthy(), { timeout: 8000 });
    expect(analysisApi.deformation).toHaveBeenCalledWith({ kamOrder: 4, kamThreshold: 7, hagbMin: 12 });
    expect(analysisApi.rxClassify).toHaveBeenCalledWith({ gosThreshold: 1.25, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 0.4 }, 2);
    expect(state().deformationEdited).toBe(false);
  }, 20000);

  it('checks the Deformation parameters before the first step', async () => {
    setUpBackend();
    state().setDeformationParams({ kamOrder: '9' });
    await openPage();
    fireEvent.click(screen.getByRole('button', { name: /run complete analysis/i }));
    await waitFor(() => expect(screen.getByText(/KAM order a whole number 1–5/)).toBeTruthy());
    expect(analysisApi.reconstructGrains).not.toHaveBeenCalled();
  });
});

describe('the quality filter', () => {
  it('draws the maps again and says that grains made before it are the old ones', async () => {
    setUpBackend();
    analysisApi.qualityStats.mockResolvedValue({ data: { bc: { max: 255, p05: 10, p50: 100, p95: 200 }, bands: null } });
    analysisApi.applyQualityFilter.mockResolvedValue({
      data: { success: true, filtered_total: 12, filtered_fraction: 0.01, filtered_by_bc: 12, filtered_by_bands: 0, indexed_after: 100, indexed_before: 112 },
    });
    await openPage();
    await waitFor(() => expect(screen.getByLabelText('Min Band Contrast:')).toBeTruthy());
    const version = state().grainsVersion;
    fireEvent.click(screen.getByRole('button', { name: /apply quality filter/i }));
    await waitFor(() => expect(screen.getByText(/Grains made before this filter are still the old ones: reconstruct them again\./)).toBeTruthy());
    expect(state().grainsVersion).toBe(version + 1);
  });
});
