// @vitest-environment jsdom
//
// The Batch & Export tab: the statistics phase of the batch, the reconstruction line, what a batch did
// (processed / failed / skipped / warnings, with the reasons), and the warnings of an Excel export on
// both paths (the Electron task and the browser's blob request). Real component, only the HTTP layer
// mocked, with the shapes of the real answers (routes/analysis.py batch task, export task, X-Orienta-Warnings).
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within, act } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    getTaskStatus: vi.fn(),
    getMap: vi.fn(),
    batch: vi.fn(),
    rxClassify: vi.fn(),
    status: vi.fn(),
    exportExcel: vi.fn(),
    downloadExcel: vi.fn(),
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { BatchTab } from '../AnalysisPage';
import { colors } from '../../../theme/components';

const POLL_TIMEOUT = { timeout: 6000 };

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.restoreAllMocks();
  delete window.electronAPI;
  useResultStore.getState().clearResults();
});

function mount() {
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  return render(<BatchTab analysisLoaded onComplete={() => {}} />);
}

function startBatch() {
  fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
  fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
  fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
}

describe('BatchTab: the statistics phase', () => {
  it('offers All phases and the phase with the most measured pixels, not the phases of the loaded map', () => {
    useResultStore.setState({ phaseOptions: [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }] });
    mount();
    const select = screen.getByLabelText('Phase');
    expect(Array.from(select.options).map((o) => [o.value, o.textContent])).toEqual([
      ['all', 'All phases'], ['most_measured', 'Phase with the most measured pixels'],
    ]);
    expect(select.value).toBe('all');
    expect(screen.getByText('Statistics over the grains of all phases.')).toBeTruthy();
    const optionTitles = Array.from(select.options).map((o) => o.title);
    expect(optionTitles[0]).toBe('Statistics over the grains of all phases.');
    expect(optionTitles[1]).toMatch(/^In each file, the phase with the most measured pixels \(ties: the lower phase number\); Statistics over the grains of this phase only\./);
  });

  it('sends null for all phases and "most_measured" for the choice', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    mount();
    fireEvent.change(screen.getByLabelText('Phase'), { target: { value: 'most_measured' } });
    startBatch();
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch.mock.calls[0].slice(0, 5)).toEqual(['F', 'O.xlsx', 'matlab_script', 'cell_area', 'most_measured']);
  });

  it('starts at "most measured" when the Grain tab says so', () => {
    useResultStore.setState({ grainStatsSettings: { ...useResultStore.getState().grainStatsSettings, statsPhase: 'most_measured' } });
    mount();
    expect(screen.getByLabelText('Phase').value).toBe('most_measured');
  });

  it('a phase id of the loaded map means nothing for other files: it starts at all phases', () => {
    useResultStore.setState({ grainStatsSettings: { ...useResultStore.getState().grainStatsSettings, statsPhase: 2 } });
    mount();
    expect(screen.getByLabelText('Phase').value).toBe('all');
  });
});

describe('BatchTab: the grain reconstruction of the batch', () => {
  it('says what it will use, from the Grain tab', () => {
    mount();
    expect(screen.getByTestId('batch-recon-line').textContent)
      .toBe('Grain reconstruction as set in Grain Analysis: 4.0°, 3 px, alpha 3.1, fill on; min intercept 1.5 µm (recorded only, not applied)');
  });

  it('follows the Grain tab and sends exactly those numbers', async () => {
    useResultStore.getState().setGrainStatsSettings({ gbThreshold: '6.5', minPixels: '5', minIntercept: '2', alpha: '0', fill: false });
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    mount();
    expect(screen.getByTestId('batch-recon-line').textContent)
      .toBe('Grain reconstruction as set in Grain Analysis: 6.5°, 5 px, alpha 0, fill off; min intercept 2 µm (recorded only, not applied)');
    startBatch();
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch.mock.calls[0][5]).toEqual({ threshold: 6.5, minPixels: 5, minIntercept: 2, alpha: 0, fill: false });
  });

  it('does not start with a field of the Grain tab that is not a number, and says why', () => {
    useResultStore.getState().setGrainStatsSettings({ alpha: '' });
    mount();
    expect(screen.getByTestId('batch-recon-line').textContent).toMatch(/parameters in Grain Analysis are incomplete/);
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(true);
  });
});

describe('BatchTab: what it reads, records and waits for', () => {
  it('names the files the batch takes: exactly the extensions of the processor', () => {
    mount();
    expect(screen.getByText('Input folder (.h5oina, .h5, .ang, .ctf files)')).toBeTruthy();
  });

  it('says what is missing while the button is disabled only for want of a path', () => {
    mount();
    const hint = 'Choose an input folder and an output file to start.';
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(true);
    expect(screen.getByText(hint)).toBeTruthy();
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    expect(screen.getByText(hint)).toBeTruthy();                     // the output file is still missing
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    expect(screen.queryByText(hint)).toBeNull();
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(false);
  });

  it('gives no hint about paths when something else disables the button', () => {
    useResultStore.getState().setGrainStatsSettings({ alpha: '' });
    mount();
    expect(screen.queryByText('Choose an input folder and an output file to start.')).toBeNull();
  });

  it('says what Sheet 1 records, under the batch and under the export', () => {
    mount();
    const sentence = 'Sheet 1 of the workbook records, for each dataset (a file in a batch), the grain reconstruction parameters (threshold, minimum pixels, minimum intercept, alpha), the edge rule, the ECD basis, the phase, the numbers of grains in total, of edge grains found and of grains used, whether the edge rule kept all grains, and, where the RX analysis ran, the RX criteria.';
    expect(screen.getByTestId('batch-records').textContent).toBe(sentence);
    // the export is one dataset: no "a file in a batch"
    expect(screen.getByTestId('export-records').textContent).toBe(
      sentence.replace(', for each dataset (a file in a batch), ', ', for the dataset, '));
    expect(screen.getByTestId('export-records').textContent).not.toMatch(/batch/);
  });

  it('says its statistics selects are its own copy of the Grain tab\'s, and that is what the store shows', () => {
    mount();
    expect(screen.getByTestId('batch-stats-own').textContent)
      .toBe("The batch's own settings: they start as on the Grain Analysis tab (a single phase there becomes All phases here) and stay when you switch tabs.");
    const before = { ...useResultStore.getState().grainStatsSettings };
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    fireEvent.change(screen.getByLabelText('Phase'), { target: { value: 'most_measured' } });
    const after = useResultStore.getState().grainStatsSettings;
    expect([after.edgeRule, after.ecdBasis, after.statsPhase]).toEqual([before.edgeRule, before.ecdBasis, before.statsPhase]);
    // and the other way: a change on the Grain tab's side does not move the open Batch tab
    act(() => { useResultStore.getState().setGrainStatsSettings({ edgeRule: 'keep_all' }); });
    expect(screen.getByLabelText('Edge grains').value).toBe('outer_row');
  });
});

describe('BatchTab: the statistics settings are the batch own and stay when the tab is left', () => {
  const grain = (patch) => useResultStore.setState({
    grainStatsSettings: { ...useResultStore.getState().grainStatsSettings, ...patch },
  });
  const values = () => [
    screen.getByLabelText('Edge grains').value, screen.getByLabelText('ECD from').value, screen.getByLabelText('Phase').value,
  ];

  it('starts as the Grain tab: its edge rule and ECD basis, and a single phase there becomes All phases', () => {
    useResultStore.setState({ phaseOptions: [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }] });
    grain({ edgeRule: 'mtex', ecdBasis: 'measured_pixels', statsPhase: 2 });
    mount();
    expect(values()).toEqual(['mtex', 'measured_pixels', 'all']);
  });

  it('starts at the phase with the most measured pixels when the Grain tab has it', () => {
    grain({ edgeRule: 'outer_row', ecdBasis: 'cell_area', statsPhase: 'most_measured' });
    mount();
    expect(values()).toEqual(['outer_row', 'cell_area', 'most_measured']);
  });

  it('a change survives leaving the tab and does not change the Grain tab; later Grain-tab changes do not move it', () => {
    grain({ edgeRule: 'mtex', ecdBasis: 'measured_pixels', statsPhase: null });
    const first = mount();
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'keep_all' } });
    fireEvent.change(screen.getByLabelText('Phase'), { target: { value: 'most_measured' } });
    first.unmount();
    grain({ edgeRule: 'outer_row' });                       // the Grain tab moves meanwhile
    mount();
    expect(values()).toEqual(['keep_all', 'measured_pixels', 'most_measured']);
    const g = useResultStore.getState().grainStatsSettings;
    expect([g.edgeRule, g.ecdBasis, g.statsPhase]).toEqual(['outer_row', 'measured_pixels', null]);
  });

  it('is not reset when another map is loaded, but a full clear puts it back', () => {
    grain({ edgeRule: 'mtex' });
    const view = mount();
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    view.unmount();
    act(() => { useResultStore.getState().resetGrainStatsSettings(); });       // what a load on the Data tab does
    mount();
    expect(screen.getByLabelText('ECD from').value).toBe('measured_pixels');
    cleanup();
    act(() => { useResultStore.getState().clearResults(); });
    expect(useResultStore.getState().batchStatsSettings).toBeNull();
  });

  it('states the preset and the tolerance angle of the texture step, as set on the Texture tab', () => {
    mount();
    expect(screen.getByTestId('batch-texture-settings-line').textContent)
      .toBe('Texture components as set on the Texture tab: preset FCC_Rolling, tolerance 10°');
    cleanup();
    useResultStore.setState({ texturePreset: 'BCC_Rolling', textureTolerance: '4.5' });
    mount();
    expect(screen.getByTestId('batch-texture-settings-line').textContent)
      .toBe('Texture components as set on the Texture tab: preset BCC_Rolling, tolerance 4.5°');
  });

  it('sends them with the batch', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    useResultStore.setState({ texturePreset: 'BCC_Rolling', textureTolerance: '4.5' });
    mount();
    startBatch();
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch.mock.calls[0][7]).toEqual({ preset: 'BCC_Rolling', toleranceDeg: 4.5 });
  });

  it('leaves the tolerance out when the field holds no number: the batch uses its own 10 degrees, and the line says so', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    useResultStore.setState({ texturePreset: 'BCC_Rolling', textureTolerance: '' });
    mount();
    expect(screen.getByTestId('batch-texture-settings-line').textContent).toMatch(/preset BCC_Rolling, tolerance 10°$/);
    startBatch();
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch.mock.calls[0][7]).toEqual({ preset: 'BCC_Rolling', toleranceDeg: undefined });
  });

  it('says in one line that the texture is of the phase with the most indexed pixels of each file', () => {
    mount();
    expect(screen.getByTestId('batch-texture-line').textContent)
      .toBe('Texture: the phase with the most indexed pixels of each file');
  });
});

describe('BatchTab: the minimum intercept is recorded only', () => {
  const run = async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    mount();
    startBatch();
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    return analysisApi.batch.mock.calls[0][5];
  };

  it.each(['', '-2', 'abc'])('"%s" does not block the batch: the line says it is not given and nothing is sent', async (typed) => {
    useResultStore.getState().setGrainStatsSettings({ minIntercept: typed });
    const recon = await run();
    expect(screen.getByTestId('batch-recon-line').textContent)
      .toBe('Grain reconstruction as set in Grain Analysis: 4.0°, 3 px, alpha 3.1, fill on; min intercept not given');
    expect(recon.minIntercept).toBeUndefined();
    expect(recon.threshold).toBe(4);
  });

  it('0 is a value: it is shown and sent', async () => {
    useResultStore.getState().setGrainStatsSettings({ minIntercept: '0' });
    const recon = await run();
    expect(screen.getByTestId('batch-recon-line').textContent).toMatch(/; min intercept 0 µm \(recorded only, not applied\)$/);
    expect(recon.minIntercept).toBe(0);
  });
});

describe('BatchTab: the phase sentences follow the batch\'s own edge rule', () => {
  const OPTION = 'most_measured';
  const PREFIX = 'In each file, the phase with the most measured pixels (ties: the lower phase number); ';
  const optionTitle = () => Array.from(screen.getByLabelText('Phase').options).find((o) => o.value === OPTION).title;

  it.each([
    ['matlab_script', "Statistics over the grains of this phase only. With the MATLAB-script rule the border strip is drawn around this phase's own extent, so grains of a minority phase next to other phases also count as edge grains."],
    ['mtex', 'Statistics over the grains of this phase only. The edge rule looks at the whole map: a grain next to another phase is not an edge grain.'],
    ['outer_row', 'Statistics over the grains of this phase only. The edge rule looks at the whole map: a grain next to another phase is not an edge grain.'],
    ['keep_all', 'Statistics over the grains of this phase only.'],
  ])('under %s the most-measured option carries the whole sentence', (rule, sentence) => {
    mount();
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: rule } });
    expect(optionTitle()).toBe(PREFIX + sentence);
  });
});

describe('BatchTab: the typed dataset name is trimmed', () => {
  it('sends the name without the blanks around it', async () => {
    window.electronAPI = { saveFile: vi.fn() };
    analysisApi.exportExcel.mockResolvedValue({ data: { message: 'ok' } });
    mount();
    fireEvent.change(screen.getByPlaceholderText('Documentation.xlsx'), { target: { value: 'C:/out/Doc.xlsx' } });
    fireEvent.change(screen.getByLabelText('Dataset name:'), { target: { value: '  mine ' } });
    fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
    await waitFor(() => expect(analysisApi.exportExcel).toHaveBeenCalledTimes(1));
    expect(analysisApi.exportExcel).toHaveBeenLastCalledWith('C:/out/Doc.xlsx', 'mine');
  });
});

describe('BatchTab: the dataset name of the export', () => {
  const nameField = () => screen.getByLabelText('Dataset name:');

  it("starts as the loaded file's name, without folder and extension", () => {
    useResultStore.setState({ sourceName: 'Sample 7050 AB1.h5oina' });
    mount();
    expect(nameField().value).toBe('Sample 7050 AB1');
  });

  it('is "Dataset" when no file name is known', () => {
    mount();
    expect(nameField().value).toBe('Dataset');
  });

  it('follows a file loaded later, until the user types', () => {
    mount();
    act(() => { useResultStore.setState({ sourceName: 'a.ctf' }); });
    expect(nameField().value).toBe('a');
    fireEvent.change(nameField(), { target: { value: 'mine' } });
    act(() => { useResultStore.setState({ sourceName: 'b.ctf' }); });
    expect(nameField().value).toBe('mine');                        // what the user typed wins
  });

  it('sends the typed name, and falls back to the file name when the field is emptied', async () => {
    window.electronAPI = { saveFile: vi.fn() };
    analysisApi.exportExcel.mockResolvedValue({ data: { message: 'ok' } });
    useResultStore.setState({ sourceName: 'scan_01.h5oina' });
    mount();
    fireEvent.change(screen.getByPlaceholderText('Documentation.xlsx'), { target: { value: 'C:/out/Doc.xlsx' } });
    fireEvent.change(nameField(), { target: { value: 'typed' } });
    fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
    await waitFor(() => expect(analysisApi.exportExcel).toHaveBeenCalledTimes(1));
    expect(analysisApi.exportExcel).toHaveBeenLastCalledWith('C:/out/Doc.xlsx', 'typed');
    await waitFor(() => expect(screen.getByRole('button', { name: /download excel/i }).disabled).toBe(false));
    fireEvent.change(nameField(), { target: { value: '  ' } });
    fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
    await waitFor(() => expect(analysisApi.exportExcel).toHaveBeenCalledTimes(2));
    expect(analysisApi.exportExcel).toHaveBeenLastCalledWith('C:/out/Doc.xlsx', 'scan_01');
  });
});

describe('BatchTab: a map without band contrast', () => {
  const mapCalls = () => analysisApi.getMap.mock.calls.map((c) => c[0]);
  const layerSelect = () => screen.getByLabelText('Layer:');

  it('starts on the map with band contrast as it did', async () => {
    useResultStore.setState({ hasNativeBc: true });
    mount();
    await waitFor(() => expect(mapCalls().length).toBeGreaterThan(0));
    expect(mapCalls()[0]).toBe('bc');
    expect(layerSelect().value).toBe('bc');
    expect(within(layerSelect()).getByRole('option', { name: 'Band Contrast' })).toBeTruthy();
  });

  it('does not know yet: asks for band contrast as before', async () => {
    mount();
    await waitFor(() => expect(mapCalls().length).toBeGreaterThan(0));
    expect(mapCalls()[0]).toBe('bc');
  });

  it('starts on IPF-Z and never asks for a band contrast it does not have', async () => {
    useResultStore.setState({ hasNativeBc: false });
    mount();
    await waitFor(() => expect(mapCalls().length).toBeGreaterThan(0));
    expect(mapCalls()).not.toContain('bc');
    expect(mapCalls()[0]).toBe('ipf');
    expect(layerSelect().value).toBe('ipf');
    // the option stays, and says why it will not draw
    expect(within(layerSelect()).getByRole('option', { name: 'Band Contrast (not in this file)' })).toBeTruthy();
  });

  it('moves to IPF-Z when the load that says so arrives after the tab is open', async () => {
    mount();
    await waitFor(() => expect(mapCalls()[0]).toBe('bc'));
    act(() => { useResultStore.setState({ hasNativeBc: false }); });
    await waitFor(() => expect(layerSelect().value).toBe('ipf'));
  });

  it('what the user picks stays, and the backend\'s sentence is shown when it refuses', async () => {
    useResultStore.setState({ hasNativeBc: false });
    mount();
    await waitFor(() => expect(mapCalls().length).toBeGreaterThan(0));
    analysisApi.getMap.mockRejectedValue({ response: { status: 400, data: { detail: 'This file has no band contrast; choose another layer.' } } });
    fireEvent.change(layerSelect(), { target: { value: 'bc' } });
    await waitFor(() => expect(screen.getByText(/This file has no band contrast; choose another layer\./)).toBeTruthy());
    expect(layerSelect().value).toBe('bc');
  });
});

describe('BatchTab: the RX criteria of the batch', () => {
  it('says what it will use, from the Recrystallization tab', () => {
    mount();
    expect(screen.getByTestId('batch-rx-line').textContent)
      .toBe('RX criteria as set in Recrystallization: GOS < 1.25°, gBC > 0.7 × largest gBC, gKAM < 0.55°, radius > 0.4 µm RX phase: the phase with the most indexed pixels of each file.');
  });

  it('follows the Recrystallization tab and sends exactly those numbers', async () => {
    useResultStore.getState().setRxCriteria({ gosThreshold: '2', gbcFraction: '0.5', gkamThreshold: '0.8', minRadius: '1.5' });
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    mount();
    expect(screen.getByTestId('batch-rx-line').textContent)
      .toBe('RX criteria as set in Recrystallization (typed, not yet applied to the open map): GOS < 2°, gBC > 0.5 × largest gBC, gKAM < 0.8°, radius > 1.5 µm RX phase: the phase with the most indexed pixels of each file.');
    startBatch();
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch.mock.calls[0][6]).toEqual({ gosThreshold: 2, gbcFraction: 0.5, gkamThreshold: 0.8, minRadius: 1.5 });
  });

  it('does not start with a criterion that is empty or outside its range, and says why', () => {
    useResultStore.getState().setRxCriteria({ gbcFraction: '1.5' });
    mount();
    expect(screen.getByTestId('batch-rx-line').textContent).toMatch(/RX criteria in Recrystallization are incomplete or out of range/);
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(true);
  });

  it('"Run with" names the criteria the batch ran with, not the ones typed since', async () => {
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    useResultStore.getState().adoptRxCriteriaUsed({ rx_gos_threshold: 2 });          // the open map has them
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({
      data: { status: 'completed', progress: 1, result: { n_files: 1, n_processed: 1, processed: ['a.h5'], failed: [], skipped: [], warnings: [] } },
    });
    mount();
    startBatch();
    await waitFor(() => expect(screen.getByTestId('batch-ran-with')).toBeTruthy(), POLL_TIMEOUT);
    useResultStore.getState().setRxCriteria({ gosThreshold: '9' });
    expect(screen.getByTestId('batch-ran-with').textContent).toMatch(/ · RX: GOS < 2°, gBC > 0\.7, gKAM < 0\.55°, radius > 0\.4 µm · Deformation: KAM order 2, KAM threshold 7°, HAGB from 15°$/);
  }, 15000);
});

describe('BatchTab: typed RX criteria that the open map does not have yet', () => {
  it('marks the line, and "Run with" of a batch started in that state', async () => {
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({
      data: { status: 'completed', progress: 1, result: { n_files: 1, n_processed: 1, processed: ['a.h5'], failed: [], skipped: [], warnings: [] } },
    });
    mount();
    expect(screen.getByTestId('batch-rx-line').textContent)
      .toBe('RX criteria as set in Recrystallization (typed, not yet applied to the open map): GOS < 2°, gBC > 0.7 × largest gBC, gKAM < 0.55°, radius > 0.4 µm RX phase: the phase with the most indexed pixels of each file.');
    startBatch();
    await waitFor(() => expect(screen.getByTestId('batch-ran-with')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByTestId('batch-ran-with').textContent)
      .toMatch(/ · RX: GOS < 2°, gBC > 0\.7, gKAM < 0\.55°, radius > 0\.4 µm \(typed, not yet applied to the open map\) · Deformation: KAM order 2, KAM threshold 7°, HAGB from 15°$/);
  }, 15000);

  it('shows no mark once the backend holds what is typed', () => {
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    useResultStore.getState().adoptRxCriteriaUsed({ rx_gos_threshold: 2 });
    mount();
    expect(screen.getByTestId('batch-rx-line').textContent).not.toMatch(/not yet applied/);
  });
});

describe('BatchTab: the export of the open map uses the typed RX criteria', () => {
  const download = () => ({ data: new Blob(['xlsx bytes']), headers: {} });
  const clickDownload = () => fireEvent.click(screen.getByRole('button', { name: /download excel/i }));

  function stubSaving() {
    URL.createObjectURL = vi.fn(() => 'blob:orienta-test');
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  }

  it('sends the RX classification with the typed values first, then exports', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    useResultStore.getState().setRxCriteria({ gosThreshold: '2', minRadius: '1.5' });
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_gos_threshold: 2, rx_gbc_fraction: 0.7, rx_gkam_threshold: 0.55, rx_min_radius_um: 1.5 } });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1);
    expect(analysisApi.rxClassify).toHaveBeenCalledWith({ gosThreshold: 2, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 1.5 }, 'most_measured');
    expect(analysisApi.rxClassify.mock.invocationCallOrder[0]).toBeLessThan(analysisApi.downloadExcel.mock.invocationCallOrder[0]);
    expect(useResultStore.getState().rxCriteriaEdited).toBe(false);          // the backend holds them now
  });

  it('says on the status line that the export applied the typed criteria first, and names them', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    useResultStore.getState().setRxCriteria({ gosThreshold: '2', minRadius: '1.5' });
    analysisApi.rxClassify.mockResolvedValue({ data: {} });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    expect(screen.getByTestId('export-rx-applied-region')).toBeTruthy();       // the live region is there before the text
    expect(screen.queryByTestId('export-rx-applied')).toBeNull();
    clickDownload();
    const line = await screen.findByTestId('export-rx-applied');
    expect(line.textContent).toBe('RX criteria applied before the export: GOS < 2°, gBC > 0.7 × largest gBC, gKAM < 0.55°, radius > 1.5 µm');
    expect(line.closest('[aria-live="polite"]')).toBe(screen.getByTestId('export-rx-applied-region'));
  });

  it('says nothing about RX when the export had nothing to apply', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(screen.queryByTestId('export-rx-applied')).toBeNull();
  });

  it('Electron: without an output path nothing is reclassified and nothing is exported', async () => {
    window.electronAPI = { saveFile: vi.fn() };
    useResultStore.setState({ hasGrains: true });
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Choose an output path first')).toBeTruthy());
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
    expect(analysisApi.exportExcel).not.toHaveBeenCalled();
    expect(useResultStore.getState().rxCriteriaEdited).toBe(true);           // still pending: nothing was sent
    expect(screen.queryByTestId('export-rx-applied')).toBeNull();
  });

  it('Electron: with a path the criteria go first, then the export, and the status line names them', async () => {
    window.electronAPI = { saveFile: vi.fn() };
    useResultStore.setState({ hasGrains: true });
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    analysisApi.rxClassify.mockResolvedValue({ data: {} });
    analysisApi.exportExcel.mockResolvedValue({ data: { message: 'Exported' } });
    mount();
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    clickDownload();
    await waitFor(() => expect(analysisApi.exportExcel).toHaveBeenCalledTimes(1));
    expect(analysisApi.rxClassify.mock.invocationCallOrder[0]).toBeLessThan(analysisApi.exportExcel.mock.invocationCallOrder[0]);
    expect((await screen.findByTestId('export-rx-applied')).textContent).toMatch(/GOS < 2°/);
  });

  it('sends nothing when nothing is pending', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
  });

  it('does not export a workbook made with other criteria than the typed ones when the backend refuses', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    analysisApi.rxClassify.mockRejectedValue({ response: { status: 409, data: { detail: 'The map changed while classifying; press Classify again.' } } });
    analysisApi.status.mockResolvedValue({ data: { loaded: false } });
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText(/press Classify again/)).toBeTruthy());
    expect(analysisApi.downloadExcel).not.toHaveBeenCalled();
    expect(analysisApi.status).toHaveBeenCalled();                          // the page asked the backend what it holds
  });

  it('does not export with a typed criterion that cannot be sent', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    useResultStore.getState().setRxCriteria({ gbcFraction: '7' });
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText(/Enter numbers inside the ranges/)).toBeTruthy());
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
    expect(analysisApi.downloadExcel).not.toHaveBeenCalled();
  });

  it('without grains there is no RX block to make: the export goes ahead and sends nothing', async () => {
    stubSaving();
    useResultStore.getState().setRxCriteria({ gosThreshold: '2' });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
  });
});

describe('BatchTab: progress', () => {
  it('shows the backend\'s fraction as percent, in the text and in the bar', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.6 } });
    mount();
    startBatch();
    await waitFor(() => expect(screen.getByText('60%')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('60');
  }, 15000);

  it('a batch task the backend forgot: "status unknown", not "failed"', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockRejectedValue({ response: { status: 404 } });
    mount();
    startBatch();
    await waitFor(() => expect(screen.getByText(/The batch's status is unknown/)).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByText('Batch processing failed')).toBeNull();
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(false);
  }, 15000);
});

describe('BatchTab: what a batch did', () => {
  const doneBatch = {
    status: 'completed', progress: 1.0,
    message: 'Processed 2 of 5 files: 1 skipped (nograins.h5 (no grains found)), 1 failed (bad.h5 (OSError: unreadable)), 1 analysis step(s) left out (a.h5 (rx: needs native band contrast in the file)).',
    result: {
      n_files: 5, n_processed: 2,
      processed: ['/data/a.h5', '/data/b.h5'],
      failed: [{ file: '/data/bad.h5', reason: 'OSError: unreadable' }, { file: '/data/exp.h5', reason: 'export: ValueError: Größe µm' }],
      skipped: [{ file: '/data/nograins.h5', reason: 'no grains found' }],
      warnings: [{ file: '/data/a.h5', reason: 'rx: needs native band contrast in the file' }],
    },
  };

  async function runBatchWith(taskPayload) {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: taskPayload });
    mount();
    startBatch();
    await waitFor(() => expect(screen.getByTestId('batch-result')).toBeTruthy(), POLL_TIMEOUT);
  }

  it('shows the message and the four lists with their reasons', async () => {
    await runBatchWith(doneBatch);
    // the status line and the log both carry the message of the task
    expect(screen.getAllByText(/Processed 2 of 5 files: 1 skipped/).length).toBeGreaterThan(0);
    const result = within(screen.getByTestId('batch-result'));
    expect(result.getByText('Processed files (2)')).toBeTruthy();
    expect(result.getByText('a.h5')).toBeTruthy();
    expect(result.getByText('b.h5')).toBeTruthy();
    expect(result.getByText('Failed files (2)')).toBeTruthy();
    expect(result.getByText('bad.h5: OSError: unreadable')).toBeTruthy();
    expect(result.getByText('exp.h5: export: ValueError: Größe µm')).toBeTruthy();
    expect(result.getByText('Skipped files (1)')).toBeTruthy();
    expect(result.getByText('nograins.h5: no grains found')).toBeTruthy();
    expect(result.getByText('Analysis steps left out (1)')).toBeTruthy();
    expect(result.getByText('a.h5: rx: needs native band contrast in the file')).toBeTruthy();
    // a batch with trouble does not end in the colour of a clean one
    const line = screen.getAllByRole('status').find((el) => /Processed 2 of 5 files/.test(el.textContent));
    expect(line.style.color).toBe(colors.orange);
    // the full path stays reachable: by hover, and by keyboard (a focusable element named with the path)
    const entry = result.getByText('bad.h5: OSError: unreadable');
    expect(entry.title).toBe('/data/bad.h5');
    expect(entry.tabIndex).toBe(0);
    expect(entry.getAttribute('aria-label')).toBe('bad.h5: OSError: unreadable (/data/bad.h5)');
    // the lists are in a polite live region
    expect(screen.getByTestId('batch-result').closest('[aria-live="polite"]')).toBeTruthy();
  }, 15000);

  it('names the files where the edge rule kept all grains, in one line', async () => {
    await runBatchWith({
      ...doneBatch,
      result: { ...doneBatch.result, edge_safeguard_files: ['a.h5', 'b.h5'] },
    });
    const line = screen.getByTestId('batch-safeguard-files');
    expect(line.textContent).toBe('The edge rule kept all grains in 2 of 2 files: a.h5, b.h5');
    // plain text: no tab stop for each name
    expect(line.querySelectorAll('[tabindex]').length).toBe(0);
  }, 15000);

  it('cuts a long list of such files with a toggle, one tab stop for the list', async () => {
    const names = Array.from({ length: 14 }, (_, i) => `f${String(i + 1).padStart(2, '0')}.h5`);
    await runBatchWith({
      ...doneBatch,
      result: { ...doneBatch.result, n_processed: 14, processed: names.map((n) => `/d/${n}`), edge_safeguard_files: names },
    });
    const line = screen.getByTestId('batch-safeguard-files');
    expect(line.textContent).toContain('in 14 of 14 files: f01.h5, f02.h5');
    expect(line.textContent).not.toContain('f14.h5');
    expect(line.querySelectorAll('button').length).toBe(1);
    fireEvent.click(within(line).getByRole('button', { name: 'Show all 14' }));
    expect(line.textContent).toContain('f14.h5');
  }, 15000);

  it('says nothing about the edge rule when it never kept all grains', async () => {
    await runBatchWith({ ...doneBatch, result: { ...doneBatch.result, edge_safeguard_files: [] } });
    expect(screen.queryByTestId('batch-safeguard-files')).toBeNull();
  }, 15000);

  it('a list that is empty is not shown', async () => {
    await runBatchWith({
      status: 'completed', progress: 1, message: 'Processed 1 of 1 files.',
      result: { n_files: 1, n_processed: 1, processed: ['/data/a.h5'], failed: [], skipped: [], warnings: [] },
    });
    const result = within(screen.getByTestId('batch-result'));
    expect(result.getByText('Processed files (1)')).toBeTruthy();
    const line = screen.getAllByRole('status').find((el) => /Processed 1 of 1 files/.test(el.textContent));
    expect(line.style.color).toBe(colors.green);
    expect(result.queryByText(/Failed files/)).toBeNull();
    expect(result.queryByText(/Skipped files/)).toBeNull();
    expect(result.queryByText(/Analysis steps left out/)).toBeNull();
  }, 15000);

  it('a batch in which no file gave a result: the error text AND the lists', async () => {
    await runBatchWith({
      status: 'failed', progress: 1.0,
      error: 'None of the 2 files gave a result, no workbook was written. 1 failed: bad.h5 (OSError: unreadable); 1 skipped: nograins.h5 (no grains found)',
      result: {
        n_files: 2, n_processed: 0, processed: [],
        failed: [{ file: '/data/bad.h5', reason: 'OSError: unreadable' }],
        skipped: [{ file: '/data/nograins.h5', reason: 'no grains found' }], warnings: [],
      },
    });
    expect(screen.getByText(/None of the 2 files gave a result, no workbook was written\./)).toBeTruthy();
    const result = within(screen.getByTestId('batch-result'));
    expect(result.getByText('bad.h5: OSError: unreadable')).toBeTruthy();
    expect(result.getByText('nograins.h5: no grains found')).toBeTruthy();
    expect(result.queryByText(/Processed files/)).toBeNull();
    // and the button is free for another try
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(false);
  }, 15000);

  it('a workbook that could not be saved: the error and the files whose results were lost', async () => {
    await runBatchWith({
      status: 'failed', progress: 1.0,
      error: 'The workbook O.xlsx could not be written (PermissionError: [Errno 13] Permission denied); the results of 2 files were lost.',
      result: {
        n_files: 2, n_processed: 0, processed: [],
        failed: [{ file: '/data/a.h5', reason: 'not saved: PermissionError' }, { file: '/data/b.h5', reason: 'not saved: PermissionError' }],
        skipped: [], warnings: [], workbook_path: null,
        save_error: 'The workbook O.xlsx could not be written (PermissionError: [Errno 13] Permission denied); the results of 2 files were lost.',
      },
    });
    expect(screen.getByText(/could not be written \(PermissionError/)).toBeTruthy();
    const result = within(screen.getByTestId('batch-result'));
    expect(result.getByText('Failed files (2)')).toBeTruthy();
    expect(result.getByText('a.h5: not saved: PermissionError')).toBeTruthy();
  }, 15000);

  it('a batch that failed with no result at all (an exception in the run) shows the error and no lists', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1.0, error: 'PermissionError: O.xlsx is open in another program' } });
    mount();
    startBatch();
    await waitFor(() => expect(screen.getByText(/PermissionError: O\.xlsx is open/)).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByTestId('batch-result')).toBeNull();
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(false);
  }, 15000);

  it('the lists of an earlier run are gone when a new run starts', async () => {
    await runBatchWith(doneBatch);
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId('batch-result')).toBeNull();
  }, 15000);

  it('a long list shows ten entries and the rest on request', async () => {
    const processed = Array.from({ length: 13 }, (_, i) => `/data/scan_${String(i + 1).padStart(2, '0')}.h5`);
    await runBatchWith({
      status: 'completed', progress: 1, message: 'Processed 13 of 13 files.',
      result: { n_files: 13, n_processed: 13, processed, failed: [], skipped: [], warnings: [] },
    });
    const result = within(screen.getByTestId('batch-result'));
    expect(result.getAllByRole('listitem')).toHaveLength(10);
    expect(result.queryByText('scan_11.h5')).toBeNull();
    expect(result.getByRole('button', { name: 'Show all 13' }).getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(result.getByRole('button', { name: 'Show all 13' }));
    expect(result.getByRole('button', { name: 'Show fewer' }).getAttribute('aria-expanded')).toBe('true');
    expect(result.getAllByRole('listitem')).toHaveLength(13);
    expect(result.getByText('scan_13.h5')).toBeTruthy();
    fireEvent.click(result.getByRole('button', { name: 'Show fewer' }));
    expect(result.getAllByRole('listitem')).toHaveLength(10);
  }, 15000);

  it('exactly ten entries need no toggle', async () => {
    const processed = Array.from({ length: 10 }, (_, i) => `/data/s${i}.h5`);
    await runBatchWith({
      status: 'completed', progress: 1, message: 'done',
      result: { n_files: 10, n_processed: 10, processed, failed: [], skipped: [], warnings: [] },
    });
    const result = within(screen.getByTestId('batch-result'));
    expect(result.getAllByRole('listitem')).toHaveLength(10);
    expect(result.queryByRole('button')).toBeNull();
  }, 15000);
});

describe('BatchTab: Excel export warnings, Electron path', () => {
  const exportTo = () => {
    window.electronAPI = { saveFile: vi.fn() };
    mount();
    fireEvent.change(screen.getByPlaceholderText('Documentation.xlsx'), { target: { value: 'C:/out/Doc.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
  };

  it('polls the export task and lists result.warnings under the success message', async () => {
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus
      .mockResolvedValueOnce({ data: { status: 'running', progress: 0.3 } })
      .mockResolvedValue({
        data: {
          status: 'completed', progress: 1.0,
          result: {
            path: 'C:/out/Doc.xlsx', requested_path: 'C:/out/Doc.xlsx',
            warnings: ['texture: ValueError: Größe µm', 'bc_gmm: too few points for the fit'],
          },
        },
      });
    exportTo();
    await waitFor(() => expect(screen.getByText('Exported to C:/out/Doc.xlsx')).toBeTruthy(), POLL_TIMEOUT);
    expect(analysisApi.exportExcel).toHaveBeenCalledWith('C:/out/Doc.xlsx', 'Dataset');
    expect(analysisApi.getTaskStatus).toHaveBeenCalledWith('e1');
    const box = within(screen.getByTestId('export-warnings'));
    expect(box.getByText('texture: ValueError: Größe µm')).toBeTruthy();
    expect(box.getByText('bc_gmm: too few points for the fit')).toBeTruthy();
    expect(screen.getByTestId('export-warnings').textContent)
      .toMatch(/The workbook is incomplete: these steps failed or were left out, and are empty in it/);
    expect(screen.queryByTestId('export-renamed')).toBeNull();      // the file was written where it was asked for
  }, 15000);

  it('a clean export has no warning box', async () => {
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus.mockResolvedValue({
      data: { status: 'completed', progress: 1, result: { path: 'C:/out/Doc.xlsx', requested_path: 'C:/out/Doc.xlsx', warnings: [] } },
    });
    exportTo();
    await waitFor(() => expect(screen.getByText('Exported to C:/out/Doc.xlsx')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByTestId('export-warnings')).toBeNull();
    expect(screen.queryByTestId('export-renamed')).toBeNull();
  }, 15000);

  it('a new file on Windows: the backend writes C:\\o\\Doc.xlsx for the requested C:/o/Doc.xlsx, and that is no rename', async () => {
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus.mockResolvedValue({
      data: {
        status: 'completed', progress: 1,
        result: { path: 'C:\\o\\Doc.xlsx', requested_path: 'C:/o/Doc.xlsx', warnings: [] },
      },
    });
    exportTo();
    await waitFor(() => expect(screen.getByText('Exported to C:\\o\\Doc.xlsx')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByTestId('export-renamed')).toBeNull();
  }, 15000);

  it('a rename on Windows is still told, mixed slashes and all', async () => {
    window.electronAPI = { saveFile: vi.fn(), platform: 'win32' };
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus.mockResolvedValue({
      data: {
        status: 'completed', progress: 1,
        result: { path: 'C:\\o\\Doc_2.xlsx', requested_path: 'C:\\o/doc.xlsx', warnings: [] },
      },
    });
    mount();
    fireEvent.change(screen.getByPlaceholderText('Documentation.xlsx'), { target: { value: 'C:\\o/doc.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
    await waitFor(() => expect(screen.getByTestId('export-renamed')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByTestId('export-renamed').textContent).toContain('C:\\o\\Doc_2.xlsx');
  }, 15000);

  it('a file that already existed: the user is told it is the rule of the app, and where the workbook went', async () => {
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus.mockResolvedValue({
      data: {
        status: 'completed', progress: 1,
        result: { path: 'C:/out/Doc_2.xlsx', requested_path: 'C:/out/Doc.xlsx', warnings: [] },
      },
    });
    exportTo();
    await waitFor(() => expect(screen.getByTestId('export-renamed')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByTestId('export-renamed').textContent)
      .toBe('Orienta never overwrites an existing workbook: C:/out/Doc.xlsx already exists, so the workbook was written to C:/out/Doc_2.xlsx');
    // the live region is there before its text (it is the parent of the line, always rendered)
    expect(screen.getByTestId('export-renamed').parentElement.getAttribute('aria-live')).toBe('polite');
    expect(screen.getByText('Exported to C:/out/Doc_2.xlsx')).toBeTruthy();
  }, 15000);

  it('a failed export task says why and does not claim success', async () => {
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1, error: 'PermissionError: Doc.xlsx is open in another program' } });
    exportTo();
    await waitFor(() => expect(screen.getByText(/PermissionError: Doc\.xlsx is open/)).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByText(/Exported to/)).toBeNull();
    expect(screen.queryByTestId('export-warnings')).toBeNull();
    expect(screen.getByRole('button', { name: /download excel/i }).disabled).toBe(false);
  }, 15000);

  it('a task the backend forgot ends the wait with "status unknown", not "failed"', async () => {
    analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
    analysisApi.getTaskStatus.mockRejectedValue({ response: { status: 404 } });
    exportTo();
    await waitFor(() => expect(screen.getByText(/The export's status is unknown/)).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByText('Export failed')).toBeNull();
    expect(screen.getByRole('button', { name: /download excel/i }).disabled).toBe(false);
  }, 15000);

  it('an export that takes longer than five minutes is polled on until it ends', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      analysisApi.exportExcel.mockResolvedValue({ data: { task_id: 'e1' } });
      analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.5 } });
      exportTo();
      await vi.waitFor(() => expect(analysisApi.exportExcel).toHaveBeenCalled());
      await vi.advanceTimersByTimeAsync(330000);
      expect(analysisApi.getTaskStatus.mock.calls.length).toBeGreaterThan(600);
      expect(screen.queryByText('Export failed')).toBeNull();
      expect(screen.getByRole('button', { name: /exporting/i })).toBeTruthy();
      analysisApi.getTaskStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1, result: { path: 'C:/out/Doc.xlsx', requested_path: 'C:/out/Doc.xlsx', warnings: [] } },
      });
      await vi.advanceTimersByTimeAsync(1000);
      await vi.waitFor(() => expect(screen.getByText('Exported to C:/out/Doc.xlsx')).toBeTruthy());
    } finally {
      vi.useRealTimers();
    }
  }, 30000);
});

describe('BatchTab: Excel export warnings, browser path', () => {
  /** What axios gives for responseType 'blob' with the header the backend sets. */
  function download(warnings, { raw } = {}) {
    const headers = {};
    if (raw !== undefined) headers['x-orienta-warnings'] = raw;
    else if (warnings) headers['x-orienta-warnings'] = encodeURIComponent(JSON.stringify(warnings));
    return { data: new Blob(['xlsx bytes']), headers };
  }

  function stubSaving() {
    const saved = [];
    URL.createObjectURL = vi.fn(() => 'blob:orienta-test');
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function click() {
      saved.push({ href: this.href, download: this.download });
    });
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    return { saved, open };
  }

  const clickDownload = () => fireEvent.click(screen.getByRole('button', { name: /download excel/i }));

  it('saves the blob as a file (no window.open) and shows the warnings of the header, a non-ASCII reason included', async () => {
    const { saved, open } = stubSaving();
    const reasons = ['texture: ValueError: Größe µm; a, b', 'rx: needs native band contrast in the file'];
    analysisApi.downloadExcel.mockResolvedValue(download(reasons));
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.downloadExcel).toHaveBeenCalledWith('Dataset');
    expect(open).not.toHaveBeenCalled();
    expect(saved).toEqual([{ href: 'blob:orienta-test', download: 'EBSD_Analysis.xlsx' }]);
    // the object URL is given back after the click (not before: the download reads it)
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:orienta-test'), { timeout: 3000 });
    const box = within(screen.getByTestId('export-warnings'));
    reasons.forEach((r) => expect(box.getByText(r)).toBeTruthy());
  });

  it('reads the header the way the backend writes it (json.dumps with \\u escapes, then quote)', async () => {
    stubSaving();
    // urllib.parse.quote(json.dumps(["texture: Größe"]), safe="")
    analysisApi.downloadExcel.mockResolvedValue(download(null, { raw: '%5B%22texture%3A%20Gr%5Cu00f6%5Cu00dfe%22%5D' }));
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('texture: Größe')).toBeTruthy());
  });

  it('no header: the file is saved and there is no warning box', async () => {
    const { saved } = stubSaving();
    analysisApi.downloadExcel.mockResolvedValue(download(null));
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(saved).toHaveLength(1);
    expect(screen.queryByTestId('export-warnings')).toBeNull();
  });

  it('a header that cannot be read is said so, the file is still saved', async () => {
    const { saved } = stubSaving();
    analysisApi.downloadExcel.mockResolvedValue(download(null, { raw: '%E0%A4%A' }));
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByTestId('export-warnings')).toBeTruthy());
    expect(screen.getByText('The server sent warnings that could not be read.')).toBeTruthy();
    expect(saved).toHaveLength(1);
  });

  it('a failed request shows the detail that is inside the error blob', async () => {
    const { saved } = stubSaving();
    analysisApi.downloadExcel.mockRejectedValue({
      response: { status: 500, data: new Blob([JSON.stringify({ detail: 'No dataset loaded' })], { type: 'application/json' }) },
    });
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('No dataset loaded')).toBeTruthy());
    expect(saved).toHaveLength(0);
    expect(screen.queryByText('Download started')).toBeNull();
  });

  it('a failed request whose blob is not JSON says "Export failed"', async () => {
    stubSaving();
    analysisApi.downloadExcel.mockRejectedValue({ response: { status: 502, data: new Blob(['<html>bad gateway</html>']) } });
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Export failed')).toBeTruthy());
  });
});
