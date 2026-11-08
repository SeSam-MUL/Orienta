// @vitest-environment jsdom
//
// The Batch & Export tab: (1) the reconstruction fields are checked against the limits of the batch route before anything
// is sent, and a 422 that still comes back (a list as `detail`) shows a message instead of taking the page down;
// (2) the batch sends the Deformation tab's parameters, names the RX phase rule, and shows the file it is at;
// (3) the Excel export applies what is typed on the Deformation, Texture and RX tabs and not run yet, before it exports.
// Real component, only the HTTP layer mocked, with the shapes of the real answers.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    getTaskStatus: vi.fn(), getMap: vi.fn(), batch: vi.fn(), rxClassify: vi.fn(), deformation: vi.fn(), texture: vi.fn(),
    status: vi.fn(), exportExcel: vi.fn(), downloadExcel: vi.fn(),
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { BatchTab } from '../AnalysisPage';
import { batchReconstruction, reconstructArgs } from '../grainStats';

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.restoreAllMocks();
  delete window.electronAPI;
  useResultStore.getState().clearResults();
});

const state = () => useResultStore.getState();
const mount = () => {
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  return render(<BatchTab analysisLoaded onComplete={() => {}} />);
};
const fillPaths = () => {
  fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
  fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
};
const processButton = () => screen.getByRole('button', { name: /process folder/i });

describe('the reconstruction limits of the batch route', () => {
  it('batchReconstruction and reconstructArgs refuse what the route would answer with a 422', () => {
    const ok = { gbThreshold: '4', minPixels: '3', alpha: '3.1', minIntercept: '1.5', fill: true };
    expect(batchReconstruction(ok)).toMatchObject({ threshold: 4, minPixels: 3, alpha: 3.1 });
    for (const bad of [{ gbThreshold: '0' }, { gbThreshold: '-1' }, { gbThreshold: '' }, { minPixels: '0' }, { minPixels: '2.7' },
      { minPixels: '' }, { alpha: '25' }, { alpha: '-0.1' }, { alpha: 'x' }]) {
      expect(batchReconstruction({ ...ok, ...bad }), JSON.stringify(bad)).toBeNull();
      expect(reconstructArgs({ ...ok, ...bad }), JSON.stringify(bad)).toBeNull();
    }
    expect(batchReconstruction({ ...ok, alpha: '0' })).toMatchObject({ alpha: 0 });          // 0 closes nothing: a value
    expect(batchReconstruction({ ...ok, alpha: '20' })).toMatchObject({ alpha: 20 });
  });

  it.each([['minPixels', '0'], ['alpha', '25'], ['gbThreshold', '0']])(
    'a %s of %s says so on the batch line and the batch is not sent', (field, value) => {
      state().setGrainStatsSettings({ [field]: value });
      mount();
      fillPaths();
      expect(screen.getByTestId('batch-recon-line').textContent).toMatch(/incomplete or out of range/);
      expect(processButton().disabled).toBe(true);
      fireEvent.click(processButton());
      expect(analysisApi.batch).not.toHaveBeenCalled();
    },
  );
});

describe('a 422 from the batch', () => {
  it('shows a message made from the list in `detail`, and the page is still there', async () => {
    analysisApi.batch.mockRejectedValue({
      response: { status: 422, data: { detail: [{ type: 'missing', loc: ['body', 'folder_path'], msg: 'Field required', input: {} }] } },
    });
    mount();
    fillPaths();
    fireEvent.click(processButton());
    await waitFor(() => expect(screen.getByText('folder_path: Field required')).toBeTruthy());
    expect(processButton().disabled).toBe(false);                    // the tab is usable: a retry is possible
  });
});

describe('the batch and the Deformation / RX tabs', () => {
  it('sends the Deformation tab\'s parameters, and says them on the batch line', async () => {
    state().setDeformationParams({ kamOrder: '3', kamThreshold: '5', hagbMin: '12' });
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    mount();
    expect(screen.getByTestId('batch-deformation-line').textContent)
      .toBe('Deformation as set in Deformation: KAM order 3, KAM threshold 5°, HAGB from 12°');
    fillPaths();
    fireEvent.click(processButton());
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch.mock.calls[0][8]).toEqual({ kamOrder: 3, kamThreshold: 5, hagbMin: 12 });
  });

  it('refuses a Deformation parameter outside the range, and says so', () => {
    state().setDeformationParams({ kamOrder: '9' });
    mount();
    fillPaths();
    expect(screen.getByTestId('batch-deformation-line').textContent).toMatch(/incomplete or out of range/);
    expect(processButton().disabled).toBe(true);
  });

  it('names the RX phase rule in its RX line', () => {
    mount();
    expect(screen.getByTestId('batch-rx-line').textContent).toMatch(/RX phase: the phase with the most indexed pixels of each file\.$/);
  });

  it('shows the file the batch is at while it runs', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.3, current_file: 'Processing file 2/5: b.h5oina' } });
    mount();
    fillPaths();
    fireEvent.click(processButton());
    const line = await screen.findByTestId('batch-current-file', {}, { timeout: 6000 });
    expect(line.textContent).toBe('File 2 of 5: b.h5oina');
  }, 15000);
});

describe('the export of the open map applies what is typed and not run yet', () => {
  const download = () => ({ data: new Blob(['xlsx bytes']), headers: {} });
  const clickDownload = () => fireEvent.click(screen.getByRole('button', { name: /download excel/i }));
  function stubSaving() {
    URL.createObjectURL = vi.fn(() => 'blob:orienta-test');
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  }
  const order = (fn) => fn.mock.invocationCallOrder[0];

  it('runs the Deformation, the Texture and the RX request, in that order, then exports, and names what it applied', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true, phaseOptions: [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }] });
    state().setDeformationParams({ kamOrder: '3' });
    state().setTexturePreset('BCC_Rolling');
    state().setTextureTolerance('12');
    state().setRxPhase(2);
    analysisApi.deformation.mockResolvedValue({ data: { deformation_params: { kam_order: 3, kam_threshold_deg: 7, hagb_min_deg: 15 } } });
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.rxClassify.mockResolvedValue({ data: {} });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.deformation).toHaveBeenCalledWith({ kamOrder: 3, kamThreshold: 7, hagbMin: 15 });
    expect(analysisApi.texture).toHaveBeenCalledWith('most_measured', 'BCC_Rolling', 12);
    expect(analysisApi.rxClassify).toHaveBeenCalledWith({ gosThreshold: 1.25, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 0.4 }, 2);
    expect(order(analysisApi.deformation)).toBeLessThan(order(analysisApi.texture));
    expect(order(analysisApi.texture)).toBeLessThan(order(analysisApi.rxClassify));
    expect(order(analysisApi.rxClassify)).toBeLessThan(order(analysisApi.downloadExcel));
    expect(state()).toMatchObject({ deformationEdited: false, textureEdited: false, rxCriteriaEdited: false });
    expect(screen.getByTestId('export-deformation-applied').textContent)
      .toBe('Deformation parameters applied before the export: KAM order 3, KAM threshold 7°, HAGB from 15°');
    expect(screen.getByTestId('export-texture-applied').textContent)
      .toBe('Texture applied before the export: BCC_Rolling, tolerance 12°, phase: phase with the most indexed pixels');
  });

  it('sends nothing when nothing is pending', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.deformation).not.toHaveBeenCalled();
    expect(analysisApi.texture).not.toHaveBeenCalled();
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
    expect(screen.queryByTestId('export-texture-applied')).toBeNull();
    expect(screen.queryByTestId('export-deformation-applied')).toBeNull();
  });

  it('without grains the texture pick is still applied (it needs none); Deformation and RX wait for grains', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: false });
    state().setTexturePreset('BCC_Rolling');
    state().setDeformationParams({ kamOrder: '3' });
    state().setRxPhase(2);
    analysisApi.texture.mockResolvedValue({ data: {} });
    analysisApi.downloadExcel.mockResolvedValue(download());
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText('Download started')).toBeTruthy());
    expect(analysisApi.texture).toHaveBeenCalledWith('most_measured', 'BCC_Rolling', 10);
    expect(analysisApi.deformation).not.toHaveBeenCalled();
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
  });

  it('a pending texture tolerance that cannot be sent stops the export and says why', async () => {
    stubSaving();
    useResultStore.setState({ hasGrains: true });
    state().setTextureTolerance('45');
    mount();
    clickDownload();
    await waitFor(() => expect(screen.getByText(/more than 0° and at most 30°/i)).toBeTruthy());
    expect(analysisApi.downloadExcel).not.toHaveBeenCalled();
    expect(analysisApi.texture).not.toHaveBeenCalled();
  });
});
