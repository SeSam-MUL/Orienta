// @vitest-environment jsdom
//
// Display of missing numbers and wording: a number the backend does not give is "not available" with the reason the page
// knows (never 0, never nothing); "Classify" runs a Deformation value that is typed and not run yet first; a missing
// component fraction or plane value shows a dash without a percent sign or a 0.0; the batch shows "File 2 of 5: name" and
// the step size note in a list of its own. Real components, only the HTTP layer mocked.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    getMap: vi.fn(), rxClassify: vi.fn(), deformation: vi.fn(), texture: vi.fn(), textureComponents: vi.fn(),
    status: vi.fn(), getTaskStatus: vi.fn(), batch: vi.fn(),
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore, { EMPTY_BATCH_RUN } from '../../../stores/useResultStore';
import { RXTab, TextureTab, BatchTab } from '../AnalysisPage';

afterEach(() => { cleanup(); vi.resetAllMocks(); useResultStore.getState().clearResults(); });

const state = () => useResultStore.getState();
const noMap = () => analysisApi.getMap.mockImplementation(() => new Promise(() => {}));

describe('RX tab: a missing number is "not available", with the reason', () => {
  const renderTab = () => {
    noMap();
    return render(<RXTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
  };
  const classify = () => fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));
  const metrics = () => document.querySelector('[style*="monospace"]').textContent;

  it('without band contrast: all four numbers say so', async () => {
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_needs_bc: true, total_grains: 30 } });
    renderTab();
    classify();
    await waitFor(() => expect(metrics()).toMatch(/not available/));
    for (const name of ['RX fraction', 'High BC fraction', 'Low gKAM fraction', 'Low KAM fraction']) {
      expect(metrics()).toContain(`${name}: not available (needs band contrast in the map)`);
    }
  });

  it('a phase without grain pixels: the four numbers are absent from the answer and the page says why, not 0', async () => {
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_grain_count: 0, total_grains: 12, rx_phase: { id: 2, name: 'Si' } } });
    renderTab();
    classify();
    await waitFor(() => expect(metrics()).toMatch(/not available/));
    expect(metrics()).toContain('RX fraction: not available (no grain pixel in the RX phase)');
    expect(metrics()).toContain('Low KAM fraction: not available (no grain pixel in the RX phase)');
    expect(metrics()).not.toMatch(/0\.0%/);
    expect(metrics()).toContain('RX Grains: 0/12');
    expect(screen.queryByText('RX Fraction')).toBeNull();          // and no badge
  });

  it('a number that is there is printed as before', async () => {
    analysisApi.rxClassify.mockResolvedValue({
      data: { rx_fraction: 12.34, high_bc_fraction: 5, low_gkam_fraction: 6, low_kam_fraction: 7, rx_grain_count: 3, total_grains: 9 },
    });
    renderTab();
    classify();
    await waitFor(() => expect(metrics()).toMatch(/RX Fraction:\s+12\.3%/));
    expect(metrics()).not.toMatch(/not available/);
  });
});

describe('RX tab: "Classify" runs a pending Deformation value first', () => {
  const renderTab = () => {
    noMap();
    return render(<RXTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
  };
  const classify = () => fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));

  it('sends the Deformation parameters, then the classification', async () => {
    useResultStore.setState({ hasGrains: true });
    state().setDeformationParams({ kamOrder: '4' });
    analysisApi.deformation.mockResolvedValue({ data: { deformation_params: { kam_order: 4, kam_threshold_deg: 7, hagb_min_deg: 15 } } });
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_fraction: 1 } });
    renderTab();
    classify();
    await waitFor(() => expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1));
    expect(analysisApi.deformation).toHaveBeenCalledWith({ kamOrder: 4, kamThreshold: 7, hagbMin: 15 });
    expect(analysisApi.deformation.mock.invocationCallOrder[0]).toBeLessThan(analysisApi.rxClassify.mock.invocationCallOrder[0]);
    expect(state().deformationEdited).toBe(false);
  });

  it('sends no Deformation request when nothing is pending', async () => {
    useResultStore.setState({ hasGrains: true });
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_fraction: 1 } });
    renderTab();
    classify();
    await waitFor(() => expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1));
    expect(analysisApi.deformation).not.toHaveBeenCalled();
  });

  it('a pending value that cannot be sent stops the classification and says why', async () => {
    useResultStore.setState({ hasGrains: true });
    state().setDeformationParams({ kamOrder: '9' });
    renderTab();
    classify();
    await waitFor(() => expect(screen.getByText(/KAM order a whole number 1–5/)).toBeTruthy());
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
  });
});

describe('Texture tab: a missing number is a dash', () => {
  it('a component without a fraction shows "—" and no percent sign; a plane without a value shows "—", not 0.0 %', async () => {
    noMap();
    analysisApi.texture.mockResolvedValue({
      data: {
        texture_phase: { id: 1, name: 'Aluminium' }, texture_preset: 'FCC_Rolling', texture_tolerance_deg: 10,
        components: [{ name: 'Brass', fraction: null }, { name: 'Goss', fraction: 12.34 }, { name: 'Cube', fraction: 0 }],
        surface_planes: { '{111}': null, '{100}': 3.04, '{110}': 0 },
        surface_planes_family: { '{111}': 40, '{100}': 20, '{110}': 10 },
      },
    });
    render(<TextureTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /analyze texture/i }));
    await waitFor(() => expect(screen.getByText('Brass')).toBeTruthy());
    const row = (name) => screen.getByText(name).parentElement;
    expect(row('Brass').textContent).toBe('Brass—');
    expect(row('Goss').textContent).toBe('Goss12.3%');
    expect(row('Cube').textContent).toBe('Cube0.0%');            // a zero is a value
    expect(screen.getByTestId('plane-{111}').textContent).toMatch(/\): —$/);
    expect(screen.getByTestId('plane-{111}').textContent).not.toMatch(/\(1 1 1\): 0\.0/);
    expect(screen.getByTestId('plane-{100}').textContent).toMatch(/: 3\.0 %$/);
    expect(screen.getByTestId('plane-{110}').textContent).toMatch(/: 0\.0 %$/);
  });
});

describe('Batch tab: wording', () => {
  const mount = () => {
    noMap();
    return render(<BatchTab analysisLoaded onComplete={() => {}} />);
  };

  it('the file the batch is at is one clean line; a message of another shape is shown as it is', () => {
    useResultStore.setState({ batchRun: { ...EMPTY_BATCH_RUN, running: true, taskId: 'b1', progress: 20, currentFile: 'Processing file 3/12: sample_A.h5oina' } });
    analysisApi.getTaskStatus.mockImplementation(() => new Promise(() => {}));
    mount();
    expect(screen.getByTestId('batch-current-file').textContent).toBe('File 3 of 12: sample_A.h5oina');
    cleanup();
    useResultStore.setState({ batchRun: { ...EMPTY_BATCH_RUN, running: true, taskId: 'b1', progress: 20, currentFile: 'Writing the workbook' } });
    mount();
    expect(screen.getByTestId('batch-current-file').textContent).toBe('Writing the workbook');
  });

  it('the step size note has its own list and is not counted under the analysis steps that were left out', () => {
    useResultStore.setState({
      batchRun: {
        ...EMPTY_BATCH_RUN,
        result: {
          n_processed: 2, processed: ['/d/a.h5', '/d/b.h5'], failed: [], skipped: [],
          warnings: [
            { file: '/d/a.h5', reason: 'step_size: step size not recorded in the file; 1.0 µm assumed' },
            { file: '/d/b.h5', reason: 'texture: ValueError: no cubic phase' },
          ],
        },
      },
    });
    mount();
    const box = within(screen.getByTestId('batch-result'));
    expect(box.getByText('Analysis steps left out (1)')).toBeTruthy();
    expect(box.getByText('b.h5: texture: ValueError: no cubic phase')).toBeTruthy();
    expect(box.queryByText(/a\.h5: step_size/)).toBeNull();
    expect(box.getByText('Step size not recorded in the file, 1.0 µm assumed (1)')).toBeTruthy();
  });
});
