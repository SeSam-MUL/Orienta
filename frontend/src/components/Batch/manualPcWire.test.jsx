// @vitest-environment jsdom
/**
 * The Batch page's "set PC manually" dialog used to update only frontend state — the typed
 * PC never reached the backend, so a batch start still refused with a placeholder-PC error
 * despite the UI showing "PC set". This holds the real wiring: the dialog converts non-Bruker
 * numbers through pcApi.convert (never a second formula here) and calls batchV2Api.setPC,
 * addressed by the file's own path, before touching the store. The real same-stem scenario
 * (two files, one set-pc call, only the addressed entry changes) is covered end to end on the
 * backend in tests/test_batch_same_stem.py; this is the frontend half of that wiring.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

vi.mock('../../services/api', () => ({
  batchV2Api: {
    setPC: vi.fn(() => Promise.resolve({ data: { success: true } })),
  },
  pcApi: {
    convert: vi.fn(() => Promise.resolve({ data: { pc: [0.51, 0.34, 0.84] } })),
  },
}));

import { batchV2Api, pcApi } from '../../services/api';
import BatchFileTree from './BatchFileTree';
import useBatchConfigStore from '../../stores/useBatchConfigStore';

function seed(fileOverrides = {}) {
  useBatchConfigStore.setState({
    files: [{
      file_path: 'D:/A/Scan5.up1', file_name: 'Scan5.up1', ready: false, pc_status: 'none',
      pc_value: null, grid_kind: 'square', pattern_shape: [60, 60],
      convention: 'tsl', convention_known: true,
      preprocessing_config: null, eds_available: false, eds_elements: [], roi_configured: false,
      config: { phases: [] },
      ...fileOverrides,
    }],
    activeFileIndex: 0,
  });
}

const open = () => fireEvent.click(screen.getByText('fileTree.setPcManually'));
const apply = () => fireEvent.click(screen.getByText('common:apply'));

describe('Batch page: manual PC dialog reaches the backend', () => {
  afterEach(cleanup);

  it('preselects the file\'s own convention, converts, and calls set-pc with the Bruker numbers', async () => {
    vi.clearAllMocks();
    seed();
    render(<BatchFileTree />);
    open();
    expect(screen.getByTitle('hoverTips.pcConventionSelect').value).toBe('tsl'); // preselected from the file

    fireEvent.change(screen.getByTitle('hoverTips.pcXInput'), { target: { value: '0.52' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcYInput'), { target: { value: '0.65' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcZInput'), { target: { value: '0.64' } });
    apply();

    await waitFor(() => expect(batchV2Api.setPC).toHaveBeenCalled());
    expect(pcApi.convert).toHaveBeenCalledWith([0.52, 0.65, 0.64], 'tsl', 'bruker', [60, 60]);
    expect(batchV2Api.setPC).toHaveBeenCalledWith('D:/A/Scan5.up1', [0.51, 0.34, 0.84]);
  });

  it('does not convert, and sends the typed numbers as-is, when the convention is Bruker', async () => {
    vi.clearAllMocks();
    seed({ convention: 'bruker', convention_known: false });
    render(<BatchFileTree />);
    open();
    fireEvent.change(screen.getByTitle('hoverTips.pcXInput'), { target: { value: '0.5' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcYInput'), { target: { value: '0.5' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcZInput'), { target: { value: '0.7' } });
    apply();

    await waitFor(() => expect(batchV2Api.setPC).toHaveBeenCalled());
    expect(pcApi.convert).not.toHaveBeenCalled();
    expect(batchV2Api.setPC).toHaveBeenCalledWith('D:/A/Scan5.up1', [0.5, 0.5, 0.7]);
  });

  it('shows the real backend error and does not pretend the PC was set', async () => {
    vi.clearAllMocks();
    batchV2Api.setPC.mockRejectedValueOnce({
      response: { data: { detail: 'not in the CalibrationStore' } },
    });
    seed({ convention: 'bruker', convention_known: false });
    render(<BatchFileTree />);
    open();
    fireEvent.change(screen.getByTitle('hoverTips.pcXInput'), { target: { value: '0.5' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcYInput'), { target: { value: '0.5' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcZInput'), { target: { value: '0.7' } });
    apply();

    await screen.findByText('not in the CalibrationStore');
    expect(useBatchConfigStore.getState().files[0].pc_status).toBe('none');  // not silently flipped
  });

  it('converts the existing PC to the preselected convention for display — never shows raw Bruker numbers under a different label', async () => {
    vi.clearAllMocks();
    // TSL<->Bruker: y flips as 1-y (the real convention, see test fixtures elsewhere in this
    // codebase), so a round trip through both directions is a meaningful check, not a stub.
    pcApi.convert.mockImplementation((pc) => Promise.resolve({ data: { pc: [pc[0], 1 - pc[1], pc[2]] } }));
    seed({ pc_value: [0.55, 0.40, 0.70] });
    render(<BatchFileTree />);
    fireEvent.click(screen.getByText('fileTree.setPcManually'));

    // Shown as TSL numbers (y = 1-0.40 = 0.60), not the raw Bruker 0.40 under a "TSL" label.
    await waitFor(() => expect(screen.getByTitle('hoverTips.pcYInput').value).toBe('0.6000'));
    expect(screen.getByTitle('hoverTips.pcXInput').value).toBe('0.5500');

    apply(); // no edits: applying the shown TSL numbers must round-trip to the ORIGINAL Bruker PC
    await waitFor(() => expect(batchV2Api.setPC).toHaveBeenCalled());
    const [, sentPc] = batchV2Api.setPC.mock.calls[0];
    expect(sentPc[0]).toBeCloseTo(0.55, 6);
    expect(sentPc[1]).toBeCloseTo(0.40, 6);
    expect(sentPc[2]).toBeCloseTo(0.70, 6);
  });

  it('reconverts the displayed numbers when the convention is changed, instead of relabeling them', async () => {
    vi.clearAllMocks();
    pcApi.convert.mockImplementation((pc) => Promise.resolve({ data: { pc: [pc[0], 1 - pc[1], pc[2]] } }));
    seed({ convention: 'bruker', convention_known: false }); // dialog opens on Bruker, no PC yet
    render(<BatchFileTree />);
    open();
    fireEvent.change(screen.getByTitle('hoverTips.pcXInput'), { target: { value: '0.55' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcYInput'), { target: { value: '0.40' } });
    fireEvent.change(screen.getByTitle('hoverTips.pcZInput'), { target: { value: '0.70' } });

    fireEvent.change(screen.getByTitle('hoverTips.pcConventionSelect'), { target: { value: 'tsl' } });

    await waitFor(() => expect(screen.getByTitle('hoverTips.pcYInput').value).toBe('0.6000'));
    expect(screen.getByTitle('hoverTips.pcXInput').value).toBe('0.5500');
  });

  it('refuses locally on a non-numeric field, without calling the backend', async () => {
    vi.clearAllMocks();
    seed({ convention: 'bruker', convention_known: false });
    render(<BatchFileTree />);
    open();
    fireEvent.change(screen.getByTitle('hoverTips.pcXInput'), { target: { value: 'abc' } });
    apply();

    expect(await screen.findByText('fileTree.msgInvalidPc')).toBeTruthy();
    expect(batchV2Api.setPC).not.toHaveBeenCalled();
  });
});
