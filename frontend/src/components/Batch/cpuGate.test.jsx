// @vitest-environment jsdom
/**
 * The Batch wizard defaults to the GPU spherical engine. On a machine without
 * CUDA that engine silently becomes torch-on-CPU — same numbers, one to two
 * orders of magnitude slower — and a batch is a whole folder of it.
 *
 * The Indexing page has guarded that since 2026-08-03 (estimate, notice,
 * confirm dialog). The wizard had none of it, so the default it now shares
 * would have been a trap. These tests hold the guard, at the two levels the
 * defect can live at:
 *
 *  1. the rule itself — one predicate, used by both screens;
 *  2. the wiring — the start button must not reach the API before the user has
 *     said yes. A rendered-but-unwired dialog is the failure this project keeps
 *     meeting (see tasks/lessons.md), and no predicate test can see it.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import {
  isSphericalCpuFallback,
  estimateCpuSphericalSeconds,
  formatRoughDuration,
} from '../Indexing/cpuEstimate';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
  Trans: ({ children }) => children ?? null,
}));

vi.mock('../../services/api', () => ({
  batchV2Api: {
    create: vi.fn(() => Promise.resolve({ data: { batch_id: 'b1', preflight: null } })),
    scanFolder: vi.fn(() => Promise.resolve({ data: { files: [] } })),
    quickLoad: vi.fn(() => Promise.resolve({ data: {} })),
    pcStatus: vi.fn(() => Promise.resolve({ data: {} })),
  },
  indexApi: {
    gpuStatus: vi.fn(() => Promise.resolve({ data: { device: 'cpu' } })),
    discoverFiles: vi.fn(() => Promise.resolve({ data: { files: [] } })),
  },
}));

vi.mock('./BatchFileTree', () => ({ default: () => null }));
vi.mock('./BatchDashboard', () => ({ default: () => null }));

import { batchV2Api, indexApi } from '../../services/api';
import BatchWorkflow from './BatchWorkflow';
import useBatchConfigStore from '../../stores/useBatchConfigStore';
import useBatchStore from '../../stores/useBatchStore';

describe('isSphericalCpuFallback — one rule for both screens', () => {
  const cuda = { device: 'cuda' };
  const cpu = { device: 'cpu' };

  it('fires for a GPU-engine spherical run on a machine without CUDA', () => {
    expect(isSphericalCpuFallback({
      method: 'spherical', backend: 'spherical_gpu', runtime: cpu,
    })).toBe(true);
  });

  it('stays quiet when CUDA is there', () => {
    expect(isSphericalCpuFallback({
      method: 'spherical', backend: 'spherical_gpu', runtime: cuda,
    })).toBe(false);
  });

  it('stays quiet for EMSphInx — a separate CPU program, not a fallback', () => {
    expect(isSphericalCpuFallback({
      method: 'spherical', backend: 'emsphinx', runtime: cpu,
    })).toBe(false);
  });

  it('stays quiet for Hough and Dictionary', () => {
    for (const method of ['hough', 'dictionary']) {
      expect(isSphericalCpuFallback({
        method, backend: 'spherical_gpu', runtime: cpu,
      })).toBe(false);
    }
  });

  it('raises no alarm before the runtime has answered', () => {
    expect(isSphericalCpuFallback({
      method: 'spherical', backend: 'spherical_gpu', runtime: null,
    })).toBe(false);
  });
});

describe('the estimate is the batch total, not one file', () => {
  it('scales with the whole folder', () => {
    const oneFile = estimateCpuSphericalSeconds(10800, 88);
    const fiveFiles = estimateCpuSphericalSeconds(5 * 10800, 88);
    expect(fiveFiles).toBeCloseTo(5 * oneFile, 6);
    expect(formatRoughDuration(fiveFiles)).toMatch(/h|min/);
  });
});

// ---------------------------------------------------------------------------

const readyFile = (name, rows = 90, cols = 120) => ({
  file_path: `D:/${name}.h5oina`,
  file_name: `${name}.h5oina`,
  ready: true,
  pc_status: 'header',
  grid_shape: [rows, cols],
  config: {
    phases: ['Al.sht'],
    backend: 'spherical_gpu',
    spherical_bandwidth: 88,
    auto_export: false,
    export_dir: '',
  },
});

function seedStore() {
  useBatchConfigStore.setState({
    files: [readyFile('a'), readyFile('b')],
    activeFileIndex: 0,
    currentStep: 6,
    method: 'spherical',
    globalPhases: [],
  });
  useBatchStore.setState({ batchId: null, preflight: null, totalJobs: 0 });
}

describe('the start button is gated, not just decorated', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    indexApi.gpuStatus.mockResolvedValue({ data: { device: 'cpu' } });
    seedStore();
  });

  // Without this every render stays in the document and the queries below
  // match the previous test's button as well as this one's.
  afterEach(cleanup);

  const clickStart = () => fireEvent.click(
    screen.getAllByText(/^step6\.startBatch/)[0]);

  it('asks before starting a CPU spherical batch, and does not call the API yet', async () => {
    render(<BatchWorkflow isActive />);
    await waitFor(() => expect(indexApi.gpuStatus).toHaveBeenCalled());

    clickStart();

    await screen.findByRole('dialog');
    expect(batchV2Api.create).not.toHaveBeenCalled();
  });

  it('starts once the user confirms', async () => {
    render(<BatchWorkflow isActive />);
    await waitFor(() => expect(indexApi.gpuStatus).toHaveBeenCalled());
    clickStart();
    await screen.findByRole('dialog');

    fireEvent.click(screen.getAllByText(/cpuConfirmStart/)[0]);
    await waitFor(() => expect(batchV2Api.create).toHaveBeenCalled());
  });

  it('does not ask when the machine has CUDA', async () => {
    indexApi.gpuStatus.mockResolvedValue({ data: { device: 'cuda' } });
    render(<BatchWorkflow isActive />);
    await waitFor(() => expect(indexApi.gpuStatus).toHaveBeenCalled());

    clickStart();

    await waitFor(() => expect(batchV2Api.create).toHaveBeenCalled());
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
