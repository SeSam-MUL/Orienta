// @vitest-environment jsdom
/**
 * setMethod (the store) silently clears picked phases when the method changes — a phase's
 * path is only valid for the method it was discovered under (see
 * useBatchConfigStore.methodStalePhases.test.js). Silent is the wrong word for something that
 * makes the user's phase picks disappear: this holds the visible notice that tells them why.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react';

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
    gpuStatus: vi.fn(() => Promise.resolve({ data: { device: 'cuda' } })),
    discoverFiles: vi.fn(() => Promise.resolve({ data: { files: [] } })),
  },
}));

vi.mock('./BatchFileTree', () => ({ default: () => null }));
vi.mock('./BatchDashboard', () => ({ default: () => null }));

import BatchWorkflow from './BatchWorkflow';
import useBatchConfigStore from '../../stores/useBatchConfigStore';
import useBatchStore from '../../stores/useBatchStore';

const NOTICE = 'step4.methodChangedClearedPhases';

function seedStoreWithPhases() {
  useBatchConfigStore.setState({
    files: [{
      file_path: 'D:/a.up1', file_name: 'a.up1', ready: true, pc_status: 'header',
      phases_configured: true,
      config: { phases: [{ name: 'Ti', path: 'beta-Ti.sht', method: 'spherical' }], method: 'spherical' },
    }],
    activeFileIndex: 0, currentStep: 4, method: 'spherical',
    globalPhases: [{ name: 'Ti', path: 'beta-Ti.sht', method: 'spherical' }],
  });
  useBatchStore.setState({ batchId: null, preflight: null, totalJobs: 0 });
}

describe('phases step: notice when a method change clears the phase picks', () => {
  afterEach(cleanup);

  it('shows the notice after switching method away from one that had phases applied', () => {
    seedStoreWithPhases();
    render(<BatchWorkflow isActive />);
    expect(screen.queryByText(NOTICE)).toBeNull();

    fireEvent.click(screen.getByText('methods.hough'));

    expect(screen.getByText(NOTICE)).toBeTruthy();
    expect(useBatchConfigStore.getState().globalPhases).toHaveLength(0);
  });

  it('disappears once phases are applied again under the new method', () => {
    seedStoreWithPhases();
    render(<BatchWorkflow isActive />);
    fireEvent.click(screen.getByText('methods.hough'));
    expect(screen.getByText(NOTICE)).toBeTruthy();

    act(() => {
      useBatchConfigStore.getState().applyGlobalPhases([
        { name: 'Ti', path: 'alpha-Ti.cif', method: 'hough' },
      ]);
    });

    expect(screen.queryByText(NOTICE)).toBeNull();
  });

  it('stays quiet on first render, and quiet for a same-method click', () => {
    seedStoreWithPhases();
    render(<BatchWorkflow isActive />);
    fireEvent.click(screen.getByText('methods.spherical')); // already the active method
    expect(screen.queryByText(NOTICE)).toBeNull();
  });

  it('stays quiet switching away from a method that had no phases applied', () => {
    useBatchConfigStore.setState({
      files: [{
        file_path: 'D:/a.up1', file_name: 'a.up1', ready: false, pc_status: 'header',
        phases_configured: false,
        config: { phases: [], method: 'spherical' },
      }],
      activeFileIndex: 0, currentStep: 4, method: 'spherical', globalPhases: [],
    });
    useBatchStore.setState({ batchId: null, preflight: null, totalJobs: 0 });
    render(<BatchWorkflow isActive />);

    fireEvent.click(screen.getByText('methods.hough'));

    expect(screen.queryByText(NOTICE)).toBeNull();
  });
});
