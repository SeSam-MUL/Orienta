// @vitest-environment jsdom
/**
 * Per-phase participation in the classification.
 *
 * The rule that matters: only a STRICT subset narrows the run. An
 * all-selected list means "everything", and sending it would freeze the
 * request against a library the user may since have extended.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: vi.fn(() => Promise.resolve({ data: { loaded: true, summary: [] } })),
    getPhaseMap: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
    cifPhases: vi.fn(() => Promise.resolve({
      data: {
        phases: [
          { key: 'a', cif_filename: 'Al.cif', formula: 'Al' },
          { key: 'b', cif_filename: 'Si.cif', formula: 'Si' },
          { key: 'c', cif_filename: 'sd_0302719.cif', formula: 'AlFeMnSi' },
        ],
      },
    })),
  },
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import { edsApi } from '../../services/api';
import { usePhaseMap } from './PhaseMapPanel';

const flush = () => act(async () => { await Promise.resolve(); });

beforeEach(() => { vi.clearAllMocks(); });

describe('per-phase participation', () => {
  it('loads the CIF phase list and starts with everything selected', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    expect(result.current.cifPhases).toHaveLength(3);
    expect(result.current.selectedPhaseKeys.size).toBe(3);
  });

  it('sends only the selected keys when a strict subset is chosen', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    act(() => { result.current.setSelectedPhaseKeys(new Set(['a'])); });
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].phaseKeys).toEqual(['a']);
  });

  it('omits phase_keys when every phase is selected', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].phaseKeys).toBeUndefined();
  });

  it('refuses to classify when nothing is selected', async () => {
    // Running the whole library here would invert the clearest instruction
    // the user can give. Earlier this silently sent no phase_keys at all,
    // which the backend reads as "use everything".
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    act(() => { result.current.setSelectedPhaseKeys(new Set()); });
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify).not.toHaveBeenCalled();
    expect(result.current.error).toBeTruthy();
  });

  it('defaults to cluster mode and passes it through', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    expect(result.current.mode).toBe('cluster');
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].mode).toBe('cluster');
  });

  it('passes an explicit cluster count when one is set', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    act(() => { result.current.setNClusters(5); });
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].nClusters).toBe(5);
  });

  it('leaves the cluster count null so the backend chooses by BIC', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    expect(result.current.nClusters).toBeNull();
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].nClusters).toBeNull();
  });

  it('survives a failing cif-phases call without breaking classification', async () => {
    edsApi.cifPhases.mockRejectedValueOnce(new Error('no database'));
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    expect(result.current.cifPhases).toEqual([]);
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify).toHaveBeenCalled();
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].phaseKeys).toBeUndefined();
  });
});
