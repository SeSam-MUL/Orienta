// @vitest-environment jsdom
/**
 * Per-phase participation in the classification.
 *
 * The rule that matters: only a STRICT subset narrows the run. An
 * all-selected list means "everything", and sending it would freeze the
 * request against a library the user may since have extended.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { StrictMode } from 'react';
import { renderHook, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: vi.fn(() => Promise.resolve({ data: { loaded: true, summary: [] } })),
    getPhaseMap: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
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

/**
 * Opening the picker asks again, because the library can grow under it.
 *
 * The backend has read `Database/CIF_Library` itself since 2026-09-12, so a
 * downloaded CIF is a phase at once — but this list was fetched once per file,
 * and a CIF downloaded mid-session was missing from the checkboxes until the
 * file was reloaded.
 */
describe('the picker refreshes when it is opened', () => {
  const FOUR = {
    data: {
      phases: [
        { key: 'a', cif_filename: 'Al.cif', formula: 'Al' },
        { key: 'b', cif_filename: 'Si.cif', formula: 'Si' },
        { key: 'c', cif_filename: 'sd_0302719.cif', formula: 'AlFeMnSi' },
        { key: 'd', cif_filename: 'Mg2Si.cif', formula: 'Mg2Si' },
      ],
    },
  };

  it('picks up a CIF downloaded after the first load', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    expect(result.current.cifPhases).toHaveLength(3);

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);           // the download
    await act(async () => { await result.current.refreshCifPhases(); });

    expect(result.current.cifPhases.map(p => p.key)).toEqual(['a', 'b', 'c', 'd']);
    // A phase that appeared since takes part by default, like every other.
    expect(result.current.selectedPhaseKeys.has('d')).toBe(true);
  });

  it('keeps a deselection the user made before opening it', async () => {
    // The whole reason the refresh cannot reuse `loadCifPhases`: that one ticks
    // everything, which would overrule the user seconds after they chose.
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    act(() => { result.current.setSelectedPhaseKeys(new Set(['a'])); });

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);
    await act(async () => { await result.current.refreshCifPhases(); });

    // 'a' as chosen, 'd' because it is new; 'b' and 'c' stay deselected.
    expect([...result.current.selectedPhaseKeys].sort()).toEqual(['a', 'd']);
    // ...and a strict subset still narrows the run, with the new phase in it.
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(edsApi.autoClassify.mock.calls.at(-1)[0].phaseKeys.sort())
      .toEqual(['a', 'd']);
  });

  it('leaves an explicit "none" empty instead of re-ticking everything', async () => {
    // At refresh time the list has been shown, so empty means the user pressed
    // "Select none" — the clearest instruction there is.
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    act(() => { result.current.setSelectedPhaseKeys(new Set()); });

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);
    await act(async () => { await result.current.refreshCifPhases(); });

    expect(result.current.selectedPhaseKeys.size).toBe(0);
    expect(result.current.cifPhases).toHaveLength(4);       // shown, not ticked
  });

  it('drops a phase that has gone from the library', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();

    edsApi.cifPhases.mockResolvedValueOnce({
      data: { phases: [{ key: 'a', cif_filename: 'Al.cif', formula: 'Al' }] },
    });
    await act(async () => { await result.current.refreshCifPhases(); });

    expect(result.current.cifPhases).toHaveLength(1);
    expect([...result.current.selectedPhaseKeys]).toEqual(['a']);
  });

  it('keeps the list on screen when the refresh call fails', async () => {
    // A refresh that empties the picker on a dropped request would be worse
    // than no refresh: the phases are still there, only the answer is missing.
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();

    edsApi.cifPhases.mockRejectedValueOnce(new Error('backend restarted'));
    await act(async () => { await result.current.refreshCifPhases(); });

    expect(result.current.cifPhases).toHaveLength(3);
    expect(result.current.selectedPhaseKeys.size).toBe(3);
  });

  it('adds the new phase even when React double-invokes the updater', async () => {
    // The reason `appeared` is computed BEFORE `setSelectedPhaseKeys` and the ref
    // is written by an effect instead of inside the updater. React 19 invokes an
    // updater twice under StrictMode (and `main.jsx` wraps the app in it), so a
    // ref written in one would make the second pass see the new keys as already
    // known and drop them. Nothing else in this file renders under StrictMode,
    // which is exactly why the claim needs its own test.
    const { result } = renderHook(() => usePhaseMap({}), { wrapper: StrictMode });
    await flush();
    expect(result.current.cifPhases).toHaveLength(3);

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);
    await act(async () => { await result.current.refreshCifPhases(); });

    expect(result.current.cifPhases).toHaveLength(4);
    expect(result.current.selectedPhaseKeys.has('d')).toBe(true);
  });

  it('keeps a deselection under StrictMode too', async () => {
    const { result } = renderHook(() => usePhaseMap({}), { wrapper: StrictMode });
    await flush();
    act(() => { result.current.setSelectedPhaseKeys(new Set(['a'])); });

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);
    await act(async () => { await result.current.refreshCifPhases(); });

    expect([...result.current.selectedPhaseKeys].sort()).toEqual(['a', 'd']);
  });

  it('does not re-tick a deselected phase on a second refresh', async () => {
    // The trap in "new since last time": once a phase has been SHOWN, it is no
    // longer new, so a later refresh must not treat a deselection as absence.
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);
    await act(async () => { await result.current.refreshCifPhases(); });
    act(() => { result.current.setSelectedPhaseKeys(new Set(['a', 'b'])); });

    edsApi.cifPhases.mockResolvedValueOnce(FOUR);
    await act(async () => { await result.current.refreshCifPhases(); });

    expect([...result.current.selectedPhaseKeys].sort()).toEqual(['a', 'b']);
  });
});
