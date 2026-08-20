// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { useEdsLayerStack } from './useEdsLayerStack';

// One-pixel base64 PNG (we don't care what it decodes to, just that
// createImageBitmap returns *something* on call).
const TINY_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';

vi.mock('../../../services/api', () => ({
  edsApi:  { getMap: vi.fn() },
  ebsdApi: { virtualBSE: vi.fn(), bandContrast: vi.fn() },
  h5Api:   { getElectronImage: vi.fn() },
  phaseMapApi: { layer: vi.fn() },
}));
vi.mock('../../../stores/useEdsColorStore', () => {
  let colors = { Al: '#8be9fd', Fe: '#ff5555' };
  const getColor = vi.fn((sym) => colors[sym] || '#bd93f9');
  // The real store is a zustand hook: callable with a selector AND carrying
  // a .getState(). useEdsLayerStack uses both forms (reactive subscription +
  // getState() at fetch time), so the mock must support both.
  const useStore = (selector) => {
    const state = { colors, getColor };
    return selector ? selector(state) : state;
  };
  useStore.getState = () => ({ colors, getColor });
  return {
    default: useStore,
    // Test helpers: the real store re-renders subscribers on setColor; in
    // tests we mutate the colour map (new reference) then rerender() manually.
    __setColor: (sym, hex) => { colors = { ...colors, [sym]: hex }; },
    __resetColors: () => { colors = { Al: '#8be9fd', Fe: '#ff5555' }; },
  };
});
import { edsApi, ebsdApi, h5Api } from '../../../services/api';
import { __setColor, __resetColors } from '../../../stores/useEdsColorStore';

// jsdom does not implement createImageBitmap; stub it.
globalThis.createImageBitmap = vi.fn().mockResolvedValue({ close: vi.fn(), width: 156, height: 128 });

beforeEach(() => {
  vi.clearAllMocks();
  edsApi.getMap.mockResolvedValue({ data: { image: TINY_PNG, shape: [128, 156] } });
  h5Api.getElectronImage.mockResolvedValue({ data: { image: TINY_PNG, shape: [128, 156] } });
  ebsdApi.virtualBSE.mockResolvedValue({ data: { image: TINY_PNG, shape: [128, 156] } });
  ebsdApi.bandContrast.mockResolvedValue({ data: { image: TINY_PNG, shape: [128, 156], label: 'native' } });
  globalThis.createImageBitmap = vi.fn().mockResolvedValue({ close: vi.fn(), width: 156, height: 128 });
});

describe('useEdsLayerStack', () => {
  it('seeds layers from initialLayers', () => {
    const initial = [
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    expect(result.current.layers.map(l => l.id)).toEqual(['bc']);
  });

  it('fetches visible layers and caches bitmaps', async () => {
    const initial = [
      { id: 'eds-Al Kα1', kind: 'eds-element', element: 'Al Kα1', visible: true, opacity: 0.7, blend: 'screen' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.has('eds-Al Kα1')).toBe(true));
    expect(edsApi.getMap).toHaveBeenCalledWith('Al Kα1', 'at_pct', 'gray', '#8be9fd');
  });

  it('invalidates EDS bitmaps when displayMode changes; keeps non-EDS', async () => {
    const initial = [
      { id: 'eds-Al', kind: 'eds-element', element: 'Al', visible: true, opacity: 0.7, blend: 'screen' },
      { id: 'bc',     kind: 'bc',          label: 'BC',  visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result, rerender } = renderHook(
      ({ mode }) => useEdsLayerStack({ initialLayers: initial, displayMode: mode }),
      { initialProps: { mode: 'at_pct' } },
    );
    await waitFor(() => expect(result.current.bitmaps.size).toBeGreaterThanOrEqual(1));
    const bcBefore = result.current.bitmaps.get('bc');
    rerender({ mode: 'wt_pct' });
    // EDS layer cache key is dropped; next fetch loop refetches with new mode.
    await waitFor(() => {
      const edsCalls = edsApi.getMap.mock.calls.map(c => c[1]);
      expect(edsCalls).toContain('wt_pct');
    });
    // BC bitmap reference preserved.
    expect(result.current.bitmaps.get('bc')).toBe(bcBefore);
  });

  it('removeLayer also drops the bitmap', async () => {
    const initial = [
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.has('bc')).toBe(true));
    act(() => { result.current.removeLayer('bc'); });
    expect(result.current.layers).toHaveLength(0);
    expect(result.current.bitmaps.has('bc')).toBe(false);
  });

  it('invalidateBitmapsForLayer drops cache for one id only', async () => {
    const initial = [
      { id: 'bc',   kind: 'bc',   label: 'BC',   visible: true, opacity: 0.5, blend: 'multiply' },
      { id: 'vbse', kind: 'vbse', label: 'V-BSE', visible: true, opacity: 0.5, blend: 'normal'   },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.size).toBe(2));
    act(() => { result.current.invalidateBitmapsForLayer('bc'); });
    // bc immediately gone; vbse intact.
    expect(result.current.bitmaps.has('vbse')).toBe(true);
    // bc will refetch on next effect — wait for it.
    await waitFor(() => expect(result.current.bitmaps.has('bc')).toBe(true));
  });

  it('puts an error in errors map for an unknown kind', async () => {
    const initial = [
      { id: 'mystery', kind: 'totally-unknown-kind', label: '?', visible: true, opacity: 1, blend: 'normal' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.errors.has('mystery')).toBe(true));
    expect(result.current.errors.get('mystery')).toMatch(/Unsupported/);
  });

  it('drops a stale EDS bitmap if displayMode changed during the fetch', async () => {
    // Make edsApi.getMap return a deferred promise so we can interleave a
    // mode-change between the call and its resolution. Both the initial at_pct
    // call and any wt_pct refetch must be controlled (otherwise the wt_pct
    // refetch could resolve first and cache an entry).
    let resolveAt;
    let resolveWt;
    edsApi.getMap
      .mockReturnValueOnce(new Promise(r => { resolveAt = r; }))
      .mockReturnValueOnce(new Promise(r => { resolveWt = r; }));

    const closeSpy = vi.fn();
    // Deferred-bitmap mock — we control exactly when the bitmap-decode await
    // resolves, so we can guarantee the mode-rerender happens FIRST.
    let resolveBitmap;
    globalThis.createImageBitmap = vi.fn().mockReturnValue(
      new Promise((r) => { resolveBitmap = r; }),
    );

    const initial = [
      { id: 'eds-Al', kind: 'eds-element', element: 'Al', visible: true, opacity: 0.7, blend: 'screen' },
    ];
    const { result, rerender } = renderHook(
      ({ mode }) => useEdsLayerStack({ initialLayers: initial, displayMode: mode }),
      { initialProps: { mode: 'at_pct' } },
    );
    // Wait for the first fetch to actually start.
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledTimes(1));
    // Resolve the network call so doFetch advances to the bitmap-decode await.
    resolveAt({ data: { image: TINY_PNG, shape: [128, 156] } });
    await waitFor(() => expect(globalThis.createImageBitmap).toHaveBeenCalled());
    // Switch mode while the bitmap decode is still pending. Sync rerender —
    // wrapping this in async-act would hang because act() would wait for the
    // pending bitmap promise that we haven't resolved yet.
    rerender({ mode: 'wt_pct' });
    // Finish the bitmap decode. The staleness guard in doFetch fires.
    resolveBitmap({ close: closeSpy, width: 156, height: 128 });
    // Poll until closeSpy has been called (the staleness branch ran).
    await waitFor(() => expect(closeSpy).toHaveBeenCalled());
    expect(result.current.bitmaps.has('eds-Al')).toBe(false);
    // Tidy up the dangling wt_pct promise (resolved after the rerender's
    // cache-flush triggered a refetch loop, then queued behind the in-flight
    // fetchingRef guard — it never actually got to start a real fetch).
    if (resolveWt) resolveWt({ data: { image: TINY_PNG, shape: [128, 156] } });
  });

  it('re-seeding with a fresh initialLayers (file switch) flushes stale bitmaps and refetches', async () => {
    // Regression: switching files keeps layer ids stable (`bc`), so the
    // fetch loop's `!cacheRef.has(l.id)` guard would skip the re-seeded layer
    // and keep rendering the PREVIOUS file's bitmap. The re-seed effect must
    // flush the whole cache so every layer refetches for the new file.
    const closeOld = vi.fn();
    const closeNew = vi.fn();
    globalThis.createImageBitmap = vi.fn()
      .mockResolvedValueOnce({ close: closeOld, width: 156, height: 128 })
      .mockResolvedValueOnce({ close: closeNew, width: 156, height: 128 });

    const fileA = [
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    // Same id, but a brand-new array reference == a different file.
    const fileB = [
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result, rerender } = renderHook(
      ({ layers }) => useEdsLayerStack({ initialLayers: layers, displayMode: 'at_pct' }),
      { initialProps: { layers: fileA } },
    );
    await waitFor(() => expect(result.current.bitmaps.has('bc')).toBe(true));
    expect(ebsdApi.bandContrast).toHaveBeenCalledTimes(1);

    rerender({ layers: fileB });
    // Old bitmap closed by the flush, then a fresh fetch for the new file.
    await waitFor(() => expect(closeOld).toHaveBeenCalled());
    await waitFor(() => expect(ebsdApi.bandContrast).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.bitmaps.has('bc')).toBe(true));
  });

  it('populates layerSources from the bandContrast response source', async () => {
    ebsdApi.bandContrast.mockResolvedValue({
      data: { image: TINY_PNG, shape: [128, 156], source: 'computed', label: 'Pattern Quality (computed)' },
    });
    const initial = [
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.has('bc')).toBe(true));
    await waitFor(() => expect(result.current.layerSources.get('bc')).toBe('computed'));
    // removeLayer must clear provenance so a later layer/file never shows stale source.
    act(() => { result.current.removeLayer('bc'); });
    expect(result.current.layerSources.has('bc')).toBe(false);
  });

  // --- crop scope ------------------------------------------------------
  it('asks for the DATASET view of an electron image, so it follows a crop', async () => {
    const initial = [
      { id: 'electron-SE1', kind: 'electron', electronName: 'SE/Elektronenbild 1',
        label: 'SE', visible: true, opacity: 1, blend: 'normal' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.has('electron-SE1')).toBe(true));
    expect(h5Api.getElectronImage).toHaveBeenCalledWith('SE/Elektronenbild 1', 'dataset');
  });

  it('keeps a "could not be cropped" verdict, and only that one', async () => {
    h5Api.getElectronImage.mockResolvedValue({
      data: { image: TINY_PNG, shape: [768, 1024],
              crop: { cropped: false, reason: 'no geometry' } },
    });
    const initial = [
      { id: 'electron-SE1', kind: 'electron', electronName: 'SE/Elektronenbild 1',
        label: 'SE', visible: true, opacity: 1, blend: 'normal' },
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.layerCropStatus.get('electron-SE1')).toEqual(
      { cropped: false, reason: 'no geometry' }));
    // A layer that DID follow the crop says nothing — "cropped" is the norm.
    expect(result.current.layerCropStatus.has('bc')).toBe(false);
    // Removing the layer takes its verdict with it.
    act(() => { result.current.removeLayer('electron-SE1'); });
    expect(result.current.layerCropStatus.has('electron-SE1')).toBe(false);
  });

  it('setThreshold attaches a threshold to the targeted layer', async () => {
    const initial = [
      { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 0.5, blend: 'multiply' },
    ];
    const { result } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.has('bc')).toBe(true));
    expect(typeof result.current.setThreshold).toBe('function');
    act(() => { result.current.setThreshold('bc', { min: 20, max: 200 }); });
    expect(result.current.layers.find(l => l.id === 'bc').threshold).toEqual({ min: 20, max: 200 });
  });

  it('closes all cached bitmaps on unmount', async () => {
    const close1 = vi.fn();
    const close2 = vi.fn();
    // Sequence the two bitmap mocks so we can assert both were closed.
    globalThis.createImageBitmap = vi.fn()
      .mockResolvedValueOnce({ close: close1, width: 156, height: 128 })
      .mockResolvedValueOnce({ close: close2, width: 156, height: 128 });

    const initial = [
      { id: 'bc',   kind: 'bc',   label: 'BC',   visible: true, opacity: 0.5, blend: 'multiply' },
      { id: 'vbse', kind: 'vbse', label: 'V-BSE', visible: true, opacity: 0.5, blend: 'normal'   },
    ];
    const { result, unmount } = renderHook(() => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }));
    await waitFor(() => expect(result.current.bitmaps.size).toBe(2));
    unmount();
    expect(close1).toHaveBeenCalled();
    expect(close2).toHaveBeenCalled();
  });

  it('refetches ONLY the EDS-element map whose colour changed', async () => {
    __resetColors();
    const initial = [
      { id: 'eds-Al', kind: 'eds-element', element: 'Al', visible: true, opacity: 0.7, blend: 'screen' },
      { id: 'eds-Fe', kind: 'eds-element', element: 'Fe', visible: true, opacity: 0.7, blend: 'screen' },
    ];
    const { result, rerender } = renderHook(
      () => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }),
    );
    await waitFor(() => expect(result.current.bitmaps.size).toBe(2));
    const feCallsBefore = edsApi.getMap.mock.calls.filter((c) => c[0] === 'Fe').length;

    // Change ONLY Al's colour. The mock has no subscription, so mutate then
    // rerender() to emulate zustand re-rendering subscribers.
    act(() => { __setColor('Al', '#123456'); });
    rerender();

    // Al refetched with the NEW colour baked into the request…
    await waitFor(() => {
      const alNew = edsApi.getMap.mock.calls.filter((c) => c[0] === 'Al' && c[3] === '#123456');
      expect(alNew.length).toBeGreaterThanOrEqual(1);
    });
    // …and Fe was NOT refetched by the colour change.
    const feCallsAfter = edsApi.getMap.mock.calls.filter((c) => c[0] === 'Fe').length;
    expect(feCallsAfter).toBe(feCallsBefore);
  });

  it('does not refetch when a colour is set to its current value (no-op)', async () => {
    __resetColors();
    const initial = [
      { id: 'eds-Al', kind: 'eds-element', element: 'Al', visible: true, opacity: 0.7, blend: 'screen' },
    ];
    const { result, rerender } = renderHook(
      () => useEdsLayerStack({ initialLayers: initial, displayMode: 'at_pct' }),
    );
    await waitFor(() => expect(result.current.bitmaps.has('eds-Al')).toBe(true));
    const alCallsBefore = edsApi.getMap.mock.calls.filter((c) => c[0] === 'Al').length;

    act(() => { __setColor('Al', '#8be9fd'); }); // same as default
    rerender();
    // Give the debounce window time to (not) fire.
    await new Promise((r) => setTimeout(r, 200));
    const alCallsAfter = edsApi.getMap.mock.calls.filter((c) => c[0] === 'Al').length;
    expect(alCallsAfter).toBe(alCallsBefore);
  });
});
