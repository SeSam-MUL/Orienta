// @vitest-environment jsdom
/**
 * A crop is a new DATASET cut from the file that is already open. The EDS page
 * used to key everything on the file: `useDefaultLayers` re-probed only on
 * [isFileOpen, filePath], and `useEdsLayerStack` re-seeded only when the
 * `initialLayers` REFERENCE changed. A crop moves neither, and every page in
 * this app stays mounted, so the bitmaps were warm long before the crop was
 * drawn.
 *
 * The result: the page composited the PARENT's full-scan maps while `shapeRef`
 * (its interaction grid, taken from those bitmaps) was the parent's — and
 * /eds/probe, /linescan and /region-stats resolved those very coordinates on
 * the CROP's grid. A hover inside the crop returned another pixel's chemistry,
 * with no error.
 *
 * These tests hold the identity in all three places.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';

const TINY_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';

vi.mock('../../../services/api', () => ({
  edsApi: { elements: vi.fn(), getMap: vi.fn() },
  ebsdApi: { datasets: vi.fn(), bandContrast: vi.fn(), virtualBSE: vi.fn() },
  h5Api: { getElectronList: vi.fn(), getElectronImage: vi.fn() },
  phaseMapApi: { layer: vi.fn() },
}));
vi.mock('../../../stores/useEdsColorStore', () => {
  const state = { colors: {}, getColor: () => '' };
  const useStore = (sel) => (sel ? sel(state) : state);
  useStore.getState = () => state;
  return { default: useStore };
});

import { edsApi, ebsdApi, h5Api } from '../../../services/api';
import { useActiveDatasetKey, DATASET_UNKNOWN } from './useActiveDatasetKey';
import { useDefaultLayers } from './useDefaultLayers';
import { useEdsLayerStack } from './useEdsLayerStack';

globalThis.createImageBitmap = vi.fn().mockResolvedValue(
  { close: vi.fn(), width: 40, height: 30 });
globalThis.fetch = vi.fn().mockResolvedValue({ blob: async () => ({}) });

beforeEach(() => {
  vi.clearAllMocks();
  ebsdApi.datasets.mockResolvedValue({ data: { active: 'Scan1' } });
  edsApi.elements.mockResolvedValue({ data: { elements: ['Al Ka1'] } });
  edsApi.getMap.mockResolvedValue({ data: { image: TINY_PNG, shape: [30, 40] } });
  h5Api.getElectronList.mockResolvedValue({ data: { images: [] } });
  ebsdApi.bandContrast.mockResolvedValue({ data: { image: TINY_PNG, label: 'native' } });
});

// ---------------------------------------------------------------------------

describe('useActiveDatasetKey', () => {
  it('is DATASET_UNKNOWN with no file open, and never probes', async () => {
    const { result } = renderHook(() => useActiveDatasetKey(false, null, true));
    expect(result.current).toBe(DATASET_UNKNOWN);
    expect(ebsdApi.datasets).not.toHaveBeenCalled();
  });

  it('resolves to the backend active dataset name', async () => {
    const { result } = renderHook(() => useActiveDatasetKey(true, '/f.h5oina', true));
    await waitFor(() => expect(result.current).toBe('Scan1'));
  });

  it('re-asks when the page becomes visible, which is when a crop happened', async () => {
    const { result, rerender } = renderHook(
      ({ active }) => useActiveDatasetKey(true, '/f.h5oina', active),
      { initialProps: { active: false } });
    await waitFor(() => expect(result.current).toBe('Scan1'));

    // The user crops on the EBSD viewer while this page is hidden.
    ebsdApi.datasets.mockResolvedValue({ data: { active: 'Scan1_crop1' } });
    rerender({ active: true });
    await waitFor(() => expect(result.current).toBe('Scan1_crop1'));
  });

  it('degrades to null (which still probes) when the read fails', async () => {
    ebsdApi.datasets.mockRejectedValue(new Error('backend down'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { result } = renderHook(() => useActiveDatasetKey(true, '/f.h5oina', true));
    await waitFor(() => expect(result.current).toBeNull());
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

// ---------------------------------------------------------------------------

describe('useDefaultLayers - dataset identity', () => {
  it('does not probe until the dataset is known', async () => {
    renderHook(() => useDefaultLayers(true, '/f.h5oina', DATASET_UNKNOWN));
    await act(async () => { await Promise.resolve(); });
    expect(edsApi.elements).not.toHaveBeenCalled();
  });

  it('re-probes when only the DATASET changes, the file staying put', async () => {
    const { result, rerender } = renderHook(
      ({ key }) => useDefaultLayers(true, '/f.h5oina', key),
      { initialProps: { key: 'Scan1' } });
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(edsApi.elements).toHaveBeenCalledTimes(1);
    const firstLayers = result.current.layers;

    rerender({ key: 'Scan1_crop1' });
    await waitFor(() => expect(edsApi.elements).toHaveBeenCalledTimes(2));
    // A fresh array, so the stack below re-seeds too.
    await waitFor(() => expect(result.current.layers).not.toBe(firstLayers));
  });

  it('does not re-probe when nothing changed', async () => {
    const { result, rerender } = renderHook(
      ({ key }) => useDefaultLayers(true, '/f.h5oina', key),
      { initialProps: { key: 'Scan1' } });
    await waitFor(() => expect(result.current.ready).toBe(true));
    rerender({ key: 'Scan1' });
    await act(async () => { await Promise.resolve(); });
    expect(edsApi.elements).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------

describe('useEdsLayerStack - dataset identity', () => {
  // The SAME array on purpose: this is exactly the case the old reference-only
  // re-seed missed, and the case a crop produces if the probe is still in
  // flight when the identity changes.
  const LAYERS = [{
    id: 'eds-Al Ka1', kind: 'eds-element', element: 'Al Ka1',
    label: 'Al', visible: true, opacity: 1, blend: 'normal',
  }];

  it('flushes the previous dataset bitmaps when the identity changes', async () => {
    const { result, rerender } = renderHook(
      ({ key }) => useEdsLayerStack({
        initialLayers: LAYERS, displayMode: 'at_pct', datasetKey: key }),
      { initialProps: { key: 'Scan1' } });

    await waitFor(() => expect(result.current.bitmaps.size).toBe(1));
    expect(edsApi.getMap).toHaveBeenCalledTimes(1);

    // The crop: same file, same layer ids, same array - different dataset.
    rerender({ key: 'Scan1_crop1' });
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledTimes(2));
  });

  it('drops the interaction grid so it is re-taken from the new dataset', async () => {
    const { result, rerender } = renderHook(
      ({ key }) => useEdsLayerStack({
        initialLayers: LAYERS, displayMode: 'at_pct', datasetKey: key }),
      { initialProps: { key: 'Scan1' } });
    await waitFor(() => expect(result.current.shape).toEqual([30, 40]));

    // The crop maps are a different size - and it is `shape` that the probe,
    // linescan and region-stats coordinates are expressed in.
    edsApi.getMap.mockResolvedValue({ data: { image: TINY_PNG, shape: [12, 16] } });
    rerender({ key: 'Scan1_crop1' });
    await waitFor(() => expect(result.current.shape).toEqual([12, 16]));
  });

  it('does not flush when the identity is unchanged', async () => {
    const { result, rerender } = renderHook(
      ({ key }) => useEdsLayerStack({
        initialLayers: LAYERS, displayMode: 'at_pct', datasetKey: key }),
      { initialProps: { key: 'Scan1' } });
    await waitFor(() => expect(result.current.bitmaps.size).toBe(1));
    rerender({ key: 'Scan1' });
    await act(async () => { await Promise.resolve(); });
    expect(edsApi.getMap).toHaveBeenCalledTimes(1);
  });
});
