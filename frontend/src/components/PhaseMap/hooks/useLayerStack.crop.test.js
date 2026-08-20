// @vitest-environment jsdom
/**
 * The phase map's SE underlay must ask for the DATASET view — under a crop the
 * electron image has to be cut to the same physical region as the map above
 * it — and it must carry back the backend's verdict when it could not be.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: 'Zm9v' } })) },
  h5Api: { getElectronImage: vi.fn(), getEDSMap: vi.fn() },
  ebsdApi: {}, analysisApi: {},
}));
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 4, height: 4, close: () => {} })));
// The h5 branch keys black to alpha, which runs the bitmap through an
// OffscreenCanvas jsdom does not have. Without this the SE fetch fails at the
// decode and never reaches the bookkeeping under test.
class FakeOffscreenCanvas {
  constructor(w, h) { this.width = w; this.height = h; }
  getContext() {
    return {
      drawImage() {},
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData() {},
    };
  }
  transferToImageBitmap() { return { width: this.width, height: this.height, close: () => {} }; }
}
vi.stubGlobal('OffscreenCanvas', FakeOffscreenCanvas);

import { useLayerStack } from './useLayerStack';
import { h5Api } from '../../../services/api';

const CLEANUP = {};
const settle = () => act(async () => {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
});

// addLayer takes an ID; `se:` is the dynamic prefix the h5 branch reads.
const SE_ID = 'se:SE/Elektronenbild 1';

function mount() {
  const hook = renderHook(() =>
    useLayerStack({ cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null }));
  act(() => { hook.result.current.setSingleLayer('phase'); });
  act(() => { hook.result.current.addLayer(SE_ID); });
  return hook;
}

describe('useLayerStack — electron images and the crop', () => {
  beforeEach(() => {
    h5Api.getElectronImage.mockReset();
    h5Api.getElectronImage.mockResolvedValue({ data: { image: 'Zm9v' } });
  });

  it('requests the dataset scope, not the file', async () => {
    mount();
    await settle();
    expect(h5Api.getElectronImage).toHaveBeenCalledWith('SE/Elektronenbild 1', 'dataset');
  });

  it('keeps a "could not be cropped" verdict', async () => {
    const verdict = { cropped: false, exact: false, reason: 'no geometry' };
    h5Api.getElectronImage.mockResolvedValue({ data: { image: 'Zm9v', crop: verdict } });
    const { result } = mount();
    await settle();
    expect(result.current.cropStatus.get(SE_ID)).toEqual(verdict);
  });

  it('keeps a CLAMPED verdict too, though the image WAS cropped', async () => {
    // LayeredCanvas stretches every layer to the composite size, so a cut-out
    // that is no longer the window's aspect ratio moves every feature on it.
    const verdict = { cropped: true, exact: false, reason: 'clamped' };
    h5Api.getElectronImage.mockResolvedValue({ data: { image: 'Zm9v', crop: verdict } });
    const { result } = mount();
    await settle();
    expect(result.current.cropStatus.get(SE_ID)).toEqual(verdict);
  });

  it('records nothing for an image that did follow the crop', async () => {
    h5Api.getElectronImage.mockResolvedValue({
      data: { image: 'Zm9v', crop: { cropped: true, exact: true, reason: null } },
    });
    const { result } = mount();
    await settle();
    expect(result.current.cropStatus.has(SE_ID)).toBe(false);
  });

  it('drops the verdict when the layer goes away', async () => {
    h5Api.getElectronImage.mockResolvedValue({
      data: { image: 'Zm9v',
              crop: { cropped: false, exact: false, reason: 'no geometry' } },
    });
    const { result } = mount();
    await settle();
    expect(result.current.cropStatus.has(SE_ID)).toBe(true);
    act(() => { result.current.removeLayer(SE_ID); });
    expect(result.current.cropStatus.has(SE_ID)).toBe(false);
  });
});

// The EDS element layer sits in the SAME composite as the phase map above it,
// and LayeredCanvas scales every bitmap onto the first layer's native size.
// A file-scope element map under a cropped phase map is therefore not merely
// "extra data" — it is stretched onto the crop's grid and every feature on it
// is displaced, with no warning anywhere.
describe('useLayerStack — EDS element layers and the crop', () => {
  beforeEach(() => {
    h5Api.getEDSMap.mockReset();
    h5Api.getEDSMap.mockResolvedValue({ data: { image: 'Zm9v' } });
  });

  it('requests the dataset scope, not the file', async () => {
    const hook = renderHook(() =>
      useLayerStack({ cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null }));
    act(() => { hook.result.current.setSingleLayer('phase'); });
    act(() => { hook.result.current.addLayer('eds:Al'); });
    await settle();
    expect(h5Api.getEDSMap).toHaveBeenCalledWith('Al', 'hot', '', 'dataset');
  });
});
