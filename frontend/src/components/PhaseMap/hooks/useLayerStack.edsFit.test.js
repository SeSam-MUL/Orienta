// @vitest-environment jsdom
/**
 * The phase map's EDS element layers and the spectrum fit.
 *
 * On a file with a spectrum cube every EDS map answers 503 until the spectra
 * are evaluated. An `eds:` layer added during the fit kept that refusal as its
 * error -- in the backend's English -- and the fetch loop never retries an
 * errored layer, so it stayed errored after the fit. Now a layer added while
 * the fit runs (or before its state is known) is not asked for at all -- a
 * refused request is a console error -- and says why in the page's language;
 * it is fetched when the state turns ready or none. A refusal that still
 * happens (the page did not know yet) reads in the page's language too, and
 * the layer is fetched again when the fit turns ready (`edsFitReadyCount`).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}|${o.percent ?? ''}` : k) }),
}));
vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: 'Zm9v' } })) },
  h5Api: { getElectronImage: vi.fn(), getEDSMap: vi.fn() },
  ebsdApi: {}, analysisApi: {},
}));
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 4, height: 4, close: () => {} })));
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
const EDS_ID = 'eds:Fe Kα1';
const settle = () => act(async () => {
  for (let i = 0; i < 4; i += 1) await new Promise((r) => setTimeout(r, 0));
});
const running = () => Object.assign(new Error('Request failed with status code 503'), {
  response: {
    status: 503,
    data: { detail: { code: 'eds_spectrum_fit_running', progress: 0.3,
                      message: 'The spectra of this map are still being evaluated (30 %).' } },
  },
});
const failed = () => Object.assign(new Error('Request failed with status code 500'), {
  response: {
    status: 500,
    data: { detail: { code: 'eds_spectrum_fit_failed', error: 'cube unreadable',
                      message: 'The spectrum fit failed: cube unreadable.' } },
  },
});

function mount() {
  const hook = renderHook((props) => useLayerStack({
    cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, ...props,
  }), { initialProps: { edsFitReadyCount: 0 } });
  act(() => { hook.result.current.setSingleLayer('phase'); });
  act(() => { hook.result.current.addLayer(EDS_ID); });
  return hook;
}

describe('useLayerStack — EDS layers during the spectrum fit', () => {
  beforeEach(() => { h5Api.getEDSMap.mockReset(); });

  it('says the spectra are being evaluated, in the page\'s language', async () => {
    h5Api.getEDSMap.mockRejectedValue(running());
    const { result } = mount();
    await settle();
    expect(result.current.errors.get(EDS_ID)).toBe('eds:spectrumFit.running|30');
  });

  it('an errored layer is fetched again when the fit turns ready', async () => {
    h5Api.getEDSMap.mockRejectedValueOnce(running())
      .mockResolvedValue({ data: { image: 'Zm9v' } });
    const hook = mount();
    await settle();
    expect(hook.result.current.errors.has(EDS_ID)).toBe(true);
    expect(h5Api.getEDSMap).toHaveBeenCalledTimes(1);
    await settle();                       // no retry loop while not ready
    expect(h5Api.getEDSMap).toHaveBeenCalledTimes(1);

    hook.rerender({ edsFitReadyCount: 1 });
    await settle();
    expect(h5Api.getEDSMap).toHaveBeenCalledTimes(2);
    expect(hook.result.current.errors.has(EDS_ID)).toBe(false);
    expect(hook.result.current.bitmaps.has(EDS_ID)).toBe(true);
  });

  it('a refusal that answers after the fit turned ready does not strand the layer', async () => {
    let reject;
    h5Api.getEDSMap
      .mockReturnValueOnce(new Promise((_res, rej) => { reject = rej; }))
      .mockResolvedValue({ data: { image: 'Zm9v' } });
    const hook = mount();
    await settle();
    hook.rerender({ edsFitReadyCount: 1 });   // ready while the 503 is in flight
    await settle();
    await act(async () => { reject(running()); });
    await settle();
    expect(hook.result.current.errors.has(EDS_ID)).toBe(false);
    expect(h5Api.getEDSMap).toHaveBeenCalledTimes(2);
    expect(hook.result.current.bitmaps.has(EDS_ID)).toBe(true);
  });

  it('a failed fit stays errored, with its own sentence', async () => {
    h5Api.getEDSMap.mockRejectedValue(failed());
    const { result } = mount();
    await settle();
    expect(result.current.errors.get(EDS_ID)).toMatch(/^eds:spectrumFit.failed/);
  });
});

describe('useLayerStack — EDS layers wait for the fit instead of asking', () => {
  beforeEach(() => { h5Api.getEDSMap.mockReset(); });

  it('asks nothing while the fit runs, says so, and fetches when it is ready', async () => {
    h5Api.getEDSMap.mockResolvedValue({ data: { image: 'Zm9v' } });
    const hook = renderHook((props) => useLayerStack({
      cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, ...props,
    }), { initialProps: { edsFitState: 'running', edsFitProgress: 0.3, edsFitReadyCount: 0 } });
    act(() => { hook.result.current.setSingleLayer('phase'); });
    act(() => { hook.result.current.addLayer(EDS_ID); });
    await settle();
    expect(h5Api.getEDSMap).not.toHaveBeenCalled();
    expect(hook.result.current.errors.get(EDS_ID)).toBe('eds:spectrumFit.running|30');

    hook.rerender({ edsFitState: 'ready', edsFitProgress: 1, edsFitReadyCount: 1 });
    await settle();
    expect(h5Api.getEDSMap).toHaveBeenCalledTimes(1);
    expect(hook.result.current.errors.has(EDS_ID)).toBe(false);
    expect(hook.result.current.bitmaps.has(EDS_ID)).toBe(true);
  });

  it('the waiting sentence follows the fit\'s progress', async () => {
    h5Api.getEDSMap.mockResolvedValue({ data: { image: 'Zm9v' } });
    const hook = renderHook((props) => useLayerStack({
      cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, ...props,
    }), { initialProps: { edsFitState: null, edsFitProgress: 0, edsFitReadyCount: 0 } });
    act(() => { hook.result.current.setSingleLayer('phase'); });
    act(() => { hook.result.current.addLayer(EDS_ID); });
    await settle();
    expect(hook.result.current.errors.get(EDS_ID)).toBe('eds:spectrumFit.checking');

    hook.rerender({ edsFitState: 'running', edsFitProgress: 0.3, edsFitReadyCount: 0 });
    await settle();
    expect(hook.result.current.errors.get(EDS_ID)).toBe('eds:spectrumFit.running|30');

    hook.rerender({ edsFitState: 'running', edsFitProgress: 0.7, edsFitReadyCount: 0 });
    await settle();
    expect(hook.result.current.errors.get(EDS_ID)).toBe('eds:spectrumFit.running|70');
    expect(h5Api.getEDSMap).not.toHaveBeenCalled();
  });

  it('a real error is not overwritten by the waiting sentence', async () => {
    h5Api.getEDSMap.mockRejectedValue(failed());
    const hook = renderHook((props) => useLayerStack({
      cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, ...props,
    }), { initialProps: { edsFitState: 'failed', edsFitReadyCount: 0 } });
    act(() => { hook.result.current.setSingleLayer('phase'); });
    act(() => { hook.result.current.addLayer(EDS_ID); });
    await settle();
    expect(hook.result.current.errors.get(EDS_ID)).toMatch(/^eds:spectrumFit.failed/);

    hook.rerender({ edsFitState: 'running', edsFitProgress: 0.5, edsFitReadyCount: 0 });
    await settle();
    expect(hook.result.current.errors.get(EDS_ID)).toMatch(/^eds:spectrumFit.failed/);
  });

  it('before the state is known it says "checking"; a file without a cube then fetches', async () => {
    h5Api.getEDSMap.mockResolvedValue({ data: { image: 'Zm9v' } });
    const hook = renderHook((props) => useLayerStack({
      cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, ...props,
    }), { initialProps: { edsFitState: null } });
    act(() => { hook.result.current.setSingleLayer('phase'); });
    act(() => { hook.result.current.addLayer(EDS_ID); });
    await settle();
    expect(h5Api.getEDSMap).not.toHaveBeenCalled();
    expect(hook.result.current.errors.get(EDS_ID)).toBe('eds:spectrumFit.checking');

    hook.rerender({ edsFitState: 'none' });
    await settle();
    expect(h5Api.getEDSMap).toHaveBeenCalledTimes(1);
    expect(hook.result.current.bitmaps.has(EDS_ID)).toBe(true);
  });

  it('other layers are not held back by the fit', async () => {
    const { phaseMapApi } = await import('../../../services/api');
    phaseMapApi.layer.mockClear();
    const hook = renderHook(() => useLayerStack({
      cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, edsFitState: 'running',
    }));
    act(() => { hook.result.current.setSingleLayer('phase'); });
    await settle();
    expect(phaseMapApi.layer).toHaveBeenCalled();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
  });
});

describe('the Phase Map page hands the fit to the stack', () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const src = readFileSync(resolve(here, '..', 'PhaseMapPage.jsx'), 'utf8');
  it('watches the open file\'s fit and passes its ready count', () => {
    expect(src).toMatch(/useSpectrumFitStatus\(isFileOpen \? openFilePath : null/);
    expect(src).toMatch(/const edsFitReadyCount = useFitReadyCount\(edsFit\.state\)/);
    const call = src.slice(src.indexOf('useLayerStack({'), src.indexOf('});', src.indexOf('useLayerStack({')));
    expect(call).toMatch(/\bedsFitReadyCount\b/);
    expect(call).toMatch(/edsFitState: isFileOpen \? edsFit\.state : 'none'/);
  });
});
