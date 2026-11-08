// @vitest-environment jsdom
/**
 * On a hexagonal result every layer must be on the display raster. A layer that is not (an add-on map, an
 * analysis map, anything still on the index grid) would be drawn half a step off the others, so the hook
 * refuses it by name on that layer's row instead.
 *
 * A picture also belongs to the geometry it was fetched for: when the expected raster changes (another
 * result, dataset or measurement) no message about the old expectation survives, no picture of the old
 * geometry stays, and an answer that lands late is judged by the expectation of the moment it lands.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: 'Zm9v' } })) },
  h5Api: {}, ebsdApi: {}, analysisApi: {},
}));
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
let bitmapSize = { width: 14, height: 12 };
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ ...bitmapSize, close: () => {} })));

import { useLayerStack } from './useLayerStack';
import { phaseMapApi } from '../../../services/api';

const RASTER = [12, 14];
const INDEX_GRID = [6, 5];
const SENTENCE = 'This layer is not drawn on the hexagonal map grid, so it cannot be shown with this scan.';
const CLEANUP = {};
const settle = () => act(async () => {
  for (let k = 0; k < 4; k += 1) await new Promise((r) => setTimeout(r, 0));
});

function mount(expectedShape, shapePending = false, extra = {}) {
  const hook = renderHook((p) => useLayerStack({
    cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null,
    expectedShape: p.expectedShape, shapePending: p.shapePending,
    rasterMatches: p.rasterMatches, rasterId: p.rasterId,
  }), { initialProps: { expectedShape, shapePending, rasterMatches: null, rasterId: '', ...extra } });
  act(() => { hook.result.current.setSingleLayer('phase'); });
  return hook;
}
const set = (hook, props) => hook.rerender({
  expectedShape: null, shapePending: false, rasterMatches: null, rasterId: '', ...props,
});

beforeEach(() => {
  bitmapSize = { width: 14, height: 12 };
  createImageBitmap.mockClear();
  phaseMapApi.layer.mockClear();
  phaseMapApi.layer.mockImplementation(() => Promise.resolve({ data: { image: 'Zm9v' } }));
});

describe('useLayerStack with an expected shape', () => {
  it('keeps a layer that is exactly the expected raster', async () => {
    const hook = mount(RASTER);
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    expect(hook.result.current.errors.has('phase')).toBe(false);
  });

  it('refuses a layer of another shape by name and does not keep its bitmap', async () => {
    bitmapSize = { width: 5, height: 6 };                 // the index grid of the scan
    const hook = mount(RASTER);
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(false);
    expect(hook.result.current.errors.get('phase')).toBe(SENTENCE);
  });

  it('checks nothing without an expected shape', async () => {
    bitmapSize = { width: 5, height: 6 };
    const hook = mount(null);
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    expect(hook.result.current.errors.has('phase')).toBe(false);
  });
});

describe('a picture belongs to the geometry it was fetched for', () => {
  it('lifts the refusal when the expected raster changes to the one the picture is on', async () => {
    bitmapSize = { width: 5, height: 6 };
    const hook = mount(RASTER);
    await settle();
    expect(hook.result.current.errors.get('phase')).toBe(SENTENCE);

    set(hook, { expectedShape: INDEX_GRID });            // the geometry turned out to be this one
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
  });

  it('lifts the refusal when no raster is expected any more', async () => {
    bitmapSize = { width: 5, height: 6 };
    const hook = mount(RASTER);
    await settle();
    expect(hook.result.current.errors.get('phase')).toBe(SENTENCE);

    set(hook, { expectedShape: null });
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
  });

  it('keeps a picture that fits the new expectation without asking again', async () => {
    const hook = mount(null);                             // 14 x 12 pixels, no check
    await settle();
    const asked = phaseMapApi.layer.mock.calls.length;
    set(hook, { expectedShape: RASTER });                 // the geometry arrives, the picture is on it
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked);
  });

  it('drops a picture that does not fit the new expectation and fetches it again', async () => {
    const hook = mount(RASTER);
    await settle();
    expect(hook.result.current.bitmaps.get('phase').height).toBe(12);
    const asked = phaseMapApi.layer.mock.calls.length;
    bitmapSize = { width: 5, height: 6 };                 // what the backend draws from now on
    set(hook, { expectedShape: INDEX_GRID });
    await settle();
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked + 1);
    expect(hook.result.current.bitmaps.get('phase').height).toBe(6);
    expect(hook.result.current.errors.has('phase')).toBe(false);
  });

  it('drops a picture of the old raster when no raster is expected any more', async () => {
    const hook = mount(RASTER);
    await settle();
    const asked = phaseMapApi.layer.mock.calls.length;
    bitmapSize = { width: 5, height: 6 };
    set(hook, { expectedShape: null });
    await settle();
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked + 1);
    expect(hook.result.current.bitmaps.get('phase').height).toBe(6);
  });

  it('judges an answer that lands after the expectation changed by the new expectation', async () => {
    let release;
    phaseMapApi.layer.mockImplementationOnce(() => new Promise((resolve) => {
      release = () => resolve({ data: { image: 'Zm9v' } });
    }));
    bitmapSize = { width: 5, height: 6 };
    const hook = mount(RASTER);                           // asked while a 12 x 14 raster was expected
    await settle();
    set(hook, { expectedShape: INDEX_GRID });             // ... and the geometry has changed since
    await act(async () => { release(); });
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);
    expect(hook.result.current.bitmaps.get('phase').height).toBe(6);
  });
});

describe('while the geometry is being re-read', () => {
  it('neither shows nor refuses a picture that does not fit, and asks again when the geometry is known', async () => {
    bitmapSize = { width: 5, height: 6 };
    const hook = mount(RASTER, true);                     // the stale raster is still expected
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);   // no message for a verdict not yet due
    expect(hook.result.current.bitmaps.has('phase')).toBe(false);

    // the answer arrives: it is the index grid after all, so the same picture is right
    set(hook, { expectedShape: INDEX_GRID, shapePending: false });
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);
    expect(hook.result.current.bitmaps.get('phase').height).toBe(6);
  });

  it('refuses it once the geometry is known and still does not fit', async () => {
    bitmapSize = { width: 5, height: 6 };
    const hook = mount(RASTER, true);
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);

    set(hook, { expectedShape: RASTER, shapePending: false });   // confirmed: the raster is expected
    await settle();
    expect(hook.result.current.errors.get('phase')).toBe(SENTENCE);
    expect(hook.result.current.bitmaps.has('phase')).toBe(false);
  });

  it('does not ask in a loop while it waits', async () => {
    bitmapSize = { width: 5, height: 6 };
    mount(RASTER, true);
    await settle();
    const asked = phaseMapApi.layer.mock.calls.length;
    await settle();
    await settle();
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked);
  });
});

describe('the raster an answer says it drew', () => {
  // the same 12 x 14 pixels on the sample, started at x0 = 0 or at x0 = 1: only the origin tells them apart
  const drewOriginAt = (x0) => phaseMapApi.layer.mockImplementation(() => Promise.resolve({
    data: { image: 'Zm9v', display: { shape: RASTER, factor: 4, step_um: 0.5, x0, y0: 0 } },
  }));
  const originAt = (x0) => (display) => display.x0 === x0;

  it('keeps a picture on the expected raster', async () => {
    drewOriginAt(0);
    const hook = mount(RASTER, false, { rasterMatches: originAt(0), rasterId: 'a' });
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    expect(hook.result.current.errors.has('phase')).toBe(false);
  });

  it('refuses a picture of the right size on a raster with another origin, and does not ask in a loop', async () => {
    drewOriginAt(1);
    const hook = mount(RASTER, false, { rasterMatches: originAt(0), rasterId: 'a' });
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(false);
    expect(hook.result.current.errors.get('phase')).toBe(SENTENCE);
    const asked = phaseMapApi.layer.mock.calls.length;
    await settle();
    await settle();
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked);
  });

  it('checks no origin where the answer names no raster, or where none is expected', async () => {
    const hook = mount(RASTER, false, { rasterMatches: originAt(0), rasterId: 'a' });   // answer without display
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    drewOriginAt(1);
    const other = mount(RASTER, false, { rasterMatches: null, rasterId: '' });
    await settle();
    expect(other.result.current.bitmaps.has('phase')).toBe(true);
  });

  it('lifts the refusal when the expected origin changes to the one the picture is on', async () => {
    drewOriginAt(1);
    const hook = mount(RASTER, false, { rasterMatches: originAt(0), rasterId: 'a' });
    await settle();
    expect(hook.result.current.errors.get('phase')).toBe(SENTENCE);
    set(hook, { expectedShape: RASTER, rasterMatches: originAt(1), rasterId: 'b' });
    await settle();
    expect(hook.result.current.errors.has('phase')).toBe(false);
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
  });

  it('drops a picture held for the old origin and fetches it again for the new one', async () => {
    drewOriginAt(0);
    const hook = mount(RASTER, false, { rasterMatches: originAt(0), rasterId: 'a' });
    await settle();
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    const asked = phaseMapApi.layer.mock.calls.length;
    drewOriginAt(1);                                      // what the backend draws from now on
    set(hook, { expectedShape: RASTER, rasterMatches: originAt(1), rasterId: 'b' });
    await settle();
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked + 1);
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
    expect(hook.result.current.errors.has('phase')).toBe(false);
  });

  it('keeps a picture held for an origin that is still the expected one', async () => {
    drewOriginAt(0);
    const hook = mount(RASTER, false, { rasterMatches: originAt(0), rasterId: 'a' });
    await settle();
    const asked = phaseMapApi.layer.mock.calls.length;
    set(hook, { expectedShape: RASTER, rasterMatches: originAt(0), rasterId: 'a2' });   // renamed, same raster
    await settle();
    expect(phaseMapApi.layer.mock.calls.length).toBe(asked);
    expect(hook.result.current.bitmaps.has('phase')).toBe(true);
  });
});
