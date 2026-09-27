// @vitest-environment jsdom
/**
 * An add-on's map as a layer: the id, the fetch, and the two ways it can lie.
 *
 * The id carries the result AND the run, and both are load-bearing. Without
 * the result the fetch cannot be built at all — `fetchLayerImage` is a module
 * function with no result in scope. Without the run token a second run is a
 * no-op that leaves the first run's numbers on screen under a label saying
 * otherwise.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/addonsApi', () => ({
  addonsApi: { mapImage: vi.fn() },
}));
vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: '' } })) },
  analysisApi: { getMap: vi.fn() },
  ebsdApi: { virtualBSE: vi.fn() },
  h5Api: { getEDSMap: vi.fn(), getElectronImage: vi.fn() },
}));

import { addonsApi } from '../../../services/addonsApi';
import { useLayerStack, parseAddonLayerId } from './useLayerStack';

afterEach(cleanup);

const ID = 'addon:bc-gmm/res-1/addon.bc_gmm/component_map#t1';

// A 1x1 transparent PNG: enough for createImageBitmap to be asked for one.
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk'
  + 'YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

beforeEach(() => {
  vi.clearAllMocks();
  addonsApi.mapImage.mockResolvedValue({ data: {
    image: PNG, shape: [2, 2],
    scale: { min: 0, max: 2, unit: '', cmap: 'viridis', stops: ['#440154'] } } });
  // jsdom has no createImageBitmap; the hook only needs something closeable.
  global.createImageBitmap = vi.fn(async () => ({ close: () => {}, width: 1, height: 1 }));
});

describe('parseAddonLayerId', () => {
  it('reads the four parts and drops the run token', () => {
    expect(parseAddonLayerId(ID)).toEqual({
      name: 'bc-gmm', resultId: 'res-1',
      analysisKey: 'addon.bc_gmm', key: 'component_map' });
  });

  it('keeps an analysis key that contains dots intact', () => {
    // Keys are `addon.<something>` by contract, so a parser that split on
    // dots would break every real id.
    expect(parseAddonLayerId('addon:a/r/addon.b.c.d/k#x').analysisKey)
      .toBe('addon.b.c.d');
  });

  it('is null for every id that is not one of ours', () => {
    expect(parseAddonLayerId('eds:Al')).toBeNull();
    expect(parseAddonLayerId('addon:too/few/parts')).toBeNull();
    expect(parseAddonLayerId('addon:a//c/d')).toBeNull();
    expect(parseAddonLayerId(undefined)).toBeNull();
  });
});

function setup(resultId = 'res-1') {
  return renderHook(() => useLayerStack({
    cleanupParams: {}, resetSignal: 0, frameSig: 'f', resultId,
    seedResultId: resultId }));
}

describe('adding one', () => {
  it('lands on the stack as an addon-source layer', async () => {
    // The branch two shipped "rendered but unwired" defects lacked: without
    // it addLayer warns and returns, and the click does nothing at all.
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    const layer = result.current.layers.find((l) => l.id === ID);
    expect(layer).toBeTruthy();
    expect(layer.source).toBe('addon');
  });

  it('takes the add-on’s declared label over the output key', async () => {
    // The id can only ever yield `component_map`, and this label is what the
    // value-scale legend prints beside a figure in a paper.
    const { result } = setup();
    act(() => {
      result.current.addLayer(ID, { label: 'Component map (GMM)' });
    });
    expect(result.current.layers.find((l) => l.id === ID).label)
      .toBe('Component map (GMM)');
  });

  it('fetches with the four parts out of the id', async () => {
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalled());
    expect(addonsApi.mapImage).toHaveBeenCalledWith(
      'bc-gmm', 'res-1', 'addon.bc_gmm', 'component_map');
  });

  it('keeps the scale, so the legend can state the unit', async () => {
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    await waitFor(() =>
      expect(result.current.scales.get(ID)).toBeTruthy());
    expect(result.current.scales.get(ID).cmap).toBe('viridis');
  });
});

describe('the id identifies the run and the result', () => {
  it('two results give two different layers and two fetches', async () => {
    // Dropping result_id from the id would make these one layer — and the
    // fetch could not be built at all, since fetchLayerImage has no result
    // in scope.
    const { result } = setup('res-1');
    act(() => { result.current.addLayer(ID); });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalledTimes(1));
    act(() => {
      result.current.addLayer('addon:bc-gmm/res-1/addon.bc_gmm/other_map#t1');
    });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalledTimes(2));
    expect(result.current.layers.filter((l) => l.source === 'addon'))
      .toHaveLength(2);
  });

  it('a second run re-fetches instead of serving the cached bitmap', async () => {
    // Asserted on the FETCH COUNT: a stale bitmap and a fresh one look the
    // same in jsdom, so identity proves nothing. Mint the same token twice
    // and this goes red — which is what says the mechanism is present and
    // not just the test.
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalledTimes(1));
    act(() => {
      result.current.addLayer('addon:bc-gmm/res-1/addon.bc_gmm/component_map#t2');
    });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalledTimes(2));
  });

  it('does not draw over a different result, and says whose it is', async () => {
    // The case that composites SILENTLY: two results of the same shape, one
    // result's add-on map over another's phase map, nothing to see.
    const { result } = setup('res-OTHER');
    act(() => { result.current.addLayer(ID); });
    await waitFor(() =>
      expect(result.current.errors.get(ID)).toBeTruthy());
    expect(addonsApi.mapImage).not.toHaveBeenCalled();
    expect(result.current.errors.get(ID)).toContain('res-1');
  });
});

describe('when the map is gone', () => {
  it('an evicted map reports why instead of drawing nothing', async () => {
    addonsApi.mapImage.mockRejectedValue({
      response: { data: { reason: 'map_not_stored',
                          detail: 'No stored map. Run the analysis again.' } } });
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    await waitFor(() => expect(result.current.errors.get(ID)).toBeTruthy());
    expect(result.current.errors.get(ID)).toMatch(/run the analysis again/i);
  });
});

describe('which dispatches may keep an add-on layer', () => {
  // The reducer keeps add-on layers only when the caller asks it to, and the
  // asking is what this file can see. Its unit test proves the reducer obeys
  // the flag; only a test at this level proves the SEED is the one dispatch
  // that sets it. Without this, adding the flag to usePreset passed 447 tests
  // -- the flag on the wrong dispatch is the same defect the branch keeps
  // finding: two halves each right, wrong together.
  it('a preset the USER picks still replaces everything', async () => {
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalled());
    expect(result.current.layers.some((l) => l.id === ID)).toBe(true);

    act(() => { result.current.usePreset('phase_default'); });
    expect(result.current.layers.some((l) => l.id === ID)).toBe(false);
  });

  it('a quick mode the USER picks still replaces everything', async () => {
    const { result } = setup();
    act(() => { result.current.addLayer(ID); });
    await waitFor(() => expect(addonsApi.mapImage).toHaveBeenCalled());

    act(() => { result.current.setSingleLayer('phase'); });
    expect(result.current.layers.some((l) => l.id === ID)).toBe(false);
  });
});
