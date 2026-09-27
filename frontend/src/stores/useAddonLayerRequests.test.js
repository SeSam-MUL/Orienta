import { describe, it, expect, beforeEach } from 'vitest';
import useAddonLayerRequests from './useAddonLayerRequests';
import { planAddonDrain } from '../components/PhaseMap/addonLayerDrain';

const MAX = 8;
const layers = (n) => Array.from({ length: n }, (_, i) => ({ id: `l${i}` }));

beforeEach(() => { useAddonLayerRequests.getState().clear(); });

describe('the queue', () => {
  it('carries the label, not just the id', () => {
    // The page builds a layer's label from the id alone, and the best that
    // can yield is `component_map`. The author's declared label exists only
    // on the other page — and it is what the legend prints beside a figure.
    useAddonLayerRequests.getState().request('addon:a/r/k/m#t', 'Component map');
    expect(useAddonLayerRequests.getState().requests)
      .toEqual([{ id: 'addon:a/r/k/m#t', label: 'Component map' }]);
  });

  it('take() returns everything and empties the queue in one step', () => {
    // One step, because StrictMode runs the drain effect twice in
    // development: a read that left the queue standing would hand the same
    // requests over again on the second run.
    const s = useAddonLayerRequests.getState();
    s.request('a', 'A');
    s.request('b', 'B');
    expect(useAddonLayerRequests.getState().take()).toHaveLength(2);
    expect(useAddonLayerRequests.getState().requests).toEqual([]);
    expect(useAddonLayerRequests.getState().take()).toEqual([]);
  });

  it('asking twice before a drain is one request', () => {
    const s = useAddonLayerRequests.getState();
    s.request('a', 'A');
    s.request('a', 'A');
    expect(useAddonLayerRequests.getState().requests).toHaveLength(1);
  });

  it('the stored array keeps its identity when nothing changes', () => {
    // zustand 5 loops on a selector that returns a NEW reference, and the
    // page selects `s.requests` directly.
    const s = useAddonLayerRequests.getState();
    s.request('a', 'A');
    const first = useAddonLayerRequests.getState().requests;
    s.request('a', 'A');                       // deduplicated: no new array
    expect(useAddonLayerRequests.getState().requests).toBe(first);
  });
});

describe('planning the drain', () => {
  it('adds what fits', () => {
    const { toAdd, refused } = planAddonDrain(
      [{ id: 'x', label: 'X' }], layers(2), MAX);
    expect(toAdd).toHaveLength(1);
    expect(refused).toEqual([]);
  });

  it('reports what the full stack refused, rather than losing it', () => {
    // With eight layers on the page, "show as layer" plus a drain that
    // cleared the queue meant no layer, no message and no request.
    const { toAdd, refused } = planAddonDrain(
      [{ id: 'x', label: 'Component map' }], layers(MAX), MAX);
    expect(toAdd).toEqual([]);
    expect(refused).toEqual([{ id: 'x', label: 'Component map' }]);
  });

  it('a layer already on the stack is NOT reported as "stack full"', () => {
    // The StrictMode case: the second pass sees the layer it just added.
    // Inferring the refusal from the reducer's unchanged return would call
    // this a full stack on every development run.
    const { toAdd, refused } = planAddonDrain(
      [{ id: 'l1', label: 'Already there' }], layers(MAX), MAX);
    expect(toAdd).toEqual([]);
    expect(refused).toEqual([]);
  });

  it('fills the last free slot and refuses only the rest', () => {
    const { toAdd, refused } = planAddonDrain(
      [{ id: 'x', label: 'X' }, { id: 'y', label: 'Y' }],
      layers(MAX - 1), MAX);
    expect(toAdd.map((r) => r.id)).toEqual(['x']);
    expect(refused.map((r) => r.id)).toEqual(['y']);
  });

  it('a re-run REPLACES its own earlier layer instead of stacking beside it', () => {
    // Found in the acceptance run: after a second run the stack held two rows
    // with the identical author label, the lower one drawing the FIRST run's
    // map. In a figure the two are indistinguishable.
    const older = [{ id: 'addon:a/r/k/m#run1' }];
    const { toAdd, toRemove, refused } = planAddonDrain(
      [{ id: 'addon:a/r/k/m#run2', label: 'M' }], older, MAX);
    expect(toRemove).toEqual(['addon:a/r/k/m#run1']);
    expect(toAdd.map((r) => r.id)).toEqual(['addon:a/r/k/m#run2']);
    expect(refused).toEqual([]);
  });

  it('leaves a DIFFERENT map of the same run alone', () => {
    // Two outputs of one analysis are two layers, not a replacement.
    const older = [{ id: 'addon:a/r/k/first#run1' }];
    const { toAdd, toRemove } = planAddonDrain(
      [{ id: 'addon:a/r/k/second#run1', label: 'Second' }], older, MAX);
    expect(toRemove).toEqual([]);
    expect(toAdd).toHaveLength(1);
  });

  it('a re-run is not refused by a full stack it is replacing into', () => {
    const full = [{ id: 'addon:a/r/k/m#run1' }, ...layers(MAX - 1)];
    const { toAdd, toRemove, refused } = planAddonDrain(
      [{ id: 'addon:a/r/k/m#run2', label: 'M' }], full, MAX);
    expect(toRemove).toHaveLength(1);
    expect(toAdd).toHaveLength(1);
    expect(refused).toEqual([]);
  });

  it('counts a duplicate as taking no room', () => {
    // Otherwise a request for a layer already present would push a real one
    // out of the last free slot.
    const { toAdd, refused } = planAddonDrain(
      [{ id: 'l0', label: 'Dup' }, { id: 'x', label: 'X' }],
      layers(MAX - 1), MAX);
    expect(toAdd.map((r) => r.id)).toEqual(['x']);
    expect(refused).toEqual([]);
  });
});
