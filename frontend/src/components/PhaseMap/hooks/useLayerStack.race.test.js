// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// Controllable layer fetches: each call parks a deferred that the test
// resolves on demand, so we can hold a fetch "in flight" across rapid layer
// switches and reproduce the epoch-drop race deterministically.
const calls = [];
vi.mock('../../../services/api', () => ({
  phaseMapApi: {
    layer: vi.fn((kind, params) => new Promise((res) => {
      const entry = {
        kind, params, resolved: false,
        resolve: () => { entry.resolved = true; res({ data: { image: 'Zm9v' } }); },
      };
      calls.push(entry);
    })),
  },
  ebsdApi: {}, h5Api: {}, analysisApi: {},
}));
// Succeeding decode pipeline so a fetch that ISN'T dropped actually caches.
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 4, height: 4, close: () => {} })));

import { useLayerStack } from './useLayerStack';

const CLEANUP = {};
const tick = () => new Promise((r) => setTimeout(r, 0));
const countCalls = (kind) => calls.filter((c) => c.kind === kind).length;
const pending = (kind) => calls.filter((c) => c.kind === kind && !c.resolved);

describe('useLayerStack rapid-switch fetch race', () => {
  beforeEach(() => { calls.length = 0; });

  it('re-fetches a layer stranded by a switch-away-and-back while its first fetch was in flight', async () => {
    const { result } = renderHook(() =>
      useLayerStack({ cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, colorOverrides: {} }));
    await act(async () => { await tick(); });

    // Switch to IPF-Z; hold its fetch in flight (do NOT resolve).
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    // Switch away to IPF-X (also in flight), then back to IPF-Z. The back-
    // switch finds IPF-Z still in fetchingRef → the fetch effect is blocked
    // from issuing a fresh IPF-Z request.
    await act(async () => { result.current.setSingleLayer('ipf-x'); await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    expect(countCalls('ipf-z')).toBe(1);  // the first (soon-stale) request only

    // The stale IPF-Z fetch resolves: its epoch is behind → it must be
    // dropped AND trigger a re-fetch of the now-current, still-uncached
    // IPF-Z layer. Without the fix the layer stays bitmap-less forever.
    await act(async () => {
      pending('ipf-z')[0].resolve();
      await tick(); await tick();
    });

    expect(countCalls('ipf-z')).toBeGreaterThanOrEqual(2);
  });
});
