// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// Non-empty image so the fetch SUCCEEDS and the decoded bitmap is CACHED —
// this is what makes the cache-flush meaningful: without the flush a colour
// change would find 'phase' still cached and never re-fetch (the production
// bug). jsdom lacks createImageBitmap / real data-URL fetch, so we stub the
// decode pipeline to return a lightweight fake bitmap.
vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: 'Zm9v' } })) },
  ebsdApi: {}, h5Api: {}, analysisApi: {},
}));
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 4, height: 4, close: () => {} })));

import { useLayerStack } from './useLayerStack';
import { phaseMapApi } from '../../../services/api';

// Stable references so the hook settles (a fresh {} each render would rebuild
// fetchLayer and re-fire the fetch effect forever).
const CLEANUP = {};
const NO_OVERRIDES = {};

// Flush the effect + async fetch/decode chain (fetch → blob → createImageBitmap
// → cacheSet → force re-render). Two macrotasks cover it comfortably.
const settle = () => act(async () => {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
});

describe('useLayerStack colorOverrides', () => {
  beforeEach(() => phaseMapApi.layer.mockClear());

  it('re-fetches the phase layer WITH color_overrides when overrides change (cache flush works)', async () => {
    const { rerender } = renderHook(
      ({ colorOverrides }) =>
        useLayerStack({ cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, colorOverrides }),
      { initialProps: { colorOverrides: NO_OVERRIDES } },
    );
    await settle();
    // Baseline: phase fetched + cached (no overrides yet).
    expect(phaseMapApi.layer.mock.calls.some(([kind]) => kind === 'phase')).toBe(true);

    phaseMapApi.layer.mockClear();
    await act(async () => {
      rerender({ colorOverrides: { Al: '#ff0000' } });
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));
    });

    // The cached 'phase' bitmap must be dropped + re-fetched WITH overrides.
    // If the flush were missing, 'phase' stays cached → zero new calls here.
    const overrideCalls = phaseMapApi.layer.mock.calls.filter(
      ([kind, params]) => kind === 'phase' && params && params.color_overrides,
    );
    expect(overrideCalls.length).toBeGreaterThan(0);
    expect(JSON.parse(overrideCalls.at(-1)[1].color_overrides)).toEqual({ Al: '#ff0000' });
  });

  it('does NOT attach color_overrides when there are no overrides', async () => {
    renderHook(
      ({ colorOverrides }) =>
        useLayerStack({ cleanupParams: CLEANUP, resetSignal: 'r1', frameSig: null, colorOverrides }),
      { initialProps: { colorOverrides: NO_OVERRIDES } },
    );
    await settle();
    const phaseCall = phaseMapApi.layer.mock.calls.find(([kind]) => kind === 'phase');
    expect(phaseCall).toBeTruthy();
    expect(phaseCall[1]?.color_overrides).toBeUndefined();
  });
});
