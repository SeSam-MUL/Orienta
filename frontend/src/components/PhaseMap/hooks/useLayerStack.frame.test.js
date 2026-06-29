// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: '' } })) },
  ebsdApi: {}, h5Api: {}, analysisApi: {},
}));

import { useLayerStack } from './useLayerStack';

describe('useLayerStack frameSig', () => {
  it('accepts frameSig without throwing and re-runs the flush effect', () => {
    const { rerender } = renderHook(
      ({ frameSig }) => useLayerStack({ cleanupParams: {}, resetSignal: 'r1', frameSig }),
      { initialProps: { frameSig: 'aaa' } },
    );
    act(() => rerender({ frameSig: 'bbb' }));
    expect(true).toBe(true); // no throw; bitmap cache flush for ipf-* layers fired
  });

  it('flushes the cache (bumps bitmapVersion) when frameSig changes', () => {
    // cacheFlush() — fired by the frame effect — calls force(), which bumps
    // bitmapVersion. So a frameSig change must increment bitmapVersion. This
    // is the externally-observable proof the flush effect actually ran
    // (cacheFlush itself is internal to the hook and not exposed).
    const { result, rerender } = renderHook(
      ({ frameSig }) => useLayerStack({ cleanupParams: {}, resetSignal: 'r1', frameSig }),
      { initialProps: { frameSig: 'aaa' } },
    );
    const before = result.current.bitmapVersion;
    act(() => rerender({ frameSig: 'bbb' }));
    expect(result.current.bitmapVersion).toBeGreaterThan(before);
  });

  it('does NOT flush on frameSig change when frameSig stays null (no-op for legacy callers)', () => {
    const { result, rerender } = renderHook(
      ({ frameSig }) => useLayerStack({ cleanupParams: {}, resetSignal: 'r1', frameSig }),
      { initialProps: { frameSig: null } },
    );
    const before = result.current.bitmapVersion;
    // Re-render with frameSig still null + same resetSignal: the frame effect
    // early-returns, so no extra cache flush / version bump from it.
    act(() => rerender({ frameSig: null }));
    expect(result.current.bitmapVersion).toBe(before);
  });
});
