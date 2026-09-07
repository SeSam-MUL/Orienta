// @vitest-environment jsdom
//
// The post-processing cleanup sliders are a VIEW filter: the backend re-renders
// the layer with the rejected pixels dropped. That only reaches the screen if
// the cached bitmap of every affected layer is thrown away when the params
// change.
//
// Measured production bug (2026-09-03): the invalidation set was
// {phase, ci, uncertainty, ci_*} — the IPF and BC layers were left out, on the
// stated reasoning that "IPF colour doesn't depend on cleanup". The COLOUR does
// not; the ALPHA does (backend/api/routes/phase_map.py: `alpha[:] =
// (effective_pid_2d >= 0)` then `alpha &= ipf_valid_2d` for ipf-*, and the same
// for bc when band contrast comes from the xmap). So dragging the CI slider
// changed nothing on screen for a user whose stack showed IPF — confirmed in
// backend-console.log: 30+ requests carried a non-zero ci_threshold and NOT ONE
// of them was a /layer request.
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// Non-empty image so the fetch SUCCEEDS and the bitmap is CACHED — a cached
// bitmap is the whole point: without it every re-render re-fetches anyway and
// the test would pass with no invalidation at all.
vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: 'Zm9v' } })) },
  ebsdApi: {}, h5Api: {}, analysisApi: {},
}));
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 4, height: 4, close: () => {} })));

import { useLayerStack } from './useLayerStack';
import { phaseMapApi } from '../../../services/api';

const NO_CLEANUP = {
  ci_threshold: 0, uncertainty_threshold: 0, min_cluster_size: 0,
  fill_unindexed: false, modal_filter_size: 0,
};
const CI_04 = { ...NO_CLEANUP, ci_threshold: 0.4 };

const settle = () => act(async () => {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
});

/** Mount the hook with one layer on the stack and let it fetch + cache. */
async function mountWithLayer(layerId) {
  const view = renderHook(
    ({ cleanupParams }) =>
      useLayerStack({ cleanupParams, resetSignal: 'r1', frameSig: null, colorOverrides: null }),
    { initialProps: { cleanupParams: NO_CLEANUP } },
  );
  await settle();
  await act(async () => { view.result.current.setSingleLayer(layerId); });
  await settle();
  return view;
}

describe('useLayerStack — cleanup params invalidate every layer the backend filters', () => {
  beforeEach(() => phaseMapApi.layer.mockClear());

  for (const layerId of ['ipf-z', 'ipf-x', 'ipf-y', 'bc', 'phase']) {
    it(`re-fetches '${layerId}' with the new ci_threshold when cleanup changes`, async () => {
      const view = await mountWithLayer(layerId);
      expect(phaseMapApi.layer.mock.calls.some(([kind]) => kind === layerId)).toBe(true);

      phaseMapApi.layer.mockClear();
      await act(async () => {
        view.rerender({ cleanupParams: CI_04 });
        await new Promise((r) => setTimeout(r, 0));
        await new Promise((r) => setTimeout(r, 0));
        await new Promise((r) => setTimeout(r, 0));
      });

      // Without the flush the bitmap stays cached → zero new calls for this id.
      const refetch = phaseMapApi.layer.mock.calls.filter(
        ([kind, params]) => kind === layerId && params?.ci_threshold === 0.4,
      );
      expect(refetch.length).toBeGreaterThan(0);
    });
  }
});
