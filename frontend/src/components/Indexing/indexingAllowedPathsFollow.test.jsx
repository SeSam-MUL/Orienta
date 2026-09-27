// @vitest-environment jsdom
/**
 * I2 (whole-branch review, feat/phase-collections): the Spherical/Dictionary
 * `allowedPaths` follow effect in IndexingPage.jsx used to re-fetch only on
 * `[activeName, method]`. `App.jsx` keeps every page mounted with
 * `display: none` rather than unmounting it, so a phase can be filed into
 * the ACTIVE collection (database browser's "move to collection", or the
 * Collection Manager) while this page sits hidden under Hough — and Hough
 * is unaffected, because it filters directly on `collectionKeys`, a memo
 * over `collections` that is always fresh. Spherical and Dictionary are NOT
 * unaffected: they filter on the SEPARATELY FETCHED `allowedPaths`, and
 * without the active collection's own membership in the effect's dependency
 * array, switching back to Spherical/Dictionary kept serving the STALE
 * answer from before the membership change — silently offering LESS than
 * the collection now contains, with nothing on screen saying so.
 *
 * `useAllowedPaths` is exported from IndexingPage.jsx specifically so this
 * can be tested directly (mirrors PhaseMapPanel.jsx's exported `usePhaseMap`
 * for the same reason: IndexingPage itself is never fully rendered in
 * tests). This file proves the WIRING, not `resolveAllowedPaths`'s own
 * maths (already covered by `indexingCollection.test.jsx`): that a
 * membership change alone, with `activeName` and `method` both held
 * constant, makes the hook re-call the resolver and pick up the new answer.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, cleanup } from '@testing-library/react';

const resolve = vi.fn();
vi.mock('../../services/api', () => ({
  collectionsApi: { resolve: (...a) => resolve(...a) },
}));

const { useAllowedPaths } = await import('./IndexingPage');

const flush = () => act(async () => { await Promise.resolve(); await Promise.resolve(); });

// Two collections sharing a NAME across "generations" of membership — the
// point of this fixture is that `activeName` never changes; only `members`
// does, exactly as a live `useCollectionStore.load()` reply would replace it
// (a brand-new array/object every refresh, per `activeKeySignature`'s own
// docstring).
const MATRIX_V1 = [{ name: 'Matrix', parent: null, members: [{ key: 'Al' }, { key: 'Si' }] }];
const MATRIX_V2 = [{ name: 'Matrix', parent: null, members: [{ key: 'Al' }, { key: 'Si' }, { key: 'Ni' }] }];

beforeEach(() => {
  vi.clearAllMocks();
  cleanup();
});
afterEach(() => cleanup());

describe('useAllowedPaths (IndexingPage.jsx, exported for I2)', () => {
  it('resolves once for the active collection and method', async () => {
    resolve.mockResolvedValue({ data: { paths: ['/lib/Al (Al) [cF4] {20kV}.sht'] } });
    const { result } = renderHook(
      ({ name, method, cols }) => useAllowedPaths(name, method, cols),
      { initialProps: { name: 'Matrix', method: 'spherical', cols: MATRIX_V1 } },
    );
    await flush();
    expect(resolve).toHaveBeenCalledWith('Matrix', 'spherical');
    expect(result.current).toEqual(new Set(['/lib/Al (Al) [cF4] {20kV}.sht']));
  });

  it('I2 REGRESSION: re-resolves when the ACTIVE collection\'s own membership changes, name and method unchanged', async () => {
    resolve.mockResolvedValueOnce({ data: { paths: ['/lib/Al.sht', '/lib/Si.sht'] } });
    const { result, rerender } = renderHook(
      ({ name, method, cols }) => useAllowedPaths(name, method, cols),
      { initialProps: { name: 'Matrix', method: 'spherical', cols: MATRIX_V1 } },
    );
    await flush();
    expect(resolve).toHaveBeenCalledTimes(1);
    expect(result.current).toEqual(new Set(['/lib/Al.sht', '/lib/Si.sht']));

    // "Ni" was just filed into "Matrix" from the database browser, on a
    // DIFFERENT page — activeName ('Matrix') and method ('spherical') are
    // BOTH unchanged. Before the fix, this alone would never re-trigger the
    // effect, and `allowedPaths` would stay pinned to the pre-membership-
    // change Set forever (until the user happened to also flip methods).
    resolve.mockResolvedValueOnce({ data: { paths: ['/lib/Al.sht', '/lib/Si.sht', '/lib/Ni.sht'] } });
    rerender({ name: 'Matrix', method: 'spherical', cols: MATRIX_V2 });
    await flush();

    expect(resolve).toHaveBeenCalledTimes(2);
    expect(resolve).toHaveBeenLastCalledWith('Matrix', 'spherical');
    expect(result.current).toEqual(new Set(['/lib/Al.sht', '/lib/Si.sht', '/lib/Ni.sht']));
  });

  it('does not re-resolve on a re-render that changes neither name, method, nor membership', async () => {
    resolve.mockResolvedValue({ data: { paths: ['/lib/Al.sht'] } });
    const { rerender } = renderHook(
      ({ name, method, cols }) => useAllowedPaths(name, method, cols),
      { initialProps: { name: 'Matrix', method: 'spherical', cols: MATRIX_V1 } },
    );
    await flush();
    expect(resolve).toHaveBeenCalledTimes(1);

    // A brand-new `collections` array reference with IDENTICAL content (the
    // ordinary case: some unrelated collection refreshed elsewhere) must
    // NOT trigger a re-fetch — that is exactly what `activeKeySignature`
    // (a content fingerprint, not a reference compare) exists to prevent.
    rerender({ name: 'Matrix', method: 'spherical', cols: [...MATRIX_V1] });
    await flush();
    expect(resolve).toHaveBeenCalledTimes(1);
  });

  it('Hough never resolves at all (resolveAllowedPaths short-circuits) — confirms Hough was never exposed to this bug', async () => {
    const { result, rerender } = renderHook(
      ({ name, method, cols }) => useAllowedPaths(name, method, cols),
      { initialProps: { name: 'Matrix', method: 'hough', cols: MATRIX_V1 } },
    );
    await flush();
    expect(resolve).not.toHaveBeenCalled();
    expect(result.current).toBeNull();

    rerender({ name: 'Matrix', method: 'hough', cols: MATRIX_V2 });
    await flush();
    expect(resolve).not.toHaveBeenCalled();
  });
});
