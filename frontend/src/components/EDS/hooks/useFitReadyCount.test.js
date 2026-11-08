// @vitest-environment jsdom
/**
 * Widgets that asked for EDS numbers ONCE -- on file load, on dialog open --
 * got a 503 while the spectrum fit ran and kept that answer: "No EDS data
 * loaded", "No EDS elements found in dataset" for a file that has both.
 * `useFitReadyCount` is what they listen to: it
 * counts the times the fit turned ready, so a fetch keyed on it runs again.
 */
import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';

import { useFitReadyCount } from './useSpectrumFitStatus';

describe('useFitReadyCount', () => {
  it('counts each turn into "ready", from the first answer or from running', () => {
    const { result, rerender } = renderHook(({ s }) => useFitReadyCount(s),
      { initialProps: { s: null } });
    expect(result.current).toBe(0);
    rerender({ s: 'running' });
    expect(result.current).toBe(0);
    rerender({ s: 'ready' });
    expect(result.current).toBe(1);
    rerender({ s: 'ready' });                    // a repeated answer is no turn
    expect(result.current).toBe(1);
    rerender({ s: null });                       // another file: not known yet
    rerender({ s: 'ready' });                    // its fit was cached
    expect(result.current).toBe(2);
    rerender({ s: 'failed' });
    rerender({ s: 'ready' });                    // a retry that worked
    expect(result.current).toBe(3);
  });

  it('a widget mounted on a ready fit asked already: no extra turn', () => {
    const { result } = renderHook(() => useFitReadyCount('ready'));
    expect(result.current).toBe(0);
  });

  it('a file without spectra never counts', () => {
    const { result, rerender } = renderHook(({ s }) => useFitReadyCount(s),
      { initialProps: { s: null } });
    rerender({ s: 'none' });
    expect(result.current).toBe(0);
  });
});
