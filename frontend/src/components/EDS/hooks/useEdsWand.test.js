// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  edsApi: { wandField: vi.fn(), wandStats: vi.fn(), wandAssign: vi.fn() },
}));
import { edsApi } from '../../../services/api';
import { useEdsWand } from './useEdsWand';

/** 6x6: a 2x2 "feature" at (2,2) sitting at distance 0, matrix at 200. */
const R = 6, C = 6;
function fieldBytes() {
  const f = new Uint8Array(R * C).fill(200);
  for (const [r, c] of [[2, 2], [2, 3], [3, 2], [3, 3]]) f[r * C + c] = 0;
  f[4 * C + 4] = 255;                       // an unmeasured pixel
  let s = '';
  for (let i = 0; i < f.length; i++) s += String.fromCharCode(f[i]);
  return btoa(s);
}

const reply = {
  data: {
    n_rows: R, n_cols: C,
    seed: { row: 2, col: 2 },
    seed_at_pct: { Al: 70, Fe: 26 },
    field_b64: fieldBytes(),
    scale: 0.5,
    growth: [
      { threshold: 0, n_pixels: 4 },
      { threshold: 200, n_pixels: 35 },
    ],
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  edsApi.wandField.mockResolvedValue(reply);
  edsApi.wandStats.mockResolvedValue({ data: { n_pixels: 4, mean_at_pct: {}, enrichment: {} } });
  edsApi.wandAssign.mockResolvedValue({ data: { loaded: true } });
});

const seed = async (h) => { await act(async () => { await h.current.seedAt(2, 2); }); };

describe('useEdsWand', () => {
  it('selects the connected region containing the seed', async () => {
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    act(() => { result.current.setStep(0); });
    const m = result.current.mask;
    expect(m[2 * C + 2]).toBe(1);
    expect(m[3 * C + 3]).toBe(1);
    expect(m[0]).toBe(0);
    expect(m.reduce((a, b) => a + b, 0)).toBe(4);
  });

  it('grows with the slider', async () => {
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    act(() => { result.current.setStep(0); });
    const small = result.current.mask.reduce((a, b) => a + b, 0);
    act(() => { result.current.setStep(1); });
    const big = result.current.mask.reduce((a, b) => a + b, 0);
    expect(big).toBeGreaterThan(small);
  });

  it('never selects an unmeasured pixel', async () => {
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    act(() => { result.current.setStep(1); });   // widest step
    expect(result.current.mask[4 * C + 4]).toBe(0);
  });

  it('packs the mask the way the backend unpacks it', async () => {
    // np.unpackbits reads MSB-first within each byte; a mismatch would
    // paint a region the user never selected.
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    act(() => { result.current.setStep(0); });
    await act(async () => { await result.current.commit(3); });
    const [packed, phase] = edsApi.wandAssign.mock.calls[0];
    expect(phase).toBe(3);
    const bin = atob(packed);
    const bits = [];
    for (let i = 0; i < bin.length; i++) {
      const byte = bin.charCodeAt(i);
      for (let b = 7; b >= 0; b--) bits.push((byte >> b) & 1);
    }
    expect(bits[2 * C + 2]).toBe(1);
    expect(bits[3 * C + 3]).toBe(1);
    expect(bits[0]).toBe(0);
  });

  it('disarms after a commit so the next click does not re-seed', async () => {
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    expect(result.current.seed).not.toBeNull();
    await act(async () => { await result.current.commit(0); });
    expect(result.current.seed).toBeNull();
    expect(result.current.mask).toBeNull();
  });

  it('does not commit without a phase', async () => {
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    await act(async () => { await result.current.commit(null); });
    expect(edsApi.wandAssign).not.toHaveBeenCalled();
  });

  it('surfaces a backend error and clears the stale selection', async () => {
    edsApi.wandField.mockRejectedValueOnce({ response: { data: { detail: 'no EDS here' } } });
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    expect(result.current.error).toBe('no EDS here');
    expect(result.current.mask).toBeNull();
  });

  it('ignores a stats reply that a newer slider step has superseded', async () => {
    const { result } = renderHook(() => useEdsWand());
    await seed(result);
    let release;
    edsApi.wandStats.mockReturnValueOnce(new Promise((r) => { release = () => r({ data: { n_pixels: 999, mean_at_pct: {}, enrichment: {} } }); }));
    edsApi.wandStats.mockResolvedValueOnce({ data: { n_pixels: 4, mean_at_pct: {}, enrichment: {} } });
    await act(async () => {
      const stale = result.current.refreshStats();
      const fresh = result.current.refreshStats();
      release();
      await Promise.all([stale, fresh]);
    });
    expect(result.current.stats.n_pixels).toBe(4);
  });
});
