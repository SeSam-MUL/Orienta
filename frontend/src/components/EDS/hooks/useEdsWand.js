import { useCallback, useMemo, useRef, useState } from 'react';
import { edsApi } from '../../../services/api';

/**
 * Seeded selection ("magic wand") on the EDS phase map.
 *
 * The backend returns ONE distance-from-seed field per click; every slider
 * tick is then a flood fill in the browser. Measured: the fill is ~2.8 ms
 * over 485 000 px, so the preview needs no round trip, no debounce and no
 * cancellation — a round trip per tick would be slower and would feel it.
 *
 * The slider is a PIXEL COUNT, not a chemical tolerance. Thresholding the
 * distance directly has dead bands: measured on a real particle, 2 % through
 * 10 % of the range all returned the identical 98 154 px. The backend ships
 * a growth curve sampled where pixels actually are, and the slider walks it.
 */
export function useEdsWand() {
  const [seed, setSeed] = useState(null);        // { row, col }
  const [growth, setGrowth] = useState([]);      // [{ threshold, n_pixels }]
  const [step, setStep] = useState(0);           // index into growth
  const [shape, setShape] = useState(null);      // { n_rows, n_cols }
  const [seedAtPct, setSeedAtPct] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fieldRef = useRef(null);                 // Uint8Array, row-major
  const maskRef = useRef(null);                  // Uint8Array 0/1
  const statsTokenRef = useRef(null);

  const clear = useCallback(() => {
    fieldRef.current = null;
    maskRef.current = null;
    statsTokenRef.current = null;
    setSeed(null); setGrowth([]); setStep(0); setShape(null);
    setSeedAtPct(null); setStats(null); setError(null); setLoading(false);
  }, []);

  /** Click a pixel: fetch the field and select the seed's own region. */
  const seedAt = useCallback(async (row, col) => {
    setLoading(true); setError(null); setStats(null);
    try {
      const res = await edsApi.wandField(row, col);
      const d = res.data;
      const bin = atob(d.field_b64);
      const field = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) field[i] = bin.charCodeAt(i);
      fieldRef.current = field;
      maskRef.current = null;
      setShape({ n_rows: d.n_rows, n_cols: d.n_cols });
      setSeed({ row, col });
      setSeedAtPct(d.seed_at_pct || null);
      setGrowth(d.growth || []);
      // Start on the step nearest a small, obviously-a-feature selection
      // rather than at either extreme: 1 px tells the user nothing and the
      // last step is usually the whole map.
      const g = d.growth || [];
      const start = g.length
        ? g.reduce((best, s, i) => (
            Math.abs(s.n_pixels - 400) < Math.abs(g[best].n_pixels - 400) ? i : best), 0)
        : 0;
      setStep(start);
    } catch (err) {
      clear();
      setError(err.response?.data?.detail || err.message || 'Wand failed.');
    } finally {
      setLoading(false);
    }
  }, [clear]);

  /** 4-connected fill from the seed over `field <= threshold`. */
  const mask = useMemo(() => {
    const field = fieldRef.current;
    if (!field || !seed || !shape || !growth.length) return null;
    const { n_rows: R, n_cols: C } = shape;
    const thr = growth[Math.min(step, growth.length - 1)].threshold;
    const start = seed.row * C + seed.col;
    const out = new Uint8Array(R * C);
    if (field[start] > thr) { maskRef.current = out; return out; }

    // Explicit stack rather than recursion: the region can be the whole map.
    const stack = new Int32Array(R * C);
    let sp = 0;
    stack[sp++] = start;
    out[start] = 1;
    while (sp > 0) {
      const p = stack[--sp];
      const r = (p / C) | 0;
      const c = p - r * C;
      if (c > 0     && !out[p - 1] && field[p - 1] <= thr) { out[p - 1] = 1; stack[sp++] = p - 1; }
      if (c < C - 1 && !out[p + 1] && field[p + 1] <= thr) { out[p + 1] = 1; stack[sp++] = p + 1; }
      if (r > 0     && !out[p - C] && field[p - C] <= thr) { out[p - C] = 1; stack[sp++] = p - C; }
      if (r < R - 1 && !out[p + C] && field[p + C] <= thr) { out[p + C] = 1; stack[sp++] = p + C; }
    }
    maskRef.current = out;
    return out;
  }, [seed, shape, growth, step]);

  const nSelected = growth.length
    ? growth[Math.min(step, growth.length - 1)].n_pixels : 0;

  /** Pack the mask the way the backend unpacks it (np.packbits order). */
  const packMask = useCallback(() => {
    const m = maskRef.current;
    if (!m) return null;
    const bytes = new Uint8Array(Math.ceil(m.length / 8));
    for (let i = 0; i < m.length; i++) {
      if (m[i]) bytes[i >> 3] |= 0x80 >> (i & 7);
    }
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
  }, []);

  /** Composition + enrichment of the current selection, for the readout. */
  const refreshStats = useCallback(async () => {
    const packed = packMask();
    if (!packed) return;
    const token = {};
    statsTokenRef.current = token;
    try {
      const res = await edsApi.wandStats(packed);
      if (statsTokenRef.current !== token) return;   // a newer step won
      setStats(res.data);
    } catch {
      if (statsTokenRef.current === token) setStats(null);
    }
  }, [packMask]);

  const commit = useCallback(async (phaseIndex) => {
    const packed = packMask();
    if (packed == null || phaseIndex == null) return null;
    setLoading(true); setError(null);
    try {
      const res = await edsApi.wandAssign(packed, phaseIndex);
      clear();                 // disarm: the next click must not re-seed by accident
      return res.data;
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Assign failed.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [packMask, clear]);

  return {
    seed, seedAtPct, growth, step, setStep, shape, mask, nSelected,
    stats, refreshStats, loading, error,
    seedAt, commit, clear,
  };
}
