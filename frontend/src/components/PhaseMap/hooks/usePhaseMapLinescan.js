import { useCallback, useRef, useState } from 'react';
import { phaseMapApi } from '../../../services/api';

/**
 * Manages linescan state for the phase map: the current line (start + end pixel
 * coords) and the backend-fetched per-layer profile arrays. PhaseMap linescans
 * are mode-agnostic (no displayMode arg, unlike the EDS counterpart). The
 * consumer drives this via fetchProfile(start, end), or fetchPoints(cells) for a
 * list of scan points; the line is remembered so the consumer can re-render it
 * as a visual indicator.
 *
 * Only the answer to the latest request (or to none, after `clear`) is kept: a
 * slower earlier answer arriving afterwards is dropped, and `loading` ends with
 * the request that is current, so it never stays on.
 */
export function usePhaseMapLinescan({ layerIds, nSamples = 128 } = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [line, setLine] = useState(null);   // { start: {row, col}, end: {row, col} } or { points: [[row, col], ...] }
  const latestRef = useRef(0);

  const run = useCallback(async (where, request) => {
    latestRef.current += 1;
    const mine = latestRef.current;
    setLine(where);
    if (!layerIds || layerIds.length === 0) { setData(null); setLoading(false); return; }
    setLoading(true); setError(null);
    try {
      const r = await phaseMapApi.linescan({ ...request, layers: layerIds });
      if (mine !== latestRef.current) return;
      setData(r.data);
    } catch (e) {
      if (mine !== latestRef.current) return;
      setError(e?.response?.data?.detail || e?.message || 'linescan failed');
      setData(null);
    } finally {
      if (mine === latestRef.current) setLoading(false);
    }
  }, [layerIds]);

  const fetchProfile = useCallback((start, end) => run({ start, end }, {
    start_row: start.row, start_col: start.col,
    end_row: end.row,     end_col: end.col,
    n_samples: nSamples,
  }), [run, nSamples]);

  /** The profile at exactly these scan points, `[[row, col], ...]` (2 to 4096), evenly spaced along the path. */
  const fetchPoints = useCallback((cells) => run({ points: cells }, { points: cells }), [run]);

  const clear = useCallback(() => {
    latestRef.current += 1;             // an answer still on its way belongs to a line that is gone
    setLine(null); setData(null); setError(null); setLoading(false);
  }, []);

  return { data, loading, error, line, fetchProfile, fetchPoints, clear };
}
