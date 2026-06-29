import { useCallback, useState } from 'react';
import { edsApi } from '../../../services/api';

/**
 * Manages linescan state: the current line (start + end pixel coords) and
 * the backend-fetched per-layer profile arrays. The consumer drives this
 * via fetchProfile(start, end); the line is remembered so the consumer
 * can re-render it as a visual indicator.
 */
export function useLinescan({ layerIds, displayMode = 'at_pct', nSamples = 128 } = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [line, setLine] = useState(null);   // { start: {row, col}, end: {row, col} }

  const fetchProfile = useCallback(async (start, end) => {
    setLine({ start, end });
    if (!layerIds || layerIds.length === 0) { setData(null); return; }
    setLoading(true); setError(null);
    try {
      const r = await edsApi.linescan({
        start_row: start.row, start_col: start.col,
        end_row: end.row,     end_col: end.col,
        n_samples: nSamples,
        layers: layerIds,
        display_mode: displayMode,
      });
      setData(r.data);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'linescan failed');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [layerIds, displayMode, nSamples]);

  const clear = useCallback(() => {
    setLine(null); setData(null); setError(null);
  }, []);

  return { data, loading, error, line, fetchProfile, clear };
}
