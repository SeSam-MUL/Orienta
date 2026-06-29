import { useCallback, useState } from 'react';
import { phaseMapApi } from '../../../services/api';

/**
 * One-shot region-stats fetcher for the phase map. Region selection happens
 * once per drag, so no debounce or LRU cache is needed. The `region` state is
 * remembered even on error so the consumer can keep drawing the selection
 * rectangle as visual feedback.
 */
export function usePhaseMapRegionStats() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [region, setRegion] = useState(null);   // { start: {row, col}, end: {row, col} }

  const fetchStats = useCallback(async (start, end) => {
    setRegion({ start, end });
    setLoading(true); setError(null);
    try {
      const r = await phaseMapApi.regionStats({
        row_start: start.row, col_start: start.col,
        row_end:   end.row,   col_end:   end.col,
      });
      setData(r.data);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'region-stats failed');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setData(null); setError(null); setRegion(null);
  }, []);

  return { data, loading, error, region, fetchStats, clear };
}
