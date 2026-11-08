import { useCallback, useRef, useState } from 'react';
import { phaseMapApi } from '../../../services/api';

/**
 * One-shot region-stats fetcher for the phase map. Region selection happens
 * once per drag, so no debounce or LRU cache is needed. The `region` state is
 * remembered even on error so the consumer can keep drawing the selection
 * rectangle as visual feedback.
 *
 * Only the answer to the latest request (or to none, after `clear`) is kept: a
 * slower earlier answer arriving afterwards is dropped, and `loading` ends with
 * the request that is current, so it never stays on.
 */
export function usePhaseMapRegionStats() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [region, setRegion] = useState(null);   // { start: {row, col}, end: {row, col} }
  const latestRef = useRef(0);

  // `extra` is spread into the request last (the mask of a hexagonal selection).
  const run = useCallback(async (start, end, extra) => {
    latestRef.current += 1;
    const mine = latestRef.current;
    setRegion({ start, end });
    setLoading(true); setError(null);
    try {
      const r = await phaseMapApi.regionStats({
        row_start: start.row, col_start: start.col,
        row_end:   end.row,   col_end:   end.col,
        ...extra,
      });
      if (mine !== latestRef.current) return;
      setData(r.data);
    } catch (e) {
      if (mine !== latestRef.current) return;
      setError(e?.response?.data?.detail || e?.message || 'region-stats failed');
      setData(null);
    } finally {
      if (mine === latestRef.current) setLoading(false);
    }
  }, []);

  const fetchStats = useCallback((start, end) => run(start, end, {}), [run]);

  /**
   * A selection on the index grid: `bbox` {row0, col0, rows, cols} and `mask`, a Uint8Array over the box
   * (row-major, 1 = selected) or null when every cell of the box is selected. The route takes both ends of
   * the box inclusive; a null mask is left out of the request.
   */
  const fetchMasked = useCallback(({ bbox, mask }) => run(
    { row: bbox.row0, col: bbox.col0 },
    { row: bbox.row0 + bbox.rows - 1, col: bbox.col0 + bbox.cols - 1 },
    mask ? { mask: Array.from(mask, (v) => !!v) } : {},
  ), [run]);

  const clear = useCallback(() => {
    latestRef.current += 1;             // an answer still on its way belongs to a selection that is gone
    setData(null); setError(null); setRegion(null); setLoading(false);
  }, []);

  return { data, loading, error, region, fetchStats, fetchMasked, clear };
}
