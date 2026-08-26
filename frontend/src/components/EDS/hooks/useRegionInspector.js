/**
 * Keeps the region inspector pointed at one region.
 *
 * A click on the map and a click in the list must land on the same thing, so
 * both funnel through here. The map has no region ids on the client — the
 * grid is one int per pixel and shipping it would cost ~1.9 MB on a 485k-px
 * scan — so the pixel lookup and the detail come back in ONE request rather
 * than two.
 *
 * Requests carry a token: a slow answer for a region the user has already
 * clicked past must not overwrite the current one.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { edsApi } from '../../../services/api';

export default function useRegionInspector({ selectedRegionId, setSelectedRegionId, mapVersion }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const token = useRef(0);

  const fetchFor = useCallback(async (sid) => {
    if (sid == null) { setDetail(null); setError(null); return null; }
    const mine = ++token.current;
    setLoading(true); setError(null);
    try {
      const res = await edsApi.regionDetail(sid);
      if (mine !== token.current) return null;
      setDetail(res.data);
      return res.data;
    } catch (e) {
      if (mine !== token.current) return null;
      setError(e.response?.data?.detail || String(e));
      setDetail(null);
      return null;
    } finally {
      if (mine === token.current) setLoading(false);
    }
  }, []);

  /** A click on the map: find the region under the pixel and describe it. */
  const inspectPixel = useCallback(async (row, col) => {
    const mine = ++token.current;
    setLoading(true); setError(null);
    try {
      const res = await edsApi.regionAt(row, col);
      if (mine !== token.current) return;
      if (res.data?.region_id == null) {
        // No data under that pixel. Saying so beats silently keeping the
        // previous region on screen as if it were the one clicked.
        setDetail(null);
        setSelectedRegionId?.(null);
        return;
      }
      setDetail(res.data);
      setSelectedRegionId?.(res.data.region_id);
    } catch (e) {
      if (mine !== token.current) return;
      setError(e.response?.data?.detail || String(e));
    } finally {
      if (mine === token.current) setLoading(false);
    }
  }, [setSelectedRegionId]);

  // Follow the selection, and refresh after anything that changes the map:
  // merging renumbers ids, splitting adds them, a boundary move changes the
  // pixels — a stale panel would describe a region that no longer exists.
  useEffect(() => {
    fetchFor(selectedRegionId);
  }, [selectedRegionId, mapVersion, fetchFor]);

  return { detail, loading, error, inspectPixel, refresh: () => fetchFor(selectedRegionId) };
}
