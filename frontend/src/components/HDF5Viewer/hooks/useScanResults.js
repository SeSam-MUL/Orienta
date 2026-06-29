/**
 * useScanResults — calls /api/h5/scan/quality and /api/h5/scan/defects.
 *
 * Replaces the per-pattern-fetch loops the viewer used to run client-side.
 * The backend endpoints (h5_viewer.py) sample N patterns server-side and
 * compute std-dev / classification in numpy, which is dramatically faster
 * than round-tripping every sample over HTTP.
 *
 * Fail-loud contract:
 *   - On error, the result is set to { error: "<detail>" } so the modal
 *     can render a red message (UI-visible failure).
 *   - The error is ALSO logged via console.warn so it's diagnosable in
 *     DevTools.
 *   - The hook does NOT re-throw: callers can simply `await scanQuality(N)`
 *     without try/catch, and rely on `result.error` for the user surface.
 *     This avoids the "throw-then-empty-catch" anti-pattern that was
 *     silently swallowing the throw upstream.
 */
import { useCallback, useState } from 'react';
import i18n from '../../../i18n';
import { h5Api } from '../../../services/api';

export function useScanResults() {
  const [quality, setQuality] = useState(null);
  const [defects, setDefects] = useState(null);

  const scanQuality = useCallback(async (samples = 50) => {
    setQuality('scanning');
    try {
      const res = await h5Api.scanQuality(samples);
      setQuality(res.data);
      return res.data;
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || i18n.t('hdf5viewer:scan.qualityFailed');
      // eslint-disable-next-line no-console
      console.warn('[useScanResults] scanQuality failed:', detail);
      setQuality({ error: detail });
      // intentionally not re-throwing: callers already render result.error
      return null;
    }
  }, []);

  const scanDefects = useCallback(async (samples = 60) => {
    setDefects('scanning');
    try {
      const res = await h5Api.scanDefects(samples);
      setDefects(res.data);
      return res.data;
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || i18n.t('hdf5viewer:scan.defectFailed');
      // eslint-disable-next-line no-console
      console.warn('[useScanResults] scanDefects failed:', detail);
      setDefects({ error: detail });
      // intentionally not re-throwing: callers already render result.error
      return null;
    }
  }, []);

  const clearResults = useCallback(() => {
    setQuality(null);
    setDefects(null);
  }, []);

  return { quality, defects, scanQuality, scanDefects, clearResults };
}

export default useScanResults;
