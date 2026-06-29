/**
 * useSourceLink — resolve the H5OINA source file for the active result.
 *
 * On result change: fetch ebsdApi.info() + analysisApi.status() and call
 * the pure resolveSource() function. Exposes a linkManual(path) method
 * for the user to pin a specific file as the source override.
 */
import { useState, useEffect, useCallback } from 'react';
import { resolveSource } from '../sourceResolve';
import { ebsdApi, analysisApi } from '../../../services/api';

export function useSourceLink({ resultEntry, resultShape }) {
  const [ebsdInfo, setEbsdInfo] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [manualOverride, setManualOverride] = useState(null);
  const [resolved, setResolved] = useState({
    linked: false, sourcePath: null, reason: 'Loading…', sourceShape: null,
  });

  // Refetch backend state whenever the active result changes
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      ebsdApi.info().catch(() => ({ data: null })),
      analysisApi.status().catch(() => ({ data: null })),
    ]).then(([info, status]) => {
      if (cancelled) return;
      setEbsdInfo(info?.data ?? null);
      setAnalysisStatus(status?.data ?? null);
    });
    return () => { cancelled = true; };
  }, [resultEntry?.id]);

  // Resolve whenever inputs change
  useEffect(() => {
    setResolved(resolveSource({
      resultEntry, ebsdInfo, analysisStatus, manualOverride, resultShape,
    }));
  }, [resultEntry, ebsdInfo, analysisStatus, manualOverride, resultShape]);

  const linkManual = useCallback((path, shape = null) => {
    setManualOverride({ path, shape });
  }, []);

  const clearManual = useCallback(() => setManualOverride(null), []);

  return { ...resolved, linkManual, clearManual };
}
