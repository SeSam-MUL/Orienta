import { useEffect, useState, useCallback } from 'react';
import { poleFigureApi } from '../../services/api';

export function usePoleFigure({ phaseId, hkl, mode = 'both', subsample = 20000, refreshKey = 0 }) {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    if (phaseId == null) { setImage(null); return () => {}; }
    let cancelled = false;
    setLoading(true); setError(null);
    poleFigureApi.image({ phaseId, hkl, mode, subsample })
      .then((r) => { if (!cancelled) { setImage(r.data.image); setLoading(false); } })
      .catch((e) => { if (!cancelled) { setError(e?.response?.data?.detail ?? e.message); setImage(null); setLoading(false); } });
    return () => { cancelled = true; };
  }, [phaseId, hkl, mode, subsample]);

  useEffect(() => load(), [load, refreshKey]);
  return { image, loading, error, reload: load };
}
