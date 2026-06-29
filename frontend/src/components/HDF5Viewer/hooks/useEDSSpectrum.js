import { useEffect, useState, useRef } from 'react';
import { h5Api } from '../../../services/api';

export function useEDSSpectrum(isFileOpen, hasEDS, row, col) {
  const [spectrum, setSpectrum] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const seq = useRef(0);

  useEffect(() => {
    if (!isFileOpen || !hasEDS || row == null || col == null) {
      setSpectrum(null); return;
    }
    const s = ++seq.current;
    setLoading(true);
    setError(null);
    h5Api.getEDSSpectrum(row, col)
      .then((res) => { if (s === seq.current) setSpectrum(res.data); })
      .catch((err) => {
        if (s !== seq.current) return;
        console.warn('[useEDSSpectrum]', err);
        setError(err.response?.data?.detail || err.message);
        setSpectrum(null);
      })
      .finally(() => { if (s === seq.current) setLoading(false); });
  }, [isFileOpen, hasEDS, row, col]);

  return { spectrum, loading, error };
}
