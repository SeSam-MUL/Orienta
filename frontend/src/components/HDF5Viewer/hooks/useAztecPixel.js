import { useEffect, useState, useRef } from 'react';
import { h5Api } from '../../../services/api';

export function useAztecPixel(isFileOpen, row, col) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);

  useEffect(() => {
    if (!isFileOpen || row == null || col == null) {
      setData(null); return;
    }
    const s = ++seq.current;
    setLoading(true);
    h5Api.getAztecPixel(row, col)
      .then((res) => { if (s === seq.current) setData(res.data); })
      .catch((err) => {
        if (s !== seq.current) return;
        console.warn('[useAztecPixel]', err);
        setData(null);
      })
      .finally(() => { if (s === seq.current) setLoading(false); });
  }, [isFileOpen, row, col]);

  return { data, loading };
}
