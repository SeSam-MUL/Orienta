import { useEffect, useState } from 'react';
import { h5Api } from '../../../services/api';

export function useHeaders(isFileOpen) {
  const [ebsd, setEbsd] = useState(null);
  const [eds, setEds] = useState(null);

  useEffect(() => {
    if (!isFileOpen) { setEbsd(null); setEds(null); return; }
    let cancelled = false;
    Promise.all([
      h5Api.getEBSDHeader().catch(() => ({ data: null })),
      h5Api.getEDSHeader().catch(() => ({ data: null })),
    ]).then(([e, eds]) => {
      if (cancelled) return;
      setEbsd(e.data); setEds(eds.data);
    });
    return () => { cancelled = true; };
  }, [isFileOpen]);

  return { ebsd, eds };
}
