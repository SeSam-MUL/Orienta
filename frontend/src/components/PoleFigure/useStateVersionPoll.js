import { useEffect, useRef, useState } from 'react';
import { stateApi } from '../../services/api';

export function useStateVersionPoll(onChange, { intervalMs = 1500 } = {}) {
  const [snap, setSnap] = useState(null);
  const lastRef = useRef({ version: null, rid: undefined });
  const cbRef = useRef(onChange);
  useEffect(() => { cbRef.current = onChange; }, [onChange]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const { data } = await stateApi.version();
        if (cancelled) return;
        const changed = data.version !== lastRef.current.version
          || data.active_result_id !== lastRef.current.rid;
        if (changed) {
          lastRef.current = { version: data.version, rid: data.active_result_id };
          setSnap(data);
          cbRef.current?.(data);
        }
      } catch { /* backend down — keep polling */ }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [intervalMs]);

  return snap;
}
