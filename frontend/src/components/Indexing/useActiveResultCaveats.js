/**
 * useActiveResultCaveats - the notes (caveats) of the result that is ACTIVE now.
 *
 * A result becomes active in several places: a run on this page, a gallery click on
 * the Phase Map page, an import of an .h5, a refinement. All of them bump the
 * backend's state version, so while `enabled` the hook polls it and, whenever it (or
 * the active result id) changes, asks `/result/last` for the caveats of whatever is
 * active. A result without notes, no result at all, or a failing `/result/last`
 * leaves the banner empty: it never keeps the note of a result that is no longer the
 * active one. A failing state-version request changes nothing (the backend may be
 * restarting).
 *
 * `apply(result)` takes the caveats from an answer the page already holds (the one
 * after a run) and `clear()` empties them (a run starts); both win over an answer
 * that is still on its way.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { indexApi, stateApi } from '../../services/api';
import { caveatsFrom } from './ResultCaveats';

export default function useActiveResultCaveats({ enabled = true, intervalMs = 1500 } = {}) {
  const [caveats, setCaveats] = useState([]);
  const seq = useRef(0);

  const apply = useCallback((result) => {
    seq.current += 1;
    setCaveats(caveatsFrom(result));
  }, []);
  const clear = useCallback(() => {
    seq.current += 1;
    setCaveats([]);
  }, []);

  useEffect(() => {
    if (!enabled) {
      seq.current += 1; // an answer still on its way belongs to a state we left
      return undefined;
    }
    let cancelled = false;
    let seen = null;
    let busy = false; // a slow answer must not be replaced by a new request every tick
    const tick = async () => {
      if (busy) return;
      busy = true;
      try {
        await check();
      } finally {
        busy = false;
      }
    };
    const check = async () => {
      let snap;
      try {
        snap = (await stateApi.version()).data;
      } catch {
        return;
      }
      if (cancelled) return;
      const key = `${snap?.version}|${snap?.active_result_id}`;
      if (key === seen) return;
      const mine = ++seq.current;
      let next = [];
      try {
        const { data } = await indexApi.getLastResult();
        next = caveatsFrom(data); // {available: false} has none
        seen = key;
      } catch {
        // unknown which result is active: say nothing, ask again at the next tick
      }
      if (!cancelled && mine === seq.current) setCaveats(next);
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled, intervalMs]);

  return { caveats, apply, clear };
}
