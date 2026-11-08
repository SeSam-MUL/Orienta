import { useCallback, useEffect, useRef, useState } from 'react';
import { pcApi } from '../../services/api';

/**
 * The number on "Open PC Refinement": how many calibration patterns the PC
 * Refinement list holds.
 *
 * The list lives in the backend and two pages change it - this one adds, the
 * PC Refinement page removes. The number used to be set only from Add's reply,
 * so a pattern removed on the other page stayed counted. Now the backend is
 * asked whenever this page is shown again, and Add still sets the number from
 * its own reply at once.
 *
 * An answer to an older question never overwrites a newer number: a status
 * reply that arrives after an Add (or after a later status question) is dropped.
 *
 * The list belongs to the active dataset, so it is asked again when that
 * changes while the page stays on screen (processing with deep copy makes a
 * new dataset whose list is empty).
 *
 * @param {string} page the router id of the page this hook lives on
 * @param {string|null} [dataset] the active dataset; a change re-asks the backend
 * @returns {[number, (n: number) => void]} the count and a setter for Add
 */
export function usePcPatternCount(page, dataset = null) {
  const [count, setCountState] = useState(0);
  const seq = useRef(0);

  const setCount = useCallback((n) => {
    seq.current += 1;                 // any status question still out is now stale
    setCountState(n);
  }, []);

  const refresh = useCallback(() => {
    const asked = ++seq.current;
    pcApi.status().then((res) => {
      if (asked !== seq.current) return;
      const list = res?.data?.patterns;
      setCountState(Array.isArray(list) ? list.length : 0);
    }).catch(() => {});              // no backend yet: keep the number shown
  }, []);

  useEffect(() => {
    const onPage = (e) => { if (e.detail?.page === page) refresh(); };
    window.addEventListener('page-changed', onPage);
    return () => window.removeEventListener('page-changed', onPage);
  }, [page, refresh]);

  // On mount, and whenever the active dataset changes.
  useEffect(() => { refresh(); }, [dataset, refresh]);

  return [count, setCount];
}
