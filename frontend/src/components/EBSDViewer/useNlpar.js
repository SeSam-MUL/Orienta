/**
 * NLPAR state and the one processing step, kept out of EBSDViewer.jsx.
 *
 * The viewer owns the busy overlay, the progress poll (`withProgress`) and the
 * auto-copy before every in-place operation. This hook owns what is specific to
 * NLPAR: the search radius, the "denoise first in the Recommended Pipeline"
 * choice, the request id the Cancel button needs, and the log lines.
 *
 * Lambda (the averaging strength) is never a user setting: the backend
 * estimates it from the scan.
 */
import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { ebsdApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';

export const MIN_RADIUS = 1;
export const MAX_RADIUS = 3;
export const DEFAULT_RADIUS = 3;

/** Reasons after which the backend would certainly refuse a run (400). */
const REFUSED_REASONS = ['too_small', 'hex', 'uniform'];

/**
 * What a processing op does about NLPAR, decided BEFORE the viewer's auto-copy so that a
 * refused run leaves nothing behind.
 *  - 'run'     NLPAR is the first step
 *  - 'skip'    the Recommended Pipeline with "denoise first" while NLPAR cannot run: the
 *              pipeline goes on without it (and says so)
 *  - 'refuse'  the NLPAR button itself while it cannot run: nothing happens
 *  - null      the op has nothing to do with NLPAR
 */
export function nlparStepFor(op, { denoiseFirst, canRun }) {
  if (op === 'nlpar') return canRun ? 'run' : 'refuse';
  if (op === 'pipeline' && denoiseFirst) return canRun ? 'run' : 'skip';
  return null;
}

/** A radius the backend accepts, from whatever the input field holds. */
export function clampRadius(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return DEFAULT_RADIUS;
  return Math.min(MAX_RADIUS, Math.max(MIN_RADIUS, n));
}

/**
 * @param {object} state what the viewer knows: `ebsdLoaded`, `activeDataset`, `busy`
 *   (a processing op is running). The advice is fetched when a dataset becomes the
 *   active one and again when an op has finished, never while one runs.
 */
export function useNlpar({ ebsdLoaded = false, activeDataset = null, busy = false } = {}) {
  const { t } = useTranslation('ebsdviewer');
  const filePath = useDataStore((s) => s.filePath);
  // null = being checked; otherwise the answer of GET /api/ebsd/nlpar/advice
  const [advice, setAdvice] = useState(null);
  const [radius, setRadiusState] = useState(DEFAULT_RADIUS);
  const [denoiseFirst, setDenoiseFirst] = useState(false);
  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  // The step closes over nothing that changes: it reads the latest radius from
  // a ref, so the viewer's doProcessing does not need it as a dependency.
  const radiusRef = useRef(radius);
  radiusRef.current = radius;
  const ridRef = useRef(null);

  const setRadius = useCallback((v) => setRadiusState(clampRadius(v)), []);

  useEffect(() => {
    if (!ebsdLoaded) { setAdvice(null); return undefined; }
    if (busy) return undefined;
    let stale = false;
    setAdvice(null);
    ebsdApi.nlparAdvice()
      .then((r) => { if (!stale) setAdvice(r.data || { recommendation: 'unavailable', reason: 'error' }); })
      .catch(() => { if (!stale) setAdvice({ recommendation: 'unavailable', reason: 'error' }); });
    return () => { stale = true; };
  }, [ebsdLoaded, busy, activeDataset, filePath]);

  const run = useCallback(async (rid) => {
    ridRef.current = rid;
    try {
      return await ebsdApi.nlpar(radiusRef.current, rid);
    } finally {
      ridRef.current = null;
    }
  }, []);

  const cancel = useCallback(async () => {
    const rid = ridRef.current;
    if (!rid) return;
    setCancelling(true);
    try {
      await ebsdApi.nlparCancel(rid);
    } catch { /* the run ends on its own either way */ }
  }, []);

  /**
   * One NLPAR step inside the viewer's `doProcessing`.
   * Returns false when the user cancelled (the caller stops the pipeline).
   */
  const runStep = useCallback(async ({ timed, withProgress, log }) => {
    setRunning(true);
    setCancelling(false);
    try {
      const res = await timed(
        t('nlpar.stepName', { radius: radiusRef.current }),
        () => withProgress((rid) => run(rid), { phase: 'compile' }),
      );
      if (res?.data?.cancelled) {
        log(t('nlpar.cancelled'));
        return false;
      }
      const d = res?.data || {};
      // The dataset now carries the mark: say so at once, not after the next fetch,
      // so there is no moment in which the control invites a second run.
      setAdvice({ recommendation: 'applied', applied: true, applied_radius: d.radius });
      if (typeof d.lam === 'number') {
        log(t('nlpar.result', { lam: d.lam.toFixed(2), sigma: Number(d.sigma_median).toFixed(1), radius: d.radius }));
      }
      return true;
    } finally {
      setRunning(false);
      setCancelling(false);
    }
  }, [t, run]);

  const applied = advice?.applied === true;
  // Not while the answer is pending: a click then could still hit an applied dataset.
  const refused = advice?.recommendation === 'unavailable' && REFUSED_REASONS.includes(advice.reason);
  const canRun = advice !== null && !applied && !refused;

  return useMemo(() => ({
    radius, setRadius, denoiseFirst, setDenoiseFirst,
    running, cancelling, run, cancel, runStep, advice, applied, canRun,
  }), [radius, setRadius, denoiseFirst, running, cancelling, run, cancel, runStep, advice, applied, canRun]);
}
