/**
 * The EBSD viewer's "EDS Composition" box: one pixel's composition and phase
 * suggestions, and the spectrum fit they wait for.
 *
 * On a file with a spectrum cube every EDS number answers 503 until the
 * spectra are evaluated. The viewer asks for pixel (0, 0) right after every
 * load -- during the fit -- and used to keep that refusal: "No EDS data
 * loaded" and a disabled "Send Filter to Indexing" on a file with EDS,
 * until the cursor moved. Here the fit's state is
 * watched: while the numbers cannot exist yet the box says why and asks
 * nothing (the request would only be refused -- two 503s per load, each a
 * console error), and once they do, the pixel asked for meanwhile is asked.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { edsApi } from '../../services/api';
import { useSpectrumFitStatus } from '../EDS/hooks/useSpectrumFitStatus';

const MODE_PARAM = { Counts: 'counts', 'Wt.%': 'wt_pct', 'At.%': 'at_pct' };

/**
 * Can EDS numbers be asked for under this fit state? ``"ready"`` (the
 * spectra are evaluated) and ``"none"`` (a file without a cube: window
 * numbers) -- yes. ``null`` (no answer for this file yet), ``"running"``
 * and ``"failed"`` -- no: every EDS number of a file with a cube answers 503
 * or 500 then, and the box already says which.
 */
export function edsNumbersExist(state) {
  return state === 'ready' || state === 'none';
}

/**
 * `{composition, phases, fetchEds, fetchEdsAfterLoad, fit}` for the open file
 * ``filePath``. ``fetchEdsAfterLoad`` is ``fetchEds`` for the pixel a LOAD asks
 * for: it asks the fit's state again first (see below).
 *
 * ``fetchEds(row, col)`` asks for a pixel; an answer that arrives after a
 * later request is dropped, so a slow one never paints over the pixel the
 * cursor moved to. While {@link edsNumbersExist} says no, it only remembers
 * the pixel (and clears the box, so no other file's numbers stay on it);
 * when the state turns ``"ready"`` or ``"none"``, the remembered pixel is
 * asked. ``fit`` is {@link useSpectrumFitStatus}'s state (asked only while
 * ``active``).
 */
export function useViewerEds({ filePath, active, edsMode }) {
  const [composition, setComposition] = useState(null);
  const [phases, setPhases] = useState(null);
  // Bumped by fetchEdsAfterLoad: the status is asked again after every load.
  const [recheck, setRecheck] = useState(0);
  const fit = useSpectrumFitStatus(filePath || null, { active: !!active, recheck });
  // Read through a ref: the viewer calls fetchEds from load callbacks that
  // were made before the new file's fit state existed. Kept current by the
  // effect below, which runs before any handler can call fetchEds again.
  const fitStateRef = useRef(fit.state);
  const pendingPixel = useRef(null);
  const requestId = useRef(0);

  const fetchEds = useCallback(async (r, c) => {
    const mine = ++requestId.current;
    if (!edsNumbersExist(fitStateRef.current)) {
      pendingPixel.current = [r, c];
      setComposition(null);
      setPhases(null);
      return;
    }
    pendingPixel.current = null;
    try {
      const res = await edsApi.quantifyPixel(r, c, MODE_PARAM[edsMode] || 'at_pct');
      if (mine === requestId.current) setComposition(res.data);
    } catch {
      if (mine === requestId.current) setComposition(null);
    }
    try {
      const res = await edsApi.suggestPhases(r, c);
      if (mine === requestId.current) setPhases(res.data);
    } catch {
      if (mine === requestId.current) setPhases(null);
    }
  }, [edsMode]);

  // After a load the viewer asks for a pixel of the file it just (re)loaded.
  // A load of the SAME path can start a new fit (the file changed on disk,
  // its result was dropped for newer files, a failed fit is retried), and the
  // status for that path would still say "ready": the request would be
  // refused. So the state is unknown again and asked for afresh; the pixel
  // waits for the answer like any other.
  const fetchEdsAfterLoad = useCallback((r, c) => {
    fitStateRef.current = null;
    setRecheck((n) => n + 1);
    return fetchEds(r, c);
  }, [fetchEds]);

  // The numbers exist now: ask for the pixel asked for while they did not.
  const fetchRef = useRef(fetchEds);
  useEffect(() => { fetchRef.current = fetchEds; }, [fetchEds]);
  useEffect(() => {
    fitStateRef.current = fit.state;
    if (edsNumbersExist(fit.state) && pendingPixel.current) {
      fetchRef.current(...pendingPixel.current);
    }
  }, [fit.state]);

  return { composition, phases, fetchEds, fetchEdsAfterLoad, fit };
}

/**
 * The text of the composition box: the pixel's numbers, or why there are
 * none. ``watching``: a file is loaded (otherwise "not known yet" is not
 * "checking", it is "no file"). ``t`` is the viewer's; the fit's sentences are
 * the EDS page's own (``eds:spectrumFit.*``).
 */
export function edsCompositionText(composition, edsMode, fit, watching, t) {
  if (composition?.data) {
    const modeKey = MODE_PARAM[edsMode] || 'at_pct';
    return Object.entries(composition.data).map(([el, vals]) => {
      const v = vals[modeKey] ?? vals.counts ?? 0;
      return `${el}: ${typeof v === 'number' ? v.toFixed(1) : v}${edsMode !== 'Counts' ? '%' : ''}`;
    }).join(' | ');
  }
  if (watching) {
    if (fit?.state === 'running') {
      return t('eds:spectrumFit.running', {
        percent: Math.round(100 * (Number(fit.progress) || 0)) });
    }
    if (fit?.state === 'failed') {
      return t('eds:spectrumFit.failed', { error: fit.error || '?' });
    }
    if (fit?.state == null) {
      return t(fit?.unreachable ? 'eds:spectrumFit.unreachable' : 'eds:spectrumFit.checking');
    }
  }
  return t('eds.noData');
}
