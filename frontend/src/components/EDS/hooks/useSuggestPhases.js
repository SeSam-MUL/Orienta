import { useCallback, useRef, useState } from 'react';
import { edsApi } from '../../../services/api';

/**
 * Phase Suggestion state for the EDS page.
 *
 * The suggestion is a per-pixel quantity, so the panel must never show a
 * result without saying which pixel produced it. Two entry points:
 *
 *   - ``run(row, col)``        — the user pressed "Suggest Phases".
 *   - ``followPixel(row, col)``— the user clicked another pixel.
 *
 * ``followPixel`` is a no-op until ``run`` has been called once, so merely
 * clicking around the map never fires unsolicited requests; once the panel
 * is showing something, it keeps up with the cursor instead of silently
 * describing the previously clicked pixel.
 */
export function useSuggestPhases() {
  const [suggestions, setSuggestions] = useState(null);
  const [pixel, setPixel] = useState(null);          // pixel `suggestions` belongs to
  const [atomicPct, setAtomicPct] = useState(null);  // measured At.% at that pixel
  const [mapPhase, setMapPhase] = useState(null);     // what the MAP says here
  const [librarySource, setLibrarySource] = useState(null);
  const [librarySize, setLibrarySize] = useState(null);
  // Why the list is empty, when it is. The backend sends a stable code plus
  // its numbers; absent on a successful answer, so it cannot become noise.
  const [cifNoMatch, setCifNoMatch] = useState(null);
  // CIFs that are on disk in Database/CIF_Library and did NOT become phases
  // (unreadable, empty composition, duplicate filename). The user put those
  // files there, so "the library never saw your phase" has to be visible in
  // the panel and not only in the backend log. [] when everything read.
  const [librarySkipped, setLibrarySkipped] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // True once the user has asked for suggestions; gates followPixel.
  const activeRef = useRef(false);
  // Cancel token for the in-flight request, so a slow reply for an old pixel
  // cannot overwrite a newer one when the user clicks around quickly.
  const inFlightRef = useRef(null);

  const run = useCallback(async (row, col) => {
    activeRef.current = true;
    if (inFlightRef.current) inFlightRef.current.cancel = true;
    const token = { cancel: false };
    inFlightRef.current = token;

    setLoading(true);
    setError(null);
    try {
      const res = await edsApi.suggestPhases(Number(row), Number(col));
      if (token.cancel) return;
      const data = res?.data || {};
      // Pixel and list are set together — the badge can never disagree
      // with the list it labels.
      setSuggestions(data.suggestions || data.phases || []);
      setPixel({ row: Number(row), col: Number(col) });
      setAtomicPct(data.atomic_pct || null);
      setMapPhase(data.map_phase || null);
      setLibrarySource(data.library_source || null);
      setLibrarySize(data.library_size ?? null);
      setCifNoMatch(data.cif_no_match || null);
      setLibrarySkipped(data.library_skipped || []);
    } catch (e) {
      if (token.cancel) return;
      setError(e?.response?.data?.detail || e?.message || 'Phase suggestion failed.');
      // Drop the stale list rather than leave it under a fresh error.
      setSuggestions(null);
      setPixel(null);
      setAtomicPct(null);
      setMapPhase(null);
      setLibrarySource(null);
      setLibrarySize(null);
      setCifNoMatch(null);
      setLibrarySkipped([]);
    } finally {
      if (!token.cancel) setLoading(false);
    }
  }, []);

  const followPixel = useCallback((row, col) => {
    if (!activeRef.current) return undefined;
    return run(row, col);
  }, [run]);

  const clear = useCallback(() => {
    activeRef.current = false;
    if (inFlightRef.current) inFlightRef.current.cancel = true;
    setSuggestions(null);
    setPixel(null);
    setAtomicPct(null);
    setMapPhase(null);
    setLibrarySource(null);
    setLibrarySize(null);
    setCifNoMatch(null);
    setLibrarySkipped([]);
    setError(null);
    setLoading(false);
  }, []);

  return {
    suggestions, pixel, atomicPct, mapPhase, librarySource, librarySize,
    cifNoMatch, librarySkipped, loading, error,
    run, followPixel, clear,
  };
}
