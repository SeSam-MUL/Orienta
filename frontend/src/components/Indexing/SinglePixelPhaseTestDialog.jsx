import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi, ebsdApi, edsApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import LinkedPatternImage from '../PatternMatch/LinkedPatternImage';
import { useLinkedPatternMarkers } from '../PatternMatch/useLinkedPatternMarkers';
import PatternExportDialog from '../PatternMatch/PatternExportDialog';

const C = {
  bg: '#282a36', bgSecondary: '#21222c', border: '#44475a', text: '#f8f8f2',
  textSecondary: '#6272a4', accent: '#bd93f9', green: '#50fa7b',
  orange: '#ffb86c', red: '#ff5555',
};
const rColor = (r) => (r == null ? C.textSecondary : r >= 0.3 ? C.green : r >= 0.15 ? C.orange : C.red);
const img = (b64) => (b64 ? `data:image/png;base64,${b64}` : null);
// Distinct tints for stacked EDS element overlays on the scan picker.
const EDS_COLORS = ['#ff5555', '#50fa7b', '#8be9fd', '#ffb86c', '#bd93f9', '#f1fa8c', '#ff79c6', '#62d6e8'];
const edsSym = (el) => (typeof el === 'string' ? el : (el?.element || el?.symbol || el?.name || ''));

export default function SinglePixelPhaseTestDialog({ open, onClose, currentMethod = 'hough', onUsePhase }) {
  const { t } = useTranslation('indexing');
  const currentIndex = useDataStore((s) => s.currentIndex) || 0;
  const gridShape = useDataStore((s) => s.gridShape) || [0, 0];
  const setPosition = useDataStore((s) => s.setPosition);
  const fileLoaded = useDataStore((s) => s.ebsdLoaded || s.isFileOpen);
  const filePath = useDataStore((s) => s.filePath);

  const [nRows, nCols] = gridShape;

  const [pixelIndex, setPixelIndex] = useState(currentIndex);
  // Simulation bandwidth (SHT) — drives the actual compute. Higher = sharper
  // simulated bands (better discrimination), slower. Switching this builds a
  // new per-(phaseset, geometry, bandwidth) backend (cold on first use).
  const [bandwidth, setBandwidth] = useState(128);
  const [edsMode, setEdsMode] = useState('filter');
  // Dynamic-background-remove the MEASURED pattern before comparison. The
  // simulated dynamical pattern is already background-free, so removing the
  // measured pattern's background makes the NCC correlate band-vs-band (better
  // R + discrimination) and reveals the Kikuchi bands the phase-ID relies on.
  // Default ON — without it the comparison degrades and the phase is misranked.
  const [bgRemove, setBgRemove] = useState(true);
  const [aperture, setAperture] = useState('circular');
  const [apertureRadius, setApertureRadius] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(null);
  const [showExcluded, setShowExcluded] = useState(false);

  // Linked crosshair + numbered markers shared across the three comparison
  // panels, plus the publication export composer (same as the Pattern Match
  // dialogs in IndexingPage / PhaseMapPage).
  const markerCtl = useLinkedPatternMarkers();
  const [exportOpen, setExportOpen] = useState(false);

  // Transient confirmation shown under the "Use for indexing" buttons after a
  // phase is added, so the click gives visible feedback now that the dialog no
  // longer closes on add (the user keeps it open to collect phases across
  // multiple pixels). ``{ formula, method, status }``; auto-clears after ~2.6 s.
  const [justUsed, setJustUsed] = useState(null);
  const justUsedTimerRef = useRef(null);
  const markUsed = useCallback((formula, methodLabel, status) => {
    setJustUsed({ formula, method: methodLabel, status });
    if (justUsedTimerRef.current) clearTimeout(justUsedTimerRef.current);
    justUsedTimerRef.current = setTimeout(() => setJustUsed(null), 2600);
  }, []);

  // Phase picker: the library phases we *can* test, plus the user's
  // current selection. Defaults to ALL phases selected once they load.
  const [phases, setPhases] = useState([]);
  const [selectedKeys, setSelectedKeys] = useState(() => new Set());
  const [showPhases, setShowPhases] = useState(false);

  // Background-job state. ``progress`` mirrors the backend poll response.
  const [progress, setProgress] = useState(null);
  // Poll interval id + a monotonic run id so a stale poll from a previous
  // job can never write into a newer run's state.
  const pollRef = useRef(null);
  const runIdRef = useRef(0);
  const jobIdRef = useRef(null);
  // The id of the LAST COMPLETED run. Unlike jobIdRef (cleared on 'done'),
  // this stays set after a run finishes so /remask can reuse the cached
  // patterns to recompute masked R without a full re-render.
  const lastJobIdRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  // Real scan overview (Band Contrast) — fetched once when the dialog
  // opens with a file loaded. Used as the ScanPicker background. Uses the
  // 'bc' mode so this picker shows the SAME crisp band-contrast map as the
  // Indexing Navigation Map / EBSD Viewer overview (native Oxford BC, with a
  // computed FFT image-quality fallback), instead of a washed-out per-pixel
  // mean-intensity map.
  const [overviewB64, setOverviewB64] = useState(null);
  // Measured pattern at the active pixel — fetched whenever pixelIndex
  // changes so the user SEES the pattern before running the test.
  const [previewB64, setPreviewB64] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  // Monotonic request counter so a fast pixel change discards a stale
  // (slower) preview response instead of flashing the wrong pattern.
  const previewReqRef = useRef(0);
  // Surfaced fetch error for the measured-pattern preview, so a failed fetch
  // shows "preview failed" instead of silently reading as "no pattern".
  const [previewError, setPreviewError] = useState(null);

  // EDS element overlays on the scan picker — the user can toggle MULTIPLE
  // elements on at once (each tinted a distinct colour) to pick pixels by
  // chemistry. ``edsOverlaySel`` is the ordered list of active symbols;
  // ``edsOverlayMaps`` caches each element's fetched (colour-tinted) map.
  const [edsElements, setEdsElements] = useState([]);
  const [edsOverlaySel, setEdsOverlaySel] = useState([]);
  const [edsOverlayMaps, setEdsOverlayMaps] = useState({});
  const [edsOpacity, setEdsOpacity] = useState(0.6);

  // Live per-pixel EDS chemistry for the ACTIVE crosshair pixel — fetched on
  // every pixel change so the EDS readout (and the user's sense of what the
  // chemistry filter will keep/drop) follows the crosshair, instead of being
  // frozen on the pixel of the last completed run. ``null`` = no EDS / loading.
  const [livePixelAtPct, setLivePixelAtPct] = useState(null);
  const chemReqRef = useRef(0);

  useEffect(() => { if (typeof currentIndex === 'number') setPixelIndex(currentIndex); }, [currentIndex]);

  // Fetch the scan overview once when the dialog opens with a file loaded.
  useEffect(() => {
    if (!open || !fileLoaded) return;
    let cancelled = false;
    ebsdApi.overview('bc')
      .then((r) => { if (!cancelled) setOverviewB64(r?.data?.image || null); })
      .catch(() => { /* silently fall back to the empty picker */ });
    return () => { cancelled = true; };
  }, [open, fileLoaded, filePath]);

  // Fetch the testable library phases once when the dialog opens. Default
  // the selection to ALL of them (so the run tests every phase by default).
  useEffect(() => {
    if (!open || !fileLoaded) return;
    let cancelled = false;
    indexApi.phaseTestPhases()
      .then((r) => {
        if (cancelled) return;
        const list = Array.isArray(r?.data) ? r.data : [];
        setPhases(list);
        setSelectedKeys(new Set(list.map((p) => p.key)));
      })
      .catch(() => { if (!cancelled) setPhases([]); });
    return () => { cancelled = true; };
  }, [open, fileLoaded, filePath]);

  // Stop polling + invalidate the run when the dialog closes or unmounts so
  // a late poll response can't setState after close/unmount.
  useEffect(() => {
    if (!open) { runIdRef.current += 1; stopPolling(); }
  }, [open, stopPolling]);
  useEffect(() => () => {
    runIdRef.current += 1; stopPolling();
    if (justUsedTimerRef.current) clearTimeout(justUsedTimerRef.current);
  }, [stopPolling]);

  // Fetch the measured pattern for the active pixel whenever it changes.
  useEffect(() => {
    if (!open || !fileLoaded || nCols <= 0) { setPreviewB64(null); setPreviewError(null); return; }
    const row = Math.floor(pixelIndex / nCols);
    const col = pixelIndex % nCols;
    const myReq = ++previewReqRef.current;
    let cancelled = false;
    setPreviewLoading(true); setPreviewError(null);
    // When BG-removal is on, fetch the preview through the dynamic-BG display
    // filter so the previewed measured pattern matches what the NCC sees.
    const cfg = bgRemove ? { params: { display_filter: 'Dynamic BG' } } : undefined;
    ebsdApi.getPattern(row, col, cfg)
      .then((res) => {
        if (cancelled || myReq !== previewReqRef.current) return;  // stale
        setPreviewB64(res?.data?.image || null);
      })
      .catch((err) => {
        if (cancelled || myReq !== previewReqRef.current) return;
        setPreviewB64(null);
        setPreviewError(err?.response?.data?.detail || err?.message || 'fetch failed');
      })
      .finally(() => {
        if (!cancelled && myReq === previewReqRef.current) setPreviewLoading(false);
      });
    return () => { cancelled = true; };
  }, [open, fileLoaded, pixelIndex, nCols, bgRemove, filePath]);

  // Fetch the live EDS chemistry for the active pixel whenever it changes, so
  // the EDS readout tracks the crosshair (not the last run's pixel).
  useEffect(() => {
    if (!open || !fileLoaded) { setLivePixelAtPct(null); return; }
    const myReq = ++chemReqRef.current;
    let cancelled = false;
    indexApi.phaseTestPixelChemistry(pixelIndex)
      .then((res) => {
        if (cancelled || myReq !== chemReqRef.current) return;  // stale
        setLivePixelAtPct(res?.data?.pixel_at_pct || null);
      })
      .catch(() => {
        if (cancelled || myReq !== chemReqRef.current) return;
        setLivePixelAtPct(null);
      });
    return () => { cancelled = true; };
  }, [open, fileLoaded, pixelIndex, filePath]);

  // Fetch the list of EDS elements once when the dialog opens, to populate the
  // scan-picker overlay selector. Empty list → the overlay control hides.
  useEffect(() => {
    if (!open || !fileLoaded) { setEdsElements([]); return; }
    let cancelled = false;
    edsApi.elements()
      .then((r) => { if (!cancelled) setEdsElements(Array.isArray(r?.data?.elements) ? r.data.elements : []); })
      .catch(() => { if (!cancelled) setEdsElements([]); });
    return () => { cancelled = true; };
  }, [open, fileLoaded, filePath]);

  // Toggle an EDS element overlay on/off. On first activation, fetch its
  // colour-tinted element map (cached in edsOverlayMaps; tint is stable per
  // element so a colour always means the same element).
  const toggleEdsEl = useCallback((sym) => {
    setEdsOverlaySel((sel) => (sel.includes(sym) ? sel.filter((x) => x !== sym) : [...sel, sym]));
    if (!edsOverlayMaps[sym]) {
      const idx = Math.max(0, edsElements.findIndex((e) => edsSym(e) === sym));
      const color = EDS_COLORS[idx % EDS_COLORS.length];
      edsApi.getMap(sym, 'counts', 'hot', color)
        .then((r) => { if (r?.data?.image) setEdsOverlayMaps((m) => ({ ...m, [sym]: r.data.image })); })
        .catch(() => { /* overlay just won't show for this element */ });
    }
  }, [edsOverlayMaps, edsElements]);

  // Bug fix: when the ACTIVE FILE changes, drop everything cached from the
  // previous file (result/preview/overview/phases) so a stale pattern from the
  // old file can never linger; the per-file effects above then refetch fresh.
  useEffect(() => {
    runIdRef.current += 1; stopPolling();
    setResult(null); setSelected(null); setProgress(null); setError(null);
    setPreviewB64(null); setPreviewError(null); setOverviewB64(null);
    setEdsOverlaySel([]); setEdsOverlayMaps({}); setEdsElements([]);
    setPhases([]); setSelectedKeys(new Set());
    markerCtl.clearMarkers(); markerCtl.setHover(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filePath]);

  // Markers live in the active pixel's detector frame — clear them when the
  // pixel changes (a new measured pattern). They persist across phase changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { markerCtl.clearMarkers(); markerCtl.setHover(null); }, [pixelIndex]);

  const previewRow = nCols > 0 ? Math.floor(pixelIndex / nCols) : 0;
  const previewCol = nCols > 0 ? pixelIndex % nCols : 0;

  // A strict subset is selected → pass phase_keys; otherwise omit it so the
  // backend tests every phase (its default).
  const allSelected = phases.length === 0 || selectedKeys.size === phases.length;
  const subsetSelected =
    phases.length > 0 && selectedKeys.size > 0 && selectedKeys.size < phases.length;
  const running = progress?.status === 'running';

  const runAuto = useCallback(async () => {
    // Invalidate any in-flight job, then start a fresh one.
    const myRun = ++runIdRef.current;
    stopPolling();
    setLoading(true); setError(null); setSelected(null); setProgress(null);

    const params = {
      pixel_index: pixelIndex, eds_weighting: edsMode,
      aperture, aperture_radius: apertureRadius, max_bandwidth: bandwidth,
      bg_remove: bgRemove,
    };
    if (subsetSelected) params.phase_keys = [...selectedKeys];

    try {
      const res = await indexApi.phaseTestStart(params);
      if (myRun !== runIdRef.current) return;  // superseded while awaiting
      const jobId = res.data.job_id;
      jobIdRef.current = jobId;
      lastJobIdRef.current = jobId;  // kept after 'done' for /remask reuse
      // Show the experimental preview + total immediately (partial result).
      setResult(res.data);
      setProgress({ done: 0, total: res.data.total, current: null, status: 'running' });

      const tick = async () => {
        if (myRun !== runIdRef.current) return;
        try {
          const pr = await indexApi.phaseTestProgress(jobId);
          if (myRun !== runIdRef.current) return;
          const d = pr.data || {};
          setProgress({ done: d.done, total: d.total, current: d.current, status: d.status });
          if (d.status === 'done') {
            stopPolling();
            const full = d.result || {};
            setResult(full);
            setSelected(full.candidates?.[0] || null);
            jobIdRef.current = null;
          } else if (d.status === 'error') {
            stopPolling();
            setError(d.error || t('phaseTest.failed'));
            jobIdRef.current = null;
          } else if (d.status === 'cancelled') {
            stopPolling();
            setError(t('phaseTest.cancelled'));
            jobIdRef.current = null;
          }
        } catch (err) {
          if (myRun !== runIdRef.current) return;
          stopPolling();
          setError(err?.response?.data?.detail || err?.message || t('phaseTest.failed'));
          jobIdRef.current = null;
        }
      };
      // Poll immediately, then every 500 ms (the immediate first poll keeps
      // tests deterministic without fake timers).
      tick();
      pollRef.current = setInterval(tick, 500);
    } catch (err) {
      if (myRun !== runIdRef.current) return;
      setError(err?.response?.data?.detail || err?.message || t('phaseTest.failed'));
      setResult(null); setSelected(null); setProgress(null);
    } finally {
      if (myRun === runIdRef.current) setLoading(false);
    }
  }, [pixelIndex, edsMode, aperture, apertureRadius, bandwidth, bgRemove, subsetSelected, selectedKeys, stopPolling, t]);

  const cancelRun = useCallback(async () => {
    const jobId = jobIdRef.current;
    if (!jobId) return;
    try { await indexApi.phaseTestCancel(jobId); } catch { /* poll will observe state */ }
  }, []);

  // Mask-only change (toggle or radius) → recompute masked R + diff/simulated
  // PNGs from the LAST run's cached patterns via /remask (cheap, no re-render).
  // Only valid after a run completed (lastJobIdRef + result set) and not mid-run.
  // On a 404 (cache expired) we fall back to a full re-run so the user isn't stuck.
  const remask = useCallback(async (apertureOverride, radiusOverride) => {
    const ap = apertureOverride ?? aperture;
    const rad = radiusOverride ?? apertureRadius;
    const jobId = lastJobIdRef.current;
    if (!jobId || !result || running || loading) return;  // need a completed run
    try {
      const res = await indexApi.phaseTestRemask({ job_id: jobId, aperture: ap, aperture_radius: rad });
      const full = res.data;
      setResult(full);
      // Keep the user's currently-selected phase selected if it still exists,
      // otherwise fall back to the top candidate.
      setSelected((prev) => {
        const keep = prev && full.candidates?.find((c) => c.phase_key === prev.phase_key);
        return keep || full.candidates?.[0] || null;
      });
    } catch (err) {
      if (err?.response?.status === 404) { runAuto(); }  // cache expired → full re-run
      else { setError(err?.response?.data?.detail || err?.message || t('phaseTest.remaskFailed')); }
    }
  }, [aperture, apertureRadius, result, running, loading, runAuto, t]);

  const toggleKey = useCallback((key) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    setSelectedKeys((prev) =>
      prev.size === phases.length ? new Set() : new Set(phases.map((p) => p.key)));
  }, [phases]);

  if (!open) return null;

  const cands = result?.candidates || [];
  const excluded = result?.excluded || [];

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex',
      alignItems: 'center', justifyContent: 'center', zIndex: 999 }} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: C.bgSecondary,
        border: `1px solid ${C.border}`, borderRadius: 8, padding: 20, width: 920,
        maxHeight: '92vh', overflow: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <span style={{ color: C.accent, fontWeight: 700, fontSize: 15 }}>{t('phaseTest.title')}</span>
          <button onClick={onClose} title={t('hoverTips.phaseTestClose')} aria-label={t('hoverTips.phaseTestClose')} style={{ background: 'transparent', border: 'none',
            color: C.textSecondary, cursor: 'pointer', fontSize: 18 }}>&times;</button>
        </div>

        {/* 2-column body: LEFT = picker + pixel input + measured preview,
            RIGHT = EDS/mask/run controls + comparison + ranked list. */}
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>

        {/* LEFT column: scan picker + pixel index + measured pattern preview */}
        <div style={{ width: 300, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: 10 }}>
            <div style={{ fontSize: 11, color: C.textSecondary, marginBottom: 6 }}>
              {nRows > 0 && nCols > 0
                ? t('phaseTest.clickScan', { rows: nRows, cols: nCols })
                : t('phaseTest.scanUnknown')}
            </div>
            {nRows > 0 && nCols > 0 ? (
              <ScanPicker
                gridShape={gridShape}
                pixelIndex={pixelIndex}
                overviewB64={overviewB64}
                edsOverlays={edsOverlaySel.map((s) => edsOverlayMaps[s]).filter(Boolean)}
                edsOpacity={edsOpacity}
                t={t}
                onPick={(r, c) => {
                  const idx = r * nCols + c;
                  setPixelIndex(idx);
                  // Sync the global store so EBSD Viewer / Phase Map jump to
                  // the same pixel when the user switches tabs.
                  if (setPosition) setPosition(r, c, idx, null);
                }}
              />
            ) : (
              <div style={{ height: 120, display: 'flex', alignItems: 'center',
                justifyContent: 'center', color: C.textSecondary, fontSize: 10, opacity: 0.6 }}>
                {t('phaseTest.loadingOverview')}
              </div>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
              <span style={{ fontSize: 11, color: C.textSecondary }}>{t('phaseTest.pixelIndex')}</span>
              <input type="number" value={pixelIndex} min={0}
                onChange={(e) => setPixelIndex(parseInt(e.target.value, 10) || 0)}
                title={t('hoverTips.phaseTestPixelIndex')}
                style={{ width: 90, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                  borderRadius: 4, padding: '3px 8px', fontSize: 12 }} />
            </div>
            {edsElements.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 11, color: C.textSecondary, marginBottom: 4 }}>{t('phaseTest.edsOverlay')}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }} title={t('hoverTips.phaseTestEdsOverlay')}>
                  {edsElements.map((el, i) => {
                    const sym = edsSym(el);
                    const on = edsOverlaySel.includes(sym);
                    const color = EDS_COLORS[i % EDS_COLORS.length];
                    return (
                      <button key={sym} type="button" onClick={() => toggleEdsEl(sym)} aria-pressed={on}
                        style={{ fontSize: 11, padding: '2px 8px', borderRadius: 10, cursor: 'pointer',
                          border: `1px solid ${on ? color : C.border}`,
                          background: on ? color : C.bg, color: on ? '#000' : C.text }}>
                        {sym}
                      </button>
                    );
                  })}
                </div>
                {edsOverlaySel.length > 0 && (
                  <input type="range" min={0} max={1} step={0.05} value={edsOpacity}
                    onChange={(e) => setEdsOpacity(parseFloat(e.target.value))}
                    title={t('hoverTips.phaseTestEdsOpacity')} style={{ width: 100, marginTop: 6 }} />
                )}
              </div>
            )}
          </div>

          {/* Measured pattern preview for the active pixel */}
          <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: 10 }}>
            <div style={{ fontSize: 11, color: C.text, marginBottom: 6 }}>
              {t('phaseTest.measuredPattern', { row: previewRow, col: previewCol })}
            </div>
            {previewB64 ? (
              <img src={img(previewB64)} alt={t('phaseTest.measuredPattern', { row: previewRow, col: previewCol })}
                style={{ width: '100%', aspectRatio: '1', objectFit: 'contain', background: '#000',
                  border: `1px solid ${C.border}`, borderRadius: 4 }} />
            ) : (
              <div title={previewError || undefined} style={{ width: '100%', aspectRatio: '1', background: C.bgSecondary,
                border: `1px solid ${C.border}`, borderRadius: 4, display: 'flex',
                alignItems: 'center', justifyContent: 'center', color: previewError ? C.orange : C.textSecondary, fontSize: 10 }}>
                {previewLoading ? t('phaseTest.loadingShort')
                  : previewError ? t('phaseTest.previewFailed')
                  : (fileLoaded ? t('phaseTest.noPattern') : t('phaseTest.loadAFile'))}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT column: all existing controls + results */}
        <div style={{ flex: 1, minWidth: 0 }}>

        {/* EDS + mask bar (pixel-index input lives in the left column) */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center',
          background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: '8px 12px', marginBottom: 12 }}>
          {(() => {
            // Prefer the LIVE chemistry of the active crosshair pixel; fall
            // back to the last run's pixel only until the live fetch lands.
            const edsPct = livePixelAtPct || result?.pixel_at_pct;
            if (!edsPct) return null;
            // Only the elements actually present (>0.5 at%) to keep it readable.
            const shown = Object.entries(edsPct).filter(([, v]) => v >= 0.5);
            if (shown.length === 0) return null;
            return (
              <span style={{ fontSize: 10, color: C.textSecondary }}
                title={t('phaseTest.edsAtTooltip', { row: previewRow, col: previewCol })}>
                {t('phaseTest.edsAt', { row: previewRow, col: previewCol, values: shown.map(([k, v]) => `${k} ${v.toFixed(0)}%`).join(' · ') })}
              </span>
            );
          })()}
          <span style={{ marginLeft: 'auto', fontSize: 10, color: C.textSecondary }}>{t('phaseTest.edsLabel')}</span>
          {['off', 'soft', 'filter'].map((m) => (
            <button key={m} onClick={() => setEdsMode(m)} title={t(`hoverTips.phaseTestEds${m.charAt(0).toUpperCase() + m.slice(1)}`)} style={{ fontSize: 10,
              background: edsMode === m ? C.green : 'transparent', color: edsMode === m ? C.bg : C.textSecondary,
              border: `1px solid ${edsMode === m ? C.green : C.border}`, borderRadius: 3, padding: '2px 8px', cursor: 'pointer' }}>{t(`phaseTest.eds${m.charAt(0).toUpperCase() + m.slice(1)}`)}</button>
          ))}
          {/* Quality = simulation bandwidth (SHT). Drives the actual NCC/R
              compute, not just the preview. */}
          <span style={{ fontSize: 10, color: C.textSecondary }}>{t('phaseTest.quality')}</span>
          <select value={bandwidth} onChange={(e) => setBandwidth(parseInt(e.target.value, 10))}
            title={t('phaseTest.qualityTip')}
            style={{ fontSize: 10, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
              borderRadius: 3, padding: '2px 6px', cursor: 'pointer' }}>
            <option value={88}>{t('phaseTest.qualityFast')}</option>
            <option value={128}>{t('phaseTest.qualityStandard')}</option>
          </select>
          <label style={{ fontSize: 10, color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}
            title={t('phaseTest.bgRemoveTip')}>
            <input type="checkbox" checked={bgRemove}
              onChange={(e) => setBgRemove(e.target.checked)} /> {t('phaseTest.bgRemove')}
          </label>
          <label style={{ fontSize: 10, color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}
            title={t('hoverTips.phaseTestMask')}>
            <input type="checkbox" checked={aperture !== 'full'}
              onChange={(e) => {
                const ap = e.target.checked ? 'circular' : 'full';
                setAperture(ap);
                // Mask-only change → recompute from cache (no full re-render).
                if (result && !running && !loading) remask(ap, undefined);
              }} /> {t('phaseTest.mask')}
          </label>
          <span style={{ fontSize: 10, color: C.textSecondary }}>{t('phaseTest.maskRadius')}</span>
          <input type="range" min={0.3} max={1.0} step={0.05} value={apertureRadius}
            disabled={aperture === 'full'}
            title={t('hoverTips.phaseTestMaskRadius')}
            onChange={(e) => setApertureRadius(parseFloat(e.target.value))}
            onMouseUp={() => { if (result && !running && !loading) remask(undefined, apertureRadius); }}
            onTouchEnd={() => { if (result && !running && !loading) remask(undefined, apertureRadius); }}
            onKeyUp={() => { if (result && !running && !loading) remask(undefined, apertureRadius); }} />
          <span style={{ fontSize: 10, color: C.textSecondary, minWidth: 30 }}>
            {Math.round(apertureRadius * 100)}%
          </span>
        </div>

        {/* Phase picker — collapsible. Hidden if no library phases loaded
            (then the run just tests all, the backend default). */}
        {phases.length > 0 && (
          <div style={{ marginBottom: 10, background: C.bg, border: `1px solid ${C.border}`,
            borderRadius: 6 }}>
            <button onClick={() => setShowPhases((v) => !v)}
              title={t('hoverTips.phaseTestPhasesToggle')}
              style={{ background: 'transparent', border: 'none', color: C.text, cursor: 'pointer',
                fontSize: 11, padding: '8px 12px', width: '100%', textAlign: 'left' }}>
              {showPhases ? '▾' : '▸'} {t('phaseTest.phasesToggle', { selected: selectedKeys.size, total: phases.length })}
            </button>
            {showPhases && (
              <div style={{ borderTop: `1px solid ${C.border}` }}>
                <button onClick={toggleAll}
                  title={t('hoverTips.phaseTestSelectToggle')}
                  style={{ background: 'transparent', border: 'none', color: C.accent, cursor: 'pointer',
                    fontSize: 10, padding: '6px 12px', width: '100%', textAlign: 'left' }}>
                  {selectedKeys.size === phases.length ? t('phaseTest.selectNone') : t('phaseTest.selectAll')}
                </button>
                <div style={{ maxHeight: 180, overflowY: 'auto', padding: '0 12px 8px' }}>
                  {phases.map((p) => (
                    <label key={p.key} title={t('hoverTips.phaseTestPhaseCheckbox', { label: p.label })} style={{ display: 'flex', alignItems: 'center', gap: 6,
                      fontSize: 11, color: C.text, padding: '3px 0', cursor: 'pointer' }}>
                      <input type="checkbox" checked={selectedKeys.has(p.key)}
                        onChange={() => toggleKey(p.key)} />
                      <span>{p.label}</span>
                      {p.space_group && (
                        <span style={{ fontSize: 9, color: C.textSecondary }}>{p.space_group}</span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
          <button onClick={runAuto} disabled={!fileLoaded || running || loading}
            title={t('hoverTips.phaseTestRun')}
            style={{ background: C.accent, color: '#fff', border: 'none', borderRadius: 4,
              padding: '8px 16px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
              opacity: (!fileLoaded || running || loading) ? 0.5 : 1 }}>
            {(running || loading) ? t('phaseTest.testing')
              : allSelected ? t('phaseTest.autoRunAll') : t('phaseTest.runSelected', { count: selectedKeys.size })}
          </button>
          {running && (
            <button onClick={cancelRun}
              title={t('hoverTips.phaseTestCancel')}
              style={{ background: 'transparent', color: C.red, border: `1px solid ${C.red}`,
                borderRadius: 4, padding: '8px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer' }}>
              {t('phaseTest.cancel')}
            </button>
          )}
        </div>

        {/* Live progress bar while the background job runs. During the cold
            backend build (done===0) the per-phase count is meaningless, so show
            the backend's "Building phase models…" message + an indeterminate
            pulsing bar instead of a frozen "Testing 0/N". */}
        {running && ((progress?.done ?? 0) === 0 ? (
          <div style={{ marginBottom: 12 }}>
            <div style={{ height: 7, background: '#44475a', borderRadius: 3, overflow: 'hidden' }}>
              <div className="sppt-indeterminate" style={{
                width: '35%', height: '100%', background: C.accent }} />
            </div>
            <div style={{ fontSize: 11, color: C.text, marginTop: 4 }}>
              ⏳ {progress?.current || t('phaseTest.preparing')}
            </div>
            <div style={{ fontSize: 10, color: C.textSecondary, marginTop: 2 }}>
              {t('phaseTest.buildingHint')}
            </div>
            <style>{`@keyframes sppt-indet{0%{margin-left:-35%}100%{margin-left:100%}}
              .sppt-indeterminate{animation:sppt-indet 1.2s ease-in-out infinite}`}</style>
          </div>
        ) : (
          <div style={{ marginBottom: 12 }}>
            <div style={{ height: 7, background: '#44475a', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                width: `${progress?.total ? Math.round(((progress.done || 0) / progress.total) * 100) : 0}%`,
                height: '100%', background: C.accent, transition: 'width 0.2s' }} />
            </div>
            <div style={{ fontSize: 10, color: C.textSecondary, marginTop: 4 }}>
              {t('phaseTest.testingProgress', { done: progress?.done ?? 0, total: progress?.total ?? 0 })}
              {progress?.current ? ` — ${progress.current}` : ''}…
            </div>
          </div>
        ))}

        {error && <div style={{ color: C.red, fontSize: 11, marginBottom: 10 }}>{t('phaseTest.errorPrefix', { error })}</div>}

        {/* Comparison panel for the selected phase */}
        {selected && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', gap: 10 }}>
              {[[t('phaseTest.measured'), result?.experimental_png], [t('phaseTest.simulated', { formula: selected.display_formula || selected.formula }), selected.simulated_png],
                [t('phaseTest.nccDifference'), selected.ncc_diff_png]].map(([label, b64]) => (
                <div key={label} style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: C.text, marginBottom: 4 }}>{label}</div>
                  {b64 ? (
                    <LinkedPatternImage src={img(b64)} alt={label}
                      markers={markerCtl.markers} hover={markerCtl.hover}
                      onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick}
                      style={{ width: '100%', aspectRatio: '1', background: '#000',
                        border: `1px solid ${C.border}`, borderRadius: 4 }} />
                  ) : <div style={{ aspectRatio: '1', background: C.bg, borderRadius: 4 }} />}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 6, display: 'flex', gap: 10, alignItems: 'center',
              justifyContent: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: 19, fontWeight: 700, color: rColor(selected.r_score) }}>
                {t('phaseTest.rScore', { value: selected.r_score?.toFixed(3) ?? '—' })}
              </span>
              {selected.orientation_source === 'hough' && (
                <span
                  style={{ fontSize: 10, color: '#ffb86c', border: '1px solid #ffb86c44',
                    borderRadius: 3, padding: '2px 6px', cursor: 'help' }}
                  title={t('phaseTest.orientationHoughTip')}
                >
                  ⬡ {t('phaseTest.orientationFromHough')}
                </span>
              )}
              {selected.chemistry_fit != null && (
                <span style={{ fontSize: 11, color: C.accent }}>
                  {t('phaseTest.chemistry', { value: selected.chemistry_fit.toFixed(2) })}
                </span>
              )}
              {markerCtl.markers.length > 0 && (
                <button onClick={markerCtl.clearMarkers}
                  title={t('hoverTips.phaseTestClearMarkers')}
                  style={{ fontSize: 10, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                    borderRadius: 3, padding: '3px 8px', cursor: 'pointer' }}>
                  {t('phaseTest.clearMarkers', { count: markerCtl.markers.length })}
                </button>
              )}
              <button onClick={() => setExportOpen(true)} title={t('phaseTest.exportTip')}
                style={{ fontSize: 10, fontWeight: 600, background: C.green, color: C.bg, border: 'none',
                  borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}>
                {t('phaseTest.export')}
              </button>
            </div>

            {/* Use this phase for indexing — pick the method; disabled methods
                have no file for this phase (✗). Adds the phase and switches the
                indexing method, but keeps this dialog OPEN so the user can click
                the next pixel and add another phase (close it with ×). */}
            {onUsePhase && (
              <>
              <div style={{ marginTop: 6, display: 'flex', gap: 6, alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
                <span style={{ fontSize: 9, color: '#6272a4' }}>{t('phaseTest.useForIndexing')}:</span>
                {[
                  { m: 'hough', path: selected.cif_path, label: 'Hough' },
                  { m: 'spherical', path: selected.sht_path, label: 'Spherical' },
                  { m: 'dictionary', path: selected.master_h5_path, label: 'Dictionary' },
                ].map(({ m, path, label }) => (
                  <button key={m} disabled={!path}
                    onClick={() => {
                      const status = onUsePhase(selected, m);
                      // Only confirm if the click actually did something (path
                      // present ⇒ button enabled ⇒ non-null status).
                      if (status) markUsed(selected.display_formula || selected.formula, label, status);
                    }}
                    title={path ? t('phaseTest.useForIndexingTip', { method: label }) : t('phaseTest.methodUnavailable', { method: label })}
                    style={{ fontSize: 9, padding: '3px 8px', borderRadius: 3, cursor: path ? 'pointer' : 'not-allowed',
                      background: path ? (m === currentMethod ? C.green : C.bg) : '#2a2a3a',
                      color: path ? (m === currentMethod ? C.bg : C.text) : '#6272a4',
                      border: `1px solid ${path ? (m === currentMethod ? C.green : C.border) : '#3a3a4a'}` }}>
                    {path ? '✓' : '✗'} {label}
                  </button>
                ))}
              </div>
              {justUsed && (
                <div role="status" aria-live="polite" style={{ marginTop: 5, textAlign: 'center', fontSize: 10,
                  color: justUsed.status === 'exists' ? C.orange : C.green }}>
                  {justUsed.status === 'exists'
                    ? t('phaseTest.alreadyInList', { formula: justUsed.formula, method: justUsed.method })
                    : t('phaseTest.addedToList', { formula: justUsed.formula, method: justUsed.method })}
                </div>
              )}
              </>
            )}
          </div>
        )}

        {/* Ranked list — own scroll area so clicking through the phases never
            pushes the measured/simulated/NCC patterns above out of view. */}
        <div style={{ maxHeight: 340, overflowY: 'auto', paddingRight: 4 }}>
        {cands.map((c) => (
          <div key={c.phase_key} onClick={() => setSelected(c)}
            style={{ display: 'grid', gridTemplateColumns: '130px 1fr 1fr', gap: 10, alignItems: 'center',
              background: C.bgSecondary, cursor: 'pointer', padding: '7px 10px', marginBottom: 5,
              border: `1px solid ${selected?.phase_key === c.phase_key ? C.green : C.border}`, borderRadius: 5 }}>
            <span style={{ fontSize: 11, color: c.rank === 1 ? C.green : C.text, fontWeight: c.rank === 1 ? 700 : 400 }}>
              {c.rank === 1 ? '🏆 ' : ''}<span>{c.display_formula || c.formula}</span>
            </span>
            <Bar value={c.r_score} label={c.r_score?.toFixed(3) ?? '—'} color={rColor(c.r_score)} />
            <Bar value={c.chemistry_fit} label={c.chemistry_fit?.toFixed(2) ?? '—'}
              color={c.chemistry_fit == null ? C.textSecondary
                : c.chemistry_fit >= 0.6 ? C.green : c.chemistry_fit >= 0.35 ? C.orange : C.red} />
          </div>
        ))}
        </div>

        {excluded.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <button onClick={() => setShowExcluded((v) => !v)} title={t('hoverTips.phaseTestExcludedToggle')} style={{ background: 'transparent',
              border: `1px dashed ${C.border}`, color: C.textSecondary, borderRadius: 5, padding: '6px 10px',
              fontSize: 10, cursor: 'pointer', width: '100%', textAlign: 'left' }}>
              {t('phaseTest.excludedByEds', { count: excluded.length })}
            </button>
            {showExcluded && excluded.map((e) => (
              <div key={e.phase_key} style={{ fontSize: 10, color: C.textSecondary, padding: '4px 10px' }}>
                {t('phaseTest.excludedRow', { formula: e.display_formula || e.formula, value: e.chemistry_fit?.toFixed(2) ?? '—' })}
              </div>
            ))}
          </div>
        )}
        </div>{/* end RIGHT column */}
        </div>{/* end 2-column body */}
        <PatternExportDialog
          open={exportOpen} onClose={() => setExportOpen(false)}
          sources={{
            experimental: result?.experimental_png || null,
            simulated: selected?.simulated_png || null,
            ncc: selected?.ncc_diff_png || null,
            heatmap: null,
          }}
          rNcc={{ r: selected?.r_score, ncc: null }}
          stepUm={null}
          mapCols={null}
          markers={markerCtl.markers}
        />
      </div>
    </div>
  );
}

function Bar({ value, label, color }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value * 100));
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 7, background: '#44475a', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color }} />
      </div>
      <span style={{ fontSize: 10, color, minWidth: 32, textAlign: 'right' }}>{label}</span>
    </div>
  );
}

/**
 * Click-to-pick scan overview with the real Band Contrast image as
 * background + an SVG crosshair at the active pixel. Mirrors the
 * CrystalHint SinglePixelMode picker. Falls back to an empty box (the
 * crosshair + click logic still works) until the overview loads.
 */
function ScanPicker({ gridShape, pixelIndex, overviewB64, onPick, t, edsOverlays = [], edsOpacity = 0.6 }) {
  const [nRows, nCols] = gridShape;
  const maxW = 280;
  const maxH = 240;
  const aspect = nCols / Math.max(nRows, 1);
  const w = aspect >= maxW / maxH ? maxW : Math.round(maxH * aspect);
  const h = aspect >= maxW / maxH ? Math.round(maxW / aspect) : maxH;
  const cy = Math.floor(pixelIndex / Math.max(nCols, 1));
  const cx = pixelIndex % Math.max(nCols, 1);
  const handleClick = (ev) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const c = Math.max(0, Math.min(nCols - 1, Math.floor((x / rect.width) * nCols)));
    const r = Math.max(0, Math.min(nRows - 1, Math.floor((y / rect.height) * nRows)));
    onPick(r, c);
  };
  const px = ((cx + 0.5) / nCols) * w;
  const py = ((cy + 0.5) / nRows) * h;
  return (
    <div
      onClick={handleClick}
      style={{
        position: 'relative', width: w, height: h,
        background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 4, cursor: 'crosshair', overflow: 'hidden',
      }}
    >
      {overviewB64 ? (
        <img
          src={`data:image/png;base64,${overviewB64}`}
          alt={t('phaseTest.scanOverviewAlt')}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'fill', imageRendering: 'pixelated', pointerEvents: 'none',
          }}
        />
      ) : (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          color: C.textSecondary, fontSize: 10, opacity: 0.5,
        }}>
          {t('phaseTest.loadingOverview')}
        </div>
      )}
      {edsOverlays.map((b64, i) => (
        <img
          key={i}
          src={`data:image/png;base64,${b64}`}
          alt={t('phaseTest.edsOverlayAlt')}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'fill', imageRendering: 'pixelated', pointerEvents: 'none',
            opacity: edsOpacity, mixBlendMode: 'screen',
          }}
        />
      ))}
      <svg width={w} height={h} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        <line x1={px} y1={0} x2={px} y2={h} stroke={C.accent} strokeWidth="1" opacity="0.7" />
        <line x1={0} y1={py} x2={w} y2={py} stroke={C.accent} strokeWidth="1" opacity="0.7" />
        <circle cx={px} cy={py} r={5} fill={C.accent} stroke="#fff" strokeWidth="1.5" />
        <text x={4} y={h - 6} fill="#fff" fontSize="10" style={{
          paintOrder: 'stroke', stroke: '#000', strokeWidth: 2,
        }}>
          ({cy}, {cx})
        </text>
      </svg>
    </div>
  );
}
