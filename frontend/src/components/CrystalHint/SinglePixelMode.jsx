/**
 * Single Pixel Deep Analysis (Mode 1).
 *
 * User picks a pixel index (or uses the active EBSD index from the store),
 * we POST to /api/crystal-hint/analyze-pixel and render symmetry + lattice
 * + ranked candidate phases.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import useDataStore from '../../stores/useDataStore';
import { crystalHintApi, ebsdApi } from '../../services/api';
import SymmetryReportPanel from './SymmetryReportPanel';
import LatticeReportPanel from './LatticeReportPanel';
import CandidateList from './CandidateList';
import PatternPreview from './PatternPreview';

const S = {
  layout: {
    display: 'grid',
    gridTemplateColumns: '320px 1fr',
    gap: 16,
    height: '100%',
  },
  leftPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  rightPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    overflow: 'auto',
  },
  card: {
    padding: 12,
    background: colors.bgSecondary,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
  },
  cardTitle: {
    fontSize: 12,
    fontWeight: 600,
    color: colors.accent,
    marginBottom: 8,
  },
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  label: { fontSize: 11, color: colors.textSecondary, minWidth: 80 },
  input: {
    background: colors.bg,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    padding: '4px 8px',
    fontSize: 12,
    flex: 1,
  },
  button: {
    background: colors.accent,
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    padding: '6px 14px',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    width: '100%',
    marginTop: 8,
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  hint: {
    fontSize: 10,
    color: colors.textSecondary,
    marginTop: 4,
  },
  error: {
    background: `${colors.red}22`,
    color: colors.red,
    border: `1px solid ${colors.red}88`,
    borderRadius: 4,
    padding: '8px 10px',
    fontSize: 11,
    marginTop: 8,
  },
};

export default function SinglePixelMode({ elements, presetKey, presetEntry, isActive, onNavigate }) {
  const { t } = useTranslation('crystalhint');
  const currentIndex = useDataStore(s => s.currentIndex) || 0;
  const gridShape = useDataStore(s => s.gridShape);
  const setPosition = useDataStore(s => s.setPosition);
  // Either path (EBSD Viewer's `ebsd/load` or HDF5 Viewer's `h5/open`)
  // sets up a usable h5_session on the backend — both gate Crystal Hint.
  const fileLoaded = useDataStore(s => s.ebsdLoaded || s.isFileOpen);
  const [pixelIndex, setPixelIndex] = useState(currentIndex);
  const [strictChemistry, setStrictChemistry] = useState(true);
  // Neighborhood averaging: 0=single pixel, 1=3x3 average (9 patterns).
  // 3x3 averaging recovers ~2x NCC on noisy raw patterns by suppressing
  // per-pixel noise while preserving band structure.
  const [avgRadius, setAvgRadius] = useState(1);
  // EDS chemistry weighting: 'off' (pattern-only) | 'soft' (damp by
  // chemistry match) | 'filter' (drop chemically inconsistent phases).
  // Default 'soft'; degrades to off server-side when the dataset has no EDS.
  const [edsWeighting, setEdsWeighting] = useState('soft');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [externalMatches, setExternalMatches] = useState([]);
  const [externalLoading, setExternalLoading] = useState(false);
  // Monotonic request counter to discard stale external-search responses.
  // Without this, rapid Analyse clicks on different pixels can land an
  // older pixel's external matches into the new pixel's state.
  const reqIdRef = useRef(0);
  // AbortController for the in-flight external search — cancels the HTTP
  // request itself when the user starts a new analyze (otherwise the
  // backend keeps doing COD+MP work for 1-5s per stale click).
  const extAbortRef = useRef(null);
  // LRU cache for analyze-pixel results (pixelIndex + avgRadius → response).
  // Repeat clicks on the same pixel are instant. Cap at 20 entries.
  const analysisCacheRef = useRef(new Map());
  // Cache for external-DB matches keyed by (system, a_low, a_high, sorted
  // elements). On the same pixel re-clicked, we don't want to re-fire 1-5s
  // worth of COD+MP queries. The reviewer flagged this as wasteful — agreed.
  const externalCacheRef = useRef(new Map());
  // Real scan overview (Band Contrast) — fetched once when the tab
  // becomes active and a file is loaded. Cached client-side; backend
  // caches by (dataset, mode) too. Uses the 'bc' mode so the picker shows
  // the same crisp band-contrast map as the Indexing Navigation Map (native
  // Oxford BC, computed FFT image-quality fallback) rather than a washed-out
  // per-pixel mean-intensity map.
  const [overviewB64, setOverviewB64] = useState(null);

  useEffect(() => {
    if (!fileLoaded || !isActive) return;
    let cancelled = false;
    ebsdApi.overview('bc')
      .then((r) => {
        if (cancelled) return;
        setOverviewB64(r?.data?.image || null);
      })
      .catch(() => { /* silently fall back to empty SVG */ });
    return () => { cancelled = true; };
  }, [fileLoaded, isActive]);

  // Keep the input in sync with the global currentIndex when it changes (e.g.
  // user clicks a pixel in the EBSD viewer / phase map).
  useEffect(() => {
    if (typeof currentIndex === 'number' && currentIndex >= 0) {
      setPixelIndex(currentIndex);
    }
  }, [currentIndex]);

  const handleAnalyze = useCallback(async () => {
    setError(null);
    setExternalMatches([]);
    // Reserve a fresh request id; only THIS handler's external response may
    // write to state. Anything older is stale and discarded.
    const myReqId = ++reqIdRef.current;

    // Cancel any in-flight external-search from a previous click.
    if (extAbortRef.current) {
      try { extAbortRef.current.abort(); } catch { /* noop */ }
      extAbortRef.current = null;
    }

    // Cache hit: identical (pixelIndex, avgRadius, strictChemistry,
    // presetKey, elements signature). Skips the network round-trip + the
    // backend symmetry+lattice work for repeat clicks. We still re-fire
    // the external DB search because it depends on the lattice range that
    // might have shifted (and the user may want to refresh after CIF dl).
    // Sort elements before stringify so [Al,Fe] and [Fe,Al] hit the same
    // cache entry (the backend treats the element list as a set).
    const elementsSorted = [...elements].sort();
    const cacheKey = JSON.stringify([
      pixelIndex, avgRadius, strictChemistry, presetKey || '', elementsSorted,
      edsWeighting,
    ]);
    const cached = analysisCacheRef.current.get(cacheKey);
    let resData;
    if (cached) {
      resData = cached;
      setAnalysisResult(resData);
    } else {
      setLoading(true);
      try {
        const res = await crystalHintApi.analyzePixel({
          pixelIndex,
          elements,
          presetKey: presetKey || null,
          strictChemistry,
          patternType: 'processed',
          avgRadius,
          edsWeighting,
        });
        if (myReqId !== reqIdRef.current) return;  // stale
        resData = res.data;
        setAnalysisResult(resData);
        // LRU insert: re-insert on hit, evict oldest at 20.
        analysisCacheRef.current.set(cacheKey, resData);
        if (analysisCacheRef.current.size > 20) {
          const firstKey = analysisCacheRef.current.keys().next().value;
          analysisCacheRef.current.delete(firstKey);
        }
      } catch (err) {
        if (myReqId !== reqIdRef.current) return;
        const msg = err?.response?.data?.detail || err?.message || 'unknown error';
        setError(msg);
        setAnalysisResult(null);
        setLoading(false);
        return;
      } finally {
        if (myReqId === reqIdRef.current) {
          setLoading(false);
        }
      }
    }

    // External search — also cached. The inputs are derived from resData
    // (which is itself cached), so a cache hit on the analyze AND identical
    // external inputs = zero network. Reviewer flagged "re-fires on cache
    // hit is wasteful" — fixed by adding this second cache layer.
    try {
      const lattice = resData?.lattice;
      const sym = resData?.symmetry;
      const aRange = lattice?.a_range_A;
      // Only constrain the EXTERNAL search by crystal system when the symmetry
      // detector is actually confident. On real EBSD most pixels give noise-
      // level rotation NCC (confidence "none"/"low"); passing the resulting
      // (often wrong) "cubic" hint then hides every non-cubic candidate — i.e.
      // exactly the monoclinic/orthorhombic/triclinic Fe-Al intermetallics
      // (Al13Fe4 C2/m, Al5Fe2, Al12Fe7 P-1, …) that this feature exists to
      // surface ("phases not in your SHT library"). When unconfident, search
      // all systems and let chemistry + lattice rank. Confident detections
      // keep the system constraint.
      const _symConfident = sym?.confidence === 'medium' || sym?.confidence === 'high';
      const sysHint = _symConfident
        ? ((sym?.compatible_systems || []).includes('cubic')
            ? 'cubic'
            : (sym?.compatible_systems || [])[0] || null)
        : null;
      // Pass the n_fold_scores + d_spacings into external-search too so
      // the backend can apply the same per-phase fit ranking to external
      // candidates. Without this, external matches stay sorted just by
      // lattice distance + E_above_hull, while local matches use the
      // smarter combined fit — confusing for the user.
      const nFoldScores = sym?.n_fold_scores || null;
      const dSpacings = lattice?.d_spacings_A || null;
      const extKey = JSON.stringify([
        sysHint || '',
        aRange ? aRange[0] : null,
        aRange ? aRange[1] : null,
        elementsSorted,
        nFoldScores,   // include in cache key so fit-aware results don't
        dSpacings,     // collide with fit-blind ones.
      ]);
      const extCached = externalCacheRef.current.get(extKey);
      if (extCached) {
        setExternalMatches(extCached);
        return;
      }
      setExternalLoading(true);
      const ctrl = new AbortController();
      extAbortRef.current = ctrl;
      crystalHintApi
        .externalSearch({
          elements,
          crystalSystem: sysHint,
          aLowA: aRange ? aRange[0] : null,
          aHighA: aRange ? aRange[1] : null,
          maxResults: 20,
          nFoldScores,
          dObservedA: dSpacings,
          signal: ctrl.signal,
        })
        .then(extRes => {
          if (myReqId !== reqIdRef.current) return;
          const matches = extRes.data || [];
          setExternalMatches(matches);
          // LRU(15) on external matches — slightly smaller than analysis
          // cache (each MP/COD entry is ~200 B and we may have 20 of them).
          externalCacheRef.current.set(extKey, matches);
          if (externalCacheRef.current.size > 15) {
            const firstKey = externalCacheRef.current.keys().next().value;
            externalCacheRef.current.delete(firstKey);
          }
        })
        .catch(err => {
          if (myReqId !== reqIdRef.current) return;
          if (err?.code === 'ERR_CANCELED' || err?.name === 'CanceledError') {
            return;  // expected on rapid re-click
          }
          console.warn('External DB search failed:', err);
          setExternalMatches([]);
        })
        .finally(() => {
          if (myReqId !== reqIdRef.current) return;
          setExternalLoading(false);
          // Clear the abort ref on success/failure so it doesn't dangle.
          if (extAbortRef.current === ctrl) {
            extAbortRef.current = null;
          }
        });
    } catch (err) {
      // Safety net for any synthesis error in the external-search setup
      // (e.g. accessing fields on an unexpected response shape). The
      // analysis result was already saved above; we just warn here.
      console.warn('External DB search setup failed:', err);
    }
  }, [pixelIndex, elements, presetKey, strictChemistry, avgRadius, edsWeighting]);

  const symmetry = analysisResult?.symmetry;
  const lattice = analysisResult?.lattice;
  const matches = analysisResult?.local_matches || [];

  const fileReadyHint = fileLoaded
    ? null
    : t('single.fileReadyHint');

  return (
    <div style={S.layout}>
      {/* Left: input controls */}
      <div style={S.leftPanel}>
        <div style={S.card}>
          <div style={S.cardTitle}>{t('single.pixelSelection')}</div>
          <div style={S.controlRow}>
            <span style={S.label}>{t('single.indexLabel')}</span>
            <input
              type="number"
              style={S.input}
              value={pixelIndex}
              min={0}
              max={gridShape[0] * gridShape[1] - 1}
              title={t('single.pixelIndexTooltip')}
              onChange={e => setPixelIndex(parseInt(e.target.value, 10) || 0)}
              onKeyDown={e => {
                if (e.key === 'Enter' && fileLoaded && !loading) {
                  e.preventDefault();
                  handleAnalyze();
                }
              }}
            />
          </div>
          {gridShape[0] > 0 && gridShape[1] > 0 && (
            <>
              <div style={{ fontSize: 10, color: colors.textSecondary, marginBottom: 4 }}>
                {t('single.clickToPick', { rows: gridShape[0], cols: gridShape[1] })}
              </div>
              <ScanPicker
                gridShape={gridShape}
                pixelIndex={pixelIndex}
                overviewB64={overviewB64}
                scanPickerTitle={t('single.scanPickerTooltip')}
                overviewAlt={t('single.overviewAlt')}
                loadingOverviewText={t('single.loadingOverview')}
                onPick={(r, c) => {
                  const idx = r * gridShape[1] + c;
                  setPixelIndex(idx);
                  // Also sync the global store so EBSD Viewer / Phase Map
                  // jump to the same pixel when the user switches tabs.
                  setPosition(r, c, idx, null);
                }}
              />
            </>
          )}
          <div style={S.hint}>
            {t('single.syncHint')}
          </div>
        </div>

        <div style={S.card}>
          <div style={S.cardTitle}>{t('single.options')}</div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
            <input
              type="checkbox"
              checked={strictChemistry}
              onChange={e => setStrictChemistry(e.target.checked)}
              title={t('single.strictChemistryTooltip')}
            />
            <span>{t('single.strictChemistry')}</span>
          </label>
          <div style={S.hint}>
            {t('single.strictChemistryHint')}
          </div>
          <div style={{ ...S.controlRow, marginTop: 10 }}>
            <span style={S.label}>{t('single.avgNeighborhood')}</span>
            <select
              style={S.input}
              value={avgRadius}
              onChange={e => setAvgRadius(parseInt(e.target.value, 10))}
              title={t('single.avgNeighborhoodTooltip')}
            >
              <option value={0}>{t('single.avgSingle')}</option>
              <option value={1}>{t('single.avg3x3')}</option>
              <option value={2}>{t('single.avg5x5')}</option>
            </select>
          </div>
          <div style={S.hint}>
            {t('single.avgHint')}
          </div>
          <div style={{ ...S.controlRow, marginTop: 10 }}>
            <span style={S.label}>{t('single.edsWeighting')}</span>
            <select
              style={S.input}
              value={edsWeighting}
              onChange={e => setEdsWeighting(e.target.value)}
              title={t('single.edsWeightingTooltip')}
            >
              <option value="off">{t('single.edsOff')}</option>
              <option value="soft">{t('single.edsSoft')}</option>
              <option value="filter">{t('single.edsFilter')}</option>
            </select>
          </div>
          <div style={S.hint}>
            {t('single.edsHint')}
          </div>
        </div>

        <button
          onClick={handleAnalyze}
          style={{
            ...S.button,
            ...((!fileLoaded || loading) ? S.buttonDisabled : {}),
          }}
          disabled={!fileLoaded || loading}
          title={t('single.analyseTooltip')}
        >
          {loading ? t('single.analysing') : t('single.analyse')}
        </button>

        {fileReadyHint && (
          <div style={S.hint}>{fileReadyHint}</div>
        )}
        {error && (
          <div style={S.error}>{t('single.error', { msg: error })}</div>
        )}
      </div>

      {/* Right: result panels */}
      <div style={S.rightPanel}>
        {!analysisResult && !loading && (
          <div style={{ ...S.card, color: colors.textSecondary, textAlign: 'center', padding: 32 }}>
            {t('single.placeholderPre')}<b>{t('single.placeholderBold')}</b>{t('single.placeholderPost')}
          </div>
        )}
        {analysisResult && (
          <>
            <PatternPreview
              patternB64={analysisResult.pattern_b64_png}
              patternShape={analysisResult.pattern_shape}
              symmetry={symmetry}
            />
            <SymmetryReportPanel symmetry={symmetry} indexedPhase={analysisResult.indexed_phase} />
            <LatticeReportPanel lattice={lattice} />
            <CandidateList
              matches={matches}
              presetEntry={presetEntry}
              warnings={analysisResult.warnings}
              externalMatches={externalMatches}
              externalLoading={externalLoading}
              onNavigate={onNavigate}
            />
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Click-to-pick scan overview with the real scan image as background.
 *
 * Falls back to an empty box if the overview isn't loaded yet — the
 * crosshair + click logic still works in that case.
 */
function ScanPicker({ gridShape, pixelIndex, overviewB64, onPick, scanPickerTitle, overviewAlt, loadingOverviewText }) {
  const [nRows, nCols] = gridShape;
  // Bigger than before — easier to pick a small region on a 174x145 scan.
  const maxW = 320;
  const maxH = 280;
  const aspect = nCols / Math.max(nRows, 1);
  const w = aspect >= maxW / maxH ? maxW : Math.round(maxH * aspect);
  const h = aspect >= maxW / maxH ? Math.round(maxW / aspect) : maxH;
  const cy = Math.floor(pixelIndex / nCols);
  const cx = pixelIndex % nCols;
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
      title={scanPickerTitle}
      style={{
        position: 'relative',
        width: w,
        height: h,
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        cursor: 'crosshair',
        marginBottom: 6,
        overflow: 'hidden',
      }}
    >
      {overviewB64 ? (
        <img
          src={`data:image/png;base64,${overviewB64}`}
          alt={overviewAlt}
          style={{
            position: 'absolute', inset: 0,
            width: '100%', height: '100%',
            objectFit: 'fill',
            imageRendering: 'pixelated',
            pointerEvents: 'none',
          }}
        />
      ) : (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: colors.textSecondary, fontSize: 10, opacity: 0.5,
        }}>
          {loadingOverviewText}
        </div>
      )}
      <svg
        width={w} height={h}
        style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      >
        <line x1={px} y1={0} x2={px} y2={h} stroke={colors.accent} strokeWidth="1" opacity="0.7" />
        <line x1={0} y1={py} x2={w} y2={py} stroke={colors.accent} strokeWidth="1" opacity="0.7" />
        <circle cx={px} cy={py} r={5} fill={colors.accent} stroke="#fff" strokeWidth="1.5" />
        <text x={4} y={h - 6} fill="#fff" fontSize="10" style={{
          paintOrder: 'stroke', stroke: '#000', strokeWidth: 2,
        }}>
          ({cy}, {cx})
        </text>
      </svg>
    </div>
  );
}
