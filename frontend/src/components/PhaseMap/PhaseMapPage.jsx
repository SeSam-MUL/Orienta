/**
 * Phase Map Viewer — React port of gui/phase_map_gui.py (PhaseMapPage).
 *
 * Layout: ResizableSplitter
 *   Left:  Results Gallery list + canvas image area
 *   Right: Settings sidebar (260-320 px) — Data Source, Scalebar, Display,
 *          Confidence Overlay, Spatial Calibration, action buttons
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import api, { phaseMapApi, analysisApi, ebsdApi, indexApi, h5Api, pcApi } from '../../services/api';
import {
  colors, spacing,
  Button, Input, NumberInput, Select, GroupBox, CollapsibleGroup,
  ResizableSplitter, FormRow, Label, ProgressBar, Separator, ScrollPanel,
  LoadingOverlay, useConfirm, ConfirmDialog,
  usePrompt, PromptDialog,
} from '../../theme/components';
import useResultStore from '../../stores/useResultStore';
import useDataStore from '../../stores/useDataStore';
import { toast } from '../../stores/useToastStore';
import usePhaseColorStore from '../../stores/usePhaseColorStore';
import LayeredCanvas from './LayeredCanvas';
import LayerStackPanel from './LayerStackPanel';
import ComputeDiagnosticsPanel from './ComputeDiagnosticsPanel';
import RefinementPanel from './RefinementPanel';
import AnomalyBrowserDrawer from './AnomalyBrowser/AnomalyBrowserDrawer';
import FileSwitcher from '../common/FileSwitcher';
import CoordinateSystemPanel from '../common/CoordinateSystemPanel';
import LinkedPatternImage from '../PatternMatch/LinkedPatternImage';
import { useLinkedPatternMarkers } from '../PatternMatch/useLinkedPatternMarkers';
import PatternExportDialog from '../PatternMatch/PatternExportDialog';
import { openPoleFigureWindow } from '../PoleFigure/openPoleFigureWindow';
import useFrameStore from '../../stores/useFrameStore';
import { useLayerStack } from './hooks/useLayerStack';
import { useSourceLink } from './hooks/useSourceLink';
import { LAYER_SOURCES } from './layerSources';

// EDS-shared building blocks (reused for Tier-2 tools)
import { CursorSyncProvider, useCursorSync, useCursorPublisher } from '../EDS/CursorSyncContext';
import { pointerToRowCol, rowColToContainerPx } from '../EDS/mapCoords';
import TileGrid from '../EDS/TileGrid';
import ThresholdHistogram from '../EDS/ThresholdHistogram';
import LinescanProfilePlot from '../EDS/LinescanProfilePlot';
import MagnifierLens from '../EDS/MagnifierLens';
import { exportComposite } from '../EDS/compositeExporter';
import { useRectangleDrag } from '../EDS/hooks/useRectangleDrag';

// PhaseMap-specific tooling (landed in earlier commits this branch)
import PhaseMapProbeOverlay from './PhaseMapProbeOverlay';
import RegionStatsPanel from './RegionStatsPanel';
import ToolToolbar from './ToolToolbar';
import PhaseLegend from './PhaseLegend';
import PhaseAdjacencyPanel from './PhaseAdjacencyPanel';
import useAnnotations from './annotations/useAnnotations';
import AnnotationLayer from './annotations/AnnotationLayer';
import AnnotationToolbar from './annotations/AnnotationToolbar';
import { usePhaseMapProbe } from './hooks/usePhaseMapProbe';
import { usePhaseMapRegionStats } from './hooks/usePhaseMapRegionStats';
import { usePhaseMapLinescan } from './hooks/usePhaseMapLinescan';

// ---------------------------------------------------------------------------
// Pattern Matches Dialog (inline — used by "View Pattern Matches" button)
// ---------------------------------------------------------------------------
// Quality preset for SHT-forward simulated patterns. Bandwidth controls
// sharpness vs. VRAM: 128 (fast, ~225 MB), 256 (sharp, ~1.7 GB),
// 384 (sharpest, ~5.7 GB). Only used for Spherical Indexing results.
const SHT_QUALITY_OPTIONS = [
  { value: 128, labelKey: 'matches.sht.fast',     descKey: 'matches.sht.fastDesc'     },
  { value: 256, labelKey: 'matches.sht.standard', descKey: 'matches.sht.standardDesc' },
  { value: 384, labelKey: 'matches.sht.high',     descKey: 'matches.sht.highDesc'     },
];


function PatternMatchesDialog({ open, onClose, initialPixel = null }) {
  const { t } = useTranslation('phasemap');
  const [heatmapClean, setHeatmapClean] = useState(null);
  const [gridDims, setGridDims] = useState({ rows: 0, cols: 0 });
  // Shared crosshair + numbered red markers across the 3 comparison panels + the
  // publication-figure export composer — the SAME shared components as the
  // Indexing Pattern-Match view, so composer changes apply to both.
  const markerCtl = useLinkedPatternMarkers();
  const [exportOpen, setExportOpen] = useState(false);
  const [cropOffset, setCropOffset] = useState({ row: 0, col: 0 });
  const [matchData, setMatchData] = useState(null);
  const [selectedPixel, setSelectedPixel] = useState(null);
  const [rank, setRank] = useState(0);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [shtQuality, setShtQuality] = useState(128);  // SHT bandwidth preset
  const [aperture, setAperture] = useState('auto');   // auto | circular | full
  const [apertureRadius, setApertureRadius] = useState(1.0);
  // Misindex-diagnose (2026-05-26): when ON, the backend additionally
  // returns phase_results — per-phase R-score + simulated PNG for the
  // clicked pixel. The user can step through phases to see which one
  // ACTUALLY matches best (vs the indexer's stored winner, which can be
  // a chemically-degenerate near-tie). Adds ~0.5 s/click on a 12-phase
  // result after the per-result backend cache is warm. First click on
  // a new result pays a ~6 s × N_phases warmup cost.
  const [comparePhases, setComparePhases] = useState(false);
  const [selectedPhaseIdx, setSelectedPhaseIdx] = useState(0);
  const heatmapRef = useRef(null);

  // First-time-open effect: fetch the heatmap. Runs ONLY when ``open``
  // flips false -> true. ``initialPixel`` is consumed within the same
  // run via a ref so an external click that re-sets initialPixel while
  // the dialog is already open doesn't blow away the user's in-dialog
  // state (the previous deps ``[open, initialPixel]`` reset
  // selectedPixel + matchData every time the parent fired
  // setMatchesInitialPixel({...}), which left the user stuck on
  // "Loading…" with no inspect data after a re-click).
  const initialPixelRef = useRef(initialPixel);
  useEffect(() => { initialPixelRef.current = initialPixel; }, [initialPixel]);
  useEffect(() => {
    if (!open) return;
    setLoading(true); setSelectedPixel(null); setMatchData(null); setRank(0);
    let cancelled = false;
    indexApi.nccHeatmap().then(r => {
      if (cancelled) return;
      setHeatmapClean(r.data.heatmap_clean);
      setGridDims({ rows: r.data.n_rows, cols: r.data.n_cols });
      setCropOffset({ row: r.data.crop_row_offset ?? 0, col: r.data.crop_col_offset ?? 0 });
      setStats({ min: r.data.min_score, max: r.data.max_score, mean: r.data.mean_score });
      setLoading(false);
      // Pre-select an external pixel (e.g. from AnomalyBrowserDrawer or
      // PhaseMap canvas click). Coordinates are in original (uncropped)
      // map space; convert to local grid coords so the crosshair
      // overlay maps correctly. Read from the ref so this picks up the
      // LATEST initialPixel set by the parent at open time.
      const ip = initialPixelRef.current;
      if (ip && Number.isFinite(ip.row) && Number.isFinite(ip.col)) {
        const offRow = r.data.crop_row_offset ?? 0;
        const offCol = r.data.crop_col_offset ?? 0;
        setSelectedPixel({
          row: ip.row,
          col: ip.col,
          localRow: ip.row - offRow,
          localCol: ip.col - offCol,
        });
      }
    }).catch(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [open]);

  // If the dialog is OPEN and the parent re-sets initialPixel (e.g. user
  // clicks a different pixel on the Phase Map canvas while the dialog
  // is up), update the in-dialog selection without re-fetching the
  // heatmap. Skips when initialPixel is null (closing or fresh open).
  useEffect(() => {
    if (!open || !initialPixel) return;
    if (!Number.isFinite(initialPixel.row) || !Number.isFinite(initialPixel.col)) return;
    setSelectedPixel((prev) => {
      // No-op if the same pixel is already selected.
      if (prev && prev.row === initialPixel.row && prev.col === initialPixel.col) return prev;
      return {
        row: initialPixel.row,
        col: initialPixel.col,
        localRow: initialPixel.row - (cropOffset.row || 0),
        localCol: initialPixel.col - (cropOffset.col || 0),
      };
    });
  }, [open, initialPixel, cropOffset.row, cropOffset.col]);

  useEffect(() => {
    if (!selectedPixel) return;
    // Race guard: rapid changes to shtQuality (256 -> 128 -> 256) fire
    // overlapping requests. Without this guard the LAST request to RESOLVE
    // wins (not the last one fired), so the dialog could show R-score from
    // a stale bandwidth while the dropdown displays the current one. The
    // `cancelled` flag drops responses whose deps have since changed.
    let cancelled = false;
    indexApi.patternMatch(
      selectedPixel.row, selectedPixel.col, rank,
      shtQuality, aperture, apertureRadius,
      comparePhases,
    )
      .then(r => {
        if (cancelled) return;
        setMatchData(r.data);
        // Reset phase selector when a fresh response lands so the user
        // sees rank-1 (best R) by default.
        setSelectedPhaseIdx(0);
      })
      .catch(() => { if (!cancelled) setMatchData(null); });
    return () => { cancelled = true; };
  }, [selectedPixel, rank, shtQuality, aperture, apertureRadius, comparePhases]);

  const handleHeatmapClick = (e) => {
    const img = heatmapRef.current;
    if (!img || !gridDims.rows) return;
    const rect = img.getBoundingClientRect();
    const localRow = Math.min(Math.max(0, Math.floor((e.clientY - rect.top) / rect.height * gridDims.rows)), gridDims.rows - 1);
    const localCol = Math.min(Math.max(0, Math.floor((e.clientX - rect.left) / rect.width * gridDims.cols)), gridDims.cols - 1);
    setSelectedPixel({ row: localRow + cropOffset.row, col: localCol + cropOffset.col, localRow, localCol }); setRank(0);
  };

  if (!open) return null;
  const C = colors;

  const crossX = selectedPixel ? ((selectedPixel.localCol + 0.5) / gridDims.cols * 100) : -10;
  const crossY = selectedPixel ? ((selectedPixel.localRow + 0.5) / gridDims.rows * 100) : -10;

  // Misindex-diagnose (2026-05-26): when compare_phases is on AND the
  // backend returned per-phase results, the dialog displays the SELECTED
  // phase's simulated pattern + R-score instead of the indexer's stored
  // winner. The phase navigator below the simulated panel lets the user
  // step through phases sorted by R-score (best first).
  const phaseResults = (comparePhases ? matchData?.phase_results : null) || null;
  const selectedPhase = phaseResults?.[selectedPhaseIdx] ?? null;
  // Build the active "displayed match" view. In normal mode this is just
  // matchData. In compare mode it overlays the selected phase's sim/R.
  const displayed = phaseResults && selectedPhase
    ? {
        simulated:    selectedPhase.simulated_b64,
        r_score:      selectedPhase.r_score,
        phase_name:   selectedPhase.phase_name,
        euler_angles: selectedPhase.euler_deg,
        ncc_score:    selectedPhase.ncc_score,
        // r_quality derives from R-score (same buckets the backend uses):
        r_quality:    selectedPhase.r_score == null ? 'poor'
                       : selectedPhase.r_score >= 0.30 ? 'good'
                       : selectedPhase.r_score >= 0.15 ? 'acceptable'
                       : 'poor',
        // Pass-through fields the rest of the dialog still reads from:
        experimental: matchData?.experimental,
        ncc_image:    matchData?.ncc_image,           // computed from indexer winner — informative even in compare mode
        circular_aperture: matchData?.circular_aperture,
        indexing_method:   matchData?.indexing_method,
        simulated_error:   null,
      }
    : matchData;

  // R-score color (green >= 0.3, orange >= 0.15, red < 0.15)
  const rColor = displayed?.r_quality === 'good' ? '#50fa7b' : displayed?.r_quality === 'acceptable' ? '#ffb86c' : '#ff5555';
  const rLabel = displayed?.r_quality === 'good' ? t('phasemap:matches.match.good') : displayed?.r_quality === 'acceptable' ? t('phasemap:matches.match.acceptable') : t('phasemap:matches.match.poor');

  const patStyle = { height: 240, objectFit: 'contain', borderRadius: 3, border: `1px solid ${C.border}`, background: '#000' };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999 }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        background: colors.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 8, padding: 20, width: 1100, maxHeight: '90vh', overflow: 'auto',
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <span style={{ color: C.accent, fontWeight: 700, fontSize: 14 }}>
            {(() => {
              const m = matchData?.indexing_method || 'dictionary';
              const label = m === 'spherical' ? t('phasemap:matches.method.spherical')
                          : m === 'hough'     ? t('phasemap:matches.method.hough')
                          : t('phasemap:matches.method.dictionary');
              return `${label} — ${t('phasemap:matches.titleSuffix')}`;
            })()}
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {/* SHT bandwidth selector (only meaningful for spherical results) */}
            {matchData?.indexing_method === 'spherical' && (
              <label style={{ fontSize: '9pt', color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}>
                {t('phasemap:matches.quality')}
                <select
                  value={shtQuality}
                  onChange={(e) => setShtQuality(Number(e.target.value))}
                  title={(() => { const o = SHT_QUALITY_OPTIONS.find(o => o.value === shtQuality); return o ? t(`phasemap:${o.descKey}`) : ''; })()}
                  style={{
                    background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                    borderRadius: 3, padding: '2px 6px', fontSize: '9pt', cursor: 'pointer',
                  }}
                >
                  {SHT_QUALITY_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{t(`phasemap:${o.labelKey}`)}</option>
                  ))}
                </select>
              </label>
            )}
            {/* Aperture mask — EDAX phosphor patterns have a circular signal
                area with dark-grey (not black) corners that auto-detection
                misses. "Circular" forces the corner mask so the R-score
                isn't diluted by the signal-free corners. */}
            <label style={{ fontSize: '9pt', color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}>
              {t('phasemap:matches.aperture')}
              <select
                value={aperture}
                onChange={(e) => setAperture(e.target.value)}
                title={t('phasemap:matches.apertureTooltip')}
                style={{
                  background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                  borderRadius: 3, padding: '2px 6px', fontSize: '9pt', cursor: 'pointer',
                }}
              >
                <option value="auto">{t('phasemap:matches.apertureAuto')}</option>
                <option value="circular">{t('phasemap:matches.apertureCircular')}</option>
                <option value="full">{t('phasemap:matches.apertureFull')}</option>
              </select>
            </label>
            {aperture === 'circular' && (
              <label
                style={{ fontSize: '9pt', color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}
                title={t('phasemap:matches.radiusTooltip')}
              >
                R={apertureRadius.toFixed(2)}
                <input
                  type="range" min="0.3" max="1.0" step="0.02"
                  value={apertureRadius}
                  onChange={(e) => setApertureRadius(Number(e.target.value))}
                  style={{ width: 70, cursor: 'pointer' }}
                />
              </label>
            )}
            {/* Compare-phases toggle: when ON the backend ALSO re-indexes
                the clicked pixel against every phase's SHT and returns
                per-phase results in phase_results. Only meaningful for
                spherical results with > 1 phase. */}
            {matchData?.indexing_method === 'spherical' && (
              <label
                style={{ fontSize: '9pt', color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
                title={t('phasemap:matches.comparePhasesTooltip')}
              >
                <input
                  type="checkbox"
                  checked={comparePhases}
                  onChange={(e) => setComparePhases(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
                {t('phasemap:matches.comparePhases')}
              </label>
            )}
            <button onClick={onClose} aria-label={t('phasemap:matches.closeAria')} style={{ background: 'transparent', border: 'none', color: C.textSecondary, cursor: 'pointer', fontSize: 18 }}>&times;</button>
          </div>
        </div>
        <div style={{ fontSize: '9pt', color: C.textSecondary, marginBottom: 12 }}>{t('phasemap:matches.clickHint')}</div>

        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          {/* Left: NCC Heatmap (30%) — vertically centred against the
              three pattern panels on the right, which are much taller.
              Without alignItems:center the small heatmap stuck to the
              top while the right column dominated the row height. */}
          <div style={{ width: '30%', minWidth: 200 }}>
            {loading && <div style={{ color: C.textSecondary, padding: 40, textAlign: 'center' }}>{t('phasemap:matches.loading')}</div>}
            {heatmapClean && (
              <div style={{ position: 'relative', cursor: 'crosshair' }} onClick={handleHeatmapClick}>
                <img ref={heatmapRef} src={`data:image/png;base64,${heatmapClean}`} alt={t('phasemap:matches.nccAlt')} style={{ width: '100%', display: 'block', borderRadius: 3, border: `1px solid ${C.border}` }} />
                {selectedPixel && (<>
                  <div style={{ position: 'absolute', left: 0, right: 0, top: `${crossY}%`, height: 1, background: '#ffb86c', pointerEvents: 'none' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${crossX}%`, width: 1, background: '#ffb86c', pointerEvents: 'none' }} />
                  <div style={{ position: 'absolute', left: `${crossX}%`, top: `${crossY}%`, width: 12, height: 12, transform: 'translate(-50%,-50%)', pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ffb86c', fontSize: 16, fontWeight: 700 }}>+</div>
                </>)}
              </div>
            )}
            {stats && <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 4 }}>
              {t('phasemap:matches.nccRange', { min: stats.min?.toFixed(3), max: stats.max?.toFixed(3), mean: stats.mean?.toFixed(3) })}
            </div>}
            {selectedPixel && matchData && (
              <div style={{ fontSize: '9pt', color: '#f8f8f2', marginTop: 6, textAlign: 'center' }}>
                {t('phasemap:matches.pixelScore', { col: selectedPixel.col, row: selectedPixel.row, score: matchData.ncc_score?.toFixed(4) ?? '—' })}
              </div>
            )}
          </div>

          {/* Right: 3-panel comparison (70%) */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {!matchData && (
              <div style={{ color: '#6272a4', fontSize: '11pt', padding: 60, textAlign: 'center' }}>
                {selectedPixel
                  ? t('phasemap:matches.loadingMatch', {
                      col: selectedPixel.col, row: selectedPixel.row,
                      suffix: comparePhases ? t('phasemap:matches.comparePhasesWarmup') : '',
                    })
                  : t('phasemap:matches.clickToInspect')}
              </div>
            )}
            {matchData && (<>
              {/* 3 panels side by side */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <div style={{ flex: 1, textAlign: 'center' }}>
                  {/* Fixed-height header so all three images align vertically —
                      the middle (simulated) panel has an extra phase-name line. */}
                  <div style={{ height: 38, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', marginBottom: 4 }}>
                    <div style={{ fontSize: '9pt', color: '#f8f8f2' }}>{t('phasemap:matches.experimental')}</div>
                  </div>
                  {matchData.experimental ? <LinkedPatternImage src={`data:image/png;base64,${matchData.experimental}`} alt={t('phasemap:matches.expAlt')} markers={markerCtl.markers} hover={markerCtl.hover} onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick} style={{ ...patStyle, width: '100%' }} /> : <div style={{ height: 240, background: '#282a36', borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6272a4' }}>{t('phasemap:matches.notAvailable')}</div>}
                </div>
                <div style={{ flex: 1, textAlign: 'center' }}>
                  {/* Same fixed-height header as the other two panels (label +
                      the phase name) so the simulated image aligns with them. */}
                  <div style={{ height: 38, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', marginBottom: 4 }}>
                    <div style={{ fontSize: '9pt', color: '#f8f8f2' }}>
                      {phaseResults ? t('phasemap:matches.perPhaseSimulated') : t('phasemap:matches.bestMatchSimulated')}
                    </div>
                    <div style={{ fontSize: '10pt', fontWeight: 700, color: '#bd93f9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                         title={displayed?.phase_name || ''}>
                      {t('phasemap:matches.phaseLabel', { name: displayed?.phase_name || '—' })}
                      {phaseResults && selectedPhase && (
                        <span style={{ fontSize: '8pt', color: '#6272a4', marginLeft: 6 }}>
                          {t('phasemap:matches.phaseRank', { rank: selectedPhase.rank, total: phaseResults.length })}
                        </span>
                      )}
                    </div>
                  </div>
                  {displayed?.simulated ? (
                    <LinkedPatternImage src={`data:image/png;base64,${displayed.simulated}`} alt={t('phasemap:matches.simAlt')} markers={markerCtl.markers} hover={markerCtl.hover} onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick} style={{ ...patStyle, width: '100%' }} />
                  ) : (
                    <div style={{ height: 240, background: '#282a36', borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6272a4', padding: 12, textAlign: 'center', fontSize: '9pt' }}>
                      {matchData.simulated_error
                        ? matchData.simulated_error
                        : (matchData.indexing_method === 'spherical'
                            ? t('phasemap:matches.shtRendererUnavailable')
                            : t('phasemap:matches.dictionaryNotInMemory'))}
                    </div>
                  )}
                </div>
                <div style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ height: 38, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', marginBottom: 4 }}>
                    <div style={{ fontSize: '9pt', color: '#f8f8f2' }}>{t('phasemap:matches.nccImage')}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'stretch' }}>
                    {matchData.ncc_image ? <LinkedPatternImage src={`data:image/png;base64,${matchData.ncc_image}`} alt={t('phasemap:matches.nccAlt')} markers={markerCtl.markers} hover={markerCtl.hover} onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick} style={{ ...patStyle, flex: 1, minWidth: 0, background: '#1a1b26' }} /> : <div style={{ height: 240, flex: 1, background: '#282a36', borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6272a4' }}>{t('phasemap:matches.bothPatternsNeeded')}</div>}
                    {/* NCC Colorbar */}
                    {matchData.ncc_image && (
                      <div style={{ width: 18, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'space-between', fontSize: '6pt', color: '#6272a4' }}>
                        <span>+1</span>
                        <div style={{
                          flex: 1, width: 10, margin: '2px 0', borderRadius: 2,
                          background: 'linear-gradient(to bottom, #2166ac, #67a9cf, #f7f7f7, #ef8a62, #b2182b)',
                        }} />
                        <span>−1</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Marker tools + publication-figure export (shared composer —
                  changes here apply to the Indexing Pattern-Match export too). */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                {markerCtl.markers.length > 0 && (
                  <button onClick={markerCtl.clearMarkers} title={t('phasemap:hoverTips.matchesClearMarkers')}
                    style={{ fontSize: '8pt', background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, padding: '3px 8px', cursor: 'pointer' }}>
                    {t('phasemap:matches.clearMarkers', { count: markerCtl.markers.length })}
                  </button>
                )}
                <button onClick={() => setExportOpen(true)} title={t('phasemap:matches.exportTip')}
                  style={{ fontSize: '8pt', fontWeight: 600, background: '#50fa7b', color: '#282a36', border: 'none', borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}>
                  {t('phasemap:matches.export')}
                </button>
              </div>

              {/* Rank browser — hidden in compare-phases mode (the
                  phase navigator below takes over). */}
              {!phaseResults && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ fontSize: '9pt', color: '#f8f8f2' }}>{t('phasemap:matches.showMatch')}</span>
                  <button onClick={() => setRank(r => Math.max(0, r - 1))} disabled={rank <= 0} title={t('phasemap:hoverTips.matchPrevRank')} aria-label={t('phasemap:hoverTips.matchPrevRank')} style={{ padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}>-</button>
                  <span style={{ fontSize: '11pt', color: C.accent, fontWeight: 700, minWidth: 24, textAlign: 'center' }}>{rank + 1}</span>
                  <button onClick={() => setRank(r => Math.min((matchData.total_ranks || 1) - 1, r + 1))} disabled={rank >= (matchData.total_ranks || 1) - 1} title={t('phasemap:hoverTips.matchNextRank')} aria-label={t('phasemap:hoverTips.matchNextRank')} style={{ padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}>+</button>
                  <span style={{ fontSize: '9pt', color: '#6272a4' }}>{t('phasemap:matches.ofKept', { total: matchData.total_ranks || '?' })}</span>
                </div>
              )}

              {/* Compare-phases phase navigator: lets the user step through
                  per-phase re-indexed results sorted by R-score, best first.
                  Renders only when compare_phases is ON and the backend
                  delivered phase_results. */}
              {phaseResults && phaseResults.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '9pt', color: '#f8f8f2' }}>{t('phasemap:matches.comparePhaseLabel')}</span>
                  <button
                    onClick={() => setSelectedPhaseIdx(i => Math.max(0, i - 1))}
                    disabled={selectedPhaseIdx <= 0}
                    title={t('phasemap:hoverTips.comparePrevPhase')}
                    aria-label={t('phasemap:hoverTips.comparePrevPhase')}
                    style={{ padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}
                  >◀</button>
                  <select
                    value={selectedPhaseIdx}
                    onChange={(e) => setSelectedPhaseIdx(Number(e.target.value))}
                    title={t('phasemap:hoverTips.comparePhaseSelect')}
                    style={{
                      background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                      borderRadius: 3, padding: '2px 6px', fontSize: '9pt', cursor: 'pointer', minWidth: 220,
                    }}
                  >
                    {phaseResults.map((pr, i) => (
                      <option key={pr.phase_id} value={i}>
                        {pr.rank}. {pr.phase_name} — R={pr.r_score != null ? pr.r_score.toFixed(3) : '—'}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => setSelectedPhaseIdx(i => Math.min(phaseResults.length - 1, i + 1))}
                    disabled={selectedPhaseIdx >= phaseResults.length - 1}
                    title={t('phasemap:hoverTips.compareNextPhase')}
                    aria-label={t('phasemap:hoverTips.compareNextPhase')}
                    style={{ padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}
                  >▶</button>
                  <span style={{ fontSize: '8pt', color: '#6272a4' }}>{t('phasemap:matches.ofPhases', { count: phaseResults.length })}</span>
                </div>
              )}

              {/* R-score — large, color-coded. Reads from `displayed` so
                  it reflects the per-phase selection in compare mode. */}
              {displayed?.r_score != null && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '18pt', fontWeight: 700, color: rColor }}>
                    {t('phasemap:matches.rValue', { value: displayed.r_score.toFixed(4) })}
                  </div>
                  <div style={{ fontSize: '10pt', color: '#bd93f9' }}>{rLabel}</div>
                </div>
              )}

              {/* Multi-phase score comparison */}
              {matchData.phase_scores?.length > 1 && (
                <div style={{ marginTop: 8, padding: '6px 12px', background: C.bg, borderRadius: 4, border: `1px solid ${C.border}` }}>
                  <div style={{ fontSize: '8pt', color: '#6272a4', marginBottom: 4 }}>{t('phasemap:matches.phaseComparison')}</div>
                  {matchData.phase_scores.map((ps, i) => {
                    const barW = Math.max(2, Math.min(100, (Math.max(0, ps.r_score) / Math.max(0.01, matchData.phase_scores[0].r_score)) * 100));
                    const barColor = ps.r_score >= 0.3 ? '#50fa7b' : ps.r_score >= 0.15 ? '#ffb86c' : '#ff5555';
                    return (
                      <div key={ps.result_id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                        <span style={{ fontSize: '8pt', color: ps.is_active ? '#f8f8f2' : '#6272a4', minWidth: 80, textAlign: 'right', fontWeight: ps.is_active ? 700 : 400 }}>
                          {ps.phase || '—'}
                        </span>
                        <div style={{ flex: 1, height: 8, background: '#44475a', borderRadius: 2, overflow: 'hidden' }}>
                          <div style={{ width: `${barW}%`, height: '100%', background: barColor, borderRadius: 2, transition: 'width 0.3s' }} />
                        </div>
                        <span style={{ fontSize: '8pt', color: ps.is_active ? '#f8f8f2' : '#6272a4', minWidth: 50, fontWeight: ps.is_active ? 700 : 400 }}>
                          {ps.r_score.toFixed(3)}{i === 0 ? ' ★' : ''}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Euler (phase is shown prominently above the simulated panel) */}
              {displayed?.euler_angles && (
                <div style={{ fontSize: '8pt', color: '#6272a4', textAlign: 'center', marginTop: 4 }}>
                  {t('phasemap:matches.euler', { values: displayed.euler_angles.map(a => a?.toFixed(1)).join(', ') })}
                  {displayed.circular_aperture && t('phasemap:matches.cornersMasked')}
                </div>
              )}
            </>)}
          </div>
        </div>
      </div>
      <PatternExportDialog
        open={exportOpen} onClose={() => setExportOpen(false)}
        sources={{
          experimental: matchData?.experimental || null,
          simulated: displayed?.simulated || null,
          ncc: matchData?.ncc_image || null,
          heatmap: heatmapClean || null,
        }}
        rNcc={{ r: displayed?.r_score, ncc: matchData?.ncc_score }}
        stepUm={null}
        mapCols={gridDims.cols || null}
        markers={markerCtl.markers}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Constants matching phase_map_gui.py
// ---------------------------------------------------------------------------
const BASE_DISPLAY_MODES = [
  { value: 'phase',       label: 'Phase Map',   tip: 'Color each pixel by identified crystal phase', group: 'Phase' },
  { value: 'ipf_z',       label: 'IPF-Z [001]', tip: 'Inverse Pole Figure colored by sample normal direction', group: 'IPF' },
  { value: 'ipf_x',       label: 'IPF-X [100]', tip: 'Inverse Pole Figure colored by rolling direction', group: 'IPF' },
  { value: 'ipf_y',       label: 'IPF-Y [010]', tip: 'Inverse Pole Figure colored by transverse direction', group: 'IPF' },
  { value: 'ci',          label: 'CI Heatmap',   tip: 'Confidence Index of the best matching phase per pixel', group: 'Confidence' },
  { value: 'forward_ncc', label: 'Forward NCC',  tip: 'Forward-simulated pattern vs experimental NCC (spherical results only). Click Compute to build the map.', group: 'Confidence' },
];
// Additional CI modes are loaded dynamically from /api/phasemap/available-maps
const DISPLAY_MODES = BASE_DISPLAY_MODES;

const SCALEBAR_POSITIONS = [
  'upper left', 'upper center', 'upper right',
  'center left', 'center', 'center right',
  'lower left', 'lower center', 'lower right',
];

const CONF_CMAPS = ['RdYlGn', 'viridis', 'hot', 'coolwarm', 'gray'];

// Dracula palette — matches backend phase_map.py PHASE_COLORS
const DRACULA_PHASE_COLORS = [
  '#5ff77a', '#82aaff', '#f78c6c', '#c892ea',
  '#ffcb6b', '#89ddff', '#ff5370', '#c3e88d',
];

// Phase names orix / various loaders emit for the unindexed sentinel entry —
// not real crystal phases, should not appear in the phase-list legend.
const UNINDEXED_PHASE_NAMES = new Set([
  '', 'not_indexed', 'not indexed', 'notindexed',
  'unindexed', 'nothing', 'none', 'null',
]);
function isUnindexedPhase(name) {
  return name == null || UNINDEXED_PHASE_NAMES.has(String(name).trim().toLowerCase());
}

// JS-compatible string hash — must equal backend phase_map._name_hash so the
// gallery swatch colour equals the colour the backend map/legend use.
function nameHash(name) {
  let h = 0;
  const s = name || '';
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}
// Kept for any 8-slot consumers; the map/legend use a continuous hue now.
function stableColorIdx(name) {
  return nameHash(name) % DRACULA_PHASE_COLORS.length;
}
// Python colorsys.hsv_to_rgb ported to JS, → '#rrggbb' with round()-to-255 to
// match backend _to_hex, so the swatch equals the legend square exactly.
function hsvToHex(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r, g, b;
  switch (((i % 6) + 6) % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return '#' + [r, g, b]
    .map((c) => Math.round(c * 255).toString(16).padStart(2, '0'))
    .join('');
}
// NAME-stable hue — mirrors backend phase_map._hsv_phase_color_by_name so a
// phase keeps the same colour across every result, with distinct hues for
// distinct names (continuous 3600-step hue, no 8-slot palette collisions).
function hsvPhaseColorByName(name) {
  const hue = (nameHash(name) % 3600) / 3600;
  return hsvToHex(hue, 0.65, 0.95);
}
function colorForPhaseName(name, overrides = null) {
  // User-set override wins; otherwise the name-stable hue the backend map uses.
  if (overrides && overrides[name]) return overrides[name];
  return hsvPhaseColorByName(name || '');
}

// Direction used by API (Z / X / Y / phase / bc / ci / ci_<phase> / uncertainty) from display mode id
function directionFromMode(modeId) {
  if (modeId === 'phase') return 'phase';
  if (modeId === 'ipf_x') return 'X';
  if (modeId === 'ipf_y') return 'Y';
  if (modeId === 'ipf_z') return 'Z';
  if (modeId === 'bc') return 'bc';
  // CI modes and uncertainty pass through directly
  if (modeId.startsWith('ci') || modeId === 'uncertainty') return modeId;
  return 'Z';
}

// ---------------------------------------------------------------------------
// Gallery item row
// ---------------------------------------------------------------------------
function GalleryItem({ entry, selected, onClick }) {
  const { t } = useTranslation('phasemap');
  const [hovered, setHovered] = useState(false);
  return (
    <div
      role="option"
      aria-selected={selected}
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick?.(); } }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={t('phasemap:hoverTips.galleryEntry')}
      style={{
        padding: '4px 8px',
        fontSize: '9pt',
        cursor: 'pointer',
        background: selected
          ? colors.sidebarActive
          : hovered ? colors.bgTertiary : 'transparent',
        color: selected ? colors.accent : colors.text,
        borderLeft: selected ? `3px solid ${colors.accent}` : '3px solid transparent',
        userSelect: 'none',
        transition: 'background 0.1s, border-left-color 0.15s',
      }}
    >
      {entry.label || entry.name || t('phasemap:gallery.resultFallback', { id: entry.id })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tier-2 helper components
// ---------------------------------------------------------------------------
// CanvasInteractionLayer wraps the LayeredCanvas (or TileGrid in grid view)
// with pointer handlers that drive the CursorSyncContext, plus ROI / linescan
// overlays and the optional magnifier lens. All tool state is owned by the
// parent — this leaf does pure event-routing + visual overlays.
function CanvasInteractionLayer({
  view, layers, bitmaps, errors, shape, tileMinWidth,
  bitmapVersion, scalebar, title, stepX, ipfKeyImage, showIpfKey, hoverPixel,
  onPixelClick, onRegionSelected, onLineComplete,
  linescanMode, magnifierEnabled,
}) {
  const hostRef = useRef(null);
  const publish = useCursorPublisher();
  const [crosshair, setCrosshair] = useState(null);
  const [linePts, setLinePts] = useState(null);
  const lineStartRef = useRef(null);
  const [lensPos, setLensPos] = useState(null);

  useCursorSync((pos) => {
    if (!shape || !pos.hovering || !hostRef.current) { setCrosshair(null); return; }
    const rect = hostRef.current.getBoundingClientRect();
    const out = rowColToContainerPx(pos.row, pos.col, rect, shape);
    if (out) setCrosshair({ x: out.x, y: out.y });
  });

  const drag = useRectangleDrag({ shape, onRegion: onRegionSelected });

  const onMouseMove = (e) => {
    if (!shape || !hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, rect, shape);
    if (out) publish({ row: out.row, col: out.col, hovering: true, screenX: e.clientX, screenY: e.clientY });
    setLensPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    if (linescanMode && lineStartRef.current) {
      if (e.buttons === 0) {
        lineStartRef.current = null;
        setLinePts(null);
        return;
      }
      setLinePts((p) => p && ({ ...p, x1: e.clientX - rect.left, y1: e.clientY - rect.top }));
    }
    if (!linescanMode) drag.onPointerMove(e);
  };
  const onMouseLeave = (e) => {
    if (!shape) return;
    publish({ row: 0, col: 0, hovering: false, screenX: e.clientX, screenY: e.clientY });
    setLensPos(null);
  };
  const onMouseDown = (e) => {
    if (linescanMode && shape && hostRef.current) {
      const rect = hostRef.current.getBoundingClientRect();
      const out = pointerToRowCol(e, rect, shape);
      if (out) {
        lineStartRef.current = { row: out.row, col: out.col, x: e.clientX - rect.left, y: e.clientY - rect.top };
        setLinePts({
          x0: lineStartRef.current.x, y0: lineStartRef.current.y,
          x1: lineStartRef.current.x, y1: lineStartRef.current.y,
        });
      }
      return;
    }
    drag.onPointerDown(e);
  };
  const onMouseUp = (e) => {
    if (linescanMode && lineStartRef.current && hostRef.current && shape) {
      const rect = hostRef.current.getBoundingClientRect();
      const end = pointerToRowCol(e, rect, shape);
      if (end) onLineComplete?.({ start: { row: lineStartRef.current.row, col: lineStartRef.current.col }, end });
      lineStartRef.current = null;
      return;
    }
    drag.onPointerUp(e);
  };
  const onClickHandler = (e) => {
    if (linescanMode) return;  // never trigger click-to-quantify in linescan mode
    if (!shape || !hostRef.current || !onPixelClick) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, rect, shape);
    if (out) onPixelClick(out.row, out.col);
  };

  return (
    <div
      ref={hostRef}
      data-phasemap-canvas-host
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onClick={onClickHandler}
      style={{ position: 'relative', width: '100%', height: '100%', cursor: shape ? 'crosshair' : 'default' }}
    >
      {view === 'grid' ? (
        <TileGrid
          layers={layers}
          bitmaps={bitmaps}
          errors={errors}
          shape={shape}
          onPixelClick={onPixelClick}
          onRegionSelected={onRegionSelected}
          minTileWidth={tileMinWidth}
        />
      ) : (
        <LayeredCanvas
          layers={layers}
          bitmaps={bitmaps}
          bitmapVersion={bitmapVersion}
          loading={false}
          error={null}
          perLayerErrors={errors}
          scalebar={scalebar}
          title={title}
          stepX={stepX}
          ipfKeyImage={ipfKeyImage}
          showIpfKey={showIpfKey}
          hoverPixel={hoverPixel}
        />
      )}
      {view === 'stack' && crosshair && (
        <div data-phasemap-crosshair style={{
          position: 'absolute', left: crosshair.x, top: crosshair.y,
          width: 14, height: 14, transform: 'translate(-50%, -50%)',
          border: `1px solid ${colors.cyan}`, borderRadius: 2,
          boxShadow: '0 0 0 1px rgba(0,0,0,.45)',
          pointerEvents: 'none',
        }} />
      )}
      {view === 'stack' && drag.overlay && (
        <div data-roi-rect style={{
          position: 'absolute',
          left: Math.min(drag.overlay.x0, drag.overlay.x1),
          top:  Math.min(drag.overlay.y0, drag.overlay.y1),
          width:  Math.abs(drag.overlay.x1 - drag.overlay.x0),
          height: Math.abs(drag.overlay.y1 - drag.overlay.y0),
          border: `1px dashed ${colors.cyan}`,
          background: 'rgba(139,233,253,.08)',
          pointerEvents: 'none',
        }} />
      )}
      {view === 'stack' && linePts && linescanMode && (
        <svg data-linescan-line style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          <line x1={linePts.x0} y1={linePts.y0} x2={linePts.x1} y2={linePts.y1}
            stroke={colors.cyan} strokeWidth="2" strokeLinecap="round" strokeDasharray="4 3" />
          <circle cx={linePts.x0} cy={linePts.y0} r="3" fill={colors.cyan} />
          <circle cx={linePts.x1} cy={linePts.y1} r="3" fill={colors.cyan} />
        </svg>
      )}
      <MagnifierLens hostRef={hostRef} pos={lensPos} visible={magnifierEnabled} />
    </div>
  );
}

// Cursor-sync subscriber for the floating hover-probe tooltip. Lives inside
// the CursorSyncProvider subtree (mounted in the page-level return); the
// tooltip's screen-fixed position state stays local so 60Hz cursor moves
// don't thrash the rest of the page.
function HoverProbeLayer({ probe, error, requestProbe, clearProbe }) {
  const [pos, setPos] = useState(null);
  useCursorSync((p) => {
    if (!p.hovering) {
      setPos(null);
      clearProbe();
      return;
    }
    setPos({ x: p.screenX ?? 0, y: p.screenY ?? 0 });
    requestProbe(p.row, p.col);
  });
  return <PhaseMapProbeOverlay probe={probe} error={error} visible={pos !== null} pos={pos} />;
}

// ---------------------------------------------------------------------------
// SaveFormatPicker — modal that lets the user pick .ang / Light .h5 / Rich .h5
// before the native save dialog opens. Visually mirrors PromptDialog from
// theme/components.jsx so it feels like part of the dialog family. Esc and
// click-outside cancel.
// ---------------------------------------------------------------------------
function SaveFormatPicker({ entry, onPick, onCancel }) {
  const { t } = useTranslation('phasemap');
  useEffect(() => {
    if (!entry) return;
    const h = (e) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [entry, onCancel]);

  if (!entry) return null;

  const OptionCard = ({ onClick, title, badge, badgeColor, lines }) => (
    <button
      onClick={onClick}
      style={{
        display: 'block', width: '100%', textAlign: 'left',
        background: colors.bgSecondary, color: colors.text,
        border: `1px solid ${colors.border}`, borderRadius: 6,
        padding: '10px 12px', marginBottom: 8, cursor: 'pointer',
        fontFamily: 'inherit', fontSize: '10pt',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.borderColor = colors.accent; }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = colors.border; }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontWeight: 600 }}>{title}</span>
        {badge && (
          <span style={{
            fontSize: '8pt', padding: '1px 6px', borderRadius: 3,
            background: badgeColor, color: '#0a0a0a', fontWeight: 700,
          }}>{badge}</span>
        )}
      </div>
      {lines.map((line, i) => (
        <div key={i} style={{ fontSize: '9pt', color: colors.textSecondary, lineHeight: 1.4 }}>
          {line}
        </div>
      ))}
    </button>
  );

  return (
    <div
      onClick={onCancel}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 10000,
      }}
    >
      <div
        role="dialog"
        aria-label={t('phasemap:save.dialogAria')}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bg, border: `1px solid ${colors.border}`,
          borderRadius: 8, padding: '20px 24px', minWidth: 420, maxWidth: 520,
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ fontSize: '11pt', fontWeight: 600, color: colors.text, marginBottom: 4 }}>
          {t('phasemap:save.pickTitle')}
        </div>
        <div style={{ fontSize: '9pt', color: colors.textSecondary, marginBottom: 12 }}>
          {t('phasemap:save.pickSubtitle', { label: entry.label || t('phasemap:save.selectedResult') })}
        </div>

        <OptionCard
          onClick={() => onPick(entry, 'ang')}
          title={t('phasemap:save.ang')}
          lines={[
            t('phasemap:save.angLine1'),
            t('phasemap:save.angLine2'),
          ]}
        />
        <OptionCard
          onClick={() => onPick(entry, 'h5_light')}
          title={t('phasemap:save.h5Light')}
          badge={t('phasemap:save.recommended')}
          badgeColor="#7ed321"
          lines={[
            t('phasemap:save.h5LightLine1'),
            t('phasemap:save.h5LightLine2'),
            t('phasemap:save.h5LightLine3'),
          ]}
        />
        <OptionCard
          onClick={() => onPick(entry, 'h5')}
          title={t('phasemap:save.h5Rich')}
          badge={t('phasemap:save.large')}
          badgeColor="#f5a623"
          lines={[
            t('phasemap:save.h5RichLine1'),
            t('phasemap:save.h5RichLine2'),
            t('phasemap:save.h5RichLine3'),
          ]}
        />

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
          <Button small onClick={onCancel}>{t('common:cancel')}</Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function PhaseMapPage({ onNavigate, isActive = false }) {
  const { t } = useTranslation(['phasemap', 'common']);
  // Gallery state — subscribe to Zustand store for reactivity
  const indexingResult = useResultStore((s) => s.indexingResult);
  const indexingMethod = useResultStore((s) => s.indexingMethod);
  const resultsList = useResultStore((s) => s.resultsList);
  const { stepSize: storeStepSize } = useDataStore();
  const syncFromBackend = useDataStore((s) => s.syncFromBackend);

  // Coordinate-system frame — drives IPF re-render (frameSig threads into
  // useLayerStack) and the Pole-Figure popout / CS panel.
  const frameSig = useFrameStore((s) => s.frameSig);
  const loadFrameForFile = useFrameStore((s) => s.loadForFile);
  // Re-pull the per-file frame whenever the active result changes (its
  // source_file drives which frame applies). Null-safe inside the store —
  // the arg is only the localStorage key; frameApi.get() returns the
  // backend ACTIVE file's frame regardless.
  useEffect(() => {
    loadFrameForFile(indexingResult?.metadata?.source_file ?? null);
  }, [indexingResult?.result_id, loadFrameForFile]);

  // Map-canvas annotations (Legend, Scalebar, Title, North-Arrow)
  // — keyed off the active result id so each result keeps its own
  // layout. localStorage-backed; survives backend restarts (not result
  // metadata) so the user's publication-figure setup sticks.
  const annotState = useAnnotations(indexingResult?.result_id ?? null);
  const [selectedAnnotId, setSelectedAnnotId] = useState(null);
  const mapContainerRef = useRef(null);
  // Phase color overrides — a flat { phaseName: '#rrggbb' } map persisted in
  // localStorage. Passed as JSON to /render + /layer + /phase-stats and used
  // for the inline legend swatches so the map, legend, swatches AND export
  // all agree. Declared here (above the LegendAnnotation stats fetch) so that
  // fetch can key off it.
  const phaseColorOverrides = usePhaseColorStore((s) => s.overrides);
  const setPhaseColor = usePhaseColorStore((s) => s.setColor);
  const resetPhaseColor = usePhaseColorStore((s) => s.resetColor);
  // Phase stats for the LegendAnnotation body (lazy fetch, shared with
  // PhaseLegend panel). Reusing phaseStats keeps the on-canvas legend
  // in sync with the side-panel legend.
  const [phaseStatsForAnnot, setPhaseStatsForAnnot] = useState(null);
  useEffect(() => {
    if (!indexingResult?.result_id) {
      setPhaseStatsForAnnot(null);
      return;
    }
    let cancelled = false;
    import('../../services/api').then(({ phaseMapApi }) => {
      phaseMapApi.phaseStats(false, phaseColorOverrides)
        .then((r) => { if (!cancelled) setPhaseStatsForAnnot(r.data); })
        .catch(() => { if (!cancelled) setPhaseStatsForAnnot(null); });
    });
    return () => { cancelled = true; };
  }, [indexingResult?.result_id, phaseColorOverrides]);

  // Phase B: Original/Refined view toggle. The original result id is the
  // CrystalMap that came out of indexing; `refinementInfo` is populated
  // (per result) once joint R+PC refinement has been computed. When the
  // user picks "Refined", layer fetches should target the refined result
  // id. v1 wires `effectiveResultId` only into the RefinementPanel; the
  // layer-stack gating still reads `indexingResult.result_id` from the
  // store directly. Full wiring (LayerStackPanel + per-layer fetchers
  // honouring `effectiveResultId`) is deferred — see TODO below.
  const [resultView, setResultView] = useState('original'); // 'original' | 'refined'
  const originalResultId = indexingResult?.result_id;
  const refinementInfo = useDataStore((s) => s.refinementComputed[originalResultId]);
  const effectiveResultId = (resultView === 'refined' && refinementInfo?.refined_result_id)
    ? refinementInfo.refined_result_id
    : originalResultId;
  // TODO(refinement-v2): thread `effectiveResultId` into LayerStackPanel
  // and the layer fetchers so all derived layers reflow when the view
  // toggle flips. v1 keeps layers on the original result and uses the
  // RefinementPanel to display the refined CrystalMap.
  const [askConfirm, confirmProps] = useConfirm();
  const [askPrompt, promptProps] = usePrompt();
  const [gallery, setGallery] = useState([]);
  const [selectedGalleryIdx, setSelectedGalleryIdx] = useState(-1);
  const [showMatchesDialog, setShowMatchesDialog] = useState(false);
  // Pre-seeded pixel for the PatternMatchesDialog when launched from the
  // Anomaly Browser. Null = user opened the dialog manually (no pre-select).
  const [matchesInitialPixel, setMatchesInitialPixel] = useState(null);
  // Forward Diagnostics anomaly browser drawer (drawer component built in Task 22).
  const [browserOpen, setBrowserOpen] = useState(false);
  // Transient hover marker driven by the AnomalyBrowserDrawer. Rendered as
  // an absolutely-positioned div on top of the LayeredCanvas. Cleared on
  // mouse leave.
  const [hoverPixel, setHoverPixel] = useState(null);

  // Save-as Format-Picker. When non-null the modal is open and holds the
  // gallery entry that triggered the save. The user picks one of three
  // formats; we then open the native save dialog with the matching default
  // filename + extension and POST /api/indexing/export with that format.
  const [saveFormatPickerEntry, setSaveFormatPickerEntry] = useState(null);

  // Image state
  const [mapImage, setMapImage] = useState(null);  // base64 PNG
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState(null);

  // Phase B: Forward-NCC map state
  const [forwardNccState, setForwardNccState] = useState({
    ready: false,        // is a cached map available?
    computing: false,    // are we currently computing?
    needsCompute: false, // user selected forward_ncc but no map cached
    stats: null,         // {min, max, mean} from latest compute/status
    bandwidth: 256,      // user-selectable; default Standard
  });
  const [infoText, setInfoText] = useState(t('phasemap:info.noPhaseMapLoaded'));

  // Data source
  const [crystalMapPath, setCrystalMapPath] = useState('');
  const [phaseArrayPath, setPhaseArrayPath] = useState('');
  const [loadingCrystal, setLoadingCrystal] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState(false);
  const [dataMsg, setDataMsg] = useState(null);
  const [dataMsgErr, setDataMsgErr] = useState(false);

  // Scalebar settings
  const [showScalebar, setShowScalebar] = useState(true);
  const [sbPosition, setSbPosition] = useState('lower right');
  const [sbLength, setSbLength] = useState(0.25);
  const [sbFontSize, setSbFontSize] = useState(10);
  const [sbBarColor, setSbBarColor] = useState('#ffffff');
  const [sbBoxColor, setSbBoxColor] = useState('#000000');
  const [sbBoxAlpha, setSbBoxAlpha] = useState(0.6);

  // Display settings
  const [displayMode, setDisplayMode] = useState('phase');
  const [showLegend, setShowLegend] = useState(true);
  const [showIpfKeys, setShowIpfKeys] = useState(true);
  // Blank by default so the canvas overlay auto-derives from the active
  // layer stack. User typing in the Title input overrides the auto value.
  const [mapTitle, setMapTitle] = useState('');
  const [availableModes, setAvailableModes] = useState(BASE_DISPLAY_MODES);

  // Confidence overlay
  const [showConfidence, setShowConfidence] = useState(false);
  const [confAlpha, setConfAlpha] = useState(50);
  const [confCmap, setConfCmap] = useState('RdYlGn');

  // Post-processing cleanup — live preview (render params only, non-persistent)
  // "Apply" writes these into the in-memory result (or batch checkpoint) via
  // /apply-cleanup so downstream exports + the Analysis handoff see them.
  const [cleanupCI, setCleanupCI] = useState(0);
  const [cleanupUnc, setCleanupUnc] = useState(0);
  const [cleanupMinCluster, setCleanupMinCluster] = useState(0);
  const [cleanupFillUnindexed, setCleanupFillUnindexed] = useState(false);
  const [cleanupModalSize, setCleanupModalSize] = useState(0); // 0 / 3 / 5
  const [cleanupApplying, setCleanupApplying] = useState(false);
  const [cleanupMsg, setCleanupMsg] = useState(null);
  const cleanupActive = (
    cleanupCI > 0 || cleanupUnc > 0 || cleanupMinCluster > 0
    || cleanupFillUnindexed || cleanupModalSize > 0
  );

  // ---- Layer stack ---------------------------------------------------
  // resetSignal bumps when *either* the selected gallery entry changes OR
  // the backend's active result changes. The latter matters because the
  // backend's `get_last_indexing_result()` is global state — when the user
  // clicks a different gallery entry we have to await activateResult()
  // before refetching layers, otherwise we'd fetch the OLD result's pixels
  // for the NEW selection. `backendSyncTick` is the version stamp the
  // gallery-select handler bumps after activation completes.
  const [backendSyncTick, setBackendSyncTick] = useState(0);
  const resetSignal = useMemo(() => {
    const entry = gallery[selectedGalleryIdx];
    return entry ? `${entry.id}-${backendSyncTick}` : null;
  }, [gallery, selectedGalleryIdx, backendSyncTick]);

  const cleanupParams = useMemo(() => ({
    ci_threshold: cleanupCI,
    uncertainty_threshold: cleanupUnc,
    min_cluster_size: cleanupMinCluster,
    fill_unindexed: cleanupFillUnindexed,
    modal_filter_size: cleanupModalSize,
  }), [cleanupCI, cleanupUnc, cleanupMinCluster, cleanupFillUnindexed, cleanupModalSize]);

  const layerStack = useLayerStack({ cleanupParams, resetSignal, frameSig, colorOverrides: phaseColorOverrides });

  // ----- Tier-2 tooling state (new) -----
  const [view, setView] = useState('stack');           // 'stack' | 'grid'
  const [tileMinWidth, setTileMinWidth] = useState(240);
  const [linescanMode, setLinescanMode] = useState(false);
  const [magnifierEnabled, setMagnifierEnabled] = useState(false);
  const [swipe, setSwipe] = useState({ a: null, b: null, splitX: 0.5 });

  // Probe / Region-Stats / Linescan hooks
  const { probe, error: probeError, requestProbe, clear: clearProbe } = usePhaseMapProbe();
  const regionStats = usePhaseMapRegionStats();
  const visibleLayerIds = useMemo(
    () => layerStack.layers.filter((l) => l.visible).map((l) => l.id),
    [layerStack.layers],
  );
  const linescan = usePhaseMapLinescan({ layerIds: visibleLayerIds });

  const onSwipeChange = useCallback((next) => setSwipe((s) => ({ ...s, ...next })), []);
  const onLineComplete = useCallback(({ start, end }) => linescan.fetchProfile(start, end), [linescan]);
  const onRegionSelected = useCallback(({ rowStart, rowEnd, colStart, colEnd }) => {
    regionStats.fetchStats({ row: rowStart, col: colStart }, { row: rowEnd, col: colEnd });
  }, [regionStats]);

  // useLayerStack doesn't expose `shape`. Derive it from the first available
  // bitmap: ImageBitmap dimensions are the natural shape of the layer grid
  // (all layers in a stack share the same native shape — the backend enforces
  // this — so any bitmap suffices). bitmapVersion is included in deps because
  // the bitmaps Map identity is stable (`cacheRef.current`) and React would
  // otherwise miss bitmap-add events.
  const stackShape = useMemo(() => {
    for (const bmp of layerStack.bitmaps.values?.() || []) {
      if (bmp && bmp.width && bmp.height) return [bmp.height, bmp.width];
    }
    return null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layerStack.bitmaps, layerStack.bitmapVersion]);

  const onExportPng = useCallback(() => {
    exportComposite({
      layers: layerStack.layers,
      bitmaps: layerStack.bitmaps,
      shape: stackShape,
      filename: 'phase-map-composite.png',
    });
  }, [layerStack.layers, layerStack.bitmaps, stackShape]);

  // Threshold extras for the LayerStackPanel — show only for scalar layers
  // where threshold-on-luma carries physical meaning. Phase / IPF layers are
  // categorical colour maps and skip the histogram.
  const SCALAR_THRESHOLD_KINDS = useMemo(() => new Set([
    'bc', 'ci', 'kam', 'gos', 'uncertainty',
    'forward_ncc', 'local_anomaly', 'pc_sensitivity', 'pattern_residual',
    'refined_ncc', 'convergence_status', 'orientation_delta',
    'pc_delta_x', 'pc_delta_y', 'pc_delta_l',
  ]), []);

  const renderLayerExtras = useCallback((layer) => {
    // Misindex-diagnose helper (2026-05-26): ci-threshold layer gets two
    // number inputs for the band [min, max]. Pixels with CI outside this
    // window are painted in `out_color` (default red) by the backend.
    // We use number inputs (not a slider) so the user can type exact
    // thresholds (e.g. 0.5). Backend re-renders the layer whenever
    // setLayerParams fires — useLayerStack auto-invalidates the bitmap.
    if (layer.id === 'ci-threshold') {
      const params = layer.params || {};
      const bmin = params.band_min ?? 0.0;
      const bmax = params.band_max ?? 0.3;
      const onMin = (e) => {
        const v = Math.max(0, Math.min(1, parseFloat(e.target.value) || 0));
        layerStack.setLayerParams(layer.id, { band_min: v });
      };
      const onMax = (e) => {
        const v = Math.max(0, Math.min(1, parseFloat(e.target.value) || 1));
        layerStack.setLayerParams(layer.id, { band_max: v });
      };
      const labelStyle = {
        fontSize: '8.5pt', color: '#9ca3af',
        display: 'inline-block', marginRight: 4,
      };
      const inputStyle = {
        width: 52, fontSize: '9pt',
        background: '#1f2937', color: '#e5e7eb',
        border: '1px solid #374151', borderRadius: 3,
        padding: '2px 4px', fontVariantNumeric: 'tabular-nums',
      };
      return (
        <div style={{ display: 'flex', gap: 12, padding: '4px 2px', alignItems: 'center' }}>
          <span style={labelStyle}>{t('phasemap:ciThreshold.highlightIn')}</span>
          <label>
            <span style={labelStyle}>{t('phasemap:ciThreshold.min')}</span>
            <input type="number" step={0.05} min={0} max={1} value={bmin}
                   onChange={onMin} style={inputStyle} title={t('phasemap:hoverTips.ciBandMin')} />
          </label>
          <span style={{ color: '#9ca3af' }}>–</span>
          <label>
            <span style={labelStyle}>{t('phasemap:ciThreshold.max')}</span>
            <input type="number" step={0.05} min={0} max={1} value={bmax}
                   onChange={onMax} style={inputStyle} title={t('phasemap:hoverTips.ciBandMax')} />
          </label>
        </div>
      );
    }
    const isCiPerPhase = typeof layer.id === 'string' && layer.id.startsWith('ci_');
    if (!SCALAR_THRESHOLD_KINDS.has(layer.kind) && !isCiPerPhase) return null;
    return (
      <ThresholdHistogram
        bitmap={layerStack.bitmaps.get(layer.id)}
        threshold={layer.threshold}
        onThresholdChange={(t) => layerStack.setThreshold(layer.id, t)}
      />
    );
  }, [layerStack, SCALAR_THRESHOLD_KINDS]);

  const activeEntry = gallery[selectedGalleryIdx] ?? null;
  const resultShape = activeEntry?.data?.shape ?? null;
  const sourceLink = useSourceLink({ resultEntry: activeEntry, resultShape });

  // -------- Dynamic catalog discovery -----------------------------------
  // The static LAYER_SOURCES only enumerates always-present layers. Per-
  // phase CI maps, EDS element maps, and electron images depend on what's
  // loaded — discover them on result/source change.
  const [dynamicLayers, setDynamicLayers] = useState([]);

  useEffect(() => {
    let cancelled = false;
    const promises = [];
    // Per-phase CI from available-maps (result-scoped)
    promises.push(
      phaseMapApi.availableMaps().then((res) => {
        const out = [];
        for (const m of res.data?.maps ?? []) {
          if (m.id && m.id.startsWith && m.id.startsWith('ci_')) {
            out.push({
              id: m.id, label: m.label ?? m.id, source: 'result',
              defaultBlend: 'normal', defaultOpacity: 0.6,
              groupLabel: 'Indexing Result',
            });
          }
        }
        return out;
      }).catch(() => [])
    );
    // EDS elements from h5 session — only relevant when source is linked
    if (sourceLink.linked) {
      promises.push(
        h5Api.getEDSElements().then((res) => {
          const elements = res.data?.elements ?? [];
          return elements.map((el) => ({
            id: `eds:${el}`, label: `EDS: ${el}`, source: 'h5',
            defaultBlend: 'screen', defaultOpacity: 0.5,
            groupLabel: 'H5OINA Source',
          }));
        }).catch(() => [])
      );
      promises.push(
        h5Api.getElectronList().then((res) => {
          const names = res.data?.images ?? res.data ?? [];
          const arr = Array.isArray(names) ? names : [];
          return arr.map((name) => ({
            id: `se:${name}`, label: `SE: ${name}`, source: 'h5',
            defaultBlend: 'normal', defaultOpacity: 1.0,
            groupLabel: 'H5OINA Source',
          }));
        }).catch(() => [])
      );
    }
    Promise.all(promises).then((groups) => {
      if (cancelled) return;
      setDynamicLayers(groups.flat());
    });
    return () => { cancelled = true; };
  }, [resetSignal, sourceLink.linked]);

  // Build the "Add Layer" dropdown options (static + dynamic, dedup'd)
  const availableToAdd = useMemo(() => {
    const opts = [];
    const used = new Set(layerStack.layers.map((l) => l.id));
    for (const [groupKey, group] of Object.entries(LAYER_SOURCES)) {
      for (const layer of group.layers) {
        if (used.has(layer.id)) continue;
        opts.push({
          value: layer.id,
          label: `${group.label}: ${layer.label}`,
          disabled: groupKey === 'h5' && !sourceLink.linked,
          tip: groupKey === 'h5' && !sourceLink.linked ? sourceLink.reason : null,
        });
      }
    }
    for (const d of dynamicLayers) {
      if (used.has(d.id)) continue;
      opts.push({
        value: d.id,
        label: `${d.groupLabel}: ${d.label}`,
        disabled: false,
      });
    }
    return opts;
  }, [layerStack.layers, dynamicLayers, sourceLink.linked, sourceLink.reason]);

  // -------- IPF-key overlay --------------------------------------------
  // Show the IPF colour-key triangles as a canvas overlay whenever any
  // IPF layer is in the stack. The PNG comes from the backend and is
  // cached per direction × result. Fetches lazily.
  const ipfLayer = useMemo(
    () => layerStack.layers.find((l) => l.id?.startsWith?.('ipf-')),
    [layerStack.layers],
  );
  const hasIpfLayer = !!ipfLayer;
  const ipfDirection = ipfLayer ? ipfLayer.id.slice(4).toUpperCase() : 'Z';
  const [ipfKeyImage, setIpfKeyImage] = useState(null);
  const ipfKeyCacheRef = useRef(new Map()); // key: `${resultId}|${direction}` -> base64

  useEffect(() => {
    if (!hasIpfLayer || !resetSignal) { setIpfKeyImage(null); return; }
    const cacheKey = `${resetSignal}|${ipfDirection}`;
    const cached = ipfKeyCacheRef.current.get(cacheKey);
    if (cached) { setIpfKeyImage(cached); return; }
    let cancelled = false;
    phaseMapApi.ipfKey(ipfDirection).then((res) => {
      if (cancelled) return;
      const img = res.data?.image ?? null;
      if (img) ipfKeyCacheRef.current.set(cacheKey, img);
      setIpfKeyImage(img);
    }).catch(() => {
      if (!cancelled) setIpfKeyImage(null);
    });
    return () => { cancelled = true; };
  }, [hasIpfLayer, ipfDirection, resetSignal]);

  // Flush IPF key cache on result change
  useEffect(() => {
    ipfKeyCacheRef.current.clear();
  }, [resetSignal]);

  // Pattern Center of the active dataset/detector (the PC the indexing ran
  // with). Surfaced on the phase map so the user can see it without going
  // back to the Indexing page. Re-reads whenever the active result/file
  // changes; falls back to "not shown" on any error.
  const [pcValue, setPcValue] = useState(null);
  useEffect(() => {
    let cancelled = false;
    pcApi.detectorInfo()
      .then((r) => {
        if (cancelled) return;
        const d = r.data;
        setPcValue(
          d?.has_detector && Array.isArray(d.pc) && d.pc.length === 3 ? d.pc : null,
        );
      })
      .catch(() => { if (!cancelled) setPcValue(null); });
    return () => { cancelled = true; };
  }, [resetSignal]);

  // Auto-derive the canvas title from the active layer stack so users
  // don't see "Phase Map" while looking at IPF-Z. The mapTitle text input
  // becomes an *override*: when it's blank, we fall through to the auto
  // value. When the user types something, their override wins.
  const autoTitle = useMemo(() => {
    const visible = layerStack.layers.filter((l) => l.visible);
    if (visible.length === 0) return '';
    if (visible.length === 1) return visible[0].label;
    // Multi-layer stack: name the topmost layer plus a counter
    const top = visible[visible.length - 1].label;
    return `${top}  +${visible.length - 1}`;
  }, [layerStack.layers]);
  const effectiveTitle = mapTitle.trim() || autoTitle;

  // Calibration
  const [stepX, setStepX] = useState(1.0);
  const [stepY, setStepY] = useState(1.0);

  // Export
  const [exportPath, setExportPath] = useState('');
  const [exportFormat, setExportFormat] = useState('png');
  const [exportDpi, setExportDpi] = useState(300);
  const [exporting, setExporting] = useState(false);
  const [exportMsg, setExportMsg] = useState(null);
  const [exportErr, setExportErr] = useState(false);

  // Send to analysis
  const [sendToAnalysisEnabled, setSendToAnalysisEnabled] = useState(false);

  // Load available map types (includes per-phase CI maps from multi-phase indexing).
  //
  // Trigger on BOTH mapImage and selectedGalleryIdx: switching gallery entries
  // runs deactivate + analysisApi.load before handleRefreshPreview, so by the
  // time mapImage updates the backend already reflects the new entry. We also
  // refetch on selectedGalleryIdx directly as a belt-and-braces — previously
  // a cached identical image could skip the effect, leaving a .h5's CI —
  // <phase> entries in the dropdown after switching to a .ang gallery entry.
  // We also drop the `maps.length > 0` guard: the backend always returns at
  // least the 5 base modes, and if the user switches to an entry without
  // per-phase data we WANT the dropdown to collapse back to 5 items, not
  // linger on the previous 8.
  useEffect(() => {
    phaseMapApi.availableMaps()
      .then((res) => {
        const maps = res.data?.maps ?? [];
        const modes = maps.length > 0
          ? maps.map((m) => ({
              value: m.id,
              label: m.label,
              tip: m.label,
              group: m.group,
            }))
          : BASE_DISPLAY_MODES;
        setAvailableModes(modes);
        // If the current selection no longer exists (switched from an entry
        // that had "CI — Al" to one that doesn't), fall back to Phase Map
        // so the preview panel doesn't try to render a non-existent mode.
        if (!modes.some((m) => m.value === displayMode)) {
          setDisplayMode('phase');
        }
      })
      .catch(() => { setAvailableModes(BASE_DISPLAY_MODES); });
  }, [mapImage, selectedGalleryIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Handlers (defined before useEffects that reference them)
  // ---------------------------------------------------------------------------
  // Phase B handler: triggers compute or fetches the cached forward-NCC heatmap
  const handleComputeForwardNcc = useCallback(async (force = false) => {
    setForwardNccState(s => ({ ...s, computing: true, needsCompute: false }));
    setMapLoading(true);
    setMapError(null);
    try {
      const res = await indexApi.forwardNccCompute(forwardNccState.bandwidth, force);
      const stats = res?.data || {};
      // Now fetch the heatmap
      const hmRes = await indexApi.forwardNccHeatmap();
      const img = hmRes?.data?.heatmap_clean;
      if (img) {
        setMapImage(img);
        setForwardNccState(s => ({
          ...s, computing: false, ready: true, needsCompute: false,
          stats: { min: stats.min, max: stats.max, mean: stats.mean },
          bandwidth: stats.bandwidth ?? s.bandwidth,
        }));
      } else {
        setMapError(t('phasemap:info.forwardNccNoImage'));
        setForwardNccState(s => ({ ...s, computing: false }));
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('phasemap:info.forwardNccComputeFailed');
      setMapError(msg);
      setForwardNccState(s => ({ ...s, computing: false }));
    } finally {
      setMapLoading(false);
    }
  }, [forwardNccState.bandwidth]);

  const handleRefreshPreview = useCallback(async () => {
    // Phase B: Forward-NCC has its own load path
    if (displayMode === 'forward_ncc') {
      setMapLoading(true);
      setMapError(null);
      try {
        const status = await indexApi.forwardNccStatus();
        if (status?.data?.ready) {
          const hmRes = await indexApi.forwardNccHeatmap();
          const img = hmRes?.data?.heatmap_clean;
          if (img) {
            setMapImage(img);
            setForwardNccState(s => ({
              ...s, ready: true, needsCompute: false,
              stats: { min: status.data.min, max: status.data.max, mean: status.data.mean },
              bandwidth: status.data.bandwidth ?? s.bandwidth,
            }));
          }
        } else {
          // Not computed yet -- show the Compute button overlay
          setMapImage(null);
          setForwardNccState(s => ({ ...s, ready: false, needsCompute: true }));
        }
      } catch (err) {
        const msg = err.response?.data?.detail || err.message || t('phasemap:info.forwardNccUnavailable');
        setMapError(msg);
      } finally {
        setMapLoading(false);
      }
      return;
    }

    setMapLoading(true);
    setMapError(null);
    try {
      const direction = directionFromMode(displayMode);
      const res = await phaseMapApi.render({
        direction,
        scalebar_enabled: showScalebar,
        scalebar_location: sbPosition,
        scalebar_length: sbLength,
        scalebar_font_size: sbFontSize,
        scalebar_bar_color: sbBarColor,
        scalebar_box_color: sbBoxColor,
        scalebar_box_alpha: sbBoxAlpha,
        show_legend: showLegend,
        show_ipf_keys: showIpfKeys,
        title: mapTitle,
        step_x: stepX,
        step_y: stepY,
        confidence_overlay: showConfidence,
        confidence_alpha: confAlpha / 100,
        confidence_cmap: confCmap,
        ci_threshold: cleanupCI,
        uncertainty_threshold: cleanupUnc,
        min_cluster_size: cleanupMinCluster,
        fill_unindexed: cleanupFillUnindexed,
        modal_filter_size: cleanupModalSize,
        color_overrides: Object.keys(phaseColorOverrides).length > 0
          ? JSON.stringify(phaseColorOverrides)
          : undefined,
      });
      const img = res.data?.image || res.data?.image_base64 || null;
      if (img) {
        setMapImage(img);
        setInfoText('');
      } else {
        setMapError(t('phasemap:info.noImageReturned'));
        setInfoText(t('phasemap:info.noPhaseMapAvailable'));
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('phasemap:info.renderFailed');
      setMapError(msg);
      setInfoText(msg);
    } finally {
      setMapLoading(false);
    }
  }, [displayMode, showScalebar, sbPosition, sbLength, sbFontSize,
      sbBarColor, sbBoxColor, sbBoxAlpha, showLegend, showIpfKeys, mapTitle,
      stepX, stepY, showConfidence, confAlpha, confCmap,
      cleanupCI, cleanupUnc, cleanupMinCluster,
      cleanupFillUnindexed, cleanupModalSize,
      phaseColorOverrides]);

  // ---------------------------------------------------------------------------
  // Batch-Dashboard handoff: load .ang into analysis dataset, then render.
  // Phase Map's render endpoint falls back to _analysis_dataset when there's
  // no _last_result from a single-file indexing — so this drives the entire
  // "open batch result in Phase Map" flow.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!isActive) return;
    let handoff;
    try { handoff = sessionStorage.getItem('batch_handoff_path'); }
    catch { return; }
    if (!handoff) return;
    try {
      sessionStorage.removeItem('batch_handoff_path');
      sessionStorage.removeItem('batch_handoff_source');
    } catch { /* ignore */ }

    // Always show the attempted path in the Load field — so when the handoff
    // fails, the user can see what was tried and either correct it (e.g.
    // a missing export_dir segment) or click Load to retry.
    setCrystalMapPath(handoff);
    import('../../services/api').then(({ analysisApi }) => {
      analysisApi.load(handoff)
        .then((res) => {
          setInfoText(t('phasemap:info.loadedFromBatch'));
          setDataMsg(t('phasemap:data.loadedFromBatchShort'));
          setDataMsgErr(false);
          setSendToAnalysisEnabled(true);
          // Add a gallery entry for the loaded file so the user can rename,
          // keep it while comparing with other results, and so the right-
          // panel "Phases:" legend actually has something to read from.
          // Without this the handoff quietly loaded the data but the
          // gallery stayed empty, which felt broken.
          const fileName = handoff.split(/[\\/]/).pop();
          const d = res?.data || {};
          const newEntry = {
            id: Date.now(),
            result_id: null,
            label: fileName + (d.shape ? ` — ${d.shape.join(' × ')}` : ''),
            data: { source_path: handoff, phases: d.phases || [], shape: d.shape },
          };
          setGallery((prev) => {
            // Avoid duplicates when the handoff fires twice (StrictMode, etc.)
            if (prev.some((e) => e.data?.source_path === handoff)) return prev;
            const next = [...prev, newEntry];
            setSelectedGalleryIdx(next.length - 1);
            return next;
          });
          // Refresh available map types + trigger an actual render
          phaseMapApi.availableMaps().catch(() => {});
          return handleRefreshPreview();
        })
        .catch((err) => {
          const msg = err.response?.data?.detail || err.message || 'Handoff load failed';
          setInfoText(t('phasemap:info.batchHandoffFailed', { msg }));
          setMapError(msg);
          // Surface in the Data Source box too — user can then fix the path
          // in the input and click Load without retyping.
          setDataMsg(t('phasemap:data.batchHandoffFailedHint', { msg }));
          setDataMsgErr(true);
          console.warn('[PhaseMap] batch handoff load failed:', err);
        });
    });
  }, [isActive, handleRefreshPreview]);

  // ---------------------------------------------------------------------------
  // Populate gallery when indexing results change
  // ---------------------------------------------------------------------------
  // Tracks the last indexingResult we've already processed (by result_id
  // when available, else by object identity). Prevents the effect from
  // re-running side-effects when handleRefreshPreview's identity changes
  // (e.g. user tweaks a display setting) but indexingResult hasn't
  // actually changed.
  const lastProcessedResultRef = useRef(null);

  useEffect(() => {
    if (!indexingResult) return;
    const resultKey = indexingResult.result_id || indexingResult;
    if (lastProcessedResultRef.current === resultKey) {
      // Same result we've already processed — don't re-run gallery /
      // selection / backendSyncTick side effects. This was the regression
      // in 35eea87: dedup branch fired setSelectedGalleryIdx every time
      // handleRefreshPreview's identity changed, which clobbered the
      // user's gallery click.
      return;
    }
    lastProcessedResultRef.current = resultKey;

    const label = [
      indexingMethod || 'Indexing',
      indexingResult.phases?.length > 0 ? indexingResult.phases.map(p => typeof p === 'object' ? p.name : p).join(', ') : null,
      indexingResult.mean_ci != null ? `CI: ${indexingResult.mean_ci.toFixed(3)}` : null,
      `${indexingResult.n_indexed ?? '?'} px`,
    ].filter(Boolean).join(' — ');
    const resultId = indexingResult.result_id || null;

    let newSelectedIdx = -1;
    setGallery((prev) => {
      if (prev.length > 0) {
        const last = prev[prev.length - 1];
        if ((resultId && last.result_id === resultId) || last.label === label) {
          // Dedup against the just-appended entry. The last entry is
          // already what this result represents — re-select it (it
          // matches what the canvas just rendered).
          newSelectedIdx = prev.length - 1;
          return prev;
        }
      }
      const newEntry = { id: Date.now(), label, data: indexingResult, result_id: resultId };
      newSelectedIdx = prev.length;  // index in the appended array
      return [...prev, newEntry];
    });
    if (newSelectedIdx >= 0) {
      setSelectedGalleryIdx(newSelectedIdx);
    }
    setSendToAnalysisEnabled(true);
    // Fresh indexing run: backend's _last_indexing_result is the new one,
    // so bump the sync tick to make the layered canvas refetch against it.
    setBackendSyncTick((t) => t + 1);
    handleRefreshPreview();
  }, [indexingResult, handleRefreshPreview]);

  // ---------------------------------------------------------------------------
  // Sync gallery with resultsList from useResultStore. This handles the case
  // where indexing runs happened while PhaseMapPage was unmounted (the
  // store accumulates entries, but the local `gallery` state is rebuilt
  // on mount). We append any store entries that are not already in the
  // gallery by result_id.
  //
  // Do not remove: the indexingResult effect above only fires for the
  // current run while the page is mounted; runs that completed on another
  // tab live only in resultsList until this effect copies them over. When
  // both effects fire on the same setIndexingResult dispatch, the dedup
  // (`seen` Set) makes this a no-op — the indexingResult effect wins.
  //
  // `data` is intentionally thinner than the indexingResult-effect entry:
  // the store only carries the summary fields, not xmap_path/source_path,
  // so entries added by this path skip the file-based handlers in the
  // rename/delete/send-to-analysis flow.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!resultsList || resultsList.length === 0) return;
    setGallery((prev) => {
      const seen = new Set(prev.map((e) => e.result_id).filter(Boolean));
      const toAdd = resultsList.filter((r) => r.result_id && !seen.has(r.result_id));
      if (toAdd.length === 0) return prev;
      const newEntries = toAdd.map((r) => ({
        id: Date.now() + Math.random(),
        label: r.label,
        data: { phases: r.phases, mean_ci: r.mean_ci, n_indexed: r.n_indexed, result_id: r.result_id },
        result_id: r.result_id,
      }));
      return [...prev, ...newEntries];
    });
  }, [resultsList]);

  // ---------------------------------------------------------------------------
  // Load spatial calibration from store (set by EBSDViewer on file load)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (storeStepSize?.x > 0) {
      setStepX(storeStepSize.x);
      setStepY(storeStepSize.y ?? storeStepSize.x);
    } else if (isActive) {
      // Fallback: fetch from API if store not yet populated (only when page is active)
      ebsdApi.getMetadata()
        .then(resp => {
          const ss = resp.data?.step_size;
          if (ss?.x > 0) {
            setStepX(ss.x);
            setStepY(ss.y ?? ss.x);
          }
        })
        .catch(() => { /* no EBSD data loaded yet — keep default */ });
    }
  }, [storeStepSize, isActive]);

  // ---------------------------------------------------------------------------
  // Sync with backend on mount — load all stored results into gallery
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!isActive) return;
    if (indexingResult || gallery.length > 0) return; // already have data from Zustand
    // Fetch all stored results from backend
    indexApi.listResults()
      .then(res => {
        const results = res.data?.results || [];
        if (results.length === 0) return;
        const entries = results.map(r => ({
          id: r.id || Date.now(),
          result_id: r.id,
          label: [
            r.method || 'Indexing',
            r.phases?.length > 0 ? r.phases.map(p => typeof p === 'object' ? p.name : p).join(', ') : null,
            r.mean_ci != null ? `CI: ${r.mean_ci.toFixed(3)}` : null,
            `${r.n_indexed} px`,
          ].filter(Boolean).join(' — '),
          data: r,
        }));
        setGallery(entries);
        // Select the active result if known, otherwise the last
        const activeIdx = results.findIndex(r => r.is_active);
        setSelectedGalleryIdx(activeIdx >= 0 ? activeIdx : entries.length - 1);
        setSendToAnalysisEnabled(true);
      })
      .catch(() => { /* no previous results — that's fine */ });
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Refresh preview when display settings change.
  //
  // Previously this was gated on gallery.length > 0, which broke the
  // Batch-Dashboard handoff: batch doesn't populate the gallery, so changing
  // the Display Mode (e.g. CI Heatmap) did nothing. We refresh whenever a
  // map has already rendered at least once (mapImage is set) — that way
  // switching modes updates the view for both gallery-based and handoff
  // workflows, without firing a render before any data exists.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!isActive) return;
    const havingData = (gallery.length > 0 && selectedGalleryIdx >= 0) || mapImage;
    if (havingData) {
      handleRefreshPreview();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, handleRefreshPreview, gallery.length, selectedGalleryIdx]);

  const handleLoadCrystalMap = async () => {
    if (!crystalMapPath.trim()) return;
    setLoadingCrystal(true);
    setDataMsg(null);
    try {
      await analysisApi.load(crystalMapPath.trim());
      setDataMsg(t('phasemap:data.crystalMapLoaded'));
      setDataMsgErr(false);
      setSendToAnalysisEnabled(true);
      await handleRefreshPreview();
    } catch (err) {
      setDataMsg(err.response?.data?.detail || err.message || t('phasemap:data.crystalMapLoadFailed'));
      setDataMsgErr(true);
    } finally {
      setLoadingCrystal(false);
    }
  };

  const handleImportArray = async () => {
    if (!phaseArrayPath.trim()) return;
    setLoadingPhase(true);
    setDataMsg(null);
    try {
      await analysisApi.load(phaseArrayPath.trim());
      setDataMsg(t('phasemap:data.phaseArrayImported'));
      setDataMsgErr(false);
      await handleRefreshPreview();
    } catch (err) {
      setDataMsg(err.response?.data?.detail || err.message || t('phasemap:data.phaseArrayImportFailed'));
      setDataMsgErr(true);
    } finally {
      setLoadingPhase(false);
    }
  };

  const handleBrowseCrystalMap = async () => {
    let p = null;
    if (window.electronAPI?.openFile) {
      p = await window.electronAPI.openFile({
        filters: [{ name: 'CrystalMap', extensions: ['ang', 'ctf', 'h5', 'hdf5'] }],
      });
    } else {
      askPrompt({
        title: t('phasemap:data.loadCrystalMapTitle'),
        message: t('phasemap:data.loadCrystalMapMessage'),
        defaultValue: crystalMapPath,
        placeholder: '/path/to/map.ang',
        submitLabel: t('common:load'),
        onSubmit: async (val) => {
          if (!val.trim()) return;
          setCrystalMapPath(val.trim());
          setLoadingCrystal(true);
          setDataMsg(null);
          try {
            await analysisApi.load(val.trim());
            setDataMsg(t('phasemap:data.crystalMapLoaded'));
            setDataMsgErr(false);
            setSendToAnalysisEnabled(true);
            await handleRefreshPreview();
          } catch (err) {
            setDataMsg(err.response?.data?.detail || err.message || t('phasemap:data.crystalMapLoadFailed'));
            setDataMsgErr(true);
          } finally { setLoadingCrystal(false); }
        },
      });
      return;
    }
    if (p && p.trim()) {
      setCrystalMapPath(p.trim());
      // Auto-load immediately after browse
      setLoadingCrystal(true);
      setDataMsg(null);
      try {
        await analysisApi.load(p.trim());
        setDataMsg(t('phasemap:data.crystalMapLoaded'));
        setDataMsgErr(false);
        setSendToAnalysisEnabled(true);
        await handleRefreshPreview();
      } catch (err) {
        setDataMsg(err.response?.data?.detail || err.message || t('phasemap:data.crystalMapLoadFailed'));
        setDataMsgErr(true);
      } finally {
        setLoadingCrystal(false);
      }
    }
  };

  const handleBrowsePhaseArray = async () => {
    let p = null;
    if (window.electronAPI?.openFile) {
      p = await window.electronAPI.openFile({
        filters: [{ name: 'Numpy Array', extensions: ['npy'] }],
      });
    } else {
      askPrompt({
        title: t('phasemap:data.importPhaseArrayTitle'),
        message: t('phasemap:data.importPhaseArrayMessage'),
        defaultValue: phaseArrayPath,
        placeholder: '/path/to/phases.npy',
        submitLabel: t('common:import'),
        onSubmit: async (val) => {
          if (!val.trim()) return;
          setPhaseArrayPath(val.trim());
          setLoadingPhase(true);
          setDataMsg(null);
          try {
            await analysisApi.load(val.trim());
            setDataMsg(t('phasemap:data.phaseArrayImported'));
            setDataMsgErr(false);
            await handleRefreshPreview();
          } catch (err) {
            setDataMsg(err.response?.data?.detail || err.message || t('phasemap:data.phaseArrayImportFailed'));
            setDataMsgErr(true);
          } finally {
            setLoadingPhase(false);
          }
        },
      });
      return;
    }
    if (p && p.trim()) {
      setPhaseArrayPath(p.trim());
      // Auto-load immediately after browse
      setLoadingPhase(true);
      setDataMsg(null);
      try {
        await analysisApi.load(p.trim());
        setDataMsg(t('phasemap:data.phaseArrayImported'));
        setDataMsgErr(false);
        await handleRefreshPreview();
      } catch (err) {
        setDataMsg(err.response?.data?.detail || err.message || t('phasemap:data.phaseArrayImportFailed'));
        setDataMsgErr(true);
      } finally {
        setLoadingPhase(false);
      }
    }
  };

  // Build the exact view-params object both /render and /export consume.
  // Centralised so we never ship a preview that disagrees with the saved file.
  const buildViewParams = () => ({
    direction: directionFromMode(displayMode),
    show_legend: showLegend,
    show_ipf_keys: showIpfKeys,
    title: mapTitle,
    step_x: stepX, step_y: stepY,
    scalebar_enabled: showScalebar,
    scalebar_location: sbPosition,
    scalebar_length: sbLength,
    scalebar_font_size: sbFontSize,
    scalebar_bar_color: sbBarColor,
    scalebar_box_color: sbBoxColor,
    scalebar_box_alpha: sbBoxAlpha,
    confidence_overlay: showConfidence,
    confidence_alpha: confAlpha / 100,
    confidence_cmap: confCmap,
    ci_threshold: cleanupCI,
    uncertainty_threshold: cleanupUnc,
    min_cluster_size: cleanupMinCluster,
    fill_unindexed: cleanupFillUnindexed,
    modal_filter_size: cleanupModalSize,
    color_overrides: Object.keys(phaseColorOverrides).length > 0
      ? JSON.stringify(phaseColorOverrides)
      : undefined,
  });

  const handleExport = async (explicitFormat = null) => {
    // User picks format via the native Save dialog. The `explicitFormat`
    // argument lets the three format buttons (PNG/SVG/PDF) pre-filter the
    // dialog and default extension — matches the reviewed Qt UX.
    if (!mapImage) {
      setExportMsg(t('phasemap:exportImage.nothingToExport'));
      setExportErr(true);
      return;
    }
    const fmt = (explicitFormat || exportFormat || 'png').toLowerCase();
    const safeTitle = (mapTitle || 'phase_map').replace(/[^\w.-]+/g, '_');
    let targetPath = null;
    if (window.electronAPI?.saveFile) {
      targetPath = await window.electronAPI.saveFile({
        defaultPath: `${safeTitle}.${fmt}`,
        filters: [
          fmt === 'png' ? { name: 'PNG image', extensions: ['png'] }
            : fmt === 'svg' ? { name: 'SVG (vector)', extensions: ['svg'] }
              : { name: 'PDF (vector)', extensions: ['pdf'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (!targetPath) { setExportMsg(t('phasemap:info.saveCancelled')); setExportErr(false); return; }
    } else {
      targetPath = await new Promise((resolve) => {
        askPrompt({
          title: t('phasemap:exportImage.saveImageTitle'),
          message: t('phasemap:exportImage.saveImageMessage', { fmt }),
          defaultValue: `${safeTitle}.${fmt}`,
          placeholder: `/path/to/${safeTitle}.${fmt}`,
          submitLabel: t('phasemap:exportImage.saveSubmit'),
          onSubmit: (v) => resolve((v || '').trim()),
        });
      });
      if (!targetPath) { setExportMsg(t('phasemap:info.saveCancelled')); setExportErr(false); return; }
    }
    setExporting(true);
    setExportMsg(null);
    try {
      const res = await phaseMapApi.export(targetPath, fmt, exportDpi, buildViewParams());
      setExportMsg(t('phasemap:exportImage.savedTo', { path: res.data?.path || targetPath }));
      setExportErr(false);
      setExportPath(res.data?.path || targetPath);
      setExportFormat(fmt);
    } catch (err) {
      setExportMsg(err.response?.data?.detail || err.message || t('phasemap:exportImage.exportFailed'));
      setExportErr(true);
    } finally {
      setExporting(false);
    }
  };

  // Copy the live preview PNG to the clipboard — great for reports.
  const handleCopyImage = async () => {
    if (!mapImage) {
      setExportMsg(t('phasemap:exportImage.nothingToCopy'));
      setExportErr(true);
      return;
    }
    try {
      const res = await fetch(`data:image/png;base64,${mapImage}`);
      const blob = await res.blob();
      if (navigator.clipboard?.write) {
        await navigator.clipboard.write([
          // eslint-disable-next-line no-undef
          new ClipboardItem({ [blob.type]: blob }),
        ]);
        setExportMsg(t('phasemap:exportImage.copied'));
        setExportErr(false);
      } else {
        setExportMsg(t('phasemap:exportImage.clipboardUnavailable'));
        setExportErr(true);
      }
    } catch (err) {
      setExportMsg(err.message || t('phasemap:exportImage.copyFailed'));
      setExportErr(true);
    }
  };

  const handleGallerySelect = async (idx) => {
    setSelectedGalleryIdx(idx);
    setSendToAnalysisEnabled(idx >= 0 && gallery.length > 0);
    const entry = gallery[idx];
    if (!entry) return;

    // Two paths:
    //   result_id present  → this is a run produced by the current session,
    //                        activate it on the backend.
    //   loaded from file   → the gallery entry carries its source_path; load
    //                        it back into the analysis dataset AND deactivate
    //                        any stale _last_result so /render falls back to
    //                        _analysis_dataset instead of rendering the
    //                        previously-activated indexing run.
    try {
      if (entry.result_id) {
        // activate_result auto-switches the active file on the backend
        // when the result's source_file differs from the current one.
        // We then call syncFromBackend so the global useDataStore (and
        // therefore the FileSwitcher and any page-mount effects) reflect
        // the new active file.
        const activateResp = await indexApi.activateResult(entry.result_id);
        if (activateResp?.data?.auto_switched_file) {
          await syncFromBackend();
        }
      } else if (entry.data?.source_path) {
        // Reload the file — cheap, keeps state fresh when user jumps between
        // multiple loaded files in the gallery.
        await indexApi.deactivateResult().catch(() => {});
        await analysisApi.load(entry.data.source_path);
      }
    } catch (err) {
      console.warn('[PhaseMap] gallery selection switch failed:', err);
      // Don't swallow this silently — a failed re-activation is exactly the
      // "my results disappeared" symptom. Tell the user what happened and why
      // (the backend holds one active dataset; switching files resets results).
      const detail = err?.response?.data?.detail || err?.message || 'unknown error';
      toast.error(t('phasemap:errors.couldntLoadResult', { detail }));
    }
    // Bump the backend-sync tick AFTER activation completes — this makes
    // resetSignal change which triggers a cacheFlush + REPLACE_ALL in
    // useLayerStack, so the layered canvas refetches against the now-
    // current backend result instead of racing with activateResult.
    setBackendSyncTick((t) => t + 1);
    handleRefreshPreview();
  };

  const handleGalleryRename = async () => {
    if (selectedGalleryIdx < 0) return;
    const name = await askPrompt({ message: t('phasemap:renamePrompt'), defaultValue: gallery[selectedGalleryIdx]?.label || '' });
    if (name?.trim()) {
      setGallery((prev) => prev.map((e, i) =>
        i === selectedGalleryIdx ? { ...e, label: name.trim() } : e
      ));
    }
  };

  const handleGalleryDelete = () => {
    if (selectedGalleryIdx < 0) return;
    askConfirm({
      title: t('phasemap:deleteDialog.title'),
      message: t('phasemap:deleteDialog.message'),
      confirmLabel: t('phasemap:deleteDialog.confirm'),
      onConfirm: () => {
        setGallery((prev) => prev.filter((_, i) => i !== selectedGalleryIdx));
        setSelectedGalleryIdx(-1);
        setSendToAnalysisEnabled(false);
        setMapImage(null);
        setInfoText(t('phasemap:info.noPhaseMapLoadedShort'));
      },
    });
  };

  // ---- Save / Load real CrystalMap files (.ang / .h5) ---------------------
  // Save: activates the selected gallery result on the backend, then exports
  //       it to .ang or .h5 (via /api/indexing/export which returns a
  //       FileResponse → browser Save-As dialog).
  // Load: opens a file picker, loads the chosen .ang/.ctf/.h5 into the
  //       analysis dataset, and adds a synthetic gallery entry so Phase Map
  //       can render it.
  //
  // The JSON-only save that was here before didn't actually preserve the
  // CrystalMap data — only labels — so reloading was not useful across
  // sessions or in MTEX. This reuses the same .ang path the batch exports
  // use, which is MTEX-compatible out of the box.

  // Click "Save as…" → open the format picker. The picker then calls
  // handleSaveWithFormat() with one of 'ang' / 'h5_light' / 'h5'. Split so
  // the existing keyboard-shortcut / programmatic callers don't have to
  // know about the picker.
  const handleGallerySave = async () => {
    if (gallery.length === 0) return;
    if (selectedGalleryIdx < 0) {
      setInfoText(t('phasemap:info.selectGalleryFirst'));
      return;
    }
    const entry = gallery[selectedGalleryIdx];
    if (!entry.result_id) {
      setInfoText(t('phasemap:info.noResultId'));
      return;
    }
    setSaveFormatPickerEntry(entry);
  };

  // Per-format save defaults. Keep `_light` separate from the plain `.h5`
  // suffix so the two h5 variants never overwrite each other when the user
  // accepts the dialog default.
  const SAVE_FORMAT_DEFAULTS = {
    ang:       { ext: 'ang', tail: '.ang',       filterName: t('phasemap:save.filterAng')     },
    h5_light:  { ext: 'h5',  tail: '_light.h5',  filterName: t('phasemap:save.filterH5Light')  },
    h5:        { ext: 'h5',  tail: '.h5',        filterName: t('phasemap:save.filterH5Rich')   },
  };

  const handleSaveWithFormat = async (entry, fmt) => {
    setSaveFormatPickerEntry(null);
    const spec = SAVE_FORMAT_DEFAULTS[fmt];
    if (!spec) {
      setInfoText(t('phasemap:info.unknownSaveFormat', { fmt }));
      return;
    }
    const safeStem = (entry.label || 'gallery').replace(/[^\w.-]+/g, '_');
    const defaultName = `${safeStem}${spec.tail}`;
    let targetPath = null;
    if (window.electronAPI?.saveFile) {
      targetPath = await window.electronAPI.saveFile({
        defaultPath: defaultName,
        filters: [
          { name: spec.filterName,  extensions: [spec.ext] },
          { name: 'All Files',      extensions: ['*']       },
        ],
      });
      if (!targetPath) { setInfoText(t('phasemap:info.saveCancelled')); return; }
    } else {
      const fmtLabel = fmt === 'h5_light' ? t('phasemap:save.h5Light') : (fmt === 'h5' ? t('phasemap:save.h5Rich') : t('phasemap:save.ang'));
      targetPath = await new Promise((resolve) => {
        askPrompt({
          title: t('phasemap:save.promptTitle', { format: fmtLabel }),
          message: t('phasemap:save.promptMessage', { tail: spec.tail }),
          defaultValue: defaultName,
          placeholder: `/path/to/${defaultName}`,
          submitLabel: t('phasemap:save.promptSubmit'),
          onSubmit: (v) => resolve((v || '').trim()),
        });
      });
      if (!targetPath) { setInfoText(t('phasemap:info.saveCancelled')); return; }
    }
    // Make sure the file ends with the right extension. The native dialog
    // doesn't always append the filter's extension when the user typed a
    // bare name, and the picker has already committed us to one format.
    const lower = targetPath.toLowerCase();
    if (fmt === 'h5_light') {
      if (!lower.endsWith('_light.h5')) {
        targetPath = lower.endsWith('.h5')
          ? targetPath.slice(0, -3) + '_light.h5'
          : targetPath + '_light.h5';
      }
    } else if (!lower.endsWith('.' + spec.ext)) {
      targetPath = targetPath + '.' + spec.ext;
    }
    try {
      await indexApi.activateResult(entry.result_id);
      await api.post('/api/indexing/export', {
        format: fmt, filename: targetPath, include_eds: true, include_detector: true,
      }, { responseType: 'blob' });
      setInfoText(t('phasemap:info.saved', { label: entry.label, path: targetPath }));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'export failed';
      setInfoText(t('phasemap:info.saveFailed', { msg }));
    }
  };

  const handleGalleryLoad = async () => {
    // Use Electron's native Open dialog — returns an absolute filesystem path
    // straight away (no browser blob → no second prompt for path). Only list
    // formats orix.io.load actually supports (ang, ctf). Our custom rich
    // result_*.h5 and result_*_light.h5 are intentionally excluded because
    // orix can't parse them; letting the user pick one would surface a
    // cryptic "NoneType has no attribute 'file_reader'" error from rsciio.
    let path = null;
    if (window.electronAPI?.openFile) {
      path = await window.electronAPI.openFile({
        filters: [
          { name: 'CrystalMap (MTEX / GUI export)', extensions: ['ang', 'ctf', 'h5', 'hdf5'] },
          { name: 'All Files',                       extensions: ['*']                        },
        ],
      });
      if (!path) { setInfoText(t('phasemap:info.loadCancelled')); return; }
    } else {
      path = await new Promise((resolve) => {
        askPrompt({
          title: t('phasemap:data.loadCrystalMapTitle'),
          message: t('phasemap:data.loadCrystalMapPathMessage'),
          defaultValue: '',
          placeholder: '/path/to/result.ang',
          submitLabel: t('common:load'),
          onSubmit: (v) => resolve((v || '').trim()),
        });
      });
      if (!path) { setInfoText(t('phasemap:info.loadCancelled')); return; }
    }
    const fileName = path.split(/[\\/]/).pop();
    const isRichOrLightH5 = /\.(h5|hdf5)$/i.test(path);
    try {
      // Deactivate any stale indexing result BEFORE the load; otherwise the
      // render path would keep using _last_result and silently ignore the
      // freshly-loaded analysis dataset.
      await indexApi.deactivateResult().catch(() => {});

      if (isRichOrLightH5) {
        // GUI-flavoured rich/light .h5 — route through /api/indexing/import-h5.
        // That route ALSO calls _store_result so we get a real result_id and
        // Save as… (+ Light export, → Analysis) work straight after load.
        // It additionally wires /1/EBSD into the EBSD viewer when the file
        // is rich (carries the source patterns).
        const res = await api.post('/api/indexing/import-h5', { path });
        const d = res.data || {};
        const caps = d.capabilities || {};
        const newEntry = {
          id: Date.now(),
          result_id: d.result_id,
          label: d.label || fileName,
          data: {
            source_path: path,
            phases: d.phases || [],
            shape: d.shape,
            capabilities: caps,
            ebsd_signal_loaded: !!d.ebsd_signal_loaded,
          },
        };
        setGallery((prev) => {
          const next = [...prev, newEntry];
          setSelectedGalleryIdx(next.length - 1);
          return next;
        });
        setSendToAnalysisEnabled(true);
        // Refresh the Phase Map preview (uses the freshly-activated result)
        // and pull EBSD/EDS state in case patterns were wired to the
        // viewer. syncFromBackend rebuilds useDataStore from the new
        // _ebsd_file_path so the next page-nav to EBSD/EDS sees the data.
        await handleRefreshPreview();
        try { await syncFromBackend?.(); } catch { /* non-fatal */ }
        // Human-readable capability summary so the user knows what got loaded.
        const features = [];
        if (caps.indexing) features.push('indexing');
        if (caps.patterns && d.ebsd_signal_loaded) features.push('patterns→EBSD Viewer');
        else if (caps.patterns) features.push('patterns (failed to wire)');
        if (caps.eds) features.push('EDS');
        if (caps.electron_image) features.push('electron images');
        setInfoText(t('phasemap:info.loaded', { name: fileName, features: features.join(', ') || t('phasemap:info.noRecognisedData') }));
      } else {
        // .ang / .ctf — go through the legacy analysis-only path. These
        // formats only carry orientations; no /1/EBSD or /1/EDS to wire.
        const res = await analysisApi.load(path);
        const d = res.data;
        const newEntry = {
          id: Date.now(),
          result_id: null,
          label: fileName + (d.shape ? ` — ${d.shape.join(' × ')}` : ''),
          data: { source_path: path, phases: d.phases || [], shape: d.shape },
        };
        setGallery((prev) => {
          const next = [...prev, newEntry];
          setSelectedGalleryIdx(next.length - 1);
          return next;
        });
        setSendToAnalysisEnabled(true);
        await handleRefreshPreview();
        setInfoText(t('phasemap:info.loadedIntoGallery', { name: fileName }));
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'load failed';
      setInfoText(t('phasemap:info.loadFailed', { msg }));
    }
  };

  const handleSendToAnalysis = async () => {
    // Load indexing result into Analysis module, then navigate
    try {
      await analysisApi.load('__from_indexing__');
    } catch {
      // Navigate anyway even if load fails — Analysis page can pick up data itself
    }
    if (onNavigate) onNavigate('analysis');
  };

  // ---------------------------------------------------------------------------
  // Sub-components
  // ---------------------------------------------------------------------------
  const inputStyle = {
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    color: colors.text,
    fontSize: '10pt',
    padding: '4px 8px',
    height: spacing.buttonHeight,
    outline: 'none',
    boxSizing: 'border-box',
    flex: 1,
  };

  // Left panel
  const leftPanel = (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      padding: spacing.outerMargin,
      gap: spacing.outerSpacing,
      boxSizing: 'border-box',
    }}>

      {/* Results Gallery */}
      <GroupBox title={t('phasemap:gallery.title')} style={{ flexShrink: 0 }}>
        {/* Phase B: Original/Refined view toggle. Only shown when a refined
            result has been computed for the active indexing result. Kept
            inline (not a refactor of the gallery layout) per v1 scope. */}
        {refinementInfo && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            marginBottom: 6, fontSize: '9pt',
          }}>
            <span style={{ color: colors.textSecondary }}>{t('phasemap:gallery.view')}</span>
            <Button
              small
              variant={resultView === 'original' ? 'primary' : 'default'}
              onClick={() => setResultView('original')}
              disabled={resultView === 'original'}
              title={t('phasemap:gallery.originalTooltip')}
            >
              {t('phasemap:gallery.original')}
            </Button>
            <Button
              small
              variant={resultView === 'refined' ? 'primary' : 'default'}
              onClick={() => setResultView('refined')}
              disabled={resultView === 'refined'}
              title={t('phasemap:gallery.refinedTooltip')}
            >
              {t('phasemap:gallery.refined')}
            </Button>
          </div>
        )}
        <div style={{ display: 'flex', gap: spacing.innerSpacing }}>
          {/* Gallery list */}
          <div className="thin-scrollbar"
            role="listbox"
            aria-label={t('phasemap:gallery.listAria')}
            style={{
              flex: 1,
              background: colors.bgSecondary,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              // Enough for 5–6 entries before scrolling — fits the typical
              // batch comparison flow (one reference + several runs).
              maxHeight: 190,
              overflowY: 'auto',
              fontSize: '9pt',
            }}>
            {gallery.length === 0 ? (
              <div style={{ padding: '12px 8px', color: colors.textSecondary, fontSize: '9pt', textAlign: 'center' }}>
                <span style={{ opacity: 0.3, marginRight: 4 }}>{'\u25A2'}</span>
                {t('phasemap:gallery.empty')}
              </div>
            ) : gallery.map((entry, i) => (
              <GalleryItem
                key={entry.id}
                entry={entry}
                selected={i === selectedGalleryIdx}
                onClick={() => handleGallerySelect(i)}
              />
            ))}
          </div>

          {/* Gallery buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <Button small onClick={handleGalleryRename} disabled={selectedGalleryIdx < 0}
              title={t('phasemap:gallery.renameTooltip')}>
              {t('phasemap:gallery.rename')}
            </Button>
            <Button small variant="danger" onClick={handleGalleryDelete} disabled={selectedGalleryIdx < 0}
              title={t('phasemap:gallery.deleteTooltip')}>
              {t('phasemap:gallery.delete')}
            </Button>
            <Button
              small onClick={handleGallerySave}
              disabled={gallery.length === 0 || selectedGalleryIdx < 0 || !gallery[selectedGalleryIdx]?.result_id}
              title={t('phasemap:gallery.saveAsTooltip')}
            >
              {t('phasemap:gallery.saveAs')}
            </Button>
            <Button
              small onClick={handleGalleryLoad}
              title={t('phasemap:gallery.addFileTooltip')}
            >
              {t('phasemap:gallery.addFile')}
            </Button>
            <Button
              small variant="primary"
              onClick={handleSendToAnalysis}
              disabled={!sendToAnalysisEnabled}
              title={t('phasemap:gallery.toAnalysisTooltip')}
            >
              {t('phasemap:gallery.toAnalysis')}
            </Button>
          </div>
        </div>

        {/* Phase legend for selected result — the swatch is now a color
            picker: click to override the hashed default, right-click to
            reset that phase. Overrides are stored globally (by phase name)
            so they persist across gallery entries and sessions. */}
        {(() => {
          const phases = gallery[selectedGalleryIdx]?.data?.phases || [];
          const realPhases = phases
            .map((p, i) => ({
              id: typeof p === 'object' ? p.id : i,
              name: (typeof p === 'object' ? p.name : p) || t('phasemap:phases.phaseFallback', { index: i }),
            }))
            .filter((p) => !isUnindexedPhase(p.name));
          if (selectedGalleryIdx < 0 || realPhases.length === 0) return null;
          return (
            <div style={{ marginTop: 6, padding: '5px 8px', background: colors.bg, borderRadius: 4, border: `1px solid ${colors.border}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 3 }}>
                <div style={{ fontSize: '8pt', fontWeight: 600, color: colors.cyan || '#89ddff' }}>{t('phasemap:phases.heading')}</div>
                <div style={{ fontSize: '7pt', color: colors.textSecondary }}>
                  {t('phasemap:phases.swatchHint')}
                </div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 12px' }}>
                {realPhases.map((p) => {
                  const currentColor = colorForPhaseName(p.name, phaseColorOverrides);
                  const hasOverride = !!phaseColorOverrides[p.name];
                  return (
                    <label
                      key={`${p.id}-${p.name}`}
                      style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '9pt', cursor: 'pointer' }}
                      onContextMenu={(e) => {
                        e.preventDefault();
                        if (hasOverride) resetPhaseColor(p.name);
                      }}
                      title={hasOverride
                        ? t('phasemap:phases.swatchOverrideTooltip', { name: p.name })
                        : t('phasemap:phases.swatchPickTooltip', { name: p.name })}
                    >
                      <span style={{
                        width: 12, height: 12, borderRadius: 2, flexShrink: 0,
                        display: 'inline-block', position: 'relative',
                        background: currentColor,
                        border: hasOverride ? `1px solid ${colors.text}` : `1px solid transparent`,
                      }}>
                        <input
                          type="color"
                          value={currentColor}
                          onChange={(e) => setPhaseColor(p.name, e.target.value)}
                          style={{
                            position: 'absolute', inset: 0, width: '100%', height: '100%',
                            opacity: 0, cursor: 'pointer', border: 'none', padding: 0,
                          }}
                          aria-label={t('phasemap:phases.colorAria', { name: p.name })}
                        />
                      </span>
                      <span style={{ color: colors.text }}>{p.name}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </GroupBox>

      {/* Tier-2 Tool Toolbar \u2014 sits directly above the canvas area. */}
      <ToolToolbar
        view={view} setView={setView}
        tileMinWidth={tileMinWidth} setTileMinWidth={setTileMinWidth}
        linescanMode={linescanMode} setLinescanMode={setLinescanMode}
        magnifierEnabled={magnifierEnabled} setMagnifierEnabled={setMagnifierEnabled}
        onExport={onExportPng}
        swipe={swipe} setSwipe={onSwipeChange}
        layers={layerStack.layers}
      />

      {/* Canvas area */}
      <div ref={mapContainerRef} className="map-container" style={{
        flex: 1,
        background: colors.bgSecondary,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        minHeight: 200,
        position: 'relative',
      }}>
        {mapLoading ? (
          <LoadingOverlay message={t('phasemap:canvas.rendering')} />
        ) : mapError ? (
          <div style={{ color: colors.red, fontSize: '10pt', padding: 16, textAlign: 'center', animation: 'fadeSlideIn 0.25s ease-out' }}>
            <div style={{ fontSize: '20pt', opacity: 0.5, marginBottom: 6 }}>{'\u26A0'}</div>
            {mapError}
          </div>
        ) : layerStack.layers.length > 0 ? (
          <CanvasInteractionLayer
            view={view}
            layers={layerStack.layers}
            bitmaps={layerStack.bitmaps}
            errors={layerStack.errors}
            shape={stackShape}
            tileMinWidth={tileMinWidth}
            bitmapVersion={layerStack.bitmapVersion}
            scalebar={{
              enabled: showScalebar,
              position: sbPosition,
              length: sbLength,
              fontSize: sbFontSize,
              barColor: sbBarColor,
              boxColor: sbBoxColor,
              boxAlpha: sbBoxAlpha,
            }}
            title={effectiveTitle}
            stepX={stepX}
            ipfKeyImage={ipfKeyImage}
            showIpfKey={hasIpfLayer}
            hoverPixel={hoverPixel}
            onPixelClick={(r, c) => {
              // Drive the existing Pattern Matches dialog flow when a user
              // clicks a pixel \u2014 surface the same affordance as anomaly-browser
              // click.
              setMatchesInitialPixel({ row: r, col: c });
              setShowMatchesDialog(true);
            }}
            onRegionSelected={onRegionSelected}
            onLineComplete={onLineComplete}
            linescanMode={linescanMode}
            magnifierEnabled={magnifierEnabled}
          />
        ) : mapImage ? (
          <>
            <img
              className="image-reveal map-image"
              src={`data:image/png;base64,${mapImage}`}
              alt={t('phasemap:canvas.modeBadgeAlt')}
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
              }}
            />
            {/* Display mode badge */}
            <div style={{
              position: 'absolute', top: 6, left: 8,
              background: '#000000aa', color: colors.accent,
              padding: '2px 8px', borderRadius: 3,
              fontSize: '8pt', fontWeight: 600,
              pointerEvents: 'none',
            }}>
              {availableModes.find(m => m.value === displayMode)?.label || displayMode}
            </div>
          </>
        ) : displayMode === 'forward_ncc' && forwardNccState.needsCompute ? (
          <div style={{ color: colors.textSecondary, fontSize: '10pt', padding: 24, textAlign: 'center', maxWidth: 480 }}>
            <div style={{ fontSize: '24pt', opacity: 0.4, marginBottom: 12 }}>{'\u25A8'}</div>
            <div style={{ color: colors.accent, fontWeight: 700, fontSize: '12pt', marginBottom: 8 }}>
              {t('phasemap:canvas.forwardNccTitle')}
            </div>
            <div style={{ marginBottom: 14, lineHeight: 1.45 }}>
              {t('phasemap:canvas.forwardNccBody')}
            </div>
            <div style={{ marginBottom: 14, fontSize: '9pt', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <label>{t('phasemap:canvas.bandwidth')}</label>
              <select
                value={forwardNccState.bandwidth}
                onChange={(e) => setForwardNccState(s => ({ ...s, bandwidth: Number(e.target.value) }))}
                disabled={forwardNccState.computing}
                title={t('phasemap:hoverTips.forwardNccBandwidth')}
                style={{
                  background: colors.bg, color: colors.text, border: `1px solid ${colors.border}`,
                  borderRadius: 3, padding: '2px 6px', fontSize: '9pt',
                }}
              >
                <option value={128}>{t('phasemap:canvas.bandwidthFast')}</option>
                <option value={256}>{t('phasemap:canvas.bandwidthStandard')}</option>
                <option value={384}>{t('phasemap:canvas.bandwidthHigh')}</option>
              </select>
            </div>
            <button
              onClick={() => handleComputeForwardNcc(false)}
              disabled={forwardNccState.computing}
              title={t('phasemap:hoverTips.forwardNccCompute')}
              style={{
                marginTop: 4, background: colors.accent, border: 'none',
                borderRadius: 4, color: '#000', cursor: forwardNccState.computing ? 'wait' : 'pointer',
                padding: '8px 18px', fontSize: '10pt', fontWeight: 700,
              }}
            >
              {forwardNccState.computing ? t('phasemap:canvas.computing') : t('phasemap:canvas.computeForwardNcc')}
            </button>
            <div style={{ fontSize: '8pt', marginTop: 10, opacity: 0.6 }}>
              {t('phasemap:canvas.forwardNccHint')}
            </div>
          </div>
        ) : (
          <div style={{ color: colors.textSecondary, fontSize: '10pt', padding: 24, textAlign: 'center' }}>
            <div style={{ fontSize: '28pt', opacity: 0.3, marginBottom: 8 }}>{'\u25A8'}</div>
            <div>{infoText}</div>
            <div style={{ fontSize: '8pt', marginTop: 8, opacity: 0.6 }}>
              {t('phasemap:canvas.noDataHint')}
            </div>
            {onNavigate && (
              <button
                onClick={() => onNavigate('indexing')}
                title={t('phasemap:hoverTips.goToIndexing')}
                style={{
                  marginTop: 10, background: 'transparent', border: `1px solid ${colors.border}`,
                  borderRadius: 4, color: colors.textSecondary, cursor: 'pointer',
                  padding: '4px 12px', fontSize: '9pt',
                  transition: 'border-color 0.15s, color 0.15s, background 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = colors.accent; e.currentTarget.style.color = colors.accent; e.currentTarget.style.background = `${colors.accent}11`; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = colors.border; e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.background = 'transparent'; }}
              >
                {t('phasemap:canvas.goToIndexing')}
              </button>
            )}
          </div>
        )}
        {/* Annotation overlay (Legend, Scalebar, Title, North-Arrow).
            Renders inside the map-container so position: absolute is
            resolved against the canvas bounding-rect. Empty until the
            user adds annotations via the toolbar in the side panel. */}
        {annotState.annotations.length > 0 && (
          <AnnotationLayer
            annotations={annotState.annotations}
            selectedId={selectedAnnotId}
            onSelect={setSelectedAnnotId}
            onDelete={(id) => {
              annotState.removeAnnotation(id);
              if (selectedAnnotId === id) setSelectedAnnotId(null);
            }}
            onUpdate={annotState.updateAnnotation}
            containerRef={mapContainerRef}
            ctx={{
              phaseStats: phaseStatsForAnnot,
              stepX,
              // Width of the scan grid (cols) so ScalebarBody can compute
              // the physical bar length: lengthUm / stepX -> scan pixels,
              // divided by scanCols -> fraction of the scan grid.
              scanCols: stackShape ? stackShape[1] : null,
            }}
          />
        )}
      </div>

      {/* Linescan profile (renders below the canvas when a line has been drawn) */}
      {linescan.data && (
        <GroupBox title={t('phasemap:linescan.title')} style={{ marginTop: 8, flexShrink: 0 }}>
          <LinescanProfilePlot data={linescan.data} layers={layerStack.layers} />
        </GroupBox>
      )}
    </div>
  );

  // Right settings panel — order: data → layers → cleanup → annotation → export.
  // Rationale: you can't choose layers before you have data loaded. The old
  // order put Layers above Data Source, which left first-time users staring
  // at an empty layer list with no obvious next step.
  const layerStackPanel = (
    <LayerStackPanel
      layers={layerStack.layers}
      bitmaps={layerStack.bitmaps}
      errors={layerStack.errors}
      onSetOpacity={layerStack.setOpacity}
      onSetBlend={layerStack.setBlend}
      onSetVisibility={layerStack.setVisibility}
      onRemove={layerStack.removeLayer}
      onReorder={layerStack.reorder}
      onAdd={layerStack.addLayer}
      onUsePreset={layerStack.usePreset}
      onSetSingleLayer={layerStack.setSingleLayer}
      availableToAdd={availableToAdd}
      renderLayerExtras={renderLayerExtras}
    />
  );
  const rightPanel = (
    <ScrollPanel style={{ padding: spacing.outerMargin }}>
      {/* Data Source */}
      <GroupBox title={t('phasemap:dataSource.title')}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
          <div>
            <Label small secondary style={{ display: 'block', marginBottom: 2 }}>
              {t('phasemap:dataSource.loadCrystalMapLabel')}
            </Label>
            <div style={{ display: 'flex', gap: 4 }}>
              <input
                type="text"
                value={crystalMapPath}
                onChange={(e) => setCrystalMapPath(e.target.value)}
                placeholder={t('phasemap:dataSource.crystalMapPlaceholder')}
                title={t('phasemap:hoverTips.crystalMapPath')}
                style={{ ...inputStyle }}
              />
              <Button
                small variant="default" onClick={handleBrowseCrystalMap}
                style={{
                  border: `1px solid ${colors.accent}`,
                  color: colors.accent,
                  background: 'transparent',
                }}
                title={t('phasemap:dataSource.browseCrystalMapTooltip')}
              >{t('common:browse')}</Button>
              <Button small variant="primary" onClick={handleLoadCrystalMap} disabled={loadingCrystal || !crystalMapPath.trim()}>
                {loadingCrystal ? t('phasemap:dataSource.loading') : t('common:load')}
              </Button>
            </div>
          </div>
          <div>
            <Label small secondary style={{ display: 'block', marginBottom: 2 }}>
              {t('phasemap:dataSource.importPhaseArrayLabel')}
            </Label>
            <div style={{ display: 'flex', gap: 4 }}>
              <input
                type="text"
                value={phaseArrayPath}
                onChange={(e) => setPhaseArrayPath(e.target.value)}
                placeholder={t('phasemap:dataSource.phaseArrayPlaceholder')}
                title={t('phasemap:hoverTips.phaseArrayPath')}
                style={{ ...inputStyle }}
              />
              <Button
                small variant="default" onClick={handleBrowsePhaseArray}
                style={{
                  border: `1px solid ${colors.accent}`,
                  color: colors.accent,
                  background: 'transparent',
                }}
                title={t('phasemap:dataSource.browsePhaseArrayTooltip')}
              >{t('common:browse')}</Button>
              <Button small variant="primary" onClick={handleImportArray} disabled={loadingPhase || !phaseArrayPath.trim()}>
                {loadingPhase ? t('phasemap:dataSource.loading') : t('common:import')}
              </Button>
            </div>
          </div>
          {dataMsg && (
            <div style={{
              fontSize: '9pt',
              color: dataMsgErr ? colors.red : colors.green,
              padding: '4px 6px',
              border: `1px solid ${dataMsgErr ? colors.red : colors.green}`,
              borderRadius: 4,
              animation: 'fadeSlideIn 0.2s ease-out',
            }}>
              {dataMsg}
            </div>
          )}
        </div>
      </GroupBox>

      <ComputeDiagnosticsPanel
        resultId={indexingResult?.result_id}
        onOpenBrowser={() => setBrowserOpen(true)}
      />

      <RefinementPanel
        resultId={originalResultId}
        onSwitchToRefined={() => setResultView('refined')}
      />

      {layerStackPanel}

      {/* Pattern Center of the active detector (the PC indexing ran with). */}
      {pcValue && (
        <div style={{
          fontSize: '9pt', color: colors.textSecondary,
          padding: '6px 10px', marginBottom: 4,
          border: `1px solid ${colors.border}`, borderRadius: 4,
          background: colors.bgSecondary,
        }}>
          <span style={{ color: colors.accent, fontWeight: 600 }}>{t('phasemap:pc.label')}</span>
          {t('phasemap:pc.value', { x: pcValue[0].toFixed(3), y: pcValue[1].toFixed(3), z: pcValue[2].toFixed(3) })}
        </div>
      )}

      {/* Phase Legend — colour↔phase mapping for the rendered map. The phase
          LAYER is a transparent-bg composite with no baked-in legend, so this
          panel is the only place the user can read which colour is which
          phase. Re-fetches whenever the active result changes. */}
      <PhaseLegend resultId={indexingResult?.result_id ?? null} />
      <PhaseAdjacencyPanel resultId={indexingResult?.result_id ?? null} />
      <AnnotationToolbar
        annotations={annotState.annotations}
        selectedId={selectedAnnotId}
        onSelect={setSelectedAnnotId}
        onAdd={annotState.addAnnotation}
        onRemove={(id) => {
          annotState.removeAnnotation(id);
          if (selectedAnnotId === id) setSelectedAnnotId(null);
        }}
        onUpdate={annotState.updateAnnotation}
        onClear={() => {
          annotState.clearAnnotations();
          setSelectedAnnotId(null);
        }}
      />

      {/* Region Statistics — populated by Shift+drag on the canvas. */}
      {(regionStats.data || regionStats.loading || regionStats.error) && (
        <RegionStatsPanel
          data={regionStats.data}
          error={regionStats.error}
          loading={regionStats.loading}
        />
      )}

      {/* Display Mode dropdown removed — the Layers panel quick-buttons
          (Phase / IPF-Z / IPF-X / IPF-Y / BC / CI) are now the single
          source of truth for what's shown on the canvas. The displayMode,
          showLegend, showIpfKeys state vars still feed the legacy /render
          payload for export, but the in-canvas legend / IPF keys come
          from LayeredCanvas overlays driven directly by the layer stack. */}

      {/* Post-processing cleanup — preview + apply to data */}
      <GroupBox title={t('phasemap:cleanup.title')}>
        <div style={{ fontSize: '8.5pt', color: colors.textSecondary, marginBottom: 6 }}>
          <Trans i18nKey="phasemap:cleanup.livePreview" components={{ 1: <strong /> }} />
        </div>

        {/* Data-safety callout — visible at a glance. Scientists are
            modifying experimental results here, so we spell out what the
            filters change and what they don't. */}
        <div style={{
          fontSize: '8.5pt',
          background: `${colors.accent}10`,
          border: `1px solid ${colors.accent}44`,
          borderRadius: 4,
          padding: '6px 8px',
          marginBottom: 8,
          lineHeight: 1.45,
        }}>
          <div style={{ color: colors.accent, fontWeight: 600, marginBottom: 3 }}>
            {t('phasemap:cleanup.dataHeading')}
          </div>
          <ul style={{ margin: 0, padding: '0 0 0 16px', color: colors.text }}>
            <li>
              <Trans i18nKey="phasemap:cleanup.dataSliders" components={{ 1: <strong />, 3: <em /> }} />
            </li>
            <li>
              <Trans i18nKey="phasemap:cleanup.dataApply" components={{ 1: <strong /> }} />
            </li>
            <li>
              <Trans i18nKey="phasemap:cleanup.dataOrientations" components={{ 1: <strong />, 3: <em />, 5: <em /> }} />
            </li>
          </ul>
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '110px 1fr 52px',
          gap: '4px 8px',
          alignItems: 'center',
          fontSize: '10pt',
          color: colors.text,
        }}>
          <span title={t('phasemap:cleanup.ciThresholdTooltip')}>{t('phasemap:cleanup.ciThresholdLabel')}</span>
          <input type="range" min="0" max="1" step="0.01" value={cleanupCI}
            onChange={(e) => setCleanupCI(parseFloat(e.target.value))}
            title={t('phasemap:hoverTips.cleanupCiSlider')}
            style={{ accentColor: colors.accent, width: '100%' }} />
          <span style={{ fontFamily: 'monospace', fontSize: '9pt' }}>{cleanupCI.toFixed(2)}</span>

          <span title={t('phasemap:cleanup.uncertaintyTooltip')}>{t('phasemap:cleanup.uncertaintyLabel')}</span>
          <input type="range" min="0" max="0.5" step="0.01" value={cleanupUnc}
            onChange={(e) => setCleanupUnc(parseFloat(e.target.value))}
            title={t('phasemap:hoverTips.cleanupUncSlider')}
            style={{ accentColor: colors.accent, width: '100%' }} />
          <span style={{ fontFamily: 'monospace', fontSize: '9pt' }}>{cleanupUnc.toFixed(2)}</span>

        </div>

        {/* Separator: score-based filters above vs spatial filters below.
            Conceptually distinct — CI/Uncertainty reject pixels by a score,
            Min-cluster/Modal/Fill reshape the spatial layout. */}
        <Separator style={{ margin: '6px 0' }} />

        <div style={{
          display: 'grid',
          gridTemplateColumns: '110px 1fr 52px',
          gap: '4px 8px',
          alignItems: 'center',
          fontSize: '10pt',
          color: colors.text,
        }}>
          <span title={t('phasemap:cleanup.minClusterTooltip')}>{t('phasemap:cleanup.minClusterLabel')}</span>
          <input type="range" min="0" max="50" step="1" value={cleanupMinCluster}
            onChange={(e) => setCleanupMinCluster(parseInt(e.target.value, 10))}
            title={t('phasemap:hoverTips.cleanupMinClusterSlider')}
            style={{ accentColor: colors.accent, width: '100%' }} />
          <span style={{ fontFamily: 'monospace', fontSize: '9pt' }}>{cleanupMinCluster}</span>
        </div>

        {/* Modal (majority-vote) smoothing — MTEX-style salt-and-pepper removal */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
          <Label small secondary style={{ minWidth: 110 }} title={t('phasemap:cleanup.modalSmoothingTooltip')}>
            {t('phasemap:cleanup.modalSmoothingLabel')}
          </Label>
          <div role="radiogroup" aria-label={t('phasemap:cleanup.modalSmoothingAria')} style={{ display: 'flex', gap: 2 }}>
            {[
              { v: 0, l: t('phasemap:cleanup.modalOff') },
              { v: 3, l: t('phasemap:cleanup.modal3') },
              { v: 5, l: t('phasemap:cleanup.modal5') },
            ].map(({ v, l }) => (
              <button
                key={v}
                type="button"
                role="radio"
                aria-checked={cleanupModalSize === v}
                onClick={() => setCleanupModalSize(v)}
                title={t('phasemap:hoverTips.modalSmoothing')}
                style={{
                  padding: '3px 10px',
                  fontSize: '9pt',
                  border: `1px solid ${cleanupModalSize === v ? colors.accent : colors.border}`,
                  background: cleanupModalSize === v ? `${colors.accent}22` : 'transparent',
                  color: cleanupModalSize === v ? colors.accent : colors.text,
                  borderRadius: 3,
                  cursor: 'pointer',
                  transition: 'all 0.12s',
                }}
              >{l}</button>
            ))}
          </div>
        </div>

        <label style={{
          display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer',
          marginTop: 6, fontSize: '10pt', color: colors.text,
        }} title={t('phasemap:cleanup.fillUnindexedTooltip')}>
          <input
            type="checkbox"
            checked={cleanupFillUnindexed}
            onChange={(e) => setCleanupFillUnindexed(e.target.checked)}
          />
          <span>{t('phasemap:cleanup.fillUnindexed')}</span>
        </label>

        <div style={{ display: 'flex', gap: 6, marginTop: 8, alignItems: 'center' }}>
          <Button
            variant="primary" small
            onClick={async () => {
              // Infer file stem from sessionStorage handoff (set by
              // BatchDashboard) or from the last-known source path. If no
              // stem is known we still try to persist — the backend will
              // fall back to mutating the in-memory indexing result so
              // subsequent .ang/.h5 exports and the Analysis handoff see
              // the cleaned map.
              let stem = null, dir = null;
              try {
                const src = sessionStorage.getItem('batch_handoff_source')
                         || localStorage.getItem('last_batch_source');
                if (src) {
                  const parts = src.replace(/\\/g, '/').split('/');
                  const fname = parts.pop();
                  stem = fname.replace(/\.[^.]+$/, '');
                  dir = parts.join('/');
                }
              } catch { /* empty */ }
              setCleanupApplying(true); setCleanupMsg(null);
              try {
                const res = await phaseMapApi.applyCleanup({
                  fileStem: stem, searchDir: dir,
                  ciThreshold: cleanupCI,
                  uncertaintyThreshold: cleanupUnc,
                  minClusterSize: cleanupMinCluster,
                  fillUnindexed: cleanupFillUnindexed,
                  modalFilterSize: cleanupModalSize,
                  target: 'auto',
                });
                const d = res.data;
                const where = d.target === 'checkpoint' ? t('phasemap:cleanup.whereCheckpoint') : t('phasemap:cleanup.whereInMemory');
                setCleanupMsg({
                  kind: 'ok',
                  text: t('phasemap:cleanup.appliedTo', { where, moved: d.moved ?? 0, unindexed: d.after_unindexed ?? 0 }),
                });
                // Re-render so the saved state is visible in the preview
                handleRefreshPreview();
              } catch (err) {
                const msg = err.response?.data?.detail || err.message || t('phasemap:cleanup.applyFailed');
                setCleanupMsg({ kind: 'err', text: msg });
              } finally {
                setCleanupApplying(false);
              }
            }}
            disabled={cleanupApplying || !cleanupActive}
            title={t('phasemap:cleanup.applyTooltip')}
            style={{ opacity: (cleanupApplying || !cleanupActive) ? 0.4 : 1 }}
          >
            {cleanupApplying ? t('phasemap:cleanup.applying') : t('phasemap:cleanup.applyButton')}
          </Button>
          <Button
            small
            onClick={() => {
              setCleanupCI(0); setCleanupUnc(0); setCleanupMinCluster(0);
              setCleanupFillUnindexed(false); setCleanupModalSize(0);
              setCleanupMsg(null);
            }}
            title={t('phasemap:cleanup.resetFiltersTooltip')}
          >{t('phasemap:cleanup.resetFilters')}</Button>
        </div>
        {cleanupMsg && (
          <div style={{
            marginTop: 6, padding: '4px 8px', fontSize: '9pt',
            color: cleanupMsg.kind === 'ok' ? colors.green : colors.red,
            background: cleanupMsg.kind === 'ok' ? `${colors.green}15` : `${colors.red}15`,
            border: `1px solid ${cleanupMsg.kind === 'ok' ? colors.green : colors.red}55`,
            borderRadius: 3,
            animation: 'fadeSlideIn 0.2s ease-out',
          }}>{cleanupMsg.text}</div>
        )}
        <details style={{ marginTop: 6, fontSize: '8.5pt', color: colors.textSecondary }}>
          <summary style={{ cursor: 'pointer', userSelect: 'none' }}>{t('phasemap:cleanup.explainSummary')}</summary>
          <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
            <li><Trans i18nKey="phasemap:cleanup.explainCi" components={{ 1: <strong /> }} /></li>
            <li><Trans i18nKey="phasemap:cleanup.explainUncertainty" components={{ 1: <strong /> }} /></li>
            <li><Trans i18nKey="phasemap:cleanup.explainMinCluster" components={{ 1: <strong /> }} /></li>
            <li><Trans i18nKey="phasemap:cleanup.explainModal" components={{ 1: <strong /> }} /></li>
            <li><Trans i18nKey="phasemap:cleanup.explainFill" components={{ 1: <strong /> }} /></li>
          </ul>
        </details>
      </GroupBox>

      {/* Annotation & calibration — merged from previous "Scalebar & Title"
          and "Spatial calibration" groups. Sub-labels group the three logical
          sections (title / scalebar / step size) inside one collapsible. */}
      <CollapsibleGroup title={t('phasemap:annotationCalibration.title')} defaultCollapsed={true}>
        {/* ── Title ───────────────────────────────────────────────────── */}
        <Label small secondary style={{
          display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px',
          fontSize: '8pt', marginBottom: 4,
        }}>{t('phasemap:annotationCalibration.titleSection')}</Label>
        <FormRow label={t('phasemap:annotationCalibration.titleLabel')}>
          <Input
            value={mapTitle}
            onChange={(e) => setMapTitle(e.target.value)}
            placeholder={autoTitle ? t('phasemap:annotationCalibration.titleAutoPlaceholder', { title: autoTitle }) : t('phasemap:annotationCalibration.titleAutoFromLayers')}
            title={t('phasemap:hoverTips.titleInput')}
          />
        </FormRow>

        {/* ── Scalebar ────────────────────────────────────────────────── */}
        <Label small secondary style={{
          display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px',
          fontSize: '8pt', marginTop: 10, marginBottom: 4,
        }}>{t('phasemap:annotationCalibration.scalebarSection')}</Label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}
          title={t('phasemap:hoverTips.showScalebar')}>
          <input
            type="checkbox"
            checked={showScalebar}
            onChange={(e) => setShowScalebar(e.target.checked)}
            title={t('phasemap:hoverTips.showScalebar')}
          />
          <Label>{t('phasemap:annotationCalibration.showScalebar')}</Label>
        </label>

        <FormRow label={t('phasemap:annotationCalibration.position')}>
          <Select
            value={sbPosition}
            onChange={(e) => setSbPosition(e.target.value)}
            options={SCALEBAR_POSITIONS}
            style={{ width: '100%' }}
            title={t('phasemap:hoverTips.scalebarPosition')}
          />
        </FormRow>

        <FormRow label={t('phasemap:annotationCalibration.length')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <NumberInput
              value={sbLength}
              onChange={(e) => setSbLength(Number(e.target.value))}
              min={0.05}
              max={0.50}
              step={0.05}
              style={{ width: 72 }}
              title={t('phasemap:hoverTips.scalebarLength')}
            />
            <Label small secondary>{t('phasemap:annotationCalibration.lengthHint')}</Label>
          </div>
        </FormRow>

        <FormRow label={t('phasemap:annotationCalibration.fontSize')}>
          <NumberInput
            value={sbFontSize}
            onChange={(e) => setSbFontSize(Number(e.target.value))}
            min={6}
            max={24}
            step={1}
            style={{ width: 72 }}
            title={t('phasemap:hoverTips.scalebarFontSize')}
          />
        </FormRow>

        <div style={{ display: 'flex', gap: spacing.innerSpacing }}>
          <div style={{ flex: 1 }}>
            <Label small secondary style={{ display: 'block', marginBottom: 2 }}>{t('phasemap:annotationCalibration.barColour')}</Label>
            <input
              type="color"
              value={sbBarColor}
              onChange={(e) => setSbBarColor(e.target.value)}
              title={t('phasemap:hoverTips.scalebarBarColour')}
              style={{
                width: '100%',
                height: spacing.buttonHeight,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                background: 'transparent',
                cursor: 'pointer',
                padding: 2,
              }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <Label small secondary style={{ display: 'block', marginBottom: 2 }}>{t('phasemap:annotationCalibration.boxColour')}</Label>
            <input
              type="color"
              value={sbBoxColor}
              onChange={(e) => setSbBoxColor(e.target.value)}
              title={t('phasemap:hoverTips.scalebarBoxColour')}
              style={{
                width: '100%',
                height: spacing.buttonHeight,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                background: 'transparent',
                cursor: 'pointer',
                padding: 2,
              }}
            />
          </div>
        </div>

        <FormRow label={t('phasemap:annotationCalibration.boxOpacity')}>
          <NumberInput
            value={sbBoxAlpha}
            onChange={(e) => setSbBoxAlpha(Number(e.target.value))}
            min={0.0}
            max={1.0}
            step={0.1}
            style={{ width: 72 }}
            title={t('phasemap:hoverTips.scalebarBoxOpacity')}
          />
        </FormRow>

        {/* ── Step size (calibration) ──────────────────────────────────── */}
        <Label small secondary style={{
          display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px',
          fontSize: '8pt', marginTop: 10, marginBottom: 4,
        }}>{t('phasemap:annotationCalibration.stepSizeSection')}</Label>
        <FormRow label={t('phasemap:annotationCalibration.stepX')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <NumberInput
              value={stepX}
              onChange={(e) => setStepX(Number(e.target.value))}
              min={0.001}
              max={1000}
              step={0.1}
              style={{ width: 80 }}
              title={t('phasemap:hoverTips.stepX')}
            />
            <Label small secondary>{t('phasemap:annotationCalibration.micron')}</Label>
          </div>
        </FormRow>
        <FormRow label={t('phasemap:annotationCalibration.stepY')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <NumberInput
              value={stepY}
              onChange={(e) => setStepY(Number(e.target.value))}
              min={0.001}
              max={1000}
              step={0.1}
              style={{ width: 80 }}
              title={t('phasemap:hoverTips.stepY')}
            />
            <Label small secondary>{t('phasemap:annotationCalibration.micron')}</Label>
          </div>
        </FormRow>
        <Label small secondary style={{ display: 'block' }}>
          {t('phasemap:annotationCalibration.stepHint')}
        </Label>
      </CollapsibleGroup>

      {/* Coordinate system — orientation reference frame + Pole-Figure popout.
          Changing the frame bumps frameSig, which flushes the IPF layer cache. */}
      <CoordinateSystemPanel defaultCollapsed />
      <Button small style={{ width: '100%', marginTop: spacing.innerSpacing }} onClick={openPoleFigureWindow}
        title={t('phasemap:hoverTips.poleFigure')}>
        {t('phasemap:poleFigure.openButton')}
      </Button>

      {/* Confidence Overlay block removed — replaced by adding "CI (Best)"
          or "CI - <phase>" as a layer in the Layers panel above with full
          opacity / blend control. State vars (showConfidence, confAlpha,
          confCmap) kept at their defaults for the legacy /render export. */}

      <Separator />

      {/* Export — pinned here, prominent because image save was the pain point */}
      <GroupBox title={t('phasemap:exportImage.title')} style={{ border: `1px solid ${colors.accent}55` }}>
        <Label small secondary style={{ display: 'block', marginBottom: 4 }}>
          {t('phasemap:exportImage.blurb')}
        </Label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
          <Label secondary small>{t('phasemap:exportImage.dpi')}</Label>
          <NumberInput
            value={exportDpi}
            onChange={(e) => setExportDpi(Number(e.target.value))}
            min={72}
            max={600}
            step={50}
            style={{ width: 80 }}
            title={t('phasemap:hoverTips.exportDpi')}
          />
          <span style={{ flex: 1 }} />
          <Label small secondary>{exportFormat.toUpperCase()}</Label>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4, marginBottom: 6 }}>
          <Button small variant="primary" onClick={() => handleExport('png')} disabled={exporting || !mapImage}
            title={t('phasemap:exportImage.pngTooltip')}>
            {t('phasemap:exportImage.png')}
          </Button>
          <Button small onClick={() => handleExport('svg')} disabled={exporting || !mapImage}
            title={t('phasemap:exportImage.svgTooltip')}>
            {t('phasemap:exportImage.svg')}
          </Button>
          <Button small onClick={() => handleExport('pdf')} disabled={exporting || !mapImage}
            title={t('phasemap:exportImage.pdfTooltip')}>
            {t('phasemap:exportImage.pdf')}
          </Button>
        </div>
        <Button small onClick={handleCopyImage} disabled={!mapImage}
          title={t('phasemap:exportImage.copyClipboardTooltip')}
          style={{ width: '100%', marginBottom: 4 }}>
          {t('phasemap:exportImage.copyClipboard')}
        </Button>
        {/* Composite export — Layer stack + Annotations baked into one PNG.
            Uses client-side Canvas2D so the on-canvas layout is exactly
            what the user sees in the preview (legend, scalebar, title,
            arrow positions are pixel-identical). Scale factor doubles
            the pixel count for crisper print. */}
        <Button small variant="primary"
          onClick={async () => {
            try {
              const { downloadComposedExport } = await import('./annotations/composeExport');
              await downloadComposedExport({
                layers: layerStack.layers,
                bitmaps: layerStack.bitmaps,
                shape: stackShape ? { rows: stackShape[0], cols: stackShape[1] } : null,
                annotations: annotState.annotations,
                scale: 2,
                phaseStats: phaseStatsForAnnot,
                stepX,  // for physical-length scalebar in the export
              }, `phase_map_composed_${Date.now()}.png`);
            } catch (e) {
              // surface to console only; existing exportMsg is for the
              // backend-driven path and we don't want to fight over it
              console.warn('Composite export failed:', e);
            }
          }}
          disabled={!stackShape || !stackShape[0] || !stackShape[1] || layerStack.layers.length === 0}
          title={t('phasemap:exportImage.composedPngTooltip')}
          style={{ width: '100%', marginBottom: 4 }}>
          {t('phasemap:exportImage.composedPng')}
        </Button>
        <Button small onClick={handleRefreshPreview} disabled={mapLoading}
          title={t('phasemap:exportImage.refreshPreviewTooltip')}
          style={{ width: '100%' }}>
          {mapLoading ? t('phasemap:exportImage.rendering') : t('phasemap:exportImage.refreshPreview')}
        </Button>
        {exportMsg && (
          <div style={{
            marginTop: 6,
            fontSize: '9pt',
            color: exportErr ? colors.red : colors.green,
            padding: '4px 8px',
            background: exportErr ? `${colors.red}15` : `${colors.green}15`,
            border: `1px solid ${exportErr ? colors.red : colors.green}55`,
            borderRadius: 4,
            animation: 'fadeSlideIn 0.2s ease-out',
            wordBreak: 'break-all',
          }}>
            {exporting ? <span className="btn-loading">{t('phasemap:exportImage.exporting')}</span> : exportMsg}
          </div>
        )}
      </GroupBox>

      {/* Secondary action — inspector dialog for dictionary indexing */}
      <Button
        variant="default"
        onClick={() => { setMatchesInitialPixel(null); setShowMatchesDialog(true); }}
        title={t('phasemap:viewMatches.tooltip')}
        style={{ width: '100%', marginTop: spacing.innerSpacing }}
      >
        {t('phasemap:viewMatches.button')}
      </Button>
    </ScrollPanel>
  );

  return (
    <CursorSyncProvider>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        color: colors.text,
        fontFamily: "'Segoe UI', system-ui, sans-serif",
        padding: '12px 16px 0',
        boxSizing: 'border-box',
      }}>
        {/* Page header */}
        <div style={{ marginBottom: 8, flexShrink: 0, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('phasemap:page.title')}</h1>
            <div style={{ fontSize: '10pt', color: colors.textSecondary, marginTop: 2 }}>
              {t('phasemap:page.subtitle')}
            </div>
          </div>
          <FileSwitcher />
        </div>
        <ResizableSplitter
          left={leftPanel}
          right={rightPanel}
          defaultLeftWidth={580}
          minLeftWidth={300}
          maxLeftWidth={900}
          style={{ flex: 1 }}
        />
        <ConfirmDialog {...confirmProps} />
        <PromptDialog {...promptProps} />
        <SaveFormatPicker
          entry={saveFormatPickerEntry}
          onPick={handleSaveWithFormat}
          onCancel={() => setSaveFormatPickerEntry(null)}
        />
        <PatternMatchesDialog
          open={showMatchesDialog}
          onClose={() => setShowMatchesDialog(false)}
          initialPixel={matchesInitialPixel}
        />
        <AnomalyBrowserDrawer
          open={browserOpen}
          onClose={() => setBrowserOpen(false)}
          resultId={indexingResult?.result_id}
          onSelectPixel={(r, c) => {
            setMatchesInitialPixel({ row: r, col: c });
            setShowMatchesDialog(true);
          }}
          onHoverPixel={(r, c) => setHoverPixel({ row: r, col: c })}
          onLeavePixel={() => setHoverPixel(null)}
        />
        {/* Hover probe tooltip — mounted inside CursorSyncProvider so it can
            subscribe to cursor events. position: fixed so it floats above. */}
        <HoverProbeLayer
          probe={probe}
          error={probeError}
          requestProbe={requestProbe}
          clearProbe={clearProbe}
        />
      </div>
    </CursorSyncProvider>
  );
}
