/**
 * IndexingPage.jsx
 *
 * React rewrite of gui/indexing_gui.py — IndexingPage + IndexingPageUI.
 *
 * Layout mirrors the PyQt5 original exactly:
 *   Left  : Navigation map (overview image + selection overlay) + EDS overlay checkbox
 *   Right : Scrollable controls panel
 *           - Dataset selector + data status
 *           - PC & Phase status summary
 *           - Method dropdown (Hough / Dictionary / Spherical / Embedding)
 *           - Multi-Phase Comparison group (collapsible)
 *           - Required Files group (auto-discovery + browse)
 *           - Method-specific parameter groups
 *           - Action buttons (Start, Stop, Compare, → Analysis, Export, Refine)
 *           - Progress bar + quality label
 *           - Log output
 */

import { useState, useEffect, useCallback, useRef, useReducer, useMemo, Fragment } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi, ebsdApi, pcApi, edsApi, dictionaryGpuApi, getGpuStatus, phaseMapApi } from '../../services/api';
import NavigationCanvas from './NavigationCanvas';
import EdsOverlayPanel from './EdsOverlayPanel';
import BatchIndexingDialog from './BatchIndexingDialog';
import GenerateDictionaryDialog from './GenerateDictionaryDialog';
import PhaseDropdown from './PhaseDropdown';
import PhaseResultModal from './PhaseResultModal';
import SinglePixelPhaseTestDialog from './SinglePixelPhaseTestDialog';
import SelectedPhasesList from './SelectedPhasesList';
import LinkedPatternImage from '../PatternMatch/LinkedPatternImage';
import { useLinkedPatternMarkers } from '../PatternMatch/useLinkedPatternMarkers';
import PatternExportDialog from '../PatternMatch/PatternExportDialog';
import PseudoSymmetryPanel from '../common/PseudoSymmetryPanel';
import NeighbourhoodZoom from '../common/NeighbourhoodZoom';
import { detectPhaseDegeneracy } from './phaseDegeneracy';
import FloatingPhasePanel from './FloatingPhasePanel';
import useResultStore from '../../stores/useResultStore';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import { colors as C, spacing } from '../../theme/tokens';
import { toast } from '../../stores/useToastStore';
import {
  Button,
  GroupBox,
  Label,
  NumberInput,
  Select,
  ResizableSplitter,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Method display labels/tips are localised inside the component via
// `methodOptions`. The circular-mask option *values* are localised likewise
// via `circmaskOptions`; the buildParams switch matches the exact English
// sentinel strings ("Off (-1)" / "Inscribed circle (0)" / "Custom radius").
const BANDWIDTH_OPTIONS = ['53', '68', '88', '113', '158', '203', '263', '338'];
const METRIC_OPTIONS    = ['ncc', 'ndp'];

const PHASE_COLORS = [
  '#ff5555', '#50fa7b', '#8be9fd', '#ffb86c',
  C.purple, C.pink, '#f1fa8c', '#6272a4',
];

// ---------------------------------------------------------------------------
// Small shared primitives (scoped to this file, matching PyQt5 look)
// ---------------------------------------------------------------------------

/** Horizontal row of items with a gap */
function Row({ children, gap = 8, style = {} }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap, ...style }}>
      {children}
    </div>
  );
}

/** Compact label (matches PyQt5 QLabel inline) */
function InlineLabel({ children, style = {} }) {
  return (
    <span style={{ color: C.text, fontSize: '10pt', whiteSpace: 'nowrap', ...style }}>
      {children}
    </span>
  );
}

/** Status text label */
function StatusLabel({ children, color = C.text, style = {} }) {
  return (
    <div style={{ fontSize: '9pt', color, padding: '3px 0', wordBreak: 'break-word', ...style }}>
      {children}
    </div>
  );
}

/** Radio button (styled like PyQt5 QRadioButton) */
function RadioOption({ label, checked, onChange, name, title }) {
  return (
    <label title={title} style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '10pt', color: C.text, userSelect: 'none' }}>
      <input
        type="radio"
        name={name}
        checked={checked}
        onChange={onChange}
        style={{ accentColor: C.purple, cursor: 'pointer' }}
      />
      {label}
    </label>
  );
}

/** Checkbox (styled like PyQt5 QCheckBox) */
function CheckOption({ label, checked, onChange, disabled = false, style = {}, title }) {
  return (
    <label title={title} style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: disabled ? 'not-allowed' : 'pointer', fontSize: '10pt', color: disabled ? C.textSecondary : C.text, userSelect: 'none', ...style }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        style={{ accentColor: C.purple, cursor: disabled ? 'not-allowed' : 'pointer' }}
      />
      {label}
    </label>
  );
}

/** EMSphinx preprocessing preview — shows Original | Gaussian BG | Circular Mask + AHE with pixel slider */
function PreprocessingPreview({ row: initRow, col: initCol, gausbckg, circmask, nregions, dataLoaded, nRows, nCols }) {
  const { t } = useTranslation('indexing');
  const [images, setImages] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pixelRow, setPixelRow] = useState(initRow || 0);
  const [pixelCol, setPixelCol] = useState(initCol || 0);
  const timerRef = useRef(null);
  const totalPixels = (nRows || 1) * (nCols || 1);

  // Flat index ↔ (row, col) conversion
  const flatIndex = pixelRow * (nCols || 1) + pixelCol;
  const setFromFlat = (idx) => {
    const nc = nCols || 1;
    setPixelRow(Math.floor(idx / nc));
    setPixelCol(idx % nc);
  };

  useEffect(() => {
    if (!dataLoaded) { setImages(null); return; }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await indexApi.previewPreprocessing({ row: pixelRow, col: pixelCol, gausbckg, circmask, nregions });
        setImages(r.data);
      } catch { setImages(null); }
      setLoading(false);
    }, 300);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    // nRows/nCols are part of the dep list so a FILE switch (which changes
    // the active dataset's grid shape) re-runs the preview even when the
    // user keeps row=col=0 across the switch — otherwise the preview shows
    // the previous file's pixel until the user nudges a control.
  }, [pixelRow, pixelCol, gausbckg, circmask, nregions, dataLoaded, nRows, nCols]);

  const imgStyle = { height: 128, objectFit: 'contain', borderRadius: 3, border: `1px solid ${C.border}`, background: C.bg };
  const labelStyle = { fontSize: '8pt', color: C.textSecondary, textAlign: 'center', marginTop: 2 };

  return (
    <div style={{ marginTop: 8, background: C.bgSecondary, border: `1px solid ${C.border}`, borderRadius: 4, padding: '8px 10px' }}>
      {/* Pixel slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: '8pt', color: C.textSecondary, whiteSpace: 'nowrap' }}>{t('preview.pixel', { row: pixelRow, col: pixelCol })}</span>
        <input type="range" min={0} max={Math.max(0, totalPixels - 1)} value={flatIndex}
          onChange={e => setFromFlat(Number(e.target.value))}
          title={t('hoverTips.previewPixelSlider')}
          style={{ flex: 1, cursor: 'pointer', accentColor: C.accent }}
        />
        <span style={{ fontSize: '7pt', color: C.textSecondary, minWidth: 55, textAlign: 'right' }}>
          {flatIndex + 1}/{totalPixels}
        </span>
      </div>
      {/* Preview images */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'center', gap: 12, minHeight: 128 }}>
        {!dataLoaded && <span style={{ color: C.textSecondary, fontSize: '9pt', padding: 20 }}>{t('preview.loadToSee')}</span>}
        {dataLoaded && loading && <span style={{ color: C.textSecondary, fontSize: '9pt', padding: 20 }}>{t('preview.loading')}</span>}
        {dataLoaded && !loading && !images && <span style={{ color: C.textSecondary, fontSize: '9pt', padding: 20 }}>{t('preview.unavailable')}</span>}
        {images && (
          <>
            <div style={{ textAlign: 'center' }}>
              <img src={`data:image/png;base64,${images.original}`} alt={t('preview.original')} style={imgStyle} />
              <div style={labelStyle}>{t('preview.original')}</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <img src={`data:image/png;base64,${images.gaussian_bg}`} alt={t('preview.gaussianBg')} style={imgStyle} />
              <div style={labelStyle}>{t('preview.gaussianBg')}</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <img src={`data:image/png;base64,${images.circmask_ahe}`} alt={t('preview.maskAhe')} style={imgStyle} />
              <div style={labelStyle}>{t('preview.maskAhe')}</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/** Refine Orientations dialog — Nelder-Mead refinement after Dictionary Indexing */
function RefineDialog({ open, onClose, onLog }) {
  const { t } = useTranslation(['indexing', 'common']);
  const [masterH5, setMasterH5] = useState('');
  const [energy, setEnergy] = useState(20);
  const [trustRegion, setTrustRegion] = useState(5.0);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');

  if (!open) return null;

  const handleBrowse = async () => {
    if (window.electronAPI?.openFile) {
      const path = await window.electronAPI.openFile({
        filters: [{ name: 'HDF5 Master Pattern', extensions: ['h5', 'hdf5'] }],
      });
      if (path) setMasterH5(path);
    }
  };

  const handleStart = async () => {
    if (!masterH5) { onLog(t('refineDialog.selectMaster')); return; }
    setRunning(true);
    setProgress(t('refineDialog.starting'));
    try {
      const r = await indexApi.refine({
        master_h5_path: masterH5,
        energy_kv: energy,
        trust_region: trustRegion,
      });
      const taskId = r.data?.task_id;
      if (!taskId) { setRunning(false); return; }

      const poll = setInterval(async () => {
        try {
          const sr = await indexApi.getStatus(taskId);
          const d = sr.data;
          setProgress(d.message || '');
          if (d.status === 'done' || d.status === 'complete') {
            clearInterval(poll);
            setRunning(false);
            onLog(t('refineDialog.complete'));
            onClose();
          } else if (d.status === 'failed' || d.status === 'error') {
            clearInterval(poll);
            setRunning(false);
            onLog(t('refineDialog.failed', { error: d.error || t('refineDialog.unknownError') }));
          }
        } catch (err) {
          clearInterval(poll);
          setRunning(false);
          onLog(t('refineDialog.pollingError', { error: err.message }));
        }
      }, 1000);
    } catch (err) {
      setRunning(false);
      onLog(t('refineDialog.error', { error: err.response?.data?.detail || err.message }));
    }
  };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999,
      }}
      onClick={onClose}
    >
      <div onClick={e => e.stopPropagation()} style={{
        background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 8, padding: 20, minWidth: 400, maxWidth: 500,
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ color: C.accent, fontWeight: 700, fontSize: 14 }}>{t('refineDialog.title')}</span>
          <button onClick={onClose} title={t('hoverTips.refineClose')} aria-label={t('hoverTips.refineClose')} style={{
            background: 'transparent', border: 'none', color: C.textSecondary, cursor: 'pointer', fontSize: 16,
          }}>&times;</button>
        </div>

        <div style={{ fontSize: '9pt', color: C.textSecondary, marginBottom: 12 }}>
          {t('refineDialog.description')}
        </div>

        {/* Master H5 file */}
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: '9pt', color: C.text, display: 'block', marginBottom: 2 }}>{t('refineDialog.masterPattern')}</label>
          <div style={{ display: 'flex', gap: 4 }}>
            <input
              type="text" value={masterH5}
              onChange={e => setMasterH5(e.target.value)}
              placeholder={t('refineDialog.masterPlaceholder')}
              title={t('hoverTips.refineMasterInput')}
              style={{
                flex: 1, padding: '4px 8px', fontSize: '9pt',
                background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text,
              }}
            />
            <button onClick={handleBrowse} title={t('hoverTips.refineMasterBrowse')} style={{
              padding: '4px 12px', fontSize: '9pt', cursor: 'pointer',
              background: C.accent, color: C.bg, border: 'none', borderRadius: 3,
            }}>{t('common:browse')}</button>
          </div>
        </div>

        {/* Energy + Trust Region */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 8 }}>
          <div>
            <label style={{ fontSize: '9pt', color: C.text, display: 'block', marginBottom: 2 }}>{t('refineDialog.energy')}</label>
            <input
              type="number" value={energy} onChange={e => setEnergy(Number(e.target.value))}
              min={5} max={40} step={0.5}
              title={t('hoverTips.refineEnergy')}
              style={{ width: 70, padding: '4px', fontSize: '9pt', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text }}
            />
          </div>
          <div>
            <label style={{ fontSize: '9pt', color: C.text, display: 'block', marginBottom: 2 }}>{t('refineDialog.trustRegion')}</label>
            <input
              type="number" value={trustRegion} onChange={e => setTrustRegion(Number(e.target.value))}
              min={1} max={15} step={0.5}
              title={t('hoverTips.refineTrustRegion')}
              style={{ width: 70, padding: '4px', fontSize: '9pt', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text }}
            />
          </div>
        </div>

        {/* Progress */}
        {running && (
          <div style={{ fontSize: '9pt', color: C.yellow, marginBottom: 8, padding: '4px 8px', background: `${C.yellow}11`, borderRadius: 3 }}>
            {progress || t('refineDialog.working')}
          </div>
        )}

        {/* Start button */}
        <button
          onClick={handleStart}
          disabled={running || !masterH5}
          title={t('hoverTips.refineStart')}
          style={{
            width: '100%', padding: '8px', fontSize: '10pt', fontWeight: 600,
            cursor: running ? 'wait' : 'pointer',
            background: running ? C.bgTertiary : C.green, color: C.bg,
            border: 'none', borderRadius: 4, opacity: (!masterH5 || running) ? 0.5 : 1,
          }}
        >
          {running ? t('refineDialog.refining') : t('refineDialog.start')}
        </button>
      </div>
    </div>
  );
}

/** Pattern Matches Viewer — 3-panel layout: NCC heatmap | Experimental | Simulated | NCC Image */
function PatternMatchesDialog({ open, onClose }) {
  const { t } = useTranslation('indexing');
  const [heatmapClean, setHeatmapClean] = useState(null);
  const [gridDims, setGridDims] = useState({ rows: 0, cols: 0 });
  const [cropOffset, setCropOffset] = useState({ row: 0, col: 0 });
  const [matchData, setMatchData] = useState(null);
  const [selectedPixel, setSelectedPixel] = useState(null);
  const [rank, setRank] = useState(0);
  const [loading, setLoading] = useState(false);
  // Neighbourhood-zoom source: a full-grid layer PNG (tiny mis-indexed
  // nests read as colour breaks there). User-switchable IPF-Z/Y/X/phase —
  // some nests only show in ONE of those maps. Fail-soft: no layer → nudge
  // only. zoomBump refetches after a grain flip (the colours changed).
  const [zoomLayer, setZoomLayer] = useState(null);
  const [zoomBump, setZoomBump] = useState(0);
  const [zoomKind, setZoomKind] = useState('ipf-z');
  useEffect(() => {
    if (!open) { setZoomLayer(null); return undefined; }
    let cancelled = false;
    phaseMapApi.layer(zoomKind)
      .then(r => { if (!cancelled) setZoomLayer(r.data); })
      .catch(() => { if (!cancelled) setZoomLayer(null); });
    return () => { cancelled = true; };
  }, [open, zoomBump, zoomKind]);
  const nudgePixel = useCallback((dr, dc) => {
    setSelectedPixel(p => {
      if (!p) return p;
      const or = cropOffset.row || 0;
      const oc = cropOffset.col || 0;
      const lr = Math.max(0, Math.min((gridDims.rows || 1) - 1,
        (p.localRow ?? (p.row - or)) + dr));
      const lc = Math.max(0, Math.min((gridDims.cols || 1) - 1,
        (p.localCol ?? (p.col - oc)) + dc));
      return { row: lr + or, col: lc + oc, localRow: lr, localCol: lc };
    });
    setRank(0);
  }, [gridDims.rows, gridDims.cols, cropOffset.row, cropOffset.col]);
  const [stats, setStats] = useState(null);
  const heatmapRef = useRef(null);
  // Linked crosshair + numbered red markers shared across the 3 comparison
  // panels, plus the publication-figure export composer — the SAME shared tools
  // as the Phase Test dialog, so any export change applies to both.
  const markerCtl = useLinkedPatternMarkers();
  const [exportOpen, setExportOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true); setSelectedPixel(null); setMatchData(null); setRank(0);
    indexApi.nccHeatmap().then(r => {
      setHeatmapClean(r.data.heatmap_clean);
      setGridDims({ rows: r.data.n_rows, cols: r.data.n_cols });
      setCropOffset({ row: r.data.crop_row_offset ?? 0, col: r.data.crop_col_offset ?? 0 });
      setStats({ min: r.data.min_score, max: r.data.max_score, mean: r.data.mean_score });
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [open]);

  useEffect(() => {
    if (!selectedPixel) return;
    // Race guard — see PhaseMapPage.jsx for the same pattern. Rapid pixel
    // or rank changes can otherwise show response data for a stale (pixel,
    // rank) pair while the controls display the current one.
    let cancelled = false;
    indexApi.patternMatch(selectedPixel.row, selectedPixel.col, rank)
      .then(r => { if (!cancelled) setMatchData(r.data); })
      .catch(() => { if (!cancelled) setMatchData(null); });
    return () => { cancelled = true; };
  }, [selectedPixel, rank]);

  // Refresh the match view after a grain flip/undo so the corrected
  // orientation (and its simulated pattern) shows immediately. Also bumps
  // the neighbourhood-zoom layer (its IPF colours changed too).
  const refetchMatch = () => {
    if (!selectedPixel) return;
    setZoomBump(x => x + 1);
    indexApi.patternMatch(selectedPixel.row, selectedPixel.col, rank)
      .then(rr => setMatchData(rr.data)).catch(() => {});
  };

  const handleHeatmapClick = (e) => {
    const img = heatmapRef.current;
    if (!img || !gridDims.rows) return;
    const rect = img.getBoundingClientRect();
    // Local coordinates within the cropped heatmap
    const localRow = Math.min(Math.max(0, Math.floor((e.clientY - rect.top) / rect.height * gridDims.rows)), gridDims.rows - 1);
    const localCol = Math.min(Math.max(0, Math.floor((e.clientX - rect.left) / rect.width * gridDims.cols)), gridDims.cols - 1);
    // Translate back to original grid coordinates for the backend
    setSelectedPixel({ row: localRow + cropOffset.row, col: localCol + cropOffset.col, localRow, localCol }); setRank(0);
  };

  if (!open) return null;

  // Crosshair uses local (cropped) coordinates
  const crossX = selectedPixel ? ((selectedPixel.localCol + 0.5) / gridDims.cols * 100) : -10;
  const crossY = selectedPixel ? ((selectedPixel.localRow + 0.5) / gridDims.rows * 100) : -10;

  // R-score color (green >= 0.3, orange >= 0.15, red < 0.15)
  const rColor = matchData?.r_quality === 'good' ? '#50fa7b' : matchData?.r_quality === 'acceptable' ? '#ffb86c' : '#ff5555';
  const rLabel = matchData?.r_quality === 'good' ? t('matchesDialog.goodMatch') : matchData?.r_quality === 'acceptable' ? t('matchesDialog.acceptableMatch') : t('matchesDialog.poorMatch');

  const patStyle = { height: 240, objectFit: 'contain', borderRadius: 3, border: `1px solid ${C.border}`, background: '#000' };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999 }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 8, padding: 20, width: 1100, maxHeight: '90vh', overflow: 'auto',
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <span style={{ color: C.accent, fontWeight: 700, fontSize: 14 }}>{t('matchesDialog.title')}</span>
          <button onClick={onClose} title={t('hoverTips.matchesClose')} aria-label={t('hoverTips.matchesClose')} style={{ background: 'transparent', border: 'none', color: C.textSecondary, cursor: 'pointer', fontSize: 18 }}>&times;</button>
        </div>
        <div style={{ fontSize: '9pt', color: '#6272a4', marginBottom: 12 }}>{t('matchesDialog.clickHint')}</div>

        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          {/* Left: NCC Heatmap (30%) — vertically centred against the
              taller right column so the heatmap sits in line with the
              middle of the pattern panels. */}
          <div style={{ width: '30%', minWidth: 200 }}>
            {loading && <div style={{ color: C.textSecondary, padding: 40, textAlign: 'center' }}>{t('matchesDialog.loading')}</div>}
            {heatmapClean && (
              <div style={{ position: 'relative', cursor: 'crosshair' }} onClick={handleHeatmapClick}>
                <img ref={heatmapRef} src={`data:image/png;base64,${heatmapClean}`} alt="NCC" style={{ width: '100%', display: 'block', borderRadius: 3, border: `1px solid ${C.border}` }} />
                {selectedPixel && (<>
                  <div style={{ position: 'absolute', left: 0, right: 0, top: `${crossY}%`, height: 1, background: '#ffb86c', pointerEvents: 'none' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${crossX}%`, width: 1, background: '#ffb86c', pointerEvents: 'none' }} />
                  <div style={{ position: 'absolute', left: `${crossX}%`, top: `${crossY}%`, width: 12, height: 12, transform: 'translate(-50%,-50%)', pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ffb86c', fontSize: 16, fontWeight: 700 }}>+</div>
                </>)}
              </div>
            )}
            {stats && <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 4 }}>
              {t('matchesDialog.nccRange', { min: stats.min?.toFixed(3), max: stats.max?.toFixed(3), mean: stats.mean?.toFixed(3) })}
            </div>}
            {selectedPixel && matchData && (
              <div style={{ fontSize: '9pt', color: '#f8f8f2', marginTop: 6, textAlign: 'center' }}>
                {t('matchesDialog.pixelScore', { col: selectedPixel.col, row: selectedPixel.row, score: matchData.ncc_score?.toFixed(4) ?? t('matchesDialog.dash') })}
              </div>
            )}
            {/* Neighbourhood zoom + 1-px nudge: tiny nests are hard to hit
                by clicking; step onto them and SEE where you stand. */}
            <NeighbourhoodZoom
              imageB64={zoomLayer?.image}
              shape={zoomLayer?.shape}
              pixel={selectedPixel}
              onNudge={nudgePixel}
              caption={t('matchesDialog.zoomCaption')}
              kind={zoomKind}
              onKindChange={setZoomKind}
              kinds={[
                { id: 'ipf-z', label: 'Z' },
                { id: 'ipf-y', label: 'Y' },
                { id: 'ipf-x', label: 'X' },
                { id: 'phase', label: t('matchesDialog.zoomKindPhase'), sep: true },
              ]}
              labels={{
                up: t('matchesDialog.nudgeUp'), down: t('matchesDialog.nudgeDown'),
                left: t('matchesDialog.nudgeLeft'), right: t('matchesDialog.nudgeRight'),
                tip: t('matchesDialog.nudgeTip'),
                clickTip: t('matchesDialog.nudgeClickTip'),
                layerLead: 'IPF',
                layerGroup: t('matchesDialog.zoomLayerGroup'),
                layerTip: t('matchesDialog.zoomLayerTip'),
              }}
            />
          </div>

          {/* Right: 3-panel comparison (70%) */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {!matchData && <div style={{ color: '#6272a4', fontSize: '11pt', padding: 60, textAlign: 'center' }}>{t('matchesDialog.clickToInspect')}</div>}
            {matchData && (<>
              {/* 3 panels side by side */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <div style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ fontSize: '9pt', color: '#f8f8f2', marginBottom: 4 }}>{t('matchesDialog.experimental')}</div>
                  {matchData.experimental ? <LinkedPatternImage src={`data:image/png;base64,${matchData.experimental}`} alt={t('matchesDialog.experimental')} markers={markerCtl.markers} hover={markerCtl.hover} onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick} style={{ ...patStyle, width: '100%' }} /> : <div style={{ height: 240, background: '#282a36', borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6272a4' }}>{t('matchesDialog.na')}</div>}
                </div>
                <div style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ fontSize: '9pt', color: '#f8f8f2', marginBottom: 4 }}>{t('matchesDialog.bestMatch')}</div>
                  {matchData.simulated ? <LinkedPatternImage src={`data:image/png;base64,${matchData.simulated}`} alt={t('matchesDialog.bestMatch')} markers={markerCtl.markers} hover={markerCtl.hover} onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick} style={{ ...patStyle, width: '100%' }} /> : <div style={{ height: 240, background: '#282a36', borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6272a4' }}>{t('matchesDialog.dictNotInMemory')}</div>}
                </div>
                <div style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ fontSize: '9pt', color: '#f8f8f2', marginBottom: 4 }}>{t('matchesDialog.nccImage')}</div>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'stretch' }}>
                    {matchData.ncc_image ? <LinkedPatternImage src={`data:image/png;base64,${matchData.ncc_image}`} alt={t('matchesDialog.nccImage')} markers={markerCtl.markers} hover={markerCtl.hover} onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick} style={{ ...patStyle, flex: 1, minWidth: 0, background: '#1a1b26' }} /> : <div style={{ height: 240, flex: 1, background: '#282a36', borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6272a4' }}>{t('matchesDialog.bothNeeded')}</div>}
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
                  changes here apply to the Phase Test export too). */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                {markerCtl.markers.length > 0 && (
                  <button onClick={markerCtl.clearMarkers} title={t('hoverTips.matchesClearMarkers')}
                    style={{ fontSize: '8pt', background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, padding: '3px 8px', cursor: 'pointer' }}>
                    {t('matchesDialog.clearMarkers', { count: markerCtl.markers.length })}
                  </button>
                )}
                <button onClick={() => setExportOpen(true)} title={t('matchesDialog.exportTip')}
                  style={{ fontSize: '8pt', fontWeight: 600, background: C.green, color: C.bg, border: 'none', borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}>
                  {t('matchesDialog.export')}
                </button>
              </div>

              {/* Rank browser */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: '9pt', color: '#f8f8f2' }}>{t('matchesDialog.showMatch')}</span>
                <button onClick={() => setRank(r => Math.max(0, r - 1))} disabled={rank <= 0} title={t('hoverTips.matchesRankPrev')} aria-label={t('hoverTips.matchesRankPrev')} style={{ padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}>-</button>
                <span style={{ fontSize: '11pt', color: C.accent, fontWeight: 700, minWidth: 24, textAlign: 'center' }}>{rank + 1}</span>
                <button onClick={() => setRank(r => Math.min((matchData.total_ranks || 1) - 1, r + 1))} disabled={rank >= (matchData.total_ranks || 1) - 1} title={t('hoverTips.matchesRankNext')} aria-label={t('hoverTips.matchesRankNext')} style={{ padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}>+</button>
                <span style={{ fontSize: '9pt', color: '#6272a4' }}>{t('matchesDialog.ofKept', { count: matchData.total_ranks || '?' })}</span>
              </div>

              {/* R-score — large, color-coded */}
              {matchData.r_score != null && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '18pt', fontWeight: 700, color: rColor }}>
                    R = {matchData.r_score.toFixed(4)}
                  </div>
                  <div style={{ fontSize: '10pt', color: '#bd93f9' }}>{rLabel}</div>
                </div>
              )}

              {/* Multi-phase score comparison */}
              {matchData.phase_scores?.length > 1 && (
                <div style={{ marginTop: 8, padding: '6px 12px', background: C.bg, borderRadius: 4, border: `1px solid ${C.border}` }}>
                  <div style={{ fontSize: '8pt', color: '#6272a4', marginBottom: 4 }}>{t('matchesDialog.phaseComparison')}</div>
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

              {/* Phase + Euler */}
              {matchData.euler_angles && (
                <div style={{ fontSize: '8pt', color: '#6272a4', textAlign: 'center', marginTop: 4 }}>
                  {t('matchesDialog.phaseEuler', { phase: matchData.phase_name || t('matchesDialog.dash'), euler: matchData.euler_angles.map(a => a?.toFixed(1)).join(', ') })}
                </div>
              )}

              {/* Orientation provenance: low-symmetry phases (z_rot==2 —
                  orthorhombic mmm and the cubic approximants m-3/23) get the
                  orientation from Hough, because the spherical SO(3) correlation
                  can't form a sharp peak for them. The exact per-class reason is
                  in orientation_source_reason (shown as the tooltip). */}
              {matchData.orientation_source === 'hough' && (
                <div
                  style={{ fontSize: '8pt', color: '#ffb86c', textAlign: 'center', marginTop: 3, cursor: 'help' }}
                  title={matchData.orientation_source_reason || t('matchesDialog.orientationHoughTip')}
                >
                  ⬡ {t('matchesDialog.orientationFromHough')}
                </div>
              )}

              {/* Universal manual pseudo-symmetry flip — shared panel (also
                  mounted in the Phase Maps pattern-match dialog). */}
              <PseudoSymmetryPanel
                selectedPixel={selectedPixel}
                matchData={matchData}
                onApplied={refetchMatch}
              />
            </>)}
          </div>
        </div>
      </div>
      <PatternExportDialog
        open={exportOpen} onClose={() => setExportOpen(false)}
        sources={{
          experimental: matchData?.experimental || null,
          simulated: matchData?.simulated || null,
          ncc: matchData?.ncc_image || null,
          heatmap: heatmapClean || null,
        }}
        rNcc={{ r: matchData?.r_score, ncc: matchData?.ncc_score }}
        stepUm={null}
        mapCols={gridDims.cols || null}
        markers={markerCtl.markers}
      />
    </div>
  );
}

/** Monospace log area */
function LogOutput({ lines, onClear }) {
  const { t } = useTranslation('indexing');
  const ref = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines, autoScroll]);
  const handleCopy = () => {
    if (lines.length === 0) return;
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }).catch(() => {});
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '2px 8px', background: C.bgSecondary,
        border: `1px solid ${C.border}`, borderBottom: 'none',
        borderRadius: '4px 4px 0 0',
      }}>
        <span style={{ fontSize: '8pt', color: C.textSecondary, opacity: 0.7 }}>
          {lines.length > 0 ? t('log.labelCount', { count: lines.length }) : t('log.label')}
        </span>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <button
            onClick={() => setAutoScroll(v => !v)}
            title={autoScroll ? t('log.autoScrollOn') : t('log.autoScrollOff')}
            aria-label={autoScroll ? t('log.pauseAutoScroll') : t('log.resumeAutoScroll')}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: autoScroll ? C.green : C.textSecondary, fontSize: '9pt', padding: '1px 4px',
              opacity: 0.7, borderRadius: 3,
              transition: 'opacity 0.15s, background 0.15s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = `${C.border}55`; }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.7'; e.currentTarget.style.background = 'none'; }}
          >
            {autoScroll ? '\u25BC' : '\u25A0'}
          </button>
          {onClear && (
            <button
              onClick={onClear}
              disabled={lines.length === 0}
              title={t('log.clearLog')}
              aria-label={t('log.clearLog')}
              style={{
                background: 'none', border: 'none', cursor: lines.length ? 'pointer' : 'default',
                color: C.textSecondary, fontSize: '9pt', padding: '1px 4px',
                opacity: lines.length ? 0.7 : 0.3, borderRadius: 3,
                transition: 'opacity 0.15s, background 0.15s',
              }}
              onMouseEnter={(e) => { if (lines.length) { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = `${C.border}55`; } }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = lines.length ? '0.7' : '0.3'; e.currentTarget.style.background = 'none'; }}
            >
              &#10005;
            </button>
          )}
          <button
            onClick={handleCopy}
            disabled={lines.length === 0}
            title={copied ? t('log.copied') : t('log.copyLog')}
            aria-label={t('log.copyLog')}
            style={{
              background: 'none', border: 'none', cursor: lines.length ? 'pointer' : 'default',
              color: copied ? C.green : C.textSecondary, fontSize: '9pt', padding: '1px 4px',
              opacity: lines.length ? 0.7 : 0.3, borderRadius: 3,
              transition: 'color 0.2s, opacity 0.15s, background 0.15s',
            }}
            onMouseEnter={(e) => { if (lines.length) { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = `${C.border}55`; } }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = lines.length ? '0.7' : '0.3'; e.currentTarget.style.background = 'none'; }}
          >
            {copied ? '\u2713' : '\uD83D\uDCCB'}
          </button>
        </div>
      </div>
      <div
        ref={ref}
        className="thin-scrollbar"
        style={{
          background: C.bgSecondary,
          border: `1px solid ${C.border}`,
          borderRadius: '0 0 4px 4px',
          padding: '6px 8px',
          height: 140,
          overflowY: 'auto',
          fontFamily: "'Courier New', monospace",
          fontSize: '9pt',
          color: C.text,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
        }}
      >
        {lines.length === 0
          ? <span style={{ color: C.textSecondary }}>{t('log.ready')}</span>
          : lines.map((l, i) => {
            const lc = l.toLowerCase();
            const color = lc.includes('error') || lc.includes('fail') ? C.red
              : lc.includes('complete') || lc.includes('success') || lc.includes('done') ? C.green
              : lc.includes('warn') ? C.orange
              : undefined;
            return (
              <div key={i} style={{ display: 'flex', gap: 8, color: color || undefined }}>
                <span style={{ color: C.textSecondary, opacity: 0.3, minWidth: 20, textAlign: 'right', userSelect: 'none', flexShrink: 0 }}>{i + 1}</span>
                <span>{l}</span>
              </div>
            );
          })
        }
      </div>
    </div>
  );
}

/** Progress bar (matches PyQt5 QProgressBar) */
function ProgressBar({ value, visible }) {
  if (!visible) return null;
  return (
    <div style={{ marginTop: 4, marginBottom: 4 }}>
      <div style={{ height: 16, background: C.border, borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
        <div style={{
          height: '100%',
          width: `${value}%`,
          background: C.green,
          transition: 'width 0.3s',
          borderRadius: 4,
        }} />
        <span style={{
          position: 'absolute', left: '50%', top: '50%',
          transform: 'translate(-50%,-50%)',
          fontSize: '9pt', fontWeight: 700,
          color: value > 50 ? C.bgSecondary : C.text,
        }}>
          {value}%
        </span>
      </div>
    </div>
  );
}

/** File path row: [dropdown/text] [Load] [Browse] [Preview] */
function FileRow({ value, onChange, onLoad, onBrowse, onPreview, canPreview = false, placeholder = '' }) {
  const { t } = useTranslation(['indexing', 'common']);
  return (
    <Row gap={4}>
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        title={t('hoverTips.fileRowInput')}
        style={{
          flex: 1,
          background: C.bg,
          border: `1px solid ${C.border}`,
          borderRadius: 4,
          color: C.text,
          fontSize: '9pt',
          padding: '3px 6px',
          height: 26,
          outline: 'none',
          minWidth: 0,
        }}
      />
      <button onClick={onLoad} title={t('hoverTips.fileRowLoad')} style={btnSmall(C.border, C.text)}>{t('common:load')}</button>
      <button onClick={onBrowse} title={t('hoverTips.fileRowBrowse')} style={btnSmall(C.border, C.text)}>{t('common:browse')}</button>
      <button
        onClick={onPreview}
        disabled={!canPreview}
        title={t('hoverTips.fileRowPreview')}
        style={btnSmall(C.border, canPreview ? C.cyan : C.textSecondary, !canPreview)}
      >
        {t('actions.preview')}
      </button>
    </Row>
  );
}

function btnSmall(bg, color, disabled = false) {
  return {
    background: bg,
    color: disabled ? C.textSecondary : color,
    border: `1px solid ${C.border}`,
    borderRadius: 4,
    padding: '2px 8px',
    fontSize: '9pt',
    cursor: disabled ? 'not-allowed' : 'pointer',
    height: 26,
    whiteSpace: 'nowrap',
    opacity: disabled ? 0.5 : 1,
    transition: 'all 0.15s',
  };
}

// ---------------------------------------------------------------------------
// Navigation Map (left panel) — shows placeholder since no canvas in React
// ---------------------------------------------------------------------------

function NavigationMapPanel({
  navImage,        // base64 image string or null
  nRows,
  nCols,
  selectionMode,   // 'full' | 'region' | 'mask'
  onSelectionModeChange,
  region,
  onRegionChange,
  savedRegions,
  savedRegionCount,
  onAddRegion,
  onClearAllRegions,
  maskInfo,
  pixelCount,
  edsOverlayEnabled,
  edsOverlayAvailable,
  onEdsOverlayToggle,
  onEdsLayersChange,
  edsOverlayLayers,
  edsFileKey,
  maskOverlayData,
  // Chemistry filter
  chemFilters,
  onAddChemFilter,
  onRemoveChemFilter,
  onChangeChemFilter,
  onApplyChemMask,
  chemPreview,
  chemCombine,
  onChemCombineChange,
  maskMargin,
  onMaskMarginChange,
  filterStats,
  phaseMapRoutingAvailable,
  usePhaseMapRouting,
  onUsePhaseMapRoutingChange,
}) {
  const { t } = useTranslation('indexing');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0, gap: spacing.innerSpacing }}>
      {/* Title */}
      <div style={{ fontSize: '11pt', fontWeight: 'bold', color: C.purple }}>{t('navMap.title')}</div>

      {/* Interactive Canvas */}
      <NavigationCanvas
        navImage={navImage}
        nRows={nRows}
        nCols={nCols}
        region={region}
        savedRegions={savedRegions}
        selectionMode={selectionMode}
        onRegionChange={onRegionChange}
        edsLayers={edsOverlayLayers}
        maskData={maskOverlayData}
      />

      {/* Pixel Selection group */}
      <GroupBox title={t('selection.groupTitle')} style={{ marginBottom: 0 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
          <RadioOption
            label={t('selection.fullImage')}
            checked={selectionMode === 'full'}
            onChange={() => onSelectionModeChange('full')}
            name="sel-mode"
            title={t('hoverTips.selFullImage')}
          />
          <RadioOption
            label={t('selection.region')}
            checked={selectionMode === 'region'}
            onChange={() => onSelectionModeChange('region')}
            name="sel-mode"
            title={t('hoverTips.selRegion')}
          />
          <RadioOption
            label={t('selection.chemistryMask')}
            checked={selectionMode === 'mask'}
            onChange={() => onSelectionModeChange('mask')}
            name="sel-mode"
            title={t('hoverTips.selChemistryMask')}
          />

          {/* Region controls */}
          {selectionMode === 'region' && (
            <div style={{ paddingLeft: 16 }}>
              <Row gap={4} style={{ flexWrap: 'wrap', marginBottom: 4 }}>
                <InlineLabel>{t('selection.rows')}</InlineLabel>
                <NumberInput
                  value={region.rowStart}
                  onChange={e => onRegionChange({ rowStart: Number(e.target.value) })}
                  min={0} max={Math.max(0, nRows - 1)}
                  style={{ width: 60 }}
                  title={t('hoverTips.selRowStart')}
                />
                <InlineLabel>{t('selection.rangeSep')}</InlineLabel>
                <NumberInput
                  value={region.rowEnd}
                  onChange={e => onRegionChange({ rowEnd: Number(e.target.value) })}
                  min={0} max={nRows}
                  style={{ width: 60 }}
                  title={t('hoverTips.selRowEnd')}
                />
                <InlineLabel>{t('selection.cols')}</InlineLabel>
                <NumberInput
                  value={region.colStart}
                  onChange={e => onRegionChange({ colStart: Number(e.target.value) })}
                  min={0} max={Math.max(0, nCols - 1)}
                  style={{ width: 60 }}
                  title={t('hoverTips.selColStart')}
                />
                <InlineLabel>{t('selection.rangeSep')}</InlineLabel>
                <NumberInput
                  value={region.colEnd}
                  onChange={e => onRegionChange({ colEnd: Number(e.target.value) })}
                  min={0} max={nCols}
                  style={{ width: 60 }}
                  title={t('hoverTips.selColEnd')}
                />
                <button
                  onClick={() => { onRegionChange({ rowStart: 0, rowEnd: nRows, colStart: 0, colEnd: nCols }); }}
                  style={btnSmall(C.border, C.red)}
                  title={t('selection.clearTip')}
                >
                  {t('selection.clear')}
                </button>
              </Row>
              {/* Multi-region controls */}
              <Row gap={6}>
                <button
                  onClick={onAddRegion}
                  style={btnSmall('transparent', C.green)}
                  title={t('selection.addRegionTip')}
                >
                  {t('selection.addRegion')}
                </button>
                <button
                  onClick={onClearAllRegions}
                  style={btnSmall('transparent', C.red)}
                  title={t('selection.clearAllTip')}
                >
                  {t('selection.clearAll')}
                </button>
                <span style={{ color: C.textSecondary, fontSize: '9pt' }}>
                  {t('selection.savedRegions', { count: savedRegionCount })}
                </span>
              </Row>
            </div>
          )}

          {/* Chemistry mask info */}
          {selectionMode === 'mask' && (
            <div style={{ paddingLeft: 16, animation: 'fadeSlideIn 0.2s ease-out' }}>
              <div style={{ color: maskInfo ? C.green : C.orange, fontSize: '9pt', marginBottom: 4, transition: 'color 0.2s' }}>
                {maskInfo || t('mask.noChemistryMask')}
              </div>

              {/* Per-phase routing toggle (only shown after EDS phase-map handoff) */}
              {phaseMapRoutingAvailable && (
                <label
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 6,
                    cursor: 'pointer', userSelect: 'none', marginBottom: 6,
                    padding: '4px 6px', borderRadius: 4,
                    background: usePhaseMapRouting
                      ? `${C.cyan}14`
                      : `${C.bgSecondary}`,
                    border: `1px solid ${usePhaseMapRouting ? C.cyan : C.border}`,
                    transition: 'all 0.15s',
                  }}
                  title={t('mask.perPhaseRoutingTip')}
                >
                  <input
                    type="checkbox"
                    checked={usePhaseMapRouting}
                    onChange={(e) => onUsePhaseMapRoutingChange?.(e.target.checked)}
                    style={{ accentColor: C.cyan, marginTop: 2 }}
                  />
                  <span style={{ fontSize: '9pt', color: usePhaseMapRouting ? C.cyan : C.text, fontWeight: 600, lineHeight: 1.3 }}>
                    {t('mask.perPhaseRouting')}
                    <span style={{ display: 'block', fontSize: '8pt', fontWeight: 400, color: C.textSecondary, marginTop: 2 }}>
                      {t('mask.perPhaseRoutingDesc')}
                    </span>
                  </span>
                </label>
              )}

              {/* Chemistry filter builder */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {chemFilters.map((f, idx) => (
                  <ChemFilterRow
                    key={idx}
                    filter={f}
                    onChange={val => onChangeChemFilter(idx, val)}
                    onRemove={() => onRemoveChemFilter(idx)}
                  />
                ))}
                {/* AND/OR toggle */}
                {chemFilters.length > 1 && (
                  <Row gap={4} style={{ justifyContent: 'center' }}>
                    <button
                      onClick={() => onChemCombineChange(chemCombine === 'and' ? 'or' : 'and')}
                      title={t('hoverTips.chemCombineToggle')}
                      style={{
                        background: chemCombine === 'and' ? C.purple : C.orange,
                        color: '#fff', border: 'none', borderRadius: 3,
                        padding: '2px 12px', fontSize: '9pt', fontWeight: 'bold', cursor: 'pointer',
                      }}
                    >
                      {chemCombine === 'and' ? t('mask.andAll') : t('mask.orAny')}
                    </button>
                  </Row>
                )}
                <Row gap={4}>
                  <button
                    onClick={onAddChemFilter}
                    title={t('hoverTips.chemAddFilter')}
                    style={{ ...btnSmall('transparent', C.green), border: `1px solid ${C.green}`, borderRadius: 3 }}
                  >
                    {t('mask.addFilter')}
                  </button>
                  <button
                    onClick={onApplyChemMask}
                    title={t('hoverTips.chemApplyMask')}
                    style={{ ...btnSmall('transparent', C.purple), border: `1px solid ${C.purple}`, fontWeight: 'bold' }}
                  >
                    {t('mask.applyMask')}
                  </button>
                </Row>
                {/* Mask Margin slider */}
                <Row gap={6} style={{ alignItems: 'center' }}>
                  <span style={{ color: C.textSecondary, fontSize: '9pt', whiteSpace: 'nowrap' }}>{t('mask.margin')}</span>
                  <input
                    type="range" min={-5} max={5} value={maskMargin}
                    onChange={e => onMaskMarginChange(Number(e.target.value))}
                    title={t('hoverTips.maskMargin')}
                    style={{ width: 80 }}
                  />
                  <span style={{ color: C.text, fontSize: '9pt', width: 36, textAlign: 'center' }}>
                    {maskMargin > 0 ? `+${maskMargin}` : maskMargin} {t('spherical.px')}
                  </span>
                </Row>
                {chemPreview && (
                  <div style={{ color: C.cyan, fontSize: '9pt' }}>{chemPreview}</div>
                )}
                {/* Per-filter breakdown */}
                {filterStats && filterStats.length > 0 && (
                  <div style={{
                    background: 'rgba(0,0,0,0.3)',
                    borderRadius: 4,
                    padding: '4px 6px',
                    marginTop: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                  }}>
                    <span style={{ color: C.textSecondary, fontSize: '8pt', fontWeight: 'bold' }}>
                      {t('mask.filterBreakdown')}
                    </span>
                    {filterStats.map((fs, i) => {
                      const unitLabel = { counts: 'Cts', wt_pct: 'Wt%', at_pct: 'At%' }[fs.unit] || fs.unit;
                      return (
                        <div key={i} style={{ display: 'flex', gap: 4, fontSize: '8pt', alignItems: 'center' }}>
                          <span style={{ color: C.text, minWidth: 28 }}>{fs.element}</span>
                          <span style={{ color: C.textSecondary }}>{fs.operator}</span>
                          <span style={{ color: C.textSecondary }}>({unitLabel})</span>
                          <div style={{
                            flex: 1, height: 6, background: 'rgba(255,255,255,0.1)',
                            borderRadius: 3, overflow: 'hidden',
                          }}>
                            <div style={{
                              width: `${fs.pct_pass}%`, height: '100%',
                              background: fs.pct_pass > 50 ? C.green : fs.pct_pass > 10 ? C.orange : C.red,
                              borderRadius: 3,
                              transition: 'width 0.3s',
                            }} />
                          </div>
                          <span style={{ color: C.cyan, minWidth: 48, textAlign: 'right' }}>
                            {fs.pct_pass}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
                {/* Mask visual legend */}
                {maskInfo && (
                  <div style={{
                    display: 'flex', gap: 10, alignItems: 'center', marginTop: 2,
                    padding: '3px 0', borderTop: `1px solid ${C.border}`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                      <span style={{
                        width: 10, height: 10, borderRadius: 2,
                        background: 'rgba(80,250,123,0.3)',
                        border: '1px solid rgba(139,233,253,0.8)',
                        display: 'inline-block',
                      }} />
                      <span style={{ color: C.green, fontSize: '7pt' }}>{t('mask.legendSelected')}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                      <span style={{
                        width: 10, height: 10, borderRadius: 2,
                        background: 'repeating-linear-gradient(45deg, rgba(200,0,0,0.5), rgba(200,0,0,0.5) 1px, rgba(0,0,0,0.4) 1px, rgba(0,0,0,0.4) 3px)',
                        display: 'inline-block',
                      }} />
                      <span style={{ color: C.red, fontSize: '7pt' }}>{t('mask.legendExcluded')}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </GroupBox>

      {/* Pixel count */}
      {pixelCount && (
        <div style={{ textAlign: 'center', color: C.cyan, fontSize: '10pt', fontWeight: 'bold' }}>
          {pixelCount}
        </div>
      )}

      {/* EDS Element Overlay */}
      <EdsOverlayPanel
        enabled={edsOverlayEnabled}
        available={edsOverlayAvailable}
        onToggle={onEdsOverlayToggle}
        onLayersChange={onEdsLayersChange}
        fileKey={edsFileKey}
      />
    </div>
  );
}

/** Chemistry filter row: [element] [unit] [operator] [min] [max?] [X] */
function ChemFilterRow({ filter, onChange, onRemove }) {
  const { t } = useTranslation('indexing');
  const unitMax = (filter.unit || 'at_pct') === 'counts' ? 100000 : 100;
  const unitStep = (filter.unit || 'at_pct') === 'counts' ? 1 : 0.1;
  const unitLabel = { counts: 'Cts', wt_pct: 'Wt%', at_pct: 'At%' }[filter.unit || 'at_pct'];
  return (
    <Row gap={4} style={{ flexWrap: 'nowrap' }}>
      <input
        type="text"
        value={filter.element}
        onChange={e => onChange({ ...filter, element: e.target.value })}
        placeholder={t('chemFilter.elementPlaceholder')}
        style={{ width: 44, ...inputSmall() }}
        title={t('chemFilter.elementTip')}
      />
      <select
        value={filter.unit || 'at_pct'}
        onChange={e => onChange({ ...filter, unit: e.target.value })}
        style={{ ...inputSmall(), width: 62 }}
        title={t('chemFilter.unitTip')}
      >
        <option value="counts">{t('chemFilter.unitCounts')}</option>
        <option value="wt_pct">{t('chemFilter.unitWt')}</option>
        <option value="at_pct">{t('chemFilter.unitAt')}</option>
      </select>
      <select
        value={filter.operator}
        onChange={e => onChange({ ...filter, operator: e.target.value })}
        style={{ ...inputSmall(), width: 72 }}
        title={t('hoverTips.chemOperator')}
      >
        <option value=">">&gt;</option>
        <option value="<">&lt;</option>
        <option value="between">{t('chemFilter.opBetween')}</option>
      </select>
      <input
        type="number"
        value={filter.min}
        onChange={e => onChange({ ...filter, min: Number(e.target.value) })}
        min={0} max={unitMax} step={unitStep}
        style={{ width: 72, ...inputSmall() }}
        title={t('chemFilter.minTip', { unit: unitLabel })}
      />
      {filter.operator === 'between' && (
        <>
          <span style={{ color: C.textSecondary, fontSize: '9pt' }}>-</span>
          <input
            type="number"
            value={filter.max}
            onChange={e => onChange({ ...filter, max: Number(e.target.value) })}
            min={0} max={unitMax} step={unitStep}
            style={{ width: 72, ...inputSmall() }}
            title={t('chemFilter.maxTip', { unit: unitLabel })}
          />
        </>
      )}
      <button
        aria-label={t('chemFilter.removeFilter')}
        onClick={onRemove}
        style={{ background: 'none', border: 'none', color: C.red, cursor: 'pointer', fontWeight: 'bold', fontSize: '10pt', padding: '0 2px' }}
      >
        X
      </button>
    </Row>
  );
}

function inputSmall() {
  return {
    background: C.bg,
    border: `1px solid ${C.border}`,
    borderRadius: 4,
    color: C.text,
    fontSize: '9pt',
    padding: '2px 4px',
    height: 22,
    outline: 'none',
  };
}


// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function IndexingPage({ isActive }) {
  const { t } = useTranslation(['indexing', 'common']);
  const { setIndexingResult } = useResultStore();
  const filePath = useDataStore(s => s.filePath);

  // Translated method labels/tooltips. The `value`s are the canonical method
  // identifiers the indexing logic switches on; only display strings change.
  const methodOptions = useMemo(() => [
    { value: 'hough',      label: t('methods.hough'),      tip: t('methods.houghTip') },
    { value: 'dictionary', label: t('methods.dictionary'), tip: t('methods.dictionaryTip') },
    { value: 'spherical',  label: t('methods.spherical'),  tip: t('methods.sphericalTip') },
    { value: 'embedding',  label: t('methods.embedding'),  tip: t('methods.embeddingTip') },
  ], [t]);

  // Localised circular-mask dropdown options. Stored value stays the English
  // sentinel ("Off (-1)"/"Inscribed circle (0)"/"Custom radius") that the
  // buildParams switch reads, so only the label is translated.
  const circmaskOptions = useMemo(() => [
    { value: 'Off (-1)',             label: t('spherical.circmaskOff') },
    { value: 'Inscribed circle (0)', label: t('spherical.circmaskInscribed') },
    { value: 'Custom radius',        label: t('spherical.circmaskCustom') },
  ], [t]);

  // --- Dataset ---
  const [datasets, setDatasets]         = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [dataStatus, setDataStatus]     = useState(t('dataset.statusNoData'));
  const [dataLoaded, setDataLoaded]     = useState(false);

  // --- Loaded files (multi-file workflow) ---
  // The Indexing dropdown shows a HIERARCHICAL list: each loaded file is a
  // top-level entry; derivatives (deepcopy from Frame Average, BG
  // subtraction, etc.) appear as indented children of the active file.
  // Non-active files have no visible derivatives because the backend only
  // keeps the active file's signals in _raw_signals.
  // Sourced from the shared store so this dropdown stays in sync with the
  // EBSD viewer's "Loaded Files" panel and the header file switcher.
  const loadedFiles = useLoadedFilesStore((s) => s.files);  // [{path, name, active}]
  const refreshLoadedFiles = useLoadedFilesStore((s) => s.refresh);
  const [datasetSwitching, setDatasetSwitching] = useState(false);

  // --- PC summary ---
  const [pcValues, setPcValues]         = useState(t('pcPhase.notLoaded'));
  const [phaseInfo, setPhaseInfo]       = useState(t('pcPhase.notLoaded'));

  // --- Navigation map ---
  const [navImage, setNavImage]         = useState(null);
  const [nRows, setNRows]               = useState(0);
  const [nCols, setNCols]               = useState(0);

  // --- Pixel selection ---
  const [selMode, setSelMode]           = useState('full');   // 'full' | 'region' | 'mask'
  const [region, setRegion]             = useState({ rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 });
  const [savedRegions, setSavedRegions] = useState([]);
  const [maskInfo, setMaskInfo]         = useState('');
  const [pixelCount, setPixelCount]     = useState('');

  // --- EDS overlay ---
  const [edsOverlayEnabled, setEdsOverlayEnabled]   = useState(false);
  const [edsOverlayAvailable, setEdsOverlayAvailable] = useState(false);
  const [edsOverlayLayers, setEdsOverlayLayers]     = useState([]);
  const [maskOverlayData, setMaskOverlayData]       = useState(null);

  // --- Chemistry filters ---
  const [chemFilters, setChemFilters]   = useState([]);
  const [chemPreview, setChemPreview]   = useState('');
  const [chemCombine, setChemCombine]   = useState('and');
  const [maskMargin, setMaskMargin]     = useState(0);
  const [filterStats, setFilterStats]   = useState(null); // per-filter pixel counts from backend

  // --- Per-phase routing (EDS phase-map handoff) ---
  // When the user comes in via "→ Send to Indexing", the M5 handoff
  // useEffect installs the union mask AND switches this on by default
  // so the run uses per-phase pixel routing instead of letting every
  // phase compete for every classified pixel. Toggle is visible only
  // after the handoff has run successfully — outside that flow the
  // backend store is empty and the flag would just produce a 400.
  const [phaseMapRoutingAvailable, setPhaseMapRoutingAvailable] = useState(false);
  const [usePhaseMapRouting, setUsePhaseMapRouting] = useState(false);

  // --- Method ---
  const [method, setMethod]             = useState('hough');
  const [pendingPhase, setPendingPhase] = useState(null);  // phase to add after a method switch from the Phase Test

  // --- Phase selection (replaces old comparison mode) ---
  const [phases, setPhases]             = useState([]); // selected phase file objects from discovery
  const [discoveredGroups, setDiscoveredGroups] = useState([]);
  const [phaseDropdownOpen, setPhaseDropdownOpen] = useState(false);
  const [phasePanelOpen, setPhasePanelOpen]       = useState(false);
  const [selectedDictPaths, setSelectedDictPaths] = useState({}); // { masterPath: dictPath }

  // --- EDS chemistry prior (per-phase influence at indexing) ---
  const [edsStrengths, setEdsStrengths]                 = useState({}); // { [phase.path]: 0..100 }
  const [edsExpectedOverrides, setEdsExpectedOverrides] = useState({}); // { [phase.path]: {El:atPct} } (editor deferred)

  // --- Required files ---
  const [discoveredFiles, setDiscoveredFiles] = useState([]);
  const [fileInput, setFileInput]       = useState('');    // typed / selected path
  const [phaseFiles, setPhaseFiles]     = useState([]);    // list of added phase files
  const [fileStatus, setFileStatus]     = useState(t('phases.noFileLoaded'));
  const [fileStatusColor, setFileStatusColor] = useState(C.red);
  const [canPreview, setCanPreview]     = useState(false);
  const [dictSection, setDictSection]   = useState(false);
  const [dictFiles, setDictFiles]       = useState([]);
  const [selectedDictFile, setSelectedDictFile] = useState('');

  // --- Hough params ---
  const [bands, setBands]               = useState(12);
  const [tSigma, setTSigma]             = useState(2.0);
  const [rSigma, setRSigma]             = useState(2.0);

  // --- Dictionary params ---
  const [metric, setMetric]             = useState('ncc');
  const [keepN, setKeepN]               = useState(20);
  const [resolution, setResolution]     = useState(5.0);
  const [energy, setEnergy]             = useState(20);
  const [dictInfo, setDictInfo]         = useState('');
  // Compute toggle for dictionary indexing: 'auto' picks GPU when available
  // and falls back to CPU; 'gpu' forces CUDA (raises GpuDictError if absent);
  // 'cpu' forces the kikuchipy CPU path. Status comes from /api/system/gpu.
  const [computeMode, setComputeMode]   = useState('auto');
  const [gpuStatus, setGpuStatus]       = useState(null);

  // --- Spherical params ---
  const [bandwidth, setBandwidth]       = useState('88');
  const [nregions, setNregions]         = useState(10);
  const [refine, setRefine]             = useState(true);
  const [gausbckg, setGausbckg]         = useState(false);
  const [circmask, setCircmask]         = useState('Off (-1)');
  const [circmaskRadius, setCircmaskRadius] = useState(30);
  // Backend selector: "spherical_gpu" runs the in-process PyTorch pipeline
  // (~11x faster than EMSphInx CPU on identical data, GPU required).
  // "emsphinx" is the original WSL EMSphInx path (CPU, slower, more battle-
  // tested in older builds).
  const [sphericalBackend, setSphericalBackend] = useState('spherical_gpu');

  // --- Embedding params ---
  const [encoderPath, setEncoderPath]   = useState('');
  const [faissPath, setFaissPath]       = useState('');
  const [embK, setEmbK]                 = useState(5);

  // --- Task / progress ---
  const [taskId, setTaskId]             = useState(null);
  const [running, setRunning]           = useState(false);
  const [startTime, setStartTime]      = useState(null);
  const [elapsed, setElapsed]          = useState('');
  const [progress, setProgress]         = useState(0);
  const [progressVisible, setProgressVisible] = useState(false);
  const [quality, setQuality]           = useState('');
  const [qualityVisible, setQualityVisible]   = useState(false);
  const [showResultModal, setShowResultModal] = useState(false);
  const [perPhaseStats, setPerPhaseStats]     = useState([]);
  const [logLines, setLogLines]         = useState([]);
  // Runtime banner (CUDA/CPU, free VRAM) — fed by status poll, shown
  // above the Start button so the user can SEE whether the run will be
  // fast and whether a previous run still holds the GPU.
  const [runtimeInfo, setRuntimeInfo]   = useState(null);
  const [releasingGpu, setReleasingGpu] = useState(false);
  const pollRef                         = useRef(null);
  const pollTerminatedRef               = useRef(false);
  const chemMaskRef                     = useRef(null);
  const datasetsRef                     = useRef([]);  // full dataset info from backend

  // --- Post-run actions ---
  const [canExport, setCanExport]       = useState(false);
  const [canViewMatches, setCanViewMatches] = useState(false);
  const [canRefine, setCanRefine]       = useState(false);
  const [canSendAnalysis, setCanSendAnalysis] = useState(false);
  const [sendToPhaseMap, setSendToPhaseMap] = useState(true);

  // --- Preprocessing-preview pixel seed (row/col for EMSphinx preview) ---
  const [qtRow, setQtRow]               = useState(0);
  const [qtCol, setQtCol]               = useState(0);

  // --- Batch indexing ---
  const [showBatchDialog, setShowBatchDialog] = useState(false);
  const [showRefineDialog, setShowRefineDialog] = useState(false);
  const [showMatchesDialog, setShowMatchesDialog] = useState(false);
  const [showPhaseTestDialog, setShowPhaseTestDialog] = useState(false);

  // --- GPU dictionary generation (state lives here so the run survives
  // closing the modal mid-flight) ---
  const [showGenDictDialog, setShowGenDictDialog] = useState(false);
  const [genDictTaskId, setGenDictTaskId] = useState(null);
  const [genDictProgress, setGenDictProgress] = useState(null);

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const log = useCallback((msg) => {
    setLogLines(prev => [...prev.slice(-200), msg]);
  }, []);

  const updateRegion = useCallback((partial) => {
    setRegion(prev => ({ ...prev, ...partial }));
  }, []);

  // ---------------------------------------------------------------------------
  // Lifecycle: load methods + discover files
  // ---------------------------------------------------------------------------

  const fetchDataState = useCallback(() => {
    refreshLoadedFiles();
    ebsdApi.datasets().then(r => {
      const ds = r.data?.datasets || [];
      const names = ds.map(d => d.name);
      setDatasets(names);
      datasetsRef.current = ds;
      const active = r.data?.active || (names.length > 0 ? names[0] : '');
      if (active) {
        setSelectedDataset(active);
        setDataLoaded(true);
        const info = ds.find(d => d.name === active);
        if (info) {
          const navShape = info.navigation_shape || [];
          const rows = navShape.length >= 2 ? navShape[1] : navShape[0] || 0;
          const cols = navShape.length >= 2 ? navShape[0] : 1;
          setNRows(rows);
          setNCols(cols);
          setDataStatus(t('dataset.summary', { name: active, rows, cols, count: info.n_patterns }));
          setEdsOverlayAvailable(!!info.has_eds);
          if (info.signal_shape && info.signal_shape.length >= 2) {
            const minDim = Math.min(info.signal_shape[0], info.signal_shape[1]);
            setCircmaskRadius(Math.floor(minDim / 2));
          }
        }
      }
    }).catch(() => {});

    // Use 'bc' (Band Contrast) — the SAME default the EBSD Viewer shows — so
    // the Indexing nav map matches the Viewer overview instead of showing a
    // visually different mean-intensity map of the same dataset. Falls back to
    // a live BC-style compute when no stored quality map exists.
    ebsdApi.overview('bc').then(r => {
      if (r.data?.image) setNavImage(r.data.image);
    }).catch(() => {});

    // Try PC refinement controller first, fall back to EBSD viewer detector
    pcApi.detectorInfo().then(r => {
      const d = r.data;
      if (d?.has_detector && d?.pc) {
        setPcValues(t('pcPhase.pcLine', { x: d.pc[0]?.toFixed(3), y: d.pc[1]?.toFixed(3), z: d.pc[2]?.toFixed(3) }));
      } else {
        // Fall back to EBSD viewer detector
        ebsdApi.getDetector().then(r2 => {
          const d2 = r2.data;
          if (d2?.has_detector && d2?.pc) {
            setPcValues(t('pcPhase.pcLine', { x: d2.pc[0]?.toFixed(3), y: d2.pc[1]?.toFixed(3), z: d2.pc[2]?.toFixed(3) }));
          }
        }).catch(() => {});
      }
    }).catch(() => {});
  }, [refreshLoadedFiles, t]);

  // Switch active dataset OR loaded file on backend when user picks from
  // dropdown. The dropdown value uses prefixes to distinguish:
  //   "file:<path>"     -> /api/ebsd/switch-file  (re-load the file)
  //   "dataset:<name>"  -> /api/ebsd/select-dataset  (within-file switch)
  // Bare names (no prefix) are treated as legacy dataset names.
  const handleDatasetSwitch = useCallback(async (value) => {
    if (!value || datasetSwitching) return;
    let kind = 'dataset';
    let payload = value;
    if (value.startsWith('file:')) {
      kind = 'file';
      payload = value.slice('file:'.length);
    } else if (value.startsWith('dataset:')) {
      kind = 'dataset';
      payload = value.slice('dataset:'.length);
    }

    if (kind === 'file') {
      // Loaded-file switch — full re-load from disk
      setDatasetSwitching(true);
      const fname = payload.split(/[\\/]/).pop();
      log(t('dataset.switching', { name: fname }));
      try {
        await ebsdApi.switchFile(payload);
        // Update the GLOBAL store first so EVERY page (EBSD viewer, Crystal
        // Hint, status bar, EDS) reconciles to the new active file. Without
        // this, switching files from the Indexing dropdown left the rest of
        // the app pointing at the previous file — wrong-file images and a wall
        // of "Dataset not found" / "out of bounds" errors when those pages
        // fired calls with the old file's dataset names and grid.
        await useDataStore.getState().syncFromBackend();
        // Refresh the per-file dataset list and file switcher. The previous
        // file's processed datasets are now stashed (not destroyed) backend-
        // side, so switching back restores them.
        fetchDataState();
        ebsdApi.overview('bc').then(r => {
          if (r.data?.image) setNavImage(r.data.image);
        }).catch(() => {});
        log(t('dataset.switchedFile', { name: fname }));
      } catch (err) {
        log(t('dataset.switchFileError', { error: err.response?.data?.detail || err.message }));
      } finally {
        setDatasetSwitching(false);
      }
      return;
    }

    // Within-file dataset switch
    setSelectedDataset(payload);
    try {
      await ebsdApi.selectDataset(payload);
      const info = datasetsRef.current.find(d => d.name === payload);
      if (info) {
        const navShape = info.navigation_shape || [];
        const rows = navShape.length >= 2 ? navShape[1] : navShape[0] || 0;
        const cols = navShape.length >= 2 ? navShape[0] : 1;
        setNRows(rows);
        setNCols(cols);
        setDataStatus(t('dataset.summary', { name: payload, rows, cols, count: info.n_patterns }));
        setEdsOverlayAvailable(!!info.has_eds);
      }
      ebsdApi.overview('bc').then(r => {
        if (r.data?.image) setNavImage(r.data.image);
      }).catch(() => {});
      log(t('dataset.switchedDataset', { name: payload }));
    } catch (err) {
      log(t('dataset.switchDatasetError', { error: err.response?.data?.detail || err.message }));
    }
  }, [log, datasetSwitching, fetchDataState, t]);

  // GPU status probe — runs once on mount. Used by the dictionary Compute
  // radio group to disable the GPU option and show a status line. We never
  // raise; if the endpoint fails we fall back to "no CUDA" so the UI stays
  // usable on CPU-only machines.
  useEffect(() => {
    getGpuStatus()
      .then(setGpuStatus)
      .catch((e) => {
        console.warn('getGpuStatus failed:', e);
        setGpuStatus({ available: false, name: '', vram_total_gb: 0, vram_free_gb: 0 });
      });
  }, []);

  // Fetch data when page becomes active OR when a new file is loaded
  // while we're already on this page. Without the filePath dep the user
  // had to tab-switch to see the new file's overview/datasets.
  useEffect(() => {
    if (!isActive) return;
    indexApi.methods().catch(() => {});
    // Must follow the currently selected method — hardcoding 'hough' here
    // caused Spherical/Dictionary users to see the Hough CIF library on
    // return-to-page, which then seeded phaseFiles with CIF paths and made
    // Spherical runs fail with "All phases failed: unknown".
    discoverFilesForMethod(method);
    fetchDataState();
  }, [isActive, filePath]); // eslint-disable-line react-hooks/exhaustive-deps

  // File switch: drop EDS overlay layers and chemistry mask painted on the
  // navigation canvas. The EdsOverlayPanel resets its own toggled-element
  // state via the fileKey prop, but the parent-side mirror must clear too,
  // otherwise the previous file's PNGs stay on the canvas for one frame.
  // Also drop the phase-map routing toggle — the new file's backend store
  // is empty until the user runs auto-classify there.
  useEffect(() => {
    setEdsOverlayLayers([]);
    setMaskOverlayData(null);
    setPhaseMapRoutingAvailable(false);
    setUsePhaseMapRouting(false);
  }, [filePath]);

  // Auto-select chemistry mask mode when navigating from "Send Filter to Indexing"
  useEffect(() => {
    if (!isActive) return;
    const pending = useDataStore.getState().pendingChemMask;
    if (pending) {
      setSelMode('mask');
      log(t('messages.chemMaskActivated'));
      useDataStore.getState().setPendingChemMask(false);
    }
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  // M5: phase-map handoff from EDS page. When the user clicks
  // "→ Send to Indexing" on the phase-map panel, useDataStore flips
  // pendingPhaseMapIndexing=true and navigates here. We then:
  //   1) switch to Hough (the only method that consumes CIFs directly)
  //   2) auto-select every CIF that appears in the phase map and is
  //      visible in the discovered phase files
  //   3) install the combined classified-pixel mask as the indexing
  //      mask so unclassified pixels are skipped entirely
  //   4) consume the flag so a back-and-forth navigation doesn't keep
  //      re-applying the prefill
  // The dependency on `discoveredFiles.length` is the wait-gate — we
  // can't match CIF filenames against an empty discovery list.
  useEffect(() => {
    if (!isActive) return;
    if (!useDataStore.getState().pendingPhaseMapIndexing) return;
    if (discoveredFiles.length === 0) return;

    let cancelled = false;
    (async () => {
      try {
        const res = await edsApi.phaseMapIndexingConfig();
        if (cancelled) return;
        const cfg = res.data;
        if (!cfg?.loaded) {
          log(t('messages.handoffSkipped'));
          useDataStore.getState().setPendingPhaseMapIndexing(false);
          return;
        }

        // Hough is the only method whose discovery list contains CIFs;
        // other methods would silently drop the auto-selection in
        // handleTogglePath because the extension wouldn't match.
        if (method !== 'hough') setMethod('hough');

        const cifFiles = discoveredFiles.filter(
          f => f.file_type === 'phase' || (f.filename || '').toLowerCase().endsWith('.cif')
        );
        const matchedFiles = cifFiles.filter(
          f => cfg.cif_filenames.includes(f.filename)
        );

        if (matchedFiles.length > 0) {
          setPhaseFiles(matchedFiles.map(f => f.path));
          setPhases(matchedFiles);
          setPhaseInfo(t('phases.phasesFromEdsPrefix', { list: matchedFiles.map(p => p.formula || p.filename).join(', ') }));
          setFileStatus(t('phases.preselectedFromEds', { count: matchedFiles.length }));
          setFileStatusColor(C.green);
        }
        const missing = cfg.cif_filenames.filter(
          name => !cifFiles.some(f => f.filename === name)
        );
        if (missing.length > 0) {
          log(t('messages.handoffMissing', { count: missing.length, list: missing.join(', ') }));
        }

        // Install the union mask + switch selection mode.
        chemMaskRef.current = cfg.combined_mask;
        setMaskOverlayData({
          mask2d: new Uint8Array(cfg.combined_mask),
          shape: [cfg.n_rows, cfg.n_cols],
        });
        setMaskInfo(
          t('mask.fromEdsMap', { classified: cfg.n_classified, total: cfg.n_total, pct: cfg.percentage, matched: matchedFiles.length, phases: cfg.cif_filenames.length })
        );
        setPixelCount(t('selection.pixelCount', { n: cfg.n_classified, total: cfg.n_total, pct: cfg.percentage }));
        setSelMode('mask');
        setPhaseMapRoutingAvailable(true);
        setUsePhaseMapRouting(true);
        log(t('messages.handoffApplied', { phases: matchedFiles.length, pixels: cfg.n_classified }));
      } catch (err) {
        log(t('messages.handoffError', { error: err.response?.data?.detail || err.message }));
      } finally {
        useDataStore.getState().setPendingPhaseMapIndexing(false);
      }
    })();

    return () => { cancelled = true; };
  }, [isActive, discoveredFiles, method]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup poll interval on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  // Poll GPU dictionary-generation task while it's active.
  useEffect(() => {
    if (!genDictTaskId) return;
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      try {
        const r = await dictionaryGpuApi.progress(genDictTaskId);
        if (cancelled) return;
        setGenDictProgress(r.data);
        const status = r.data?.status;
        if (status === 'done' || status === 'error') {
          if (timer) { clearInterval(timer); timer = null; }
        }
      } catch (e) {
        if (cancelled) return;
        setGenDictProgress((prev) => ({ ...(prev || {}), status: 'error', error: t('genDictDialog.progressPollFailed', { error: e?.message || e }) }));
        if (timer) { clearInterval(timer); timer = null; }
      }
    };
    poll();
    timer = setInterval(poll, 500);
    return () => { cancelled = true; if (timer) clearInterval(timer); };
  }, [genDictTaskId]);

  useEffect(() => {
    // Clear files from previous method — can't use .sht for Dictionary etc.
    setPhaseFiles([]);
    setPhases([]);
    // Drop any per-phase EDS strengths/overrides — their phase.path keys belong
    // to the just-cleared list and must not linger into the new method's run.
    setEdsStrengths({});
    setEdsExpectedOverrides({});
    setPhaseDropdownOpen(false);
    setFileInput('');
    setFileStatus(t('phases.noFileLoaded'));
    setFileStatusColor(C.red);
    setPhaseInfo(t('pcPhase.notLoaded'));
    discoverFilesForMethod(method);
  }, [method]);

  // Apply a phase chosen via the Phase Test "Use for indexing" picker. Runs AFTER
  // the method-change effect above cleared the (method-incompatible) list, so the
  // newly-chosen phase survives the switch.
  useEffect(() => {
    if (!pendingPhase) return;
    setPhases([pendingPhase]);
    setPhaseFiles([pendingPhase.path]);
    setFileStatus(t('phases.selectedCount', { count: 1 }));
    setFileStatusColor(C.green);
    setPhaseInfo(t('phases.phasesPrefix', { list: pendingPhase.formula || pendingPhase.filename }));
    setPendingPhase(null);
  }, [pendingPhase, method]);

  // "Use for indexing" from the Phase Test: add this phase for the chosen method,
  // switching the indexing method if needed (the method-change effect clears the
  // old method-incompatible list; same-method just appends).
  //
  // Deliberately does NOT close the dialog — the user builds up a multi-phase
  // list by clicking through pixels, identifying the phase at each and adding it,
  // so the picker stays open until they close it with ×. Returns a status
  // ('switched' | 'added' | 'exists') the dialog uses for its confirmation line.
  const handleUsePhaseFromTest = (phase, targetMethod) => {
    const path = { hough: phase.cif_path, spherical: phase.sht_path, dictionary: phase.master_h5_path }[targetMethod];
    if (!path) return null;
    const file = {
      path,
      filename: String(path).split(/[\\/]/).pop(),
      formula: phase.display_formula || phase.formula || phase.phase_key,
    };
    if (targetMethod !== method) {
      setPendingPhase(file);        // applied by the effect above, after the switch
      setMethod(targetMethod);
      return 'switched';
    }
    if (!phaseFiles.includes(path)) {
      setPhases(p => [...p, file]);
      setPhaseFiles(p => [...p, path]);
      setFileStatus(t('phases.selectedCount', { count: phaseFiles.length + 1 }));
      setFileStatusColor(C.green);
      setPhaseInfo(t('phases.phasesPrefix', { list: [...phases.map(p => p.formula || p.filename), file.formula].join(', ') }));
      return 'added';
    }
    return 'exists';
  };

  async function discoverFilesForMethod(m) {
    try {
      // Parse current PC from display string like "PC: (0.503, 0.327, 0.846)"
      let pc = null;
      if (pcValues) {
        const pcMatch = pcValues.match(/PC:\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);
        if (pcMatch) pc = [parseFloat(pcMatch[1]), parseFloat(pcMatch[2]), parseFloat(pcMatch[3])];
      }
      const r = await indexApi.discoverFiles(m, '', pc);
      const files = r.data?.files || [];
      const groups = r.data?.groups || [];
      setDiscoveredFiles(files);
      setDiscoveredGroups(groups);
      if (files.length > 0 && phaseFiles.length === 0) {
        setFileInput(files[0].path || files[0].name || '');
      }
    } catch {
      setDiscoveredFiles([]);
      setDiscoveredGroups([]);
    }
  }

  function handleTogglePath(file) {
    if (!file) {
      // Manual file fallback — trigger native file picker
      handleBrowsePhaseFile();
      return;
    }
    const path = file.path;
    const isSelected = phaseFiles.includes(path);

    // Guard: reject files whose extension doesn't match the current method.
    // Previously the dropdown could show mixed content after a method switch,
    // and adding e.g. a .cif while Spherical was active silently produced a
    // request with sht_paths=[] and cif_paths>=2 — that triggered the
    // multi-phase path and then the "All phases failed: unknown" error.
    if (!isSelected) {
      const ext = (path || '').toLowerCase().split('.').pop();
      const methodExt = { hough: 'cif', dictionary: 'h5', spherical: 'sht' }[method];
      const altExt = method === 'dictionary' ? 'hdf5' : null;
      if (methodExt && ext !== methodExt && ext !== altExt) {
        log(t('phases.cannotAdd', { name: file.filename || path, method, ext: methodExt, got: ext }));
        return;
      }
    }

    if (isSelected) {
      // Remove phase
      const newPaths = phaseFiles.filter(p => p !== path);
      const newPhases = phases.filter(p => p.path !== path);
      setPhaseFiles(newPaths);
      setPhases(newPhases);
      pruneEdsMaps(newPhases);
      if (newPhases.length === 0) {
        setPhaseInfo(t('pcPhase.notLoaded'));
        setFileStatus(t('phases.noFileLoaded'));
        setFileStatusColor(C.red);
      } else {
        setPhaseInfo(t('phases.phasesPrefix', { list: newPhases.map(p => p.formula || p.filename).join(', ') }));
        setFileStatus(t('phases.selectedCount', { count: newPhases.length }));
      }
    } else {
      // Add phase
      const newPaths = [...phaseFiles, path];
      const newPhases = [...phases, file];
      setPhaseFiles(newPaths);
      setPhases(newPhases);
      setFileStatus(t('phases.selectedCount', { count: newPaths.length }));
      setFileStatusColor(C.green);
      setPhaseInfo(t('phases.phasesPrefix', { list: newPhases.map(p => p.formula || p.filename).join(', ') }));

      // For dictionary: auto-select best dict for this phase
      if (method === 'dictionary') {
        const formula = (file.formula || '').toLowerCase();
        const dicts = discoveredFiles.filter(
          f => f.file_type === 'dictionary' && (f.formula || '').toLowerCase() === formula
        );
        if (dicts.length > 0) {
          // Pick dict with lowest PC delta, then finest resolution
          const best = dicts.reduce((a, b) => {
            const da = a.pc_delta_percent ?? 999;
            const db = b.pc_delta_percent ?? 999;
            if (da !== db) return da < db ? a : b;
            return (a.resolution_deg || 999) < (b.resolution_deg || 999) ? a : b;
          });
          setSelectedDictPaths(prev => ({ ...prev, [path]: best.path }));
        }
      }
    }
  }

  function handleSetAllPaths(paths) {
    // Drop any path whose extension doesn't match the current method —
    // same contract as handleTogglePath so bulk select/"Select all"
    // can't smuggle CIFs into a Spherical run.
    const methodExt = { hough: 'cif', dictionary: 'h5', spherical: 'sht' }[method];
    const altExt = method === 'dictionary' ? 'hdf5' : null;
    const filteredPaths = methodExt
      ? paths.filter(p => {
          const ext = (p || '').toLowerCase().split('.').pop();
          return ext === methodExt || ext === altExt;
        })
      : paths;
    const dropped = paths.length - filteredPaths.length;
    if (dropped > 0) {
      log(t('phases.skippedNoFile', { count: dropped, ext: methodExt, method }));
    }
    // Sync phases array to match the new set of paths
    const newPhases = filteredPaths.map(p => discoveredFiles.find(f => f.path === p)).filter(Boolean);
    setPhaseFiles(filteredPaths);
    setPhases(newPhases);
    pruneEdsMaps(newPhases);
    if (newPhases.length === 0) {
      setPhaseInfo(t('pcPhase.notLoaded'));
      setFileStatus(t('phases.noFileLoaded'));
      setFileStatusColor(C.red);
    } else {
      setPhaseInfo(t('phases.phasesPrefix', { list: newPhases.map(p => p.formula || p.filename).join(', ') }));
      setFileStatus(t('phases.selectedCount', { count: newPhases.length }));
      setFileStatusColor(C.green);
    }
  }

  // Keep handleRemovePhase as a thin delegate for SelectedPhasesList backward compat
  function handleRemovePhase(index) {
    const path = phaseFiles[index];
    if (path) {
      const fileObj = phases[index];
      if (fileObj) handleTogglePath(fileObj);
    }
  }

  // Live crystallographic-degeneracy detection over the selected phases.
  const phaseDegeneracy = useMemo(
    () => detectPhaseDegeneracy(phases),
    [phases],
  );

  // Keep the per-phase EDS chemistry-prior maps in sync with the surviving
  // phase set: drop any phase.path key that is no longer selected so a removed
  // phase can never leak a stale strength into buildParams()'s payload. Called
  // with [] this clears everything; called with newPhases it prunes surgically
  // and preserves the strengths of phases that remain.
  function pruneEdsMaps(keepPhases) {
    const keep = new Set((keepPhases || []).map(p => p.path));
    setEdsStrengths(s => Object.fromEntries(Object.entries(s).filter(([k]) => keep.has(k))));
    setEdsExpectedOverrides(o => Object.fromEntries(Object.entries(o).filter(([k]) => keep.has(k))));
  }

  // Remove several phases at once (used by the "reduce to one" banner action).
  // phases[] and phaseFiles[] are kept index-aligned by handleTogglePath /
  // handleSetAllPaths, so a positional filter is safe.
  function handleReducePhases(indicesToRemove) {
    if (!indicesToRemove || indicesToRemove.length === 0) return;
    const removeSet = new Set(indicesToRemove);
    const newPhases = phases.filter((_, i) => !removeSet.has(i));
    const newPaths = phaseFiles.filter((_, i) => !removeSet.has(i));
    setPhases(newPhases);
    setPhaseFiles(newPaths);
    pruneEdsMaps(newPhases);
    if (newPhases.length === 0) {
      setPhaseInfo(t('pcPhase.notLoaded'));
      setFileStatus(t('phases.noFileLoaded'));
      setFileStatusColor(C.red);
    } else {
      setPhaseInfo(t('phases.phasesPrefix', { list: newPhases.map(p => p.formula || p.filename).join(', ') }));
      setFileStatus(t('phases.selectedCount', { count: newPhases.length }));
      setFileStatusColor(C.green);
    }
  }

  function handleSelectDict(phase, dictFile) {
    // Map master/phase path → chosen dictionary path
    setSelectedDictPaths(prev => ({ ...prev, [phase.path]: dictFile.path }));
  }

  // ---------------------------------------------------------------------------
  // GPU status banner — probe and every 5s while NOT indexing AND while this
  // page is visible. The page never unmounts (App.jsx keeps all pages mounted
  // and toggles display), so without the isActive gate this poller hammered
  // /gpu-status every 5s for the whole session even when the user was on
  // another page — wasted backend load that contributed to cross-page lag.
  // During an indexing run the runtime info comes from the status poll
  // (see /status handler above), so we don't double-poll.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!isActive) return undefined;
    let cancelled = false;
    const probe = async () => {
      try {
        const r = await indexApi.gpuStatus();
        if (!cancelled && r?.data) setRuntimeInfo(r.data);
      } catch (err) {
        // Silent — banner just stays hidden if backend is unreachable.
      }
    };
    probe();
    if (running) return () => { cancelled = true; };
    const t = setInterval(probe, 5000);
    return () => { cancelled = true; clearInterval(t); };
  }, [running, isActive]);

  // ---------------------------------------------------------------------------
  // Elapsed time counter
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!running) { setElapsed(''); return; }
    const t = setInterval(() => {
      if (!startTime) return;
      const s = Math.floor((Date.now() - startTime) / 1000);
      const m = Math.floor(s / 60);
      const sec = s % 60;
      setElapsed(m > 0 ? `${m}m ${sec.toString().padStart(2, '0')}s` : `${sec}s`);
    }, 1000);
    return () => clearInterval(t);
  }, [running, startTime]);

  // ---------------------------------------------------------------------------
  // Pixel count calculation
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (nRows === 0 || nCols === 0) {
      setPixelCount('');
      return;
    }
    if (selMode === 'full') {
      const n = nRows * nCols;
      setPixelCount(t('selection.pixelCountFull', { n, total: n }));
    } else if (selMode === 'region') {
      const r0 = region.rowStart, r1 = region.rowEnd;
      const c0 = region.colStart, c1 = region.colEnd;
      const n = Math.max(0, r1 - r0) * Math.max(0, c1 - c0);
      const tot = nRows * nCols;
      const pct = tot > 0 ? (100 * n / tot).toFixed(1) : '0.0';
      setPixelCount(t('selection.pixelCount', { n, total: tot, pct }));
    }
    // mask mode: set externally via setMaskInfo / setPixelCount
  }, [selMode, region, nRows, nCols]);

  // Validate phase file compatibility (kV, PC, detector shape)
  useEffect(() => {
    if (phaseFiles.length === 0) return;
    const warnings = [];
    for (const fp of phaseFiles) {
      const disc = discoveredFiles.find(d => (d.path || d.name) === fp);
      if (!disc) continue;
      // kV mismatch warning
      if (disc.energy_kv && disc.energy_kv !== energy) {
        warnings.push(t('warnings.kvMismatch', { label: disc.display_label || fp.split(/[\\/]/).pop(), fileKv: disc.energy_kv, datasetKv: energy }));
      }
      // PC deviation warning (only for dictionaries with stored PC)
      if (disc.pc && disc.pc.length === 3 && pcValues) {
        const pcMatch = pcValues.match(/PC:\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);
        if (pcMatch) {
          const currentPC = [parseFloat(pcMatch[1]), parseFloat(pcMatch[2]), parseFloat(pcMatch[3])];
          const maxDev = Math.max(
            Math.abs(disc.pc[0] - currentPC[0]),
            Math.abs(disc.pc[1] - currentPC[1]),
            Math.abs(disc.pc[2] - currentPC[2])
          );
          if (maxDev > 0.05) {
            warnings.push(t('warnings.pcDeviation', { label: disc.display_label || fp.split(/[\\/]/).pop(), pct: (maxDev * 100).toFixed(1) }));
          }
        }
      }
    }
    if (warnings.length > 0) {
      setFileStatus(`⚠ ${warnings.join(' | ')}`);
      setFileStatusColor(C.yellow);
    }
  }, [phaseFiles, discoveredFiles, energy, pcValues]);

  // ---------------------------------------------------------------------------
  // Start indexing
  // ---------------------------------------------------------------------------

  async function handleStart() {
    if (running) return;

    const params = buildParams();
    log(t('messages.startingIndexing', { time: new Date().toLocaleTimeString(), method }));
    setRunning(true);
    setStartTime(Date.now());
    setProgress(0);
    setProgressVisible(true);
    setQualityVisible(false);
    setCanExport(false);
    setCanViewMatches(false);
    setCanRefine(false);
    setCanSendAnalysis(false);

    try {
      const r = await indexApi.start(params);
      const id = r.data?.task_id;
      setTaskId(id);
      if (method === 'dictionary') {
        log(t('dict.preparing'));
      }
      if (id) startPolling(id);
    } catch (err) {
      log(t('messages.errorPrefix', { error: err?.response?.data?.detail || err.message }));
      setRunning(false);
      setProgressVisible(false);
    }
  }

  function buildParams() {
    const selectionParams =
      selMode === 'full'   ? { selection_mode: 'full' } :
      selMode === 'region' ? {
        selection_mode: 'region',
        row_start: region.rowStart, row_end: region.rowEnd,
        col_start: region.colStart, col_end: region.colEnd,
      } :
      { selection_mode: 'mask', mask: chemMaskRef.current || undefined };

    // Collect all phase files (list + any single fileInput not yet added)
    let allFiles = phaseFiles.length > 0
      ? phaseFiles
      : (fileInput ? [fileInput] : []);

    // For dictionary method: replace master paths with selected dictionary paths
    if (method === 'dictionary' && Object.keys(selectedDictPaths).length > 0) {
      allFiles = allFiles.map(f => selectedDictPaths[f] || f);
    }

    // Same substitution as `allFiles` above, as a per-key function — so the EDS
    // maps (keyed on phase.path) get re-keyed to whatever path actually lands in
    // cif_paths/master_h5_paths/sht_paths. For Dictionary that's the selected
    // dictionary path; for Hough/Spherical it's phase.path unchanged.
    const edsRemapPath = p =>
      (method === 'dictionary' && selectedDictPaths[p]) ? selectedDictPaths[p] : p;

    const common = {
      method,
      dataset: selectedDataset || undefined,
      cif_paths: allFiles.filter(f => f.toLowerCase().endsWith('.cif')),
      master_h5_paths: allFiles.filter(f => /\.(h5|hdf5)$/i.test(f)),
      sht_paths: allFiles.filter(f => f.toLowerCase().endsWith('.sht')),
      send_to_phase_map: sendToPhaseMap,
      // Only forward the flag when there's actually a phase map on the
      // backend, otherwise the route returns 400. The flag overrides
      // selection_mode + mask on the backend, so we still ship those
      // for the legacy/non-phasemap codepath.
      use_phase_map_routing: phaseMapRoutingAvailable && usePhaseMapRouting,
      // EDS chemistry prior (per-phase) — only shipped when EDS is available AND
      // the user actually moved a strength slider above 0. Off by default ⇒ the
      // keys are absent and the payload is byte-identical to before this feature.
      // The maps are keyed on phase.path, but for Dictionary we substitute the
      // selected dictionary path into master_h5_paths (mirrors the allFiles map
      // above). The backend iterates those substituted paths and looks up the
      // strength by that key, so we re-key through the SAME substitution here or
      // every dictionary lookup would miss and the prior would silently drop.
      ...(edsOverlayAvailable && Object.values(edsStrengths).some(v => v > 0) ? {
        eds_phase_strengths: Object.fromEntries(
          Object.entries(edsStrengths)
            .filter(([, v]) => v > 0)
            .map(([p, v]) => [edsRemapPath(p), Number(v) / 100])),
        eds_expected_overrides: Object.fromEntries(
          Object.entries(edsExpectedOverrides).map(([p, d]) => [edsRemapPath(p), d])),
      } : {}),
      ...selectionParams,
    };

    if (method === 'hough') {
      return { ...common, n_bands: bands, t_sigma: tSigma, r_sigma: rSigma };
    } else if (method === 'dictionary') {
      return { ...common, metric, keep_n: keepN, resolution, energy_kv: energy, compute_mode: computeMode };
    } else if (method === 'spherical') {
      return {
        ...common,
        bandwidth: Number(bandwidth),
        nregions,
        refine,
        gausbckg,
        circmask: circmask === 'Inscribed circle (0)' ? 0 : (circmask === 'Custom radius' ? circmaskRadius : -1),
        backend: sphericalBackend,
      };
    } else if (method === 'embedding') {
      return { ...common, encoder_path: encoderPath, faiss_path: faissPath, k: embK };
    }
    return common;
  }

  function startPolling(id) {
    if (pollRef.current) clearInterval(pollRef.current);
    pollTerminatedRef.current = false;
    pollRef.current = setInterval(async () => {
      // Guard: an earlier tick may already have handled the terminal
      // status and called fetchResult. setInterval keeps firing in-flight
      // async callbacks even after clearInterval, so we must check here.
      if (pollTerminatedRef.current) return;
      try {
        const r = await indexApi.getStatus(id);
        if (pollTerminatedRef.current) return; // re-check after await
        const d = r.data;
        if (d.progress !== undefined) setProgress(Math.round(d.progress * 100));
        if (d.log) log(d.log);
        if (d.runtime) setRuntimeInfo(d.runtime);
        if (d.status === 'done' || d.status === 'complete' || d.status === 'completed' || d.status === 'error' || d.status === 'failed') {
          pollTerminatedRef.current = true;
          clearInterval(pollRef.current);
          pollRef.current = null;
          setRunning(false);
          if (d.status !== 'error' && d.status !== 'failed') {
            fetchResult();
          } else {
            log(t('messages.indexingFailed', { error: d.error || t('refineDialog.unknownError') }));
            setProgressVisible(false);
          }
        }
      } catch (err) {
        if (pollTerminatedRef.current) return;
        pollTerminatedRef.current = true;
        log(t('messages.pollingError', { error: err?.message || t('messages.connectionLost') }));
        clearInterval(pollRef.current);
        pollRef.current = null;
        setRunning(false);
        setProgressVisible(false);
      }
    }, 1000);
  }

  async function fetchResult() {
    try {
      const r = await indexApi.getLastResult();
      const d = r.data;
      if (d?.available === false) return;
      setIndexingResult(d, method);
      const ci = d?.mean_ci ?? d?.ci;
      const nPx = d?.n_indexed ?? d?.n_pixels;
      const pps = d?.per_phase_stats || [];
      setPerPhaseStats(pps);
      if (ci !== undefined) {
        const elapsedStr = startTime ? ` — ${((Date.now() - startTime) / 1000).toFixed(1)}s` : '';
        const ts = new Date().toLocaleTimeString();
        if (d?.multi_phase === true && pps.length > 0) {
          // Multi-phase summary: show best phase by CI
          const best = pps.reduce((a, b) => ((a.ci ?? 0) >= (b.ci ?? 0) ? a : b));
          const nPhases = pps.length;
          const totalPx = pps.reduce((sum, s) => sum + (s.pixels || 0), 0);
          setQuality(t('quality.bestSummary', { name: best.name, ci: (best.ci ?? 0).toFixed(4), phases: nPhases, pixels: totalPx, elapsed: elapsedStr, time: ts }));
        } else {
          const phases = d?.phases || [];
          const phaseNames = phases.map(p => typeof p === 'object' ? p.name : p);
          const phaseStr = phaseNames.length > 0 ? t('quality.phasesSuffix', { list: phaseNames.join(', ') }) : '';
          setQuality(t('quality.meanSummary', { ci: ci.toFixed(4), pixels: nPx ?? '?', elapsed: elapsedStr, time: ts }) + phaseStr);
        }
        setQualityVisible(true);
      }
      setCanExport(true);
      setCanViewMatches(method === 'dictionary');
      setCanRefine(method === 'dictionary');
      setCanSendAnalysis(true);
      // Update phase info from result
      if (d?.phases && d.phases.length > 0) {
        const phaseNames = d.phases.map(p => typeof p === 'object' ? p.name : p);
        setPhaseInfo(t('phases.phasesPrefix', { list: phaseNames.join(', ') }));
      }
      const ciStr = ci !== undefined ? ci.toFixed(4) : '—';
      log(t('messages.indexingComplete', { ci: ciStr }));
      toast.success(t('messages.indexingCompleteToast', { ci: ciStr }));
    } catch (err) {
      log(t('messages.couldNotFetchResult', { error: err.response?.data?.detail || err.message }));
    }
    setProgressVisible(false);
  }

  // ---------------------------------------------------------------------------
  // Stop
  // ---------------------------------------------------------------------------

  async function handleStop() {
    if (!taskId) return;
    try {
      await indexApi.stop(taskId);
      log(t('messages.indexingStopped'));
    } catch {
      log(t('messages.stopFailed'));
    } finally {
      pollTerminatedRef.current = true;
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      setRunning(false);
      setProgressVisible(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Browse helpers
  // ---------------------------------------------------------------------------

  async function browseFile(setter, filters) {
    if (window.electronAPI?.openFile) {
      const path = await window.electronAPI.openFile({ filters });
      if (path) setter(path);
    } else {
      log(t('messages.fileBrowseRequiresElectron'));
    }
  }

  function handleBrowsePhaseFile() {
    const filters =
      method === 'hough'      ? [{ name: 'CIF Files', extensions: ['cif'] }] :
      method === 'dictionary' ? [{ name: 'Master Pattern', extensions: ['h5', 'hdf5'] }] :
      method === 'spherical'  ? [{ name: 'SHT Files', extensions: ['sht'] }] :
      [];
    browseFile(setFileInput, filters);
  }

  function handleBrowseEncoder() {
    browseFile(setEncoderPath, [{ name: 'PyTorch Model', extensions: ['pt', 'pth'] }]);
  }

  function handleBrowseFaiss() {
    browseFile(setFaissPath, [{ name: 'Directory', extensions: [] }]);
  }


  // ---------------------------------------------------------------------------
  // Chemistry filters
  // ---------------------------------------------------------------------------

  function addChemFilter() {
    setChemFilters(prev => [...prev, { element: '', operator: '>', min: 2, max: 100 }]);
  }

  function removeChemFilter(i) {
    setChemFilters(prev => prev.filter((_, idx) => idx !== i));
  }

  function changeChemFilter(i, val) {
    setChemFilters(prev => prev.map((f, idx) => idx === i ? val : f));
  }

  async function applyChemMask() {
    const validFilters = chemFilters.filter(f => f.element);
    if (validFilters.length === 0) return;
    setChemPreview(t('mask.generating'));
    setFilterStats(null);
    try {
      const res = await edsApi.chemistryMask(validFilters, chemCombine, maskMargin);
      const data = res.data;
      setMaskInfo(t('mask.chemistryMaskInfo', { n: data.n_selected, total: data.n_total, pct: data.percentage }));
      setChemPreview(t('mask.maskGenerated', { count: data.n_selected }));
      setPixelCount(t('selection.pixelCount', { n: data.n_selected, total: data.n_total, pct: data.percentage }));
      chemMaskRef.current = data.mask;
      // Store per-filter statistics
      if (data.filter_stats) {
        setFilterStats(data.filter_stats);
      }
      // Set mask overlay for canvas visualization
      if (data.mask && data.shape) {
        setMaskOverlayData({
          mask2d: new Uint8Array(data.mask),
          shape: data.shape,
        });
      }
    } catch (err) {
      setChemPreview(t('mask.maskFailed', { error: err.response?.data?.detail || err.message }));
    }
  }

  const handleEdsLayersChange = useCallback((layers) => {
    setEdsOverlayLayers(layers);
  }, []);

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const showHough      = method === 'hough';
  const showDict       = method === 'dictionary';
  const showSpherical  = method === 'spherical';
  const showEmbedding  = method === 'embedding';
  const canStart       = dataLoaded && !running;

  // Master to seed the "Generate Dictionary" dialog: the single file field, or
  // a master (.h5 that is not a pre-generated _dict_) already in the Selected
  // Phases list — prefer a true master, fall back to any .h5. Without this the
  // Generate button greys out the moment the master is moved into the phase
  // list (which clears fileInput), even though a master is clearly selected.
  const genDictMaster =
    fileInput ||
    phaseFiles.find(f => /\.(h5|hdf5)$/i.test(f) && !/_dict_/i.test(f)) ||
    phaseFiles.find(f => /\.(h5|hdf5)$/i.test(f)) ||
    '';

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const leftPanel = (
    <NavigationMapPanel
      navImage={navImage}
      nRows={nRows}
      nCols={nCols}
      selectionMode={selMode}
      onSelectionModeChange={setSelMode}
      region={region}
      onRegionChange={updateRegion}
      savedRegions={savedRegions}
      savedRegionCount={savedRegions.length}
      onAddRegion={() => setSavedRegions(prev => [...prev, { ...region }])}
      onClearAllRegions={() => setSavedRegions([])}
      maskInfo={maskInfo}
      pixelCount={pixelCount}
      edsOverlayEnabled={edsOverlayEnabled}
      edsOverlayAvailable={edsOverlayAvailable}
      onEdsOverlayToggle={setEdsOverlayEnabled}
      onEdsLayersChange={handleEdsLayersChange}
      edsOverlayLayers={edsOverlayLayers}
      edsFileKey={`${filePath || ''}::${selectedDataset || ''}`}
      maskOverlayData={maskOverlayData}
      chemFilters={chemFilters}
      onAddChemFilter={addChemFilter}
      onRemoveChemFilter={removeChemFilter}
      onChangeChemFilter={changeChemFilter}
      onApplyChemMask={applyChemMask}
      chemPreview={chemPreview}
      chemCombine={chemCombine}
      onChemCombineChange={setChemCombine}
      maskMargin={maskMargin}
      onMaskMarginChange={setMaskMargin}
      filterStats={filterStats}
      phaseMapRoutingAvailable={phaseMapRoutingAvailable}
      usePhaseMapRouting={usePhaseMapRouting}
      onUsePhaseMapRoutingChange={setUsePhaseMapRouting}
    />
  );

  const rightPanel = (
    <div className="thin-scrollbar" style={{
      height: '100%',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
      gap: spacing.outerSpacing,
      paddingLeft: spacing.outerMargin,
      paddingRight: spacing.outerMargin,
      paddingTop: 0,
      paddingBottom: spacing.outerMargin,
    }}>

      {/* Dataset selector — hierarchical multi-file list.
          Top-level options are loaded files (selecting one re-loads via
          /switch-file). Indented options under the active file are
          within-file dataset derivatives (deepcopy from Frame Average, BG
          subtraction, etc.) — selecting one calls /select-dataset. */}
      <Row gap={8}>
        <InlineLabel>{t('dataset.label')}</InlineLabel>
        <select
          value={(() => {
            const activeFile = loadedFiles.find(f => f.active);
            // If user has selected a derivative (not the main signal whose
            // name matches the file stem), show it as dataset:<name>.
            // Otherwise show the active file's path so the file row is
            // highlighted as the current selection.
            if (selectedDataset && datasets.includes(selectedDataset)
                && activeFile && selectedDataset !== activeFile.name) {
              return `dataset:${selectedDataset}`;
            }
            if (activeFile) return `file:${activeFile.path}`;
            // Fallback for the legacy (no loaded-files) case
            if (selectedDataset) return `dataset:${selectedDataset}`;
            return '';
          })()}
          disabled={datasetSwitching}
          onChange={e => handleDatasetSwitch(e.target.value)}
          style={{ flex: 1, ...selectStyle() }}
          title={t('dataset.select')}
        >
          {loadedFiles.length === 0 && datasets.length === 0 && (
            <option value="">{t('dataset.noFileLoaded')}</option>
          )}
          {loadedFiles.map((f) => {
            const isActiveFile = !!f.active;
            const fileLabel = `${isActiveFile ? '● ' : '○ '}${f.name}`;
            // For the active file, list its within-file datasets as
            // indented children. For non-active files, list only the file
            // itself (its in-memory datasets aren't loaded).
            return (
              <Fragment key={f.path}>
                <option value={`file:${f.path}`}>{fileLabel}</option>
                {isActiveFile && datasets.length > 0 && datasets.map(d => {
                  // Hide the "main" dataset entry that just mirrors the
                  // file name (its stem) — that's already represented by
                  // the file row above. Show only derivatives.
                  const isMainSignal = d === f.name;
                  if (isMainSignal) return null;
                  return (
                    <option key={`${f.path}::${d}`} value={`dataset:${d}`}>
                      {`    └ ${d}`}
                    </option>
                  );
                })}
              </Fragment>
            );
          })}
          {/* Legacy fallback: if loaded-files endpoint is empty but
              datasets is not, show flat list (preserves old behaviour for
              setups where loaded-files registry wasn't populated). */}
          {loadedFiles.length === 0 && datasets.length > 0 && datasets.map(d => (
            <option key={d} value={`dataset:${d}`}>{d}</option>
          ))}
        </select>
      </Row>

      {/* Data status */}
      <div style={{
        fontSize: '10pt', fontWeight: 'bold',
        color: dataLoaded ? C.green : C.red,
        padding: '5px 8px',
        border: `1px solid ${dataLoaded ? `${C.green}44` : C.border}`,
        borderRadius: 4,
        background: dataLoaded ? `${C.green}0a` : 'transparent',
        transition: 'color 0.3s, border-color 0.3s, background 0.3s',
      }}>
        {dataStatus}
      </div>

      {/* PC & Phase summary */}
      <GroupBox title={t('pcPhase.groupTitle')}>
        <StatusLabel>{pcValues}</StatusLabel>
        <StatusLabel>{phaseInfo}</StatusLabel>
        <div style={{ marginTop: 4 }}>
          <Button
            variant="default"
            small
            title={t('pcPhase.openPcRefinementTip')}
            onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'pcrefinement' } }))}
          >
            {t('pcPhase.openPcRefinement')}
          </Button>
        </div>
      </GroupBox>

      {/* Method selector */}
      <Row gap={8}>
        <InlineLabel>{t('method.label')}</InlineLabel>
        <select
          value={method}
          onChange={e => setMethod(e.target.value)}
          style={{ flex: 1, ...selectStyle() }}
          title={t('method.tooltip')}
        >
          {methodOptions.map(m => (
            <option key={m.value} value={m.value} title={m.tip}>{m.label}</option>
          ))}
        </select>
      </Row>

      {/* Phase Selection — inline list + floating picker */}
      <GroupBox title={t('phases.groupTitle')}>
        <SelectedPhasesList
          phases={phases}
          method={method}
          allDiscoveredFiles={discoveredFiles}
          onRemovePhase={handleRemovePhase}
          degeneracy={phaseDegeneracy}
          onReducePhases={handleReducePhases}
          onAddClick={() => setPhasePanelOpen(true)}
          currentPc={(() => {
            if (!pcValues) return null;
            const m = pcValues.match(/PC:\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);
            return m ? [parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3])] : null;
          })()}
          onSelectDict={handleSelectDict}
          selectedDictPaths={selectedDictPaths}
          edsAvailable={edsOverlayAvailable}
          edsStrengths={edsStrengths}
          onStrengthChange={(path, v) => setEdsStrengths(s => ({ ...s, [path]: v }))}
          expectedOverrides={edsExpectedOverrides}
          onExpectedChange={(path, d) => setEdsExpectedOverrides(o => ({ ...o, [path]: d }))}
        />
        <StatusLabel color={fileStatusColor} style={{ marginTop: 4 }}>{fileStatus}</StatusLabel>
      </GroupBox>

      {/* Hough Parameters */}
      {showHough && (
        <GroupBox title={t('hough.groupTitle')}>
          <Row gap={8} style={{ flexWrap: 'wrap' }}>
            <InlineLabel>{t('hough.bands')}</InlineLabel>
            <NumberInput
              value={bands}
              onChange={e => setBands(Number(e.target.value))}
              min={3} max={30} step={1}
              style={{ width: 64 }}
              title={t('hough.bandsTip')}
            />
            <InlineLabel>{t('hough.tSigma')}</InlineLabel>
            <NumberInput
              value={tSigma}
              onChange={e => setTSigma(Number(e.target.value))}
              min={0.5} max={10} step={0.5}
              style={{ width: 64 }}
              title={t('hough.tSigmaTip')}
            />
            <InlineLabel>{t('hough.rSigma')}</InlineLabel>
            <NumberInput
              value={rSigma}
              onChange={e => setRSigma(Number(e.target.value))}
              min={0.5} max={10} step={0.5}
              style={{ width: 64 }}
              title={t('hough.rSigmaTip')}
            />
          </Row>
        </GroupBox>
      )}

      {/* Dictionary Parameters */}
      {showDict && (
        <GroupBox title={t('dict.groupTitle')}>
          <Row gap={8} style={{ flexWrap: 'wrap', marginBottom: 6 }}>
            <InlineLabel>{t('dict.metric')}</InlineLabel>
            <select
              value={metric}
              onChange={e => setMetric(e.target.value)}
              style={{ ...selectStyle(), width: 80 }}
              title={t('dict.metricTip')}
            >
              {METRIC_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <InlineLabel>{t('dict.keepN')}</InlineLabel>
            <NumberInput
              value={keepN}
              onChange={e => setKeepN(Number(e.target.value))}
              min={1} max={100} step={1}
              style={{ width: 64 }}
              title={t('dict.keepNTip')}
            />
          </Row>
          <Row gap={8} style={{ flexWrap: 'wrap' }}>
            <InlineLabel>{t('dict.resolution')}</InlineLabel>
            <NumberInput
              value={resolution}
              onChange={e => setResolution(Number(e.target.value))}
              min={1} max={10} step={0.5}
              style={{ width: 64 }}
              title={t('dict.resolutionTip')}
            />
            <InlineLabel>{t('dict.energy')}</InlineLabel>
            <NumberInput
              value={energy}
              onChange={e => setEnergy(Number(e.target.value))}
              min={1} max={40} step={1}
              style={{ width: 56 }}
              title={t('dict.energyTip')}
            />
            <InlineLabel>{t('dict.kv')}</InlineLabel>
            <button
              style={btnSmall(C.border, C.text, !genDictMaster)}
              disabled={!genDictMaster}
              onClick={() => setShowGenDictDialog(true)}
              title={t('dict.generateDictionaryTip')}
            >
              {t('dict.generateDictionary')}
            </button>
          </Row>
          {/* Compute toggle (Auto / GPU / CPU). The GPU radio is disabled when
              /api/system/gpu reports no CUDA device. When the user explicitly
              picks GPU and the backend raises GpuDictError, the toast shows
              the message verbatim — no silent CPU fall-back. */}
          <Row gap={8} style={{ flexWrap: 'wrap', marginTop: 6 }}>
            <InlineLabel>{t('dict.compute')}</InlineLabel>
            {[
              { value: 'auto', label: t('dict.computeAuto') },
              { value: 'gpu',  label: t('dict.computeGpu') },
              { value: 'cpu',  label: t('dict.computeCpu') },
            ].map(opt => {
              const disabled = opt.value === 'gpu' && gpuStatus !== null && !gpuStatus.available;
              return (
                <label
                  key={opt.value}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 4,
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    fontSize: '10pt',
                    color: disabled ? C.textSecondary : C.text,
                    opacity: disabled ? 0.5 : 1,
                    userSelect: 'none',
                  }}
                  title={
                    opt.value === 'auto' ? t('dict.computeAutoTip') :
                    opt.value === 'gpu'  ? (disabled ? t('dict.computeGpuDisabledTip') : t('dict.computeGpuTip')) :
                                           t('dict.computeCpuTip')
                  }
                >
                  <input
                    type="radio"
                    name="compute_mode"
                    value={opt.value}
                    checked={computeMode === opt.value}
                    disabled={disabled}
                    onChange={e => setComputeMode(e.target.value)}
                    style={{ accentColor: C.purple, cursor: disabled ? 'not-allowed' : 'pointer' }}
                  />
                  {opt.label}
                </label>
              );
            })}
          </Row>
          {gpuStatus !== null && (
            <StatusLabel color={C.textSecondary} style={{ fontSize: '8.5pt', marginTop: 2 }}>
              {gpuStatus.available
                ? t('dict.gpuDetected', { name: gpuStatus.name, total: (gpuStatus.vram_total_gb ?? 0).toFixed(1), free: (gpuStatus.vram_free_gb ?? 0).toFixed(1) })
                : t('dict.gpuNone')}
            </StatusLabel>
          )}
          {dictInfo && <StatusLabel color={C.textSecondary}>{dictInfo}</StatusLabel>}
        </GroupBox>
      )}

      {/* Spherical Parameters */}
      {showSpherical && (
        <GroupBox title={t('spherical.groupTitle')}>
          <Row gap={8} style={{ flexWrap: 'wrap', marginBottom: 6 }}>
            <InlineLabel>{t('spherical.backend')}</InlineLabel>
            <select
              value={sphericalBackend}
              onChange={e => setSphericalBackend(e.target.value)}
              style={{ ...selectStyle(), width: 200 }}
              title={t('spherical.backendTip')}
            >
              <option value="spherical_gpu">{t('spherical.backendGpu')}</option>
              <option value="emsphinx">{t('spherical.backendEmsphinx')}</option>
            </select>
          </Row>
          <Row gap={8} style={{ flexWrap: 'wrap', marginBottom: 6 }}>
            <InlineLabel>{t('spherical.bandwidth')}</InlineLabel>
            <select
              value={bandwidth}
              onChange={e => setBandwidth(e.target.value)}
              style={{ ...selectStyle(), width: 80 }}
              title={t('spherical.bandwidthTip')}
            >
              {BANDWIDTH_OPTIONS.map(b => <option key={b} value={b}>{b}</option>)}
            </select>
            <InlineLabel>{t('spherical.regions')}</InlineLabel>
            <NumberInput
              value={nregions}
              onChange={e => setNregions(Number(e.target.value))}
              min={1} max={50} step={1}
              style={{ width: 56 }}
              title={t('spherical.regionsTip')}
            />
            <CheckOption
              label={t('spherical.refine')}
              checked={refine}
              onChange={e => setRefine(e.target.checked)}
              title={t('spherical.refineTip')}
            />
          </Row>
          <Row gap={8} style={{ flexWrap: 'wrap' }}>
            <CheckOption
              label={t('spherical.gaussianBg')}
              checked={gausbckg}
              onChange={e => setGausbckg(e.target.checked)}
              title={t('spherical.gaussianBgTip')}
            />
            <InlineLabel style={{ marginLeft: 8 }}>{t('spherical.circularMask')}</InlineLabel>
            <select
              value={circmask}
              onChange={e => setCircmask(e.target.value)}
              style={{ ...selectStyle(), width: 160 }}
              title={t('spherical.circularMaskTip')}
            >
              {circmaskOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            {circmask === 'Custom radius' && (
              <>
                <NumberInput
                  value={circmaskRadius}
                  onChange={e => setCircmaskRadius(Number(e.target.value))}
                  min={1} max={500} step={1}
                  style={{ width: 56 }}
                  title={t('spherical.customRadiusTip')}
                />
                <InlineLabel>{t('spherical.px')}</InlineLabel>
              </>
            )}
          </Row>

          {/* Pattern preprocessing preview */}
          <PreprocessingPreview
            row={qtRow} col={qtCol}
            gausbckg={gausbckg}
            circmask={circmask === 'Off (-1)' ? -1 : circmask === 'Inscribed circle (0)' ? 0 : circmaskRadius}
            nregions={nregions}
            dataLoaded={dataLoaded}
            nRows={nRows} nCols={nCols}
          />
        </GroupBox>
      )}

      {/* Embedding Parameters */}
      {showEmbedding && (
        <GroupBox title={t('embedding.groupTitle')}>
          <div style={{ fontSize: '9pt', color: C.text, marginBottom: 8, lineHeight: 1.5 }}>
            {t('embedding.description')}
          </div>
          <Row gap={6} style={{ marginBottom: 4 }}>
            <InlineLabel>{t('embedding.encoder')}</InlineLabel>
            <span title={encoderPath || undefined} style={{ fontSize: '9pt', color: encoderPath ? C.green : C.red, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {encoderPath || t('embedding.notLoaded')}
            </span>
            <button onClick={handleBrowseEncoder} style={btnSmall(C.border, C.text)}>{t('common:browse')}</button>
          </Row>
          <Row gap={6} style={{ marginBottom: 4 }}>
            <InlineLabel>{t('embedding.faissIndex')}</InlineLabel>
            <span title={faissPath || undefined} style={{ fontSize: '9pt', color: faissPath ? C.green : C.red, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {faissPath || t('embedding.notLoaded')}
            </span>
            <button onClick={handleBrowseFaiss} style={btnSmall(C.border, C.text)}>{t('common:browse')}</button>
          </Row>
          <Row gap={6}>
            <InlineLabel>{t('embedding.kNeighbors')}</InlineLabel>
            <NumberInput
              value={embK}
              onChange={e => setEmbK(Number(e.target.value))}
              min={1} max={50} step={1}
              style={{ width: 56 }}
              title={t('embedding.kNeighborsTip')}
            />
          </Row>
        </GroupBox>
      )}

      {/* Runtime banner — shows whether the run will use CUDA or fall
          back to CPU, plus how much VRAM is actually free. Critical
          because a stuck cache from a previous run can drop Spherical-GPU
          throughput to ~0 pat/s; turning the banner red gives the user
          a chance to hit "Release GPU" before they start. */}
      {runtimeInfo && (
        <Row gap={8} style={{ flexWrap: 'wrap', alignItems: 'center', marginBottom: 6 }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '6px 12px', borderRadius: 6,
            fontSize: '10pt', fontFamily: 'monospace', fontWeight: 600,
            background: runtimeInfo.device === 'cuda'
              ? (runtimeInfo.low_vram_warning ? `${C.red}22` : `${C.green}22`)
              : `${C.yellow}22`,
            color: runtimeInfo.device === 'cuda'
              ? (runtimeInfo.low_vram_warning ? C.red : C.green)
              : C.yellow,
            border: `1px solid ${
              runtimeInfo.device === 'cuda'
                ? (runtimeInfo.low_vram_warning ? C.red : C.green)
                : C.yellow
            }44`,
          }}>
            <span style={{
              display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
              background: runtimeInfo.device === 'cuda'
                ? (runtimeInfo.low_vram_warning ? C.red : C.green)
                : C.yellow,
            }} />
            {runtimeInfo.device === 'cuda'
              ? t('runtime.cuda', { name: runtimeInfo.device_name, free: runtimeInfo.free_vram_gb, total: runtimeInfo.total_vram_gb })
              : t('runtime.cpu', { name: runtimeInfo.device_name || t('runtime.noCuda') })}
          </span>
          {runtimeInfo.low_vram_warning && (
            <span style={{ fontSize: '9pt', color: C.red, fontWeight: 600 }}>
              {t('runtime.lowVram')}
            </span>
          )}
        </Row>
      )}

      {/* Action row 1 */}
      <Row gap={8} style={{ flexWrap: 'wrap' }}>
        <button
          onClick={handleStart}
          disabled={!canStart}
          style={{
            background: canStart ? C.green : C.textSecondary,
            color: C.bgSecondary,
            border: 'none', borderRadius: 4,
            padding: '7px 20px', fontSize: '11pt', fontWeight: 'bold',
            cursor: canStart ? 'pointer' : 'not-allowed',
            transition: 'background 0.15s, opacity 0.15s, transform 0.1s',
          }}
          onMouseEnter={e => { if (canStart) { e.currentTarget.style.opacity = '0.88'; e.currentTarget.style.transform = 'scale(1.02)'; } }}
          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'scale(1)'; }}
          title={running ? t('actions.indexingInProgress') : t('actions.startIndexingTip')}
        >
          {running ? <><span style={{ display: 'inline-block', width: 12, height: 12, border: '2px solid currentColor', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.7s linear infinite', marginRight: 6, verticalAlign: 'middle' }} />{t('actions.indexing')}</> : t('actions.startIndexing')}
        </button>

        {running && elapsed && (
          <span style={{
            fontSize: '10pt', color: C.yellow, fontFamily: 'monospace', fontWeight: 600,
            padding: '2px 8px', borderRadius: 8,
            background: `${C.yellow}11`, border: `1px solid ${C.yellow}33`,
          }}>
            {'\u23F1'} {elapsed}
          </span>
        )}

        <button
          onClick={handleStop}
          disabled={!running}
          style={{
            background: running ? C.red : C.textSecondary,
            color: 'white',
            border: 'none', borderRadius: 4,
            padding: '7px 14px', fontSize: '11pt', fontWeight: 'bold',
            cursor: running ? 'pointer' : 'not-allowed',
            transition: 'background 0.15s, opacity 0.15s, transform 0.1s',
          }}
          onMouseEnter={e => { if (running) { e.currentTarget.style.opacity = '0.88'; e.currentTarget.style.transform = 'scale(1.02)'; } }}
          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'scale(1)'; }}
          title={t('actions.stopTip')}
        >
          {t('actions.stop')}
        </button>

        {/* Emergency: release GPU memory held by PyTorch's caching
            allocator. Useful when a previous run got stuck and free
            VRAM is near 0 — Spherical-GPU would otherwise run at
            batch=1 / ~0 pat/s. Cheap when nothing is stuck. */}
        <button
          onClick={async () => {
            setReleasingGpu(true);
            try {
              const r = await indexApi.releaseGpu();
              const d = r.data || {};
              if (d.device === 'cuda') {
                setRuntimeInfo({
                  device: 'cuda',
                  device_name: d.device_name || 'CUDA',
                  free_vram_gb: d.free_after_gb || 0,
                  total_vram_gb: runtimeInfo?.total_vram_gb || 0,
                  low_vram_warning: (d.free_after_gb || 0) < 1.0,
                });
                // freed_mb is measured before any eviction, so it's the honest
                // total; show it in GB once it's ≥1 GB. The residual note is
                // honest about WHY the card isn't at 0 used: after release the
                // backend holds ~0 (torch_reserved_after_gb); the rest is the
                // GPU shared with other programs (Windows dwm/compositor,
                // browsers, Aztec, …) + the ~1-2 GB CUDA context — none of which
                // this button can free. Earlier this was mislabelled "CUDA
                // context", which made a shared-GPU situation look like a leak.
                const freedMb = d.freed_mb || 0;
                const freedStr = freedMb >= 1024
                  ? `${(freedMb / 1024).toFixed(2)} GB`
                  : `${Math.round(freedMb)} MiB`;
                const residual = (d.used_after_gb != null)
                  ? t('messages.gpuReleasedResidual', {
                      used: d.used_after_gb,
                      torch: d.torch_reserved_after_gb != null ? d.torch_reserved_after_gb : 0,
                    })
                  : '';
                log(t('messages.gpuReleased', { freed: freedStr, after: d.free_after_gb })
                  + (d.backends_evicted ? t('messages.gpuReleasedEvicted', { count: d.backends_evicted }) : '')
                  + (d.render_phases_cleared ? t('messages.gpuReleasedPhases', { count: d.render_phases_cleared }) : '')
                  + residual);
              } else {
                log(t('messages.gpuReleaseNoCuda'));
              }
            } catch (err) {
              log(t('messages.gpuReleaseFailed', { error: err?.message || err }));
            } finally {
              setReleasingGpu(false);
            }
          }}
          disabled={releasingGpu}
          style={{
            background: 'transparent',
            color: C.yellow,
            border: `1px solid ${C.yellow}`, borderRadius: 4,
            padding: '6px 12px', fontSize: '10pt', fontWeight: 600,
            cursor: releasingGpu ? 'wait' : 'pointer',
            opacity: releasingGpu ? 0.6 : 1,
          }}
          title={t('actions.releaseGpuTip')}
        >
          {releasingGpu ? t('actions.releasing') : t('actions.releaseGpu')}
        </button>

        <CheckOption
          label={t('actions.sendToPhaseMap')}
          checked={sendToPhaseMap}
          onChange={e => setSendToPhaseMap(e.target.checked)}
          title={t('hoverTips.sendToPhaseMap')}
        />

        <select
          onChange={async (e) => {
            const fmt = e.target.value;
            if (!fmt) return;
            e.target.value = '';
            log(t('messages.exporting', { fmt }));
            try {
              const res = await indexApi.exportResult(fmt, true, true);
              const url = window.URL.createObjectURL(res.data);
              const a = document.createElement('a');
              a.href = url;
              a.download = `indexing_result.${fmt}`;
              a.click();
              window.URL.revokeObjectURL(url);
              log(t('messages.exportComplete', { fmt }));
            } catch (err) {
              log(t('messages.exportFailed', { error: err.response?.data?.detail || err.message }));
            }
          }}
          disabled={!canExport}
          style={{ ...btnSmall(C.border, canExport ? C.text : C.textSecondary, !canExport), width: 80 }}
          title={t('actions.exportTip')}
        >
          <option value="">{t('actions.export')}</option>
          <option value="h5">{t('actions.exportH5')}</option>
          <option value="ang">{t('actions.exportAng')}</option>
        </select>
      </Row>

      {/* Action row 2 */}
      <Row gap={8} style={{ flexWrap: 'wrap' }}>
        <button
          disabled={!canViewMatches}
          onClick={() => setShowMatchesDialog(true)}
          style={btnSmall(C.border, canViewMatches ? C.text : C.textSecondary, !canViewMatches)}
          title={t('actions.patternMatchesTip')}
        >
          {t('actions.patternMatches')}
        </button>

        <button
          disabled={!canRefine}
          onClick={() => setShowRefineDialog(true)}
          style={btnSmall(C.border, canRefine ? C.text : C.textSecondary, !canRefine)}
          title={t('actions.refineTip')}
        >
          {t('actions.refine')}
        </button>

        <button
          disabled={!dataLoaded}
          onClick={() => setShowPhaseTestDialog(true)}
          style={btnSmall(C.border, dataLoaded ? C.text : C.textSecondary, !dataLoaded)}
          title={t('actions.phaseTestTip')}
        >
          {t('actions.phaseTest')}
        </button>

        <button
          onClick={() => setShowBatchDialog(true)}
          style={btnSmall('transparent', C.purple)}
          title={t('actions.batchTip')}
        >
          {t('actions.batch')}
        </button>

        <button
          disabled={!canSendAnalysis}
          onClick={() => {
            // Load indexing result into Analysis module, then navigate
            fetch('/api/analysis/load', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ xmap_path: '__from_indexing__' }),
            })
              .then(r => r.json())
              .then(d => {
                if (d.success) {
                  log(t('messages.sentToAnalysis', { phases: d.n_phases ?? 0, shape: JSON.stringify(d.shape) }));
                  window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'analysis' } }));
                } else {
                  log(t('messages.analysisLoadFailed', { detail: d.detail || JSON.stringify(d) }));
                }
              })
              // This is a raw fetch() (not axios), so a network error is a
              // plain Error with no `.response` — use err.message directly.
              .catch(err => log(t('messages.failedSendAnalysis', { error: err?.message || err })));
          }}
          style={{
            background: canSendAnalysis ? C.green : C.textSecondary,
            color: C.bgSecondary,
            border: 'none', borderRadius: 4,
            padding: '4px 12px', fontWeight: 'bold',
            cursor: canSendAnalysis ? 'pointer' : 'not-allowed',
            transition: 'background 0.15s, opacity 0.15s, transform 0.1s',
          }}
          onMouseEnter={e => { if (canSendAnalysis) { e.currentTarget.style.opacity = '0.88'; e.currentTarget.style.transform = 'scale(1.02)'; } }}
          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'scale(1)'; }}
          title={t('actions.analysisTip')}
        >
          {t('actions.analysis')}
        </button>
      </Row>

      {/* Progress bar */}
      <ProgressBar value={progress} visible={progressVisible} />

      {/* Quality summary */}
      {qualityVisible && (
        <div style={{
          fontSize: '9pt', fontWeight: 600, color: C.green, padding: '4px 8px',
          background: `${C.green}11`, border: `1px solid ${C.green}33`,
          borderRadius: 4, wordBreak: 'break-word',
          animation: 'pageFadeIn 0.3s ease-out',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span style={{ fontSize: '11pt' }}>{'\u2713'}</span>
          {quality}
          {perPhaseStats.length > 1 && (
            <button
              onClick={() => setShowResultModal(true)}
              title={t('hoverTips.resultDetails')}
              style={{
                marginLeft: 6, padding: '1px 8px', fontSize: '8pt', fontWeight: 600,
                background: `${C.green}22`, border: `1px solid ${C.green}55`,
                borderRadius: 3, color: C.green, cursor: 'pointer',
              }}
            >
              {t('actions.details')}
            </button>
          )}
        </div>
      )}

      {/* Log output */}
      <LogOutput lines={logLines} onClear={() => setLogLines([])} />

    </div>
  );

  return (
    <div style={{
      background: C.bg,
      color: C.text,
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      fontSize: 13,
      padding: spacing.outerMargin,
      boxSizing: 'border-box',
      overflow: 'hidden',
    }}>

      {/* Page title */}
      <div style={{ marginBottom: spacing.innerSpacing, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: C.accent }}>{t('page.title')}</h1>
          <span style={{
            fontSize: '8pt', fontWeight: 600, padding: '2px 8px', borderRadius: 10,
            background: `${C.cyan}1a`, color: C.cyan, letterSpacing: 0.3,
          }}>
            {methodOptions.find(m => m.value === method)?.label?.split(' ')[0] || 'Hough'}
          </span>
        </div>
        <div style={{ fontSize: '10pt', color: C.textSecondary, marginTop: 2 }}>
          {t('page.subtitle')}
        </div>
      </div>

      {/* Main splitter */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <ResizableSplitter
          left={leftPanel}
          right={rightPanel}
          defaultLeftWidth={360}
          minLeftWidth={200}
          maxLeftWidth={600}
          style={{ height: '100%' }}
        />
      </div>

      {/* Batch Indexing Dialog */}
      <BatchIndexingDialog open={showBatchDialog} onClose={() => setShowBatchDialog(false)} />

      {/* Refine Orientations Dialog */}
      <RefineDialog open={showRefineDialog} onClose={() => setShowRefineDialog(false)} onLog={log} />

      {/* Pattern Matches Viewer Dialog */}
      <PatternMatchesDialog open={showMatchesDialog} onClose={() => setShowMatchesDialog(false)} />

      <SinglePixelPhaseTestDialog open={showPhaseTestDialog} onClose={() => setShowPhaseTestDialog(false)}
        currentMethod={method} onUsePhase={handleUsePhaseFromTest} />

      {/* Multi-Phase Result Modal */}
      <PhaseResultModal
        open={showResultModal}
        onClose={() => setShowResultModal(false)}
        stats={perPhaseStats}
        method={method}
        totalPixels={perPhaseStats.reduce((sum, s) => sum + (s.pixels || 0), 0)}
        elapsed={quality.match(/(\d+\.\d+s)/)?.[1] || ''}
        onGoPhaseMap={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'phase-map' } }))}
        onGoAnalysis={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'analysis' } }))}
      />

      {/* Floating Phase Picker Panel */}
      <FloatingPhasePanel
        open={phasePanelOpen}
        onClose={() => setPhasePanelOpen(false)}
        title={t('phases.addPhaseTitle', { count: phases.length })}
        defaultX={window.innerWidth - 440}
        defaultY={100}
      >
        <PhaseDropdown
          discoveredFiles={discoveredFiles}
          groups={discoveredGroups}
          onTogglePath={handleTogglePath}
          onSetAll={handleSetAllPaths}
          selectedPaths={phaseFiles}
          method={method}
          open={true}
          onClose={() => setPhasePanelOpen(false)}
        />
      </FloatingPhasePanel>

      <GenerateDictionaryDialog
        isOpen={showGenDictDialog}
        onClose={() => setShowGenDictDialog(false)}
        initialMasterPath={genDictMaster}
        taskId={genDictTaskId}
        setTaskId={setGenDictTaskId}
        progress={genDictProgress}
        setProgress={setGenDictProgress}
        onGenerationDone={() => {
          // Refresh discovery so the new dictionary appears under the
          // phase card. discoverFilesForMethod is the canonical refresh
          // hook in this file.
          discoverFilesForMethod(method);
          // Auto-close after a short delay so the user sees the "done"
          // state, then clear the task id + progress. Clearing the id
          // matters because the polling effect re-fires on remount if
          // genDictTaskId is still set, which would hit /progress with
          // a stale id and surface a "Progress poll failed" error.
          setTimeout(() => {
            setShowGenDictDialog(false);
            setGenDictTaskId(null);
            setGenDictProgress(null);
          }, 800);
        }}
      />

    </div>
  );
}

// ---------------------------------------------------------------------------
// Small utility: select style
// ---------------------------------------------------------------------------
function selectStyle() {
  return {
    background: C.bg,
    border: `1px solid ${C.border}`,
    borderRadius: 4,
    color: C.text,
    fontSize: '10pt',
    padding: '3px 6px',
    height: 28,
    outline: 'none',
    cursor: 'pointer',
  };
}
