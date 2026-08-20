/**
 * EBSD Viewer — React port of gui/kikuchi_gui.py (KikuchiGuiUI)
 *
 * Layout (redesigned 2026-05-19):
 *   Header: title + subtitle | FileSwitcher · HDF5/EDS nav links
 *   Left panel (scroll): Datasets, Load, Pattern Processing,
 *                        EDS Composition, PC Refinement
 *   Right panel: Overview + Pattern (content-sized) | navigation bar
 *                | pattern action footer | collapsible log footer
 *   Status bar: file | grid | pattern | [row,col] | PC | memory
 *
 * Uses ONLY shared components from theme/components.jsx — no local primitives.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { ebsdApi, edsApi, pcApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useEdsColorStore from '../../stores/useEdsColorStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import { addRecentFile } from '../Dashboard/Dashboard';
import FileSwitcher from '../common/FileSwitcher';
import InfoTooltip from '../common/InfoTooltip';
import { qualityProvenanceKey } from '../common/qualityProvenance';
import LoadProgressModal from './LoadProgressModal';
import CropPanel from './CropPanel';
import {
  boundsOf, bytesPerSample, maskForTool, selectionPointsFor,
} from './navSelection';
import { datasetGeometry } from './datasetGeometry';
import { useSettledPath } from './useSettledPath';
// Same zoom mechanics as the EDS analysis maps — imported rather than copied so
// the two pages cannot drift apart. See ../EDS/zoomView.js for the derivation.
import {
  IDENTITY_VIEW, isZoomed, viewToTransform, zoomAt, panBy, wheelFactor,
} from '../EDS/zoomView';
import { zoomRectPct } from './zoomOverlay';
import ContextMenu from '../common/ContextMenu';
import ImageExportDialog from '../common/ImageExportDialog';
import {
  colors, alpha, spacing,
  Button, NumberInput, Select, Label,
  GroupBox, ResizableSplitter,
  FormRow, Separator, ScrollPanel, useConfirm, ConfirmDialog,
  usePrompt, PromptDialog,
} from '../../theme/components';

// A drag shorter than this (in CSS px) still counts as a click, so panning a
// zoomed overview never navigates to a pattern by accident. Same slop as the
// EDS tiles use for click-to-quantify.
const CLICK_SLOP_PX = 3;

// How long an ABANDONED lasso drag may hold the previous mask — a release we
// never heard about, e.g. over another application. Not a debounce interval:
// the mask settles the moment a drag ends, and this timer fires at most once
// per drag (see ./useSettledPath.js). Comfortably above the slowest realistic
// per-pixel interval of a careful trace, so an ordinary drag never reaches it,
// and short enough that a lost mouse-up cannot leave a stale mask standing
// while the hand travels to the Crop button.
const LASSO_BACKSTOP_MS = 2000;

// ---------------------------------------------------------------------------
// Small local helpers that have no equivalent in shared components
// ---------------------------------------------------------------------------

/** Inline checkbox row matching PyQt5 QCheckBox */
function CheckRow({ checked, onChange, label, title }) {
  return (
    <label
      title={title}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: '10pt', color: colors.text, cursor: 'pointer',
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: colors.purple, width: 13, height: 13, flexShrink: 0 }}
      />
      {label}
    </label>
  );
}

/** Compact range input (slider) matching QSlider */
function Slider({ value, onChange, min, max, title, label, displayValue }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      {label && (
        <span style={{ fontSize: '9pt', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
          {label}
        </span>
      )}
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        title={title}
        style={{ flex: 1, accentColor: colors.purple }}
      />
      {displayValue !== undefined && (
        <span style={{ fontSize: '9pt', color: colors.text, minWidth: 32, textAlign: 'right' }}>
          {displayValue}
        </span>
      )}
    </div>
  );
}

/** Quick mode toggle button (compact, matching pyqt5 quick-switch row) */
function QuickModeBtn({ label, active, onClick, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        background: active ? colors.purple : 'transparent',
        color: active ? colors.textOnAccent : colors.textSecondary,
        border: `1px solid ${active ? colors.purple : colors.border}`,
        borderRadius: 3,
        padding: '1px 6px',
        fontSize: '8pt',
        fontWeight: 600,
        cursor: 'pointer',
        minWidth: 28,
        textAlign: 'center',
        lineHeight: 1.6,
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// EDS composition overlay rendered on top of the pattern image
// ---------------------------------------------------------------------------

function EdsOverlay({ composition, mode }) {
  // Subscribe to the element→colour map so an "EDS Colors" edit re-renders the
  // pills/labels. Reading via getState().getColor() (as before) is a
  // non-reactive snapshot, so the colour never updated until some other prop
  // forced a re-render. Must run before the early return (rules of hooks).
  const elementColors = useEdsColorStore((s) => s.colors);
  if (!composition?.data) return null;

  const modeKey = { 'Counts': 'counts', 'Wt.%': 'wt_pct', 'At.%': 'at_pct' }[mode] || 'at_pct';
  const unit = mode === 'Counts' ? '' : '%';

  // Sort elements by value descending, take top 4
  const entries = Object.entries(composition.data)
    .map(([el, vals]) => ({ el, value: vals[modeKey] ?? vals.counts ?? 0 }))
    .filter((e) => e.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 4);

  if (entries.length === 0) return null;

  return (
    <div style={{
      position: 'absolute', bottom: 22, left: 6,
      background: '#00000099', borderRadius: 4,
      padding: '4px 6px', pointerEvents: 'none',
      display: 'flex', flexDirection: 'column', gap: 3,
      backdropFilter: 'blur(2px)',
    }}>
      {entries.map(({ el, value }) => {
        const color = elementColors[el] || '#bd93f9';
        const displayVal = mode === 'Counts'
          ? Math.round(value).toLocaleString('en-US')
          : value.toFixed(1);
        return (
          <div key={el} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{
              width: 10, height: 10, borderRadius: 2,
              background: color, flexShrink: 0,
            }} />
            <span style={{ fontSize: '8pt', color, fontWeight: 700, minWidth: 22 }}>{el}</span>
            <span style={{ fontSize: '8pt', color: '#ffffff', minWidth: 46, textAlign: 'right' }}>
              {displayVal}{unit}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
import { toast } from '../../stores/useToastStore';
import { disambiguateNames } from './disambiguateNames';

export default function EBSDViewer({ onNavigate, isActive }) {
  const { t } = useTranslation(['ebsdviewer', 'imageexport', 'common']);
  const { ebsdLoaded, ebsdInfo, setEBSDLoaded, setFileData, setMetadata,
          setNavigationGrid, setPendingChemMask, stepSize } = useDataStore();
  const [askConfirm, confirmProps] = useConfirm();
  const [askPrompt, promptProps] = usePrompt();

  // --- File loading ---
  const [filePath, setFilePath] = useState('');
  const [loadLoading, setLoadLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);
  // Progress modal: opens immediately on Load click, advances through 4
  // backend-driven stages (reading_metadata → building_signal →
  // detecting_features → finalising), stays open on error until dismissed.
  const [loadProgressOpen, setLoadProgressOpen] = useState(false);
  const [loadProgressState, setLoadProgressState] = useState(null);

  // --- Datasets ---
  const [datasets, setDatasets] = useState([]);
  const [activeDataset, setActiveDataset] = useState(null);
  const [deepcopy, setDeepcopy] = useState(true);

  // --- Loaded files (multi-file switcher in the left panel) ---
  // Sourced from the shared useLoadedFilesStore so this panel, the header
  // <FileSwitcher/>, and the Indexing dataset dropdown never disagree:
  // pruning the list in any of them refreshes all of them. Clicking an entry
  // calls /api/ebsd/switch-file (re-load from disk) and refetches the
  // EBSDViewer state.
  const loadedFiles = useLoadedFilesStore((s) => s.files);
  const refreshLoadedFiles = useLoadedFilesStore((s) => s.refresh);
  const pruneLoadedFile = useLoadedFilesStore((s) => s.remove);
  const clearLoadedFilesKeepActive = useLoadedFilesStore((s) => s.clearKeepActive);
  const clearAllLoadedFiles = useLoadedFilesStore((s) => s.clearAll);
  const [fileSwitching, setFileSwitching] = useState(false);

  // --- Pattern display ---
  const [pattern, setPattern] = useState(null);
  const [patternIsThumb, setPatternIsThumb] = useState(false); // true when showing atlas thumbnail
  const [patternLoading, setPatternLoading] = useState(false);
  const [row, setRow] = useState(0);
  const [col, setCol] = useState(0);

  // --- Overview ---
  const [overviewImage, setOverviewImage] = useState(null);
  // Band Contrast is sourced from Aztec's precomputed per-pixel array (read
  // in ~ms regardless of file size). Mean Intensity, by contrast, has to read
  // every pattern off a lazy multi-GB signal, so it must not be the default —
  // it made the overview appear to hang on large files (session 2026-06-01).
  const [overviewMode, setOverviewMode] = useState('Band Contrast');
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState(null);
  const [overviewSampled, setOverviewSampled] = useState(false);
  // Provenance of the current Band Contrast overview: "native" (Oxford, read
  // from file) vs "computed" (kikuchipy FFT pattern quality). Surfaced as a
  // caption chip so the user knows whether the map is measured or derived.
  const [overviewSource, setOverviewSource] = useState(null);
  const [qualityMask, setQualityMask] = useState(false);
  const [qualityThreshold, setQualityThreshold] = useState(25);
  const overviewRef = useRef(null);
  // Tracks the visible image rect inside the object-fit:contain element (for overlays)
  const [ovImgRect, setOvImgRect] = useState(null); // { left, top, width, height } in %

  // --- Processing ---
  const [windowSize, setWindowSize] = useState(3);
  const [gamma, setGamma] = useState(100);
  const [filter, setFilter] = useState('None');
  const [interpolation, setInterpolation] = useState('nearest');
  const [processingBusy, setProcessingBusy] = useState(false);
  const [processingLabel, setProcessingLabel] = useState('');
  // Granular progress for long ops (CLAHE) — {fraction, done, total, elapsed,
  // stage} or null when the op reports no chunk progress (→ plain spinner).
  const [procProgress, setProcProgress] = useState(null);
  const [claheKernel, setClaheKernel] = useState(8);

  // --- Circular detector signal mask ---
  // Local UI state; mirrored to backend via ebsdApi.setSignalMask. The mask
  // only makes sense for vendors that store patterns with a circular detector
  // aperture (EDAX, Bruker — the corners are zero in the raw data). Oxford
  // H5OINA stores the full rectangular camera frame, so a circular mask
  // would discard real diffraction data.
  //
  // Default is OFF for safety. After a file loads we auto-enable it for
  // EDAX-style files (see post-load effect below); the user can still toggle
  // either way and the choice is honoured until the next file load.
  const [maskEnabled, setMaskEnabled] = useState(false);
  const [maskRadius, setMaskRadius] = useState(100);  // percent of inscribed circle

  // --- Detector ---
  const [detector, setDetector] = useState(null);
  const [pcPatternCount, setPcPatternCount] = useState(0);

  // --- EDS ---
  const [edsMode, setEdsMode] = useState('At.%');
  const [edsComposition, setEdsComposition] = useState(null);
  const [edsPhases, setEdsPhases] = useState(null);
  const [showEdsOverlay, setShowEdsOverlay] = useState(false);

  // --- Zoom/pan for the two images ---
  // Each image keeps its OWN view; they show unrelated things (a map of the
  // sample vs. one detector pattern), so there is nothing to synchronise.
  // A view is { scale, cx, cy } with a normalised centre — see ../EDS/zoomView.
  const [ovView, setOvView] = useState(IDENTITY_VIEW);
  const [patView, setPatView] = useState(IDENTITY_VIEW);
  // Boxes the CSS transforms are relative to; also the reference rect for
  // turning pointer deltas into fractions.
  const ovBoxRef = useRef(null);
  const patBoxRef = useRef(null);
  // Pan bookkeeping: last pointer position plus distance travelled, so a drag
  // that pans does not also fire the overview's click-to-navigate.
  const ovPanRef = useRef(null);
  const ovSuppressClickRef = useRef(false);
  const patDragRef = useRef(null);

  // --- Right-click export ---
  // `contextMenu` holds the click point plus which image was hit; `exportFor`
  // is the image the dialog is currently open for.
  const [contextMenu, setContextMenu] = useState(null);
  const [exportFor, setExportFor] = useState(null);

  // --- Compare mode ---
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareLeft, setCompareLeft] = useState(null);
  const [compareRight, setCompareRight] = useState(null);
  const [compareLeftImg, setCompareLeftImg] = useState(null);
  const [compareRightImg, setCompareRightImg] = useState(null);

  // --- ROI selection (Shift+Drag on overview) ---
  // Two consumers: the image export (which reads `roi` for its crop rectangle)
  // and the crop panel below the overview. Both only READ it, so neither
  // changes the other's behaviour.
  const [roi, setRoi] = useState(null);
  // Doubles as the "a Shift+Drag is in progress" sentinel for ALL three tools:
  // cleared on mouse-up, so a later move with Shift still held cannot extend a
  // finished selection. The lasso sets it without reading its value.
  const roiStartRef = useRef(null);
  // The freehand path, kept apart from `roi` on purpose: the image export reads
  // `roi` as a crop rectangle, and a lasso is not one.
  const [lassoPoints, setLassoPoints] = useState([]);
  // Whether a lasso drag is in progress. The MASK is rebuilt when this goes
  // false, not while the path grows — see ./useSettledPath.js. Set true by
  // mouse-down; set false only by the window listener below, so that flag has
  // exactly one owner and cannot be left standing by a code path that forgot.
  const [lassoDrawing, setLassoDrawing] = useState(false);

  // --- Crop to selection ---
  const [cropTool, setCropTool] = useState('rect');
  const [cropBusy, setCropBusy] = useState(false);
  const [cropSaving, setCropSaving] = useState(false);
  // Where the ACTIVE dataset was cut from, or null when it is a full scan.
  const [cropWindow, setCropWindow] = useState(null);

  // --- Pattern atlas (for instant drag navigation) ---
  const [atlasInfo, setAtlasInfo] = useState(null); // {thumb_h, thumb_w, grid_rows, grid_cols}
  const atlasCanvasRef = useRef(null); // offscreen canvas with atlas image
  const fileInputRef = useRef(null);
  const atlasImageRef = useRef(null); // HTMLImageElement of atlas

  // --- Log ---
  const [logLines, setLogLines] = useState([]);
  const [logCopied, setLogCopied] = useState(false);
  const [logExpanded, setLogExpanded] = useState(false);
  const logRef = useRef(null);

  // --- Drag and drop ---
  const [dragOver, setDragOver] = useState(false);

  // --- Recent files ---
  const [recentFiles, setRecentFiles] = useState(() => {
    try { return JSON.parse(localStorage.getItem('kikuchipy_recent_files') || '[]'); } catch { return []; }
  });
  const [showRecent, setShowRecent] = useState(false);

  // Derived
  const gridShape = ebsdInfo?.grid_shape || null;
  const maxRow = gridShape ? gridShape[0] - 1 : 0;
  const maxCol = gridShape ? gridShape[1] - 1 : 0;

  const log = useCallback((msg) => {
    setLogLines((prev) => [
      ...prev.slice(-200),
      `[${new Date().toLocaleTimeString()}] ${msg}`,
    ]);
  }, []);

  // --- Persist recent files ---
  useEffect(() => {
    try { localStorage.setItem('kikuchipy_recent_files', JSON.stringify(recentFiles)); } catch { /* storage unavailable */ }
  }, [recentFiles]);

  // --- Auto-scroll log ---
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  // --- Auto-load from HDF5 Viewer (sessionStorage handoff) ---
  // Re-check when isActive changes because all pages stay mounted (QStackedWidget pattern),
  // so the initial [] effect fires before sessionStorage is set by HDF5 Viewer.
  useEffect(() => {
    try {
      const preload = sessionStorage.getItem('ebsd_preload_path');
      if (preload) {
        sessionStorage.removeItem('ebsd_preload_path');
        setFilePath(preload);
        setTimeout(() => doLoadFile(preload), 100);
      }
    } catch { /* sessionStorage unavailable */ }
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Sync datasets with backend when page becomes active ---
  useEffect(() => {
    if (!isActive || ebsdLoaded) return;
    ebsdApi.datasets().then(res => {
      const list = res.data?.datasets || [];
      if (list.length > 0) {
        const active = list.find(d => d.active) || list[0];
        // Same hyperspy (x, y) -> (rows, cols) flip as everywhere else, from
        // the one tested place. pattern_shape used to be written as the RAW
        // signal_shape, i.e. [w, h], while every other writer and both readers
        // (HDF5Viewer SummaryModal, PatternPanel) use [h, w] — so a pattern was
        // reported transposed whenever the viewer was entered with a file
        // already open in the backend.
        const geom = datasetGeometry(active);
        const gridShape = geom?.gridShape || [0, 0];
        setEBSDLoaded({
          grid_shape: gridShape,
          signal_shape: active.signal_shape || [0, 0],
          n_patterns: active.n_patterns || 0,
        });
        // Also set file data so other pages (EDS, etc.) see isFileOpen=true
        setFileData({
          file_path: active.name || '',
          format_type: 'h5oina',
          grid_shape: gridShape,
          pattern_count: active.n_patterns || 0,
          pattern_shape: geom?.patternShape || [0, 0],
          has_patterns: true,
          has_raw_patterns: false,
          has_eds: !!active.has_eds,
          has_electron_images: false,
          eds_elements: active.eds_elements || [],
          electron_images: [],
        });
        setDatasets(list);
        if (res.data?.active) setActiveDataset(res.data.active);
        // Persist file path for Recent Files on Dashboard
        const backendPath = res.data?.file_path;
        if (backendPath) {
          setFilePath(backendPath);
          addRecentFile(backendPath);
        }
        log(t('logMessages.syncedBackend', {
          name: active.name || 'dataset', rows: gridShape[0], cols: gridShape[1],
        }));
        fetchOverview(overviewMode);
        loadPattern(row, col);
        loadDetector();
        fetchMetadata();
        refreshLoadedFiles();
        if (active.has_eds) fetchEds(row, col);
      }
    }).catch(() => {});
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------------
  // AbortController ref to cancel stale pattern requests during drag
  const patternAbortRef = useRef(null);

  const loadPattern = useCallback(async (r, c, isDrag = false) => {
    // Cancel any in-flight request
    if (patternAbortRef.current) patternAbortRef.current.abort();
    const controller = new AbortController();
    patternAbortRef.current = controller;

    if (!isDrag) setPatternLoading(true);
    try {
      const config = { signal: controller.signal };
      if (filter && filter !== 'None') {
        config.params = { display_filter: filter };
      }
      const res = await ebsdApi.getPattern(r, c, config);
      if (controller.signal.aborted) return; // stale response
      const img = res.data?.image || null;
      setPattern(img);
      setPatternIsThumb(false);
    } catch (err) {
      if (err.name === 'AbortError' || err.code === 'ERR_CANCELED' || controller.signal.aborted) return;
      log(t('logMessages.loadError', { row: r, col: c, error: err.response?.data?.detail || err.message }));
      setPattern(null);
    } finally {
      if (!controller.signal.aborted) setPatternLoading(false);
    }
  }, [log, filter, t]);

  // Re-fetch pattern when display filter changes
  const filterRef = useRef(filter);
  useEffect(() => {
    if (filterRef.current !== filter) {
      filterRef.current = filter;
      if (pattern !== null) loadPattern(row, col);
    }
  }, [filter, pattern, row, col, loadPattern]);

  // Push circular signal mask state to the backend (debounced) and reload
  // the pattern so the user sees the change live.
  const maskPushDebounceRef = useRef(null);
  useEffect(() => {
    if (!ebsdLoaded) return;
    if (maskPushDebounceRef.current) clearTimeout(maskPushDebounceRef.current);
    maskPushDebounceRef.current = setTimeout(async () => {
      try {
        await ebsdApi.setSignalMask(maskEnabled, maskRadius / 100);
        loadPattern(row, col);
      } catch (err) {
        log(t('logMessages.setMaskError', { error: err.response?.data?.detail || err.message }));
      }
    }, 120);
    return () => {
      if (maskPushDebounceRef.current) clearTimeout(maskPushDebounceRef.current);
    };
  }, [maskEnabled, maskRadius, ebsdLoaded, row, col, loadPattern, log, t]);

  const loadDetector = useCallback(async () => {
    try {
      const res = await ebsdApi.getDetector();
      setDetector(res.data || null);
    } catch { setDetector(null); }
  }, []);

  const fetchMetadata = useCallback(async () => {
    try {
      const res = await ebsdApi.getMetadata();
      if (res.data) setMetadata(res.data);
    } catch {}
  }, [setMetadata]);

  // Load the pattern atlas — all patterns as one tiled image for instant navigation
  const fetchAtlas = useCallback(async () => {
    try {
      const res = await ebsdApi.getPatternAtlas(2);
      const d = res.data;
      if (!d?.atlas) return;

      // Create an Image from the atlas base64
      const img = new window.Image();
      img.src = `data:image/jpeg;base64,${d.atlas}`;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
      atlasImageRef.current = img;

      // Create an offscreen canvas for pixel extraction
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      atlasCanvasRef.current = canvas;

      setAtlasInfo({
        thumb_h: d.thumb_h,
        thumb_w: d.thumb_w,
        grid_rows: d.grid_rows,
        grid_cols: d.grid_cols,
      });
      log(t('logMessages.atlasLoaded'));
    } catch (err) {
      console.warn('Atlas load failed (falling back to per-pattern API):', err.message);
    }
  }, [log, t]);

  // Extract a pattern thumbnail from the atlas (instant, no API call)
  const getPatternFromAtlas = useCallback((r, c) => {
    if (!atlasCanvasRef.current || !atlasInfo) return null;
    const { thumb_h, thumb_w } = atlasInfo;
    const sx = c * thumb_w;
    const sy = r * thumb_h;
    // Create a small canvas for this tile
    const tile = document.createElement('canvas');
    tile.width = thumb_w;
    tile.height = thumb_h;
    const ctx = tile.getContext('2d');
    ctx.drawImage(atlasCanvasRef.current, sx, sy, thumb_w, thumb_h, 0, 0, thumb_w, thumb_h);
    return tile.toDataURL('image/png').replace('data:image/png;base64,', '');
  }, [atlasInfo]);

  const fetchOverview = useCallback(async (mode) => {
    setOverviewLoading(true);
    setOverviewError(null);
    try {
      const modeMap = {
        'Mean Intensity': 'mean', 'Std Dev (Quality)': 'std', 'Max Intensity': 'max',
        'Band Contrast': 'bc', 'Sharpness (Laplacian)': 'sharpness',
        'SNR (MAD)': 'snr', 'Neighbor Correlation': 'ncc', 'Entropy': 'entropy',
      };
      const res = await ebsdApi.overview(modeMap[mode] || 'mean');
      setOverviewImage(res.data?.image || null);
      setOverviewSampled(!!res.data?.sampled);
      setOverviewSource(res.data?.source || null);
    } catch (e) {
      // Surface the failure instead of silently leaving the empty placeholder —
      // a swallowed error here looked identical to "loaded but no image".
      setOverviewError(e?.response?.data?.detail || e?.message || 'Overview failed');
    } finally {
      setOverviewLoading(false);
    }
  }, []);

  // Returns the list as well as storing it: setDatasets is async, so a caller
  // that has just changed the active dataset cannot read the fresh entry back
  // out of state in the same tick.
  const fetchDatasets = useCallback(async () => {
    try {
      const res = await ebsdApi.datasets();
      const list = res.data?.datasets || [];
      const arr = Array.isArray(list) ? list : [];
      setDatasets(arr);
      if (res.data?.active) setActiveDataset(res.data.active);
      return arr;
    } catch { /* endpoint may not exist */ }
    return null;
  }, []);


  const fetchEds = useCallback(async (r, c) => {
    try {
      const modeMap = { 'Counts': 'counts', 'Wt.%': 'wt_pct', 'At.%': 'at_pct' };
      const res = await edsApi.quantifyPixel(r, c, modeMap[edsMode] || 'at_pct');
      setEdsComposition(res.data);
    } catch { setEdsComposition(null); }
    try {
      const res = await edsApi.suggestPhases(r, c);
      setEdsPhases(res.data);
    } catch { setEdsPhases(null); }
  }, [edsMode]);

  const navigateTo = useCallback((r, c) => {
    const nr = Math.max(0, Math.min(r, maxRow));
    const nc = Math.max(0, Math.min(c, maxCol));
    setRow(nr);
    setCol(nc);
    loadPattern(nr, nc);
    fetchEds(nr, nc);
  }, [maxRow, maxCol, loadPattern, fetchEds]);

  const doLoadFile = useCallback(async (path) => {
    if (!path?.trim()) { setLoadError(t('load.enterPath')); return; }
    setLoadLoading(true);
    setLoadError(null);
    setPattern(null);
    setDetector(null);
    setOverviewImage(null);
    log(t('logMessages.loadingFile', { name: path.split(/[\\/]/).pop() }));
    // Open the progress modal immediately with an initial state so the
    // user sees the breadcrumb before the first backend poll lands. The
    // backend writes the real stage within ~100 ms; until then we show
    // "Starting…" at stage 1/4.
    setLoadProgressOpen(true);
    setLoadProgressState({
      stage: 'reading_metadata',
      stage_idx: 1, stage_total: 4,
      elapsed_seconds: 0, message: t('logMessages.starting'),
    });
    try {
      const res = await ebsdApi.loadWithProgress(path.trim(), {
        onProgress: (state) => setLoadProgressState(state),
        pollIntervalMs: 250,
      });
      const info = res.data || {};
      setEBSDLoaded(info);
      setFileData({ ...info, file_path: path });
      setRow(0);
      setCol(0);
      setFilePath(path);
      // EDAX UP1/UP2 carry no pattern centre. If we couldn't recover it from
      // the .osc sidecar, indexing runs on kikuchipy's placeholder PC
      // (0.5,0.5,0.5) — almost certainly wrong. Warn loudly + persistently so
      // the user calibrates the PC before trusting any indexing result.
      if (info.pc_defaulted) {
        const warnMsg = t('logMessages.pcDefaultWarning');
        toast.warning(warnMsg, 15000);
        log(warnMsg);
      } else if (info.pc_source === 'osc') {
        log(t('logMessages.pcFromOsc'));
      }
      // Vendor-dependent: EDAX-style files store patterns with a circular
      // detector aperture (corners are 0 in raw data) → mask helps.
      // Oxford H5OINA stores the full rectangular camera frame → mask
      // would discard real data. Default is OFF; auto-enable for EDAX.
      setMaskEnabled(info.format_type === 'EDAX');
      // Reset the radius too, so a stale value (e.g. 25% from a prior EDAX
      // session) can't leak onto the next file. The mask state is fully
      // re-derived per file (see the comment block at the maskEnabled state).
      setMaskRadius(100);
      addRecentFile(path);
      setRecentFiles((prev) => {
        const filtered = prev.filter((f) => f !== path);
        return [path, ...filtered].slice(0, 10);
      });
      // Await only the ESSENTIAL fetches — the ones the user needs before
      // they can navigate patterns. fetchOverview is moved to the
      // fire-and-forget bucket below because on a 25k-pattern dataset it
      // reads every pattern from disk to compute mean intensity (3+ min
      // on slow disks). Blocking the modal on it makes the load FEEL
      // like it takes 3 minutes when in fact the file was ready after a
      // few seconds. The overview pane shows its own loading state and
      // fills in when the result lands; meanwhile the user can already
      // click around the navigation grid (which uses the pattern atlas
      // / direct pattern fetches, not the overview).
      await Promise.allSettled([
        loadPattern(0, 0),
        loadDetector(),
        fetchMetadata(),
        fetchDatasets(),
        refreshLoadedFiles(),
      ]);
      fetchOverview(overviewMode); // background — runs server-side in to_thread
      fetchEds(0, 0);
      fetchAtlas(); // load pattern atlas in background for instant drag
      const fname = path.split(/[\\/]/).pop();
      if (info.content_mode === 'eds_only') {
        // Aztec "Elementverteilungsdaten" acquisitions carry no diffraction
        // patterns at all. Say that plainly and point at what IS available,
        // so the empty pattern panes don't read as a failed load.
        const msg = t('logMessages.edsOnlyLoaded', {
          name: fname,
          elements: (info.eds_elements || []).length,
          images: (info.electron_images || []).length,
        });
        log(msg);
        toast.info(msg, 15000);
      } else {
        log(t('logMessages.loadedFile', { name: fname, rows: info.grid_shape?.[0], cols: info.grid_shape?.[1] }));
        toast.success(t('logMessages.loadedToast', { name: fname, rows: info.grid_shape?.[0], cols: info.grid_shape?.[1] }));
      }
      // Success: close the modal. Essential fetches are awaited above —
      // by here the viewer is interactive for navigation; overview and
      // atlas fill in async without blocking the user.
      setLoadProgressOpen(false);
    } catch (err) {
      // Modal stays open in error state — user clicks Close to dismiss.
      // loadProgressState was set to {stage:'error', message: detail} by
      // loadWithProgress's rejection path.
      const msg = err.response?.data?.detail || err.message || t('load.loadFailed');
      setLoadError(msg);
      log(t('logMessages.errorPrefix', { error: msg }));
      toast.error(t('logMessages.loadFailedToast', { error: msg }));
    } finally {
      setLoadLoading(false);
    }
  }, [log, t, setEBSDLoaded, setFileData, setMetadata, loadPattern, loadDetector, fetchMetadata, fetchDatasets, refreshLoadedFiles, fetchOverview, overviewMode, fetchEds, fetchAtlas]);

  // ---------------------------------------------------------------------------
  // Switch active file from the left-panel Loaded Files list. Calls
  // /api/ebsd/switch-file which re-loads the chosen file from disk, then
  // refreshes the viewer's local state to match the new active file.
  // ---------------------------------------------------------------------------
  const handleSwitchFile = useCallback(async (path) => {
    if (!path || fileSwitching) return;
    if (path === filePath) return; // already active
    setFileSwitching(true);
    setPattern(null);
    setDetector(null);
    setOverviewImage(null);
    const fname = path.split(/[\\/]/).pop();
    log(t('logMessages.switchingFile', { name: fname }));
    try {
      const res = await ebsdApi.switchFile(path);
      const info = res.data || {};
      setEBSDLoaded(info);
      setFileData({ ...info, file_path: path });
      setRow(0);
      setCol(0);
      setFilePath(path);
      // Re-detect vendor on switch — mask only applies to EDAX-style files.
      setMaskEnabled(info.format_type === 'EDAX');
      setMaskRadius(100);  // reset radius so a stale value can't leak across files
      // Mirrors doLoadFile: fetchOverview is fire-and-forget to keep the
      // switch responsive on large files. See the comment block in
      // doLoadFile for the full rationale.
      await Promise.allSettled([
        loadPattern(0, 0),
        loadDetector(),
        fetchMetadata(),
        fetchDatasets(),
        refreshLoadedFiles(),
      ]);
      fetchOverview(overviewMode);
      fetchEds(0, 0);
      fetchAtlas();
      log(t('logMessages.switchedFile', { name: fname }));
      toast.success(t('logMessages.switchedToast', { name: fname }));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('logMessages.switchFailedDefault');
      log(t('logMessages.switchError', { error: msg }));
      toast.error(t('logMessages.switchFailedToast', { error: msg }));
    } finally {
      setFileSwitching(false);
    }
  }, [filePath, fileSwitching, log, t, setEBSDLoaded, setFileData, setFilePath, loadPattern, loadDetector, fetchMetadata, fetchDatasets, refreshLoadedFiles, fetchOverview, overviewMode, fetchEds, fetchAtlas]);

  // The header <FileSwitcher/> does the backend switch + global-store sync
  // itself. This callback re-syncs the EBSD viewer's OWN local state
  // (filePath, pattern, overview, datasets) to the now-active file. Without
  // it, switching from the header left the viewer showing the PREVIOUS file's
  // overview/pattern while a DIFFERENT file was actually active backend-side —
  // the "I came back to the viewer and the patterns are wrong / out of bounds
  // for the grid" symptom. No extra switch-file call here (FileSwitcher already
  // did it), so no redundant disk re-load.
  const handleHeaderSwitcherSwitched = useCallback(async (path) => {
    if (!path) return;
    setPattern(null);
    setDetector(null);
    setOverviewImage(null);
    setRow(0);
    setCol(0);
    setFilePath(path);
    // Mask only applies to EDAX-style files; the store was just synced.
    setMaskEnabled(useDataStore.getState().formatType === 'EDAX');
    setMaskRadius(100);  // reset radius so a stale value can't leak across files
    await Promise.allSettled([
      loadPattern(0, 0),
      loadDetector(),
      fetchMetadata(),
      fetchDatasets(),
      refreshLoadedFiles(),
    ]);
    fetchOverview(overviewMode);
    fetchEds(0, 0);
    fetchAtlas();
  }, [setFilePath, loadPattern, loadDetector, fetchMetadata, fetchDatasets, refreshLoadedFiles, fetchOverview, overviewMode, fetchEds, fetchAtlas]);

  // Remove a single file from the loaded-files registry (the backend refuses
  // to remove the active file). The store action refreshes every consumer.
  const handleRemoveFile = useCallback(async (path) => {
    if (!path || fileSwitching) return;
    try {
      await pruneLoadedFile(path);
      const fname = path.split(/[\\/]/).pop();
      log(t('logMessages.removedFile', { name: fname }));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('logMessages.removeFailedDefault');
      toast.error(msg);
    }
  }, [fileSwitching, log, t, pruneLoadedFile]);

  // Clear every entry except the active file.
  const handleClearFiles = useCallback(async () => {
    if (fileSwitching) return;
    try {
      await clearLoadedFilesKeepActive();
      log(t('logMessages.clearedKeepActive'));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('logMessages.clearFailedDefault');
      toast.error(msg);
    }
  }, [fileSwitching, log, t, clearLoadedFilesKeepActive]);

  // Clear EVERY entry, including the active file (full reset). The active file
  // stays open in the viewer; it just disappears from the switcher list until
  // re-loaded — the one-click "make it empty" the user expects after a restart.
  const handleClearAllFiles = useCallback(async () => {
    if (fileSwitching) return;
    try {
      await clearAllLoadedFiles();
      log(t('logMessages.clearedAll'));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('logMessages.clearAllFailedDefault');
      toast.error(msg);
    }
  }, [fileSwitching, log, t, clearAllLoadedFiles]);

  // Run CLAHE while polling its backend chunk-progress so the overlay can
  // show a real bar (+ %, count, it/s, ETA) instead of an indeterminate
  // spinner. The backend runs CLAHE in a worker thread, so the poll is
  // served while it computes.
  const claheWithProgress = useCallback(async (kernel) => {
    const rid = (window.crypto?.randomUUID?.() || `clahe-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    let stopped = false;
    setProcProgress({ fraction: 0, stage: 'running' });
    (async () => {
      while (!stopped) {
        try {
          const r = await ebsdApi.processingProgress(rid);
          if (!stopped && r.data?.found) {
            setProcProgress({
              fraction: typeof r.data.fraction === 'number' ? r.data.fraction : 0,
              done: r.data.done, total: r.data.total,
              elapsed: r.data.elapsed_seconds, stage: r.data.stage,
            });
          }
        } catch { /* poll best-effort */ }
        await new Promise((f) => setTimeout(f, 400));
      }
    })();
    try {
      return await ebsdApi.clahe(kernel, rid);
    } finally {
      stopped = true;
      setProcProgress(null);
    }
  }, []);

  const doProcessing = useCallback(async (op, label) => {
    setProcessingBusy(true);
    setProcessingLabel(label);
    log(`${label}...`);
    // Helper: wrap any async step with start/done log lines + elapsed time
    // shown in the bottom log. Backend also emits its own logger.info per
    // step → these auto-stream to the DEV panel on the right via the
    // WebSocketLogHandler. Two channels, two granularities.
    const timed = async (name, fn) => {
      const t0 = performance.now();
      log(t('logMessages.stepArrow', { name }));
      try {
        const res = await fn();
        const ms = performance.now() - t0;
        const elapsed = res?.data?.elapsed_seconds;
        const extra = (typeof elapsed === 'number')
          ? ` (${elapsed.toFixed(1)}s backend, ${(ms / 1000).toFixed(1)}s total)`
          : ` (${(ms / 1000).toFixed(1)}s)`;
        log(t('logMessages.stepDone', { name, extra }));
        return res;
      } catch (err) {
        log(t('logMessages.stepFailed', { name, error: err.response?.data?.detail || err.message }));
        throw err;
      }
    };
    try {
      // Auto-deepcopy before destructive processing (matches PyQt5 _start_processing)
      // Names follow PyQt5 pattern: {base}_{suffix} (e.g. dataset_bg_dyn, dataset_avg)
      if (deepcopy) {
        const suffix = op === 'batch' ? 'ac' : op;
        const baseName = activeDataset || 'dataset';
        const copyName = `${baseName}_${suffix}`;
        const copyRes = await ebsdApi.deepcopy(copyName);
        const newName = copyRes.data?.name;
        if (newName) {
          await ebsdApi.selectDataset(newName);
          setActiveDataset(newName);
          log(t('logMessages.copiedTo', { name: newName }));
        }
      }
      if (op === 'bg_dyn') await timed(t('logMessages.stepBgDynamic'), () => ebsdApi.backgroundRemoval('dynamic'));
      else if (op === 'bg_stat') await timed(t('logMessages.stepBgStatic'), () => ebsdApi.backgroundRemoval('static'));
      else if (op === 'avg') await timed(t('logMessages.stepFrameAverage'), () => ebsdApi.frameAverage(windowSize));
      else if (op === 'autocontrast') await timed(t('logMessages.stepAutoContrast'), () => ebsdApi.autocontrast());
      else if (op === 'batch') await timed(t('logMessages.stepBatchAutoContrast'), () => ebsdApi.autocontrast());
      else if (op === 'clahe') await timed(t('logMessages.stepClahe', { kernel: claheKernel }), () => claheWithProgress(claheKernel));
      else if (op === 'pipeline') {
        // Recommended: Static BG → Dynamic BG → CLAHE in one shot.
        // Each step logs its own start/elapsed line; the user sees
        // exactly where the pipeline is + how long each step took.
        await timed(t('logMessages.stepPipeline1'), () => ebsdApi.backgroundRemoval('static'));
        await timed(t('logMessages.stepPipeline2'), () => ebsdApi.backgroundRemoval('dynamic'));
        await timed(t('logMessages.stepPipeline3', { kernel: claheKernel }), () => claheWithProgress(claheKernel));
      }
      log(t('logMessages.opComplete', { label }));
      await fetchDatasets();
      await loadPattern(row, col);
      fetchOverview(overviewMode);
      fetchAtlas();
    } catch (err) {
      // Per-step error is already logged by `timed`; this is the catch-all
      // for the non-timed parts (deepcopy, dataset reload).
      if (!err._loggedByTimed) {
        log(t('logMessages.opError', { label, error: err.response?.data?.detail || err.message }));
      }
    } finally {
      setProcessingBusy(false);
    }
  }, [log, t, windowSize, claheKernel, row, col, deepcopy, activeDataset, overviewMode, fetchDatasets, loadPattern, fetchOverview, fetchAtlas]);

  // --- Keyboard shortcuts matching PyQt5 ---
  useEffect(() => {
    const handler = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === 'ArrowLeft') { e.preventDefault(); navigateTo(row, col - (e.shiftKey ? 10 : 1)); }
      if (e.key === 'ArrowRight') { e.preventDefault(); navigateTo(row, col + (e.shiftKey ? 10 : 1)); }
      if (e.key === 'ArrowUp') { e.preventDefault(); navigateTo(row - (e.shiftKey ? 10 : 1), col); }
      if (e.key === 'ArrowDown') { e.preventDefault(); navigateTo(row + (e.shiftKey ? 10 : 1), col); }
      if (e.key === 'Home') { e.preventDefault(); navigateTo(0, 0); }
      if (e.key === 'End' && gridShape) { e.preventDefault(); navigateTo(gridShape[0] - 1, gridShape[1] - 1); }
      // "R" resets the zoom. Now covers BOTH images — the overview gained a
      // zoom of its own, and a shortcut that silently skipped it would leave
      // the user stuck at 16x with no keyboard way out.
      if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.altKey) { setPatView(IDENTITY_VIEW); setOvView(IDENTITY_VIEW); }
      if (e.ctrlKey && e.key.toLowerCase() === 'k') { e.preventDefault(); setCompareOpen((v) => !v); }
      if (e.altKey && e.key.toLowerCase() === 'd') { e.preventDefault(); doProcessing('bg_dyn', t('logMessages.labelBgDynamic')); }
      if (e.altKey && e.key.toLowerCase() === 's') { e.preventDefault(); doProcessing('bg_stat', t('logMessages.labelBgStatic')); }
      if (e.altKey && e.key.toLowerCase() === 'a') { e.preventDefault(); doProcessing('avg', t('logMessages.labelFrameAverage')); }
      if (e.altKey && e.key.toLowerCase() === 'c') { e.preventDefault(); doProcessing('autocontrast', t('logMessages.labelAutoContrast')); }
      if (e.altKey && e.key.toLowerCase() === 'r') { e.preventDefault(); setGamma(100); setFilter('None'); setInterpolation('nearest'); }
      if (e.ctrlKey && e.key.toLowerCase() === 'l') { e.preventDefault(); fileInputRef.current?.focus(); fileInputRef.current?.select(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [row, col, navigateTo, gridShape, doProcessing, t]);

  // --- Compute visible image bounds inside object-fit:contain element ---
  const updateOvImgRect = useCallback(() => {
    if (!overviewRef.current) return;
    const img = overviewRef.current.querySelector('img');
    if (!img || !img.naturalWidth) return;
    // offsetWidth/Height, NOT getBoundingClientRect(): the image now carries
    // the zoom transform, so its client rect is the ZOOMED rect, while
    // ovImgRect must stay the unzoomed letterbox geometry (the zoom is applied
    // on top of it at render time via zoomRectPct). Layout metrics ignore CSS
    // transforms, so they give the untransformed box directly.
    //
    // It must be the IMAGE's box, not the bordered container's: ovImgRect is
    // consumed as percentages of the overlay's parent, which is the container's
    // PADDING box. Measuring the border box instead is off by the 1px border on
    // each side — invisible at 1x, but zoomRectPct multiplies that error by the
    // magnification, drifting the crosshair and ROI off their features.
    const contW = img.offsetWidth;
    const contH = img.offsetHeight;
    if (!contW || !contH) return;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    const scale = Math.min(contW / natW, contH / natH);
    const imgW = natW * scale;
    const imgH = natH * scale;
    const offsetX = (contW - imgW) / 2;
    const offsetY = (contH - imgH) / 2;
    setOvImgRect({
      left: (offsetX / contW) * 100,
      top: (offsetY / contH) * 100,
      width: (imgW / contW) * 100,
      height: (imgH / contH) * 100,
    });
  }, []);

  // Recalculate on window resize AND container resize (e.g. sidebar toggle, layout shift)
  useEffect(() => {
    window.addEventListener('resize', updateOvImgRect);
    const el = overviewRef.current;
    let ro;
    if (el) {
      ro = new ResizeObserver(updateOvImgRect);
      ro.observe(el);
    }
    return () => {
      window.removeEventListener('resize', updateOvImgRect);
      if (ro) ro.disconnect();
    };
  }, [updateOvImgRect]);

  // --- Wheel zoom for both images ---
  // Attached natively with { passive: false }: React's onWheel is passive, so
  // preventDefault() there is ignored and Electron would zoom the whole app
  // instead. Plain wheel AND Ctrl+wheel both zoom — unlike the EDS tile grid,
  // neither image sits inside a scrollable list that a plain wheel must reach.
  useEffect(() => {
    const bind = (el, setView) => {
      if (!el) return () => {};
      const onWheel = (e) => {
        e.preventDefault();
        const rect = el.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        setView((v) => zoomAt(
          v,
          wheelFactor(e.deltaY),
          (e.clientX - rect.left) / rect.width,
          (e.clientY - rect.top) / rect.height,
        ));
      };
      el.addEventListener('wheel', onWheel, { passive: false });
      return () => el.removeEventListener('wheel', onWheel);
    };
    const unbind = [
      bind(ovBoxRef.current, setOvView),
      bind(patBoxRef.current, setPatView),
    ];
    return () => unbind.forEach((fn) => fn());
  }, []);

  // --- Overview pointer interaction ---
  // Zoom-aware for free: the <img> carries the CSS transform, so its
  // getBoundingClientRect() below IS the zoomed rect, and the object-fit math
  // scales with it. Every pixel mapping (navigate, ROI, crosshair) therefore
  // stays exact while zoomed without touching this function.
  const calcOverviewPos = (e) => {
    if (!overviewRef.current || !gridShape) return null;
    const img = overviewRef.current.querySelector('img');
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    // Account for object-fit: contain — the actual image may be smaller than the element
    const natW = img.naturalWidth || gridShape[1];
    const natH = img.naturalHeight || gridShape[0];
    const scale = Math.min(rect.width / natW, rect.height / natH);
    const imgW = natW * scale;
    const imgH = natH * scale;
    const offsetX = (rect.width - imgW) / 2;
    const offsetY = (rect.height - imgH) / 2;
    const rx = (e.clientX - rect.left - offsetX) / imgW;
    const ry = (e.clientY - rect.top - offsetY) / imgH;
    if (rx < 0 || rx > 1 || ry < 0 || ry > 1) return null;
    return {
      r: Math.max(0, Math.min(Math.round(ry * gridShape[0]), maxRow)),
      c: Math.max(0, Math.min(Math.round(rx * gridShape[1]), maxCol)),
      rx, ry,
    };
  };

  const handleOverviewPointer = (e, isDrag = false) => {
    const pos = calcOverviewPos(e);
    if (!pos) return;
    if (pos.r !== row || pos.c !== col) {
      setRow(pos.r);
      setCol(pos.c);

      if (isDrag) {
        // During drag: show atlas thumbnail instantly (no API call)
        const thumb = getPatternFromAtlas(pos.r, pos.c);
        if (thumb) {
          setPattern(thumb);
          setPatternIsThumb(true);
          return; // atlas is enough during drag
        }
      }

      // Not dragging or no atlas: fetch full-resolution pattern
      clearTimeout(window._ebsdDragTimer);
      window._ebsdDragTimer = setTimeout(() => {
        loadPattern(pos.r, pos.c, isDrag);
        if (!isDrag) fetchEds(pos.r, pos.c);
      }, isDrag ? 80 : 50);
    }
  };

  const handleOverviewMouseMove = (e) => {
    // Panning a zoomed overview takes precedence over scrubbing. Started only
    // in onMouseDown when the view is actually zoomed, so at 1x this branch is
    // dead and the drag-to-scrub behaviour below is bit-for-bit unchanged.
    if (ovPanRef.current) {
      if (e.buttons !== 1) { ovPanRef.current = null; return; }
      const box = ovBoxRef.current?.getBoundingClientRect();
      if (!box?.width || !box?.height) return;
      const dx = e.clientX - ovPanRef.current.x;
      const dy = e.clientY - ovPanRef.current.y;
      ovPanRef.current = {
        x: e.clientX,
        y: e.clientY,
        moved: ovPanRef.current.moved + Math.abs(dx) + Math.abs(dy),
      };
      if (ovPanRef.current.moved > CLICK_SLOP_PX) ovSuppressClickRef.current = true;
      setOvView((v) => panBy(v, dx / box.width, dy / box.height));
      return;
    }
    if (e.buttons === 1 && e.shiftKey && gridShape) {
      const pos = calcOverviewPos(e);
      if (!pos || !roiStartRef.current) return;
      if (cropTool === 'lasso') {
        // Only ever extend a path belonging to a drag that STARTED here.
        // `roiStartRef` alone is not enough: it survives a release outside the
        // overview, and a stale anchor would let a press that begins off-image
        // append to the path with `lassoDrawing` false — which makes
        // useSettledPath a passthrough and rebuilds the whole mask on every
        // single mouse-move, the exact freeze the settle exists to prevent.
        // Gating on the flag the hook itself reads keeps the two in step: a
        // path can only grow while the mask is frozen.
        if (!lassoDrawing) return;
        // Sampled at mouse-move rate, so a fast drag leaves gaps between
        // consecutive points. That is fine and deliberate: lassoMask
        // rasterises the segment BETWEEN neighbouring points, so the traced
        // stroke is closed either way and the path stays short enough to
        // re-fill the mask on every move. Only genuinely new grid cells are
        // appended — a slow drag across one pixel must not push hundreds of
        // identical points.
        setLassoPoints((pts) => {
          const last = pts[pts.length - 1];
          if (last && last.r === pos.r && last.c === pos.c) return pts;
          return [...pts, { r: pos.r, c: pos.c }];
        });
        return;
      }
      setRoi({
        startRow: Math.min(roiStartRef.current.row, pos.r),
        startCol: Math.min(roiStartRef.current.col, pos.c),
        endRow: Math.max(roiStartRef.current.row, pos.r),
        endCol: Math.max(roiStartRef.current.col, pos.c),
      });
    } else if (e.buttons === 1 && !e.shiftKey) {
      handleOverviewPointer(e, true);
    }
  };

  const crosshair = gridShape
    ? { x: ((col + 0.5) / gridShape[1]) * 100, y: ((row + 0.5) / gridShape[0]) * 100 }
    : null;

  // The drawn selection as a crop window. Every drag coordinate is INCLUSIVE
  // and already clamped to the grid by calcOverviewPos, so the hull boundsOf
  // returns always fits the scan — which is what keeps the backend from
  // refusing the window with a 400. CropPanel still rejects a non-finite box
  // on its own; see its isUsableBbox.
  //
  // The tool decides which drawn points count and which mask they build; both
  // decisions live in navSelection.js so they can be checked without a canvas.
  const selectionPoints = useMemo(
    () => selectionPointsFor(cropTool, lassoPoints, roi),
    [cropTool, lassoPoints, roi],
  );

  const cropBbox = useMemo(() => boundsOf(selectionPoints), [selectionPoints]);

  // The lasso path as it stands once the hand has let go. Rebuilding a mask
  // costs O(rows·cols·points) — 392 ms for a 361×461 selection with 1200
  // sampled points on this machine, 3.0 s at 693×923 — and nothing needs it
  // before the drag ends, so it is not paid until then.
  const lassoMaskPath = useSettledPath(lassoPoints, lassoDrawing, LASSO_BACKSTOP_MS);

  // The points the MASK is built from. For the box tools that is the very same
  // drag; for the lasso it is the SETTLED copy of the path, so the mask does
  // not have to be rebuilt from scratch on every mouse-move. The crop WINDOW
  // above — and with it the memory estimate, the number that decides whether a
  // crop is a good idea — stays live either way.
  const maskPoints = useMemo(
    () => selectionPointsFor(cropTool, lassoMaskPath, roi),
    [cropTool, lassoMaskPath, roi],
  );

  const cropMask = useMemo(
    () => maskForTool(cropTool, maskPoints, cropBbox),
    [cropTool, maskPoints, cropBbox],
  );

  // A lasso drag ends wherever the button is released — over the sidebar, over
  // the log, outside the window entirely. The overview's own onMouseUp only
  // hears about a release ON it, and a drag that stayed "in progress" would
  // hold the mask at the shape it had when the hand left. Listening on the
  // window makes the end of a drag a fact rather than a hope; the backstop in
  // useSettledPath covers only what even this misses (a mouse-up the browser
  // never delivers, e.g. released over another application).
  useEffect(() => {
    if (!lassoDrawing) return undefined;
    const end = () => {
      setLassoDrawing(false);
      // Both sentinels, cleared by the same event. The overview's own
      // onMouseUp never runs for a release over the sidebar, so without this
      // `roiStartRef` keeps the anchor of a finished drag — and a later press
      // that begins OUTSIDE the overview and drags in would find it still set
      // and carry on extending that drag.
      roiStartRef.current = null;
    };
    window.addEventListener('mouseup', end);
    return () => window.removeEventListener('mouseup', end);
  }, [lassoDrawing]);

  // Drop the drawn selection — both halves of it. One function rather than a
  // pair of setters at each site, because forgetting the second leaves a stale
  // path that the next tool switch would resurrect. It deliberately does NOT
  // touch `lassoDrawing`: that flag has one owner (the window mouse-up above),
  // and clearing it from here would let a still-live drag settle its mask on
  // every mouse-move — the exact cost useSettledPath exists to avoid.
  const clearSelection = useCallback(() => {
    setRoi(null);
    setLassoPoints([]);
  }, []);

  // Switching tools. A path and a box cannot be read as each other, so moving
  // in or out of the lasso drops both — which also keeps the image export from
  // holding a rectangle the viewer has stopped drawing. Rectangle and ellipse
  // keep the box: that is one drag read two ways, not a new selection.
  const chooseTool = useCallback((tool) => {
    if (tool === cropTool) return;
    if (tool === 'lasso' || cropTool === 'lasso') clearSelection();
    setCropTool(tool);
  }, [cropTool, clearSelection]);

  // Sample width for the crop-size estimate. /api/ebsd/datasets reports dtype
  // per dataset and is already fetched on load, so the estimate is right for a
  // uint16 detector from the first file — no extra request, no assumed 8 bits.
  // ebsdInfo.dtype is the fallback (applyGrid puts it there after a crop).
  const activeDtype = useMemo(
    () => datasets.find((d) => d.name === activeDataset)?.dtype || ebsdInfo?.dtype || null,
    [datasets, activeDataset, ebsdInfo],
  );

  const ovZoomed = isZoomed(ovView);
  const patZoomed = isZoomed(patView);
  // The ROI box and crosshair sit in an untransformed overlay so their border
  // and cross arms keep a constant on-screen size at any magnification; the
  // overlay is moved numerically instead. See ./zoomOverlay.js.
  const ovRectZoomed = useMemo(
    () => (ovImgRect ? zoomRectPct(ovImgRect, ovView) : null),
    [ovImgRect, ovView],
  );

  // --- Export ---------------------------------------------------------------
  // Short, human-recognisable stem for the suggested filename.
  const exportStem = useMemo(() => {
    const raw = filePath || activeDataset || 'ebsd';
    return raw.split(/[\\/]/).pop().replace(/\.[^.]+$/, '') || 'ebsd';
  }, [filePath, activeDataset]);

  const openExport = useCallback(async (target) => {
    // During a drag over the overview the pattern pane shows a ~1/4-resolution
    // atlas thumbnail. Exporting that would silently write a blurry file, so
    // fetch the real pattern before opening the dialog.
    if (target === 'pattern' && patternIsThumb) {
      await loadPattern(row, col);
    }
    setExportFor(target);
  }, [patternIsThumb, loadPattern, row, col]);

  const exportProps = useMemo(() => {
    if (!exportFor) return null;
    const units = stepSize?.units || 'µm';
    if (exportFor === 'overview') {
      const modeName = t(`overview.modeNames.${overviewMode}`, { defaultValue: overviewMode });
      return {
        src: overviewImage ? `data:image/png;base64,${overviewImage}` : null,
        title: t('overview.captionTitle'),
        defaultBaseName: `${exportStem}_${modeName}${gridShape ? `_${gridShape[0]}x${gridShape[1]}` : ''}`,
        // The overview carries no CSS correction — what is in the PNG is what
        // is on screen, so "as displayed" and "raw" are the same thing here.
        displayFilter: null,
        unitsPerPixel: stepSize?.x ?? null,
        unitLabel: units,
        annotations: {
          crosshair: gridShape ? { row, col } : null,
          roi: roi || null,
          label: `${exportStem} · ${modeName}${gridShape ? ` · ${gridShape[0]}×${gridShape[1]}` : ''}`,
        },
      };
    }
    return {
      src: pattern ? `data:image/png;base64,${pattern}` : null,
      title: t('pattern.captionTitle'),
      defaultBaseName: `${exportStem}_r${row}_c${col}`,
      // Brightness lives only in CSS on the <img>; mirror it so "as displayed"
      // really matches the screen.
      displayFilter: gamma !== 100 ? `brightness(${gamma / 100})` : null,
      // A detector pixel is not a length on the specimen — no scalebar.
      unitsPerPixel: null,
      unitLabel: units,
      annotations: {
        crosshair: null,
        roi: null,
        label: `${exportStem} · [${row}, ${col}]`,
      },
    };
  }, [exportFor, exportStem, overviewImage, overviewMode, gridShape, roi, row, col,
      pattern, gamma, stepSize, t]);

  // The active dataset's origin. Re-read on every dataset change because
  // switching moves between a crop and its parent.
  const refreshCropWindow = useCallback(async () => {
    try {
      const res = await ebsdApi.getCrop();
      setCropWindow(res.data?.window || null);
    } catch { setCropWindow(null); }
  }, []);

  useEffect(() => {
    if (!ebsdLoaded) { setCropWindow(null); return; }
    refreshCropWindow();
  }, [ebsdLoaded, activeDataset, refreshCropWindow]);

  // Publish a new navigation grid to BOTH places that hold one. They are
  // separate store fields — ebsdInfo (this page's own geometry) and gridShape
  // (written by setFileData, read by the status bar and, critically, by
  // PCRefinement, which turns row/col into flat backend indices with
  // `row * gridShape[1] + col`). The file-load path keeps them in step;
  // cropping is the first operation that changes the grid WITHOUT a load, so
  // it has to keep them in step itself. Updating only one leaves the pattern
  // centre — the value this application is most sensitive to — optimised
  // against the wrong pixels.
  const applyGrid = useCallback((geom) => {
    if (!geom) return false;
    setEBSDLoaded({
      ...(ebsdInfo || {}),
      grid_shape: geom.gridShape,
      navigation_shape: geom.navigationShape,
      pattern_shape: geom.patternShape || ebsdInfo?.pattern_shape,
      n_patterns: geom.patternCount,
      pattern_count: geom.patternCount,
      dtype: geom.dtype || ebsdInfo?.dtype,
    });
    setNavigationGrid({
      gridShape: geom.gridShape,
      patternShape: geom.patternShape,
      patternCount: geom.patternCount,
    });
    return true;
  }, [ebsdInfo, setEBSDLoaded, setNavigationGrid]);

  // Move the viewer onto a (possibly different) dataset's grid: publish it,
  // drop a selection drawn on the old one, and pull the cursor inside the new
  // bounds. Returns the position to load, because setRow/setCol are async.
  // Both paths that hand the viewer a dataset it did not choose a position
  // for — switching and deleting — go through here, so a grid change cannot
  // be forgotten in one of them again. handleCrop calls applyGrid directly
  // instead: it knows where to land (the origin of the cut-out) rather than
  // clamping the old position to an arbitrary corner of it.
  const adoptGrid = useCallback((geom) => {
    applyGrid(geom);
    // A selection drawn on one grid means nothing on another.
    clearSelection();
    const r = geom ? Math.min(row, geom.gridShape[0] - 1) : row;
    const c = geom ? Math.min(col, geom.gridShape[1] - 1) : col;
    setRow(r);
    setCol(c);
    return [r, c];
  }, [applyGrid, row, col, clearSelection]);

  // Cut the active dataset down to the drawn selection. The backend makes the
  // crop the active dataset, so everything on screen has to be re-read: the
  // dataset list, the metadata, the overview, the atlas and the pattern.
  // ebsdInfo goes with them — unlike a deepcopy, a crop CHANGES the navigation
  // grid, and a stale grid_shape would leave calcOverviewPos mapping every
  // click onto the parent's grid.
  const handleCrop = useCallback(async (payload) => {
    setCropBusy(true);
    try {
      const res = await ebsdApi.crop(payload);
      // NOT destructured as `window`: that shadows the global object this
      // file uses elsewhere (window.electronAPI, window.Image).
      const { name, n_selected: nSel, materialised, window: cropWin,
              n_patterns: nPx } = res.data || {};
      log(t('crop.done', { name, selected: nSel }));
      if (!materialised) log(t('crop.notMaterialised'));
      clearSelection();
      setActiveDataset(name);
      setCropWindow(cropWin || null);
      // The old position may lie outside the cut-out entirely.
      setRow(0);
      setCol(0);
      // The datasets list reads the grid off the live signal, so it is the
      // authority. The crop RESPONSE is the fallback (fetchDatasets swallows
      // its errors); the crop REQUEST is never used — it says what was asked
      // for, and only the backend's refusal-rather-than-clamp makes the two
      // agree today.
      const list = await fetchDatasets();
      const applied = applyGrid(datasetGeometry(list?.find((d) => d.name === name)));
      if (!applied && cropWin) {
        applyGrid({
          gridShape: [cropWin.rows, cropWin.cols],
          navigationShape: [cropWin.cols, cropWin.rows],
          patternShape: null,
          patternCount: nPx ?? cropWin.rows * cropWin.cols,
          dtype: null,
        });
      }
      await fetchMetadata();
      setOverviewImage(null);
      fetchOverview(overviewMode);
      fetchAtlas();
      loadPattern(0, 0);
    } catch (e) {
      log(t('crop.failed', { message: e?.response?.data?.detail || e?.message || String(e) }));
    } finally {
      setCropBusy(false);
    }
  }, [log, t, applyGrid, fetchDatasets, fetchMetadata,
      fetchOverview, fetchAtlas, overviewMode, loadPattern, clearSelection]);

  // Write the active crop to a file. The BACKEND writes it — the dialog only
  // supplies a destination — so this uses `saveFile` (returns a path), not
  // `saveImage` (shows a dialog AND writes renderer bytes).
  const doExportCrop = useCallback(async (path) => {
    setCropSaving(true);
    try {
      const res = await ebsdApi.exportCrop(path);
      const d = res.data || {};
      log(t('crop.saveCropDone', {
        points: (d.n_points ?? 0).toLocaleString(), path: d.path || path,
      }));
      // Say what actually went into the file. "Saved" alone leaves the user
      // to open it to find out whether the EDS maps and the electron image
      // came along.
      log(t('crop.saveCropDetail', {
        cut: d.datasets_cut ?? 0,
        copied: d.datasets_copied ?? 0,
        images: d.electron_images_cut ?? 0,
      }));
      if (d.electron_images_full > 0) {
        log(t('crop.saveCropElectronFull', { count: d.electron_images_full }));
      }
    } catch (e) {
      log(t('crop.saveCropFailed', {
        message: e?.response?.data?.detail || e?.message || String(e),
      }));
    } finally {
      setCropSaving(false);
    }
  }, [log, t]);

  const handleExportCrop = useCallback(async () => {
    // Suggest a name, but never invent a directory: the Electron dialog and
    // the prompt fallback both let the user say where it goes.
    const suggested = `${String(activeDataset || 'crop').replace(/[^\w.-]+/g, '_')}.h5oina`;
    if (window.electronAPI?.saveFile) {
      const selected = await window.electronAPI.saveFile({
        defaultPath: suggested,
        filters: [
          { name: 'Oxford H5OINA', extensions: ['h5oina'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (selected) doExportCrop(selected);
      return;
    }
    // Plain-browser mode (start_app.py): no native dialog, so ask for the
    // path the same way the file-open button does. It is the backend's
    // filesystem either way, so a typed path is as valid as a picked one.
    askPrompt({
      title: t('crop.savePromptTitle'),
      message: t('crop.savePromptMessage'),
      defaultValue: suggested,
      placeholder: t('crop.savePromptPlaceholder'),
      submitLabel: t('crop.savePromptSubmit'),
      onSubmit: (p) => { if (p && p.trim()) doExportCrop(p.trim()); },
    });
  }, [activeDataset, askPrompt, doExportCrop, t]);

  const switchDataset = async (name) => {
    try {
      await ebsdApi.selectDataset(name);
      setActiveDataset(name);
      // Before crops every dataset shared its parent's grid, so switching one
      // could never change it. A crop can, and leaving the old grid up
      // mis-maps every click, puts the crosshair in the wrong place, and lets
      // a new drag produce a window the backend refuses with a 400.
      const [r, c] = adoptGrid(datasetGeometry(datasets.find((d) => d.name === name)));
      await loadPattern(r, c);
      fetchOverview(overviewMode);
      fetchAtlas(); // re-build atlas for new dataset
      log(t('logMessages.switchedDataset', { name }));
    } catch (err) {
      log(t('logMessages.switchDatasetError', { error: err.response?.data?.detail || err.message }));
    }
  };

  const handleBrowse = async () => {
    if (window.electronAPI?.openFile) {
      const selected = await window.electronAPI.openFile({
        filters: [
          { name: 'EBSD Files', extensions: ['h5oina', 'h5', 'hdf5', 'up1', 'up2'] },
          { name: 'EDAX Patterns', extensions: ['up1', 'up2'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (selected) doLoadFile(selected);
    } else {
      askPrompt({
        title: t('load.promptTitle'),
        message: t('load.promptMessage'),
        placeholder: t('load.promptPlaceholder'),
        submitLabel: t('load.promptSubmit'),
        onSubmit: (path) => { if (path.trim()) doLoadFile(path.trim()); },
      });
    }
  };

  // Opens the same dialog the right-click menu uses. This button used to fire
  // an immediate, silent browser download with no format, location or crop —
  // keeping two different export behaviours around would only confuse.
  const handleExport = () => {
    if (!pattern) return;
    openExport('pattern');
  };

  // Add the currently displayed pattern to the PC Refinement list (used by
  // both the sidebar PC Refinement section and the pattern action footer).
  const handleAddCurrentPattern = async () => {
    try {
      const res = await pcApi.addPattern(row, col);
      if (res.data?.duplicate) {
        log(t('logMessages.patternDuplicate', { row, col }));
        return;
      }
      const n = res.data?.n_patterns || 0;
      setPcPatternCount(n);
      log(t('logMessages.addedPattern', { row, col, count: n }));
      window.dispatchEvent(new CustomEvent('pc-patterns-updated'));
      if (n === 1) onNavigate?.('pcrefinement');
    } catch (err) {
      log(t('logMessages.addPatternError', { error: err.response?.data?.detail || err.message }));
    }
  };

  // Short, distinguishable labels for the loaded-files switcher (common
  // prefix/suffix stripped — see disambiguateNames).
  const loadedFileLabels = useMemo(
    () => disambiguateNames(loadedFiles.map((f) => f.name || '')),
    [loadedFiles]
  );

  const quickModes = [
    { label: 'σ',   mode: 'Std Dev (Quality)',     tip: t('quickModes.stdDevTooltip') },
    { label: 'BC',  mode: 'Band Contrast',          tip: t('quickModes.bandContrastTooltip') },
    { label: '∇',   mode: 'Sharpness (Laplacian)',  tip: t('quickModes.sharpnessTooltip') },
    { label: 'SNR', mode: 'SNR (MAD)',              tip: t('quickModes.snrTooltip') },
    { label: 'NCC', mode: 'Neighbor Correlation',   tip: t('quickModes.neighborCorrelationTooltip') },
    { label: 'H',   mode: 'Entropy',                tip: t('quickModes.entropyTooltip') },
  ];

  // =========================================================================
  // Sub-panels
  // =========================================================================

  /** Left panel content — wrapped in ScrollPanel */
  const leftPanel = (
    <ScrollPanel style={{ background: colors.bgSecondary, padding: spacing.groupMargin }}>

      {/* Loaded Files — multi-file switcher (only shown when more than 1 file loaded) */}
      {loadedFiles.length > 1 && (
        <GroupBox title={t('loadedFiles.title', { count: loadedFiles.length })}>
          <div className="thin-scrollbar" style={{
            maxHeight: 140, overflowY: 'auto',
            border: `1px solid ${colors.border}`, borderRadius: 3,
            background: colors.bg,
          }}>
            {loadedFiles.map((f, idx) => {
              const isActive = !!f.active;
              // Show only the part of the name that distinguishes this file
              // from the others (common prefix/suffix stripped). Full path is
              // in the title tooltip.
              const label = loadedFileLabels[idx] || f.name;
              return (
                <div
                  key={f.path}
                  className="list-item-interactive"
                  onClick={() => handleSwitchFile(f.path)}
                  title={isActive ? t('loadedFiles.fileTooltipActive', { path: f.path }) : t('loadedFiles.fileTooltip', { path: f.path })}
                  style={{
                    padding: '5px 8px', cursor: fileSwitching ? 'wait' : 'pointer',
                    fontSize: '9pt',
                    background: isActive ? alpha(colors.purple, 13) : 'transparent',
                    color: isActive ? colors.purple : colors.cyan,
                    fontWeight: isActive ? 'bold' : 'normal',
                    borderBottom: `1px solid ${colors.border}`,
                    borderLeft: isActive ? `3px solid ${colors.purple}` : '3px solid transparent',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6,
                    transition: 'background 0.1s, border-left-color 0.15s',
                    opacity: fileSwitching && !isActive ? 0.5 : 1,
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                    {isActive ? '● ' : '○ '}{label}
                  </span>
                  {fileSwitching && !isActive ? (
                    <span style={{ fontSize: '8pt', color: colors.textSecondary }}>…</span>
                  ) : !isActive ? (
                    <button
                      type="button"
                      title={t('loadedFiles.removeTooltip')}
                      onClick={(e) => { e.stopPropagation(); handleRemoveFile(f.path); }}
                      style={{
                        flex: '0 0 auto', border: 'none', background: 'transparent',
                        color: colors.textSecondary, cursor: 'pointer', fontSize: '10pt',
                        lineHeight: 1, padding: '0 2px',
                      }}
                    >✕</button>
                  ) : null}
                </div>
              );
            })}
          </div>
          {fileSwitching ? (
            <div style={{ fontSize: '8.5pt', color: colors.cyan, marginTop: 4, paddingLeft: 4 }}>
              {t('loadedFiles.switching')}
            </div>
          ) : loadedFiles.length > 1 && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6, marginTop: 4 }}>
              <button
                type="button"
                onClick={handleClearFiles}
                title={t('loadedFiles.clearListTooltip')}
                style={{
                  border: `1px solid ${colors.border}`, background: 'transparent',
                  color: colors.textSecondary, cursor: 'pointer', fontSize: '8.5pt',
                  borderRadius: 3, padding: '2px 8px',
                }}
              >{t('loadedFiles.clearList')}</button>
              <button
                type="button"
                onClick={handleClearAllFiles}
                title={t('loadedFiles.clearAllTooltip')}
                style={{
                  border: `1px solid ${colors.border}`, background: 'transparent',
                  color: colors.textSecondary, cursor: 'pointer', fontSize: '8.5pt',
                  borderRadius: 3, padding: '2px 8px',
                }}
              >{t('loadedFiles.clearAll')}</button>
            </div>
          )}
        </GroupBox>
      )}

      {/* Datasets (within active file — used by deepcopy / derived datasets) */}
      <GroupBox title={datasets.length ? t('datasets.titleWithCount', { count: datasets.length }) : t('datasets.title')}>
        {/* List */}
        <div className="thin-scrollbar" style={{
          maxHeight: 120, overflowY: 'auto', marginBottom: spacing.innerSpacing,
          border: `1px solid ${colors.border}`, borderRadius: 3,
          background: colors.bg,
        }}>
          {datasets.length === 0 ? (
            <div style={{ padding: '10px 8px', fontSize: '9pt', color: colors.textSecondary, textAlign: 'center' }}>
              <span style={{ opacity: 0.4, marginRight: 4 }}>{'\u25A2'}</span>
              {t('datasets.empty')}
            </div>
          ) : datasets.map((ds) => {
            // PyQt5 parity: originals cyan/bold, derivatives purple with └ prefix
            const isDerivative = /_(?:bg_dyn|bg_stat|avg|autocontrast|ac)/.test(ds.name);
            const isSelected = ds.name === activeDataset;
            const sizeStr = ds.navigation_shape
              ? `${ds.navigation_shape[1]}×${ds.navigation_shape[0]}`
              : '';
            return (
              <div
                key={ds.name}
                className="list-item-interactive"
                onClick={() => switchDataset(ds.name)}
                title={isDerivative
                  ? t('datasets.derivedTooltip', { name: ds.name })
                  : (sizeStr
                    ? t('datasets.datasetTooltipWithSize', { name: ds.name, size: sizeStr })
                    : t('datasets.datasetTooltip', { name: ds.name }))}
                style={{
                  padding: '4px 8px', cursor: 'pointer', fontSize: '9pt',
                  background: isSelected ? alpha(colors.purple, 13) : 'transparent',
                  color: isDerivative ? colors.purple : colors.cyan,
                  fontWeight: isDerivative ? 'normal' : 'bold',
                  borderBottom: `1px solid ${colors.border}`,
                  borderLeft: isSelected ? `3px solid ${colors.purple}` : '3px solid transparent',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  transition: 'background 0.1s, border-left-color 0.15s',
                }}
              >
                <span>{isDerivative ? `  └ ${ds.name}` : ds.name}</span>
                <span style={{ fontSize: '8pt', color: colors.textSecondary }}>
                  {sizeStr}
                </span>
              </div>
            );
          })}
        </div>

        {/* Deepcopy checkbox + remove button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <CheckRow
            checked={deepcopy}
            onChange={setDeepcopy}
            label={t('datasets.deepcopyLabel')}
            title={t('datasets.deepcopyTooltip')}
          />
          <Button
            variant="danger"
            small
            title={t('datasets.removeTooltip')}
            style={{ marginLeft: 'auto', padding: '2px 6px', fontSize: '9pt', minWidth: 28 }}
            onClick={() => {
              if (!activeDataset) return;
              askConfirm({
                title: t('datasets.removeConfirmTitle'),
                message: t('datasets.removeConfirmMessage', { name: activeDataset }),
                confirmLabel: t('datasets.removeConfirmLabel'),
                onConfirm: async () => {
                  try {
                    const res = await ebsdApi.deleteDataset(activeDataset);
                    log(t('logMessages.removedDataset', { name: activeDataset }));
                    // The backend hands back a DIFFERENT dataset, which
                    // since crops exist can be on a different grid — same
                    // reason switchDataset has to adopt it.
                    const newActive = res.data?.active || '';
                    setActiveDataset(newActive);
                    const list = await fetchDatasets();
                    if (newActive) {
                      const [r, c] = adoptGrid(
                        datasetGeometry(list?.find((d) => d.name === newActive)));
                      await loadPattern(r, c);
                      fetchOverview(overviewMode);
                      fetchAtlas();
                    }
                  } catch (err) {
                    log(t('logMessages.removeDatasetError', { error: err.response?.data?.detail || err.message }));
                  }
                },
              });
            }}
          >
            ✖
          </Button>
        </div>
      </GroupBox>

      {/* Load Data */}
      <GroupBox title={t('load.title')}>
        <input
          ref={fileInputRef}
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') doLoadFile(filePath); }}
          placeholder={t('load.pathPlaceholder')}
          title={t('hoverTips.filePathInput')}
          style={{
            width: '100%', boxSizing: 'border-box',
            background: colors.bg, border: `1px solid ${colors.border}`,
            borderRadius: 4, color: colors.text, fontSize: '10pt',
            padding: '4px 8px', height: spacing.buttonHeight, outline: 'none',
            marginBottom: spacing.innerSpacing,
          }}
        />
        <div style={{ display: 'flex', gap: spacing.innerSpacing, marginBottom: spacing.innerSpacing }}>
          <Button
            variant="primary"
            onClick={() => doLoadFile(filePath)}
            disabled={loadLoading || !filePath.trim()}
            title={t('load.loadDataTooltip')}
            style={{ flex: 1 }}
          >
            {loadLoading ? <span className="btn-loading">{t('load.loading')}</span> : t('load.loadData')}
          </Button>
          <Button
            onClick={handleBrowse}
            title={t('load.browseTooltip')}
            style={{ padding: '5px 10px' }}
          >
            {t('load.browse')}
          </Button>
          <Button
            onClick={() => setShowRecent(!showRecent)}
            disabled={recentFiles.length === 0}
            title={t('load.recentTooltip', { count: recentFiles.length })}
            small
          >
            {t('load.recent')}
          </Button>
        </div>
        {showRecent && recentFiles.length > 0 && (
          <div style={{
            background: colors.bg, border: `1px solid ${colors.border}`,
            borderRadius: 4, marginBottom: spacing.innerSpacing,
          }}>
            {recentFiles.map((f, i) => (
              <div
                key={i}
                onClick={() => { setShowRecent(false); doLoadFile(f); }}
                title={f}
                style={{
                  padding: '4px 8px', fontSize: '9pt', cursor: 'pointer', color: colors.text,
                  borderBottom: i < recentFiles.length - 1 ? `1px solid ${colors.border}` : 'none',
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = colors.sidebarActive}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                {f.split(/[\\/]/).pop()}
              </div>
            ))}
          </div>
        )}
        {loadError && (
          <div role="alert" style={{ fontSize: '9pt', color: colors.red, marginTop: 4, display: 'flex', alignItems: 'flex-start', gap: 4, animation: 'fadeSlideIn 0.2s ease-out' }}>
            <span style={{ flexShrink: 0, fontSize: '10pt', lineHeight: 1 }}>{'\u26A0'}</span>
            <span>{loadError}</span>
          </div>
        )}
      </GroupBox>

      {/* Detector Signal Mask */}
      <GroupBox title={t('mask.title')}>
        <div style={{ marginBottom: spacing.innerSpacing }}>
          <CheckRow
            checked={maskEnabled}
            onChange={(v) => setMaskEnabled(v)}
            label={t('mask.circularLabel')}
            title={t('mask.circularTooltip')}
          />
        </div>
        {ebsdInfo?.format_type && (
          <div style={{
            fontSize: '8pt', color: colors.textSecondary,
            marginBottom: spacing.innerSpacing,
            padding: '2px 6px',
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
          }}>
            {t('mask.fileVendor')} <span style={{
              color: ebsdInfo.format_type === 'EDAX' ? colors.green : colors.yellow,
              fontWeight: 600,
            }}>{ebsdInfo.format_type}</span>
            {' — '}
            {ebsdInfo.format_type === 'EDAX'
              ? t('mask.vendorRecommended')
              : t('mask.vendorDiscard')}
          </div>
        )}
        <FormRow label={t('mask.radiusLabel')}>
          <Slider
            value={maskRadius}
            onChange={setMaskRadius}
            min={50}
            max={100}
            step={1}
            displayValue={`${maskRadius}%`}
            title={t('mask.radiusTooltip')}
            style={{ opacity: maskEnabled ? 1 : 0.4, pointerEvents: maskEnabled ? 'auto' : 'none' }}
          />
        </FormRow>
      </GroupBox>

      {/* Pattern Processing */}
      <GroupBox title={t('processing.title')}>
        {/* BG removal row */}
        <div style={{ display: 'flex', gap: spacing.innerSpacing, marginBottom: spacing.innerSpacing }}>
          <Button
            onClick={() => doProcessing('bg_dyn', t('logMessages.labelBgDynamic'))}
            disabled={!ebsdLoaded || processingBusy}
            title={t('processing.bgDynamicTooltip')}
            style={{ flex: 1 }}
            small
          >
            {t('processing.bgDynamic')}
          </Button>
          <Button
            onClick={() => doProcessing('bg_stat', t('logMessages.labelBgStatic'))}
            disabled={!ebsdLoaded || processingBusy}
            title={t('processing.bgStaticTooltip')}
            style={{ flex: 1 }}
            small
          >
            {t('processing.bgStatic')}
          </Button>
        </div>

        {/* Frame average row */}
        <FormRow label={t('processing.windowSizeLabel')}>
          <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
            <NumberInput
              value={windowSize}
              onChange={(e) => setWindowSize(Number(e.target.value))}
              min={1}
              max={50}
              title={t('processing.windowSizeTooltip')}
              style={{ width: 60 }}
            />
            <Button
              onClick={() => doProcessing('avg', t('logMessages.labelFrameAverage'))}
              disabled={!ebsdLoaded || processingBusy}
              title={t('processing.frameAverageTooltip')}
              small
              style={{ flex: 1 }}
            >
              {t('processing.frameAverage')}
            </Button>
          </div>
        </FormRow>

        {/* Autocontrast */}
        <Button
          onClick={() => doProcessing('autocontrast', t('logMessages.labelAutoContrast'))}
          disabled={!ebsdLoaded || processingBusy}
          title={t('processing.autoContrastTooltip')}
          small
          style={{ width: '100%', marginBottom: spacing.innerSpacing }}
        >
          {t('processing.autoContrast')}
        </Button>

        {/* CLAHE — Adaptive Histogram Equalization */}
        <FormRow label={(
          <span style={{ display: 'inline-flex', alignItems: 'center' }}>
            {t('processing.claheKernelLabel')}
            <InfoTooltip>
              <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
                {t('processing.claheTooltipTitle')}
              </div>
              <p style={{ margin: '4px 0' }}>
                {t('processing.claheTooltipP1')}
              </p>
              <p style={{ margin: '4px 0' }}>
                {t('processing.claheTooltipP2')}
              </p>
              <div style={{ fontWeight: 700, marginTop: 6, marginBottom: 2 }}>
                {t('processing.claheTooltipChooseTitle')}
              </div>
              <ul style={{ margin: 0, paddingLeft: 16 }}>
                <li>{t('processing.claheTooltipChoose1')}</li>
                <li>{t('processing.claheTooltipChoose2')}</li>
                <li>{t('processing.claheTooltipChoose3')}</li>
                <li>{t('processing.claheTooltipChoose4')}</li>
              </ul>
              <p style={{ margin: '6px 0 0 0', fontSize: 10,
                color: colors.textSecondary }}>
                {t('processing.claheTooltipRule')}
              </p>
            </InfoTooltip>
          </span>
        )}>
          <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
            <NumberInput
              value={claheKernel}
              onChange={(e) => setClaheKernel(Number(e.target.value))}
              min={2}
              max={64}
              title={t('hoverTips.claheKernel')}
              style={{ width: 60 }}
            />
            <Button
              onClick={() => doProcessing('clahe', t('logMessages.labelClahe'))}
              disabled={!ebsdLoaded || processingBusy}
              small
              title={t('hoverTips.claheRun')}
              style={{ flex: 1 }}
            >
              {t('processing.clahe')}
            </Button>
          </div>
        </FormRow>

        {/* One-click recommended pipeline */}
        <div style={{
          display: 'flex', gap: spacing.innerSpacing, alignItems: 'center',
          marginBottom: spacing.innerSpacing,
        }}>
          <Button
            onClick={() => doProcessing('pipeline', t('logMessages.labelRecommendedPipeline'))}
            disabled={!ebsdLoaded || processingBusy}
            small
            title={t('hoverTips.pipelineRun')}
            style={{ flex: 1 }}
          >
            {t('processing.recommendedPipeline')}
          </Button>
          <InfoTooltip>
            <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
              {t('processing.recommendedPipelineTooltipTitle')}
            </div>
            <p style={{ margin: '4px 0' }}>
              {t('processing.recommendedPipelineTooltipP1')}
            </p>
            <ol style={{ margin: '4px 0', paddingLeft: 18 }}>
              <li>{t('processing.recommendedPipelineTooltipStep1')}</li>
              <li>{t('processing.recommendedPipelineTooltipStep2')}</li>
              <li>{t('processing.recommendedPipelineTooltipStep3')}</li>
            </ol>
            <p style={{ margin: '4px 0', fontSize: 10,
              color: colors.textSecondary }}>
              {t('processing.recommendedPipelineTooltipNote')}
            </p>
          </InfoTooltip>
        </div>

        {/* Brightness slider (client-side CSS filter — not true gamma) */}
        <FormRow label={t('processing.brightnessLabel')}>
          <Slider
            value={gamma}
            onChange={setGamma}
            min={10}
            max={300}
            title={t('processing.brightnessTooltip')}
            displayValue={(gamma / 100).toFixed(1)}
          />
        </FormRow>

        {/* Filter */}
        <FormRow label={t('processing.filterLabel')}>
          <Select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            title={t('processing.filterTooltip')}
            options={['None', 'Sobel', 'Canny', 'Difference', 'FFT Highpass', 'CLAHE', 'Sharpen', 'Denoise']}
            style={{ width: '100%' }}
          />
        </FormRow>

        {/* Interpolation */}
        <FormRow label={t('processing.interpLabel')}>
          <Select
            value={interpolation}
            onChange={(e) => setInterpolation(e.target.value)}
            title={t('processing.interpTooltip')}
            options={['nearest', 'bilinear', 'bicubic', 'lanczos']}
            style={{ width: '100%' }}
          />
        </FormRow>

        <Separator />

        {/* Batch auto-contrast */}
        <Button
          onClick={() => {
            askConfirm({
              title: t('processing.batchConfirmTitle'),
              message: t('processing.batchConfirmMessage'),
              confirmLabel: t('processing.batchConfirmLabel'),
              variant: 'warning',
              onConfirm: () => doProcessing('batch', t('logMessages.labelBatchAutoContrast')),
            });
          }}
          disabled={!ebsdLoaded || processingBusy}
          title={t('processing.batchAutoContrastTooltip')}
          style={{ width: '100%', marginBottom: spacing.innerSpacing }}
          small
        >
          {t('processing.batchAutoContrast')}
        </Button>

        {/* Export pattern */}
        <Button
          onClick={handleExport}
          disabled={!pattern}
          title={t('processing.exportPatternTooltip')}
          style={{ width: '100%', marginBottom: spacing.innerSpacing }}
          small
        >
          {t('processing.exportPattern')}
        </Button>

        {/* Reset processing */}
        {(gamma !== 100 || filter !== 'None' || interpolation !== 'nearest') && (
          <Button
            onClick={() => { setGamma(100); setFilter('None'); setInterpolation('nearest'); }}
            title={t('processing.resetProcessingTooltip')}
            style={{ width: '100%', marginBottom: spacing.innerSpacing }}
            small
          >
            {t('processing.resetProcessing')}
          </Button>
        )}

        {/* Compare datasets */}
        {datasets.length >= 2 && (
          <Button
            onClick={() => setCompareOpen(!compareOpen)}
            title={t('processing.compareDatasetsTooltip')}
            style={{ width: '100%' }}
            small
          >
            {compareOpen ? t('processing.hideCompare') : <>{`${t('processing.compareDatasets')} `}<span className="kbd">Ctrl+K</span></>}
          </Button>
        )}
      </GroupBox>

      {/* EDS Composition */}
      <GroupBox title={t('eds.title')}>
        <FormRow label={t('eds.modeLabel')}>
          <Select
            value={edsMode}
            onChange={(e) => { setEdsMode(e.target.value); fetchEds(row, col); }}
            title={t('eds.modeTooltip')}
            options={['Counts', 'Wt.%', 'At.%']}
            style={{ width: '100%' }}
          />
        </FormRow>
        <div style={{ marginBottom: spacing.innerSpacing }}>
          <CheckRow
            checked={showEdsOverlay}
            onChange={setShowEdsOverlay}
            label={t('eds.showOverlay')}
            title={t('eds.showOverlayTooltip')}
          />
        </div>

        {/* Composition display */}
        <div style={{
          padding: '6px 8px', border: `1px solid ${colors.border}`, borderRadius: 3,
          fontSize: '9pt', color: colors.textSecondary, marginBottom: spacing.innerSpacing,
          minHeight: 28, wordBreak: 'break-word',
        }}>
          {edsComposition?.data
            ? Object.entries(edsComposition.data).map(([el, vals]) => {
                const modeKey = { 'Counts': 'counts', 'Wt.%': 'wt_pct', 'At.%': 'at_pct' }[edsMode] || 'at_pct';
                const v = vals[modeKey] ?? vals.counts ?? 0;
                return `${el}: ${typeof v === 'number' ? v.toFixed(1) : v}${edsMode !== 'Counts' ? '%' : ''}`;
              }).join(' | ')
            : t('eds.noData')}
        </div>

        {/* Phase suggestions */}
        {(edsPhases?.suggestions || edsPhases?.phases) && (
          <div style={{
            padding: '4px 8px', fontSize: '9pt', color: colors.cyan,
            fontStyle: 'italic', marginBottom: spacing.innerSpacing,
            animation: 'fadeSlideIn 0.2s ease-out',
          }}>
            {t('eds.phasesLabel')} {(edsPhases.suggestions || edsPhases.phases || []).map((p) => {
              const name = p.name || p.phase || t('eds.unknownPhase');
              const score = p.score != null
                ? `${(p.score * 100).toFixed(0)}%`
                : (p.confidence != null ? `${p.confidence}%` : '');
              return score ? `${name} (${score})` : name;
            }).join(' | ')}
          </div>
        )}

        {/* Send filter to indexing */}
        <Button
          disabled={!edsComposition?.data}
          onClick={() => {
            log(t('logMessages.phaseFilterSent'));
            setPendingChemMask(true);
            onNavigate?.('indexing');
          }}
          title={t('eds.sendFilterTooltip')}
          style={{ width: '100%' }}
        >
          {t('eds.sendFilter')}
        </Button>
      </GroupBox>

      {/* PC Refinement — highlighted workflow section */}
      <div style={{
        border: `1px solid ${alpha(colors.purple, 55)}`,
        borderRadius: 6,
        background: alpha(colors.purple, 8),
        padding: spacing.innerMargin,
      }}>
        <div style={{
          fontSize: '10pt', fontWeight: 700, color: colors.purple, marginBottom: 6,
        }}>
          {t('pcRefinement.title')}
        </div>
        <div style={{
          fontSize: '8.5pt', color: colors.textSecondary, lineHeight: 1.5,
          marginBottom: spacing.innerSpacing,
        }}>
          {t('pcRefinement.description')}
        </div>
        <Button
          variant="purple"
          onClick={handleAddCurrentPattern}
          disabled={!ebsdLoaded}
          title={t('pcRefinement.addCurrentPatternTooltip')}
          style={{ width: '100%', marginBottom: spacing.innerSpacing }}
        >
          {t('pcRefinement.addCurrentPattern')}
        </Button>
        <Button
          onClick={() => onNavigate?.('pcrefinement')}
          title={t('pcRefinement.openPcRefinementTooltip')}
          style={{ width: '100%' }}
        >
          {t('pcRefinement.openPcRefinement')}
          {pcPatternCount > 0 && (
            <span style={{
              marginLeft: 6, background: colors.purple, color: colors.textOnAccent,
              fontSize: '8pt', fontWeight: 700, borderRadius: 8, padding: '1px 6px',
            }}>
              {pcPatternCount}
            </span>
          )}
        </Button>
      </div>
    </ScrollPanel>
  );

  // -------------------------------------------------------------------------
  // Overview + Pattern panel (top of right vertical splitter)
  // -------------------------------------------------------------------------
  const plotPanel = (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Overview mode toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: spacing.innerSpacing,
        padding: `${spacing.compactMargin}px ${spacing.innerMargin}px`,
        background: colors.bgSecondary, borderBottom: `1px solid ${colors.border}`,
        flexShrink: 0, flexWrap: 'wrap',
      }}>
        <Label secondary small>{t('overview.label')}</Label>
        <Select
          value={overviewMode}
          onChange={(e) => { setOverviewMode(e.target.value); fetchOverview(e.target.value); }}
          options={[
            'Mean Intensity', 'Std Dev (Quality)', 'Max Intensity',
            'Band Contrast', 'Sharpness (Laplacian)', 'SNR (MAD)',
            'Neighbor Correlation', 'Entropy',
          ].map((m) => ({ value: m, label: t(`overview.modeNames.${m}`, { defaultValue: m }) }))}
          title={t('overview.modeTooltip')}
          style={{ width: 180 }}
        />

        {/* Quality mask */}
        <CheckRow
          checked={qualityMask}
          onChange={(v) => setQualityMask(v)}
          label={t('overview.maskLabel')}
          title={t('overview.maskTooltip')}
        />
        <Slider
          value={qualityThreshold}
          onChange={setQualityThreshold}
          min={0}
          max={100}
          title={t('overview.thresholdTooltip')}
          displayValue={`${qualityThreshold}%`}
          // disabled when mask is off
          style={{ opacity: qualityMask ? 1 : 0.4, pointerEvents: qualityMask ? 'auto' : 'none', minWidth: 80, maxWidth: 80 }}
        />

        <div style={{ flex: 1 }} />
      </div>

      {/* Quick mode buttons row */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 3,
        padding: `2px ${spacing.innerMargin}px`,
        background: colors.bgSecondary, borderBottom: `1px solid ${colors.border}`,
        flexShrink: 0,
      }}>
        {quickModes.map((qm) => (
          <QuickModeBtn
            key={qm.label}
            label={qm.label}
            active={overviewMode === qm.mode}
            onClick={() => { setOverviewMode(qm.mode); fetchOverview(qm.mode); }}
            title={qm.tip}
          />
        ))}
        <div style={{ flex: 1 }} />
      </div>

      {/* Overview + Pattern side by side. Each column has a caption strip
          ABOVE the image (outside the data area) and the image fills the
          rest. Padding reduced from 14 → 4 so the data uses the space. */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>

        {/* Overview column */}
        <div style={{
          flex: 1, minWidth: 0, minHeight: 0,
          display: 'flex', flexDirection: 'column',
          background: colors.bgSecondary,
        }}>
          {/* Caption strip — outside the image */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '2px 8px', fontSize: '8pt',
            color: colors.textSecondary,
            background: colors.bg, borderBottom: `1px solid ${colors.border}`,
            flexShrink: 0, minHeight: 18, whiteSpace: 'nowrap', overflow: 'hidden',
          }}>
            <span style={{ color: colors.cyan, fontWeight: 600 }}>{t('overview.captionTitle')}</span>
            <span>· {t(`overview.modeNames.${overviewMode}`, { defaultValue: overviewMode })}</span>
            {overviewMode === 'Band Contrast' && overviewSource && (
              <span style={{
                padding: '0 6px', borderRadius: 3,
                background: colors.bgSecondary, border: `1px solid ${colors.border}`,
                color: overviewSource === 'native' ? colors.green : colors.yellow,
                fontWeight: 600,
              }}>
                {t(qualityProvenanceKey(overviewSource))}
              </span>
            )}
            {gridShape && <span>· {gridShape[0]}×{gridShape[1]}</span>}
            {ovZoomed && (
              <span data-overview-zoom-badge style={{ color: colors.accent, fontWeight: 600 }}>
                · {ovView.scale.toFixed(1)}×
              </span>
            )}
            <span style={{ flex: 1 }} />
            <span style={{ opacity: 0.7 }}>{t('overview.captionHint')}</span>
          </div>

          {/* Selection tool — which shape a Shift+Drag draws. Shown on the
              same condition as the crop panel it feeds, so a viewer with no
              file loaded is untouched. */}
          {ebsdLoaded && (
            <div
              data-crop-tools
              style={{
                display: 'flex', alignItems: 'center', gap: 3,
                padding: '2px 8px', fontSize: '8pt', color: colors.textSecondary,
                background: colors.bgSecondary, borderBottom: `1px solid ${colors.border}`,
                flexShrink: 0,
              }}
            >
              <span style={{ marginRight: 3 }}>{t('crop.toolLabel')}</span>
              {['rect', 'ellipse', 'lasso'].map((tool) => (
                <QuickModeBtn
                  key={tool}
                  label={t(`crop.tool.${tool}`)}
                  active={cropTool === tool}
                  onClick={() => chooseTool(tool)}
                  title={t(`crop.toolHint.${tool}`)}
                />
              ))}
              <div style={{ flex: 1 }} />
            </div>
          )}

        {/* Image container */}
        <div
          ref={overviewRef}
          style={{
            flex: 1, minHeight: 0, position: 'relative',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: ebsdLoaded ? (ovZoomed ? 'grab' : 'crosshair') : 'default',
            overflow: 'hidden', padding: 4,
          }}
          onClick={(e) => {
            // A drag that panned must not also navigate. Short clicks (below
            // CLICK_SLOP_PX) never set this, so click-to-navigate keeps working
            // while zoomed in.
            if (ovSuppressClickRef.current) { ovSuppressClickRef.current = false; return; }
            if (!e.shiftKey) handleOverviewPointer(e);
          }}
          onMouseDown={(e) => {
            if (e.shiftKey && gridShape) {
              const pos = calcOverviewPos(e);
              if (pos) {
                // A new drag replaces the old selection rather than adding to
                // it — for the lasso that means starting a fresh path, not
                // continuing the previous one.
                roiStartRef.current = { row: pos.r, col: pos.c };
                if (cropTool === 'lasso') {
                  setLassoPoints([{ r: pos.r, c: pos.c }]);
                  setLassoDrawing(true);
                }
              }
            } else if (ovZoomed && e.button === 0) {
              // Zoomed in, a plain drag pans. Navigation is left to onClick so
              // that a pan does not scrub through patterns on the way.
              ovPanRef.current = { x: e.clientX, y: e.clientY, moved: 0 };
              ovSuppressClickRef.current = false;
            } else {
              handleOverviewPointer(e);
            }
          }}
          onMouseUp={() => {
            ovPanRef.current = null;
            roiStartRef.current = null;
            loadPattern(row, col);
            fetchEds(row, col);
          }}
          onMouseMove={handleOverviewMouseMove}
          onMouseLeave={() => { ovPanRef.current = null; }}
          onDoubleClick={() => setOvView(IDENTITY_VIEW)}
          onContextMenu={(e) => {
            e.preventDefault();
            setContextMenu({ x: e.clientX, y: e.clientY, target: 'overview' });
          }}
        >
          <div
            ref={ovBoxRef}
            data-overview-box
            style={{
              position: 'relative',
              width: '100%', height: '100%',
              border: `1px solid ${colors.border}`, borderRadius: 6, overflow: 'hidden',
              background: '#000',
            }}
          >
          {overviewImage ? (
            <div style={{ position: 'relative', width: '100%', height: '100%' }}>
              {/* Zoom layer — a CSS transform only, so the overview PNG is
                  never re-fetched or re-rendered by a zoom. */}
              <div data-overview-zoom-layer style={{
                position: 'absolute', inset: 0,
                transform: viewToTransform(ovView),
                transformOrigin: '50% 50%',
                willChange: ovZoomed ? 'transform' : 'auto',
              }}>
                <img
                  className="image-reveal"
                  src={`data:image/png;base64,${overviewImage}`}
                  alt={t('overview.imageAlt')}
                  draggable={false}
                  onDragStart={(e) => e.preventDefault()}
                  onLoad={updateOvImgRect}
                  style={{ width: '100%', height: '100%', display: 'block', objectFit: 'contain', imageRendering: 'pixelated', userSelect: 'none' }}
                />
              </div>
              {/* Overlay positioned exactly over the visible image area.
                  Deliberately OUTSIDE the zoom transform: its ROI border and
                  crosshair arms are absolute pixel sizes that must not grow
                  with the magnification, so the rect is moved numerically
                  through the very same view instead. */}
              {ovRectZoomed && (
                <div style={{
                  position: 'absolute', pointerEvents: 'none',
                  left: `${ovRectZoomed.left}%`, top: `${ovRectZoomed.top}%`,
                  width: `${ovRectZoomed.width}%`, height: `${ovRectZoomed.height}%`,
                }}>
                  {/* ROI selection box — the rectangle, and the ellipse
                      inscribed in the same drag. One shape drawn two ways,
                      because that is exactly how the mask reads it. */}
                  {roi && gridShape && cropTool !== 'lasso' && (
                    <div data-roi-box style={{
                      position: 'absolute',
                      left: `${(roi.startCol / gridShape[1]) * 100}%`,
                      top: `${(roi.startRow / gridShape[0]) * 100}%`,
                      width: `${((roi.endCol - roi.startCol) / gridShape[1]) * 100}%`,
                      height: `${((roi.endRow - roi.startRow) / gridShape[0]) * 100}%`,
                      border: `2px solid ${colors.yellow}`,
                      background: alpha(colors.yellow, 8),
                      borderRadius: cropTool === 'ellipse' ? '50%' : 0,
                    }}>
                      <span style={{
                        position: 'absolute', top: -14, left: 0,
                        fontSize: '7pt', color: colors.yellow, whiteSpace: 'nowrap',
                      }}>
                        {t('overview.roiLabel', { startRow: roi.startRow, startCol: roi.startCol, endRow: roi.endRow, endCol: roi.endCol })}
                      </span>
                    </div>
                  )}
                  {/* The traced lasso path. An <svg> in this same untransformed
                      overlay rather than a second coordinate system: the
                      percentages below are the ones the rectangle above
                      already uses, so the zoom mathematics is shared.
                      Drawn CLOSED because lassoMask closes the path too — what
                      is outlined is what would be cropped on mouse-up.
                      non-scaling-stroke keeps the line one constant width
                      despite preserveAspectRatio="none" stretching the box. */}
                  {cropTool === 'lasso' && gridShape && lassoPoints.length > 0 && (
                    <svg
                      data-lasso-path
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                      style={{
                        position: 'absolute', inset: 0,
                        width: '100%', height: '100%', overflow: 'visible',
                      }}
                    >
                      <polygon
                        points={lassoPoints.map((p) => (
                          `${((p.c + 0.5) / gridShape[1]) * 100},${((p.r + 0.5) / gridShape[0]) * 100}`
                        )).join(' ')}
                        fill={alpha(colors.yellow, 8)}
                        stroke={colors.yellow}
                        strokeWidth={2}
                        strokeLinejoin="round"
                        vectorEffect="non-scaling-stroke"
                      />
                    </svg>
                  )}
                  {/* Crosshair */}
                  {crosshair && (
                    <div data-overview-crosshair style={{
                      position: 'absolute', left: `${crosshair.x}%`, top: `${crosshair.y}%`,
                      transform: 'translate(-50%, -50%)',
                    }}>
                      <div style={{ width: 14, height: 2, background: colors.red, position: 'absolute', top: -1, left: -7 }} />
                      <div style={{ width: 2, height: 14, background: colors.red, position: 'absolute', top: -7, left: -1 }} />
                    </div>
                  )}
                </div>
              )}
              {overviewSampled && (
                <div style={{
                  position: 'absolute', bottom: 4, right: 4, pointerEvents: 'none',
                  fontSize: '7pt', color: colors.yellow,
                  background: alpha('#000', 55), padding: '1px 4px', borderRadius: 3,
                }}>
                  {t('overview.sampledPreview')}
                </div>
              )}
            </div>
          ) : overviewError ? (
            <div style={{ textAlign: 'center', color: colors.red, padding: 12 }}>
              <div style={{ fontSize: 28, opacity: 0.5 }}>&#9888;</div>
              <div style={{ fontSize: '9pt' }}>{t('overview.failed')}</div>
              <div style={{ fontSize: '8pt', opacity: 0.8, marginTop: 4, maxWidth: 220 }}>{overviewError}</div>
            </div>
          ) : overviewLoading ? (
            <div style={{ textAlign: 'center', color: colors.textSecondary }}>
              <div className="spinner" style={{
                width: 22, height: 22, margin: '0 auto 8px',
                border: `2px solid ${colors.border}`, borderTopColor: colors.accent,
                borderRadius: '50%', animation: 'spin 0.8s linear infinite',
              }} />
              <div style={{ fontSize: '10pt' }}>{t('overview.computing')}</div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: colors.textSecondary }}>
              <div style={{ fontSize: 28, opacity: 0.3 }}>&#9635;</div>
              <div style={{ fontSize: '10pt' }}>{t('overview.emptyHint')}</div>
            </div>
          )}
          </div>
        </div>

        {/* Crop to selection — reads the very ROI rectangle drawn above.
            Hidden until a file is loaded: with no data there is nothing to
            crop, and the empty viewer stays exactly as it was. */}
        {ebsdLoaded && (
          <CropPanel
            bbox={cropBbox}
            mask={cropMask}
            /* The dataset-sync path (page re-entry with data already in the
               backend) sets signal_shape but not pattern_shape — both are the
               detector [h, w], and estimateBytes only multiplies them. */
            patternShape={ebsdInfo?.pattern_shape || ebsdInfo?.signal_shape || null}
            bytesPerPixel={bytesPerSample(activeDtype)}
            shape={cropTool}
            busy={cropBusy}
            origin={cropWindow}
            onCrop={handleCrop}
            onExport={handleExportCrop}
            saving={cropSaving}
          />
        )}
        </div>

        {/* Pattern column */}
        <div style={{
          flex: 1, minWidth: 0, minHeight: 0,
          display: 'flex', flexDirection: 'column',
          background: colors.bgSecondary,
          borderLeft: `1px solid ${colors.border}`,
        }}>
          {/* Caption strip — all status tags live here, outside the image */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '2px 8px', fontSize: '8pt',
            color: colors.textSecondary,
            background: colors.bg, borderBottom: `1px solid ${colors.border}`,
            flexShrink: 0, minHeight: 18, whiteSpace: 'nowrap', overflow: 'hidden',
          }}>
            <span style={{ color: colors.cyan, fontWeight: 600 }}>{t('pattern.captionTitle')}</span>
            <span>· [{row}, {col}]</span>
            {ebsdInfo?.pattern_shape && (
              <span>· {ebsdInfo.pattern_shape[0]}×{ebsdInfo.pattern_shape[1]}</span>
            )}
            {patZoomed && <span style={{ color: colors.accent }}>· {patView.scale.toFixed(1)}×</span>}
            {filter !== 'None' && <span style={{ color: colors.orange }}>· {filter}</span>}
            {gamma !== 100 && <span style={{ color: colors.yellow }}>· B {(gamma / 100).toFixed(1)}</span>}
            {maskEnabled && <span style={{ color: colors.green }}>· mask {maskRadius}%</span>}
            {patternIsThumb && <span style={{ color: colors.orange }}>· {t('pattern.preview')}</span>}
            <span style={{ flex: 1 }} />
            <span style={{ opacity: 0.7 }}>{t('pattern.captionHint')}</span>
          </div>

        {/* Image container */}
        <div
          style={{
            flex: 1, minHeight: 0, position: 'relative',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            overflow: 'hidden', cursor: patZoomed ? 'grab' : 'default',
            padding: 4,
          }}
          onMouseDown={(e) => {
            if (patZoomed && e.button === 0) {
              patDragRef.current = { x: e.clientX, y: e.clientY };
            }
          }}
          onMouseMove={(e) => {
            if (!patDragRef.current) return;
            if (e.buttons !== 1) { patDragRef.current = null; return; }
            const box = patBoxRef.current?.getBoundingClientRect();
            if (!box?.width || !box?.height) return;
            const dx = e.clientX - patDragRef.current.x;
            const dy = e.clientY - patDragRef.current.y;
            patDragRef.current = { x: e.clientX, y: e.clientY };
            setPatView((v) => panBy(v, dx / box.width, dy / box.height));
          }}
          onMouseUp={() => { patDragRef.current = null; }}
          onMouseLeave={() => { patDragRef.current = null; }}
          onDoubleClick={() => setPatView(IDENTITY_VIEW)}
          onContextMenu={(e) => {
            e.preventDefault();
            setContextMenu({ x: e.clientX, y: e.clientY, target: 'pattern' });
          }}
        >
          <div
            ref={patBoxRef}
            data-pattern-box
            style={{
              position: 'relative',
              width: '100%', height: '100%',
              border: `1px solid ${colors.border}`, borderRadius: 6, overflow: 'hidden',
              background: '#000',
            }}
          >
          {patternLoading ? (
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', gap: 8,
            }}>
              <div style={{
                width: 20, height: 20,
                border: `2px solid ${colors.border}`,
                borderTop: `2px solid ${colors.accent}`,
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }} />
              <Label secondary small>{t('pattern.loading')}</Label>
            </div>
          ) : pattern ? (
            <>
              {/* Zoom layer — the transform lives on a wrapper so the EDS
                  composition overlay below stays put and unscaled. No
                  transition: with zoom-to-cursor an animated transform makes
                  the point under the pointer visibly drift while it runs. */}
              <div data-pattern-zoom-layer style={{
                position: 'absolute', inset: 0,
                transform: viewToTransform(patView),
                transformOrigin: '50% 50%',
                willChange: patZoomed ? 'transform' : 'auto',
              }}>
                <img
                  className="image-reveal"
                  src={`data:image/png;base64,${pattern}`}
                  alt={t('pattern.alt', { row, col })}
                  draggable={false}
                  style={{
                    width: '100%', height: '100%', objectFit: 'contain',
                    imageRendering: patternIsThumb ? 'auto' : (interpolation === 'nearest' ? 'pixelated' : 'auto'),
                    filter: gamma !== 100 ? `brightness(${gamma / 100})` : 'none',
                  }}
                />
              </div>
              {showEdsOverlay && (
                <EdsOverlay composition={edsComposition} mode={edsMode} />
              )}
            </>
          ) : (
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', color: colors.textSecondary,
            }}>
              <div style={{ fontSize: 28, opacity: 0.3 }}>&#9634;</div>
              <div style={{ fontSize: '10pt' }}>{t('pattern.emptyHint')}</div>
            </div>
          )}
          </div>
        </div>
        </div>
      </div>

      {/* Navigation arrow bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: spacing.innerSpacing,
        padding: `${spacing.compactMargin}px ${spacing.innerMargin}px`,
        background: colors.bgSecondary, borderTop: `1px solid ${colors.border}`, flexShrink: 0,
      }}>
        <Button small onClick={() => navigateTo(row, col - 1)} disabled={!ebsdLoaded || col <= 0} title={t('hoverTips.navLeft')} aria-label={t('hoverTips.navLeft')}>←</Button>
        <Button small onClick={() => navigateTo(row - 1, col)} disabled={!ebsdLoaded || row <= 0} title={t('hoverTips.navUp')} aria-label={t('hoverTips.navUp')}>↑</Button>
        <Button small onClick={() => navigateTo(row + 1, col)} disabled={!ebsdLoaded || row >= maxRow} title={t('hoverTips.navDown')} aria-label={t('hoverTips.navDown')}>↓</Button>
        <Button small onClick={() => navigateTo(row, col + 1)} disabled={!ebsdLoaded || col >= maxCol} title={t('hoverTips.navRight')} aria-label={t('hoverTips.navRight')}>→</Button>
        {gridShape && (
          <span style={{ fontSize: '8pt', color: colors.textSecondary, fontFamily: 'monospace' }}
            title={t('navigation.patternCountTooltip', { current: row * gridShape[1] + col + 1, total: gridShape[0] * gridShape[1] })}>
            #{row * gridShape[1] + col + 1}/{gridShape[0] * gridShape[1]}
          </span>
        )}
        <div style={{ width: 1, height: 18, background: colors.border, margin: '0 4px' }} />
        <Button
          small
          onClick={async () => {
            try {
              const modeMap = { 'Mean Intensity': 'mean', 'Std Dev (Quality)': 'std', 'Band Contrast': 'bc' };
              const res = await ebsdApi.overview(modeMap[overviewMode] || 'std');
              if (res.data?.best_pos) {
                navigateTo(res.data.best_pos[0], res.data.best_pos[1]);
                log(t('logMessages.bestPattern', { pos: res.data.best_pos }));
              }
            } catch { log(t('logMessages.noBestPattern')); }
          }}
          disabled={!ebsdLoaded}
          title={t('navigation.bestTooltip')}
        >
          {t('navigation.best')}
        </Button>
        <Button
          small
          onClick={async () => {
            try {
              const modeMap = { 'Mean Intensity': 'mean', 'Std Dev (Quality)': 'std', 'Band Contrast': 'bc' };
              const res = await ebsdApi.overview(modeMap[overviewMode] || 'std');
              if (res.data?.worst_pos) {
                navigateTo(res.data.worst_pos[0], res.data.worst_pos[1]);
                log(t('logMessages.worstPattern', { pos: res.data.worst_pos }));
              }
            } catch { log(t('logMessages.noWorstPattern')); }
          }}
          disabled={!ebsdLoaded}
          title={t('navigation.worstTooltip')}
        >
          {t('navigation.worst')}
        </Button>
        <Button
          small
          onClick={() => {
            if (gridShape) {
              navigateTo(
                Math.floor(Math.random() * gridShape[0]),
                Math.floor(Math.random() * gridShape[1]),
              );
            }
          }}
          disabled={!ebsdLoaded}
          title={t('navigation.randomTooltip')}
        >
          {t('navigation.random')}
        </Button>
        <div style={{ flex: 1 }} />
        <Label secondary small>{t('navigation.rowLabel')}</Label>
        <NumberInput
          value={row}
          // Clamp typed values to the grid: NumberInput's max only limits the
          // spinner arrows, not keyboard entry. An out-of-range row/col would
          // be read by the filter/mask effects (loadPattern(row,col)) and fire
          // an out-of-bounds /pattern/{r}/{c} request → 400. Clamp here so the
          // state can never leave the grid.
          onChange={(e) => setRow(Math.max(0, Math.min(Number(e.target.value) || 0, maxRow)))}
          min={0}
          max={maxRow}
          title={t('hoverTips.rowInput')}
          style={{ width: 60 }}
        />
        <Label secondary small>{t('navigation.colLabel')}</Label>
        <NumberInput
          value={col}
          onChange={(e) => setCol(Math.max(0, Math.min(Number(e.target.value) || 0, maxCol)))}
          min={0}
          max={maxCol}
          title={t('hoverTips.colInput')}
          style={{ width: 60 }}
        />
        <Button onClick={() => navigateTo(row, col)} disabled={!ebsdLoaded} small title={t('hoverTips.goButton')}>
          {t('navigation.go')}
        </Button>
      </div>

      {/* Pattern action footer — PC-refinement quick action */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: spacing.innerSpacing,
        padding: `${spacing.compactMargin}px ${spacing.innerMargin}px`,
        background: colors.bgSecondary, borderTop: `1px solid ${colors.border}`,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: '8pt', color: colors.textSecondary }}>
          {t('patternFooter.zoom', { percent: Math.round(patView.scale * 100) })}
        </span>
        <div style={{ flex: 1 }} />
        <CheckRow
          checked={showEdsOverlay}
          onChange={setShowEdsOverlay}
          label={t('patternFooter.edsOverlay')}
          title={t('patternFooter.edsOverlayTooltip')}
        />
        <Button
          variant="purple"
          small
          onClick={handleAddCurrentPattern}
          disabled={!ebsdLoaded}
          title={t('pcRefinement.addToPcRefinementTooltip')}
        >
          {t('pcRefinement.addToPcRefinement')}
          {pcPatternCount > 0 && (
            <span style={{
              marginLeft: 4, background: colors.textOnAccent, color: colors.purple,
              fontSize: '8pt', fontWeight: 700, borderRadius: 8, padding: '1px 6px',
            }}>
              {pcPatternCount}
            </span>
          )}
        </Button>
      </div>
    </div>
  );

  // -------------------------------------------------------------------------
  // Log + Merge panel (bottom of right vertical splitter)
  // -------------------------------------------------------------------------
  const logPanel = (
    <div style={{ display: 'flex', flexDirection: 'column', flexShrink: 0 }}>

      {/* Compare panel (Ctrl+K) */}
      {compareOpen && datasets.length >= 2 && (
        <div style={{
          display: 'flex', gap: spacing.innerSpacing, padding: spacing.innerMargin,
          background: colors.bgSecondary, borderBottom: `1px solid ${alpha(colors.cyan, 27)}`, flexShrink: 0,
        }}>
          {[
            { key: 'left', label: t('compare.left'), state: compareLeft, set: setCompareLeft, imgSet: setCompareLeftImg, img: compareLeftImg },
            { key: 'right', label: t('compare.right'), state: compareRight, set: setCompareRight, imgSet: setCompareRightImg, img: compareRightImg },
          ].map(({ key, label, state, set, imgSet, img }) => (
            <div key={key} style={{ flex: 1 }}>
              <Label secondary small>{label}</Label>
              <select
                value={state || ''}
                title={t('hoverTips.compareSelect')}
                onChange={async (e) => {
                  const name = e.target.value;
                  set(name);
                  if (name) {
                    try {
                      await ebsdApi.selectDataset(name);
                      const r = await ebsdApi.getPattern(row, col);
                      imgSet(r.data?.image);
                      await ebsdApi.selectDataset(activeDataset || datasets[0]?.name);
                    } catch { imgSet(null); }
                  }
                }}
                style={{
                  width: '100%', background: colors.bg, border: `1px solid ${colors.border}`,
                  borderRadius: 3, color: colors.text, fontSize: '9pt', padding: '3px 6px',
                  marginTop: 2,
                }}
              >
                <option value="">{t('compare.selectPlaceholder')}</option>
                {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
              </select>
              {img && (
                <img
                  className="image-reveal"
                  src={`data:image/png;base64,${img}`}
                  alt={label}
                  style={{ width: '100%', marginTop: 4, imageRendering: 'pixelated' }}
                />
              )}
            </div>
          ))}
        </div>
      )}

      {/* Collapsible log bar */}
      <div
        onClick={() => setLogExpanded((v) => !v)}
        title={logExpanded ? t('hoverTips.logToggleCollapse') : t('hoverTips.logToggleExpand')}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: `2px ${spacing.innerMargin}px`, height: 26,
          background: colors.bgSecondary, borderTop: `1px solid ${colors.border}`,
          cursor: 'pointer', flexShrink: 0,
        }}
      >
        <span style={{
          fontSize: '9pt', color: colors.textSecondary,
          transform: logExpanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s',
        }}>{'\u25B6'}</span>
        <span style={{ fontSize: '8pt', color: colors.textSecondary, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          {t('log.label')}
        </span>
        <span style={{
          flex: 1, fontFamily: "'Courier New', monospace", fontSize: '9pt',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          color: logLines.length
            ? (logLines[logLines.length - 1].includes('ERROR') ? colors.red : colors.green)
            : colors.textSecondary,
        }}>
          {logLines.length ? logLines[logLines.length - 1] : t('log.ready')}
        </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (logLines.length === 0) return;
            navigator.clipboard.writeText(logLines.join('\n')).then(() => {
              setLogCopied(true);
              setTimeout(() => setLogCopied(false), 1200);
            }).catch(() => {});
          }}
          disabled={logLines.length === 0}
          title={logCopied ? t('log.copied') : t('log.copyTooltip')}
          aria-label={t('log.copyAriaLabel')}
          style={{
            background: 'none', border: 'none',
            cursor: logLines.length ? 'pointer' : 'default',
            color: logCopied ? colors.green : colors.textSecondary,
            fontSize: '9pt', padding: '1px 4px',
            opacity: logLines.length ? 0.7 : 0.3,
          }}
        >
          {logCopied ? '\u2713' : '\uD83D\uDCCB'}
        </button>
      </div>

      {/* Expanded log body */}
      {logExpanded && (
        <div
          ref={logRef}
          aria-label={t('log.ariaLabel')}
          className="thin-scrollbar"
          style={{
            maxHeight: 140, overflowY: 'auto',
            padding: `${spacing.compactMargin}px ${spacing.innerMargin}px`,
            background: colors.bg, fontFamily: "'Courier New', monospace",
            fontSize: '9pt', color: colors.textSecondary,
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          {logLines.length === 0 ? (
            <span style={{ opacity: 0.5 }}>{t('log.ready')}</span>
          ) : logLines.map((line, i) => {
            const color = line.includes('ERROR') ? colors.red
              : line.includes('complete') || line.includes('loaded') || line.includes('Loaded') ? colors.green
              : line.includes('WARNING') ? colors.orange
              : colors.textSecondary;
            return (
              <div key={i} style={{ color, display: 'flex', gap: 8 }}>
                <span style={{ color: colors.textSecondary, opacity: 0.3, minWidth: 20, textAlign: 'right', userSelect: 'none' }}>{i + 1}</span>
                <span>{line}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  // =========================================================================
  // ROOT RENDER
  // =========================================================================
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer?.files?.[0];
        if (!file) return;
        // Electron exposes .path on dropped files
        const path = file.path || file.name;
        if (path && /\.(h5oina|h5|hdf5|up1|up2)$/i.test(path)) {
          doLoadFile(path);
        }
      }}
      style={{
        display: 'flex', flexDirection: 'column', height: '100%',
        color: colors.text, fontFamily: "'Segoe UI', system-ui, sans-serif",
        fontSize: '10pt', background: colors.bg,
        position: 'relative',
        outline: dragOver ? `2px dashed ${colors.accent}` : 'none',
      }}
    >

      {/* Load progress modal — fixed-position overlay, position:fixed inset:0
          so it always sits above all viewer chrome regardless of layout. */}
      <LoadProgressModal
        isOpen={loadProgressOpen}
        progressState={loadProgressState}
        onClose={() => setLoadProgressOpen(false)}
      />

      {/* Right-click menu for the two images */}
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={() => setContextMenu(null)}
          items={[{
            id: 'export',
            label: t('imageexport:menuExport'),
            disabled: contextMenu.target === 'overview' ? !overviewImage : !pattern,
            onSelect: () => openExport(contextMenu.target),
          }]}
        />
      )}

      {/* Export dialog. Guarded on `src` because the image can only be exported
          once it exists — the menu entry is already disabled in that case, but
          the dialog would otherwise open empty after a file switch. */}
      {exportProps?.src && (
        <ImageExportDialog
          open
          onClose={() => setExportFor(null)}
          onExported={(res) => log(
            res.via === 'electron'
              ? t('logMessages.exportedImageTo', { path: res.path })
              : t('logMessages.exportedImage', { name: res.path }),
          )}
          {...exportProps}
        />
      )}

      {/* Drag overlay */}
      {dragOver && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 60,
          background: `${colors.accent}15`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none',
        }}>
          <div style={{
            fontSize: '14pt', color: colors.accent, fontWeight: 700,
            padding: '16px 32px', borderRadius: 8,
            background: `${colors.bg}ee`, border: `2px dashed ${colors.accent}`,
          }}>
            {t('drag.dropHint')}
          </div>
        </div>
      )}

      {/* Full-page loading overlay during file load */}
      {loadLoading && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 50,
          background: 'rgba(26,27,38,0.75)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 12,
        }}>
          <div style={{ fontSize: 28, animation: 'spin 1.2s linear infinite' }}>&#9676;</div>
          <div style={{ fontSize: '12pt', color: colors.accent, fontWeight: 600 }}>{t('loadingOverlay.title')}</div>
          <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('loadingOverlay.subtitle')}</div>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* Processing overlay for pattern operations */}
      {processingBusy && !loadLoading && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 50,
          background: 'rgba(26,27,38,0.65)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 12,
        }}>
          <div style={{ fontSize: 28, animation: 'spin 1.2s linear infinite' }}>&#9881;</div>
          <div style={{ fontSize: '12pt', color: colors.accent, fontWeight: 600 }}>{processingLabel}...</div>
          {(() => {
            const pp = procProgress;
            const frac = typeof pp?.fraction === 'number' ? Math.max(0, Math.min(1, pp.fraction)) : null;
            if (frac === null) {
              return <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('processingOverlay.subtitle')}</div>;
            }
            const pct = Math.round(frac * 100);
            const el = typeof pp.elapsed === 'number' ? pp.elapsed : 0;
            const rate = el > 0 && frac > 0 ? (frac / el) : 0;           // fraction/s
            const etaS = rate > 0 ? Math.max(0, Math.round((1 - frac) / rate)) : null;
            const itps = (pp.total && el > 0) ? (pp.done / el) : null;   // chunks/s
            const elStr = el.toFixed(0);
            let statsText;
            if (itps && etaS != null) {
              statsText = t('processingOverlay.statsRateEta', { elapsed: elStr, rate: itps.toFixed(1), eta: etaS });
            } else if (itps) {
              statsText = t('processingOverlay.statsRate', { elapsed: elStr, rate: itps.toFixed(1) });
            } else if (etaS != null) {
              statsText = t('processingOverlay.statsEta', { elapsed: elStr, eta: etaS });
            } else {
              statsText = t('processingOverlay.stats', { elapsed: elStr });
            }
            return (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, width: 320, maxWidth: '70%' }}>
                <div style={{ width: '100%', height: 8, borderRadius: 4, background: alpha(colors.accent, 20), overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: colors.accent, transition: 'width 0.3s ease' }} />
                </div>
                <div style={{ fontSize: '10pt', color: colors.text, fontVariantNumeric: 'tabular-nums' }}>
                  {pp.total ? t('processingOverlay.chunks', { percent: pct, done: pp.done, total: pp.total }) : t('processingOverlay.percentOnly', { percent: pct })}
                </div>
                <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
                  {statsText}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Page header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: `${spacing.compactMargin}px ${spacing.innerMargin}px`,
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('header.title')}</h1>
          <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
            {t('header.subtitle')}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing }}>
          <FileSwitcher onSwitched={handleHeaderSwitcherSwitched} />
          <Button
            variant="ghost" small
            onClick={() => onNavigate?.('h5viewer')}
            title={t('header.hdf5ViewerTooltip')}
          >
            {t('header.hdf5Viewer')}
          </Button>
          <Button
            variant="ghost" small
            onClick={() => onNavigate?.('eds')}
            title={t('header.edsPageTooltip')}
          >
            {t('header.edsPage')}
          </Button>
        </div>
      </div>

      {/* Main area: horizontal splitter (left panel | right panel) */}
      <ResizableSplitter
        defaultLeftWidth={270}
        minLeftWidth={200}
        maxLeftWidth={480}
        style={{ flex: 1, minHeight: 0 }}
        left={leftPanel}
        right={
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
            <div style={{ flex: 1, minHeight: 0 }}>{plotPanel}</div>
            {logPanel}
          </div>
        }
      />

      {/* Info bar */}
      <div style={{
        padding: `2px ${spacing.innerMargin}px`,
        background: colors.bgSecondary, borderTop: `1px solid ${colors.border}`,
        fontSize: '9pt', color: colors.textSecondary,
        display: 'flex', gap: 16, flexShrink: 0, alignItems: 'center',
      }}>
        {ebsdInfo ? (
          <>
            <span style={{ fontWeight: 500, color: colors.text }} title={filePath}>{filePath.split(/[\\/]/).pop()}</span>
            {gridShape && <span title={t('infoBar.gridTooltip', { rows: gridShape[0], cols: gridShape[1], pixels: (gridShape[0] * gridShape[1]).toLocaleString() })}>{t('infoBar.grid', { rows: gridShape[0], cols: gridShape[1] })}</span>}
            {ebsdInfo.pattern_shape && (
              <span>{t('infoBar.pattern', { rows: ebsdInfo.pattern_shape[0], cols: ebsdInfo.pattern_shape[1] })}</span>
            )}
            <span style={{ color: colors.accent }} title={t('infoBar.pixelTooltip')}>
              [{row}, {col}]
            </span>
            {activeDataset && (datasets.length > 1 || /_/.test(activeDataset)) && (
              <span style={{ color: colors.yellow }}>{activeDataset}</span>
            )}
            {detector?.has_detector && detector.pc?.length > 0 && (
              <span title={t('infoBar.pcTooltip')}>{t('infoBar.pcLabel', { values: detector.pc.map((v) => v.toFixed(3)).join(', ') })}</span>
            )}
            {gridShape && ebsdInfo.pattern_shape && (() => {
              const mb = (gridShape[0] * gridShape[1] * ebsdInfo.pattern_shape[0] * ebsdInfo.pattern_shape[1]) / (1024 * 1024);
              return <span style={{ opacity: 0.6 }} title={t('infoBar.sizeTooltip', { gridRows: gridShape[0], gridCols: gridShape[1], patRows: ebsdInfo.pattern_shape[0], patCols: ebsdInfo.pattern_shape[1] })}>{mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`}</span>;
            })()}
          </>
        ) : (
          <span style={{ fontStyle: 'italic', opacity: 0.6 }}>
            {'\u25A3'} {t('infoBar.noFile')}
          </span>
        )}
      </div>
      <ConfirmDialog {...confirmProps} />
      <PromptDialog {...promptProps} />
    </div>
  );
}
