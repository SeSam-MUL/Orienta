/**
 * EDS Analysis — Overlay + Tile-Grid layout (FEAT-33 redesign).
 *
 * Replaces the four-tab layout (Element / Phase / Virtual BSE / Band Contrast)
 * with a persistent composite Overlay + a responsive Tile-Grid of every visible
 * layer. The right rail keeps the existing Pixel Quantification, Region Average,
 * Phase Map Controls, and Phase Suggestion blocks.
 *
 * Layer stack is owned by useEdsLayerStack (which reuses PhaseMap's reducer
 * for state-shape compatibility with LayerStackPanel).
 */
import { useState, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { edsApi } from '../../services/api';
import {
  colors, alpha, spacing,
  Button, NumberInput, GroupBox, Label,
} from '../../theme/components';
import useDataStore from '../../stores/useDataStore';
import useResultStore from '../../stores/useResultStore';
import { CursorSyncProvider, useCursorSync } from './CursorSyncContext';
import { pixelSizeForLayer } from './layerPixelSize';
import OverlayCard from './OverlayCard';
import SwipeCompareController from './SwipeCompareController';
import TileGrid from './TileGrid';
import LayerStackPanel from '../PhaseMap/LayerStackPanel';
import FileSwitcher from '../common/FileSwitcher';
import { qualityProvenanceKey } from '../common/qualityProvenance';
import CropWarningChip, { cropWarningFor } from '../common/CropWarningChip';
import HoverProbeOverlay from './HoverProbeOverlay';
import ThresholdHistogram from './ThresholdHistogram';
import LinescanProfilePlot from './LinescanProfilePlot';
import { useActiveDatasetKey } from './hooks/useActiveDatasetKey';
import { useDefaultLayers } from './hooks/useDefaultLayers';
import { useEdsLayerStack } from './hooks/useEdsLayerStack';
import { allMapsLayersFor } from './edsLayerSources';
import { useHoverProbe } from './hooks/useHoverProbe';
import { useLinescan } from './hooks/useLinescan';
import { useSuggestPhases } from './hooks/useSuggestPhases';
import { useZoomViews, SYNC_ALL, SYNC_SINGLE } from './hooks/useZoomViews';
import { usePhaseMap, PhaseMapCanvas, PhaseMapControls } from './PhaseMapPanel';
import StructureInspector from './StructureInspector';
import useStructureInspector from './hooks/useStructureInspector';
import { exportComposite } from './compositeExporter';
import ContextMenu from '../common/ContextMenu';
import ImageExportDialog from '../common/ImageExportDialog';
import {
  buildSingleCanvas, buildCompositeCanvas, buildMontageCanvas,
  sourceBitmapFor, canvasToDataUrl,
} from './edsExportSources';

const DISPLAY_MODES = [
  { id: 'counts', labelKey: 'mode.counts', tipKey: 'mode.countsTip' },
  { id: 'wt_pct', labelKey: 'mode.wtPct',  tipKey: 'mode.wtPctTip' },
  { id: 'at_pct', labelKey: 'mode.atPct',  tipKey: 'mode.atPctTip' },
];

// Zoom-view id of the composite overlay. The tiles use their layer id, so this
// only has to be distinct from those.
const OVERLAY_VIEW_ID = '__overlay__';

/**
 * Zoom sync switch + reset, shown in the "All Maps" header. 'All' locks every
 * map (tiles AND the composite overlay) to one view; 'Single' gives each its
 * own. Reset returns everything to 1x.
 */
function ZoomToolbar({ mode, onModeChange, onReset, resetDisabled }) {
  const { t } = useTranslation('eds');
  const options = [
    { id: SYNC_ALL, label: t('allMaps.zoomSyncAll'), tip: t('allMaps.zoomSyncAllTooltip') },
    { id: SYNC_SINGLE, label: t('allMaps.zoomSyncSingle'), tip: t('allMaps.zoomSyncSingleTooltip') },
  ];

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }} title={t('allMaps.zoomHint')}>
      <span>{t('allMaps.zoomSync')}</span>
      <div role="group" aria-label={t('allMaps.zoomSyncAriaLabel')}
        style={{ display: 'flex', border: `1px solid ${colors.border}`, borderRadius: 4, overflow: 'hidden' }}>
        {options.map((opt, i) => (
          <button
            key={opt.id}
            onClick={() => onModeChange(opt.id)}
            aria-pressed={mode === opt.id}
            data-zoom-sync={opt.id}
            title={opt.tip}
            style={{
              padding: '2px 8px',
              background: mode === opt.id ? colors.purple : 'transparent',
              color: mode === opt.id ? colors.bg : colors.textSecondary,
              fontWeight: mode === opt.id ? 700 : 400,
              fontSize: '8.5pt',
              border: 'none',
              borderLeft: i > 0 ? `1px solid ${colors.border}` : 'none',
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <button
        onClick={onReset}
        disabled={resetDisabled}
        data-zoom-reset
        title={t('allMaps.zoomResetTooltip')}
        style={{
          background: 'transparent',
          border: `1px solid ${resetDisabled ? colors.border : colors.cyan}`,
          color: resetDisabled ? colors.textSecondary : colors.cyan,
          borderRadius: 4,
          padding: '2px 8px',
          fontSize: '8.5pt',
          fontWeight: 600,
          cursor: resetDisabled ? 'not-allowed' : 'pointer',
          opacity: resetDisabled ? 0.5 : 1,
        }}
      >
        {t('allMaps.zoomReset')}
      </button>
    </div>
  );
}

// Mode toggle — matches EDSOverlayPanel mode_combo
function ModeToggle({ value, onChange }) {
  const { t } = useTranslation('eds');
  return (
    <div role="group" aria-label={t('mode.groupLabel')}
      style={{ display: 'flex', border: `1px solid ${colors.border}`, borderRadius: 5, overflow: 'hidden' }}>
      {DISPLAY_MODES.map((opt, i) => (
        <button key={opt.id} onClick={() => onChange(opt.id)} aria-pressed={value === opt.id} title={t(opt.tipKey)}
          style={{
            flex: 1, padding: '6px 12px',
            background: value === opt.id ? colors.purple : 'transparent',
            color: value === opt.id ? colors.bg : colors.textSecondary,
            fontWeight: value === opt.id ? 700 : 400,
            fontSize: '9pt', border: 'none',
            borderLeft: i > 0 ? `1px solid ${colors.border}` : 'none',
            cursor: 'pointer', transition: 'all 0.15s',
          }}>
          {t(opt.labelKey)}
        </button>
      ))}
    </div>
  );
}

function QuantTable({ data }) {
  const { t } = useTranslation('eds');
  if (!data || data.length === 0) return null;
  const th = { padding: '5px 8px', textAlign: 'left', color: colors.textSecondary, fontWeight: 600, fontSize: '9pt', borderBottom: `1px solid ${colors.border}`, whiteSpace: 'nowrap' };
  const td = (i) => ({ padding: '4px 8px', fontSize: '9pt', borderBottom: `1px solid ${alpha(colors.border, 13)}`, background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20) });
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead><tr>{[t('table.element'), t('table.counts'), t('table.wtPct'), t('table.atPct')].map((h) => <th key={h} style={th}>{h}</th>)}</tr></thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={row.element || i} className="table-row-hover">
            <td style={{ ...td(i), color: colors.cyan, fontWeight: 600 }}>{row.element}</td>
            <td style={td(i)}>{typeof row.counts === 'number' ? Math.round(row.counts).toLocaleString('en-US') : row.counts ?? t('table.empty')}</td>
            <td style={td(i)}>{typeof row.wt_pct === 'number' ? row.wt_pct.toFixed(2) : row.wt_pct ?? t('table.empty')}</td>
            <td style={td(i)}>{typeof row.at_pct === 'number' ? row.at_pct.toFixed(2) : row.at_pct ?? t('table.empty')}</td>
          </tr>
        ))}
      </tbody>
      {(() => {
        const atSum = data.reduce((s, r) => s + (typeof r.at_pct === 'number' ? r.at_pct : 0), 0);
        const wtSum = data.reduce((s, r) => s + (typeof r.wt_pct === 'number' ? r.wt_pct : 0), 0);
        if (atSum === 0 && wtSum === 0) return null;
        const ok = Math.abs(atSum - 100) < 1;
        return (
          <tfoot>
            <tr>
              <td style={{ ...th, fontWeight: 700 }} title={t('table.sumTooltip')}>{'Σ'}</td>
              <td style={th}></td>
              <td style={{ ...th, fontWeight: 600 }}>{wtSum.toFixed(1)}%</td>
              <td style={{
                ...th, fontWeight: 700, color: ok ? colors.green : colors.orange,
                background: ok ? `${colors.green}11` : `${colors.orange}11`,
                borderRadius: 3,
              }}>
                {ok ? '✓ ' : '⚠ '}{atSum.toFixed(1)}%
              </td>
            </tr>
          </tfoot>
        );
      })()}
    </table>
  );
}

function RegionQuantTable({ data }) {
  const { t } = useTranslation('eds');
  if (!data?.data || Object.keys(data.data).length === 0) return null;
  const th = { padding: '5px 8px', textAlign: 'left', color: colors.textSecondary, fontWeight: 600, fontSize: '8pt', borderBottom: `1px solid ${colors.border}`, whiteSpace: 'nowrap' };
  const td = (i) => ({ padding: '4px 8px', fontSize: '8pt', borderBottom: `1px solid ${alpha(colors.border, 13)}`, background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20) });
  const fmtStat = (s) => {
    if (!s) return t('table.empty');
    if (typeof s === 'number') return s.toFixed(2);
    return `${s.mean?.toFixed(1) ?? t('table.empty')} ± ${s.std?.toFixed(1) ?? t('table.empty')}`;
  };
  const entries = Object.entries(data.data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ fontSize: '8pt', color: colors.textSecondary, marginBottom: 4 }}>
        {t('region.summary', { rowStart: data.row_start, colStart: data.col_start, rowEnd: data.row_end, colEnd: data.col_end, nPixels: data.n_pixels })}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {[t('table.element'), t('region.countsHeader'), t('region.wtPctHeader'), t('region.atPctHeader')].map((h) => (
              <th key={h} style={th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map(([el, vals], i) => (
            <tr key={el}>
              <td style={{ ...td(i), color: colors.cyan, fontWeight: 600 }}>{el}</td>
              <td style={td(i)}>{fmtStat(vals.counts)}</td>
              <td style={td(i)}>{fmtStat(vals.wt_pct)}</td>
              <td style={td(i)}>{fmtStat(vals.at_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Map a LayerStackPanel quick-mode id into an EDS layer object that
// useEdsLayerStack.addLayer accepts. The quick-mode buttons (phase / ipf-z /
// bc / ci) are hardcoded in LayerStackPanel; here we wire them up as
// single-layer shortcuts on the EDS page.
function quickModeToLayer(id) {
  const base = { id, visible: true, opacity: 1, blend: 'normal', key: `${id}-${Date.now()}` };
  if (id === 'phase') return { ...base, kind: 'phase', label: 'Phase Map' };
  if (id === 'ipf-z') return { ...base, kind: 'ipf-z', label: 'IPF-Z' };
  if (id === 'bc')    return { ...base, kind: 'bc',    label: 'BC' };
  if (id === 'ci')    return { ...base, kind: 'ci',    label: 'CI' };
  return null;
}

function EmptyState() {
  const { t } = useTranslation('eds');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', color: colors.text, fontFamily: "'Segoe UI', system-ui, sans-serif", padding: spacing.outerMargin, gap: spacing.outerSpacing }}>
      <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('empty.title')}</h1>
      <div style={{ fontSize: '10pt', color: colors.textSecondary }}>{t('empty.subtitle')}</div>
      <div style={{
        background: colors.bgSecondary, border: `1px solid ${colors.border}`, borderRadius: 6,
        padding: '40px 24px', textAlign: 'center',
        animation: 'pageFadeIn 0.3s ease-out',
      }}>
        <div style={{ fontSize: '36pt', opacity: 0.2, marginBottom: 10 }}>{'⬢'}</div>
        <Label secondary style={{ fontSize: '11pt' }}>{t('empty.noFile')}</Label>
        <div style={{ fontSize: '9pt', color: colors.textSecondary, marginTop: 6, opacity: 0.6, lineHeight: 1.5 }}>
          {t('empty.hint')}
          <br />{t('empty.hintLine2')}
        </div>
      </div>
    </div>
  );
}

// Cursor-sync subscriber for the hover-probe tooltip. Must live INSIDE the
// CursorSyncProvider subtree so useCursorSync has provider context. EDSPage
// itself sits above the provider in JSX order and cannot subscribe directly.
//
// Tooltip-position state is confined to this leaf so 60Hz cursor moves do not
// re-render the entire EDSPage subtree (OverlayCard / TileGrid / right rail).
function HoverProbeLayer({ probe, error, displayMode, requestProbe, clearProbe }) {
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
  return (
    <HoverProbeOverlay
      probe={probe}
      error={error}
      displayMode={displayMode}
      visible={pos !== null}
      pos={pos}
    />
  );
}

export default function EDSPage({ onNavigate, isActive = true }) {
  const { t } = useTranslation(['eds', 'imageexport']);
  const isFileOpen = useDataStore((s) => s.isFileOpen);
  const filePath = useDataStore((s) => s.filePath);
  // EDS maps sit on the scan grid, so one pixel is one step — same physical
  // scale the EBSD overview uses for its bar.
  const stepSize = useDataStore((s) => s.stepSize);
  const pixelSizes = useDataStore((s) => s.pixelSizes);
  const rawSetPending = useDataStore((s) => s.setPendingPhaseMapIndexing);
  const setPendingPhaseMap = useMemo(() => rawSetPending ?? (() => {}), [rawSetPending]);

  const [displayMode, setDisplayMode] = useState('at_pct');
  const [overlayW, setOverlayW] = useState(360);    // splitter-controlled

  // Which DATASET the maps below belong to. Not the same question as which
  // file: a crop is a new dataset cut from the file already open, and every
  // map on this page comes from the ACTIVE dataset while the probe endpoints
  // resolve coordinates on its grid. Keyed into both hooks so the two can
  // never describe different datasets — see useActiveDatasetKey.
  const datasetKey = useActiveDatasetKey(isFileOpen, filePath, isActive);

  // Default-layer probe + stack hook.
  const def = useDefaultLayers(isFileOpen, filePath, datasetKey);
  const stack = useEdsLayerStack({ initialLayers: def.layers, displayMode, datasetKey });

  // "All Maps" tile grid is DECOUPLED from the curated composite overlay: it
  // shows EVERY detected map (all elements + electron images + BC) via its own
  // uncapped stack with a larger bitmap cache. The composite `stack` above
  // stays curated/capped (MAX_LAYERS) for the overlay + LayerStackPanel.
  const allMapsLayers = useMemo(
    () => allMapsLayersFor({
      elements: def.elements,
      electronImages: def.electronImages,
      hasBC: def.layers.some((l) => l.id === 'bc'),
    }),
    [def.elements, def.electronImages, def.layers],
  );
  const allMaps = useEdsLayerStack({
    initialLayers: allMapsLayers, displayMode, cacheSize: 64, datasetKey,
  });

  // PhaseMap handoff (lifted from today).
  const handlePhaseMapHandoff = useCallback(() => {
    setPendingPhaseMap(true);
    onNavigate?.('indexing');
  }, [onNavigate, setPendingPhaseMap]);
  const phaseMapHandle = usePhaseMap({ onIndexingHandoff: handlePhaseMapHandoff });
  // The inspector follows whichever structure is selected, from the map or
  // from the list, and re-reads after anything that changes the grouping
  // (merge renumbers ids, split adds them).
  const inspector = useStructureInspector({
    selectedStructureId: phaseMapHandle.selectedStructureId,
    setSelectedStructureId: phaseMapHandle.setSelectedStructureId,
    mapVersion: phaseMapHandle.mapVersion,
  });
  // The phase map grew a tool set of its own - structures, the inspector,
  // four boundary tools - and sharing one screen with the element overlay
  // and the tile grid left neither usable. It gets its own tab; everything
  // else on this page stays exactly where it was.
  const [edsTab, setEdsTab] = useState('elements');
  const inStructureView = phaseMapHandle.mapView === 'structures'
    && (phaseMapHandle.phaseMap?.structures || []).length > 0;

  // Pixel Quantification state (lifted).
  const [pixelRow, setPixelRow] = useState(0);
  const [pixelCol, setPixelCol] = useState(0);
  const [quantData, setQuantData] = useState(null);
  const [quantLoading, setQuantLoading] = useState(false);
  const [quantError, setQuantError] = useState(null);
  const [quantCopied, setQuantCopied] = useState(false);

  // Region Average state (lifted).
  const [regionRowStart, setRegionRowStart] = useState(0);
  const [regionRowEnd, setRegionRowEnd] = useState(0);
  const [regionColStart, setRegionColStart] = useState(0);
  const [regionColEnd, setRegionColEnd] = useState(0);
  const [regionData, setRegionData] = useState(null);
  const [regionLoading, setRegionLoading] = useState(false);
  const [regionError, setRegionError] = useState(null);
  const [regionCopied, setRegionCopied] = useState(false);

  // Phase Suggestion state (lifted). The suggestion is per-pixel, so the hook
  // also tracks WHICH pixel the shown list belongs to and follows the cursor
  // once the user has asked for suggestions.
  // Destructured (rather than used as `suggest.x`) because `suggest` is a fresh
  // object each render — depending on it would churn every consumer's identity.
  const {
    suggestions: suggestedPhases,
    pixel: suggestPixel,
    atomicPct: suggestAtPct,
    mapPhase: suggestMapPhase,
    librarySource: suggestLibrarySource,
    librarySize: suggestLibrarySize,
    loading: suggestLoading,
    error: suggestError,
    run: suggestRun,
    followPixel: suggestFollowPixel,
  } = useSuggestPhases();

  // Hover probe — drives the multi-layer tooltip near the cursor.
  const { probe, error: probeError, requestProbe, clear: clearProbe } = useHoverProbe({ displayMode });

  // Linescan tool — toggle plus per-layer profile fetched from the backend.
  // visibleLayerIds is memoised so the useLinescan callback identity is stable
  // across renders that don't change layer-visibility (otherwise every parent
  // render would invalidate the OverlayCard's onLineComplete prop).
  const [linescanMode, setLinescanMode] = useState(false);
  const [magnifierEnabled, setMagnifierEnabled] = useState(false);
  const [tileMinWidth, setTileMinWidth] = useState(240);

  // Zoom for the composite overlay + every tile. 'all' (default) keeps them
  // locked together; 'single' gives each map its own view.
  const zoom = useZoomViews(SYNC_ALL);
  const overlayView = zoom.viewFor(OVERLAY_VIEW_ID);
  const onOverlayZoom = useCallback(
    (factor, px, py) => zoom.zoomAtPointer(OVERLAY_VIEW_ID, factor, px, py),
    [zoom],
  );
  const onOverlayPan = useCallback(
    (dx, dy) => zoom.pan(OVERLAY_VIEW_ID, dx, dy),
    [zoom],
  );
  const onOverlayResetView = useCallback(() => zoom.resetOne(OVERLAY_VIEW_ID), [zoom]);

  // Indexing-result availability gates the "+ Add Layer" options below.
  const indexingResult = useResultStore((s) => s.indexingResult);
  const hasIndexingResult = !!indexingResult;
  const [swipe, setSwipe] = useState({ a: null, b: null, splitX: 0.5 });
  const onSwipeChange = useCallback((next) => setSwipe((s) => ({ ...s, ...next })), []);
  const onSwipeSplitChange = useCallback((splitX) => setSwipe((s) => ({ ...s, splitX })), []);
  const visibleLayerIds = useMemo(
    () => stack.layers.filter((l) => l.visible).map((l) => l.id),
    [stack.layers],
  );
  const linescan = useLinescan({ layerIds: visibleLayerIds, displayMode });
  const { fetchProfile: linescanFetch } = linescan;
  const onLineComplete = useCallback(({ start, end }) => {
    linescanFetch(start, end);
  }, [linescanFetch]);

  // handleQuantify (lifted). Accept (row, col) override for click-to-quantify.
  const handleQuantify = useCallback(async (rowOverride = null, colOverride = null) => {
    const row = rowOverride != null ? rowOverride : Number(pixelRow);
    const col = colOverride != null ? colOverride : Number(pixelCol);
    setQuantLoading(true); setQuantError(null); setQuantData(null);
    try {
      const res = await edsApi.quantifyPixel(row, col, displayMode);
      const raw = res.data?.data || res.data?.elements || {};
      const rows = Array.isArray(raw)
        ? raw
        : Object.entries(raw).map(([el, vals]) => ({ element: el, counts: vals.counts ?? 0, wt_pct: vals.wt_pct ?? 0, at_pct: vals.at_pct ?? 0 }));
      setQuantData(rows);
    } catch (err) {
      setQuantError(err.response?.data?.detail || err.message || t('quantify.error'));
    } finally { setQuantLoading(false); }
  }, [pixelRow, pixelCol, displayMode, t]);

  // Click-to-quantify (drives both Overlay and tiles). The phase suggestion is
  // a per-pixel quantity too, so it follows the click instead of continuing to
  // describe the previously clicked pixel (no-op until the user has asked for
  // suggestions once).
  const onPixelClick = useCallback((row, col) => {
    setPixelRow(row); setPixelCol(col);
    handleQuantify(row, col);
    suggestFollowPixel(row, col);
  }, [handleQuantify, suggestFollowPixel]);

  // handleRegionQuantify (lifted).
  const handleRegionQuantify = useCallback(async () => {
    setRegionLoading(true); setRegionError(null); setRegionData(null);
    try {
      const res = await edsApi.regionQuantify(
        Number(regionRowStart), Number(regionRowEnd),
        Number(regionColStart), Number(regionColEnd),
        displayMode,
      );
      setRegionData(res.data);
    } catch (err) {
      setRegionError(err.response?.data?.detail || err.message || t('regionAvg.error'));
    } finally { setRegionLoading(false); }
  }, [regionRowStart, regionRowEnd, regionColStart, regionColEnd, displayMode, t]);

  // Shift+drag ROI → auto-populate Region-Average fields and fire compute.
  const onRegionSelected = useCallback(({ rowStart, rowEnd, colStart, colEnd }) => {
    setRegionRowStart(rowStart);
    setRegionRowEnd(rowEnd);
    setRegionColStart(colStart);
    setRegionColEnd(colEnd);
    // Fire on the next tick so the state updates have a chance to land.
    // handleRegionQuantify reads from state, so calling it synchronously here
    // would race against the setState batch.
    setTimeout(() => { handleRegionQuantify(); }, 0);
  }, [handleRegionQuantify]);

  // handleSuggestPhases (lifted) — explicit "Suggest Phases" press.
  const handleSuggestPhases = useCallback(
    () => suggestRun(pixelRow, pixelCol),
    [suggestRun, pixelRow, pixelCol],
  );

  // The shown list belongs to `suggestPixel`; the row/col boxes can be typed
  // into without clicking the map, so flag the mismatch rather than let the
  // panel silently describe a pixel the user is no longer looking at.
  const suggestStale = !!suggestPixel && !suggestLoading && (
    suggestPixel.row !== Number(pixelRow) || suggestPixel.col !== Number(pixelCol)
  );

  // LayerStackPanel.onSetSingleLayer adapter — quick-mode button maps to an EDS layer object.
  const onSetSingleLayer = useCallback((id) => {
    const layer = quickModeToLayer(id);
    if (!layer) return;
    // Localize the quick-mode layer label (the helper is module-level and has
    // no access to t()). Acronyms BC / CI stay verbatim.
    const quickLabels = {
      phase: t('addLayer.phaseMap'),
      'ipf-z': 'IPF-Z',
      bc: 'BC',
      ci: 'CI',
    };
    if (quickLabels[id]) layer.label = quickLabels[id];
    stack.clear();
    stack.addLayer(layer);
  }, [stack, t]);

  // "+ Add Layer" dropdown — offers every mountable source:
  //   - PhaseMap-result layers (Phase / IPF / CI) when an indexing result exists
  //   - Every electron image detected in the H5OINA (real SE + each FSE
  //     detector — Aztec's pre-indexed /Layered Image/ group is excluded
  //     server-side by tools/h5_viewer_backend.get_available_electron_images)
  //   - Every EDS element from the metadata
  // Existing layer ids are filtered out so the dropdown never advertises a
  // duplicate (the reducer dedupes on ADD anyway, but a disappearing option
  // is the clearer affordance).
  const availableToAdd = useMemo(() => {
    const existing = new Set(stack.layers.map((l) => l.id));
    const opts = [];

    // From indexing result: Phase + IPF + CI
    if (hasIndexingResult) {
      opts.push(
        { value: 'phase', label: t('addLayer.phaseMap') },
        { value: 'ipf-z', label: t('addLayer.ipfZ') },
        { value: 'ipf-x', label: t('addLayer.ipfX') },
        { value: 'ipf-y', label: t('addLayer.ipfY') },
        { value: 'ci',    label: t('addLayer.ci') },
      );
    }

    // From H5OINA: every electron image (real SE + all FSE detectors)
    for (const name of (def.electronImages || [])) {
      opts.push({
        value: `electron-${name}`,
        label: t('addLayer.electron', { name }),
      });
    }

    // From EDS metadata: every element not already in the stack
    for (const el of (def.elements || [])) {
      opts.push({
        value: `eds-${el}`,
        label: t('addLayer.eds', { element: el }),
      });
    }

    return opts.filter((opt) => !existing.has(opt.value));
  }, [hasIndexingResult, stack.layers, def.electronImages, def.elements, t]);

  // Map a dropdown id to a full EDS layer object with reasonable defaults
  // (kind, blend, opacity, label). useEdsLayerStack's reducer dedupes on
  // id so adding the same option twice is a no-op, not a crash.
  const onAddLayer = useCallback((id) => {
    // PhaseMap-result layers (existing behaviour)
    const resultLabels = {
      phase:   t('addLayer.phaseMap'),
      'ipf-z': t('addLayer.ipfZ'),
      'ipf-x': t('addLayer.ipfX'),
      'ipf-y': t('addLayer.ipfY'),
      ci:      'CI',          // acronym — kept verbatim in all languages
    };
    if (resultLabels[id]) {
      const opacities = { phase: 0.8, 'ipf-z': 0.8, 'ipf-x': 0.8, 'ipf-y': 0.8, ci: 0.6 };
      stack.addLayer({
        id, kind: id, label: resultLabels[id],
        visible: true, opacity: opacities[id], blend: 'normal',
        key: `${id}-${Date.now()}`,
      });
      return;
    }

    // Electron image (real SE or FSE detector)
    if (typeof id === 'string' && id.startsWith('electron-')) {
      const name = id.slice('electron-'.length);
      stack.addLayer({
        id, kind: 'electron', electronName: name,
        label: name,                          // "FSE/Oben links" or "SE/Elektronenbild 1"
        visible: true, opacity: 0.7, blend: 'normal',
        key: `${id}-${Date.now()}`,
      });
      return;
    }

    // EDS element
    if (typeof id === 'string' && id.startsWith('eds-')) {
      const element = id.slice('eds-'.length);
      stack.addLayer({
        id, kind: 'eds-element', element,
        label: element,
        visible: true, opacity: 0.7, blend: 'screen',
        displayMode: displayMode,             // ensure the new layer respects the active mode
        key: `${id}-${Date.now()}`,
      });
      return;
    }

    // Unknown id — silently no-op (reducer would dedupe anyway).
    console.warn('[EDSPage.onAddLayer] unknown layer id:', id);
  }, [stack, displayMode, t]);

  // Per-layer threshold control — scalar/element layers only (BC, V-BSE, EDS).
  // PhaseMap layers (phase/IPF/CI) skip the histogram: thresholding on a
  // categorical colour map doesn't carry physical meaning. Mask layers
  // (kind='mask') also skip — they already represent a binary selection.
  const renderLayerExtras = useCallback((layer) => {
    // Ahead of the kind filter: an electron image has none of the controls
    // below, but it is the one layer that can fail to follow a crop, and the
    // warning has to reach the user rather than only the log.
    const cropWarn = cropWarningFor(stack.layerCropStatus, layer.id);
    if (cropWarn) return <CropWarningChip status={cropWarn} />;
    if (!['eds-element', 'bc', 'vbse'].includes(layer.kind)) return null;
    // BC provenance: the /band-contrast endpoint returns "h5oina" (native
    // per-pixel Band Contrast) or "computed" (FFT pattern-quality fallback).
    // Normalise "h5oina" → "native" so the shared quality helper picks the
    // right ebsdviewer key; anything else falls through to "computed".
    const bcSourceRaw = layer.kind === 'bc' ? stack.layerSources?.get(layer.id) : null;
    const bcSource = bcSourceRaw === 'h5oina' ? 'native' : bcSourceRaw;
    return (
      <>
        {bcSourceRaw && (
          <div style={{
            fontSize: '7.5pt',
            color: colors.textSecondary,
            fontStyle: 'italic',
            padding: '1px 2px',
          }}>
            {t(qualityProvenanceKey(bcSource), { ns: 'ebsdviewer' })}
          </div>
        )}
        <ThresholdHistogram
          bitmap={stack.bitmaps.get(layer.id)}
          threshold={layer.threshold}
          onThresholdChange={(t) => stack.setThreshold(layer.id, t)}
        />
        {layer.threshold && (
          <button
            onClick={() => stack.addMaskFromThreshold(layer.id)}
            style={{
              background: 'transparent',
              border: `1px solid ${colors.cyan}`,
              color: colors.cyan,
              borderRadius: 3,
              padding: '2px 8px',
              fontSize: '8pt',
              cursor: 'pointer',
              marginTop: 4,
            }}
            title={t('threshold.convertToMaskTooltip')}
          >
            {t('threshold.convertToMask')}
          </button>
        )}
      </>
    );
  }, [stack, t]);

  // Splitter drag (manipulates --overlay-w via React state).
  const onSplitterMouseDown = (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = overlayW;
    const onMove = (ev) => {
      const next = Math.max(240, Math.min(900, startW + (ev.clientX - startX)));
      setOverlayW(next);
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  // --- Right-click export -----------------------------------------------
  // `menu` holds the click point plus what was hit; `exportSrc` is the rendered
  // image handed to the shared dialog.
  const [menu, setMenu] = useState(null);
  const [exportSrc, setExportSrc] = useState(null);
  const [exportError, setExportError] = useState(null);

  const exportStem = useMemo(() => {
    const raw = filePath || 'eds';
    return raw.split(/[\\/]/).pop().replace(/\.[^.]+$/, '') || 'eds';
  }, [filePath]);

  // Building a canvas can fail (a layer whose bitmap has not arrived yet).
  // Surface that instead of opening an empty dialog.
  // `scale` is the physical pixel size OF THE IMAGE BEING EXPORTED. It has to
  // travel with it: electron images sit on the SEM raster and the maps on the
  // scan raster, measured 10.6x apart on a real file, so one global step size
  // mis-scales half the exports.
  const openExport = useCallback((build, name, label, scale = null) => {
    try {
      setExportError(null);
      // `build` may return a canvas it composed, or a URL for something the
      // backend already rendered — the phase and structure maps arrive as
      // base64 PNGs and re-drawing them into a canvas would only re-encode
      // the same pixels.
      const built = build();
      const src = typeof built === 'string' ? built : canvasToDataUrl(built);
      setExportSrc({ src, name, label, scale });
    } catch (err) {
      setExportError(err?.message || String(err));
    }
  }, []);

  const layerName = useCallback((l) => String(l.label ?? l.id), []);

  const exportMenuItems = useCallback((hit) => {
    const items = [];
    if (hit?.kind === 'tile' && hit.layer) {
      const l = hit.layer;
      items.push({
        id: 'this',
        label: t('imageexport:menuExportThis'),
        onSelect: () => openExport(
          () => buildSingleCanvas(l, sourceBitmapFor(l, allMaps.bitmaps)),
          `${exportStem}_${layerName(l)}`,
          `${exportStem} \u00b7 ${layerName(l)}`,
          pixelSizeForLayer(l.id, pixelSizes),
        ),
      });
    }
    if (hit?.kind === 'phasemap') {
      // Both renderings are offered whichever one is on screen, so exporting
      // the other does not mean switching the view and switching back. Both
      // sit on the scan raster, so the map's own pixel size applies.
      const pm = phaseMapHandle.phaseMap;
      const mapScale = pixelSizeForLayer('phase', pixelSizes);
      if (pm?.structure_image) {
        items.push({
          id: 'structure-map',
          label: t('imageexport:menuExportStructureMap',
            { defaultValue: 'Export the structure map…' }),
          onSelect: () => openExport(
            () => `data:image/png;base64,${pm.structure_image}`,
            `${exportStem}_structures`,
            `${exportStem} \u00b7 ${t('tabs.viewStructuresLabel', { defaultValue: 'Structures' })}`,
            mapScale,
          ),
        });
      }
      if (pm?.image) {
        items.push({
          id: 'phase-map',
          label: t('imageexport:menuExportPhaseMap',
            { defaultValue: 'Export the phase map…' }),
          onSelect: () => openExport(
            () => `data:image/png;base64,${pm.image}`,
            `${exportStem}_phases`,
            `${exportStem} \u00b7 ${t('phaseMap.mapTitle', { defaultValue: 'Phase map' })}`,
            mapScale,
          ),
        });
      }
    }
    if (hit?.kind === 'overlay') {
      items.push({
        id: 'overlay',
        label: t('imageexport:menuExportOverlay'),
        onSelect: () => openExport(
          () => buildCompositeCanvas({ layers: stack.layers, bitmaps: stack.bitmaps, shape: stack.shape }),
          `${exportStem}_overlay`,
          `${exportStem} \u00b7 ${t('overlay.title', { defaultValue: 'Overlay' })}`,
          // The composite is drawn on the bottom layer's raster.
          pixelSizeForLayer(stack.layers?.[0]?.id, pixelSizes),
        ),
      });
    }
    items.push({
      id: 'all',
      label: t('imageexport:menuExportAll'),
      onSelect: () => openExport(
        () => buildMontageCanvas({
          layers: allMaps.layers,
          bitmaps: allMaps.bitmaps,
          shape: allMaps.shape,
          labelFor: layerName,
        }),
        `${exportStem}_all-maps`,
        `${exportStem} \u00b7 ${t('allMaps.title', { defaultValue: 'All maps' })}`,
        // A montage mixes rasters with different pixel sizes \u2014 no single scale
        // bar can be correct for it, so offer none.
        null,
      ),
    });
    return items;
  }, [t, openExport, exportStem, layerName, allMaps, stack, pixelSizes]);


  if (!isFileOpen) return <EmptyState />;

  return (
    <CursorSyncProvider>
      <div style={{
        display: 'flex', flexDirection: 'column', height: '100%',
        color: colors.text, fontFamily: "'Segoe UI', system-ui, sans-serif",
        padding: spacing.outerMargin, gap: spacing.outerSpacing, boxSizing: 'border-box',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={() => onNavigate?.('ebsdviewer')}
            style={{
              background: 'transparent', border: `1px solid ${colors.border}`,
              borderRadius: 5, padding: '4px 10px', cursor: 'pointer',
              color: colors.textSecondary, fontSize: '9pt',
              transition: 'all 0.15s',
            }}
            onMouseEnter={(e) => { e.target.style.borderColor = colors.cyan; e.target.style.color = colors.cyan; }}
            onMouseLeave={(e) => { e.target.style.borderColor = colors.border; e.target.style.color = colors.textSecondary; }}
            title={t('header.backTooltip')}>
            {t('header.back')}
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('header.title')}</h1>
            <div style={{ fontSize: '10pt', color: colors.textSecondary, marginTop: 2 }}>
              {t('header.subtitle')}
            </div>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            <FileSwitcher />
            <Label secondary small style={{ marginRight: 4 }}>{t('header.displayLabel')}</Label>
            <ModeToggle value={displayMode} onChange={setDisplayMode} />
          </div>
        </div>

        {/* --- Tabs --------------------------------------------------- */}
        <div style={{ display: 'flex', gap: 4, flexShrink: 0 }}>
          {[
            { id: 'elements', label: t('tabs.elements'), tip: t('tabs.elementsTooltip') },
            { id: 'phasemap', label: t('tabs.phaseMap'), tip: t('tabs.phaseMapTooltip') },
          ].map((tab) => {
            const active = edsTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setEdsTab(tab.id)}
                title={tab.tip}
                style={{
                  fontSize: '9.5pt', padding: '5px 16px', cursor: 'pointer',
                  borderRadius: '4px 4px 0 0',
                  background: active ? colors.bgSecondary : 'transparent',
                  color: active ? colors.cyan : colors.textSecondary,
                  border: `1px solid ${active ? colors.border : 'transparent'}`,
                  borderBottom: active
                    ? `1px solid ${colors.bgSecondary}`
                    : `1px solid ${colors.border}`,
                  fontWeight: active ? 600 : 400,
                }}
              >
                {tab.label}
              </button>
            );
          })}
          <div style={{ flex: 1, borderBottom: `1px solid ${colors.border}` }} />
        </div>

        {/* --- Tab: phase map ----------------------------------------- */}
        {edsTab === 'phasemap' && (
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 300px',
            gap: spacing.outerSpacing, flex: 1, minHeight: 0,
          }}>
            {/* The map gets the room it needs, and the inspector sits under
                it where its three columns fit side by side. */}
            {/* The column must NOT scroll: the map sizes itself to its
                container, so an unbounded parent lets it grow past the
                viewport instead of fitting. It takes the space the inspector
                leaves, and the inspector scrolls if it needs to. */}
            <div style={{ display: 'flex', flexDirection: 'column',
                          minHeight: 0, gap: spacing.outerSpacing }}>
              <GroupBox
                title={t('phaseMap.mapTitle')}
                style={{ flex: 1, display: 'flex', flexDirection: 'column',
                         minHeight: 220 }}
              >
                {phaseMapHandle.phaseMap?.image ? (
                  /* Right-click exports the map. On the wrapper rather than on
                     the canvas: the canvas already owns its own pointer
                     gestures, and a context menu is not one of them. */
                  <div
                    style={{ display: 'flex', flexDirection: 'column',
                             flex: 1, minHeight: 0 }}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      setMenu({ x: e.clientX, y: e.clientY, kind: 'phasemap' });
                    }}
                  >
                  <PhaseMapCanvas
                    handle={phaseMapHandle}
                    onInspect={onPixelClick}
                    wand={phaseMapHandle.wand}
                    onPickStructure={inStructureView
                      ? inspector.inspectPixel : undefined}
                    onAssignPixel={phaseMapHandle.selectedPhaseIndex != null
                      ? ((row, col) => phaseMapHandle.handleAssignPixel(
                          row, col, phaseMapHandle.selectedPhaseIndex))
                      : undefined}
                  />
                  </div>
                ) : (
                  <Label secondary small>{t('tabs.noMapYet')}</Label>
                )}
              </GroupBox>
              {inStructureView && (
                <GroupBox
                  title={t('inspector.title')}
                  style={{ flexShrink: 0, maxHeight: '42vh',
                           overflowY: 'auto' }}
                >
                  <StructureInspector
                    detail={inspector.detail}
                    loading={inspector.loading}
                    error={inspector.error}
                    busy={phaseMapHandle.structureBusy}
                    phaseIndex={inspector.detail?.phase_index}
                    onAssign={(phaseIndex) => phaseMapHandle.handleAssignStructure(
                      inspector.detail.structure_id, phaseIndex)}
                    onSelectStructure={phaseMapHandle.setSelectedStructureId}
                  />
                </GroupBox>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column',
                          gap: spacing.outerSpacing, overflowY: 'auto' }}>
              <PhaseMapControls handle={phaseMapHandle} />
            </div>
          </div>
        )}

        {/* Main 3-col grid: overlay+layers | splitter | tile-grid | right rail */}
        {edsTab === 'elements' && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: `${overlayW}px 8px 1fr 280px`,
          gap: 0, flex: 1, minHeight: 0,
        }}>
          {/* LEFT: Overlay + Layer panel */}
          <div style={{
            display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing,
            paddingRight: 4, minHeight: 0, overflow: 'hidden',
          }}>
            <GroupBox title={(
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <span>{t('overlay.title')}</span>
                <span style={{ display: 'flex', gap: 6 }}>
                  <button
                    onClick={() => setMagnifierEnabled((m) => !m)}
                    title={t('overlay.lensTooltip')}
                    style={{
                      background: magnifierEnabled ? colors.cyan : 'transparent',
                      color: magnifierEnabled ? colors.bg : colors.cyan,
                      border: `1px solid ${colors.cyan}`,
                      borderRadius: 3,
                      padding: '2px 8px',
                      fontSize: '8.5pt',
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                  >
                    {magnifierEnabled ? t('overlay.lensOn') : t('overlay.lensOff')}
                  </button>
                  <button
                    onClick={() => setLinescanMode((m) => !m)}
                    title={t('overlay.linescanTooltip')}
                    style={{
                      background: linescanMode ? colors.cyan : 'transparent',
                      color: linescanMode ? colors.bg : colors.cyan,
                      border: `1px solid ${colors.cyan}`,
                      borderRadius: 3,
                      padding: '2px 8px',
                      fontSize: '8.5pt',
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                  >
                    {linescanMode ? t('overlay.linescanOn') : t('overlay.linescanOff')}
                  </button>
                  <button
                    onClick={() => exportComposite({
                      layers: stack.layers,
                      bitmaps: stack.bitmaps,
                      shape: stack.shape,
                    })}
                    disabled={!stack.shape || stack.layers.length === 0}
                    style={{
                      background: 'transparent',
                      border: `1px solid ${colors.purple}`,
                      color: colors.purple,
                      borderRadius: 3,
                      padding: '2px 8px',
                      fontSize: '8.5pt',
                      cursor: (!stack.shape || stack.layers.length === 0) ? 'not-allowed' : 'pointer',
                      fontWeight: 600,
                      opacity: (!stack.shape || stack.layers.length === 0) ? 0.5 : 1,
                    }}
                    title={t('overlay.exportPngTooltip')}
                  >
                    {t('overlay.exportPng')}
                  </button>
                </span>
              </span>
            )}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <SwipeCompareController
                  layers={stack.layers}
                  value={swipe}
                  onChange={onSwipeChange}
                />
                <OverlayCard
                  layers={stack.layers}
                  bitmaps={stack.bitmaps}
                  bitmapVersion={stack.bitmapVersion}
                  errors={stack.errors}
                  shape={stack.shape}
                  onPixelClick={onPixelClick}
                  onRegionSelected={onRegionSelected}
                  linescanMode={linescanMode}
                  onLineComplete={onLineComplete}
                  swipe={swipe}
                  onSwipeSplitChange={onSwipeSplitChange}
                  magnifierEnabled={magnifierEnabled}
                  view={overlayView}
                  onZoomAt={onOverlayZoom}
                  onPan={onOverlayPan}
                  onResetView={onOverlayResetView}
                  onContextMenu={(x, y) => setMenu({ x, y, kind: 'overlay' })}
                />
              </div>
            </GroupBox>
            {linescan.data && (
              <GroupBox title={t('linescan.title')}>
                <LinescanProfilePlot data={linescan.data} layers={stack.layers} />
              </GroupBox>
            )}
            <GroupBox title={t('layers.title')} style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <LayerStackPanel
                layers={stack.layers}
                bitmaps={stack.bitmaps}
                errors={stack.errors}
                onSetOpacity={stack.setOpacity}
                onSetBlend={stack.setBlend}
                onSetVisibility={stack.setVisibility}
                onRemove={stack.removeLayer}
                onReorder={stack.reorder}
                onAdd={onAddLayer}
                onUsePreset={() => {}}          /* presets land in a later task */
                onSetSingleLayer={onSetSingleLayer}
                availableToAdd={availableToAdd}
                renderLayerExtras={renderLayerExtras}
              />
            </GroupBox>
          </div>

          {/* SPLITTER */}
          <div onMouseDown={onSplitterMouseDown} title={t('overlay.splitterTooltip')}
            style={{ cursor: 'col-resize', position: 'relative' }}>
            <div style={{
              position: 'absolute', left: '50%', top: 0, bottom: 0, width: 2,
              background: colors.border, transform: 'translateX(-50%)',
            }} />
          </div>

          {/* CENTER: the element tile grid. The phase map moved to its own
              tab - it brought a tool set of its own, and one screen could
              not hold both. */}
          <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, paddingLeft: 4 }}>
            <GroupBox
              title={(
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', gap: 12 }}>
                  <span>{t('allMaps.title')}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: '9pt', color: colors.textSecondary, fontWeight: 400, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    <ZoomToolbar
                      mode={zoom.mode}
                      onModeChange={zoom.setMode}
                      onReset={zoom.resetAll}
                      resetDisabled={!zoom.anyZoomed}
                    />
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span>{t('allMaps.tileSize')}</span>
                    <input
                      type="range"
                      min={140} max={520} step={10}
                      value={tileMinWidth}
                      onChange={(e) => setTileMinWidth(Number(e.target.value))}
                      style={{ width: 120, accentColor: colors.purple }}
                      aria-label={t('allMaps.tileSizeAriaLabel')}
                      title={t('allMaps.tileSizeTooltip', { value: tileMinWidth })}
                    />
                    <span style={{ minWidth: 40, textAlign: 'right' }}>{t('allMaps.tileSizeValue', { value: tileMinWidth })}</span>
                    </div>
                  </div>
                </div>
              )}
              style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
            >
              <TileGrid
                layers={allMaps.layers}
                bitmaps={allMaps.bitmaps}
                errors={allMaps.errors}
                shape={allMaps.shape}
                onPixelClick={onPixelClick}
                onRegionSelected={onRegionSelected}
                minTileWidth={tileMinWidth}
                emptyMessage={t('allMaps.empty')}
                zoom={zoom}
                onTileContextMenu={(layer, x, y) => setMenu({ x, y, kind: 'tile', layer })}
              />
            </GroupBox>
          </div>

          {/* RIGHT rail: Quantify + Region + PhaseMapControls + Phase Suggestion */}
          <div style={{
            paddingLeft: 8, display: 'flex', flexDirection: 'column',
            gap: spacing.outerSpacing, overflowY: 'auto',
          }}>
            <GroupBox title={t('quantify.title')}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                  <div>
                    <Label secondary small style={{ display: 'block', marginBottom: 2 }}>{t('quantify.row')}</Label>
                    <NumberInput value={pixelRow} onChange={(e) => setPixelRow(e.target.value)} min={0} style={{ width: 68 }} title={t('hoverTips.pixelRow')} />
                  </div>
                  <div>
                    <Label secondary small style={{ display: 'block', marginBottom: 2 }}>{t('quantify.col')}</Label>
                    <NumberInput value={pixelCol} onChange={(e) => setPixelCol(e.target.value)} min={0} style={{ width: 68 }} title={t('hoverTips.pixelCol')} />
                  </div>
                  {/* Wrap in arrow fn so React's click event is not
                      forwarded as ``rowOverride`` to handleQuantify —
                      that triggered ``JSON.stringify(<event>)`` and the
                      circular-structure error visible in the screenshot. */}
                  <Button variant="warning" onClick={() => handleQuantify()} disabled={quantLoading} style={{ flexShrink: 0 }} title={t('hoverTips.quantifyRun')}>
                    {quantLoading ? <span className="btn-loading">{t('quantify.run')}</span> : t('quantify.run')}
                  </Button>
                </div>
                {quantError && <div role="alert" style={{ fontSize: '9pt', color: colors.red, animation: 'fadeSlideIn 0.2s ease-out' }}>{quantError}</div>}
                {quantData && (
                  <div style={{ overflowX: 'auto', marginTop: 4, animation: 'fadeSlideIn 0.2s ease-out' }}>
                    <QuantTable data={quantData} />
                    <button
                      onClick={() => {
                        const header = 'Element\tCounts\tWt.%\tAt.%';
                        const rows = quantData.map(r =>
                          `${r.element}\t${r.counts ?? ''}\t${r.wt_pct?.toFixed(2) ?? ''}\t${r.at_pct?.toFixed(2) ?? ''}`
                        );
                        navigator.clipboard.writeText([header, ...rows].join('\n')).then(() => {
                          setQuantCopied(true);
                          setTimeout(() => setQuantCopied(false), 1200);
                        }).catch(() => {});
                      }}
                      aria-label={t('quantify.copyAria')}
                      title={quantCopied ? t('quantify.copiedTooltip') : t('quantify.copyTooltip')}
                      style={{
                        background: 'none', border: `1px solid ${quantCopied ? colors.green : colors.border}`, borderRadius: 4,
                        color: quantCopied ? colors.green : colors.textSecondary, fontSize: '8pt', padding: '2px 8px',
                        cursor: 'pointer', marginTop: 4, width: '100%',
                        transition: 'all 0.2s',
                      }}
                    >
                      {quantCopied ? t('quantify.copied') : t('quantify.copy')}
                    </button>
                  </div>
                )}
              </div>
            </GroupBox>

            <GroupBox title={t('regionAvg.title')}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
                <Label secondary small>
                  {t('regionAvg.description')}
                </Label>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                  <div>
                    <Label secondary small style={{ display: 'block', marginBottom: 2 }}>{t('regionAvg.rowStart')}</Label>
                    <NumberInput value={regionRowStart} onChange={(e) => setRegionRowStart(e.target.value)} min={0} style={{ width: 62 }} title={t('hoverTips.regionRowStart')} />
                  </div>
                  <div>
                    <Label secondary small style={{ display: 'block', marginBottom: 2 }}>{t('regionAvg.rowEnd')}</Label>
                    <NumberInput value={regionRowEnd} onChange={(e) => setRegionRowEnd(e.target.value)} min={0} style={{ width: 62 }} title={t('hoverTips.regionRowEnd')} />
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                  <div>
                    <Label secondary small style={{ display: 'block', marginBottom: 2 }}>{t('regionAvg.colStart')}</Label>
                    <NumberInput value={regionColStart} onChange={(e) => setRegionColStart(e.target.value)} min={0} style={{ width: 62 }} title={t('hoverTips.regionColStart')} />
                  </div>
                  <div>
                    <Label secondary small style={{ display: 'block', marginBottom: 2 }}>{t('regionAvg.colEnd')}</Label>
                    <NumberInput value={regionColEnd} onChange={(e) => setRegionColEnd(e.target.value)} min={0} style={{ width: 62 }} title={t('hoverTips.regionColEnd')} />
                  </div>
                </div>
                <Button
                  variant="warning"
                  onClick={() => handleRegionQuantify()}
                  disabled={regionLoading}
                  style={{ width: '100%' }}
                  title={t('regionAvg.computeTooltip')}
                >
                  {regionLoading ? <span className="btn-loading">{t('regionAvg.compute')}</span> : t('regionAvg.compute')}
                </Button>
                {regionError && <div role="alert" style={{ fontSize: '9pt', color: colors.red, animation: 'fadeSlideIn 0.2s ease-out' }}>{regionError}</div>}
                {regionData && (
                  <div style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
                    <RegionQuantTable data={regionData} />
                    <button
                      onClick={() => {
                        const header = 'Element\tCounts (mean±σ)\tWt.% (mean±σ)\tAt.% (mean±σ)';
                        const fmtStat = (s) => {
                          if (!s) return '';
                          if (typeof s === 'number') return s.toFixed(2);
                          return `${s.mean?.toFixed(1) ?? ''} ± ${s.std?.toFixed(1) ?? ''}`;
                        };
                        const rows = Object.entries(regionData.data).map(([el, vals]) =>
                          `${el}\t${fmtStat(vals.counts)}\t${fmtStat(vals.wt_pct)}\t${fmtStat(vals.at_pct)}`
                        );
                        navigator.clipboard.writeText([header, ...rows].join('\n')).then(() => {
                          setRegionCopied(true);
                          setTimeout(() => setRegionCopied(false), 1200);
                        }).catch(() => {});
                      }}
                      aria-label={t('regionAvg.copyAria')}
                      title={regionCopied ? t('regionAvg.copiedTooltip') : t('regionAvg.copyTooltip')}
                      style={{
                        background: 'none', border: `1px solid ${regionCopied ? colors.green : colors.border}`, borderRadius: 4,
                        color: regionCopied ? colors.green : colors.textSecondary, fontSize: '8pt', padding: '2px 8px',
                        cursor: 'pointer', marginTop: 4, width: '100%',
                        transition: 'all 0.2s',
                      }}
                    >
                      {regionCopied ? t('regionAvg.copied') : t('regionAvg.copy')}
                    </button>
                  </div>
                )}
              </div>
            </GroupBox>


            <GroupBox title={t('suggest.title')} style={{ flex: 1 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
                <Label secondary small>{t('suggest.description')}</Label>
                <Button variant="primary" onClick={handleSuggestPhases} disabled={suggestLoading} style={{ width: '100%' }} title={t('hoverTips.suggestRun')}>
                  {suggestLoading ? <span className="btn-loading">{t('suggest.analyzing')}</span> : t('suggest.run')}
                </Button>
                {suggestError && <div role="alert" style={{ fontSize: '9pt', color: colors.red, marginTop: 4, animation: 'fadeSlideIn 0.2s ease-out' }}>{suggestError}</div>}
                {(suggestLibrarySource || suggestPixel) && (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, marginTop: 2, flexWrap: 'wrap' }}>
                    {suggestPixel && (
                      <span
                        style={{
                          fontSize: '8pt', color: colors.textSecondary,
                          background: alpha(colors.cyan, 10),
                          border: `1px solid ${alpha(colors.cyan, 25)}`,
                          borderRadius: 3, padding: '1px 6px', fontWeight: 600,
                          fontVariantNumeric: 'tabular-nums',
                        }}
                        title={t('suggest.atPixelTooltip')}
                      >
                        {t('suggest.atPixel', { row: suggestPixel.row, col: suggestPixel.col })}
                      </span>
                    )}
                    {suggestLibrarySource && <span style={{
                      fontSize: '8pt',
                      color: suggestLibrarySource === 'cif' ? colors.green : colors.orange,
                      background: alpha(suggestLibrarySource === 'cif' ? colors.green : colors.orange, 12),
                      border: `1px solid ${alpha(suggestLibrarySource === 'cif' ? colors.green : colors.orange, 30)}`,
                      borderRadius: 3, padding: '1px 6px', fontWeight: 600,
                    }}
                    title={suggestLibrarySource === 'cif'
                      ? t('suggest.sourceCifTooltip')
                      : t('suggest.sourceDefaultTooltip')}>
                      {suggestLibrarySource === 'cif' ? t('suggest.sourceCif') : t('suggest.sourceDefault')}
                      {suggestLibrarySize != null && ` (${suggestLibrarySize})`}
                    </span>}
                  </div>
                )}
                {/* The row/col boxes can be typed into without clicking the map,
                    which would leave the list describing a different pixel.
                    Say so instead of quietly showing the wrong pixel. */}
                {suggestStale && (
                  <div style={{ fontSize: '8pt', color: colors.orange, marginTop: 2 }}>
                    {t('suggest.staleHint', { row: Number(pixelRow), col: Number(pixelCol) })}
                  </div>
                )}
                {/* What the MAP decided here, first. The list below ranks THIS ONE
                    PIXEL; the map, in cluster mode, decides by the mean of the
                    pixel's whole composition group. Measured on SampleB the two
                    disagree on 53 % of pixels in cluster mode and 1 % in pixel
                    mode — not a bug in either, but nothing said so, and the panel
                    appeared to contradict the map. The map's answer is the more
                    reliable one: a single pixel carries several at% of error. */}
                {suggestMapPhase && (
                  <div style={{
                    marginTop: 4, padding: '6px 8px', borderRadius: 4,
                    background: alpha(colors.cyan, 8),
                    border: `1px solid ${alpha(colors.cyan, 25)}`,
                  }}>
                    <div style={{ fontSize: '8pt', color: colors.textSecondary }}>
                      {t('suggest.onTheMap')}
                    </div>
                    <div style={{ fontSize: '10pt', fontWeight: 600, color: colors.cyan,
                                 overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {suggestMapPhase.unclassified
                        ? t('phaseMap.unclassified')
                        : suggestMapPhase.cif_filename}
                      {suggestMapPhase.hand_set && ` · ${t('suggest.handSet')}`}
                    </div>
                  </div>
                )}
                {suggestedPhases && suggestedPhases.length > 0 && suggestMapPhase && (
                  <Label secondary small style={{ marginTop: 6 }}>
                    {t('suggest.thisPixelAlone')}
                  </Label>
                )}
                {suggestedPhases && (
                  <div style={{ marginTop: 6, animation: 'fadeSlideIn 0.2s ease-out' }}>
                    {suggestedPhases.length === 0 ? (
                      <Label secondary small>{t('suggest.none')}</Label>
                    ) : suggestedPhases.map((phase, i) => {
                      const isCif = !!phase.cif_filename;
                      // CIF entries: filename as primary line, formula as secondary,
                      // space group + crystal system as tertiary tag row.
                      // Default library entries: full name as primary, formula as
                      // secondary (no extra metadata available).
                      const primary = isCif ? phase.cif_filename : (phase.name || phase.phase || t('suggest.unknown'));
                      const secondary = phase.formula || '';
                      const score = typeof phase.score === 'number'
                        ? phase.score
                        : (typeof phase.confidence === 'number' ? phase.confidence : null);
                      return (
                        <div key={`${primary}-${i}`} style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          padding: '7px 10px', borderRadius: 4,
                          background: alpha(colors.green, 7),
                          border: `1px solid ${alpha(colors.green, 20)}`,
                          marginBottom: spacing.innerSpacing,
                        }}>
                          <div style={{ minWidth: 0, flex: 1 }}>
                            <div style={{
                              fontSize: '10pt', fontWeight: 600, color: colors.green,
                              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            }} title={primary}>
                              {primary}
                            </div>
                            {secondary && (
                              <div style={{ fontSize: '9pt', color: colors.textSecondary, marginTop: 1 }}>
                                {secondary}
                              </div>
                            )}
                            {/* Measured vs expected per element. Without this
                                the list is a bare ranking: a user who knows
                                the region is a Cu phase cannot see whether
                                the candidate lost on Cu, on Fe, or not at
                                all. Off-elements are the ones that decide. */}
                            {phase.expected && suggestAtPct && (
                              <div style={{
                                display: 'flex', gap: 6, marginTop: 3, flexWrap: 'wrap',
                                fontSize: '7.5pt', fontVariantNumeric: 'tabular-nums',
                              }}>
                                {Object.entries(phase.expected)
                                  .sort((a, b) => b[1] - a[1])
                                  .map(([el, exp]) => {
                                    const meas = Number(suggestAtPct[el] ?? 0);
                                    const ratio = exp > 0 ? meas / exp : 1;
                                    // Red where the element the phase needs
                                    // is largely absent — that is what
                                    // vetoes a candidate.
                                    const col = ratio < 0.3 ? colors.red
                                      : ratio < 0.6 ? colors.orange
                                      : colors.textSecondary;
                                    return (
                                      <span key={el} style={{ color: col }}
                                        title={t('suggest.elementTooltip', {
                                          el, measured: meas.toFixed(1), expected: exp.toFixed(1),
                                        })}>
                                        {el} {meas.toFixed(1)}/{exp.toFixed(0)}
                                      </span>
                                    );
                                  })}
                              </div>
                            )}
                            {isCif && (phase.space_group || phase.crystal_system) && (
                              <div style={{
                                display: 'flex', gap: 4, marginTop: 3, flexWrap: 'wrap',
                                fontSize: '7.5pt', color: colors.textSecondary,
                              }}>
                                {phase.space_group && (
                                  <span style={{ background: alpha(colors.cyan, 10), padding: '1px 5px', borderRadius: 2 }}>
                                    {phase.space_group}
                                  </span>
                                )}
                                {phase.crystal_system && (
                                  <span style={{ background: alpha(colors.purple, 10), padding: '1px 5px', borderRadius: 2 }}>
                                    {phase.crystal_system}
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                          {score != null && (
                            <div style={{
                              fontSize: '10pt', color: colors.cyan, fontWeight: 700,
                              marginLeft: 8, flexShrink: 0,
                            }}>
                              {(score * 100).toFixed(0)}%
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </GroupBox>
          </div>
        </div>
        )}

        {/* Hover-probe wiring: subscriber + overlay live in one leaf so cursor
            moves only re-render the leaf, not the entire EDSPage subtree. */}
        <HoverProbeLayer
          probe={probe}
          error={probeError}
          displayMode={displayMode}
          requestProbe={requestProbe}
          clearProbe={clearProbe}
        />

        {/* Right-click export: menu on any tile and on the composite overlay */}
        {menu && (
          <ContextMenu
            x={menu.x}
            y={menu.y}
            onClose={() => setMenu(null)}
            items={exportMenuItems(menu)}
          />
        )}
        {exportError && (
          <div
            data-eds-export-error
            role="alert"
            onClick={() => setExportError(null)}
            style={{
              position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
              zIndex: 3600, background: colors.bgSecondary,
              border: `1px solid ${colors.red}`, color: colors.red,
              borderRadius: 6, padding: '8px 14px', fontSize: '9pt', cursor: 'pointer',
            }}
          >
            {exportError}
          </div>
        )}
        {exportSrc && (
          <ImageExportDialog
            open
            onClose={() => setExportSrc(null)}
            src={exportSrc.src}
            title={exportSrc.label}
            defaultBaseName={exportSrc.name}
            unitsPerPixel={exportSrc.scale?.x ?? null}
            unitLabel={exportSrc.scale?.units || stepSize?.units || 'µm'}
            annotations={{ label: exportSrc.label }}
          />
        )}
      </div>
    </CursorSyncProvider>
  );
}
