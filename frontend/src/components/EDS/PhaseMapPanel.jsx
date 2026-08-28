/**
 * PhaseMapPanel — owns the EDS phase-map state on the EDS page.
 *
 * Responsibilities split into two render slots so the parent decides
 * the page layout:
 *   - ``renderControls`` returns the right-rail controls (auto-classify,
 *     legend, region painting, send-to-indexing).
 *   - ``renderCanvas`` returns the phase-map image + click handler for
 *     the center column. The canvas is also returned wrapped in the
 *     same letterbox-tolerant click conversion the element map uses,
 *     so a click on a 4:3 letterboxed phase map still hits the right
 *     pixel.
 *
 * The two slots share a single state instance, so toggling between
 * the element map and the phase map in the parent does NOT lose the
 * classification or the user's manual edits.
 *
 * Lifecycle:
 *   - on mount + when the file path changes, query GET /phase-map so
 *     the panel reflects whatever the backend already has (e.g. after
 *     a page navigation away-and-back).
 *   - the backend's ``close_file()`` hook clears the store on file
 *     switch, so a stale phase map can never bleed into a new file.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { edsApi } from '../../services/api';
import {
  colors as C, alpha,
  Button, NumberInput, GroupBox, Label,
} from '../../theme/components';
import useDataStore from '../../stores/useDataStore';
import usePhaseColorStore from '../../stores/usePhaseColorStore';
import { pointerToRowCol } from './mapCoords';
import {
  IDENTITY_VIEW, isZoomed, viewToTransform, wheelFactor, zoomedRect,
} from './zoomView';
import RegionPanel from './RegionPanel';

/** Store key for a phase colour.
 *
 * The EBSD phase map names a phase by the file stem, the EDS candidate list
 * carries the full filename. Stripping `.cif` makes them the same key, so a
 * colour picked on either page shows on both. The backend normalises the
 * same way (`phase_map_store._norm_phase_name`).
 */
export function phaseNameKey(cifFilename) {
  const n = String(cifFilename || '').trim();
  return n.toLowerCase().endsWith('.cif') ? n.slice(0, -4) : n;
}
import WandOverlay from './WandOverlay';
import { useEdsWand } from './hooks/useEdsWand';

const DEFAULT_TOLERANCE = 15.0;
const DEFAULT_MIN_SCORE = 0.3;

/**
 * usePhaseMap — small custom hook so EDSPage doesn't grow a dozen
 * useState lines just to host the phase map. Returns a stable handle
 * with state + actions; consumers use the helpers, never the setters.
 */
export function usePhaseMap({ onIndexingHandoff } = {}) {
  const { t } = useTranslation('eds');
  const filePath = useDataStore(s => s.filePath);

  const [phaseMap, setPhaseMap] = useState(null);   // GET /phase-map response
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tolerance, setTolerance] = useState(DEFAULT_TOLERANCE);
  const [minScore, setMinScore] = useState(DEFAULT_MIN_SCORE);

  // Manual region paint state (M4)
  const [selectedPhaseIndex, setSelectedPhaseIndex] = useState(null);
  const [region, setRegion] = useState({ rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 });
  const [assignBusy, setAssignBusy] = useState(false);

  // Paint mode: 'rectangle' uses drag-to-rect (M4-rect), 'polygon' uses
  // click-to-add-vertex with explicit close (added later — rectangle is
  // good for axis-aligned regions, polygon for irregular grain
  // outlines).
  const [paintMode, setPaintMode] = useState('rectangle');  // 'rectangle' | 'polygon' | 'wand'
  const wand = useEdsWand();
  const [replaceFrom, setReplaceFrom] = useState(null);
  // Regions first: the classification groups pixels by composition alone,
  // and naming them is a separate act. 'regions' | 'phases'.
  const [mapView, setMapView] = useState('regions');
  const [selectedRegionId, setSelectedRegionId] = useState(null);
  // Smoothing box width for the clustering. null = the backend default (5).
  // Without it KMeans returns confetti and the cluster count has to stay tiny
  // to hide that - measured at k=8 on SampleB: 3491 pieces, median size 1 px.
  const [scale, setScale] = useState(null);
  const [regionBusy, setRegionBusy] = useState(false);
  // User-authored rules. They decide which phases may COMPETE for a region,
  // and they only take effect on the next classification - like the scale and
  // the region count, and for the same reason: changing them silently under
  // an existing map would leave the picture and its explanation disagreeing.
  const [rules, setRules] = useState(null);
  // Hand-declared regions and per-element clustering weights. Both are
  // opt-in: empty means the automatic grouping runs exactly as before.
  const [regionDefs, setRegionDefs] = useState([]);
  const [elementWeights, setElementWeights] = useState({});
  const [clusterRemainder, setClusterRemainder] = useState(true);
  // Bumped by anything that changes the grouping. Merging renumbers
  // ids, splitting adds them, a boundary move changes the pixels — the
  // inspector has to re-read or it describes a region that is gone.
  const [mapVersion, setMapVersion] = useState(0);
  // The SAME store the EBSD phase map uses, so a phase keeps its colour on
  // both pages and the user's choice survives a reload.
  const colorOverrides = usePhaseColorStore((s) => s.overrides);
  const setPhaseColor = usePhaseColorStore((s) => s.setColor);
  const resetPhaseColor = usePhaseColorStore((s) => s.resetColor);
  const [polygonVertices, setPolygonVertices] = useState([]);  // [[col, row], ...]

  // Last clicked pixel (for the readout under the canvas)
  const [hoveredPixel, setHoveredPixel] = useState(null);

  // Grouping mode. 'cluster' groups the composition and matches each
  // group's mean; 'pixel' matches every pixel on its own. Cluster is the
  // default because per-pixel composition on real data carries several
  // at% of systematic error (see the 2026-08-19 design spec).
  const [mode, setMode] = useState('cluster');
  const [nClusters, setNClusters] = useState(null);   // null -> backend picks it

  // Which library phases take part. Empty set = "not loaded yet"; the
  // request only narrows when a STRICT subset is ticked, so a stale list
  // can never silently drop phases from a run.
  const [cifPhases, setCifPhases] = useState([]);
  const [selectedPhaseKeys, setSelectedPhaseKeys] = useState(new Set());

  const loadCifPhases = useCallback(async () => {
    try {
      const res = await edsApi.cifPhases();
      const phases = res.data?.phases || [];
      setCifPhases(phases);
      setSelectedPhaseKeys(new Set(phases.map(p => p.key)));
      return phases;
    } catch {
      setCifPhases([]);
      return [];
    }
  }, []);

  // Load the phase list once per file so the picker has something to show
  // before the first classification.
  useEffect(() => { loadCifPhases(); }, [loadCifPhases, filePath]);

  /**
   * Take the region definitions and weights a stored map was built with.
   *
   * Only ever ADDS: a map that carries none leaves the editor as it is, so
   * arriving at a page whose backend map predates this feature cannot wipe
   * definitions the user is halfway through writing. Called only where the
   * map came from disk — after an action the user just took, the editor
   * already holds the truth and the round trip would only fight it.
   */
  const adoptMapProvenance = useCallback((data) => {
    if (!data) return;
    if (Array.isArray(data.region_defs) && data.region_defs.length) {
      setRegionDefs(data.region_defs);
    }
    if (data.element_weights && Object.keys(data.element_weights).length) {
      setElementWeights(data.element_weights);
    }
  }, []);

  // On mount or file change, sync from backend. The backend clears its
  // store on close_file(), so a 200-with-loaded:false response is the
  // expected "nothing to show" state, not an error.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    // Windows are written in at% against ONE dataset's chemistry. Carrying
    // them to the next file leaves numbers that mean something else silently
    // armed, and the next Classify applies them. Whatever the new file's own
    // stored map carries is adopted below.
    setRegionDefs([]);
    setElementWeights({});
    setClusterRemainder(true);
    edsApi.getPhaseMap(true)
      .then((res) => {
        if (cancelled) return;
        setPhaseMap(res.data?.loaded ? res.data : null);
        if (res.data?.loaded) adoptMapProvenance(res.data);
      })
      .catch(() => { /* 4xx is fine — just means nothing classified yet */ });
    return () => { cancelled = true; };
  }, [filePath, adoptMapProvenance]);

  // Push the colour choices to the backend, which renders the PNG. Runs on
  // mount too: the store is persisted, so a returning user's colours must
  // reach a freshly started backend before the first render.
  useEffect(() => {
    let cancelled = false;
    edsApi.setPhaseColors(colorOverrides)
      .then((res) => {
        // Deliberately NOT adopting provenance here. This effect runs on
        // every colour change, and adopting would replace a definition the
        // user just deleted or a threshold they just retyped with the last
        // classified map's copy - an edit silently rolled back by clicking a
        // swatch. Adoption belongs where the map arrives from disk.
        if (!cancelled && res.data?.loaded) setPhaseMap(res.data);
      })
      .catch(() => { /* colours are a preference; never break the map */ });
    return () => { cancelled = true; };
  }, [colorOverrides]);

  // Every region tool answers with the whole refreshed map, so they share
  // one caller. Keeping them separate would mean five copies of the same
  // error handling and the same chance for one of them to drift.
  const runRegionOp = useCallback(async (fn, args) => {
    setRegionBusy(true);
    setError(null);
    try {
      const res = await fn(args);
      if (res.data?.loaded) setPhaseMap(res.data);
      setMapVersion((v) => v + 1);
      return res.data;
    } catch (e) {
      setError(e.response?.data?.detail || String(e));
      return null;
    } finally {
      setRegionBusy(false);
    }
  }, []);

  const handleAssignRegionPhase = useCallback((regionId, phaseIndex) =>
    runRegionOp(edsApi.assignRegionPhase, { regionId, phaseIndex }),
  [runRegionOp]);

  const handleMergeRegions = useCallback((keepId, dropId) =>
    runRegionOp(edsApi.mergeRegions, { keepId, dropId }),
  [runRegionOp]);

  const handleSplitRegion = useCallback((regionId, nParts = 2) =>
    runRegionOp(edsApi.splitRegion, { regionId, nParts }),
  [runRegionOp]);

  const handleGrowRegion = useCallback((regionId, nPixels) =>
    runRegionOp(edsApi.growRegion, { regionId, nPixels }),
  [runRegionOp]);

  const handleSnapEdges = useCallback((strength) =>
    runRegionOp(edsApi.snapRegionEdges, { strength }),
  [runRegionOp]);

  const handleAutoClassify = useCallback(async () => {
    // "None selected" must not run the whole library — that inverts the
    // clearest instruction the user can give. Refuse instead.
    if (cifPhases.length > 0 && selectedPhaseKeys.size === 0) {
      setError(t('phaseMap.noneSelected'));
      return;
    }
    setLoading(true); setError(null);
    try {
      // Only send phase_keys for a STRICT subset — an all-selected list
      // means "everything", and sending it would freeze the run against a
      // library the user may since have extended.
      const isSubset = cifPhases.length > 0
        && selectedPhaseKeys.size > 0
        && selectedPhaseKeys.size < cifPhases.length;
      const res = await edsApi.autoClassify({
        tolerance,
        minScore,
        mode,
        nClusters,
        ...(scale != null ? { scale } : {}),
        ...(rules && (rules.rules?.length || rules.phase_keys) ? { rules } : {}),
        ...(isSubset ? { phaseKeys: [...selectedPhaseKeys] } : {}),
        elementWeights,
        // A definition with no clause in it claims nothing on the backend,
        // but sending it would still flip the run onto the manual path and
        // put every pixel in the leftover region. Drop the half-written
        // ones here so an in-progress edit cannot wipe the map.
        regionDefs: regionDefs.filter(
          (d) => d.elements?.length || d.ratios?.length || d.enrichment?.length),
        clusterRemainder,
      });
      setPhaseMap(res.data);
      setSelectedPhaseIndex(null);
      setSelectedRegionId(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorClassify'));
    } finally {
      setLoading(false);
    }
  }, [tolerance, minScore, mode, nClusters, scale, rules, regionDefs,
      elementWeights, clusterRemainder, cifPhases, selectedPhaseKeys, t]);

  const handleClearMap = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      await edsApi.clearPhaseMap();
      setPhaseMap(null);
      setSelectedPhaseIndex(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorClear'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  // Assign the single clicked pixel. Goes through the same endpoint as a
  // rectangle (r0==r1, c0==c1) so it inherits the grid-mismatch guard and
  // the locked_mask bookkeeping rather than opening a second write path.
  const handleAssignPixel = useCallback(async (row, col, phaseIndex) => {
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.assignRegion(
        Number(row), Number(row), Number(col), Number(col), Number(phaseIndex));
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    } finally {
      setAssignBusy(false);
    }
  }, [t]);

  const handleUndo = useCallback(async () => {
    setError(null);
    try {
      const res = await edsApi.phaseMapUndo();
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    }
  }, [t]);

  // The correction a seeded selection cannot make: the recorded case is
  // "sd_0302719 won 55 % of my map and it should be Al". Lassoing 55 % of a
  // map by hand is not a workflow.
  const handleReplacePhase = useCallback(async (fromIdx, toIdx) => {
    if (fromIdx == null || toIdx == null || fromIdx === toIdx) return;
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.replacePhase(Number(fromIdx), Number(toIdx));
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    } finally {
      setAssignBusy(false);
    }
  }, [t]);

  const handleAssignRegion = useCallback(async (phaseIndex) => {
    if (!phaseMap) return;
    const r = region;
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.assignRegion(
        Number(r.rowStart), Number(r.rowEnd),
        Number(r.colStart), Number(r.colEnd),
        Number(phaseIndex),
      );
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    } finally {
      setAssignBusy(false);
    }
  }, [phaseMap, region, t]);

  const handleSendToIndexing = useCallback(() => {
    if (!phaseMap) return;
    onIndexingHandoff?.(phaseMap);
  }, [phaseMap, onIndexingHandoff]);

  const handleClosePolygon = useCallback(async (phaseIndex) => {
    if (polygonVertices.length < 3) return;
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.assignPolygon(polygonVertices, Number(phaseIndex));
      setPhaseMap(res.data);
      setPolygonVertices([]);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorPolygon'));
    } finally {
      setAssignBusy(false);
    }
  }, [polygonVertices, t]);

  const handleCancelPolygon = useCallback(() => {
    setPolygonVertices([]);
  }, []);

  return {
    phaseMap, loading, error,
    tolerance, setTolerance, minScore, setMinScore,
    mode, setMode, nClusters, setNClusters,
    cifPhases, loadCifPhases,
    colorOverrides, setPhaseColor, resetPhaseColor,
    mapView, setMapView,
    selectedRegionId, setSelectedRegionId,
    scale, setScale, regionBusy, mapVersion,
    rules, setRules,
    regionDefs, setRegionDefs,
    elementWeights, setElementWeights,
    clusterRemainder, setClusterRemainder,
    handleAssignRegionPhase, handleMergeRegions, handleSplitRegion,
    handleGrowRegion, handleSnapEdges,
    selectedPhaseKeys, setSelectedPhaseKeys,
    selectedPhaseIndex, setSelectedPhaseIndex,
    replaceFrom, setReplaceFrom,
    region, setRegion,
    assignBusy,
    hoveredPixel, setHoveredPixel,
    paintMode, setPaintMode,
    wand,
    // The wand's "Assign" handler writes the returned map straight back.
    setPhaseMap,
    polygonVertices, setPolygonVertices,
    handleAutoClassify, handleClearMap,
    handleAssignRegion, handleAssignPixel, handleUndo, handleReplacePhase,
    handleClosePolygon, handleCancelPolygon,
    handleSendToIndexing,
  };
}


/**
 * Convert a click on the phase map to (row, col), through the zoom.
 *
 * `hostRect` must come from an element that is NOT itself transformed:
 * `getBoundingClientRect()` already includes a CSS transform, so measuring the
 * zoomed node would count the zoom twice. `zoomedRect` turns that untransformed
 * box into the virtual box the pixels are actually drawn in, and
 * `mapCoords.pointerToRowCol` does the letterbox arithmetic — the same routine
 * the element tiles use, rather than a second copy of it that can drift.
 */
export function pixelFromClickAt(e, hostRect, view, nRows, nCols) {
  if (!hostRect || !hostRect.width || !hostRect.height) return null;
  if (!nRows || !nCols) return null;
  return pointerToRowCol(e, zoomedRect(hostRect, view || IDENTITY_VIEW),
                         [nRows, nCols]);
}


/**
 * PhaseMapCanvas — renders the colored phase map and supports two
 * mouse interactions on it:
 *
 *   - **Click** = inspect pixel: store the (row, col) in the handle
 *     so the controls panel can render a per-pixel readout (which
 *     phase, which score) without a round-trip.
 *   - **Drag** = paint a rectangular region: write start/end into the
 *     ``region`` state. The overlay rectangle uses an SVG layer with
 *     ``viewBox="0 0 nCols nRows"`` and the same
 *     ``preserveAspectRatio`` as the img's ``objectFit: contain``, so
 *     the rectangle always lines up exactly with the underlying
 *     pixels — even when the image is letterboxed.
 *
 * No phase map → empty placeholder so the parent can still slot the
 * canvas into the layout without flicker.
 */
export function PhaseMapCanvas({ handle, onInspect, onAssignPixel, wand,
                                onPickRegion, view, onZoomAt, onPan,
                                claimOverlay,
                                onResetView, background }) {
  const { t } = useTranslation('eds');
  const {
    phaseMap, hoveredPixel, setHoveredPixel,
    region, setRegion,
    paintMode, polygonVertices, setPolygonVertices,
    mapView,
  } = handle;

  // The region view draws its own image. Falling back to the phase image
  // keeps a map made before regions existed (or a per-pixel run, which has
  // no groups at all) visible instead of blank.
  const shownImage = (mapView === 'regions' && phaseMap?.region_image)
    ? phaseMap.region_image
    : phaseMap?.image;
  const [drag, setDrag] = useState(null);  // { startRow, startCol, endRow, endCol } | null
  // Measured for the pointer maths; deliberately the UNtransformed box.
  const hostRef = useRef(null);
  const [panFrom, setPanFrom] = useState(null);
  const activeView = view || IDENTITY_VIEW;
  // The wheel listener is attached natively (it has to be non-passive to
  // preventDefault), so it reads the view through a ref rather than
  // closing over a stale one.
  const viewRef = useRef(activeView);
  viewRef.current = activeView;
  const zoomed = isZoomed(activeView);

  // Read the shape from the payload rather than from `nRows`/`nCols`: those
  // are declared after this component's early return, so referencing them here
  // would be a use-before-define at render time.
  // Ctrl+wheel zooms to the cursor, matching the element tiles. Attached
  // natively because React's synthetic wheel handler is passive and cannot
  // preventDefault, which the browser needs to not zoom the whole page.
  // A plain wheel is left alone so the panel can still be scrolled.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !onZoomAt) return undefined;
    const onWheel = (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const rect = host.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      onZoomAt(
        wheelFactor(e.deltaY),
        (e.clientX - rect.left) / rect.width,
        (e.clientY - rect.top) / rect.height,
      );
    };
    host.addEventListener('wheel', onWheel, { passive: false });
    return () => host.removeEventListener('wheel', onWheel);
  }, [onZoomAt]);

  const pixelFromClick = useCallback((e) => pixelFromClickAt(
    e, hostRef.current?.getBoundingClientRect(), viewRef.current,
    phaseMap?.n_rows, phaseMap?.n_cols,
  ), [phaseMap?.n_rows, phaseMap?.n_cols]);

  const isPolygon = paintMode === 'polygon';
  const isWand = paintMode === 'wand';

  /** Drag with Ctrl (or in pan mode) moves a zoomed view instead of painting. */
  const beginPan = useCallback((e) => {
    setPanFrom({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (onPan && zoomed && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      beginPan(e);
      return;
    }
    if (isPolygon) return;  // polygon mode uses click-to-add-vertex, not drag
    const px = pixelFromClick(e);
    if (!px || !phaseMap) return;
    e.preventDefault();
    setDrag({ startRow: px.row, startCol: px.col, endRow: px.row, endCol: px.col });
  }, [phaseMap, isPolygon, onPan, zoomed, beginPan]);

  const handleMouseMove = useCallback((e) => {
    if (panFrom) {
      const host = hostRef.current;
      if (host && onPan) {
        const rect = host.getBoundingClientRect();
        // Deltas relative to the LAST point, not the drag origin: the
        // controller accumulates, so sending the whole offset each move
        // would accelerate the pan quadratically.
        onPan(
          -(e.clientX - panFrom.x) / Math.max(1, rect.width),
          -(e.clientY - panFrom.y) / Math.max(1, rect.height),
        );
        setPanFrom({ ...panFrom, x: e.clientX, y: e.clientY });
      }
      return;
    }
    if (!drag) return;
    const px = pixelFromClick(e);
    if (!px) return;
    setDrag(d => d && { ...d, endRow: px.row, endCol: px.col });
  }, [drag, panFrom, onPan]);

  const handleMouseUp = useCallback((e) => {
    if (panFrom) { setPanFrom(null); return; }
    if (!drag) return;
    const px = pixelFromClick(e);
    const endRow = px ? px.row : drag.endRow;
    const endCol = px ? px.col : drag.endCol;
    const sameSpot = endRow === drag.startRow && endCol === drag.startCol;
    if (sameSpot) {
      // Treat as a click — don't create a 1-pixel "region", just inspect.
      setHoveredPixel({ row: endRow, col: endCol });
      // Drive the SAME per-pixel panels every other map on this page drives.
      // Without this the phase map was the one clickable surface that did
      // not move Pixel Quantification or Phase Suggestion, so the
      // suggestion silently kept describing a pixel picked somewhere else —
      // a user clicking a Cu-rich region got the candidate list for
      // wherever they last clicked a tile.
      onInspect?.(endRow, endCol);
      if (onPickRegion) {
        // Region view: the click selects the region under it. It must NOT
        // fall through to the single-pixel paint below — that is the phase
        // view's gesture, and here it would quietly hand-edit one pixel
        // instead of picking the region the user aimed at.
        onPickRegion(endRow, endCol);
      } else if (isWand) {
        // Wand mode: the click seeds a selection instead of assigning.
        wand?.seedAt(endRow, endCol);
      } else {
        // Armed phase -> the click also assigns that single pixel.
        onAssignPixel?.(endRow, endCol);
      }
    } else {
      const r0 = Math.min(drag.startRow, endRow);
      const r1 = Math.max(drag.startRow, endRow);
      const c0 = Math.min(drag.startCol, endCol);
      const c1 = Math.max(drag.startCol, endCol);
      setRegion({ rowStart: r0, rowEnd: r1, colStart: c0, colEnd: c1 });
      setHoveredPixel(null);
    }
    setDrag(null);
  }, [drag, setHoveredPixel, setRegion, onInspect, onAssignPixel, isWand, wand,
      onPickRegion]);

  const handleDoubleClick = useCallback(() => {
    if (onResetView) onResetView();
  }, [onResetView]);

  const handleMouseLeave = useCallback(() => {
    if (panFrom) setPanFrom(null);
    // Cancel the drag if the mouse leaves the image — otherwise a
    // mouseup outside the canvas would commit a stale rectangle.
    if (drag) setDrag(null);
  }, [drag]);

  const handleClick = useCallback((e) => {
    // Polygon mode only — rectangle mode handles selection via
    // mousedown/up. Each click adds a vertex; closing happens via
    // the explicit "Close polygon" button in the controls panel.
    if (!isPolygon) return;
    const px = pixelFromClick(e);
    if (!px) return;
    e.preventDefault();
    setPolygonVertices(prev => [...prev, [px.col, px.row]]);
  }, [isPolygon, setPolygonVertices]);

  if (!phaseMap?.image) {
    return (
      <div style={{
        flex: 1, background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: 250, color: C.textSecondary, textAlign: 'center', padding: 20,
      }}>
        <div>
          <div style={{ fontSize: '20pt', opacity: 0.3, marginBottom: 8 }}>{'□'}</div>
          <div style={{ fontSize: '10pt' }}>{t('phaseMap.noMapTitle')}</div>
          <div style={{ fontSize: '9pt', marginTop: 4, opacity: 0.7 }}>
            {t('phaseMap.noMapHintPre')}<b>{t('phaseMap.noMapHintAction')}</b>{t('phaseMap.noMapHintPost')}
          </div>
        </div>
      </div>
    );
  }

  const nRows = phaseMap.n_rows;
  const nCols = phaseMap.n_cols;

  // Pick which rectangle to draw: the live drag (during mousedown/move)
  // overrides any committed region so the user gets immediate feedback.
  const liveRect = drag
    ? {
        x: Math.min(drag.startCol, drag.endCol),
        y: Math.min(drag.startRow, drag.endRow),
        w: Math.abs(drag.endCol - drag.startCol) + 1,
        h: Math.abs(drag.endRow - drag.startRow) + 1,
      }
    : (region && (region.rowEnd > region.rowStart || region.colEnd > region.colStart))
      ? {
          x: region.colStart,
          y: region.rowStart,
          w: region.colEnd - region.colStart + 1,
          h: region.rowEnd - region.rowStart + 1,
        }
      : null;

  let readout = null;
  if (hoveredPixel) {
    const { row, col } = hoveredPixel;
    readout = (
      <div style={{
        position: 'absolute', bottom: 4, left: 4, right: 4,
        background: alpha(C.bg, 80), border: `1px solid ${C.border}`,
        borderRadius: 4, padding: '4px 8px', fontSize: '9pt',
        color: C.text, pointerEvents: 'none',
      }}>
        {t('phaseMap.readout', { row, col })}
      </div>
    );
  }

  return (
    <div style={{
      flex: 1, background: C.bgSecondary, border: `1px solid ${C.border}`,
      borderRadius: 6, position: 'relative',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      overflow: 'hidden', minHeight: 250,
    }}>
      {/* Image + overlay both stretch to the same logical box, so the
          SVG's preserveAspectRatio aligns the rectangle to the pixel
          grid regardless of how the image is letterboxed. The wrapper
          fills the container in BOTH dimensions — using only
          ``maxWidth/maxHeight`` would let a small EBSD grid (e.g.
          270×120) sit as a postage-stamp in the middle while the
          panel has 1500×800 px of empty real estate around it.
          ``objectFit: contain`` on the img + matching
          ``preserveAspectRatio: xMidYMid meet`` on the SVG keeps both
          letterbox calculations identical, so the rectangle / polygon
          overlay never drifts off the underlying pixels. */}
      {/* Two nested boxes on purpose. The OUTER one is what the pointer
          maths measures and what the wheel listener hangs on - it never
          moves, so getBoundingClientRect() stays the UNtransformed box and
          the zoom is not counted twice. The INNER one carries the
          transform, and the image, the wand overlay and the
          rectangle/polygon SVG all live inside it so they move together
          and cannot drift apart. */}
      <div
        ref={hostRef}
        onDoubleClick={handleDoubleClick}
        style={{
          position: 'relative', width: '100%', height: '100%',
          overflow: 'hidden',
          cursor: panFrom ? 'grabbing' : undefined,
        }}
      >
      <div style={{
        position: 'relative', width: '100%', height: '100%',
        transform: viewToTransform(activeView),
        transformOrigin: '0 0',
      }}>
        {/* Background first, the map over it. Both use objectFit:contain in the
            same box, so they register as long as they cover the same field of
            view - measured 60.00 um against 60.39 um on a real file. The MAP is
            what fades, not the background: the map is the thing being placed,
            and fading the background instead would leave the categorical
            colours at full strength over a washed-out image. */}
        {background?.image && (
          <img
            src={`data:image/png;base64,${background.image}`}
            alt=""
            aria-hidden="true"
            draggable={false}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%', objectFit: 'contain',
              borderRadius: 4, display: 'block', pointerEvents: 'none',
              userSelect: 'none',
            }}
          />
        )}
        <img
          src={`data:image/png;base64,${shownImage}`}
          alt={t('phaseMap.canvasAltText')}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          draggable={false}
          title={isPolygon
            ? t('phaseMap.canvasPolygonTooltip')
            : t('phaseMap.canvasRectTooltip')}
          style={{
            width: '100%', height: '100%', objectFit: 'contain',
            borderRadius: 4, cursor: 'crosshair',
            opacity: background?.image ? background.opacity : 1,
            // Phase maps are categorical — interpolation across class
            // boundaries would render garbage colours.
            imageRendering: 'pixelated',
            display: 'block',
            userSelect: 'none',
          }}
        />
        {/* What the last "Try it" claimed, laid over the map. A count
            cannot tell you whether you caught the right pixels - 202 px of
            matrix and 202 px of particle read identically - and this is the
            same answer as a picture, before anything is committed. Pinned
            at 0.75 rather than full: it has to read as an ANSWER ABOUT the
            map, not as the map. */}
        {claimOverlay && (
          <img
            src={`data:image/png;base64,${claimOverlay}`}
            alt=""
            aria-hidden="true"
            draggable={false}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%', objectFit: 'contain',
              imageRendering: 'pixelated', opacity: 0.75,
              pointerEvents: 'none', userSelect: 'none', display: 'block',
            }}
          />
        )}
        {isWand && wand?.mask && (
          <WandOverlay mask={wand.mask} shape={wand.shape} seed={wand.seed} />
        )}
        {(liveRect || (isPolygon && polygonVertices.length > 0)) && (
          <svg
            viewBox={`0 0 ${nCols} ${nRows}`}
            preserveAspectRatio="xMidYMid meet"
            style={{
              position: 'absolute', inset: 0, pointerEvents: 'none',
              width: '100%', height: '100%',
            }}
          >
            {liveRect && !isPolygon && (
              /* Drawn as a dark casing under a white dashed line ("marching
                 ants") instead of one tinted stroke. A single cyan outline
                 was invisible on this very map: the phase colours are a
                 golden-ratio HSV walk, so a fixed accent colour lands on a
                 near-match sooner or later — cyan on the teal Al-Fe-Mn-Si
                 region had almost no contrast. Black-under-white reads on
                 any fill. The old width also collapsed to ~1 screen px,
                 because non-scaling-stroke makes strokeWidth a SCREEN
                 length while the value was computed in viewBox units. */
              <>
                <rect
                  x={liveRect.x} y={liveRect.y}
                  width={liveRect.w} height={liveRect.h}
                  fill="rgba(255, 255, 255, 0.16)"
                  stroke="rgba(0, 0, 0, 0.85)"
                  strokeWidth={4}
                  vectorEffect="non-scaling-stroke"
                />
                <rect
                  x={liveRect.x} y={liveRect.y}
                  width={liveRect.w} height={liveRect.h}
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  vectorEffect="non-scaling-stroke"
                />
              </>
            )}
            {isPolygon && polygonVertices.length > 0 && (
              <>
                {/* Filled polygon (only when closed) gives a preview of
                    what the assign will paint. While the polygon is
                    still open we show only the polyline so the user
                    can tell open vs. closed at a glance. */}
                {/* Same black-under-white treatment as the rectangle, for
                    the same reason: readable over any phase colour. */}
                {polygonVertices.length >= 3 && (
                  <>
                    <polygon
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="rgba(255, 255, 255, 0.16)"
                      stroke="rgba(0, 0, 0, 0.85)"
                      strokeWidth={4}
                      vectorEffect="non-scaling-stroke"
                    />
                    <polygon
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      vectorEffect="non-scaling-stroke"
                    />
                  </>
                )}
                {polygonVertices.length < 3 && (
                  <>
                    <polyline
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="none"
                      stroke="rgba(0, 0, 0, 0.85)"
                      strokeWidth={4}
                      vectorEffect="non-scaling-stroke"
                    />
                    <polyline
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      vectorEffect="non-scaling-stroke"
                    />
                  </>
                )}
                {polygonVertices.map(([x, y], i) => (
                  <circle
                    key={`v${i}`}
                    cx={x + 0.5} cy={y + 0.5}
                    r={Math.max(1.2, Math.min(nCols, nRows) / 90)}
                    fill="#ffffff"
                    stroke="rgba(0, 0, 0, 0.85)"
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
              </>
            )}
          </svg>
        )}
      </div>
      </div>
      {readout}
    </div>
  );
}


/** PhaseMapControls — controls + legend + region painting in the right rail. */
export function PhaseMapControls({ handle }) {
  const { t } = useTranslation('eds');
  const {
    phaseMap, setPhaseMap, loading, error,
    tolerance, setTolerance, minScore, setMinScore,
    mode, setMode, nClusters, setNClusters,
    cifPhases, selectedPhaseKeys, setSelectedPhaseKeys,
    selectedPhaseIndex, setSelectedPhaseIndex,
    replaceFrom, setReplaceFrom,
    region, setRegion,
    assignBusy,
    paintMode, setPaintMode,
    // Without this the first render in wand mode threw ReferenceError and the
    // ErrorBoundary replaced the entire EDS page — one click was enough.
    wand,
    polygonVertices,
    handleAutoClassify, handleClearMap,
    handleAssignRegion, handleAssignPixel, handleUndo, handleReplacePhase,
    handleClosePolygon, handleCancelPolygon,
    handleSendToIndexing,
    colorOverrides, setPhaseColor, resetPhaseColor,
    mapView, setMapView,
  } = handle;

  const hasMap = !!phaseMap?.loaded;
  const summary = phaseMap?.summary || [];
  const realPhases = summary.filter(s => !s.is_unclassified);
  const unclassifiedRow = summary.find(s => s.is_unclassified);
  const clusters = phaseMap?.clusters || [];
  const isWandMode = paintMode === 'wand';
  // The region view only exists when there ARE regions. A per-pixel run
  // has no groups and an old map predates them; without this the toggle is
  // hidden AND the phase legend is hidden, leaving no legend at all.
  const showRegions = mapView === 'regions'
    && (phaseMap?.regions || []).length > 0;
  // The legend doubles as the phase PICKER, and `summary` deliberately omits
  // phases with zero pixels. That made the one phase a manual correction is
  // usually FOR — "this particle is beta-AlFeSi, the classifier missed it" —
  // impossible to select. `all_phases` carries every candidate.
  const allPhases = phaseMap?.all_phases || [];
  const unplacedPhases = allPhases.filter(p => p.n_pixels === 0);

  const togglePhase = (key) => {
    const next = new Set(selectedPhaseKeys);
    if (next.has(key)) next.delete(key); else next.add(key);
    setSelectedPhaseKeys(next);
  };

  return (
    <GroupBox title={t('phaseMap.title')}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {/* --- Grouping mode. The rail is a fixed 280 px, so this row wraps
                rather than overflowing it — an earlier version pushed the
                cluster-count box 22 px past the edge. --- */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <Label secondary small title={t('phaseMap.modeTooltip')}>
            {t('phaseMap.grouping')}
          </Label>
          <div style={{ display: 'flex', gap: 0, flexShrink: 0 }} role="group" aria-label={t('phaseMap.grouping')}>
            {[['cluster', t('phaseMap.modeCluster')], ['pixel', t('phaseMap.modePixel')]].map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id)}
                aria-pressed={mode === id}
                title={t('phaseMap.modeTooltip')}
                style={{
                  padding: '2px 10px', fontSize: '8.5pt', cursor: 'pointer',
                  border: `1px solid ${alpha(C.purple, mode === id ? 70 : 25)}`,
                  background: mode === id ? alpha(C.purple, 30) : 'transparent',
                  color: mode === id ? C.text : C.textSecondary,
                  borderRadius: id === 'cluster' ? '3px 0 0 3px' : '0 3px 3px 0',
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {mode === 'cluster' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 0 }}>
              <Label secondary small title={t('phaseMap.nClustersTooltip')}>
                {t('phaseMap.nClusterCount')}
              </Label>
              <input
                type="number" min={2} max={20}
                value={nClusters ?? ''}
                placeholder={t('phaseMap.nClustersAuto')}
                onChange={(e) => setNClusters(e.target.value === '' ? null : Number(e.target.value))}
                title={t('phaseMap.nClustersTooltip')}
                style={{
                  width: 58, minWidth: 0, flexShrink: 1,
                  fontSize: '8.5pt', padding: '1px 4px',
                  background: 'transparent', color: C.text,
                  border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
                }}
              />
            </div>
          )}
        </div>

        {/* --- Tolerance + min-score. Tolerance only affects the legacy
                per-pixel rule; hide it in cluster mode rather than show a
                slider that does nothing. --- */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          {mode === 'pixel' && (
            <div style={{ flex: 1 }}>
              <Label secondary small style={{ display: 'block', marginBottom: 2 }} title={t('phaseMap.toleranceTooltip')}>
                {t('phaseMap.tolerance', { value: tolerance.toFixed(1) })}
              </Label>
              <input
                type="range" min={1} max={40} step={0.5}
                value={tolerance}
                onChange={(e) => setTolerance(Number(e.target.value))}
                style={{ width: '100%', accentColor: C.purple }}
                title={t('hoverTips.toleranceSlider')}
              />
            </div>
          )}
          <div style={{ flex: 1 }}>
            <Label secondary small style={{ display: 'block', marginBottom: 2 }} title={t('phaseMap.minScoreTooltip')}>
              {t('phaseMap.minScore', { value: minScore.toFixed(2) })}
            </Label>
            <input
              type="range" min={0} max={1} step={0.05}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              style={{ width: '100%', accentColor: C.purple }}
              title={t('hoverTips.minScoreSlider')}
            />
          </div>
        </div>

        {/* --- Which phases take part --- */}
        {cifPhases.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{
              cursor: 'pointer', fontSize: '8.5pt', color: C.textSecondary,
              userSelect: 'none',
            }} title={t('phaseMap.phaseSelectionTooltip')}>
              {t('phaseMap.phaseSelection')} ({selectedPhaseKeys.size}/{cifPhases.length})
            </summary>
            <div style={{ display: 'flex', gap: 6, margin: '4px 0' }}>
              <button
                type="button"
                onClick={() => setSelectedPhaseKeys(new Set(cifPhases.map(p => p.key)))}
                style={{ fontSize: '8pt', cursor: 'pointer', background: 'transparent',
                         border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
                         color: C.textSecondary, padding: '1px 8px' }}
              >{t('phaseMap.selectAll')}</button>
              <button
                type="button"
                onClick={() => setSelectedPhaseKeys(new Set())}
                style={{ fontSize: '8pt', cursor: 'pointer', background: 'transparent',
                         border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
                         color: C.textSecondary, padding: '1px 8px' }}
              >{t('phaseMap.selectNone')}</button>
            </div>
            <div style={{ maxHeight: 160, overflowY: 'auto', paddingRight: 4 }}>
              {cifPhases.map((p) => (
                <label key={p.key} style={{
                  display: 'flex', alignItems: 'center', gap: 6, fontSize: '8.5pt',
                  padding: '1px 0', cursor: 'pointer', color: C.text,
                }} title={p.formula || p.cif_filename}>
                  <input
                    type="checkbox"
                    checked={selectedPhaseKeys.has(p.key)}
                    onChange={() => togglePhase(p.key)}
                    style={{ accentColor: C.purple }}
                  />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.cif_filename}
                  </span>
                </label>
              ))}
            </div>
            {selectedPhaseKeys.size === 0 && (
              <div style={{ fontSize: '8pt', color: C.orange, marginTop: 2 }}>
                {t('phaseMap.noneSelected')}
              </div>
            )}
          </details>
        )}

        {/* --- Action buttons --- */}
        <div style={{ display: 'flex', gap: 6 }}>
          <Button
            variant="primary"
            onClick={handleAutoClassify}
            disabled={loading}
            style={{ flex: 1 }}
            title={t('phaseMap.autoClassifyTooltip')}
          >
            {loading ? t('phaseMap.classifying') : (hasMap ? t('phaseMap.reclassify') : t('phaseMap.autoClassify'))}
          </Button>
          {hasMap && (
            <Button
              variant="secondary"
              onClick={handleClearMap}
              disabled={loading}
              title={t('phaseMap.clearTooltip')}
            >
              {t('phaseMap.clear')}
            </Button>
          )}
        </div>

        {error && (
          <div role="alert" style={{
            fontSize: '9pt', color: C.red,
            background: alpha(C.red, 8), border: `1px solid ${alpha(C.red, 30)}`,
            borderRadius: 4, padding: '4px 8px',
          }}>
            {error}
          </div>
        )}

        {/* --- Stats --- */}
        {hasMap && (
          <div style={{ fontSize: '9pt', color: C.textSecondary }}>
            {t('phaseMap.stats', { classified: phaseMap.n_classified, total: phaseMap.n_classified + phaseMap.n_unclassified })}
            {realPhases.length === 1
              ? t('phaseMap.statsPhasesOne', { count: realPhases.length })
              : t('phaseMap.statsPhasesOther', { count: realPhases.length })}
            {phaseMap.k_used > 0 && ` · ${t('phaseMap.kUsed', { k: phaseMap.k_used })}`}
          </div>
        )}

        {/* --- Cluster report. The point of clustering is that ambiguity can
                be stated per region, which is impossible per pixel: "this
                group matches A at 0.98 and B at 0.95, chemistry cannot
                separate them". --- */}
        {clusters.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{
              cursor: 'pointer', fontSize: '8.5pt', color: C.textSecondary,
              userSelect: 'none',
            }}>
              {t('phaseMap.clusterReport')} ({clusters.length})
            </summary>
            <div style={{
              maxHeight: 240, overflowY: 'auto', marginTop: 4,
              border: `1px solid ${C.border}`, borderRadius: 4, padding: 4,
            }}>
              {clusters.map((c) => (
                <div key={c.cluster_id} style={{
                  padding: '4px 2px',
                  borderBottom: `1px solid ${alpha(C.border, 40)}`,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                    <span style={{
                      fontSize: '9pt', fontWeight: 600,
                      color: c.cif_filename ? C.green : C.orange,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {c.cif_filename || t('phaseMap.clusterUnmatched')}
                    </span>
                    <span style={{ fontSize: '8.5pt', color: C.textSecondary, flexShrink: 0 }}>
                      {c.percentage}% · {c.score}
                    </span>
                  </div>
                  <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 1 }}>
                    {Object.entries(c.mean_at_pct)
                      .map(([el, v]) => `${el} ${v}`).join(' · ')}
                  </div>
                  {c.ambiguous && (
                    <div style={{ fontSize: '8pt', color: C.orange, marginTop: 1 }}
                         title={t('phaseMap.ambiguousTooltip')}>
                      ⚠ {t('phaseMap.ambiguous')}
                      {c.runners_up?.length > 0 && `: ${c.runners_up
                        .map(r => `${r.cif_filename} (${r.score})`).join(', ')}`}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </details>
        )}

        {/* --- Regions vs phases ----------------------------------
            Regions come first because that is the order of the work: the
            data says which pixels belong together, the user says what they
            are. The phase view is the same map seen through those names. */}
        {hasMap && (phaseMap?.regions || []).length > 0 && (
          <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
            {[
              { id: 'regions', label: t('regions.viewRegions'),
                tip: t('regions.viewRegionsTooltip') },
              { id: 'phases', label: t('regions.viewPhases'),
                tip: t('regions.viewPhasesTooltip') },
            ].map((v) => {
              const active = mapView === v.id;
              return (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => setMapView(v.id)}
                  title={v.tip}
                  style={{
                    flex: 1, fontSize: '8.5pt', padding: '3px 6px',
                    borderRadius: 3, cursor: 'pointer',
                    background: active ? alpha(C.cyan, 22) : 'transparent',
                    color: active ? C.text : C.textSecondary,
                    border: `1px solid ${active ? alpha(C.cyan, 50) : C.border}`,
                  }}
                >
                  {v.label}
                </button>
              );
            })}
          </div>
        )}

        {hasMap && showRegions && <RegionPanel handle={handle} />}

        {/* --- Legend --- */}
        {hasMap && !showRegions && realPhases.length > 0 && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 2,
            maxHeight: 220, overflowY: 'auto',
            border: `1px solid ${C.border}`, borderRadius: 4, padding: 4,
          }}>
            {/* Without this the swatch reads as decoration and nobody
                discovers the picker. */}
            <div style={{ fontSize: '7.5pt', color: C.textSecondary, padding: '0 2px 2px' }}>
              {t('phaseMap.colorHint')}
            </div>
            {realPhases.map((s) => {
              const active = selectedPhaseIndex === s.phase_index;
              return (
                <div
                  key={s.phase_index}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedPhaseIndex(active ? null : s.phase_index)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelectedPhaseIndex(active ? null : s.phase_index);
                    }
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '3px 6px', borderRadius: 3,
                    background: active ? alpha(C.cyan, 18) : 'transparent',
                    border: active ? `1px solid ${alpha(C.cyan, 50)}` : '1px solid transparent',
                    cursor: 'pointer',
                  }}
                  title={t('phaseMap.legendTooltip', { cif: s.cif_filename, formula: s.formula, pct: s.percentage })}
                >
                  {/* Swatch doubles as the colour picker, exactly as on the
                      EBSD phase map: click to pick, right-click to reset.
                      Keyed on the phase name WITHOUT the .cif, which is what
                      the EBSD page uses, so one choice covers both pages. */}
                  <span
                    style={{
                      width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                      display: 'inline-block', position: 'relative',
                      background: s.color || '#3c3c3c',
                      border: phaseNameKey(s.cif_filename) in colorOverrides
                        ? `1px solid ${C.text}` : `1px solid ${C.border}`,
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      resetPhaseColor(phaseNameKey(s.cif_filename));
                    }}
                    title={t('phaseMap.colorTooltip', { name: s.cif_filename })}
                  >
                    <input
                      type="color"
                      value={s.color || '#3c3c3c'}
                      onChange={(e) => setPhaseColor(phaseNameKey(s.cif_filename), e.target.value)}
                      style={{
                        position: 'absolute', inset: 0, width: '100%', height: '100%',
                        opacity: 0, cursor: 'pointer', border: 'none', padding: 0,
                      }}
                      aria-label={t('phaseMap.colorAria', { name: s.cif_filename })}
                    />
                  </span>
                  <span style={{
                    flex: 1, minWidth: 0, fontSize: '9pt',
                    color: active ? C.text : C.textSecondary,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {s.cif_filename}
                  </span>
                  <span style={{ fontSize: '8pt', color: C.textSecondary, flexShrink: 0 }}>
                    {s.percentage}%
                  </span>
                </div>
              );
            })}
            {unclassifiedRow && unclassifiedRow.n_pixels > 0 && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 6px', borderRadius: 3, opacity: 0.8,
              }}>
                <span style={{
                  width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                  background: '#3c3c3c', border: `1px solid ${C.border}`,
                }} />
                <span style={{ flex: 1, fontSize: '9pt', color: C.textSecondary, fontStyle: 'italic' }}>
                  {t('phaseMap.unclassified')}
                </span>
                <span style={{ fontSize: '8pt', color: C.textSecondary }}>
                  {unclassifiedRow.percentage}%
                </span>
              </div>
            )}
          </div>
        )}

        {/* --- Hand tools. Collapsed while grouping, because they belong to
            the phase view and stacking both toolsets is what made this rail
            unreadable. --- */}
        {hasMap && showRegions && (
          <Label secondary small style={{ display: 'block', marginTop: 6 }}>
            {t('regions.handToolsHint')}
          </Label>
        )}
        <details open={!showRegions} style={{ marginTop: 2 }}>
          <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                            color: C.textSecondary, userSelect: 'none' }}
                   title={t('regions.handToolsTooltip')}>
            {t('regions.handTools')}
          </summary>
        {/* --- Region painting (M4) --- */}
        {hasMap && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 4,
            border: `1px solid ${C.border}`, borderRadius: 4, padding: 6,
            background: alpha(C.bgSecondary, 50),
          }}>
            {/* Mode toggle: rectangle (drag) vs polygon (click vertices) */}
        {/* Undo — one level, and it redoes. Every mutation snapshots first, so
            this covers a re-classify as well as a paint. */}
        {hasMap && phaseMap.undo_label && (
          <Button
            variant="secondary"
            onClick={handleUndo}
            style={{ width: '100%', marginTop: 2 }}
            title={t('phaseMap.undoTooltip', { what: phaseMap.undo_label })}
          >
            {t('phaseMap.undo', { what: phaseMap.undo_label })}
          </Button>
        )}
        {/* Replace one phase with another, map-wide. The wand is a feature-scale
            instrument; this is the map-scale one. */}
        {hasMap && realPhases.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                              color: C.textSecondary, userSelect: 'none' }}
                     title={t('phaseMap.replaceTooltip')}>
              {t('phaseMap.replaceTitle')}
            </summary>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4,
                          flexWrap: 'wrap' }}>
              <select
                value={replaceFrom ?? ''}
                onChange={(e) => setReplaceFrom(e.target.value === '' ? null : Number(e.target.value))}
                aria-label={t('phaseMap.replaceFrom')}
                style={{ flex: 1, minWidth: 0, fontSize: '8.5pt', padding: '2px 4px',
                         background: 'transparent', color: C.text,
                         border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3 }}
              >
                <option value="">{t('phaseMap.replaceFrom')}</option>
                {realPhases.map((s) => (
                  <option key={s.phase_index} value={s.phase_index}>
                    {s.cif_filename} ({s.percentage}%)
                  </option>
                ))}
                <option value={-1}>{t('phaseMap.unclassified')}</option>
              </select>
              <span style={{ fontSize: '9pt', color: C.textSecondary }}>{'→'}</span>
              <Button
                variant="warning"
                disabled={replaceFrom == null || selectedPhaseIndex == null
                          || replaceFrom === selectedPhaseIndex || assignBusy}
                onClick={() => handleReplacePhase(replaceFrom, selectedPhaseIndex)}
                title={selectedPhaseIndex == null
                  ? t('phaseMap.replaceNeedTarget') : t('phaseMap.replaceGo')}
              >
                {t('phaseMap.replaceButton')}
              </Button>
            </div>
            <Label secondary small style={{ display: 'block', marginTop: 3 }}>
              {t('phaseMap.replaceHint')}
            </Label>
          </details>
        )}
        {/* --- Seeded selection (wand) --- */}
        {isWandMode && wand.seed && wand.growth.length > 0 && (
          <div style={{
            marginTop: 4, padding: '7px 8px', borderRadius: 4,
            background: alpha(C.cyan, 8),
            border: `1px solid ${alpha(C.cyan, 25)}`,
          }}>
            {/* A phase is rarely one blob. "Connected" grows from the seed;
                "everywhere" takes every pixel like it across the whole map — a
                chemistry-space selection rather than a spatial one. */}
            <div style={{ display: 'flex', marginBottom: 3 }} role="group" aria-label={t('wand.scope')}>
              {[['connected', t('wand.scopeConnected')], ['all', t('wand.scopeAll')]].map(([id, label], i) => (
                <button
                  key={id} type="button"
                  onClick={() => wand.setScope(id)}
                  aria-pressed={wand.scope === id}
                  title={id === 'all' ? t('wand.scopeAllTooltip') : t('wand.scopeConnectedTooltip')}
                  style={{
                    padding: '1px 8px', fontSize: '8pt', cursor: 'pointer',
                    border: `1px solid ${alpha(C.cyan, wand.scope === id ? 60 : 22)}`,
                    background: wand.scope === id ? alpha(C.cyan, 25) : 'transparent',
                    color: wand.scope === id ? C.text : C.textSecondary,
                    borderRadius: i === 0 ? '3px 0 0 3px' : '0 3px 3px 0',
                  }}
                >{label}</button>
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
              <Label secondary small>{t('wand.selection')}</Label>
              <span style={{ fontSize: '9pt', fontWeight: 700, color: C.cyan,
                             fontVariantNumeric: 'tabular-nums' }}>
                {t('wand.pixels', { count: wand.nSelected })}
              </span>
            </div>
            {/* The slider walks a growth curve sampled where pixels actually are.
                Thresholding the chemistry directly has dead bands — 2 % through
                10 % of the range returned an identical selection on real data —
                so the axis is the pixel count instead. The count is also the leak
                detector: a jump from 300 to 48 000 announces itself here. */}
            <input
              type="range" min={0} max={wand.growth.length - 1} step={1}
              value={wand.step}
              onChange={(e) => wand.setStep(Number(e.target.value))}
              onMouseUp={wand.refreshStats}
              onKeyUp={wand.refreshStats}
              aria-label={t('wand.sliderAria')}
              title={t('wand.sliderTooltip')}
              style={{ width: '100%', accentColor: C.cyan, marginTop: 2 }}
            />
            {wand.stats && wand.stats.n_pixels > 0 && (
              <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 2 }}>
                {Object.entries(wand.stats.mean_at_pct)
                  .sort((a, b) => b[1] - a[1]).slice(0, 5)
                  .map(([el, v]) => el + ' ' + v).join(' · ')}
              </div>
            )}
            {wand.stats && Object.keys(wand.stats.enrichment || {}).length > 0 && (
              <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 1 }}
                   title={t('wand.enrichmentTooltip')}>
                {t('wand.enrichment')}
                {': '}
                {Object.entries(wand.stats.enrichment)
                  .filter(([, v]) => v >= 1.3)
                  .sort((a, b) => b[1] - a[1]).slice(0, 4)
                  .map(([el, v]) => el + ' ' + v + 'x').join(' · ') || t('wand.enrichmentNone')}
              </div>
            )}
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <Button
                variant="primary"
                disabled={selectedPhaseIndex == null || wand.loading}
                onClick={async () => {
                  const res = await wand.commit(selectedPhaseIndex);
                  if (res) setPhaseMap(res);
                }}
                style={{ flex: 1 }}
                title={selectedPhaseIndex == null
                  ? t('wand.assignTooltipNoPhase') : t('wand.assignTooltip')}
              >
                {t('wand.assign')}
              </Button>
              <Button variant="secondary" onClick={wand.clear} title={t('wand.cancelTooltip')}>
                {t('wand.cancel')}
              </Button>
            </div>
            {wand.error && (
              <div role="alert" style={{ fontSize: '8pt', color: C.red, marginTop: 3 }}>
                {wand.error}
              </div>
            )}
          </div>
        )}
        {isWandMode && !wand.seed && (
          <Label secondary small style={{ marginTop: 4 }}>{t('wand.hint')}</Label>
        )}
        {/* Phases the classifier placed nowhere. The legend cannot show
            them (zero pixels) but they are exactly what a manual
            correction usually needs to assign. */}
        {hasMap && unplacedPhases.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                              color: C.textSecondary, userSelect: 'none' }}>
              {t('phaseMap.unplaced', { count: unplacedPhases.length })}
            </summary>
            <div style={{ maxHeight: 150, overflowY: 'auto', marginTop: 4 }}>
              {unplacedPhases.map((s) => {
                const active = selectedPhaseIndex === s.phase_index;
                return (
                  <div
                    key={s.phase_index}
                    onClick={() => setSelectedPhaseIndex(active ? null : s.phase_index)}
                    title={s.formula || s.cif_filename}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      padding: '3px 5px', borderRadius: 3, cursor: 'pointer',
                      fontSize: '8.5pt',
                      background: active ? alpha(C.purple, 25) : 'transparent',
                      border: `1px solid ${active ? alpha(C.purple, 60) : 'transparent'}`,
                    }}
                  >
                    <span style={{ width: 10, height: 10, borderRadius: 2,
                                   background: s.color, flexShrink: 0 }} />
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis',
                                   whiteSpace: 'nowrap' }}>{s.cif_filename}</span>
                  </div>
                );
              })}
            </div>
          </details>
        )}
            {/* Only the paint-mode label and its toggle belong on one
                row; everything above stacks. */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Label secondary small>{t('phaseMap.paintMode')}</Label>
              <div style={{
                display: 'flex', border: `1px solid ${C.border}`, borderRadius: 4,
                overflow: 'hidden', flex: 1,
              }}>
                {[
                  { id: 'rectangle', label: t('phaseMap.rectangle'), tip: t('hoverTips.paintModeRectangle') },
                  { id: 'polygon', label: t('phaseMap.polygon'), tip: t('hoverTips.paintModePolygon') },
                  { id: 'wand', label: t('wand.mode'), tip: t('wand.modeTooltip') },
                ].map((m) => {
                  const active = paintMode === m.id;
                  return (
                    <button
                      key={m.id}
                      onClick={() => setPaintMode(m.id)}
                      aria-pressed={active}
                      title={m.tip}
                      style={{
                        flex: 1, padding: '4px 0',
                        background: active ? C.purple : 'transparent',
                        color: active ? C.bg : C.textSecondary,
                        fontWeight: active ? 700 : 400,
                        fontSize: '9pt', border: 'none',
                        cursor: 'pointer', transition: 'all 0.15s',
                      }}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {paintMode === 'rectangle' ? (
              <>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  <NumberInput
                    value={region.rowStart}
                    onChange={(e) => setRegion(r => ({ ...r, rowStart: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.rowStartTooltip')}
                  />
                  <span style={{ alignSelf: 'center', fontSize: '8pt', color: C.textSecondary }}>–</span>
                  <NumberInput
                    value={region.rowEnd}
                    onChange={(e) => setRegion(r => ({ ...r, rowEnd: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.rowEndTooltip')}
                  />
                  <NumberInput
                    value={region.colStart}
                    onChange={(e) => setRegion(r => ({ ...r, colStart: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.colStartTooltip')}
                  />
                  <span style={{ alignSelf: 'center', fontSize: '8pt', color: C.textSecondary }}>–</span>
                  <NumberInput
                    value={region.colEnd}
                    onChange={(e) => setRegion(r => ({ ...r, colEnd: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.colEndTooltip')}
                  />
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <Button
                    variant="warning"
                    onClick={() => handleAssignRegion(selectedPhaseIndex ?? -1)}
                    disabled={assignBusy || selectedPhaseIndex == null}
                    style={{ flex: 1 }}
                    title={selectedPhaseIndex == null
                      ? t('phaseMap.assignRegionTooltipNoPhase')
                      : t('phaseMap.assignRegionTooltip', { rowStart: region.rowStart, rowEnd: region.rowEnd, colStart: region.colStart, colEnd: region.colEnd, phase: selectedPhaseIndex })}
                  >
                    {t('phaseMap.assignRegion')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => handleAssignRegion(-1)}
                    disabled={assignBusy}
                    title={t('phaseMap.clearRegionTooltip')}
                  >
                    {t('phaseMap.clearRegion')}
                  </Button>
                </div>
              </>
            ) : (
              <>
                <Label secondary small>
                  {t('phaseMap.polygonHint', {
                    count: polygonVertices.length,
                    suffix: polygonVertices.length >= 3
                      ? t('phaseMap.polygonHintReady')
                      : t('phaseMap.polygonHintNeedMore'),
                  })}
                </Label>
                <div style={{ display: 'flex', gap: 4 }}>
                  <Button
                    variant="warning"
                    onClick={() => handleClosePolygon(selectedPhaseIndex ?? -1)}
                    disabled={assignBusy || selectedPhaseIndex == null || polygonVertices.length < 3}
                    style={{ flex: 1 }}
                    title={selectedPhaseIndex == null
                      ? t('phaseMap.closeAssignTooltipNoPhase')
                      : polygonVertices.length < 3
                        ? t('phaseMap.closeAssignTooltipNeedMore')
                        : t('phaseMap.closeAssignTooltip', { count: polygonVertices.length, phase: selectedPhaseIndex })}
                  >
                    {t('phaseMap.closeAssign')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => handleClosePolygon(-1)}
                    disabled={assignBusy || polygonVertices.length < 3}
                    title={t('phaseMap.polygonClearTooltip')}
                  >
                    {t('phaseMap.clearRegion')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={handleCancelPolygon}
                    disabled={polygonVertices.length === 0}
                    title={t('phaseMap.polygonCancelTooltip')}
                  >
                    ×
                  </Button>
                </div>
              </>
            )}
          </div>
        )}

        </details>

        {/* --- Send to indexing (M5 entry point) --- */}
        {hasMap && realPhases.length > 0 && (
          <Button
            variant="primary"
            onClick={handleSendToIndexing}
            style={{ width: '100%' }}
            title={t('phaseMap.sendToIndexingTooltip')}
          >
            {t('phaseMap.sendToIndexing')}
          </Button>
        )}
      </div>
    </GroupBox>
  );
}
