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
  const [paintMode, setPaintMode] = useState('rectangle');
  const [polygonVertices, setPolygonVertices] = useState([]);  // [[col, row], ...]

  // Last clicked pixel (for the readout under the canvas)
  const [hoveredPixel, setHoveredPixel] = useState(null);

  // On mount or file change, sync from backend. The backend clears its
  // store on close_file(), so a 200-with-loaded:false response is the
  // expected "nothing to show" state, not an error.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    edsApi.getPhaseMap(true)
      .then((res) => { if (!cancelled) setPhaseMap(res.data?.loaded ? res.data : null); })
      .catch(() => { /* 4xx is fine — just means nothing classified yet */ });
    return () => { cancelled = true; };
  }, [filePath]);

  const handleAutoClassify = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await edsApi.autoClassify(tolerance, minScore);
      setPhaseMap(res.data);
      setSelectedPhaseIndex(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorClassify'));
    } finally {
      setLoading(false);
    }
  }, [tolerance, minScore, t]);

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
    selectedPhaseIndex, setSelectedPhaseIndex,
    region, setRegion,
    assignBusy,
    hoveredPixel, setHoveredPixel,
    paintMode, setPaintMode,
    polygonVertices, setPolygonVertices,
    handleAutoClassify, handleClearMap,
    handleAssignRegion, handleClosePolygon, handleCancelPolygon,
    handleSendToIndexing,
  };
}


/** Convert a click event on the phase-map img element to (row, col). */
function pixelFromClick(e) {
  const img = e.currentTarget;
  if (!img || !img.naturalWidth || !img.naturalHeight) return null;
  const rect = img.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  const containerAspect = rect.width / rect.height;
  const imageAspect = natW / natH;

  let dispW, dispH, offX, offY;
  if (containerAspect > imageAspect) {
    dispH = rect.height;
    dispW = rect.height * imageAspect;
    offX = (rect.width - dispW) / 2;
    offY = 0;
  } else {
    dispW = rect.width;
    dispH = rect.width / imageAspect;
    offX = 0;
    offY = (rect.height - dispH) / 2;
  }
  const px = (x - offX) / dispW * natW;
  const py = (y - offY) / dispH * natH;
  if (px < 0 || py < 0 || px >= natW || py >= natH) return null;
  return { row: Math.floor(py), col: Math.floor(px) };
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
export function PhaseMapCanvas({ handle }) {
  const { t } = useTranslation('eds');
  const {
    phaseMap, hoveredPixel, setHoveredPixel,
    region, setRegion,
    paintMode, polygonVertices, setPolygonVertices,
  } = handle;
  const [drag, setDrag] = useState(null);  // { startRow, startCol, endRow, endCol } | null

  const isPolygon = paintMode === 'polygon';

  const handleMouseDown = useCallback((e) => {
    if (isPolygon) return;  // polygon mode uses click-to-add-vertex, not drag
    const px = pixelFromClick(e);
    if (!px || !phaseMap) return;
    e.preventDefault();
    setDrag({ startRow: px.row, startCol: px.col, endRow: px.row, endCol: px.col });
  }, [phaseMap, isPolygon]);

  const handleMouseMove = useCallback((e) => {
    if (!drag) return;
    const px = pixelFromClick(e);
    if (!px) return;
    setDrag(d => d && { ...d, endRow: px.row, endCol: px.col });
  }, [drag]);

  const handleMouseUp = useCallback((e) => {
    if (!drag) return;
    const px = pixelFromClick(e);
    const endRow = px ? px.row : drag.endRow;
    const endCol = px ? px.col : drag.endCol;
    const sameSpot = endRow === drag.startRow && endCol === drag.startCol;
    if (sameSpot) {
      // Treat as a click — don't create a 1-pixel "region", just inspect.
      setHoveredPixel({ row: endRow, col: endCol });
    } else {
      const r0 = Math.min(drag.startRow, endRow);
      const r1 = Math.max(drag.startRow, endRow);
      const c0 = Math.min(drag.startCol, endCol);
      const c1 = Math.max(drag.startCol, endCol);
      setRegion({ rowStart: r0, rowEnd: r1, colStart: c0, colEnd: c1 });
      setHoveredPixel(null);
    }
    setDrag(null);
  }, [drag, setHoveredPixel, setRegion]);

  const handleMouseLeave = useCallback(() => {
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
      <div style={{
        position: 'relative', width: '100%', height: '100%',
      }}>
        <img
          src={`data:image/png;base64,${phaseMap.image}`}
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
            // Phase maps are categorical — interpolation across class
            // boundaries would render garbage colours.
            imageRendering: 'pixelated',
            display: 'block',
            userSelect: 'none',
          }}
        />
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
              <rect
                x={liveRect.x} y={liveRect.y}
                width={liveRect.w} height={liveRect.h}
                fill="rgba(139, 233, 253, 0.18)"
                stroke="rgba(139, 233, 253, 0.95)"
                strokeWidth={Math.max(1, Math.min(nCols, nRows) / 200)}
                vectorEffect="non-scaling-stroke"
              />
            )}
            {isPolygon && polygonVertices.length > 0 && (
              <>
                {/* Filled polygon (only when closed) gives a preview of
                    what the assign will paint. While the polygon is
                    still open we show only the polyline so the user
                    can tell open vs. closed at a glance. */}
                {polygonVertices.length >= 3 && (
                  <polygon
                    points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                    fill="rgba(189, 147, 249, 0.18)"
                    stroke="rgba(189, 147, 249, 0.95)"
                    strokeWidth={Math.max(1, Math.min(nCols, nRows) / 200)}
                    strokeDasharray={polygonVertices.length < 3 ? "2 2" : undefined}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {polygonVertices.length < 3 && (
                  <polyline
                    points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                    fill="none"
                    stroke="rgba(189, 147, 249, 0.95)"
                    strokeWidth={Math.max(1, Math.min(nCols, nRows) / 200)}
                    strokeDasharray="2 2"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {polygonVertices.map(([x, y], i) => (
                  <circle
                    key={`v${i}`}
                    cx={x + 0.5} cy={y + 0.5}
                    r={Math.max(1.0, Math.min(nCols, nRows) / 120)}
                    fill="rgba(189, 147, 249, 0.95)"
                    stroke="white"
                    strokeWidth={Math.max(0.5, Math.min(nCols, nRows) / 400)}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
              </>
            )}
          </svg>
        )}
      </div>
      {readout}
    </div>
  );
}


/** PhaseMapControls — controls + legend + region painting in the right rail. */
export function PhaseMapControls({ handle }) {
  const { t } = useTranslation('eds');
  const {
    phaseMap, loading, error,
    tolerance, setTolerance, minScore, setMinScore,
    selectedPhaseIndex, setSelectedPhaseIndex,
    region, setRegion,
    assignBusy,
    paintMode, setPaintMode,
    polygonVertices,
    handleAutoClassify, handleClearMap,
    handleAssignRegion, handleClosePolygon, handleCancelPolygon,
    handleSendToIndexing,
  } = handle;

  const hasMap = !!phaseMap?.loaded;
  const summary = phaseMap?.summary || [];
  const realPhases = summary.filter(s => !s.is_unclassified);
  const unclassifiedRow = summary.find(s => s.is_unclassified);

  return (
    <GroupBox title={t('phaseMap.title')}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {/* --- Tolerance + min-score --- */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
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
          </div>
        )}

        {/* --- Legend --- */}
        {hasMap && realPhases.length > 0 && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 2,
            maxHeight: 220, overflowY: 'auto',
            border: `1px solid ${C.border}`, borderRadius: 4, padding: 4,
          }}>
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
                  <span style={{
                    width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                    background: s.color || '#3c3c3c',
                    border: `1px solid ${C.border}`,
                  }} />
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

        {/* --- Region painting (M4) --- */}
        {hasMap && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 4,
            border: `1px solid ${C.border}`, borderRadius: 4, padding: 6,
            background: alpha(C.bgSecondary, 50),
          }}>
            {/* Mode toggle: rectangle (drag) vs polygon (click vertices) */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Label secondary small>{t('phaseMap.paintMode')}</Label>
              <div style={{
                display: 'flex', border: `1px solid ${C.border}`, borderRadius: 4,
                overflow: 'hidden', flex: 1,
              }}>
                {[
                  { id: 'rectangle', label: t('phaseMap.rectangle'), tip: t('hoverTips.paintModeRectangle') },
                  { id: 'polygon', label: t('phaseMap.polygon'), tip: t('hoverTips.paintModePolygon') },
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
