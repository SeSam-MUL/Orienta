import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import {
  defaultModel, addPanel, addElement, removeElement, updateElement,
  moveElement, resizeElement, alignElements, distributeElements, sameSize,
  niceScaleLength, niceCount, reseedIdsFrom, SOURCE_META,
} from './figureModel';
import { paintFigure } from './paintFigure';
import {
  loadSources, composePatternFigure, downloadFigure, copyFigureToClipboard,
} from './composePatternFigure';
import { savePreset, listPresets, loadPreset, deletePreset } from './layoutPresets';

const C = colors;
// Preview pane budget (CSS px). The canvas is fit (contain) into this box using
// the figure's TRUE aspect (model.canvas.hPx/wPx) instead of a fixed width, so a
// near-square multi-panel figure no longer renders as a short, wide strip.
const PREVIEW_MAX_W = 640;   // max preview width  (fits the left column)
const PREVIEW_MAX_H = 560;   // max preview height (keeps the modal within 94vh)
const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
const ALL_SOURCES = ['experimental', 'simulated', 'ncc', 'heatmap'];

// Topmost element under a normalized point (last drawn = on top).
function elementAt(model, nx, ny) {
  for (let i = model.elements.length - 1; i >= 0; i--) {
    const e = model.elements[i];
    if (nx >= e.x && nx <= e.x + e.w && ny >= e.y && ny <= e.y + e.h) return e;
  }
  return null;
}

export default function PatternExportDialog({ open, onClose, sources, rNcc, stepUm, mapCols = null, markers = [] }) {
  const { t } = useTranslation(['patternmatch', 'common']);
  // Localized panel-source label, falling back to the model's English label/key.
  const srcLabel = (s) => (SOURCE_META[s] ? t(`patternmatch:source.${s}`) : s);
  const [model, setModel] = useState(null);
  const [imgs, setImgs] = useState({ markers, stepUm, mapCols });
  const [selectedIds, setSelectedIds] = useState([]);
  const [resolution, setResolution] = useState({ mode: 'scale', scale: 2, widthPx: 2400 });
  const [format, setFormat] = useState('png');
  const [jpegQuality, setJpegQuality] = useState(0.92);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [copyOk, setCopyOk] = useState(false);
  const [presets, setPresets] = useState(() => listPresets());
  const [presetName, setPresetName] = useState('');

  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const drag = useRef(null);

  // Build a fresh model + decode images when opened. Keyed on a STABLE presence
  // string, NOT the `sources` object — the parent passes a fresh object literal
  // every render, so depending on it would reset the user's composition on any
  // parent re-render. The user can't change the underlying pixel while this
  // modal is up, so presence is the only thing that can meaningfully change.
  const sourcesKey = ALL_SOURCES.map((s) => (sources?.[s] ? '1' : '0')).join('');
  useEffect(() => {
    if (!open) return;
    const present = ALL_SOURCES.filter((s) => sources?.[s]);
    setModel(defaultModel(present));
    setSelectedIds([]);
    setError(null);
    let cancelled = false;
    loadSources({
      experimental: sources?.experimental, simulated: sources?.simulated,
      ncc: sources?.ncc, heatmap: sources?.heatmap,
    }).then((decoded) => {
      if (!cancelled) setImgs({ ...decoded, markers, stepUm, mapCols });
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, sourcesKey]);

  // Keep markers and the map scale fresh without re-decoding images. The scale
  // rides WITH the sources because that is where the heatmap image lives, and
  // the painter must never have to guess it from somewhere else.
  useEffect(() => {
    setImgs((p) => ({ ...p, markers, stepUm, mapCols }));
  }, [markers, stepUm, mapCols]);

  // Fit the figure's true aspect into the preview budget (contain). Whichever of
  // width/height hits its cap first sets the scale; the other stays under.
  const aspect = model ? (model.canvas.hPx / model.canvas.wPx) : 0.5; // h/w
  let previewW = PREVIEW_MAX_W;
  let previewH = Math.round(previewW * aspect);
  if (previewH > PREVIEW_MAX_H) {
    previewH = PREVIEW_MAX_H;
    previewW = Math.round(previewH / aspect);
  }

  // Repaint the live preview from the SAME painter used for export.
  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs || !model) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return; // jsdom guard
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    paintFigure(ctx, model, imgs, { W: cvs.width, H: cvs.height });
  }, [model, imgs, previewW, previewH]);

  const patch = useCallback((id, p) => setModel((m) => updateElement(m, id, p)), []);
  const patchMany = useCallback((patches) => setModel((m) => {
    let next = m;
    for (const { id, patch: p } of patches) next = updateElement(next, id, p);
    return next;
  }), []);

  // ---- Overlay drag / resize -------------------------------------------------
  const toNorm = (e) => {
    const r = overlayRef.current.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
  };
  const onOverlayDown = (e) => {
    if (!model) return;
    const norm = toNorm(e);
    const hit = elementAt(model, norm.x, norm.y);
    if (hit) {
      setSelectedIds(e.shiftKey ? (ids) => Array.from(new Set([...ids, hit.id])) : [hit.id]);
      drag.current = { mode: 'move', id: hit.id, start: norm, startEl: hit };
      overlayRef.current.setPointerCapture?.(e.pointerId);
    } else {
      setSelectedIds([]);
    }
  };
  const onHandleDown = (e, handle, el) => {
    e.stopPropagation();
    drag.current = { mode: 'resize', id: el.id, handle, start: toNorm(e), startEl: el };
    overlayRef.current.setPointerCapture?.(e.pointerId);
  };
  const onOverlayMove = (e) => {
    if (!drag.current) return;
    const norm = toNorm(e);
    const dx = norm.x - drag.current.start.x;
    const dy = norm.y - drag.current.start.y;
    const { mode, id, handle, startEl } = drag.current;
    if (mode === 'move') {
      const moved = moveElement(startEl, dx, dy);
      patch(id, { x: moved.x, y: moved.y });
    } else {
      const r = resizeElement(startEl, handle, dx, dy, { aspectLock: startEl.aspectLock });
      patch(id, { x: r.x, y: r.y, w: r.w, h: r.h });
    }
  };
  const onOverlayUp = () => { drag.current = null; };

  // ---- Toolbar actions -------------------------------------------------------
  const placedSources = (model?.elements || []).filter((e) => e.type === 'panel').map((e) => e.source);
  const addableSources = ALL_SOURCES.filter((s) => sources?.[s] && !placedSources.includes(s));

  const onAddPanel = (source) => { if (source) setModel((m) => addPanel(m, source)); };
  const onAddText = () => setModel((m) => addElement(m, {
    type: 'text', text: t('patternmatch:element.defaultLabel'), x: 0.4, y: 0.05, w: 0.2, h: 0.06,
    fontSize: 16, color: '#000000', weight: 'normal',
  }));
  const onAddRNcc = () => setModel((m) => addElement(m, {
    type: 'text', x: 0.35, y: 0.9, w: 0.3, h: 0.06, fontSize: 16, color: '#000000', weight: 'bold',
    text: rNccCaption(rNcc),
  }));
  const onAddColorbar = () => setModel((m) => addElement(m, {
    type: 'colorbar', kind: 'ncc', x: 0.92, y: 0.3, w: 0.03, h: 0.4,
    vmin: -1, vmax: 1, color: '#000000', fontSize: 11, visible: true,
  }));
  /**
   * Can this figure state a length at all?
   *
   * Only the heatmap carries a scale — a detector pattern's pixels are not a
   * length on the specimen — so a bar needs that panel plus, at minimum, how
   * many scan columns it spans. With a step size on top, the bar can be in
   * micrometres; without one it can still be in honest map pixels.
   *
   * Until 2026-08-27 the button was always enabled and always produced a
   * "100 px" label over a bar 30 % of its own box wide. Refusing is the honest
   * answer, and the tooltip says which half is missing.
   */
  const scalebarAvailability = useMemo(() => {
    const hasHeatmap = !!model?.elements?.some(
      (e) => e.type === 'panel' && e.source === 'heatmap',
    );
    if (!hasHeatmap || !(mapCols > 0)) return { can: false, um: false };
    return { can: true, um: stepUm > 0 };
  }, [model, mapCols, stepUm]);

  const onAddScalebar = () => setModel((m) => {
    const heatmapPanel = m.elements.find(
      (e) => e.type === 'panel' && e.source === 'heatmap');
    if (!heatmapPanel || !(mapCols > 0)) return m;
    // Sits under the heatmap by default and spans the panel's box, so there is
    // room for the bar the painter will size from the physics. Its own width is
    // a frame only — dragging the handles moves and pads it, never rescales it.
    const base = {
      type: 'scalebar', x: heatmapPanel.x,
      y: heatmapPanel.y + heatmapPanel.h + 0.02,
      w: heatmapPanel.w, h: 0.06, color: '#000000', fontSize: 12,
    };
    if (stepUm > 0) {
      const { valueUm } = niceScaleLength(stepUm, 1.0, mapCols);
      if (valueUm > 0) {
        return addElement(m, { ...base, mode: 'um', unit: 'µm', lengthValue: valueUm });
      }
    }
    // No step size: measure in scan pixels, which is still true about the map.
    const nice = niceCount(mapCols / 3);
    return addElement(m, { ...base, mode: 'px', unit: 'px', lengthValue: nice });
  });

  const setBg = (bg) => setModel((m) => ({ ...m, canvas: { ...m.canvas, bg } }));

  const onAlign = (mode) => {
    const els = model.elements.filter((e) => selectedIds.includes(e.id));
    if (els.length >= 2) patchMany(alignElements(els, mode));
  };
  const onDistribute = (axis) => {
    const els = model.elements.filter((e) => selectedIds.includes(e.id));
    if (els.length >= 3) patchMany(distributeElements(els, axis));
  };
  const onSameSize = (dim) => {
    const els = model.elements.filter((e) => selectedIds.includes(e.id));
    if (els.length >= 2) patchMany(sameSize(els, dim));
  };
  const onLetterAll = () => setModel((m) => ({
    ...m,
    elements: m.elements.map((e) => e.type === 'panel' ? { ...e, letter: { ...e.letter, on: true } } : e),
  }));

  // ---- Layout presets --------------------------------------------------------
  const onSavePreset = () => {
    const name = presetName.trim();
    if (!name) return;
    savePreset(name, model);
    setPresets(listPresets());
  };
  const onLoadPreset = (name) => {
    if (!name) return;
    const m = loadPreset(name);
    if (!m) return;
    reseedIdsFrom(m); // avoid id collisions with subsequently-added elements
    // Drop panels referencing sources not present in THIS dialog (e.g. a preset
    // saved from a Dictionary result loaded into a Spherical result).
    m.elements = (m.elements || []).filter((e) => e.type !== 'panel' || sources?.[e.source]);
    setModel(m);
    setSelectedIds([]);
  };
  const onDeletePreset = (name) => {
    if (!name) return;
    deletePreset(name);
    setPresets(listPresets());
  };

  // ---- Export ----------------------------------------------------------------
  const exportOpts = () => (resolution.mode === 'width'
    ? { targetWidthPx: resolution.widthPx, format, jpegQuality }
    : { scale: resolution.scale, format, jpegQuality });

  const onExport = async () => {
    setBusy(true); setError(null);
    try {
      const blob = await composePatternFigure(model, imgs, exportOpts());
      downloadFigure(blob, `pattern-figure.${format === 'jpeg' ? 'jpg' : 'png'}`);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const onCopy = async () => {
    setBusy(true); setError(null); setCopyOk(false);
    try {
      const blob = await composePatternFigure(model, imgs, { ...exportOpts(), format: 'png' });
      await copyFigureToClipboard(blob);
      setCopyOk(true); setTimeout(() => setCopyOk(false), 1500);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  if (!open || !model) return null;

  const selected = model.elements.find((e) => e.id === selectedIds[selectedIds.length - 1]) || null;

  return (
    // The backdrop deliberately does not close the dialog — a stray click
    // beside the window would throw away the whole composition.
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: C.bgSecondary, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16,
        width: 1200, maxWidth: '96vw', maxHeight: '94vh', overflow: 'hidden',
        display: 'flex', flexDirection: 'column', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span style={{ color: C.accent, fontWeight: 700, fontSize: 15 }}>{t('patternmatch:export.header')}</span>
          <button onClick={onClose} aria-label={t('patternmatch:export.closeAria')} title={t('patternmatch:export.closeTooltip')} style={{ background: 'transparent', border: 'none', color: C.textSecondary, cursor: 'pointer', fontSize: 20 }}>&times;</button>
        </div>

        {/* Toolbar — labelled clusters that wrap as whole units */}
        <div style={tbBar}>
          <div style={tbGroup}>
            <span style={tbGroupLabel}>{t('patternmatch:toolbar.addGroup')}</span>
            <select value="" onChange={(e) => { onAddPanel(e.target.value); e.target.value = ''; }}
              disabled={addableSources.length === 0} style={tbSelect} title={t('patternmatch:toolbar.addPanelTooltip')}>
              <option value="">{t('patternmatch:toolbar.addPanelOption')}</option>
              {addableSources.map((s) => <option key={s} value={s}>{srcLabel(s)}</option>)}
            </select>
            <button onClick={onAddText} style={tbBtn} title={t('patternmatch:toolbar.addTextTooltip')}>{t('patternmatch:toolbar.addText')}</button>
            <button onClick={onAddRNcc} style={tbBtn} title={t('patternmatch:toolbar.addRNccTooltip')}>{t('patternmatch:toolbar.addRNcc')}</button>
            <button
              onClick={onAddScalebar}
              disabled={!scalebarAvailability.can}
              style={{ ...tbBtn, opacity: scalebarAvailability.can ? 1 : 0.45,
                cursor: scalebarAvailability.can ? 'pointer' : 'not-allowed' }}
              title={scalebarAvailability.can
                ? t('patternmatch:toolbar.addScaleBarTooltip')
                : t('patternmatch:toolbar.addScaleBarUnavailable')}
            >{t('patternmatch:toolbar.addScaleBar')}</button>
            <button onClick={onAddColorbar} style={tbBtn} title={t('patternmatch:toolbar.addColorbarTooltip')}>{t('patternmatch:toolbar.addColorbar')}</button>
            <button onClick={onLetterAll} style={tbBtn} title={t('patternmatch:toolbar.addLettersTooltip')}>{t('patternmatch:toolbar.addLetters')}</button>
          </div>

          <div style={tbGroup}>
            <span style={tbGroupLabel}>{t('patternmatch:toolbar.backgroundGroup')}</span>
            <select value={isPreset(model.canvas.bg) ? model.canvas.bg : 'custom'}
              onChange={(e) => setBg(e.target.value === 'custom' ? '#cccccc' : e.target.value)} style={tbSelect}
              title={t('patternmatch:toolbar.backgroundTooltip')}>
              <option value="#ffffff">{t('patternmatch:toolbar.bgWhite')}</option>
              <option value="#000000">{t('patternmatch:toolbar.bgBlack')}</option>
              <option value="transparent">{t('patternmatch:toolbar.bgTransparent')}</option>
              <option value="custom">{t('patternmatch:toolbar.bgCustom')}</option>
            </select>
            {!isPreset(model.canvas.bg) && (
              <input type="color" value={model.canvas.bg} onChange={(e) => setBg(e.target.value)}
                title={t('patternmatch:toolbar.customColorTooltip')}
                style={{ width: 26, height: 24, padding: 0, border: `1px solid ${C.border}`, borderRadius: 3, background: 'none', cursor: 'pointer' }} />
            )}
          </div>

          <span style={{ flex: 1 }} />

          <button onClick={onCopy} disabled={busy} style={{ ...tbBtn, ...(copyOk ? tbBtnOk : null) }}
            title={t('patternmatch:toolbar.copyTooltip')}>{copyOk ? t('patternmatch:toolbar.copyDone') : t('patternmatch:toolbar.copy')}</button>
        </div>

        <div style={{ display: 'flex', gap: 14, flex: 1, minHeight: 0 }}>
          {/* Preview + overlay — centred on a framed checkerboard backdrop */}
          <div style={{ flex: '0 0 auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: PREVIEW_MAX_W, height: PREVIEW_MAX_H, padding: 8, boxSizing: 'border-box',
              background: checker, borderRadius: 6, border: `1px solid ${C.border}`,
            }}>
              <div style={{ position: 'relative', width: previewW, height: previewH, boxShadow: '0 2px 12px rgba(0,0,0,0.45)' }}>
                <canvas ref={canvasRef} width={previewW} height={previewH}
                  style={{ position: 'absolute', inset: 0, width: previewW, height: previewH }} />
                <div ref={overlayRef} onPointerDown={onOverlayDown} onPointerMove={onOverlayMove} onPointerUp={onOverlayUp}
                  title={t('patternmatch:export.previewOverlayTooltip')}
                  style={{ position: 'absolute', inset: 0, cursor: 'default' }}>
                {/* Selection outline + handles for the active element (DOM-only; never exported). */}
                {selectedIds.map((id) => {
                  const el = model.elements.find((e) => e.id === id);
                  if (!el) return null;
                  const isActive = selected && selected.id === id;
                  return (
                    <div key={id} style={{ position: 'absolute', left: `${el.x * 100}%`, top: `${el.y * 100}%`,
                      width: `${el.w * 100}%`, height: `${el.h * 100}%`,
                      outline: `1.5px solid ${isActive ? C.accent : C.purple}`, outlineOffset: -1, pointerEvents: 'none' }}>
                      {isActive && HANDLES.map((h) => (
                        <div key={h} onPointerDown={(e) => onHandleDown(e, h, el)}
                          title={t('patternmatch:export.resizeHandleTooltip')} aria-label={t('patternmatch:export.resizeHandleTooltip')}
                          style={{ position: 'absolute', width: 9, height: 9, background: C.accent, border: '1px solid #000',
                            pointerEvents: 'auto', cursor: `${h}-resize`, ...handlePos(h) }} />
                      ))}
                    </div>
                  );
                })}
                </div>{/* overlay */}
              </div>{/* sized canvas wrapper */}
            </div>{/* checkerboard backdrop */}
            <div style={{ fontSize: '11px', color: C.textSecondary, lineHeight: 1.4, maxWidth: PREVIEW_MAX_W }}>
              {t('patternmatch:export.previewHelpPre')}<em>{t('patternmatch:export.previewHelpAspectLock')}</em>{t('patternmatch:export.previewHelpPost')}
            </div>
          </div>

          {/* Inspector + export controls — scrolls independently of the preview */}
          <div style={{ flex: 1, minWidth: 300, display: 'flex', flexDirection: 'column', gap: 10, overflow: 'auto', paddingRight: 4 }}>
            <Inspector el={selected} patch={patch} onRemove={(id) => { setModel((m) => removeElement(m, id)); setSelectedIds([]); }} t={t} srcLabel={srcLabel} />

            {selectedIds.length >= 2 && (
              <div style={box}>
                <div style={boxTitle}>{t('patternmatch:align.title', { count: selectedIds.length })}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {['left', 'hcenter', 'right', 'top', 'vcenter', 'bottom'].map((m) => (
                    <button key={m} onClick={() => onAlign(m)} style={miniBtn} title={t('patternmatch:align.alignTooltip')}>{t(`patternmatch:align.${m}`)}</button>
                  ))}
                  <button onClick={() => onDistribute('x')} style={miniBtn} disabled={selectedIds.length < 3} title={t('patternmatch:align.distributeTooltip')}>{t('patternmatch:align.distX')}</button>
                  <button onClick={() => onDistribute('y')} style={miniBtn} disabled={selectedIds.length < 3} title={t('patternmatch:align.distributeTooltip')}>{t('patternmatch:align.distY')}</button>
                  <button onClick={() => onSameSize('w')} style={miniBtn} title={t('patternmatch:align.sameSizeTooltip')}>{t('patternmatch:align.sameW')}</button>
                  <button onClick={() => onSameSize('h')} style={miniBtn} title={t('patternmatch:align.sameSizeTooltip')}>{t('patternmatch:align.sameH')}</button>
                </div>
              </div>
            )}

            <div style={box}>
              <div style={boxTitle}>{t('patternmatch:presets.title')}</div>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                <input value={presetName} onChange={(e) => setPresetName(e.target.value)} placeholder={t('patternmatch:presets.namePlaceholder')} title={t('patternmatch:presets.nameTooltip')} style={{ ...txtInput, flex: 1, minWidth: 90 }} />
                <button onClick={onSavePreset} disabled={!presetName.trim()} style={miniBtn} title={t('patternmatch:presets.saveTooltip')}>{t('common:save')}</button>
                {presets.length > 0 && (
                  <>
                    <select value="" onChange={(e) => { onLoadPreset(e.target.value); e.target.value = ''; }} style={tbSelect} title={t('patternmatch:presets.loadTooltip')}>
                      <option value="">{t('patternmatch:presets.loadOption')}</option>
                      {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
                    </select>
                    <select value="" onChange={(e) => { onDeletePreset(e.target.value); e.target.value = ''; }} style={tbSelect} title={t('patternmatch:presets.deleteTooltip')}>
                      <option value="">{t('patternmatch:presets.deleteOption')}</option>
                      {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
                    </select>
                  </>
                )}
              </div>
            </div>

            <div style={box}>
              <div style={boxTitle}>{t('patternmatch:resolution.title')}</div>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
                <select value={resolution.mode} onChange={(e) => setResolution((r) => ({ ...r, mode: e.target.value }))} style={tbSelect} title={t('patternmatch:resolution.modeTooltip')}>
                  <option value="scale">{t('patternmatch:resolution.modeScale')}</option>
                  <option value="width">{t('patternmatch:resolution.modeWidth')}</option>
                </select>
                {resolution.mode === 'scale'
                  ? [1, 2, 4, 8].map((s) => (
                      <button key={s} onClick={() => setResolution((r) => ({ ...r, scale: s }))} title={t('patternmatch:resolution.scaleTooltip')}
                        style={{ ...miniBtn, background: resolution.scale === s ? C.accent : C.bg, color: resolution.scale === s ? C.bg : C.text }}>{`${s}×`}</button>
                    ))
                  : <input type="number" min={200} max={12000} step={100} value={resolution.widthPx} title={t('patternmatch:resolution.widthTooltip')}
                      onChange={(e) => setResolution((r) => ({ ...r, widthPx: Number(e.target.value) }))} style={numInput} />}
              </div>
              <div style={{ fontSize: '10px', color: C.textSecondary, marginBottom: 6 }}>
                {t('patternmatch:resolution.outputEstimate', {
                  w: resolution.mode === 'width' ? resolution.widthPx : model.canvas.wPx * resolution.scale,
                  h: resolution.mode === 'width' ? Math.round(resolution.widthPx * model.canvas.hPx / model.canvas.wPx) : model.canvas.hPx * resolution.scale,
                })}
              </div>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                {['png', 'jpeg'].map((f) => (
                  <label key={f} style={{ fontSize: '9pt', color: C.text, cursor: 'pointer' }} title={t('patternmatch:resolution.formatTooltip')}>
                    <input type="radio" name="fmt" checked={format === f} onChange={() => setFormat(f)} /> {f.toUpperCase()}
                  </label>
                ))}
                {format === 'jpeg' && (
                  <label style={{ fontSize: '8pt', color: C.textSecondary }} title={t('patternmatch:resolution.jpegQualityTooltip')}>{t('patternmatch:resolution.jpegQualityLabel')}
                    <input type="range" min={0.5} max={1} step={0.01} value={jpegQuality} onChange={(e) => setJpegQuality(Number(e.target.value))} style={{ width: 70 }} />
                    {jpegQuality.toFixed(2)}
                  </label>
                )}
              </div>
            </div>

            {error && <div style={{ color: C.red, fontSize: '9pt' }}>{error}</div>}
            <button onClick={onExport} disabled={busy} title={t('patternmatch:resolution.exportTooltip')} style={{
              padding: '10px', fontSize: '11pt', fontWeight: 700, cursor: busy ? 'wait' : 'pointer',
              background: C.green, color: C.bg, border: 'none', borderRadius: 4, opacity: busy ? 0.6 : 1,
            }}>{busy ? t('patternmatch:resolution.rendering') : t('common:export')}</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function rNccCaption(rNcc) {
  const r = rNcc?.r != null ? Number(rNcc.r).toFixed(4) : '—';
  const ncc = rNcc?.ncc != null ? Number(rNcc.ncc).toFixed(4) : '—';
  return `R = ${r} · NCC = ${ncc}`;
}

function Inspector({ el, patch, onRemove, t, srcLabel }) {
  if (!el) return <div style={box}><div style={{ fontSize: '9pt', color: C.textSecondary }}>{t('patternmatch:inspector.empty')}</div></div>;
  return (
    <div style={box}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={boxTitle}>{el.type === 'panel' ? t('patternmatch:inspector.panelTitle', { label: srcLabel(el.source) }) : el.type}</div>
        <button onClick={() => onRemove(el.id)} style={{ ...miniBtn, color: C.red }} title={t('patternmatch:inspector.removeTooltip')}>{t('common:remove')}</button>
      </div>
      {el.type === 'panel' && (
        <>
          <Row><Chk label={t('patternmatch:inspector.border')} title={t('patternmatch:inspector.borderTooltip')} v={el.border.on} on={(v) => patch(el.id, { border: { on: v } })} />
            <input type="color" value={el.border.color} onChange={(e) => patch(el.id, { border: { color: e.target.value } })} style={colorInput} title={t('patternmatch:inspector.colorTooltip')} />
            <input type="number" min={1} max={20} value={el.border.width} onChange={(e) => patch(el.id, { border: { width: Number(e.target.value) } })} style={numInput} title={t('patternmatch:inspector.borderWidthTooltip')} /></Row>
          <Row><Chk label={t('patternmatch:inspector.caption')} title={t('patternmatch:inspector.captionTooltip')} v={el.label.on} on={(v) => patch(el.id, { label: { on: v } })} />
            <input value={el.label.text} onChange={(e) => patch(el.id, { label: { text: e.target.value } })} style={txtInput} title={t('patternmatch:inspector.captionTextTooltip')} /></Row>
          <Row><Chk label={t('patternmatch:inspector.letter')} title={t('patternmatch:inspector.letterTooltip')} v={el.letter.on} on={(v) => patch(el.id, { letter: { on: v } })} />
            <select value={el.letter.style} onChange={(e) => patch(el.id, { letter: { style: e.target.value } })} style={tbSelect} title={t('patternmatch:inspector.letterTooltip')}>
              <option value="paren">{t('patternmatch:inspector.letterParen')}</option><option value="rparen">{t('patternmatch:inspector.letterRparen')}</option><option value="upper">{t('patternmatch:inspector.letterUpper')}</option>
            </select></Row>
          <Row><Chk label={t('patternmatch:inspector.markers')} title={t('patternmatch:inspector.markersTooltip')} v={el.includeMarkers} on={(v) => patch(el.id, { includeMarkers: v })} />
            <Chk label={t('patternmatch:inspector.aspectLock')} title={t('patternmatch:inspector.aspectLockTooltip')} v={el.aspectLock} on={(v) => patch(el.id, { aspectLock: v })} /></Row>
          <Row><label style={lbl} title={t('patternmatch:inspector.samplingTooltip')}>{t('patternmatch:inspector.sampling')}
            <select value={el.smoothing} onChange={(e) => patch(el.id, { smoothing: e.target.value })} style={tbSelect}>
              <option value="nearest">{t('patternmatch:inspector.samplingUpscaled')}</option><option value="bilinear">{t('patternmatch:inspector.samplingNative')}</option>
            </select></label></Row>
        </>
      )}
      {el.type === 'text' && (
        <>
          <Row><input value={el.text} onChange={(e) => patch(el.id, { text: e.target.value })} style={{ ...txtInput, flex: 1 }} title={t('patternmatch:inspector.textTooltip')} /></Row>
          <Row><label style={lbl} title={t('patternmatch:inspector.sizeTooltip')}>{t('patternmatch:inspector.size')}<input type="number" min={6} max={72} value={el.fontSize} onChange={(e) => patch(el.id, { fontSize: Number(e.target.value) })} style={numInput} /></label>
            <input type="color" value={el.color} onChange={(e) => patch(el.id, { color: e.target.value })} style={colorInput} title={t('patternmatch:inspector.colorTooltip')} />
            <Chk label={t('patternmatch:inspector.bold')} title={t('patternmatch:inspector.boldTooltip')} v={el.weight === 'bold'} on={(v) => patch(el.id, { weight: v ? 'bold' : 'normal' })} /></Row>
        </>
      )}
      {el.type === 'colorbar' && (
        <>
          <Row><Chk label={t('patternmatch:inspector.visible')} title={t('patternmatch:inspector.visibleTooltip')} v={el.visible !== false} on={(v) => patch(el.id, { visible: v })} /></Row>
          <Row><label style={lbl} title={t('patternmatch:inspector.minMaxTooltip')}>{t('patternmatch:inspector.min')}<input type="number" value={el.vmin} step={0.1} onChange={(e) => patch(el.id, { vmin: Number(e.target.value) })} style={numInput} /></label>
            <label style={lbl} title={t('patternmatch:inspector.minMaxTooltip')}>{t('patternmatch:inspector.max')}<input type="number" value={el.vmax} step={0.1} onChange={(e) => patch(el.id, { vmax: Number(e.target.value) })} style={numInput} /></label></Row>
        </>
      )}
      {el.type === 'scalebar' && (
        <>
          <Row><label style={lbl} title={t('patternmatch:inspector.lengthTooltip')}>{t('patternmatch:inspector.length')}<input type="number" value={el.lengthValue} onChange={(e) => patch(el.id, { lengthValue: Number(e.target.value), labelText: `${e.target.value} ${el.unit}` })} style={numInput} /></label>
            <span style={{ fontSize: '8pt', color: C.textSecondary }}>{el.unit}</span></Row>
          <Row><input type="color" value={el.color} onChange={(e) => patch(el.id, { color: e.target.value })} style={colorInput} title={t('patternmatch:inspector.colorTooltip')} /></Row>
        </>
      )}
    </div>
  );
}

// ---- tiny styled helpers ----------------------------------------------------
const Row = ({ children }) => <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>{children}</div>;
const Chk = ({ label, v, on, title }) => <label style={{ fontSize: '9pt', color: C.text, cursor: 'pointer' }} title={title}><input type="checkbox" checked={!!v} onChange={(e) => on(e.target.checked)} /> {label}</label>;
const isPreset = (bg) => bg === '#ffffff' || bg === '#000000' || bg === 'transparent';
function handlePos(h) {
  const m = -5;
  const map = {
    nw: { left: m, top: m }, n: { left: '50%', top: m, marginLeft: -4 }, ne: { right: m, top: m },
    e: { right: m, top: '50%', marginTop: -4 }, se: { right: m, bottom: m }, s: { left: '50%', bottom: m, marginLeft: -4 },
    sw: { left: m, bottom: m }, w: { left: m, top: '50%', marginTop: -4 },
  };
  return map[h];
}
const tbBtn = { padding: '4px 10px', height: 28, background: C.bgSecondary, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, cursor: 'pointer', fontSize: '11px', lineHeight: 1, whiteSpace: 'nowrap' };
const miniBtn = { padding: '2px 6px', background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, cursor: 'pointer', fontSize: '8pt' };
const tbSelect = { background: C.bgSecondary, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '3px 8px', height: 28, fontSize: '11px', cursor: 'pointer', boxSizing: 'border-box' };
// toolbar cluster styling (publication composer)
const tbBar = { display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'flex-start', marginBottom: 12, paddingBottom: 12, borderBottom: `1px solid ${C.border}` };
const tbGroup = { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'nowrap', padding: '5px 10px', borderRadius: 6, border: `1px solid ${C.border}`, background: C.bg };
const tbGroupLabel = { fontSize: '10px', fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: C.textSecondary, marginRight: 2, userSelect: 'none' };
const tbBtnOk = { background: C.green, color: C.bg, borderColor: C.green };
const lbl = { fontSize: '9pt', color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 };
const numInput = { width: 64, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, padding: '2px 4px', fontSize: '9pt' };
const txtInput = { background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, padding: '2px 6px', fontSize: '9pt' };
const colorInput = { width: 28, height: 22, padding: 0, border: `1px solid ${C.border}`, background: 'none' };
const box = { background: C.bg, border: `1px solid ${C.border}`, borderRadius: 4, padding: 8 };
const boxTitle = { fontSize: '9pt', fontWeight: 700, color: C.text, marginBottom: 2 };
const checker = 'repeating-conic-gradient(#2a2a2a 0% 25%, #1e1e1e 0% 50%) 50% / 16px 16px';
