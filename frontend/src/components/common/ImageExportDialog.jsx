import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { colors, Button, Select, NumberInput, Input } from '../../theme/components';
import {
  FORMATS, formatById, SCALE_FACTORS,
  DEFAULT_CAPTION_STYLE, DEFAULT_SCALEBAR_STYLE,
  clampCrop, fullCrop, isFullCrop, resizeCrop,
  outputSize, defaultFactor, niceScalebar, scalebarOptions, formatUnits, buildFilename,
  captionLayout, scalebarLayout,
  moveAnnotation, resizeCaptionStyle, resizeScalebarStyle,
  renderExportCanvas, drawAnnotations, canvasToBlob, loadImage, saveImageBlob, rgba,
  NO_MARGINS, BORDER_INPUT_MAX_FRACTION, canvasSizeWithMargins, widerMargins,
} from './imageExport';
import { makeFolderWriter, planBatchFiles, runImageBatch } from './batchImageExport';

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
const HANDLE_CURSOR = {
  nw: 'nwse-resize', se: 'nwse-resize',
  ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize',
  e: 'ew-resize', w: 'ew-resize',
};
// A press shorter than this counts as "clicked the image", not "dragged a new
// crop" — so a stray click does not silently shrink the selection to nothing.
const NEW_CROP_MIN_PX = 4;

/**
 * Collapse any line break to a space.
 *
 * The caption is one logical line that gets wrapped at draw time, so a typed
 * newline would silently vanish into the wrap. Built from char codes rather
 * than a regex literal purely for readability of the escapes.
 */
const LF = String.fromCharCode(10);
const CR = String.fromCharCode(13);
const oneLine = (v) => String(v ?? "").split(CR).join(" ").split(LF).join(" ");

/**
 * Geometry plus the typography the block was drawn with — the in-place editor
 * reuses it so you edit the text at the size and colour it actually has,
 * rather than in a small generic box next to it.
 */
const pick = (L) => ({
  x: L.x, y: L.y, w: L.w, h: L.h,
  fontPx: L.fontPx, lineH: L.lineH, pad: L.pad,
  color: L.st?.color, bold: L.st?.bold,
  background: L.st?.background, backgroundOpacity: L.st?.backgroundOpacity,
});

function Field({ label, children, hint }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary, marginBottom: 3 }}>{label}</div>
      {children}
      {hint && <div style={{ fontSize: '7.5pt', color: colors.textSecondary, marginTop: 2, opacity: 0.8 }}>{hint}</div>}
    </div>
  );
}

/** Compact labelled row used inside the style editors. */
function Row({ label, children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
      <span style={{ fontSize: '8pt', color: colors.textSecondary, width: 74, flexShrink: 0 }}>{label}</span>
      {children}
    </div>
  );
}

/**
 * Text field that selects its whole content the first time you click into it,
 * and behaves like a normal field on every click after that.
 *
 * The mouseup guard is the load-bearing part: focus fires first and selects,
 * then the browser's own mouseup would collapse that selection to a caret. We
 * swallow only that one mouseup.
 */
function SelectAllInput({ value, onChange, style, title, inputRef }) {
  // Armed only by a press on an UNFOCUSED field. Arming in onFocus instead
  // caught keyboard focus too, and then the first real click could not place a
  // caret because its mouseup was being swallowed.
  const armed = useRef(false);
  return (
    <Input
      ref={inputRef}
      value={value}
      onChange={onChange}
      title={title}
      style={style}
      onMouseDown={(e) => { armed.current = document.activeElement !== e.currentTarget; }}
      onFocus={(e) => { if (armed.current) e.target.select(); }}
      onMouseUp={(e) => { if (armed.current) { e.preventDefault(); armed.current = false; } }}
      onBlur={() => { armed.current = false; }}
    />
  );
}

function ColorBox({ value, onChange, title }) {
  return (
    <input
      type="color"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      title={title}
      style={{
        width: 34, height: 22, padding: 0, flexShrink: 0,
        background: colors.bg, border: `1px solid ${colors.border}`,
        borderRadius: 4, cursor: 'pointer',
      }}
    />
  );
}

function Range({ value, onChange, min, max, step = 0.05, title }) {
  return (
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      title={title}
      style={{ flex: 1, minWidth: 0, accentColor: colors.purple }}
    />
  );
}

/**
 * Appearance editor shared by the caption and the scale bar.
 *
 * `extra` carries whatever is specific to one of them (the caption's free text,
 * the bar's thickness), so the common controls stay in one place.
 */
function StyleEditor({ style, onChange, t, showBarScale, children }) {
  const set = (patch) => onChange({ ...style, ...patch });
  return (
    <div style={{
      marginLeft: 19, marginBottom: 8, paddingLeft: 8,
      borderLeft: `2px solid ${colors.border}`,
    }}>
      {children}
      <Row label={t('imageexport:style.size')}>
        <Range value={style.fontScale} onChange={(v) => set({ fontScale: v })} min={0.4} max={4} title={t('imageexport:style.sizeTooltip')} />
        <span style={{ fontSize: '7.5pt', color: colors.textSecondary, width: 34, textAlign: 'right' }}>
          {style.fontScale.toFixed(1)}×
        </span>
      </Row>
      {showBarScale && (
        <Row label={t('imageexport:style.thickness')}>
          <Range value={style.barScale} onChange={(v) => set({ barScale: v })} min={0.3} max={6} title={t('imageexport:style.thicknessTooltip')} />
          <span style={{ fontSize: '7.5pt', color: colors.textSecondary, width: 34, textAlign: 'right' }}>
            {style.barScale.toFixed(1)}×
          </span>
        </Row>
      )}
      <Row label={t('imageexport:style.color')}>
        <ColorBox value={style.color} onChange={(v) => set({ color: v })} title={t('imageexport:style.colorTooltip')} />
        <Check
          checked={style.bold}
          onChange={(v) => set({ bold: v })}
          label={t('imageexport:style.bold')}
          title={t('imageexport:style.boldTooltip')}
        />
      </Row>
      <Row label={t('imageexport:style.background')}>
        <ColorBox value={style.background} onChange={(v) => set({ background: v })} title={t('imageexport:style.backgroundTooltip')} />
        <Range value={style.backgroundOpacity} onChange={(v) => set({ backgroundOpacity: v })} min={0} max={1} title={t('imageexport:style.opacityTooltip')} />
        <span style={{ fontSize: '7.5pt', color: colors.textSecondary, width: 34, textAlign: 'right' }}>
          {style.backgroundOpacity === 0 ? t('imageexport:style.none') : `${Math.round(style.backgroundOpacity * 100)}%`}
        </span>
      </Row>
      <div style={{ fontSize: '7.5pt', color: colors.textSecondary, marginTop: 3 }}>
        {t('imageexport:style.dragHint')}
      </div>
    </div>
  );
}

/**
 * The scale bar the caller owns, edited from inside the export dialog.
 *
 * Deliberately smaller than the dialog's own StyleEditor: it offers only what
 * the bound bar actually HAS — length, colour, text size. Showing this dialog's
 * background and boldness controls here would put three dead sliders in front
 * of the user.
 */
function BoundScalebarEditor({ binding, unitLabel, t }) {
  const choices = binding.lengthChoices?.length ? binding.lengthChoices : [1, 2, 5, 10, 20, 50, 100];
  return (
    <div style={{
      marginLeft: 19, marginBottom: 8, paddingLeft: 8,
      borderLeft: `2px solid ${colors.border}`,
    }}>
      <Row label={t('imageexport:style.length')}>
        {/* The wrapper carries the test handle: `Select` renders a fixed set of
            attributes and drops anything else. */}
        <div data-bound-scalebar-length style={{ flex: 1, minWidth: 0 }}>
          <Select
            value={String(binding.lengthUm)}
            onChange={(e) => binding.onChange({ lengthUm: Number(e.target.value) })}
            options={choices.map((v) => ({ value: String(v), label: `${v} ${unitLabel}` }))}
            style={{ width: '100%' }}
            title={t('imageexport:style.lengthTooltip')}
          />
        </div>
      </Row>
      <Row label={t('imageexport:style.size')}>
        <Range
          value={binding.fontSize}
          onChange={(v) => binding.onChange({ fontSize: Math.round(v) })}
          min={6} max={48}
          title={t('imageexport:style.sizeTooltip')}
        />
        <span style={{ fontSize: '7.5pt', color: colors.textSecondary, width: 34, textAlign: 'right' }}>
          {Math.round(binding.fontSize)}
        </span>
      </Row>
      <Row label={t('imageexport:style.color')}>
        <ColorBox
          value={binding.color}
          onChange={(v) => binding.onChange({ color: v })}
          title={t('imageexport:style.colorTooltip')}
        />
      </Row>
      <div style={{ fontSize: '7.5pt', color: colors.textSecondary, marginTop: 3 }}>
        {t('imageexport:scalebarShared')}
      </div>
    </div>
  );
}

function Check({ checked, onChange, label, disabled, title }) {
  return (
    <label
      title={title}
      style={{
        display: 'flex', alignItems: 'center', gap: 6, fontSize: '9pt',
        color: disabled ? colors.textSecondary : colors.text,
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'default' : 'pointer', marginBottom: 4,
      }}
    >
      <input
        type="checkbox"
        checked={!!checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: colors.purple, width: 13, height: 13, flexShrink: 0 }}
      />
      {label}
    </label>
  );
}

/**
 * Export one image to PNG/JPEG/WebP with an optional crop.
 *
 * The preview is produced by the SAME functions that write the file
 * (`renderExportCanvas` + `drawAnnotations`), so what is shown is what is
 * written — no second rendering path that could drift.
 *
 * Props
 *   src              data: URL of the image as the viewer holds it
 *   title            heading, e.g. "Overview"
 *   defaultBaseName  suggested filename without extension
 *   displayFilter    CSS filter the viewer applies on screen (or null)
 *   unitsPerPixel    physical size of one image pixel, or null when there is none
 *   unitLabel        'µm'
 *   annotations      { crosshair, roi, label } — each may be null/absent
 */
export default function ImageExportDialog({
  open, onClose, src, title,
  defaultBaseName = 'export',
  displayFilter = null,
  unitsPerPixel = null,
  unitLabel = 'µm',
  annotations = {},
  // Region to start on, in image pixels. Used where the viewer itself shows
  // only part of the image (the phase map auto-zooms to the indexed area), so
  // the export opens on what you were looking at instead of the whole grid.
  defaultCrop = null,
  // Two halves of the same idea: `overlay(rect)` renders editable extras ON the
  // preview (positioned against the image rect it is handed), and `drawOverlay`
  // burns the same extras into the file. The phase map uses them to keep its
  // legend/title/scale bar/arrow adjustable right here.
  overlay = null,
  drawOverlay = null,
  // Caller-supplied controls, rendered at the top of the settings column.
  // The overlay's own editor lives here so it does not cover the picture.
  sidePanel = null,
  // A border the picture NEEDS, as per-side fractions of the image: the phase
  // map places its colour key and value scales beside the map, and without the
  // room for them the file would cut off what the preview shows. Applied
  // whenever `fitMarginsKey` changes — that key names the set of things being
  // fitted, so ticking a scale re-fits the border while merely dragging one
  // does not resize the picture under the user's hand.
  fitMargins = null,
  fitMarginsKey = null,
  // Hands this dialog's scale-bar controls to a bar the CALLER owns — the one
  // the user placed on the map. There is then exactly one bar in the figure,
  // reachable from both places, instead of the dialog adding a second one.
  //   { present, lengthUm, color, fontSize, lengthChoices,
  //     onToggle(bool), onChange({lengthUm?, color?, fontSize?}) }
  scalebarBinding = null,
  // Turns this dialog into the settings for a SERIES. `items` are the other
  // pictures the same settings should be applied to:
  //   { items: [{ id, label, build }], stem }
  // where `build()` returns what the host's own export builders return —
  // `{ canvas, umPerPx }` — so a batch picture is built by the same code as a
  // single one. Absent (the default) the dialog behaves exactly as before.
  batch = null,
  onExported,
}) {
  const { t } = useTranslation(['imageexport', 'common']);

  const [img, setImg] = useState(null);
  const [natural, setNatural] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const [crop, setCrop] = useState(null);
  const [format, setFormat] = useState('png');
  const [quality, setQuality] = useState(92);
  const [scaleMode, setScaleMode] = useState({ type: 'factor', factor: 1, targetWidth: 2000 });
  const [smoothing, setSmoothing] = useState(false);
  const [rawValues, setRawValues] = useState(false);
  const [showScalebar, setShowScalebar] = useState(false);
  const [showCrosshair, setShowCrosshair] = useState(false);
  const [showRoi, setShowRoi] = useState(false);
  const [showLabel, setShowLabel] = useState(false);
  const [captionText, setCaptionText] = useState('');
  const [captionStyle, setCaptionStyle] = useState(DEFAULT_CAPTION_STYLE);
  const [scalebarStyle, setScalebarStyle] = useState(DEFAULT_SCALEBAR_STYLE);
  // Extra space added around the image, as a fraction of the image size —
  // somewhere to put a caption without covering the data.
  const [margins, setMargins] = useState(NO_MARGINS);
  const [marginColor, setMarginColor] = useState('#000000');
  const [baseName, setBaseName] = useState(defaultBaseName);
  const [busy, setBusy] = useState(false);
  const [maximised, setMaximised] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const previewBoxRef = useRef(null);
  const dragRef = useRef(null);
  const measureRef = useRef(null);
  const [previewRect, setPreviewRect] = useState(null); // image area inside the box, CSS px
  // Where the caption / scale bar were drawn on the preview, in canvas px plus
  // the crop origin offset. Feeds the drag overlay.
  const [boxes, setBoxes] = useState({ caption: null, scalebar: null });
  // Set when the caption block on the preview was clicked (not dragged), so
  // the text can be retyped where it actually sits.
  // Holds the caption block's geometry, frozen at the moment editing started.
  // Deriving it live from `boxes.caption` made the field jump and resize on
  // every keystroke — and vanish completely the moment the text was empty,
  // because an empty caption draws no block.
  const [editingCaption, setEditingCaption] = useState(null);
  const inlineRef = useRef(null);
  const boxesRef = useRef(boxes);
  boxesRef.current = boxes;

  const hasScale = Number.isFinite(unitsPerPixel) && unitsPerPixel > 0;
  const hasCrosshair = !!annotations?.crosshair;
  const hasRoi = !!annotations?.roi;
  const hasLabel = !!annotations?.label;

  // --- load the image -------------------------------------------------------
  useEffect(() => {
    if (!open || !src) return undefined;
    let cancelled = false;
    setImg(null); setNatural(null); setLoadError(null); setError(null);
    loadImage(src)
      .then((image) => {
        if (cancelled) return;
        const nat = { width: image.naturalWidth, height: image.naturalHeight };
        setImg(image);
        setNatural(nat);
        setCrop(defaultCrop ? clampCrop(defaultCrop, nat) : fullCrop(nat));
        // A montage of every EDS map is already thousands of pixels wide; a
        // flat 4x there would blow past what the browser can encode.
        setScaleMode((m) => ({ ...m, type: 'factor', factor: defaultFactor(nat) }));
      })
      .catch((e) => { if (!cancelled) setLoadError(e.message); });
    return () => { cancelled = true; };
  }, [open, src, defaultCrop]);

  useEffect(() => { if (open) setBaseName(defaultBaseName); }, [open, defaultBaseName]);
  // Seed the editable caption from the auto-generated one each time the dialog
  // opens, so it always starts from the current file/pixel rather than from
  // whatever was typed for a previous image.
  useEffect(() => {
    if (open) {
      setCaptionText(annotations?.label || '');
      setEditingCaption(null);
      setMargins(NO_MARGINS);
    }
  }, [open, annotations?.label]);

  // Fit the border to whatever sits beside the picture. Deliberately keyed on
  // WHICH things are out there rather than on where they are, so the sheet
  // does not resize on every mouse-move of a drag.
  useEffect(() => {
    if (!open || !fitMargins) return;
    setMargins({
      top: fitMargins.top || 0, right: fitMargins.right || 0,
      bottom: fitMargins.bottom || 0, left: fitMargins.left || 0,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, fitMarginsKey]);

  // --- geometry of the preview ---------------------------------------------
  const measurePreview = useCallback(() => {
    const box = previewBoxRef.current;
    if (!box || !natural || !crop) return;
    // Re-fitting mid-drag would make the crop rectangle swim away from the
    // pointer, so the scale is only recomputed between drags.
    if (dragRef.current) return;
    // clientWidth/Height rather than the client RECT: the box has a 1px
    // border and the canvas is inset:0 inside it, so the border box is 2px
    // wider than the area actually drawn into.
    const b = { width: box.clientWidth, height: box.clientHeight };
    if (!b.width || !b.height) return;

    // Fit what is actually being EXPORTED — the crop plus its border — into the
    // box, not the whole source. A phase map covers about half its scan grid,
    // so fitting the grid left the exported part small in a large empty frame.
    // The rest of the image is still drawn around it (dimmed) for context and
    // simply clipped by the box.
    const mg = margins || NO_MARGINS;
    const fitW = crop.width * (1 + Math.max(0, mg.left) + Math.max(0, mg.right));
    const fitH = crop.height * (1 + Math.max(0, mg.top) + Math.max(0, mg.bottom));
    const k = Math.min(b.width / fitW, b.height / fitH) * 0.98;

    // Centre the crop in the box; the image is placed relative to it.
    setPreviewRect({
      left: b.width / 2 - (crop.x + crop.width / 2) * k,
      top: b.height / 2 - (crop.y + crop.height / 2) * k,
      width: natural.width * k,
      height: natural.height * k,
      scale: k,
    });
  }, [natural, crop, margins]);

  measureRef.current = measurePreview;

  useEffect(() => {
    measurePreview();
    const box = previewBoxRef.current;
    if (!box) return undefined;
    const ro = new ResizeObserver(measurePreview);
    ro.observe(box);
    window.addEventListener('resize', measurePreview);
    return () => { ro.disconnect(); window.removeEventListener('resize', measurePreview); };
  }, [measurePreview, open]);

  const fmt = formatById(format);
  const output = useMemo(
    () => (crop ? outputSize(crop, scaleMode) : null),
    [crop, scaleMode],
  );
  // Bar lengths that fit the current crop — recomputed as the crop changes, so
  // the menu never offers a bar wider than the image.
  const barLengthChoices = useMemo(
    () => (hasScale && crop ? scalebarOptions(unitsPerPixel, crop.width) : []),
    [hasScale, unitsPerPixel, crop],
  );

  // The annotation spec, resolved for a given output geometry. Scalebar length
  // is chosen from the CROP width, so zooming into a corner still gets a bar
  // that fits and reads correctly.
  /**
   * What gets drawn ON the exported picture.
   *
   * `opts` lets a SERIES pass the picture's own numbers: every map has its own
   * micrometres per pixel (on a real file the electron images sit on the SEM
   * raster and the maps on the scan raster, 10.6x apart), and its own name —
   * a series where every file is captioned "Al" would be mislabelled figures.
   */
  const buildSpec = useCallback((c, opts = null) => {
    const upp = opts && 'unitsPerPx' in opts ? opts.unitsPerPx : unitsPerPixel;
    const scalable = Number.isFinite(upp) && upp > 0;
    const caption = opts?.label ?? captionText;
    let bar = null;
    // With a binding, the bar is drawn by whoever owns it; this dialog only
    // edits it. Gating just the checkbox would leave a second bar on screen
    // when the owner's bar appears after the box was ticked.
    if (showScalebar && scalable && !scalebarBinding) {
      const forced = scalebarStyle.lengthUnits;
      // A forced length that no longer fits the crop would draw a bar wider
      // than the image, so fall back to the automatic pick in that case.
      bar = (forced && forced / upp <= c.width)
        ? { lengthUnits: forced, lengthPx: forced / upp, text: formatUnits(forced) }
        : niceScalebar(upp, c.width, 0.25);
    }
    return {
      scalebar: bar ? { ...bar, unitLabel } : null,
      scalebarStyle,
      crosshair: showCrosshair && hasCrosshair ? annotations.crosshair : null,
      roi: showRoi && hasRoi ? annotations.roi : null,
      // Empty text means the user cleared the field — treat that as "no caption"
      // rather than drawing an empty plate.
      label: showLabel && String(caption).trim() ? caption : null,
      labelStyle: captionStyle,
    };
  }, [showScalebar, scalebarBinding, unitsPerPixel, unitLabel, scalebarStyle, showCrosshair,
      hasCrosshair, showRoi, hasRoi, showLabel, captionText, captionStyle, annotations]);

  // --- draw the preview -----------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img || !natural || !crop || !previewRect) return;
    // The canvas covers the whole preview box; the image sits inside it at
    // `previewRect`, and the exported region (crop + border) is drawn on top of
    // that. Keeping the canvas box-sized lets the border extend past the image
    // without changing how the image is fitted.
    const boxEl = previewBoxRef.current;
    if (!boxEl) return;
    const boxW = Math.max(1, boxEl.clientWidth);
    const boxH = Math.max(1, boxEl.clientHeight);
    canvas.width = boxW;
    canvas.height = boxH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const s = previewRect.scale;
    const ix = previewRect.left;
    const iy = previewRect.top;
    const cx = ix + crop.x * s;
    const cy = iy + crop.y * s;
    const cw = crop.width * s;
    const ch = crop.height * s;

    // Border, in the same proportion it will have in the file: the margins are
    // fractions of the exported image, which on screen is the crop rectangle.
    const mg = margins || NO_MARGINS;
    const mL = Math.max(0, mg.left) * cw;
    const mR = Math.max(0, mg.right) * cw;
    const mT = Math.max(0, mg.top) * ch;
    const mB = Math.max(0, mg.bottom) * ch;

    ctx.clearRect(0, 0, boxW, boxH);
    if (mL || mR || mT || mB) {
      ctx.fillStyle = marginColor;
      ctx.fillRect(cx - mL, cy - mT, cw + mL + mR, ch + mT + mB);
    }

    ctx.imageSmoothingEnabled = smoothing;
    if (!rawValues && displayFilter) ctx.filter = displayFilter;
    ctx.drawImage(img, 0, 0, natural.width, natural.height,
      ix, iy, previewRect.width, previewRect.height);
    ctx.filter = 'none';

    // Dim everything outside the crop so the kept region reads at a glance.
    if (!isFullCrop(crop, natural)) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(ix, iy, previewRect.width, cy - iy);
      ctx.fillRect(ix, cy + ch, previewRect.width, iy + previewRect.height - cy - ch);
      ctx.fillRect(ix, cy, cx - ix, ch);
      ctx.fillRect(cx + cw, cy, ix + previewRect.width - cx - cw, ch);
    }

    // Annotations sit in the exported region (border included), exactly as on
    // export — so the caption can be dragged into the border here too.
    const spec = buildSpec(crop);
    const geom = {
      crop,
      output: { width: cw + mL + mR, height: ch + mT + mB },
      origin: { x: mL, y: mT },
      sx: cw / crop.width,
      sy: ch / crop.height,
    };
    ctx.save();
    ctx.translate(cx - mL, cy - mT);
    // While the caption is being edited the textarea IS the caption; drawing it
    // as well left the old text showing through underneath the new one.
    drawAnnotations(ctx, editingCaption ? { ...spec, label: null } : spec, geom);
    ctx.restore();

    // Publish where the blocks landed so the drag overlay sits exactly on top
    // of what was drawn — same layout functions, no second guess. ox/oy carry
    // the exported region's offset inside the preview box.
    const at = { ox: cx - mL, oy: cy - mT };
    setBoxes({
      caption: spec.label ? { ...pick(captionLayout(ctx, spec, geom)), ...at } : null,
      scalebar: spec.scalebar ? { ...pick(scalebarLayout(ctx, spec, geom)), ...at } : null,
    });
  }, [img, natural, crop, previewRect, smoothing, rawValues, displayFilter, buildSpec, margins, marginColor, editingCaption]);

  // --- crop dragging --------------------------------------------------------
  const toImage = useCallback((clientX, clientY) => {
    const box = previewBoxRef.current;
    if (!box || !previewRect || !natural) return null;
    const b = box.getBoundingClientRect();
    return {
      x: (clientX - b.left - previewRect.left) / previewRect.scale,
      y: (clientY - b.top - previewRect.top) / previewRect.scale,
    };
  }, [previewRect, natural]);

  // The window listeners are added on mousedown and removed on mouseup. Their
  // identity must survive the whole drag, but the real handlers close over
  // `previewRect` and `natural`, which can change mid-drag (a window resize).
  // Indirecting through refs keeps the add/remove pair matched — otherwise a
  // re-render mid-drag orphans the listener and the crop follows the mouse for
  // ever.
  const moveRef = useRef(null);
  const upRef = useRef(null);
  const stableMove = useCallback((e) => moveRef.current?.(e), []);
  const stableUp = useCallback((e) => upRef.current?.(e), []);

  const onPointerDown = (e, handle, kind = 'crop') => {
    if (!natural || !crop || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const p = toImage(e.clientX, e.clientY);
    if (!p) return;
    dragRef.current = {
      kind, handle,
      startCrop: crop,
      startImg: p,
      startClient: { x: e.clientX, y: e.clientY },
      startStyle: kind === 'caption' ? captionStyle : (kind === 'scalebar' ? scalebarStyle : null),
      // Annotation positions are fractions of the CROP area, so deltas must be
      // divided by the crop's on-screen size, not the whole preview.
      cropPx: previewRect
        ? { w: crop.width * previewRect.scale, h: crop.height * previewRect.scale }
        : null,
    };
    window.addEventListener('mousemove', stableMove);
    window.addEventListener('mouseup', stableUp);
  };

  const onWindowMove = useCallback((e) => {
    const d = dragRef.current;
    if (!d || !natural) return;

    if (d.kind === 'caption' || d.kind === 'scalebar') {
      const cp = d.cropPx;
      if (!cp?.w || !cp?.h) return;
      const dxF = (e.clientX - d.startClient.x) / cp.w;
      const dyF = (e.clientY - d.startClient.y) / cp.h;
      const apply = d.kind === 'caption' ? setCaptionStyle : setScalebarStyle;
      const resize = d.kind === 'caption' ? resizeCaptionStyle : resizeScalebarStyle;
      apply(d.handle === 'move'
        ? moveAnnotation(d.startStyle, dxF, dyF)
        : resize(d.startStyle, d.handle, dxF, dyF));
      return;
    }

    const p = toImage(e.clientX, e.clientY);
    if (!p) return;
    if (d.handle === 'new') {
      const moved = Math.abs(e.clientX - d.startClient.x) + Math.abs(e.clientY - d.startClient.y);
      if (moved < NEW_CROP_MIN_PX) return;
      setCrop(clampCrop({
        x: Math.min(d.startImg.x, p.x),
        y: Math.min(d.startImg.y, p.y),
        width: Math.abs(p.x - d.startImg.x),
        height: Math.abs(p.y - d.startImg.y),
      }, natural));
      return;
    }
    setCrop(resizeCrop(d.handle, d.startCrop, p.x - d.startImg.x, p.y - d.startImg.y, natural));
  }, [natural, toImage]);

  const onWindowUp = useCallback((e) => {
    const d = dragRef.current;
    // A press on the caption that never travelled is a click, not a drag —
    // that opens the in-place editor.
    if (d && d.kind === 'caption' && d.handle === 'move' && e) {
      const moved = Math.abs(e.clientX - d.startClient.x) + Math.abs(e.clientY - d.startClient.y);
      const b = boxesRef.current?.caption;
      if (moved <= NEW_CROP_MIN_PX && b) {
        setEditingCaption({ ...b, w: Math.max(160, b.w), h: Math.max(b.fontPx + b.pad / 2, b.h) });
      }
    }
    const wasCrop = d && d.kind === 'crop';
    dragRef.current = null;
    window.removeEventListener('mousemove', stableMove);
    window.removeEventListener('mouseup', stableUp);
    // The fit is held still while dragging so the crop stays under the pointer;
    // once released, re-fit to the region that is now being exported.
    if (wasCrop) measureRef.current?.();
  }, [stableMove, stableUp]);

  moveRef.current = onWindowMove;
  upRef.current = onWindowUp;

  // Unmounting mid-drag (Escape, or the parent closing) must not leave the
  // listeners attached to the window.
  useEffect(() => () => {
    window.removeEventListener('mousemove', stableMove);
    window.removeEventListener('mouseup', stableUp);
  }, [stableMove, stableUp]);

  // A press anywhere outside the in-place editor leaves it.
  //
  // onBlur alone is not enough: the preview calls preventDefault() on mousedown
  // to drive the crop drag, and that also cancels the browser's focus change —
  // so the textarea never blurred and you stayed "in" it after clicking away.
  // Capture phase, so it runs before anything can swallow the event.
  useEffect(() => {
    if (!editingCaption) return undefined;
    const onDown = (e) => {
      if (!inlineRef.current?.contains(e.target)) setEditingCaption(null);
    };
    document.addEventListener('mousedown', onDown, true);
    return () => document.removeEventListener('mousedown', onDown, true);
  }, [editingCaption]);

  // --- escape closes --------------------------------------------------------
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape' && !busy) { e.stopPropagation(); onClose?.(); } };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [open, busy, onClose]);

  // --- export ---------------------------------------------------------------

  /**
   * One export, rendered to a blob.
   *
   * The button below runs it once for the picture on screen; a batch runs it
   * per picture with the SAME settings. One path on purpose — a series that
   * went through a second renderer could come out looking unlike the single
   * export the user checked it against.
   */
  const renderToBlob = useCallback(async ({ image, crop: c, unitsPerPx, label, overlayFor }) => {
    // The border the figure NEEDS, whatever the border currently IS. The
    // fit effect runs only when `fitMarginsKey` changes — by design, so
    // dragging a body does not resize the picture mid-gesture — so a body
    // that grew or moved after that point had no room reserved and the
    // canvas simply cut it off. Taking the wider of the two at save time
    // keeps the gesture calm AND the file complete.
    const exportMargins = widerMargins(margins, fitMargins);
    const out = outputSize(c, scaleMode);
    const canvas = renderExportCanvas({
      image,
      crop: c,
      output: out,
      smoothing,
      filter: rawValues ? null : displayFilter,
      // Lossy formats cannot carry alpha; compositing onto black matches the
      // viewer's own backdrop instead of the browser's default.
      background: fmt.lossy ? '#000000' : null,
      margins: exportMargins,
      marginColor,
    });
    const total = canvasSizeWithMargins(out, exportMargins);
    const ctx = canvas.getContext('2d');
    const geom = {
      crop: c,
      output: { width: total.width, height: total.height },
      origin: total.origin,
      sx: out.width / c.width,
      sy: out.height / c.height,
      // How much bigger the FILE is than the PREVIEW the user arranged on.
      //
      // Overlay extras are laid out in preview CSS pixels, so this — not
      // `sx` — is the factor their lettering has to grow by. `sx` counts
      // output pixels per SOURCE DATA pixel, which is a different number
      // entirely whenever the preview does not happen to show the data at
      // 1:1. On a 136x39 scan exported at 8x it was ~4x too large: the
      // scale bar's plate filled 79% of the picture height and its label
      // fell off the bottom edge (reported 2026-09-03).
      //
      // The dialog computes it because only the dialog knows both halves.
      previewScale: previewRect?.scale ?? null,
      textScale: previewRect?.scale > 0
        ? (out.width / c.width) / previewRect.scale
        : 1,
    };
    // The caller's extras go down first, so the dialog's own caption and
    // scale bar sit on top of them — the same stacking as on the preview.
    overlayFor?.(ctx, geom);
    drawAnnotations(ctx, buildSpec(c, { unitsPerPx, label }), geom);
    return canvasToBlob(canvas, fmt.mime, fmt.lossy ? quality / 100 : undefined);
  }, [margins, fitMargins, scaleMode, smoothing, rawValues, displayFilter, fmt,
      marginColor, previewRect, buildSpec, quality]);

  const doExport = async () => {
    if (!img || !crop || !output) return;
    setBusy(true);
    setError(null);
    try {
      const blob = await renderToBlob({
        image: img, crop, unitsPerPx: unitsPerPixel, overlayFor: drawOverlay,
      });
      const filename = buildFilename(baseName, format);
      const res = await saveImageBlob(blob, filename, format);
      if (!res) { setBusy(false); return; }   // user cancelled the save dialog
      onExported?.(res);
      onClose?.();
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  // --- the same settings, applied to a series --------------------------------
  const [batchRun, setBatchRun] = useState(null);
  const [batchReport, setBatchReport] = useState(null);
  const batchCancelRef = useRef(false);
  const batchCount = batch?.items?.length ?? 0;

  const doBatchExport = async () => {
    if (!batchCount || !crop || !natural) return;
    setError(null);
    setBatchReport(null);
    batchCancelRef.current = false;

    // The folder is chosen once; a save dialog per picture is exactly what a
    // series has to avoid.
    const writer = makeFolderWriter();
    let target = null;
    try {
      target = await writer.pick();
    } catch (e) {
      setError(e?.message || String(e));
      return;
    }
    if (!target) return;   // cancelled the folder dialog

    setBusy(true);
    const plan = planBatchFiles(batch.items, {
      stem: batch.stem ?? baseName,
      ext: fmt.ext,
      labelOf: (i) => i.label,
    });
    // The crop as FRACTIONS of the picture it was drawn on. The maps of a
    // series need not share a raster, and "the same crop" across pictures of
    // different sizes can only mean the same relative region.
    const frac = {
      x: crop.x / natural.width, y: crop.y / natural.height,
      w: crop.width / natural.width, h: crop.height / natural.height,
    };
    let res;
    try {
      res = await runImageBatch({
        plan,
        isCancelled: () => batchCancelRef.current,
        onProgress: (p) => setBatchRun(p.done ? null : p),
        render: async (item) => {
          const built = await item.build();
          const image = built?.canvas ?? built;
          if (!image?.width || !image?.height) {
            throw new Error(t('imageexport:batchNoPicture'));
          }
          const c = {
            x: Math.round(frac.x * image.width),
            y: Math.round(frac.y * image.height),
            width: Math.max(1, Math.round(frac.w * image.width)),
            height: Math.max(1, Math.round(frac.h * image.height)),
          };
          // `drawOverlay` is deliberately NOT passed: the host's extras are
          // placed against the picture on screen, and drawing them onto a
          // different map would put them somewhere they do not belong.
          return renderToBlob({
            image,
            crop: c,
            unitsPerPx: built?.umPerPx ?? item.umPerPx ?? null,
            label: item.label,
          });
        },
        write: (blob, filename) => writer.write(blob, filename),
      });
    } catch (e) {
      setError(e?.message || String(e));
      setBusy(false);
      setBatchRun(null);
      return;
    }
    setBusy(false);
    setBatchRun(null);
    setBatchReport({ ...res, kind: writer.kind, target: writer.target });
  };

  if (!open) return null;

  const canExport = !!img && !!crop && !!output && !busy;
  const electronSave = typeof window !== 'undefined' && !!window.electronAPI?.saveImage;

  return createPortal(
    // The backdrop deliberately does NOT close the dialog: losing a crop, a
    // retyped caption and a dragged scale bar to a stray click beside the
    // window is forgivable exactly once. Close via the X, Cancel or Escape.
    <div
      data-image-export-backdrop
      // React portals still bubble SYNTHETIC events through the React tree, so
      // a click in here would reach the backdrop of whatever modal opened us
      // and close it. Only `click` is stopped: `mouseup` must keep reaching
      // window, or the crop drag never ends (it did once — the crop then
      // followed the pointer with no button held).
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'fixed', inset: 0, zIndex: 3500,
        background: 'rgba(0,0,0,.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 20,
      }}
    >
      <div
        data-image-export-dialog
        role="dialog"
        aria-modal="true"
        aria-label={t('imageexport:title', { what: title })}
        style={{
          background: colors.bg,
          border: `1px solid ${colors.border}`,
          borderRadius: 8,
          width: maximised ? '99vw' : 'min(1100px, 96vw)',
          height: maximised ? '97vh' : 'min(760px, 94vh)',
          display: 'flex', flexDirection: 'column',
          boxShadow: '0 24px 64px rgba(0,0,0,.5)',
          color: colors.text,
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 14px', borderBottom: `1px solid ${colors.border}`,
          flexShrink: 0,
        }}>
          <span style={{ color: colors.cyan, fontWeight: 600, fontSize: '11pt' }}>
            {t('imageexport:title', { what: title })}
          </span>
          {natural && (
            <span style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
              · {t('imageexport:sourceSize', { w: natural.width, h: natural.height })}
            </span>
          )}
          <div style={{ flex: 1 }} />
          {/* The shared Button does not forward unknown props, so the test
              hook lives on a wrapper rather than on the button itself. */}
          <span data-image-export-maximise>
            <Button
              small
              onClick={() => setMaximised((v) => !v)}
              title={maximised ? t('imageexport:restoreTooltip') : t('imageexport:maximiseTooltip')}
              aria-label={maximised ? t('imageexport:restoreTooltip') : t('imageexport:maximiseTooltip')}
            >
              {maximised ? '🗗' : '🗖'}
            </Button>
          </span>
          <Button small onClick={() => !busy && onClose?.()} title={t('imageexport:cancel')}>✕</Button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          {/* Preview */}
          <div style={{
            flex: 1, minWidth: 0, minHeight: 0,
            display: 'flex', flexDirection: 'column',
            background: colors.bgSecondary, padding: 10, gap: 6,
          }}>
            <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
              {t('imageexport:previewHint')}
            </div>
            <div
              ref={previewBoxRef}
              data-image-export-preview
              onMouseDown={(e) => onPointerDown(e, 'new')}
              style={{
                flex: 1, minHeight: 0, position: 'relative',
                background: '#000',
                border: `1px solid ${colors.border}`, borderRadius: 6,
                overflow: 'hidden', cursor: 'crosshair',
              }}
            >
              {loadError ? (
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                  color: colors.red, fontSize: '9pt', padding: 16, textAlign: 'center',
                }}>
                  {loadError}
                </div>
              ) : (
                <>
                  <canvas
                    ref={canvasRef}
                    data-image-export-canvas
                    style={{
                      position: 'absolute',
                      inset: 0,
                      imageRendering: smoothing ? 'auto' : 'pixelated',
                    }}
                  />
                  {previewRect && overlay && (
                    <div
                      data-image-export-overlay
                      style={{
                        position: 'absolute',
                        left: previewRect.left, top: previewRect.top,
                        width: previewRect.width, height: previewRect.height,
                        // The crop frame and its handles must stay reachable
                        // above this; the overlay's own widgets opt back in.
                        pointerEvents: 'none',
                      }}
                    >
                      {overlay(previewRect)}
                    </div>
                  )}
                  {previewRect && crop && natural && (
                    <div
                      data-image-export-croprect
                      onMouseDown={(e) => onPointerDown(e, 'move')}
                      style={{
                        position: 'absolute',
                        left: previewRect.left + crop.x * previewRect.scale,
                        top: previewRect.top + crop.y * previewRect.scale,
                        width: crop.width * previewRect.scale,
                        height: crop.height * previewRect.scale,
                        border: `1px solid ${colors.cyan}`,
                        boxShadow: '0 0 0 1px rgba(0,0,0,.6)',
                        cursor: 'move',
                      }}
                    >
                      {HANDLES.map((h) => {
                        const pos = {
                          left: h.includes('w') ? -4 : (h.includes('e') ? undefined : '50%'),
                          right: h.includes('e') ? -4 : undefined,
                          top: h.includes('n') ? -4 : (h.includes('s') ? undefined : '50%'),
                          bottom: h.includes('s') ? -4 : undefined,
                        };
                        return (
                          <div
                            key={h}
                            data-crop-handle={h}
                            onMouseDown={(e) => onPointerDown(e, h)}
                            style={{
                              position: 'absolute', width: 9, height: 9,
                              ...pos,
                              transform: `translate(${pos.left === '50%' ? '-50%' : '0'}, ${pos.top === '50%' ? '-50%' : '0'})`,
                              background: colors.cyan,
                              border: '1px solid rgba(0,0,0,.6)',
                              borderRadius: 2,
                              cursor: HANDLE_CURSOR[h],
                            }}
                          />
                        );
                      })}
                    </div>
                  )}

                  {/* Draggable / resizable annotation blocks, drawn from the
                      same layout the renderer used. */}
                  {previewRect && ['scalebar', 'caption'].map((kind) => {
                    const b = boxes[kind];
                    if (!b) return null;
                    // While the caption is being retyped its box would swallow
                    // the clicks meant for the text field underneath it.
                    if (kind === 'caption' && editingCaption) return null;
                    return (
                      <div
                        key={kind}
                        data-annot-box={kind}
                        onMouseDown={(e) => onPointerDown(e, 'move', kind)}
                        style={{
                          position: 'absolute',
                          left: b.ox + b.x,
                          top: b.oy + b.y,
                          width: b.w,
                          height: b.h,
                          border: `1px dashed ${colors.purple}`,
                          cursor: 'move',
                        }}
                      >
                        {HANDLES.map((h) => {
                          const pos = {
                            left: h.includes('w') ? -3 : (h.includes('e') ? undefined : '50%'),
                            right: h.includes('e') ? -3 : undefined,
                            top: h.includes('n') ? -3 : (h.includes('s') ? undefined : '50%'),
                            bottom: h.includes('s') ? -3 : undefined,
                          };
                          return (
                            <div
                              key={h}
                              data-annot-handle={`${kind}-${h}`}
                              onMouseDown={(e) => onPointerDown(e, h, kind)}
                              style={{
                                position: 'absolute', width: 7, height: 7,
                                ...pos,
                                transform: `translate(${pos.left === '50%' ? '-50%' : '0'}, ${pos.top === '50%' ? '-50%' : '0'})`,
                                background: colors.purple,
                                border: '1px solid rgba(0,0,0,.6)',
                                borderRadius: 2,
                                cursor: HANDLE_CURSOR[h],
                              }}
                            />
                          );
                        })}
                      </div>
                    );
                  })}

                  {/* Retype the caption where it sits, in the size and colour it
                      actually has. A textarea rather than an input so a wrapped
                      caption looks the same while it is being edited. */}
                  {previewRect && editingCaption && (
                    <textarea
                      data-caption-inline-editor
                      ref={inlineRef}
                      value={captionText}
                      autoFocus
                      spellCheck={false}
                      onFocus={(e) => e.target.select()}
                      onChange={(e) => setCaptionText(oneLine(e.target.value))}
                      onBlur={() => setEditingCaption(null)}
                      onKeyDown={(e) => {
                        e.stopPropagation();
                        if (e.key === 'Enter' || e.key === 'Escape') {
                          e.preventDefault();
                          setEditingCaption(null);
                        }
                      }}
                      style={{
                        position: 'absolute',
                        left: editingCaption.ox + editingCaption.x,
                        top: editingCaption.oy + editingCaption.y,
                        width: Math.max(editingCaption.w, 160),
                        height: Math.max(editingCaption.h, editingCaption.fontPx * 1.6),
                        font: `${editingCaption.bold ? 600 : 400} ${editingCaption.fontPx}px sans-serif`,
                        lineHeight: `${editingCaption.lineH}px`,
                        color: editingCaption.color,
                        padding: `${editingCaption.pad / 4}px ${editingCaption.pad / 2}px`,
                        // Exactly the caption's own backdrop — including none —
                        // so what you type is what the export will show. The
                        // dashed outline is the only editing-only decoration.
                        background: editingCaption.backgroundOpacity > 0
                          ? rgba(editingCaption.background, editingCaption.backgroundOpacity)
                          : 'transparent',
                        caretColor: editingCaption.color,
                        border: `1px dashed ${colors.purple}`,
                        borderRadius: 2,
                        boxSizing: 'border-box',
                        resize: 'none',
                        overflow: 'hidden',
                        outline: 'none',
                        zIndex: 5,
                      }}
                    />
                  )}
                </>
              )}
            </div>
          </div>

          {/* Settings */}
          <div style={{
            width: 330, flexShrink: 0, overflow: 'auto',
            padding: 12, borderLeft: `1px solid ${colors.border}`,
          }}>
            {sidePanel}
            <Field label={t('imageexport:format')}>
              <Select
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                options={FORMATS.map((f) => ({ value: f.id, label: f.id.toUpperCase() }))}
                style={{ width: '100%' }}
                title={t('imageexport:formatTooltip')}
              />
            </Field>

            {fmt.lossy && (
              <Field label={t('imageexport:quality', { value: quality })}>
                <input
                  type="range" min={10} max={100} value={quality}
                  onChange={(e) => setQuality(Number(e.target.value))}
                  style={{ width: '100%', accentColor: colors.purple }}
                  title={t('imageexport:qualityTooltip')}
                />
              </Field>
            )}

            <Field
              label={t('imageexport:resolution')}
              hint={output
                ? `${t('imageexport:outputSize', {
                  w: output.width, h: output.height, factor: output.factor.toFixed(2),
                })}${output.limited ? ` — ${t('imageexport:sizeLimited')}` : ''}`
                : ''}
            >
              <Select
                value={scaleMode.type === 'factor' ? String(scaleMode.factor) : 'custom'}
                onChange={(e) => {
                  const v = e.target.value;
                  setScaleMode((m) => (v === 'custom'
                    ? { ...m, type: 'width' }
                    : { ...m, type: 'factor', factor: Number(v) }));
                }}
                options={[
                  ...SCALE_FACTORS.map((f) => ({ value: String(f), label: `${f}×` })),
                  { value: 'custom', label: t('imageexport:customWidth') },
                ]}
                style={{ width: '100%', marginBottom: 5 }}
                title={t('imageexport:factorTooltip')}
              />
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <span style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
                  {t('imageexport:targetWidth')}
                </span>
                <NumberInput
                  value={scaleMode.targetWidth}
                  min={1}
                  max={20000}
                  onChange={(e) => setScaleMode((m) => ({
                    ...m, type: 'width', targetWidth: Number(e.target.value) || 1,
                  }))}
                  style={{ flex: 1 }}
                  title={t('imageexport:targetWidthTooltip')}
                />
              </div>
              <div style={{ marginTop: 6 }}>
                <Check
                  checked={!smoothing}
                  onChange={(v) => setSmoothing(!v)}
                  label={t('imageexport:pixelated')}
                  title={t('imageexport:pixelatedTooltip')}
                />
              </div>
            </Field>

            <Field
              label={t('imageexport:border')}
              // The same margins `doExport` uses — a body that grew after the
              // last fit is held by `widerMargins` at save time, and a readout
              // that ignored that announced a smaller file than it wrote.
              hint={output ? (() => {
                const tt = canvasSizeWithMargins(output, widerMargins(margins, fitMargins));
                if (tt.width === output.width && tt.height === output.height) {
                  return t('imageexport:borderHint');
                }
                return `${t('imageexport:borderTotal', { w: tt.width, h: tt.height })}`
                  + (tt.limited ? ` — ${t('imageexport:sizeLimited')}` : '');
              })() : ''}
            >
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                {['top', 'bottom', 'left', 'right'].map((side) => (
                  <div key={side} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ fontSize: '8pt', color: colors.textSecondary, width: 34 }}>
                      {t(`imageexport:borderSide.${side}`)}
                    </span>
                    <NumberInput
                      value={Math.round((margins[side] ?? 0) * 100)}
                      min={0}
                      // The field offers half the image; a border FITTED to a
                      // colour key placed beside the map is often wider than
                      // that, and the field must be able to show and hold the
                      // value it was given instead of snapping it back down.
                      max={Math.round(Math.max(
                        BORDER_INPUT_MAX_FRACTION, margins[side] ?? 0,
                      ) * 100)}
                      step={5}
                      onChange={(e) => setMargins((m) => ({
                        ...m,
                        [side]: Math.max(0, (Number(e.target.value) || 0) / 100),
                      }))}
                      style={{ flex: 1, minWidth: 0 }}
                      title={t('imageexport:borderTooltip')}
                    />
                    <span style={{ fontSize: '8pt', color: colors.textSecondary }}>%</span>
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
                <ColorBox value={marginColor} onChange={setMarginColor} title={t('imageexport:borderColorTooltip')} />
                <Button
                  small
                  onClick={() => setMargins(NO_MARGINS)}
                  disabled={!Object.values(margins).some((v) => v > 0)}
                  style={{ flex: 1 }}
                  title={t('imageexport:borderResetTooltip')}
                >
                  {t('imageexport:borderReset')}
                </Button>
              </div>
            </Field>

            <Field label={t('imageexport:crop')}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                {['x', 'y', 'width', 'height'].map((k) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ fontSize: '8pt', color: colors.textSecondary, width: 16 }}>
                      {t(`imageexport:cropField.${k}`)}
                    </span>
                    <NumberInput
                      value={crop?.[k] ?? 0}
                      min={0}
                      onChange={(e) => natural && setCrop((c) => clampCrop(
                        { ...c, [k]: Number(e.target.value) || 0 }, natural,
                      ))}
                      style={{ flex: 1, minWidth: 0 }}
                    />
                  </div>
                ))}
              </div>
              <Button
                small
                onClick={() => natural && setCrop(fullCrop(natural))}
                disabled={!natural || (crop && natural && isFullCrop(crop, natural))}
                style={{ width: '100%', marginTop: 6 }}
                title={t('imageexport:resetCropTooltip')}
              >
                {t('imageexport:resetCrop')}
              </Button>
            </Field>

            <Field label={t('imageexport:overlays')}>
              <Check
                checked={scalebarBinding ? scalebarBinding.present : showScalebar}
                onChange={scalebarBinding ? scalebarBinding.onToggle : setShowScalebar}
                disabled={!hasScale}
                label={t('imageexport:scalebar')}
                title={hasScale ? t('imageexport:scalebarTooltip') : t('imageexport:scalebarUnavailable')}
              />
              {scalebarBinding?.present && (
                // The very same bar the map shows — edited here, drawn there.
                <BoundScalebarEditor binding={scalebarBinding} unitLabel={unitLabel} t={t} />
              )}
              {showScalebar && hasScale && !scalebarBinding && (
                <StyleEditor
                  style={scalebarStyle}
                  onChange={setScalebarStyle}
                  t={t}
                  showBarScale
                >
                  <Row label={t('imageexport:style.length')}>
                    <Select
                      value={scalebarStyle.lengthUnits == null ? 'auto' : String(scalebarStyle.lengthUnits)}
                      onChange={(e) => setScalebarStyle((s) => ({
                        ...s,
                        lengthUnits: e.target.value === 'auto' ? null : Number(e.target.value),
                      }))}
                      options={[
                        { value: 'auto', label: t('imageexport:style.lengthAuto') },
                        ...barLengthChoices.map((v) => ({
                          value: String(v), label: `${formatUnits(v)} ${unitLabel}`,
                        })),
                      ]}
                      style={{ flex: 1, minWidth: 0 }}
                      title={t('imageexport:style.lengthTooltip')}
                    />
                  </Row>
                </StyleEditor>
              )}

              <Check
                checked={showCrosshair} onChange={setShowCrosshair} disabled={!hasCrosshair}
                label={t('imageexport:crosshair')}
                title={hasCrosshair ? t('imageexport:crosshairTooltip') : t('imageexport:overlayUnavailable')}
              />
              <Check
                checked={showRoi} onChange={setShowRoi} disabled={!hasRoi}
                label={t('imageexport:roi')}
                title={hasRoi ? t('imageexport:roiTooltip') : t('imageexport:roiUnavailable')}
              />

              <Check
                checked={showLabel} onChange={setShowLabel} disabled={!hasLabel}
                label={t('imageexport:caption')}
                title={hasLabel ? t('imageexport:captionTooltip') : t('imageexport:overlayUnavailable')}
              />
              {showLabel && hasLabel && (
                <StyleEditor style={captionStyle} onChange={setCaptionStyle} t={t}>
                  <Row label={t('imageexport:style.text')}>
                    <SelectAllInput
                      value={captionText}
                      onChange={(e) => setCaptionText(e.target.value)}
                      style={{ flex: 1, minWidth: 0, height: 24, fontSize: '9pt' }}
                      title={t('imageexport:style.textTooltip')}
                    />
                  </Row>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 5 }}>
                    <Button
                      small
                      onClick={() => setCaptionText(annotations?.label || '')}
                      disabled={captionText === (annotations?.label || '')}
                      title={t('imageexport:style.resetTextTooltip')}
                    >
                      {t('imageexport:style.resetText')}
                    </Button>
                  </div>
                </StyleEditor>
              )}
            </Field>

            <Field
              label={t('imageexport:values')}
              hint={displayFilter
                ? (rawValues ? t('imageexport:valuesRawActive') : t('imageexport:valuesDisplayActive'))
                : t('imageexport:valuesNoCorrection')}
            >
              <Check
                checked={rawValues}
                onChange={setRawValues}
                disabled={!displayFilter}
                label={t('imageexport:raw')}
                title={t('imageexport:rawTooltip')}
              />
            </Field>

            <Field
              label={t('imageexport:filename')}
              hint={electronSave ? t('imageexport:willAskLocation') : t('imageexport:willDownload')}
            >
              <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                <Input
                  value={baseName}
                  onChange={(e) => setBaseName(e.target.value)}
                  style={{ flex: 1, minWidth: 0 }}
                  title={t('imageexport:filenameTooltip')}
                />
                <span style={{ fontSize: '8.5pt', color: colors.textSecondary }}>.{fmt.ext}</span>
              </div>
            </Field>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '10px 14px', borderTop: `1px solid ${colors.border}`,
          flexShrink: 0,
        }}>
          {error && (
            <span data-image-export-error style={{ color: colors.red, fontSize: '8.5pt' }}>{error}</span>
          )}
          {batchRun && (
            <span data-image-export-batch-progress
              style={{ color: colors.textSecondary, fontSize: '8.5pt' }}>
              {t('imageexport:batchProgress', {
                n: batchRun.index + 1, total: batchRun.total, name: batchRun.filename,
              })}
            </span>
          )}
          {batchReport && (
            <span data-image-export-batch-report
              style={{
                color: batchReport.failed.length ? colors.red : colors.textSecondary,
                fontSize: '8.5pt',
              }}
              title={batchReport.failed.map((f) => `${f.filename}: ${f.message}`).join('\n') || undefined}
            >
              {t('imageexport:batchDone', {
                written: batchReport.written.length,
                total: batchReport.total,
              })}
              {batchReport.failed.length > 0
                && ` — ${t('imageexport:batchFailed', { n: batchReport.failed.length })}`}
              {batchReport.cancelled && ` — ${t('imageexport:batchCancelled')}`}
            </span>
          )}
          <div style={{ flex: 1 }} />
          {batchCount > 1 && (
            batchRun ? (
              <Button
                onClick={() => { batchCancelRef.current = true; }}
                title={t('imageexport:batchStopTooltip')}
              >
                {t('imageexport:batchStop')}
              </Button>
            ) : (
              <Button
                data-image-export-batch
                onClick={doBatchExport}
                disabled={!canExport}
                title={t('imageexport:batchTooltip', { n: batchCount })}
              >
                {t('imageexport:batchExport', { n: batchCount })}
              </Button>
            )
          )}
          <Button onClick={() => !busy && onClose?.()} disabled={busy} title={t('imageexport:cancelTooltip')}>
            {t('imageexport:cancel')}
          </Button>
          <Button
            variant="primary"
            onClick={doExport}
            disabled={!canExport}
            title={t('imageexport:exportTooltip')}
          >
            {busy ? t('imageexport:exporting') : t('imageexport:export')}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
