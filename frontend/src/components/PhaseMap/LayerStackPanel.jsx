/**
 * LayerStackPanel - controls the active layer stack: visibility, opacity,
 * blend, drag-to-reorder, remove. Lives at the top of the right settings
 * panel in PhaseMapPage.
 *
 * Props:
 *   layers           - array from useLayerStack
 *   bitmaps          - Map (for ready-state visualisation; not edited here)
 *   errors           - Map<id, string> for per-layer errors
 *   onSetOpacity     - (id, value)
 *   onSetBlend       - (id, value)
 *   onSetVisibility  - (id, bool)
 *   onRemove         - (id)
 *   onReorder        - (fromIdx, toIdx)
 *   onAdd            - (id)
 *   onUsePreset      - (name)
 *   availableToAdd   - array of {value, label, disabled, tip?}
 *
 * Note on the Select API used here:
 *   The shared <Select> from theme/components is a thin wrapper around a
 *   native <select>. It takes onChange={(e) => ...} (event-based, like a
 *   real <select>) and has no `placeholder`/`small` props. Placeholder
 *   behaviour is therefore emulated by injecting a disabled sentinel
 *   option with empty value as the first entry, and keeping `value=""` on
 *   the control so the sentinel is shown until the user picks an option.
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing, CollapsibleGroup, Select } from '../../theme/components';
import { PRESETS } from './presets';
import { MAX_LAYERS } from './layerStackReducer';
import { findLayerDef } from './layerSources';
import useDataStore from '../../stores/useDataStore';
import useResultStore from '../../stores/useResultStore';

function LayerRow({
  layer, index, bitmapReady, error,
  onSetOpacity, onSetBlend, onSetVisibility, onRemove, onReorder,
  onRetype, typeOptions = [],
  renderLayerExtras,
}) {
  const { t } = useTranslation('phasemap');
  const BLEND_OPTIONS = [
    { value: 'normal',   label: t('phasemap:layers.blendNormal') },
    { value: 'multiply', label: t('phasemap:layers.blendMultiply') },
    { value: 'screen',   label: t('phasemap:layers.blendScreen') },
    { value: 'overlay',  label: t('phasemap:layers.blendOverlay') },
  ];
  const [dragging, setDragging] = useState(false);
  const pct = Math.round(layer.opacity * 100);

  // Gating for derived layers: layers flagged `requiresCompute` need forward
  // diagnostics computed for the active result; layers flagged
  // `requiresRefinement` need joint R+PC refinement computed. Until then they
  // render visually disabled with a locked visibility toggle (user can still
  // remove). Phase A pattern, extended for Phase B refinement.
  const def = findLayerDef(layer.id);
  const diagnosticsComputed = useDataStore((s) => s.diagnosticsComputed);
  const refinementComputed = useDataStore((s) => s.refinementComputed);
  const activeResultId = useResultStore((s) => s.indexingResult?.result_id ?? null);
  const isGatedDiag = !!def?.requiresCompute
    && !(activeResultId && diagnosticsComputed?.[activeResultId]);
  const isGatedRef = !!def?.requiresRefinement
    && !(activeResultId && refinementComputed?.[activeResultId]);
  const isGated = isGatedDiag || isGatedRef;
  const gateHint = isGatedRef
    ? t('phasemap:layers.gateRefHint')
    : t('phasemap:layers.gateDiagHint');
  const gateSuffix = isGatedRef ? t('phasemap:layers.gateRefSuffix') : t('phasemap:layers.gateDiagSuffix');

  // The legend matters for layers whose colours encode a meaning the user
  // cannot infer (the assignment-source provenance map), so it rides along in
  // the row tooltip — behind an error, which is the more urgent message.
  const label = layer.label;
  const legend = def?.tipKey ? t(def.tipKey) : null;

  const baseTitle = error
    ? `${label}: ${error}`
    : (bitmapReady
        ? (legend ? `${label} — ${legend}` : label)
        : t('phasemap:layers.loadingSuffix', { label }));
  const rowTitle = isGated
    ? `${label} — ${gateHint}`
    : baseTitle;
  const rowOpacity = isGated ? 0.4 : (layer.visible ? 1 : 0.55);
  const rowBorder = isGated
    ? `1px dashed ${colors.border}`
    : `1px solid ${error ? colors.red : colors.border}`;

  const extras = renderLayerExtras?.(layer);
  return (
    <div style={{ marginBottom: 2 }}>
      {/* Two lines on purpose. On one line the panel is ~285px wide and the
          fixed columns (handle, box, slider, %, blend, ×) ate all of it: the
          name column resolved to ONE pixel, so no layer showed its name. The
          slider gets its own line and a real width instead of 80px. */}
      <div
        onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }}
        onDrop={(e) => {
          e.preventDefault();
          const from = parseInt(e.dataTransfer.getData('text/plain'), 10);
          if (!Number.isNaN(from) && from !== index) onReorder(from, index);
        }}
        title={rowTitle}
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 3,
          padding: '4px 6px',
          background: dragging ? colors.bgSecondary : colors.bgTertiary,
          border: rowBorder,
          borderRadius: 3,
          opacity: rowOpacity,
        }}
      >
      <div style={{
        display: 'grid',
        gridTemplateColumns: '14px 16px minmax(0, 1fr) 70px 18px',
        alignItems: 'center',
        columnGap: 6,
      }}>
        {/* ONLY the handle starts a drag. With `draggable` on the whole row the
            browser began an HTML5 drag as soon as the pointer went down on the
            slider, so the slider emitted no input at all — measured: 0 input
            events, value unchanged, one dragstart. That is the "sliders don't
            work" report, and it was literally true. */}
        <span
          draggable
          onDragStart={(e) => {
            setDragging(true);
            e.dataTransfer.setData('text/plain', String(index));
            e.dataTransfer.effectAllowed = 'move';
          }}
          onDragEnd={() => setDragging(false)}
          style={{ cursor: 'grab', color: colors.textSecondary, fontSize: 11, lineHeight: 1, textAlign: 'center', userSelect: 'none' }}
          title={t('phasemap:layers.dragToReorder')}
        >&#9776;</span>
        <input
          type="checkbox"
          checked={layer.visible}
          onChange={(e) => onSetVisibility(layer.id, e.target.checked)}
          disabled={isGated}
          aria-label={t('phasemap:layers.toggleVisibilityAria', { label })}
          title={isGated ? gateHint : undefined}
          style={{ margin: 0 }}
        />
        {onRetype ? (
          // The name doubles as a type picker: opacity, blend, visibility and
          // stack position are properties of the SLOT, not of the map in it,
          // so trying the same arrangement on IPF-X should not mean building
          // it again from scratch.
          <Select
            value={layer.id}
            onChange={(e) => onRetype(layer.id, e.target.value)}
            options={typeOptions}
            style={{
              width: '100%', fontSize: '8.5pt', height: 22, padding: '0 4px',
              fontStyle: (isGated || !bitmapReady) ? 'italic' : 'normal',
              color: error ? colors.red : undefined,
            }}
            title={t('phasemap:hoverTips.changeLayerType')}
          />
        ) : (
          <span
            title={isGated ? gateHint : undefined}
            style={{
              fontSize: '9pt', color: error ? colors.red : (bitmapReady ? colors.text : colors.textSecondary),
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              fontStyle: (isGated || !bitmapReady) ? 'italic' : 'normal',
            }}
          >
            {label}{isGated ? gateSuffix : ''}
          </span>
        )}
        <Select
          value={layer.blend}
          onChange={(e) => onSetBlend(layer.id, e.target.value)}
          options={BLEND_OPTIONS}
          style={{ width: '100%', fontSize: '8.5pt', height: 22, padding: '0 4px' }}
          title={t('phasemap:hoverTips.blendMode')}
        />
        <button
          onClick={() => onRemove(layer.id)}
          aria-label={t('phasemap:layers.removeAria', { label })}
          title={t('phasemap:layers.removeTooltip')}
          style={{
            background: 'transparent', border: 'none',
            color: colors.red, cursor: 'pointer',
            padding: 0, fontSize: 14, lineHeight: 1,
          }}
        >&times;</button>
      </div>

      {/* Line 2: the slider, with the room it needs. */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr) 40px',
        alignItems: 'center',
        columnGap: 6,
      }}>
        <input
          type="range"
          min={0} max={100} value={pct}
          disabled={isGated}
          onChange={(e) => onSetOpacity(layer.id, Number(e.target.value) / 100)}
          onDoubleClick={() => onSetOpacity(layer.id, 1)}
          style={{ width: '100%', margin: 0 }}
          aria-label={t('phasemap:layers.opacityAria', { label })}
          title={t('phasemap:hoverTips.opacitySlider')}
        />
        <span style={{
          fontSize: '8pt', color: colors.textSecondary, fontFamily: 'monospace', textAlign: 'right',
        }}>{pct}%</span>
      </div>
      </div>
      {extras}
    </div>
  );
}

export default function LayerStackPanel({
  layers, bitmaps, errors,
  onSetOpacity, onSetBlend, onSetVisibility,
  onRemove, onReorder, onAdd, onUsePreset,
  onSetSingleLayer,
  onRetype,
  activeMode = null,
  availableToAdd = [],
  renderLayerExtras,
}) {
  const { t } = useTranslation('phasemap');
  const atLimit = layers.length >= MAX_LAYERS;

  // Quick single-layer modes — replace the old Display Mode dropdown.
  // One click here resets the stack to a single layer at full opacity,
  // giving back the "pick a view" UX while the layer panel below stays
  // available for stacking on top.
  const QUICK_MODES = [
    { id: 'phase', label: t('phasemap:layers.quickPhase') },
    { id: 'ipf-z', label: t('phasemap:layers.quickIpfZ') },
    { id: 'ipf-x', label: t('phasemap:layers.quickIpfX') },
    { id: 'ipf-y', label: t('phasemap:layers.quickIpfY') },
    { id: 'bc',    label: t('phasemap:layers.quickBc') },
    { id: 'ci',    label: t('phasemap:layers.quickCi') },
  ];

  // Sentinel-based placeholder: first option has empty value and is the
  // label the user sees while no selection is made. We keep value="" so
  // the sentinel stays visible after each pick (one-shot dropdown).
  const addOptions = [
    { value: '', label: atLimit ? t('phasemap:layers.limit', { max: MAX_LAYERS }) : t('phasemap:layers.addLayer') },
    ...availableToAdd,
  ];
  const presetOptions = [
    { value: '', label: t('phasemap:layers.preset') },
    ...Object.entries(PRESETS).map(([name, p]) => ({ value: name, label: p.label })),
  ];

  // What a layer may become: the same catalogue "+ Add Layer" offers, plus
  // its own type so the picker can show what it currently is. Types already in
  // the stack stay out — two layers with one id would share a bitmap slot.
  const typeOptionsFor = (layer) => {
    const own = { value: layer.id, label: layer.label };
    const others = availableToAdd.filter((o) => o.value && o.value !== layer.id);
    return [own, ...others];
  };

  // Display top-down (UI top = topmost layer = last in layers array).
  const displayLayers = [...layers].reverse();

  // Which mode button reads as selected. Taken from the stack itself only as
  // a fallback: once a mode holds more than one layer, "the stack is a single
  // layer called X" stops being true while the user is still inside mode X.
  const activeQuickMode = activeMode ?? (layers.length === 1 ? layers[0].id : null);

  return (
    <CollapsibleGroup title={t('phasemap:layers.title')} defaultCollapsed={false}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
        {/* Quick single-layer modes — replaces the old Display Mode dropdown */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 3,
        }}>
          {QUICK_MODES.map((m) => {
            const active = activeQuickMode === m.id;
            return (
              <button
                key={m.id}
                onClick={() => onSetSingleLayer?.(m.id)}
                title={t('phasemap:layers.quickModeTooltip', { label: m.label })}
                style={{
                  fontSize: '9pt',
                  padding: '4px 6px',
                  background: active ? colors.accent : colors.bgTertiary,
                  color: active ? '#000' : colors.text,
                  border: `1px solid ${active ? colors.accent : colors.border}`,
                  borderRadius: 3,
                  cursor: 'pointer',
                  fontWeight: 600,
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  minHeight: 26,
                }}
              >
                {m.label}
              </button>
            );
          })}
        </div>

        {layers.length === 0 && (
          <div style={{ color: colors.textSecondary, fontSize: '9pt', padding: 8, textAlign: 'center' }}>
            {t('phasemap:layers.empty')}
          </div>
        )}

        <div>
          {displayLayers.map((layer, displayIdx) => {
            const realIdx = layers.length - 1 - displayIdx;
            return (
              <LayerRow
                key={layer.key ?? layer.id}
                layer={layer}
                index={realIdx}
                bitmapReady={bitmaps?.has(layer.id) ?? false}
                error={errors?.get(layer.id) ?? null}
                onSetOpacity={onSetOpacity}
                onSetBlend={onSetBlend}
                onSetVisibility={onSetVisibility}
                onRemove={onRemove}
                onRetype={onRetype}
                typeOptions={typeOptionsFor(layer)}
                onReorder={onReorder}
                renderLayerExtras={renderLayerExtras}
              />
            );
          })}
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          <Select
            value=""
            onChange={(e) => { const v = e.target.value; if (v) onAdd(v); }}
            options={addOptions}
            disabled={atLimit}
            style={{ flex: 1, minWidth: 0, fontSize: '9pt', height: 26, padding: '2px 6px' }}
            title={t('phasemap:hoverTips.addLayer')}
          />
          <Select
            value=""
            onChange={(e) => { const v = e.target.value; if (v) onUsePreset(v); }}
            options={presetOptions}
            style={{ flex: 1, minWidth: 0, fontSize: '9pt', height: 26, padding: '2px 6px' }}
            title={t('phasemap:hoverTips.presetSelect')}
          />
        </div>
        {(() => {
          const hint = availableToAdd.find((o) => o.disabled && o.tip)?.tip;
          if (!hint) return null;
          return (
            <div style={{
              color: colors.textSecondary,
              fontSize: '8pt',
              padding: '4px 2px',
              fontStyle: 'italic',
              lineHeight: 1.3,
            }}>
              {hint}
            </div>
          );
        })()}
      </div>
    </CollapsibleGroup>
  );
}
