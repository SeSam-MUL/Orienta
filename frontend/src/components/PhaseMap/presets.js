/**
 * Built-in layer-stack presets.
 *
 * Each preset's `layers` array is bottom-up: layers[0] is drawn first
 * (base layer), subsequent entries composited on top. The LayerStackPanel
 * displays this top-down, so the UI order is reversed for display.
 */
import { findLayerDef } from './layerSources';

export const PRESETS = {
  phase_default: {
    label: 'Phase Map',
    layers: [{ id: 'phase', opacity: 1.0, blend: 'normal' }],
  },
  aztec_style: {
    label: 'Aztec-Style (IPF + BC)',
    layers: [
      { id: 'ipf-z', opacity: 1.0, blend: 'normal' },
      { id: 'bc',    opacity: 0.6, blend: 'multiply' },
    ],
  },
  quality_check: {
    label: 'Phase + CI',
    layers: [
      { id: 'phase', opacity: 1.0, blend: 'normal' },
      { id: 'ci',    opacity: 0.5, blend: 'normal' },
    ],
  },
  eds_verify: {
    label: 'EDS Phase Verify (add element manually)',
    layers: [
      { id: 'ipf-z', opacity: 1.0, blend: 'normal' },
      { id: 'bc',    opacity: 0.5, blend: 'multiply' },
    ],
  },
};

/** Materialise a preset into a full layer-stack state array. */
export function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) throw new Error(`Unknown preset: ${name}`);
  return preset.layers.map((entry, idx) => {
    const def = findLayerDef(entry.id);
    return {
      id: entry.id,
      label: def?.label ?? entry.id,
      source: def?.source ?? 'result',
      opacity: entry.opacity,
      blend: entry.blend,
      visible: true,
      key: `${entry.id}-${idx}-${Date.now()}`,
    };
  });
}
