/**
 * Static catalog of layer sources available in PhaseMapPage.
 *
 * - `result` layers come from /api/phasemap/layer (new endpoint)
 * - `analysis` layers come from /api/analysis/map/{type} (when xmap loaded)
 * - `ebsd` layers come from /api/ebsd/* (when EBSD signal loaded)
 * - `h5` layers come from /api/h5/* (when source H5OINA linked)
 *
 * Dynamic discovery (per-phase CI, EDS elements, electron images) is added
 * at runtime by useLayerStack — these static entries are the always-present
 * ones.
 */
import { defaultBands as defaultGrainBoundaryBands } from './grainBoundaryBands';

const GB_DEFAULT_BANDS = defaultGrainBoundaryBands();

export const LAYER_SOURCES = {
  result: {
    label: 'Indexing Result',
    layers: [
      { id: 'phase',       label: 'Phase Map',     kind: 'phase',       defaultBlend: 'normal',   defaultOpacity: 1.0 },
      { id: 'ipf-z',       label: 'IPF-Z [001]',   kind: 'ipf-z',       defaultBlend: 'normal',   defaultOpacity: 1.0 },
      { id: 'ipf-x',       label: 'IPF-X [100]',   kind: 'ipf-x',       defaultBlend: 'normal',   defaultOpacity: 1.0 },
      { id: 'ipf-y',       label: 'IPF-Y [010]',   kind: 'ipf-y',       defaultBlend: 'normal',   defaultOpacity: 1.0 },
      { id: 'bc',          label: 'Band Contrast', kind: 'bc',          defaultBlend: 'multiply', defaultOpacity: 0.6 },
      { id: 'ci',          label: 'CI (Best)',     kind: 'ci',          defaultBlend: 'normal',   defaultOpacity: 0.5 },
      { id: 'uncertainty', label: 'Uncertainty',   kind: 'uncertainty', defaultBlend: 'normal',   defaultOpacity: 0.5 },
      // Misindex-diagnose helper (2026-05-26): pixels with CI outside the
      // configured band are painted in `out_color` (default red @ 60%
      // alpha). Default band [0, 0.3] flags low-confidence pixels.
      { id: 'ci-threshold',label: 'CI Threshold',  kind: 'ci-threshold',defaultBlend: 'normal',   defaultOpacity: 0.6,
        params: { band_min: 0.0, band_max: 0.3, out_color: 'ff3333', out_alpha: 153 } },
      // Grain boundaries drawn ON the interfaces between pixels, classified by
      // misorientation angle. The three classes travel as one JSON param so the
      // backend gets them atomically — half-applied bands would paint a map
      // that matches no setting the user ever chose.
      { id: 'grain-boundaries', label: 'Grain Boundaries', kind: 'grain-boundaries',
        defaultBlend: 'normal', defaultOpacity: 1.0,
        params: { gb_bands: JSON.stringify(GB_DEFAULT_BANDS) } },
    ],
  },
  analysis: {
    label: 'Analysis (xmap)',
    layers: [
      { id: 'kam', label: 'KAM',   defaultBlend: 'normal', defaultOpacity: 0.6 },
      { id: 'gos', label: 'GOS',   defaultBlend: 'normal', defaultOpacity: 0.6 },
    ],
  },
  ebsd: {
    label: 'EBSD',
    layers: [
      { id: 'vbse', label: 'Virtual BSE', defaultBlend: 'normal', defaultOpacity: 1.0 },
    ],
  },
  diagnostics: {
    label: 'Forward Diagnostics',
    layers: [
      { id: 'forward-ncc',      label: 'Forward NCC',      kind: 'forward_ncc',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresCompute: true },
      { id: 'local-anomaly',    label: 'NCC Anomaly',      kind: 'local_anomaly',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresCompute: true },
      { id: 'pc-sensitivity',   label: 'PC Sensitivity',   kind: 'pc_sensitivity',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresCompute: true },
      { id: 'pattern-residual', label: 'Pattern Residual', kind: 'pattern_residual',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresCompute: true },
      // Render-verified phase check (Stage A): per-grain margin of the
      // stored phase vs the best alternative phase. Computed by the
      // "Check phases" button (POST /api/indexing/phase-check); 404s until
      // then (per-layer failure isolation shows the error chip).
      { id: 'phase-margin',     label: 'Phase Check',      kind: 'phase_margin',
        defaultBlend: 'normal', defaultOpacity: 0.85 },
    ],
  },
  refinement: {
    label: 'R+PC Refinement',
    layers: [
      { id: 'refined-ncc',         label: 'Refined NCC',     kind: 'refined_ncc',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresRefinement: true },
      { id: 'convergence-status',  label: 'Convergence',     kind: 'convergence_status',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresRefinement: true },
      { id: 'orientation-delta',   label: 'Orientation Δ',   kind: 'orientation_delta',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresRefinement: true },
      { id: 'pc-delta-x',          label: 'PC Δx (px)',      kind: 'pc_delta_x',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresRefinement: true },
      { id: 'pc-delta-y',          label: 'PC Δy (px)',      kind: 'pc_delta_y',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresRefinement: true },
      { id: 'pc-delta-l',          label: 'PC ΔL (µm)',      kind: 'pc_delta_l',
        defaultBlend: 'normal', defaultOpacity: 0.7, requiresRefinement: true },
    ],
  },
  h5: {
    label: 'H5OINA Source',
    layers: [],   // populated dynamically (SE images + EDS elements)
  },
};

/** Look up a layer definition by id across all source groups.
 *  Returns null if not found. Result includes the source key for routing. */
export function findLayerDef(id) {
  for (const [source, group] of Object.entries(LAYER_SOURCES)) {
    const def = group.layers.find((l) => l.id === id);
    if (def) return { ...def, source };
  }
  return null;
}

/** The name to show for a layer definition.
 *
 * Catalog names are English literals — the convention every layer here has
 * always followed. A definition opts into translation by declaring a
 * `labelKey`; only then is `t` consulted. Definitions without one come back
 * byte-identical, so opting one layer in changes nothing for the others.
 *
 * `fallback` covers the dynamically discovered layers (per-phase CI, EDS
 * elements, electron images), which have no catalog entry at all.
 */
export function layerLabel(def, t, fallback = '') {
  if (def?.labelKey && typeof t === 'function') return t(def.labelKey);
  return def?.label ?? fallback;
}

/** Options for the "+ Add Layer" dropdown: every catalog layer plus every
 *  dynamically discovered one, minus what is already in the stack.
 *
 *  Lives here rather than inline in PhaseMapPage so the captions can be
 *  tested — a full page mount is not done in this repo, and an untested
 *  caption silently reverts to the untranslated literal.
 *
 *  Each option says whether it can actually be drawn right now. A layer that
 *  needs data nobody has loaded would only produce an error chip in the stack,
 *  so the picker hides those by default and names the missing piece in `tip`.
 *
 *  @param {object}   args
 *  @param {Array}    args.layers        layers currently in the stack
 *  @param {Array}    args.dynamicLayers runtime-discovered defs (per-phase
 *                                       CI, EDS elements, electron images)
 *  @param {boolean}  args.sourceLinked  is an H5OINA source linked?
 *  @param {?string}  args.sourceReason  why it is not, shown as the tip
 *  @param {Function} [args.t]           translator, for labelKey layers
 *  @param {object}   [args.availability] per-gate readiness, each entry
 *         `{ ok: boolean, reason?: string }`. Keys: the LAYER_SOURCES group
 *         names (`result`, `analysis`, `ebsd`, `diagnostics`, `refinement`,
 *         `h5`) plus `diagnosticsComputed`, `refinementComputed` and
 *         `assignmentSource`. A missing key means "no claim" — available.
 *  @returns {Array<{value, label, group, name, disabled, tip}>}
 */
export function buildAddLayerOptions({
  layers = [], dynamicLayers = [], sourceLinked = false,
  sourceReason = null, t, availability = null,
}) {
  const av = { ...(availability || {}) };
  // The H5OINA gate has always come in as two loose arguments; fold it in so
  // there is one place that decides, and let an explicit entry win.
  if (av.h5 === undefined) av.h5 = { ok: !!sourceLinked, reason: sourceReason };

  /** The first unmet requirement of a layer, or null when it can be drawn. */
  const gateFor = (layer, groupKey) => {
    const checks = [av[groupKey]];
    if (layer?.requiresCompute) checks.push(av.diagnosticsComputed);
    if (layer?.requiresRefinement) checks.push(av.refinementComputed);
    if (layer?.id === 'assignment-source') checks.push(av.assignmentSource);
    for (const c of checks) {
      if (c && c.ok === false) return c;
    }
    return null;
  };

  const opts = [];
  const used = new Set(layers.map((l) => l.id));
  for (const [groupKey, group] of Object.entries(LAYER_SOURCES)) {
    for (const layer of group.layers) {
      if (used.has(layer.id)) continue;
      const gate = gateFor(layer, groupKey);
      const name = layerLabel(layer, t);
      opts.push({
        value: layer.id,
        label: `${group.label}: ${name}`,
        group: group.label,
        name,
        disabled: !!gate,
        tip: gate ? (gate.reason ?? null) : null,
      });
    }
  }
  for (const d of dynamicLayers) {
    if (used.has(d.id)) continue;
    // A discovered layer belongs to the group it was discovered under, and
    // shares its gate: EDS elements read from the linked H5OINA like the rest.
    const gate = gateFor(d, d.source);
    opts.push({
      value: d.id,
      label: `${d.groupLabel}: ${d.label}`,
      group: d.groupLabel,
      name: d.label,
      disabled: !!gate,
      tip: gate ? (gate.reason ?? null) : null,
    });
  }
  return opts;
}

/** Blend mode keys → Canvas2D globalCompositeOperation values. */
export const BLEND_MAP = {
  normal: 'source-over',
  multiply: 'multiply',
  screen: 'screen',
  overlay: 'overlay',
};
