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

/** Blend mode keys → Canvas2D globalCompositeOperation values. */
export const BLEND_MAP = {
  normal: 'source-over',
  multiply: 'multiply',
  screen: 'screen',
  overlay: 'overlay',
};
