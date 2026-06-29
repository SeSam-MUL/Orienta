// frontend/src/components/common/coordinateSystemPresets.js
// Mirrors backend reference_frame_state.PRESETS (ids only — the actual rotations
// are resolved server-side; the frontend just sends the chosen preset id).
export const PRESETS = [
  { id: 'identity',     label: 'Identity (native)' },
  { id: 'mtex_default', label: 'MTEX default' },
  { id: 'aztec_oxford', label: 'Aztec / Oxford (Rz +90°)' },
  { id: 'edax',         label: 'EDAX / TSL' },
];

export const X_DIRECTIONS = [
  { id: 'east', label: 'X → East (right)' },
  { id: 'north', label: 'X → North (up)' },
];
export const PROJECTIONS = [
  { id: 'equal_area', label: 'Equal-area (Schmidt)' },
  { id: 'stereographic', label: 'Stereographic (Wulff)' },
];
export const HEMISPHERES = [
  { id: 'upper', label: 'Upper' },
  { id: 'lower', label: 'Lower' },
];
