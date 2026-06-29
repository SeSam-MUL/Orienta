/**
 * Pure reducer for the PhaseMap layer stack.
 *
 * State shape: { layers: Layer[] }
 *
 * Layer shape: { id, label, source, opacity, blend, visible, key }
 *
 * Actions: ADD, REMOVE, SET_OPACITY, SET_BLEND, SET_VISIBILITY,
 *          REORDER, REPLACE_ALL, CLEAR.
 *
 * Bitmap cache and fetch lifecycle live in useLayerStack.js — this
 * module is purely state transitions.
 */

export const MAX_LAYERS = 8;

export const initialState = {
  layers: [],
};

function clamp01(v) {
  if (Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

export function layerStackReducer(state, action) {
  switch (action.type) {
    case 'ADD': {
      if (state.layers.length >= MAX_LAYERS) return state;
      if (state.layers.some((l) => l.id === action.layer.id)) return state;
      return { ...state, layers: [...state.layers, action.layer] };
    }
    case 'REMOVE': {
      return { ...state, layers: state.layers.filter((l) => l.id !== action.id) };
    }
    case 'SET_OPACITY': {
      const value = clamp01(action.value);
      return {
        ...state,
        layers: state.layers.map((l) => (l.id === action.id ? { ...l, opacity: value } : l)),
      };
    }
    case 'SET_BLEND': {
      return {
        ...state,
        layers: state.layers.map((l) => (l.id === action.id ? { ...l, blend: action.value } : l)),
      };
    }
    case 'SET_VISIBILITY': {
      return {
        ...state,
        layers: state.layers.map((l) => (l.id === action.id ? { ...l, visible: !!action.value } : l)),
      };
    }
    case 'REORDER': {
      const { from, to } = action;
      if (from === to || from < 0 || to < 0 || from >= state.layers.length || to >= state.layers.length) {
        return state;
      }
      const next = [...state.layers];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return { ...state, layers: next };
    }
    case 'REPLACE_ALL': {
      return { ...state, layers: action.layers ?? [] };
    }
    case 'CLEAR': {
      return { ...state, layers: [] };
    }
    case 'SET_THRESHOLD': {
      return {
        ...state,
        layers: state.layers.map((l) =>
          l.id === action.id ? { ...l, threshold: action.value } : l
        ),
      };
    }
    case 'SET_LAYER_PARAMS': {
      // Merge per-layer backend-request params (e.g. band_min/band_max for
      // ci-threshold). The fetch layer in useLayerStack.js threads
      // ``layer.params`` into the GET /api/phasemap/layer query string.
      return {
        ...state,
        layers: state.layers.map((l) =>
          l.id === action.id
            ? { ...l, params: { ...(l.params || {}), ...(action.value || {}) } }
            : l
        ),
      };
    }
    case 'ADD_MASK_FROM_THRESHOLD': {
      const src = state.layers.find((l) => l.id === action.sourceId);
      if (!src || !src.threshold) return state;
      if (state.layers.length >= MAX_LAYERS) return state;
      const stamp = Date.now();
      const maskLayer = {
        id: `mask-${src.id}-${stamp}`,
        kind: 'mask',
        label: `Mask: ${src.label}`,
        isMaskFor: src.id,
        threshold: { ...src.threshold },
        visible: true,
        opacity: 1,
        blend: 'multiply',
        key: `mask-${src.id}-${stamp}`,
      };
      return { ...state, layers: [...state.layers, maskLayer] };
    }
    default:
      return state;
  }
}
