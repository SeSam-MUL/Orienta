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
import { parseAddonLayerId } from './addonLayerDrain';

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
    // Same slot, different map: keeps position, opacity, blend, visibility and
    // threshold, and swaps only what the layer SHOWS. Rebuilding it through
    // REMOVE + ADD would drop it to the end of the stack at default settings,
    // which is exactly the work this saves.
    case 'RETYPE': {
      const at = state.layers.findIndex((l) => l.id === action.id);
      if (at < 0) return state;
      // Refuse a duplicate: two layers with the same id would fight over one
      // bitmap cache entry.
      if (state.layers.some((l) => l.id === action.layer.id)) return state;
      const prev = state.layers[at];
      const next = {
        ...action.layer,
        opacity: prev.opacity,
        blend: prev.blend,
        visible: prev.visible,
        // A threshold is a window on THIS layer's values (band contrast 0..255,
        // CI 0..1, …). Carrying it to a different quantity would silently mask
        // the wrong pixels, so the new layer starts unthresholded.
        threshold: undefined,
      };
      const layers = state.layers.slice();
      layers[at] = next;
      return { ...state, layers };
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
    // `keepAddonsFor` is a RESULT ID, and only the re-seed passes it.
    //
    // Measured in the running app: the first "show as layer" on a Phase Maps
    // page that had not been opened yet was lost. The add-on page queues the
    // layer and navigates; the drain adds it; and then useLayerStack's
    // resetSignal effect fires — the result id has just arrived — and re-seeds
    // the stack from the last quick mode, or from the default preset when
    // there is none (which is what happened here), replacing everything
    // including the layer the user is being carried over to look at. The
    // second click worked, which is what made it look like a fluke rather
    // than an ordering.
    //
    // Kept by RESULT, not by "is an add-on layer": a layer bound to a
    // different result draws that result's pixels over this one's map, and
    // says nothing about it — fetchLayer's wrong-result guard only bites once
    // the id it compares against has caught up. Which is why the caller must
    // pass a result id that moves WITH the selection; see the note there.
    //
    // Appended, so the seeded layers keep the order the preset gives them and
    // the add-on map stays on top, where the drain put it. MAX_LAYERS is
    // honoured: a seed that fills the stack wins over a kept layer, because
    // the seed is what the user's own last choice asked for.
    case 'REPLACE_ALL': {
      const seeded = action.layers ?? [];
      const keepFor = action.keepAddonsFor;
      if (!keepFor) return { ...state, layers: seeded };
      const kept = state.layers.filter((l) => {
        const p = parseAddonLayerId(l.id);
        return p && p.resultId === keepFor && !seeded.some((s) => s.id === l.id);
      });
      return { ...state, layers: [...seeded, ...kept].slice(0, MAX_LAYERS) };
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
