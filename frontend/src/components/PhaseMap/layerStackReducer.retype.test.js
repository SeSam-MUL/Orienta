import { describe, it, expect } from 'vitest';
import { layerStackReducer } from './layerStackReducer';

const stack = (...layers) => ({ layers });
const L = (id, extra = {}) => ({
  id, label: id, source: 'result', opacity: 1, blend: 'normal', visible: true, ...extra,
});

const retype = (state, id, nextId, extra = {}) => layerStackReducer(state, {
  type: 'RETYPE', id, layer: L(nextId, { opacity: 1, blend: 'normal', visible: true, ...extra }),
});

describe('RETYPE — same slot, different map', () => {
  it('keeps opacity, blend, visibility and position', () => {
    // The whole point: arranging a stack is the work, the map inside a layer
    // is the variable. Trying IPF-X with the settings built for the phase map
    // must not mean building them again.
    const before = stack(
      L('bc'),
      L('phase', { opacity: 0.42, blend: 'multiply', visible: false }),
      L('ci'),
    );
    const after = retype(before, 'phase', 'ipf-x');

    expect(after.layers.map((l) => l.id)).toEqual(['bc', 'ipf-x', 'ci']);
    const swapped = after.layers[1];
    expect(swapped.opacity).toBe(0.42);
    expect(swapped.blend).toBe('multiply');
    expect(swapped.visible).toBe(false);
  });

  it('drops a threshold instead of carrying it to another quantity', () => {
    // A band-contrast window of 40..200 means nothing on a CI layer, whose
    // values run 0..1 — it would blank the map and look like a broken layer.
    const before = stack(L('bc', { threshold: { min: 40, max: 200 } }));
    const after = retype(before, 'bc', 'ci');
    expect(after.layers[0].id).toBe('ci');
    expect(after.layers[0].threshold).toBeUndefined();
  });

  it('refuses a type the stack already shows', () => {
    const before = stack(L('phase'), L('ipf-z', { opacity: 0.3 }));
    const after = retype(before, 'ipf-z', 'phase');
    expect(after).toBe(before);
  });

  it('ignores an unknown layer', () => {
    const before = stack(L('phase'));
    expect(retype(before, 'nope', 'ipf-z')).toBe(before);
  });

  it('takes the new layer\'s own params, not the old one\'s', () => {
    // Params are request parameters for THIS map (phase filter, band limits).
    // Keeping them would send a phase map's filter to an IPF request.
    const before = stack(L('ipf-z', { params: { phase_filter: 2 } }));
    const after = layerStackReducer(before, {
      type: 'RETYPE', id: 'ipf-z', layer: L('bc', { params: { band_min: 10 } }),
    });
    expect(after.layers[0].params).toEqual({ band_min: 10 });
  });
});
