import { describe, it, expect } from 'vitest';
import { layerStackReducer, initialState, MAX_LAYERS } from './layerStackReducer';

const sampleLayer = (id, overrides = {}) => ({
  id, label: id, source: 'result',
  opacity: 1.0, blend: 'normal', visible: true,
  key: `${id}-test`,
  ...overrides,
});

describe('layerStackReducer', () => {
  it('initial state has empty layers', () => {
    expect(initialState.layers).toEqual([]);
  });

  it('ADD appends a layer', () => {
    const s = layerStackReducer(initialState, {
      type: 'ADD', layer: sampleLayer('phase'),
    });
    expect(s.layers).toHaveLength(1);
    expect(s.layers[0].id).toBe('phase');
  });

  it('ADD rejects duplicate id (same layer cannot stack twice)', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('phase') });
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('phase') });
    expect(s.layers).toHaveLength(1);
  });

  it('ADD beyond MAX_LAYERS is rejected', () => {
    let s = initialState;
    for (let i = 0; i < MAX_LAYERS; i++) {
      s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer(`layer-${i}`) });
    }
    expect(s.layers).toHaveLength(MAX_LAYERS);
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('one-too-many') });
    expect(s.layers).toHaveLength(MAX_LAYERS);
  });

  it('REMOVE deletes the layer by id', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('phase') });
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('bc') });
    s = layerStackReducer(s, { type: 'REMOVE', id: 'phase' });
    expect(s.layers.map(l => l.id)).toEqual(['bc']);
  });

  it('SET_OPACITY updates only the targeted layer', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('phase') });
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('bc') });
    s = layerStackReducer(s, { type: 'SET_OPACITY', id: 'bc', value: 0.4 });
    expect(s.layers.find(l => l.id === 'phase').opacity).toBe(1.0);
    expect(s.layers.find(l => l.id === 'bc').opacity).toBe(0.4);
  });

  it('SET_OPACITY clamps to [0,1]', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('phase') });
    s = layerStackReducer(s, { type: 'SET_OPACITY', id: 'phase', value: 1.5 });
    expect(s.layers[0].opacity).toBe(1.0);
    s = layerStackReducer(s, { type: 'SET_OPACITY', id: 'phase', value: -0.2 });
    expect(s.layers[0].opacity).toBe(0.0);
  });

  it('SET_BLEND updates blend mode', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('bc') });
    s = layerStackReducer(s, { type: 'SET_BLEND', id: 'bc', value: 'multiply' });
    expect(s.layers[0].blend).toBe('multiply');
  });

  it('SET_VISIBILITY toggles visible flag', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('phase') });
    s = layerStackReducer(s, { type: 'SET_VISIBILITY', id: 'phase', value: false });
    expect(s.layers[0].visible).toBe(false);
  });

  it('REORDER moves a layer to a new index', () => {
    let s = initialState;
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('a') });
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('b') });
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('c') });
    s = layerStackReducer(s, { type: 'REORDER', from: 0, to: 2 });
    expect(s.layers.map(l => l.id)).toEqual(['b', 'c', 'a']);
  });

  it('REPLACE_ALL substitutes the entire stack (for preset application)', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('old') });
    s = layerStackReducer(s, {
      type: 'REPLACE_ALL',
      layers: [sampleLayer('new-1'), sampleLayer('new-2')],
    });
    expect(s.layers.map(l => l.id)).toEqual(['new-1', 'new-2']);
  });

  it('CLEAR empties the stack', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('phase') });
    s = layerStackReducer(s, { type: 'CLEAR' });
    expect(s.layers).toEqual([]);
  });

  it('unknown action returns state unchanged', () => {
    const s = layerStackReducer(initialState, { type: 'BOGUS' });
    expect(s).toBe(initialState);
  });

  it('SET_THRESHOLD attaches threshold to the targeted layer only', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('eds-Al') });
    s = layerStackReducer(s, { type: 'ADD', layer: sampleLayer('bc') });
    s = layerStackReducer(s, { type: 'SET_THRESHOLD', id: 'bc', value: { min: 20, max: 200 } });
    expect(s.layers.find(l => l.id === 'eds-Al').threshold).toBeUndefined();
    expect(s.layers.find(l => l.id === 'bc').threshold).toEqual({ min: 20, max: 200 });
  });

  it('SET_THRESHOLD with undefined clears the threshold', () => {
    let s = layerStackReducer(initialState, { type: 'ADD', layer: sampleLayer('bc') });
    s = layerStackReducer(s, { type: 'SET_THRESHOLD', id: 'bc', value: { min: 10, max: 250 } });
    expect(s.layers[0].threshold).toEqual({ min: 10, max: 250 });
    s = layerStackReducer(s, { type: 'SET_THRESHOLD', id: 'bc', value: undefined });
    expect(s.layers[0].threshold).toBeUndefined();
  });

  describe('ADD_MASK_FROM_THRESHOLD', () => {
    it('adds a mask layer derived from the source layer threshold', () => {
      const state = {
        layers: [{ id: 'src', kind: 'eds-element', label: 'Al', threshold: { min: 20, max: 200 }, visible: true, opacity: 0.7, blend: 'screen', key: 'src-1' }],
      };
      const next = layerStackReducer(state, { type: 'ADD_MASK_FROM_THRESHOLD', sourceId: 'src' });
      expect(next.layers).toHaveLength(2);
      const mask = next.layers[1];
      expect(mask.kind).toBe('mask');
      expect(mask.isMaskFor).toBe('src');
      expect(mask.threshold).toEqual({ min: 20, max: 200 });
      expect(mask.blend).toBe('multiply');
    });
    it('no-op when source has no threshold', () => {
      const state = {
        layers: [{ id: 'src', kind: 'eds-element', label: 'Al', visible: true, opacity: 0.7, blend: 'screen', key: 'src-1' }],
      };
      const next = layerStackReducer(state, { type: 'ADD_MASK_FROM_THRESHOLD', sourceId: 'src' });
      expect(next.layers).toHaveLength(1);
    });
    it('no-op when source not found', () => {
      const state = { layers: [] };
      const next = layerStackReducer(state, { type: 'ADD_MASK_FROM_THRESHOLD', sourceId: 'nonexistent' });
      expect(next.layers).toEqual([]);
    });
  });
});
