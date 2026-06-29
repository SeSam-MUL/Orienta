import { describe, it, expect } from 'vitest';
import { PRESETS, applyPreset } from './presets';
import { LAYER_SOURCES, findLayerDef } from './layerSources';

describe('layerSources catalog', () => {
  it('exposes 6 source groups', () => {
    expect(Object.keys(LAYER_SOURCES)).toEqual(
      ['result', 'analysis', 'ebsd', 'diagnostics', 'refinement', 'h5']
    );
  });
  it('result group includes phase + ipf + bc + ci + uncertainty', () => {
    const ids = LAYER_SOURCES.result.layers.map((l) => l.id);
    expect(ids).toContain('phase');
    expect(ids).toContain('ipf-z');
    expect(ids).toContain('bc');
    expect(ids).toContain('ci');
    expect(ids).toContain('uncertainty');
  });
  it('findLayerDef resolves by id across groups', () => {
    expect(findLayerDef('phase')).toMatchObject({ id: 'phase', source: 'result' });
    expect(findLayerDef('vbse')).toMatchObject({ id: 'vbse', source: 'ebsd' });
    expect(findLayerDef('unknown')).toBeNull();
  });
});

describe('presets', () => {
  it('has the 4 documented presets', () => {
    expect(Object.keys(PRESETS).sort()).toEqual(
      ['aztec_style', 'eds_verify', 'phase_default', 'quality_check']
    );
  });
  it('phase_default has exactly one layer at full opacity', () => {
    const layers = applyPreset('phase_default');
    expect(layers).toHaveLength(1);
    expect(layers[0]).toMatchObject({ id: 'phase', opacity: 1.0, blend: 'normal', visible: true });
  });
  it('aztec_style stacks IPF-Z under BC Multiply', () => {
    const layers = applyPreset('aztec_style');
    expect(layers).toHaveLength(2);
    expect(layers[0].id).toBe('ipf-z');
    expect(layers[1]).toMatchObject({ id: 'bc', blend: 'multiply' });
  });
  it('every preset references only known layer ids', () => {
    for (const [name, preset] of Object.entries(PRESETS)) {
      for (const layer of preset.layers) {
        expect(findLayerDef(layer.id), `${name} → ${layer.id}`).not.toBeNull();
      }
    }
  });
});
