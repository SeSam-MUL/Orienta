import { describe, it, expect } from 'vitest';
import { buildEdsLayerSources, defaultLayersFor, pickPreferredElectronImage, allMapsLayersFor } from './edsLayerSources';

describe('buildEdsLayerSources', () => {
  it('emits one source per detected EDS element', () => {
    const sources = buildEdsLayerSources({ elements: ['Al Kα1', 'Fe Kα1'] });
    const ids = sources.map(s => s.id);
    expect(ids).toContain('eds-Al Kα1');
    expect(ids).toContain('eds-Fe Kα1');
  });
  it('marks element layers with correct kind, blend, and opacity', () => {
    const [first] = buildEdsLayerSources({ elements: ['Al Kα1'] });
    expect(first.kind).toBe('eds-element');
    expect(first.defaultBlend).toBe('screen');
    expect(first.defaultOpacity).toBeCloseTo(0.7);
  });
  it('emits one source per electron image when provided', () => {
    const sources = buildEdsLayerSources({ elements: [], electronImages: ['SE1', 'SE2'] });
    const ids = sources.map(s => s.id);
    expect(ids).toEqual(['electron-SE1', 'electron-SE2']);
    expect(sources[0].kind).toBe('electron');
    expect(sources[0].defaultBlend).toBe('normal');
  });
  it('returns an empty array for an empty input', () => {
    expect(buildEdsLayerSources()).toEqual([]);
    expect(buildEdsLayerSources({})).toEqual([]);
  });
});

describe('defaultLayersFor', () => {
  it('uses real SE when hasSE=true', () => {
    const out = defaultLayersFor({ hasSE: true, hasBC: true, electronImageName: 'SE1', elements: ['Al Kα1', 'Fe Kα1', 'Cu Kα1'] });
    const kinds = out.map(l => l.kind);
    expect(kinds).toContain('electron');
    expect(kinds).not.toContain('vbse');
  });
  it('falls back to V-BSE when hasSE=false with explicit fallback label', () => {
    const out = defaultLayersFor({ hasSE: false, hasBC: true, elements: ['Al Kα1'] });
    const vbse = out.find(l => l.kind === 'vbse');
    expect(vbse).toBeDefined();
    expect(vbse.label).toMatch(/SE fallback/i);
  });
  it('omits BC layer when hasBC=false', () => {
    const out = defaultLayersFor({ hasSE: false, hasBC: false, elements: [] });
    expect(out.find(l => l.kind === 'bc')).toBeUndefined();
  });
  it('caps element defaults at the first 3 elements', () => {
    const out = defaultLayersFor({ hasSE: false, hasBC: false, elements: ['Al', 'Fe', 'Cu', 'O', 'Mg'] });
    const elLayers = out.filter(l => l.kind === 'eds-element');
    expect(elLayers).toHaveLength(3);
    expect(elLayers.map(l => l.element)).toEqual(['Al', 'Fe', 'Cu']);
  });
  it('seeds all default layers visible with At.% display mode', () => {
    const out = defaultLayersFor({ hasSE: true, hasBC: true, electronImageName: 'SE1', elements: ['Al'] });
    for (const l of out) expect(l.visible).toBe(true);
    const al = out.find(l => l.kind === 'eds-element');
    expect(al.displayMode).toBe('at');
  });
  it('returns an empty array when no inputs are usable', () => {
    expect(defaultLayersFor({ hasSE: false, hasBC: false, elements: [] })).toEqual([]);
  });
});

describe('allMapsLayersFor', () => {
  it('emits a layer for EVERY element (uncapped, unlike defaultLayersFor)', () => {
    const out = allMapsLayersFor({ elements: ['Al', 'Fe', 'Cu', 'O', 'Mg', 'Si', 'Zn'] });
    const elLayers = out.filter(l => l.kind === 'eds-element');
    expect(elLayers.map(l => l.element)).toEqual(['Al', 'Fe', 'Cu', 'O', 'Mg', 'Si', 'Zn']);
  });
  it('orders preferred SE image first, then BC, then element maps; all visible', () => {
    const out = allMapsLayersFor({
      elements: ['Al'],
      electronImages: ['FSE/A', 'SE/Main'],
      hasBC: true,
    });
    const ids = out.map(l => l.id);
    expect(ids[0]).toBe('electron-SE/Main');                   // preferred SE first
    expect(ids).toContain('electron-FSE/A');                   // other electron images kept
    expect(ids.indexOf('bc')).toBeLessThan(ids.indexOf('eds-Al')); // BC before elements
    for (const l of out) expect(l.visible).toBe(true);
  });
  it('omits BC when hasBC is false and handles empty inputs', () => {
    expect(allMapsLayersFor({ elements: [], electronImages: [], hasBC: false })).toEqual([]);
    const out = allMapsLayersFor({ elements: ['Al'], hasBC: false });
    expect(out.find(l => l.id === 'bc')).toBeUndefined();
  });
});

describe('pickPreferredElectronImage', () => {
  it('prefers SE over FSE', () => {
    expect(pickPreferredElectronImage(['FSE/Oben links', 'SE/Elektronenbild 1', 'FSE/Unten Mitte']))
      .toBe('SE/Elektronenbild 1');
  });
  it('prefers BSE over FSE when no SE present', () => {
    expect(pickPreferredElectronImage(['FSE/Oben links', 'BSE/Detector 1']))
      .toBe('BSE/Detector 1');
  });
  it('falls back to FSE when no SE or BSE present', () => {
    expect(pickPreferredElectronImage(['FSE/Oben links', 'FSE/Oben rechts']))
      .toBe('FSE/Oben links');
  });
  it('returns null for empty list', () => {
    expect(pickPreferredElectronImage([])).toBeNull();
    expect(pickPreferredElectronImage()).toBeNull();
    expect(pickPreferredElectronImage(undefined)).toBeNull();
  });
  it('falls back to first item for unknown prefixes', () => {
    expect(pickPreferredElectronImage(['Custom/Image 1', 'Other/Image 2']))
      .toBe('Custom/Image 1');
  });
});
