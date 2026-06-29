// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest';
import { savePreset, listPresets, loadPreset, deletePreset } from '../layoutPresets';

beforeEach(() => localStorage.clear());

describe('layoutPresets', () => {
  it('round-trips a model', () => {
    const model = { canvas: { wPx: 100, hPx: 50 }, elements: [{ id: 'p', type: 'panel' }] };
    savePreset('fig1', model);
    expect(listPresets().map(p => p.name)).toContain('fig1');
    expect(loadPreset('fig1')).toEqual(model);
    deletePreset('fig1');
    expect(loadPreset('fig1')).toBeNull();
  });
  it('returns a deep copy on load (mutation-safe)', () => {
    const model = { canvas: { wPx: 1, hPx: 1 }, elements: [{ id: 'a' }] };
    savePreset('x', model);
    const a = loadPreset('x');
    a.elements[0].id = 'mutated';
    expect(loadPreset('x').elements[0].id).toBe('a');
  });
  it('survives corrupt storage', () => {
    localStorage.setItem('pm_figure_presets', '{not json');
    expect(listPresets()).toEqual([]);
    expect(loadPreset('any')).toBeNull();
  });
});
