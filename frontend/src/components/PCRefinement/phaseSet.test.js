import { describe, it, expect } from 'vitest';
import {
  MAX_PC_PHASES, phaseLabelFor, loadedPaths, planPhaseSync,
  summarisePatternPhases, previewPhaseName, createSerialQueue,
} from './phaseSet';

describe('phaseLabelFor', () => {
  it('is the fallback with nothing loaded, the name for one, "A + B" for several', () => {
    expect(phaseLabelFor([], 'No phase loaded')).toBe('No phase loaded');
    expect(phaseLabelFor(['Al'], 'x')).toBe('Al');
    expect(phaseLabelFor(['austenite', 'ferrite'], 'x')).toBe('austenite + ferrite');
  });
});

describe('loadedPaths', () => {
  const files = [{ path: 'C:\\lib\\austenite.cif' }, { path: 'C:\\lib\\ferrite.cif' },
                 { path: 'C:\\lib\\Al.cif' }];
  it('marks library entries whose stem is loaded', () => {
    expect(loadedPaths(files, ['ferrite', 'Al'])).toEqual(['C:\\lib\\ferrite.cif', 'C:\\lib\\Al.cif']);
  });
  it('includes typed-in paths that are loaded and is not fooled by ones that are not', () => {
    expect(loadedPaths(files, ['mine'], ['/x/mine.cif', '/x/other.cif'])).toEqual(['/x/mine.cif']);
  });
  it('lists a path once even if both sources have it', () => {
    expect(loadedPaths(files, ['Al'], ['C:\\lib\\Al.cif'])).toEqual(['C:\\lib\\Al.cif']);
  });
});

describe('planPhaseSync', () => {
  it('adds what is wanted and not loaded, removes what is loaded and not wanted', () => {
    expect(planPhaseSync(['Al', 'Si'], ['/l/Si.cif', '/l/Ni.cif']))
      .toEqual({ add: ['/l/Ni.cif'], remove: ['Al'] });
  });
  it('does nothing when the sets agree', () => {
    expect(planPhaseSync(['Al'], ['/l/Al.cif'])).toEqual({ add: [], remove: [] });
  });
  it('"None" removes everything loaded', () => {
    expect(planPhaseSync(['Al', 'Si'], [])).toEqual({ add: [], remove: ['Al', 'Si'] });
  });
});

describe('summarisePatternPhases', () => {
  it('counts patterns per phase in order of first appearance', () => {
    const pp = [{ phase_name: 'ferrite' }, { phase_name: 'austenite' }, { phase_name: 'ferrite' },
                { phase_name: null }];
    expect(summarisePatternPhases(pp, 'none')).toEqual([
      { name: 'ferrite', count: 2 }, { name: 'austenite', count: 1 }, { name: 'none', count: 1 },
    ]);
  });
  it('is empty without a result', () => {
    expect(summarisePatternPhases(undefined, 'none')).toEqual([]);
  });
});

describe('previewPhaseName', () => {
  const patterns = [{ phase: 'ferrite' }, { phase: null }];
  it('with one phase it is the label, whatever the patterns say (unchanged behaviour)', () => {
    expect(previewPhaseName({ phaseNames: ['Ni'], phaseLabel: 'Ni', patterns: [{ phase: 'Zz' }], selectedIdx: 0 }))
      .toBe('Ni');
    expect(previewPhaseName({ phaseNames: [], phaseLabel: 'No phase loaded', patterns, selectedIdx: 0 }))
      .toBe('No phase loaded');
  });
  it('with several it is the selected pattern\'s own phase', () => {
    expect(previewPhaseName({ phaseNames: ['austenite', 'ferrite'], phaseLabel: 'austenite + ferrite', patterns, selectedIdx: 0 }))
      .toBe('ferrite');
  });
  it('with several and no result for the pattern, the first loaded phase', () => {
    const args = { phaseNames: ['austenite', 'ferrite'], phaseLabel: 'austenite + ferrite', patterns };
    expect(previewPhaseName({ ...args, selectedIdx: 1 })).toBe('austenite');
    expect(previewPhaseName({ ...args, selectedIdx: null })).toBe('austenite');
  });
});

describe('createSerialQueue', () => {
  it('runs jobs one after another, in order, even if an earlier one fails', async () => {
    const run = createSerialQueue();
    const log = [];
    const slow = run(async () => { log.push('a-start'); await new Promise((r) => setTimeout(r, 15)); log.push('a-end'); });
    const bad = run(async () => { log.push('b'); throw new Error('boom'); });
    const last = run(async () => { log.push('c'); return 7; });
    await slow;
    await expect(bad).rejects.toThrow('boom');
    expect(await last).toBe(7);
    expect(log).toEqual(['a-start', 'a-end', 'b', 'c']);
  });
});

describe('MAX_PC_PHASES', () => {
  it('is the backend limit', () => {
    expect(MAX_PC_PHASES).toBe(8);   // PCController.MAX_PHASES
  });
});
