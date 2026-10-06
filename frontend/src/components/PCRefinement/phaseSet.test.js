import { describe, it, expect } from 'vitest';
import {
  MAX_PC_PHASES, phaseLabelFor, loadedPaths, planPhaseSync, loadedPhaseFor, sameNameOtherFile,
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
  const loaded = (...paths) => paths.map((p) => ({ name: p.split(/[\\/]/).pop().replace(/\.cif$/, ''), path: p }));
  it('marks library entries that are the loaded file', () => {
    expect(loadedPaths(files, loaded('C:\\lib\\ferrite.cif', 'C:\\lib\\Al.cif')))
      .toEqual(['C:\\lib\\ferrite.cif', 'C:\\lib\\Al.cif']);
  });
  it('includes typed-in paths that are loaded and is not fooled by ones that are not', () => {
    expect(loadedPaths(files, loaded('/x/mine.cif'), ['/x/mine.cif', '/x/other.cif'])).toEqual(['/x/mine.cif']);
  });
  it('lists a path once even if both sources have it', () => {
    expect(loadedPaths(files, loaded('C:\\lib\\Al.cif'), ['C:\\lib\\Al.cif'])).toEqual(['C:\\lib\\Al.cif']);
  });
  it('a file that only shares its NAME with a loaded phase is not loaded', () => {
    // library Al is loaded; an external Al.cif is another file
    expect(loadedPaths(files, loaded('C:\\lib\\Al.cif'), ['D:\\mine\\Al.cif'])).toEqual(['C:\\lib\\Al.cif']);
  });
  it('spellings of one Windows path agree (case, slashes)', () => {
    expect(loadedPaths(files, loaded('c:/LIB/al.cif'))).toEqual(['C:\\lib\\Al.cif']);
  });
  it('a phase whose file is not known is matched by its name (single-phase Load button)', () => {
    expect(loadedPaths(files, [{ name: 'Al' }])).toEqual(['C:\\lib\\Al.cif']);
  });
});

describe('loadedPhaseFor / sameNameOtherFile', () => {
  const phases = [{ name: 'Al', path: 'C:\\lib\\Al.cif' }];
  it('finds the phase by file', () => {
    expect(loadedPhaseFor(phases, 'C:/lib/Al.cif')).toBe(phases[0]);
    expect(loadedPhaseFor(phases, 'D:\\mine\\Al.cif')).toBeUndefined();
  });
  it('sees another file of the same name, whatever its case', () => {
    expect(sameNameOtherFile(phases, 'D:\\mine\\AL.cif')).toBe(phases[0]);
    expect(sameNameOtherFile(phases, 'C:\\lib\\Al.cif')).toBeUndefined();   // the same file
    expect(sameNameOtherFile(phases, 'D:\\mine\\Ni.cif')).toBeUndefined();
  });
});

describe('planPhaseSync', () => {
  const ph = (...names) => names.map((n) => ({ name: n, path: `/l/${n}.cif` }));
  it('adds what is wanted and not loaded, removes what is loaded and not wanted', () => {
    expect(planPhaseSync(ph('Al', 'Si'), ['/l/Si.cif', '/l/Ni.cif']))
      .toEqual({ add: ['/l/Ni.cif'], remove: ['Al'] });
  });
  it('does nothing when the sets agree', () => {
    expect(planPhaseSync(ph('Al'), ['/l/Al.cif'])).toEqual({ add: [], remove: [] });
  });
  it('"None" removes everything loaded', () => {
    expect(planPhaseSync(ph('Al', 'Si'), [])).toEqual({ add: [], remove: ['Al', 'Si'] });
  });
  it('asking for another file called Al while library Al is loaded swaps nothing silently', () => {
    expect(planPhaseSync(ph('Al'), ['/mine/Al.cif'])).toEqual({ add: ['/mine/Al.cif'], remove: ['Al'] });
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

import { shortSpaceGroup } from './phaseSet';

describe('shortSpaceGroup', () => {
  it('takes the symbol out of the long description', () => {
    expect(shortSpaceGroup('SpaceGroup #225 (Fm-3m, Cubic). Symmetry matrices: 192, point sym. matr.: 48'))
      .toBe('Fm-3m');
  });
  it('leaves anything else alone', () => {
    expect(shortSpaceGroup('Fm-3m')).toBe('Fm-3m');
    expect(shortSpaceGroup('')).toBe('');
    expect(shortSpaceGroup(undefined)).toBe('');
  });
});


describe('files identified by their resolved path', () => {
  // The listing reaches the library through a link; the phase was loaded from the
  // real folder (or the other way round).
  const files = [{ path: 'D:/work/Database/Al.cif', real_path: 'D:/main/Database/Al.cif' },
                 { path: 'D:/work/Database/Si.cif', real_path: 'D:/main/Database/Si.cif' }];
  const phases = [{ name: 'Al', path: 'D:/main/Database/Al.cif', real_path: 'D:/main/Database/Al.cif' }];

  it('loadedPhaseFor finds a phase by the real file of a record', () => {
    expect(loadedPhaseFor(phases, files[0])).toBe(phases[0]);
    expect(loadedPhaseFor(phases, files[1])).toBeUndefined();
  });
  it('a bare path is resolved through the files it can be looked up in', () => {
    expect(loadedPhaseFor(phases, files[0].path, files)).toBe(phases[0]);
  });
  it('sameNameOtherFile does not call the same real file "another file"', () => {
    expect(sameNameOtherFile(phases, files[0])).toBeUndefined();
    const other = { path: 'E:/mine/Al.cif', real_path: 'E:/mine/Al.cif' };
    expect(sameNameOtherFile(phases, other)).toBe(phases[0]);
  });
  it('loadedPaths marks the library entry whose real file is loaded', () => {
    expect(loadedPaths(files, phases)).toEqual([files[0].path]);
  });
  it('planPhaseSync keeps a loaded phase that the wanted library path names', () => {
    expect(planPhaseSync(phases, [files[0].path], files)).toEqual({ add: [], remove: [] });
    expect(planPhaseSync(phases, [files[1].path], files)).toEqual({ add: [files[1].path], remove: ['Al'] });
  });
});
