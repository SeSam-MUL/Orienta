// The statistics phase and the page sync, in their pure parts: what the answers carry, how the settings
// follow `GET /status`, how the export warnings are read from a header. Shapes are the backend's
// (backend/api/routes/analysis.py: `_edge_fields`, `/status`, the batch task, `X-Orienta-Warnings`).
import { describe, it, expect } from 'vitest';
import {
  DEFAULT_GRAIN_STATS_SETTINGS, batchReconstruction, datasetNameFromFile, grainStatsFrom, mergeGrainSize, reconstructArgs,
  samePath, settingsFromStatus, statsPhaseLabel, warningsFromHeader,
} from '../grainStats';

const PHASES = [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }];

describe('the statistics phase in the answers', () => {
  const phaseResult = {
    edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: 2, stats_phase_name: 'Si',
    n_grains_total: 12, n_edge_grains: 5, n_grains_used: 7, edge_safeguard_applied: false,
    mean_ecd: 3.5, median_ecd: 3.1, std_ecd: 0.9, mean_aspect_ratio: 1.4,
  };

  it('grainStatsFrom carries the phase id and name next to the counts', () => {
    const s = grainStatsFrom({ status: 'completed', grain_count: 98, result: { grain_count: 98, n_grains: 98, ...phaseResult } });
    expect(s).toMatchObject({ count: 98, statsPhase: 2, statsPhaseName: 'Si', nGrainsTotal: 12 });
  });

  it('a /grain-size answer for a phase does not turn the phase count into the total of the map', () => {
    // no statistics shown yet (tab opened again): the answer's n_grains_total is that of the phase
    const merged = mergeGrainSize(null, { success: true, ...phaseResult });
    expect(merged.count).toBeUndefined();
    expect(merged).toMatchObject({ nGrainsTotal: 12, statsPhase: 2, statsPhaseName: 'Si' });
    // with all phases it is the total
    expect(mergeGrainSize(null, { success: true, ...phaseResult, stats_phase: null, stats_phase_name: null }).count).toBe(12);
  });

  it('the total of the map stays when the phase changes', () => {
    const prev = grainStatsFrom({ grain_count: 98, ...phaseResult, stats_phase: null, n_grains_total: 98 });
    expect(mergeGrainSize(prev, { success: true, ...phaseResult }).count).toBe(98);
  });

  it('statsPhaseLabel prefers the answer\'s name, then the map\'s phase list', () => {
    expect(statsPhaseLabel({ statsPhase: 2, statsPhaseName: 'Si' }, [])).toBe('Si');
    expect(statsPhaseLabel({ statsPhase: 1 }, PHASES)).toBe('Aluminium');
    expect(statsPhaseLabel({ statsPhase: 9 }, PHASES)).toBeNull();
    expect(statsPhaseLabel({ statsPhase: null }, PHASES)).toBeNull();
  });
});

describe('the phase in the requests', () => {
  it('reconstructArgs ends with the phase, null for all phases', () => {
    expect(reconstructArgs(DEFAULT_GRAIN_STATS_SETTINGS).at(-1)).toBeNull();
    expect(reconstructArgs({ ...DEFAULT_GRAIN_STATS_SETTINGS, statsPhase: 2 }).at(-1)).toBe(2);
  });

  it('batchReconstruction gives the Grain tab\'s numbers as numbers, null when one is not a number', () => {
    expect(batchReconstruction(DEFAULT_GRAIN_STATS_SETTINGS)).toEqual({ threshold: 4, minPixels: 3, minIntercept: 1.5, alpha: 3.1, fill: true });
    expect(batchReconstruction({ ...DEFAULT_GRAIN_STATS_SETTINGS, gbThreshold: '6.5', fill: false }))
      .toEqual({ threshold: 6.5, minPixels: 3, minIntercept: 1.5, alpha: 3.1, fill: false });
    // the min intercept is recorded only: never a reason to refuse; a value that is not a number of 0 or more is left out
    expect(batchReconstruction({ ...DEFAULT_GRAIN_STATS_SETTINGS, minIntercept: '' }))
      .toEqual({ threshold: 4, minPixels: 3, minIntercept: undefined, alpha: 3.1, fill: true });
    expect(batchReconstruction({ ...DEFAULT_GRAIN_STATS_SETTINGS, minIntercept: '-1' }).minIntercept).toBeUndefined();
    expect(batchReconstruction({ ...DEFAULT_GRAIN_STATS_SETTINGS, minIntercept: 'abc' }).minIntercept).toBeUndefined();
    expect(batchReconstruction({ ...DEFAULT_GRAIN_STATS_SETTINGS, minIntercept: '0' }).minIntercept).toBe(0);
    expect(batchReconstruction({ ...DEFAULT_GRAIN_STATS_SETTINGS, alpha: '' })).toBeNull();
  });
});

describe('datasetNameFromFile', () => {
  it('is the file name without folder and extension', () => {
    expect(datasetNameFromFile('Sample 7050 AB1.h5oina')).toBe('Sample 7050 AB1');
    expect(datasetNameFromFile('C:\\data\\scan.01.ctf')).toBe('scan.01');
    expect(datasetNameFromFile('/home/me/scan.ang')).toBe('scan');
    expect(datasetNameFromFile('noextension')).toBe('noextension');
    expect(datasetNameFromFile('.hidden')).toBe('.hidden');
  });

  it('is empty when there is no name', () => {
    expect(datasetNameFromFile(null)).toBe('');
    expect(datasetNameFromFile(undefined)).toBe('');
    expect(datasetNameFromFile('')).toBe('');
  });
});

describe('settingsFromStatus', () => {
  const status = {
    loaded: true, has_grains: true, edge_rule: 'outer_row', ecd_basis: 'measured_pixels', stats_phase: 2,
    recon_params: { threshold: 6.5, min_pixels: 5, min_intercept: 2.0, alpha: 0.0, fill: false },
  };

  it('takes the statistics settings and the reconstruction parameters of the grains it has', () => {
    expect(settingsFromStatus(DEFAULT_GRAIN_STATS_SETTINGS, status, PHASES)).toEqual({
      gbThreshold: '6.5', minPixels: '5', minIntercept: '2', alpha: '0', fill: false,
      edgeRule: 'outer_row', ecdBasis: 'measured_pixels', statsPhase: 2,
    });
  });

  it('leaves a field as typed when it already shows the number (4.0 stays 4.0)', () => {
    const out = settingsFromStatus(DEFAULT_GRAIN_STATS_SETTINGS, {
      ...status, recon_params: { threshold: 4, min_pixels: 3, min_intercept: 1.5, alpha: 3.1, fill: true },
    }, PHASES);
    expect(out).toMatchObject({ gbThreshold: '4.0', minPixels: '3', minIntercept: '1.5', alpha: '3.1', fill: true });
  });

  it('keeps the reconstruction parameters when the backend does not know them', () => {
    const typed = { ...DEFAULT_GRAIN_STATS_SETTINGS, gbThreshold: '9' };
    expect(settingsFromStatus(typed, { ...status, recon_params: null }, PHASES).gbThreshold).toBe('9');
    // a key that is null inside the object (fill of older code) is not applied either
    expect(settingsFromStatus(typed, { ...status, recon_params: { threshold: null, fill: null } }, PHASES))
      .toMatchObject({ gbThreshold: '9', fill: true });
  });

  it('knows "most_measured" and all phases (null)', () => {
    expect(settingsFromStatus(DEFAULT_GRAIN_STATS_SETTINGS, { ...status, stats_phase: 'most_measured' }, PHASES).statsPhase)
      .toBe('most_measured');
    expect(settingsFromStatus({ ...DEFAULT_GRAIN_STATS_SETTINGS, statsPhase: 2 }, { ...status, stats_phase: null }, PHASES).statsPhase)
      .toBeNull();
  });

  it('ignores a rule or basis it does not know', () => {
    const out = settingsFromStatus(DEFAULT_GRAIN_STATS_SETTINGS, { ...status, edge_rule: 'nonsense', ecd_basis: 7 }, PHASES);
    expect(out).toMatchObject({ edgeRule: 'matlab_script', ecdBasis: 'cell_area' });
  });

  it('without grains the user\'s choices stay, except a phase the map does not have', () => {
    const chosen = { ...DEFAULT_GRAIN_STATS_SETTINGS, edgeRule: 'keep_all', statsPhase: 2 };
    const none = { loaded: true, has_grains: false, edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, recon_params: null };
    expect(settingsFromStatus(chosen, none, PHASES)).toMatchObject({ edgeRule: 'keep_all', statsPhase: 2 });
    expect(settingsFromStatus(chosen, none, [{ id: 1, name: 'Aluminium' }])).toMatchObject({ edgeRule: 'keep_all', statsPhase: null });
  });
});

describe('warningsFromHeader', () => {
  it('decodes the percent-encoded JSON list, a non-ASCII reason included', () => {
    const warnings = ['texture: ValueError: Größe µm; a, b', 'rx: needs the KAM of the deformation step, which failed'];
    // what the backend sends: urllib.parse.quote(json.dumps(list), safe="") with json's ASCII escapes ...
    const escaped = encodeURIComponent(JSON.stringify(warnings).replace(/[\u0080-￿]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`));
    expect(warningsFromHeader(escaped)).toEqual({ warnings, unreadable: false });
    // ... and the same with the characters themselves
    expect(warningsFromHeader(encodeURIComponent(JSON.stringify(warnings)))).toEqual({ warnings, unreadable: false });
  });

  it('no header means no warnings', () => {
    expect(warningsFromHeader(undefined)).toEqual({ warnings: [], unreadable: false });
    expect(warningsFromHeader('')).toEqual({ warnings: [], unreadable: false });
  });

  it('a header that cannot be read is reported as such, never dropped', () => {
    expect(warningsFromHeader('%E0%A4%A')).toEqual({ warnings: [], unreadable: true });       // bad percent sequence
    expect(warningsFromHeader(encodeURIComponent('not json'))).toEqual({ warnings: [], unreadable: true });
    expect(warningsFromHeader(encodeURIComponent('{"a": 1}'))).toEqual({ warnings: [], unreadable: true });
    expect(warningsFromHeader(encodeURIComponent('[1, 2]'))).toEqual({ warnings: [], unreadable: true });
  });
});

describe('samePath', () => {
  it('a Windows path in mixed slashes is the path the backend gives back as str(Path(...))', () => {
    expect(samePath('C:/data/Documentation.xlsx', 'C:\\data\\Documentation.xlsx', 'win32')).toBe(true);
    expect(samePath('C:\\data/Documentation.xlsx', 'C:\\data\\Documentation.xlsx', 'win32')).toBe(true);
  });

  it('on Windows the case and a trailing slash do not matter, a drive-letter path counts as Windows without a platform', () => {
    expect(samePath('c:/Data/doc.xlsx', 'C:\\data\\DOC.xlsx', 'win32')).toBe(true);
    expect(samePath('c:/Data/doc.xlsx', 'C:\\data\\DOC.xlsx')).toBe(true);
    expect(samePath('C:/data/', 'C:\\data')).toBe(true);
  });

  it('elsewhere the case matters', () => {
    expect(samePath('/home/a/Doc.xlsx', '/home/a/doc.xlsx', 'linux')).toBe(false);
    expect(samePath('/home/a/Doc.xlsx', '/home/a/Doc.xlsx', 'linux')).toBe(true);
  });

  it('a real rename is a different path on every platform', () => {
    expect(samePath('C:/out/Doc.xlsx', 'C:\\out\\Doc_2.xlsx', 'win32')).toBe(false);
    expect(samePath('/out/Doc.xlsx', '/out/Doc_2.xlsx', 'linux')).toBe(false);
  });
});
