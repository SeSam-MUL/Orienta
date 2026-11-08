import { describe, it, expect } from 'vitest';
import {
  formatSetting, settingRows, sameCount, differenceLines, fitState, offerKind, versionMeta, isoDate,
  changeInPixels, replacesOf, formatPixels, savedWarning, savedContrast, baseName, sourceKindOf, noteWords,
  formatSigned, pcChange, contrastText, followerQuestion, olderResults, applyRows, copyKeptOf, otherAspect,
} from './calibrationText';

// A suggestion as the backend answers it (fake numbers): the file differs in the working distance only.
const FILE = { pattern_shape: [128, 156], binning: 8, camera_tilt: 5, sample_tilt: 70, working_distance_mm: 15, detector_insertion_mm: 55 };
const CAL = { ...FILE, working_distance_mm: 20 };
const offer = (extra = {}) => ({
  id: 'sem', name: 'SEM', version: 1, pc: [0.5, 0.5, 0.5], match: false, via_member: true,
  differences: [{ key: 'working_distance_mm', calibration: 20, file: 15, tolerance: 0.1 }],
  unknown: [], unverified: false,
  checked: ['pattern_shape', 'binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm'],
  settings: { file: FILE, calibration: CAL }, ...extra,
});

describe('formatSetting', () => {
  it('writes each setting for the table', () => {
    expect(formatSetting('pattern_shape', [128, 156])).toBe('128 × 156');
    expect(formatSetting('binning', 8)).toBe('8');
    expect(formatSetting('camera_tilt', 5)).toBe('5.0°');
    expect(formatSetting('sample_tilt', 70)).toBe('70.0°');
    expect(formatSetting('working_distance_mm', 20)).toBe('20 mm');
    expect(formatSetting('working_distance_mm', 15.55)).toBe('15.55 mm');
    expect(formatSetting('detector_insertion_mm', 55)).toBe('55 mm');
  });
  it('gives null for what is not stated, never 0 or "NaN"', () => {
    expect(formatSetting('binning', null)).toBeNull();
    expect(formatSetting('camera_tilt', undefined)).toBeNull();
    expect(formatSetting('working_distance_mm', NaN)).toBeNull();
    expect(formatSetting('pattern_shape', [128])).toBeNull();
    expect(formatSetting('pattern_shape', null)).toBeNull();
  });
});

describe('settingRows', () => {
  it('lists the six settings in their table order, with both values and the check', () => {
    const rows = settingRows(offer());
    expect(rows.map((r) => r.key)).toEqual(['pattern_shape', 'binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm']);
    expect(rows.map((r) => r.check)).toEqual(['same', 'same', 'same', 'same', 'different', 'same']);
    expect(rows[0]).toMatchObject({ file: '128 × 156', calibration: '128 × 156' });
    expect(rows[4]).toMatchObject({ file: '15 mm', calibration: '20 mm', check: 'different' });
  });
  it('marks a setting that is in "unknown" as not stated, with the side that has a value still showing it', () => {
    const o = offer({
      differences: [], unknown: ['binning', 'detector_insertion_mm'], checked: ['pattern_shape', 'camera_tilt', 'sample_tilt', 'working_distance_mm'],
      settings: { file: { ...FILE, binning: null, detector_insertion_mm: null, working_distance_mm: 20 }, calibration: CAL },
    });
    const rows = Object.fromEntries(settingRows(o).map((r) => [r.key, r]));
    expect(rows.binning).toMatchObject({ file: null, calibration: '8', check: 'notStated' });
    expect(rows.detector_insertion_mm).toMatchObject({ file: null, calibration: '55 mm', check: 'notStated' });
    expect(rows.working_distance_mm.check).toBe('same');
  });
  it('still shows the two values of a difference when the backend sent no settings', () => {
    const rows = settingRows(offer({ settings: undefined }));
    expect(rows[4]).toMatchObject({ file: '15 mm', calibration: '20 mm', check: 'different' });
    expect(rows[0]).toMatchObject({ file: null, calibration: null, check: 'same' });
  });
});

describe('sameCount', () => {
  it('is the number of settings really compared, the pattern size included', () => {
    expect(sameCount(offer())).toBe(6);
    expect(sameCount(offer({ checked: ['pattern_shape', 'camera_tilt', 'sample_tilt', 'working_distance_mm'] }))).toBe(4);
    expect(sameCount({})).toBe(0);
  });
});

describe('fitState and offerKind', () => {
  it('a match fits (A)', () => {
    const o = offer({ match: true, differences: [] });
    expect([fitState(o), offerKind(o)]).toEqual(['fits', 'fits']);
  });
  it('a difference does not fit (B)', () => {
    expect([fitState(offer()), offerKind(offer())]).toEqual(['no', 'differs']);
  });
  it('no match and no difference cannot be told (C)', () => {
    const o = offer({ differences: [], unverified: true });
    expect([fitState(o), offerKind(o)]).toEqual(['unstated', 'unstated']);
  });
  it('a match with some settings not stated still fits', () => {
    expect(fitState(offer({ match: true, differences: [], unknown: ['binning'] }))).toBe('fits');
  });
});

describe('differenceLines', () => {
  it('gives one line per difference with both values formatted', () => {
    expect(differenceLines(offer({ differences: [
      { key: 'working_distance_mm', calibration: 20, file: 15 }, { key: 'camera_tilt', calibration: 5, file: 4.32 },
    ] }))).toEqual([
      { key: 'working_distance_mm', file: '15 mm', calibration: '20 mm', aspect: false },
      { key: 'camera_tilt', file: '4.3°', calibration: '5.0°', aspect: false },
    ]);
  });
});

describe('versionMeta', () => {
  it('version 1 was saved, a later one was updated from a file', () => {
    expect(versionMeta({ version: 1, version_created: '2026-10-14T09:00:00+00:00', source_name: 'scan_a.h5oina' }))
      .toEqual({ key: 'versionSaved', params: { version: 1, date: '2026-10-14' } });
    expect(versionMeta({ version: 2, version_created: '2026-10-20T09:00:00+00:00', source_name: 'scan_x.h5oina' }))
      .toEqual({ key: 'versionUpdated', params: { version: 2, date: '2026-10-20', source: 'scan_x.h5oina' } });
  });
  it('leaves out what is not known', () => {
    expect(versionMeta({ version: 2, version_created: '2026-10-20', source_name: null }).key).toBe('versionUpdatedNoSource');
    expect(versionMeta({ version: 2, version_created: null, source_name: 'x' })).toEqual({ key: 'versionOnly', params: { version: 2 } });
    expect(versionMeta({ version: 1, version_created: 'yesterday' }).key).toBe('versionOnly');
  });
});

describe('isoDate', () => {
  it('keeps the date of an ISO stamp and nothing else', () => {
    expect(isoDate('2026-10-21T10:00:00')).toBe('2026-10-21');
    expect(isoDate('21.10.2026')).toBe('');
    expect(isoDate(null)).toBe('');
  });
});

describe('changeInPixels', () => {
  const SHAPE = [128, 156];                            // rows, columns
  it('counts x in columns, y and z in rows, as the length of the vector', () => {
    // 3 pixels in x (3 / 156) and 4 pixels in y (4 / 128): 3-4-5
    expect(changeInPixels([0.5, 0.5, 0.5], [0.5 + 3 / 156, 0.5 + 4 / 128, 0.5], SHAPE)).toBeCloseTo(5, 9);
    // z counts in rows, not columns: 2 pixels
    expect(changeInPixels([0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 2 / 128], SHAPE)).toBeCloseTo(2, 9);
  });
  it('these numbers give 0.6215 pixels (hand calculation: sqrt(0.1092^2 + 0.3712^2 + 0.4864^2))', () => {
    expect(changeInPixels([0.5019, 0.3301, 0.842], [0.5026, 0.3272, 0.8458], SHAPE)).toBeCloseTo(0.62153, 4);
    expect(formatPixels(0.62153)).toBe('0.6');
  });
  it('is null when a piece is missing', () => {
    expect(changeInPixels(null, [1, 1, 1], SHAPE)).toBeNull();
    expect(changeInPixels([1, 1, 1], [1, 1, 1], null)).toBeNull();
    expect(changeInPixels([1, 1, 1], [1, 1, NaN], SHAPE)).toBeNull();
  });
});

describe('replacesOf', () => {
  it('names what is replaced and how far the PC moves', () => {
    const o = offer({ pc: [0.5 + 1 / 156, 0.5, 0.5], replaces: { pc: [0.5, 0.5, 0.5], kind: 'file' } });
    const r = replacesOf(o);
    expect(r.kind).toBe('file');
    expect(r.pixels).toBeCloseTo(1, 9);
  });
  it('says nothing for a file without a PC and for a kind it does not know', () => {
    expect(replacesOf(offer({ replaces: { pc: null, kind: 'none' } }))).toBeNull();
    expect(replacesOf(offer({ replaces: null }))).toBeNull();
    expect(replacesOf(offer({ replaces: { pc: [0.5, 0.5, 0.5], kind: 'mystery' } }))).toBeNull();
  });
  it('gives no pixels for the mean of a PC map, and none when the pattern size is unknown', () => {
    expect(replacesOf(offer({ replaces: { pc: [0.4, 0.4, 0.4], kind: 'grid' } }))).toEqual({ kind: 'grid', pixels: null });
    expect(replacesOf(offer({ settings: undefined, replaces: { pc: [0.4, 0.4, 0.4], kind: 'find' } }))).toEqual({ kind: 'find', pixels: null });
  });
});

describe('what a saved calibration says of its result', () => {
  it('reads the warning of the three kinds and nothing else', () => {
    expect(savedWarning({ warning: 'edge', axes: ['y'] })).toBe('edge');
    expect(savedWarning({ warning: 'notAMinimum' })).toBe('notAMinimum');
    expect(savedWarning({ warning: 'secondPosition' })).toBe('secondPosition');
    expect(savedWarning({ warning: null })).toBeNull();
    expect(savedWarning(null)).toBeNull();
  });
  it('writes the contrast with one decimal, or nothing', () => {
    expect(savedContrast({ contrast: 0.43 })).toBe('0.4');
    expect(savedContrast({ contrast: 0.8 })).toBe('0.8');
    expect(savedContrast({ contrast: null })).toBeNull();
    expect(savedContrast(null)).toBeNull();
  });
});

describe('baseName and sourceKindOf', () => {
  it('takes the file name of either kind of path', () => {
    expect(baseName('C:\\data\\sample_c.ang')).toBe('sample_c.ang');
    expect(baseName('/data/sample_c.ang')).toBe('sample_c.ang');
    expect(baseName('')).toBe('');
  });
  it('knows the five kinds of file and calls the rest "other"', () => {
    expect(['ang', 'osc', 'edax_h5', 'oxford_h5', 'orienta_export'].map(sourceKindOf)).toEqual(['ang', 'osc', 'edax_h5', 'oxford_h5', 'orienta_export']);
    expect(sourceKindOf('xyz')).toBe('other');
  });
});

describe('noteWords', () => {
  it('translates by code and leaves a code it does not know to the backend text', () => {
    expect(noteWords({ code: 'assumedShape', params: { rows: 128, cols: 156, convention: 'tsl' } }, { kind: '.ang file', file: 'scan_a' }))
      .toEqual({ key: 'assumedShape.tsl', params: { kind: '.ang file', file: 'scan_a', rows: 128, cols: 156 } });
    expect(noteWords({ code: 'assumedShape', params: { rows: 128, cols: 156, convention: 'oxford' } }).key).toBe('assumedShape.oxford');
    expect(noteWords({ code: 'pixelsLeftOut', params: { left_out: 3, total: 40 } })).toEqual({ key: 'pixelsLeftOut', params: { left: 3, total: 40 } });
    expect(noteWords({ code: 'outsideDetectorRange', params: { axes: ['x', 'z'], limits: 'x 0-1' } }))
      .toEqual({ key: 'outsideDetectorRange', params: { axes: 'x, z', limits: 'x 0-1' } });
    expect(noteWords({ code: 'stdOverMap', params: { x: 0.0123456, y: 12.3456, z: 0.5 } }).params).toEqual({ x: 0.0123, y: 12.3, z: 0.5 });
    expect(noteWords({ code: 'meanOfMap', params: {} }).key).toBe('meanOfMap');
    expect(noteWords({ code: 'orientaExport', params: {} }).key).toBe('orientaExport');
    expect(noteWords({ code: 'other', params: {}, text: 'x' })).toBeNull();
    expect(noteWords({ code: 'somethingNew', params: {} })).toBeNull();
  });
  it('does not translate a note whose numbers are missing: its own text is better than a hole', () => {
    expect(noteWords({ code: 'assumedShape', params: { convention: 'tsl' } })).toBeNull();
    expect(noteWords({ code: 'assumedShape', params: { rows: 1, cols: 2, convention: 'bruker' } })).toBeNull();
    expect(noteWords({ code: 'pixelsLeftOut', params: {} })).toBeNull();
  });
});

describe('the change a refine made (screen 7)', () => {
  it('writes each axis with its sign and four decimals', () => {
    expect(formatSigned(0.0007)).toBe('+0.0007');
    expect(formatSigned(-0.0029)).toBe('-0.0029');
  });
  it('gives a number that rounds to zero no sign, whichever side of zero it came from', () => {
    expect(formatSigned(0)).toBe('0.0000');
    expect(formatSigned(-0.00001)).toBe('0.0000');       // 4 decimals: -0.00001 is 0.0000, not "-0.0000"
    expect(formatSigned(0.00004)).toBe('0.0000');
    expect(formatSigned(-0)).toBe('0.0000');
    expect(formatSigned(-0.00007)).toBe('-0.0001');      // rounds to 0.0001: not a zero, keeps its sign
    expect(formatSigned(0.00006)).toBe('+0.0001');
  });
  it('is the difference per axis, to minus from; null where a PC is missing', () => {
    const c = pcChange([0.5019, 0.3301, 0.842], [0.5026, 0.3272, 0.8458]);
    expect(c.delta.map((v) => Number(v.toFixed(4)))).toEqual([0.0007, -0.0029, 0.0038]);
    expect(pcChange(null, [1, 2, 3])).toBeNull();
    expect(pcChange([1, 2], [1, 2, 3])).toBeNull();
  });
  it('writes the contrast with one decimal', () => {
    expect(contrastText(0.7)).toBe('0.7');
    expect(contrastText(0.84)).toBe('0.8');
    expect(contrastText(null)).toBeNull();
  });
});

describe('followerQuestion', () => {
  const one = { dataset: 'b', per_pixel: false, map_kind: null };
  const block = { calibration: { id: 'a', name: 'A', version: 1, latest: 1 }, diverged: true, followers: [one] };
  it('is due only when the file left a calibration that other files are still on', () => {
    expect(followerQuestion(block)).toEqual({
      calibration: block.calibration, followers: ['b'], maps: { grid: [], header: [] } });
    expect(followerQuestion({ ...block, diverged: false })).toBeNull();
    expect(followerQuestion({ ...block, followers: [] })).toBeNull();
    expect(followerQuestion({ ...block, calibration: null })).toBeNull();
    expect(followerQuestion(undefined)).toBeNull();
  });
  it('names the followers that hold a map by its kind: a grid map is replaced, a header map is moved', () => {
    const got = followerQuestion({ ...block, followers: [
      one,
      { dataset: 'g', per_pixel: true, map_kind: 'grid' },
      { dataset: 'h', per_pixel: true, map_kind: 'header' },
      { dataset: 'u', per_pixel: true, map_kind: null },          // a kind it does not know is told as the milder one
    ] });
    expect(got.followers).toEqual(['b', 'g', 'h', 'u']);
    expect(got.maps).toEqual({ grid: ['g'], header: ['h', 'u'] });
  });
  it('still reads a follower that is a plain name', () => {
    expect(followerQuestion({ ...block, followers: ['b'] }).followers).toEqual(['b']);
  });
});

describe('olderResults', () => {
  it('keeps the results computed with an older version only', () => {
    expect(olderResults([{ result_id: 'a', state: 'older' }, { result_id: 'b', state: 'current' }])).toEqual([{ result_id: 'a', state: 'older' }]);
    expect(olderResults(undefined)).toEqual([]);
  });
});

describe('applyRows', () => {
  const base = { has_pc: true, pc: [0.5, 0.4, 0.6], ref: null, unknown: [], unverified: false, on_this_version: false, path: '/x',
    settings: { file: { working_distance_mm: 20 } } };
  const rows = applyRows([
    { ...base, dataset: 'origin', match: true, differences: [] },
    { ...base, dataset: 'fits', match: true, differences: [] },
    { ...base, dataset: 'same', match: true, differences: [], on_this_version: true },
    { ...base, dataset: 'far', match: false, differences: [{ key: 'working_distance_mm', file: 15, calibration: 20 }], settings: { file: { working_distance_mm: 15 } } },
    { ...base, dataset: 'old', match: false, unverified: true, unknown: ['working_distance_mm'], settings: { file: { working_distance_mm: null } } },
  ], 'origin');
  it('leaves the file the PC is from out, ticks what fits and would change, and lists the rest unticked with the reason', () => {
    expect(rows.fit.map((r) => [r.dataset, r.ticked, r.why])).toEqual([['fits', true, ''], ['same', false, 'sameVersion']]);
    expect(rows.others.map((r) => [r.dataset, r.kind, r.ticked, r.why])).toEqual([['far', 'differs', false, 'differs'], ['old', 'unstated', false, 'unstated']]);
  });
  it('says the working distance of each row and whether it is the same', () => {
    expect(rows.fit[0]).toMatchObject({ wd: '20 mm', wdCheck: 'same' });
    expect(rows.others[0]).toMatchObject({ wd: '15 mm', wdCheck: 'different' });
    expect(rows.others[1]).toMatchObject({ wd: null, wdCheck: 'notStated' });
  });
  it('is empty for no candidates', () => {
    expect(applyRows(undefined, 'x')).toEqual({ fit: [], others: [] });
  });
});

describe('copyKeptOf', () => {
  const answer = (extra = {}) => ({ dataset: 'root', status: 'applied', pc_map_actions: { root: 'single' }, ...extra });
  it('names the open copy when the apply went to its file and did not write the copy', () => {
    expect(copyKeptOf(answer(), 'copy')).toBe('copy');
  });
  it('is null when the copy was written too, or when the open dataset is the file itself', () => {
    expect(copyKeptOf(answer({ pc_map_actions: { root: 'single', copy: 'single' } }), 'copy')).toBeNull();
    expect(copyKeptOf(answer(), 'root')).toBeNull();
  });
  it('is null for an answer that says nothing of the datasets written', () => {
    expect(copyKeptOf(null, 'copy')).toBeNull();
    expect(copyKeptOf({ status: 'applied' }, 'copy')).toBeNull();
  });
  it('treats an answer without a list of the datasets written as: the copy was not written', () => {
    expect(copyKeptOf(answer({ pc_map_actions: undefined }), 'copy')).toBe('copy');
  });
});

describe('otherAspect and the aspect of a difference line', () => {
  it('is true for another height-to-width ratio, false for the same ratio at another size', () => {
    expect(otherAspect([128, 156], [156, 128])).toBe(true);
    expect(otherAspect([480, 640], [240, 320])).toBe(false);          // binning 2: 0.75 and 0.75
    expect(otherAspect([480, 640], [480, 640])).toBe(false);
    expect(otherAspect([1024, 1244], [128, 155])).toBe(false);         // rounded to whole pixels: 0.8231 and 0.8258
    expect(otherAspect([480, 640], [480, 600])).toBe(true);            // 0.75 and 0.8
  });
  it('says nothing for a shape that is not two positive numbers', () => {
    expect(otherAspect(null, [128, 156])).toBe(false);
    expect(otherAspect([128, 156], undefined)).toBe(false);
    expect(otherAspect([0, 156], [128, 156])).toBe(false);
    expect(otherAspect([128, 'x'], [128, 156])).toBe(false);
  });
  it('marks only the pattern shape difference line', () => {
    const lines = differenceLines({ differences: [
      { key: 'pattern_shape', file: [128, 156], calibration: [156, 128] },
      { key: 'working_distance_mm', file: 15, calibration: 20 },
    ] });
    expect(lines.map((l) => l.aspect)).toEqual([true, false]);
  });
});
