import { describe, it, expect, vi } from 'vitest';
import {
  isApexCandidate, measurementTitle, scanEndDate, stepParts, patternSize, rowOf, rowContext,
  defaultSelection, buildLoadJobs, summaryOf, refusalOf, candidateAsChoice, pairingDroppedMessage,
  stoppedRun, routeLoad, mergeFolderSearch, loadedAs, reopenPath, singleFlight, runLoads, sidebarLabels,
  announcePairingDropped, activeApexEntry, statusFileName, fileNameSafe, exportStem,
  sizeOf, sizeMismatch,
} from './apexLoad';

// A measurement as GET /api/ebsd/inspect lists it (the fields the dialog reads). Values are made up.
const m = (n, over = {}) => ({
  group: `/Area 1/OIM Map ${n}`, id: `guid-${n}`, area: 'Area 1', name: `OIM Map ${n}`, hex: true,
  n_rows: 3, n_cols: 5, n_points: 15, step_x: 0.3, step_y: 0.2598, pc: [0.5, 0.4, 0.6],
  loadable: false, problem: 'apexNoPatternFile', measurement_problem: null, pattern_files: [],
  measured_at: null, measured_at_kind: null, ...over,
});
const file = (name, over = {}) => ({
  path: `/d/${name}`, name, n_patterns: 15, pattern_shape: [128, 156], pairing: 'guid', count: 'fits',
  matches_grid: true, problem: null, names_measurement: 'guid-1', names_group: '/Area 1/OIM Map 1', ...over,
});
const ready = (n, name = `m${n}.up1`) => m(n, { loadable: true, problem: null, pattern_files: [file(name)] });

describe('isApexCandidate', () => {
  it('asks about HDF5 files only', () => {
    for (const p of ['a.h5', 'C:\\d\\a.HDF5', ' x.h5 ']) expect(isApexCandidate(p)).toBe(true);
    for (const p of ['a.h5oina', 'a.up1', 'Pattern.dat', '', null, undefined]) expect(isApexCandidate(p)).toBe(false);
  });
});

describe('measurementTitle', () => {
  it('is area and map, or whatever exists for a group that could not be read', () => {
    expect(measurementTitle(m(1))).toBe('Area 1 / OIM Map 1');
    expect(measurementTitle({ group: '/Area 2/OIM Map 9', area: 'Area 2', name: null })).toBe('Area 2');
    expect(measurementTitle({ group: '/Area 2/OIM Map 9' })).toBe('/Area 2/OIM Map 9');
  });
});

describe('scanEndDate', () => {
  it('shows the date the scan ended, taken from the text and never shifted by a time zone', () => {
    expect(scanEndDate({ measured_at: '2026-03-14T10:42:11', measured_at_kind: 'scan_end' })).toBe('2026-03-14');
    // 23:59 must stay on the 14th: no Date object, no conversion to the viewer's zone.
    expect(scanEndDate({ measured_at: '2026-03-14T23:59:59', measured_at_kind: 'scan_end' })).toBe('2026-03-14');
  });
  it('shows nothing for anything but the end of the scan, and nothing for no value', () => {
    expect(scanEndDate({ measured_at: '2026-03-14T10:42:11', measured_at_kind: 'scan_start' })).toBeNull();
    expect(scanEndDate({ measured_at: null, measured_at_kind: null })).toBeNull();
    expect(scanEndDate({ measured_at: 'yesterday', measured_at_kind: 'scan_end' })).toBeNull();
    expect(scanEndDate(null)).toBeNull();
  });
});

describe('stepParts', () => {
  it('gives the row spacing where it differs from the step (hexagonal: 0.3 and 0.26)', () => {
    expect(stepParts(m(1))).toEqual({ step: 0.3, rowStep: 0.26 });
  });
  it('gives no row spacing for equal steps, and the second step of a square scan that has one', () => {
    expect(stepParts({ step_x: 1, step_y: 1 })).toEqual({ step: 1, rowStep: null });
    expect(stepParts({ step_x: 1, step_y: 1.2 })).toEqual({ step: 1, rowStep: 1.2 });
  });
  it('gives no step for a step that is not a positive number', () => {
    expect(stepParts({ step_x: 0, step_y: 0 })).toEqual({ step: null, rowStep: null });
    expect(stepParts({ step_x: null, step_y: null })).toEqual({ step: null, rowStep: null });
  });
});

describe('patternSize', () => {
  it('is width x height from the [rows, cols] of the pattern file', () => {
    expect(patternSize({ pattern_shape: [128, 156] })).toEqual({ width: 156, height: 128 });
    expect(patternSize({ pattern_shape: null })).toBeNull();
    expect(patternSize(null)).toBeNull();
  });
});

describe('rowOf: where a measurement stands', () => {
  it('is ready when a file that names it fits, and loadable is what the backend said', () => {
    const row = rowOf(ready(1), {});
    expect(row).toMatchObject({ state: 'ready', loadable: true, extra: 0 });
    expect(row.file.name).toBe('m1.up1');
    // The backend's verdict, not the count of fitting files, decides.
    expect(rowOf({ ...ready(1), loadable: false }, {}).loadable).toBe(false);
  });

  it('says how many more files name the measurement and never picks silently among them', () => {
    const two = m(1, { loadable: true, problem: null, pattern_files: [file('a.up1'), file('b.up1')] });
    const row = rowOf(two, {});
    expect(row.file.name).toBe('a.up1');
    expect(row.extra).toBe(1);
    expect(row.others.map((f) => f.name)).toEqual(['b.up1']);
  });

  it('is wrong when files name it and none fits, with the first file and its code', () => {
    const bad = file('w.up1', { matches_grid: false, problem: 'apexPatternCountMismatch', count: 'too_few', n_patterns: 9 });
    expect(rowOf(m(1, { pattern_files: [bad], problem: 'apexPatternCountMismatch' }), {})).toMatchObject({
      state: 'wrong', loadable: false, problem: 'apexPatternCountMismatch' });
  });

  it('is missing when nothing names it, and folder when the folder could not be listed', () => {
    expect(rowOf(m(1), {})).toMatchObject({ state: 'missing', loadable: false });
    expect(rowOf(m(1, { problem: 'apexFolderUnreadable' }), {}).state).toBe('folder');
  });

  it('is blocked, whatever file it has, when the measurement itself cannot be opened', () => {
    const row = rowOf(m(1, { measurement_problem: 'apexGridMismatch', problem: 'apexGridMismatch',
      pattern_files: [file('a.up1')] }), { chosen: { '/Area 1/OIM Map 1': { state: 'ok', path: '/d/c.up1', name: 'c.up1',
      entry: file('c.up1') } } });
    expect(row).toMatchObject({ state: 'blocked', loadable: false, problem: 'apexGridMismatch' });
  });

  it('trusts a chosen file only once the backend has described it as fitting', () => {
    const base = m(1);
    const key = base.group;
    const entry = file('c.up1', { pairing: 'count', names_measurement: null });
    const attempt = (state, extra = {}) => ({ [key]: { state, path: '/d/c.up1', name: 'c.up1', ...extra } });
    // an attempt (being checked, refused, failed) never makes the row openable
    for (const state of ['checking', 'refused', 'error']) {
      const row = rowOf(base, rowContext({}, [], attempt(state, { entry })));
      expect(row).toMatchObject({ state: 'missing', loadable: false, attempt: { state } });
    }
    // 'ok' without a fitting entry is not proof either
    expect(rowOf(base, rowContext({ [key]: { state: 'ok', path: '/d/c.up1', name: 'c.up1',
      entry: { ...entry, matches_grid: false } } }, [])).loadable).toBe(false);
    expect(rowOf(base, rowContext({ [key]: { state: 'ok', path: '/d/c.up1', name: 'c.up1', entry } }, [])))
      .toMatchObject({ state: 'chosen', loadable: true });
  });

  it('keeps a file that was found when an attempt with another file is refused', () => {
    const found = ready(1);
    const refused = { [found.group]: { state: 'refused', path: '/d/x.up1', name: 'x.up1',
      entry: file('x.up1', { matches_grid: false, problem: 'apexPatternCountMismatch' }) } };
    const row = rowOf(found, rowContext({}, [], refused));
    expect(row).toMatchObject({ state: 'ready', loadable: true, attempt: { state: 'refused' } });
  });

  it('keeps a file the user chose when a later attempt is refused', () => {
    const base = m(1);
    const accepted = { [base.group]: { state: 'ok', path: '/d/c.up1', name: 'c.up1',
      entry: file('c.up1', { pairing: 'count' }) } };
    const later = { [base.group]: { state: 'refused', path: '/d/x.up1', name: 'x.up1', entry: file('x.up1') } };
    expect(rowOf(base, rowContext(accepted, [], later))).toMatchObject({
      state: 'chosen', loadable: true, attempt: { state: 'refused' } });
  });

  it('says which row has already taken a file of the folder that carries no ID', () => {
    const loose = { path: '/d/loose.up1', name: 'loose.up1', n_patterns: 15, fits: ['guid-1', 'guid-2'] };
    const taken = { [m(1).group]: candidateAsChoice(loose) };
    const other = rowOf(m(2), rowContext(taken, [loose]));
    expect(other.candidates).toEqual([{ ...loose, usedFor: '/Area 1/OIM Map 1' }]);
    // the row that took it does not see it as used
    expect(rowOf(m(1), rowContext(taken, [loose])).candidates[0].usedFor).toBeNull();
    // nobody took it
    expect(rowOf(m(2), rowContext({}, [loose])).candidates[0].usedFor).toBeNull();
  });

  it('lists the files of the folder that name no measurement and fit this one', () => {
    const unmatched = [
      { path: '/d/loose.up1', name: 'loose.up1', n_patterns: 15, fits: ['guid-1', 'guid-2'] },
      { path: '/d/other.up1', name: 'other.up1', n_patterns: 40, fits: ['guid-2'] },
    ];
    expect(rowOf(m(1), rowContext({}, unmatched)).candidates.map((c) => c.name)).toEqual(['loose.up1']);
    expect(rowOf(m(3), rowContext({}, unmatched)).candidates).toEqual([]);
  });

  it('keys on the group, never on the id: rows with the same or no id stay apart', () => {
    const a = m(1, { id: null, group: '/Area 1/OIM Map 1' });
    const b = m(2, { id: null, group: '/Area 1/OIM Map 2' });
    const chosen = { [a.group]: candidateAsChoice({ path: '/d/x.up1', name: 'x.up1', n_patterns: 15 }) };
    expect(rowOf(a, rowContext(chosen, [])).state).toBe('chosen');
    expect(rowOf(b, rowContext(chosen, [])).state).toBe('missing');
    expect(rowOf(a, rowContext({}, [{ path: '/d/l.up1', name: 'l.up1', fits: [null] }])).candidates).toEqual([]);
  });
});

describe('defaultSelection: ticked when the FILE holds exactly one measurement, and it can be opened', () => {
  it('ticks the only measurement of a file when it can be opened', () => {
    expect(defaultSelection([ready(1)], {})).toEqual(['/Area 1/OIM Map 1']);
  });
  it('ticks nothing when the only measurement cannot be opened', () => {
    expect(defaultSelection([m(1)], {})).toEqual([]);
  });
  it('ticks nothing when the file holds several, even if only one of them can be opened', () => {
    expect(defaultSelection([ready(1), m(2)], {})).toEqual([]);
    expect(defaultSelection([ready(1), ready(2)], {})).toEqual([]);
  });
  it('ticks nothing for no list', () => {
    expect(defaultSelection(undefined, {})).toEqual([]);
    expect(defaultSelection([], {})).toEqual([]);
  });
});

describe('buildLoadJobs', () => {
  const a = ready(1, 'a.up1');
  const c = ready(3, 'c.up1');
  const b = m(2);

  it('lists the ticked measurements in the order they are listed, each with its pairing', () => {
    expect(buildLoadJobs('/d/s.h5', [a, b, c], ['/Area 1/OIM Map 3', '/Area 1/OIM Map 1'], {})).toEqual([
      { path: '/d/a.up1', label: 'Area 1 / OIM Map 1', apex: { h5_path: '/d/s.h5', measurement_id: 'guid-1' } },
      { path: '/d/c.up1', label: 'Area 1 / OIM Map 3', apex: { h5_path: '/d/s.h5', measurement_id: 'guid-3' } },
    ]);
  });

  it('leaves out a ticked measurement that cannot be opened', () => {
    expect(buildLoadJobs('/d/s.h5', [a, b], [a.group, b.group], {}).map((j) => j.apex.measurement_id)).toEqual(['guid-1']);
  });

  it('leaves out a ticked measurement whose pattern file was found but does not fit', () => {
    const wrong = m(4, { problem: 'apexPatternCountMismatch', pattern_files: [file('t.up1', {
      n_patterns: 9, count: 'too_few', matches_grid: false, problem: 'apexPatternCountMismatch' })] });
    expect(rowOf(wrong, {})).toMatchObject({ state: 'wrong', loadable: false, file: { path: '/d/t.up1' } });
    expect(buildLoadJobs('/d/s.h5', [a, wrong], [a.group, wrong.group], {}).map((j) => j.path)).toEqual(['/d/a.up1']);
  });

  it('uses the file the user chose when there was none', () => {
    const chosen = { [b.group]: { state: 'ok', path: '/e/x.up1', name: 'x.up1', entry: file('x.up1', { path: '/e/x.up1' }) } };
    expect(buildLoadJobs('/d/s.h5', [b], [b.group], rowContext(chosen, []))).toEqual([
      { path: '/e/x.up1', label: 'Area 1 / OIM Map 2', apex: { h5_path: '/d/s.h5', measurement_id: 'guid-2' } }]);
  });
});

describe('summaryOf', () => {
  it('counts ready, needing a pattern file, a pattern file that does not fit, and not openable at all', () => {
    const blocked = m(3, { measurement_problem: 'apexUnknownGrid', problem: 'apexUnknownGrid' });
    const wrong = m(4, { problem: 'apexPatternCountMismatch', pattern_files: [file('w.up1', {
      n_patterns: 9, count: 'too_few', matches_grid: false, problem: 'apexPatternCountMismatch' })] });
    expect(summaryOf([ready(1), m(2), blocked], {}))
      .toEqual({ total: 3, ready: 1, needFile: 1, wrongFile: 0, blocked: 1 });
    expect(summaryOf([ready(1), m(2), blocked, wrong], {}))
      .toEqual({ total: 4, ready: 1, needFile: 1, wrongFile: 1, blocked: 1 });
    expect(summaryOf([], {})).toEqual({ total: 0, ready: 0, needFile: 0, wrongFile: 0, blocked: 0 });
  });

  it('counts a folder that could not be listed as needing a file, not as one that does not fit', () => {
    const folder = m(5, { problem: 'apexFolderUnreadable' });
    expect(summaryOf([folder], {})).toMatchObject({ needFile: 1, wrongFile: 0 });
  });
});

describe('refusalOf: why a pattern file does not do', () => {
  const grid = m(1, { n_rows: 50, n_cols: 40 });
  it('says too few with the numbers (a file with too few patterns may be incomplete)', () => {
    expect(refusalOf(file('w.up1', { matches_grid: false, problem: 'apexPatternCountMismatch', count: 'too_few',
      n_patterns: 1856 }), grid)).toEqual({ kind: 'tooFew', have: 1856, need: 2000, rows: 50, cols: 40 });
  });
  it('says too many, which is not "incomplete"', () => {
    expect(refusalOf(file('w.up1', { matches_grid: false, problem: 'apexPatternCountMismatch', count: 'too_many',
      n_patterns: 2400 }), grid)).toEqual({ kind: 'tooMany', have: 2400, need: 2000, rows: 50, cols: 40 });
  });
  it('names the measurement a file belongs to when it is one of this file\'s', () => {
    const other = m(2);
    const e = file('w.up1', { matches_grid: false, problem: 'apexGuidMismatch', names_group: other.group });
    expect(refusalOf(e, grid, [grid, other])).toEqual({ kind: 'otherMeasurement', other: 'Area 1 / OIM Map 2' });
    expect(refusalOf({ ...e, names_group: null }, grid, [grid, other])).toEqual({ kind: 'otherMeasurement', other: null });
  });
  it('hands any other code on, and a count mismatch without numbers too', () => {
    expect(refusalOf(file('w.up1', { matches_grid: false, problem: 'apexGridMismatch' }), grid))
      .toEqual({ kind: 'code', code: 'apexGridMismatch' });
    expect(refusalOf(file('w.up1', { matches_grid: false, problem: 'apexPatternCountMismatch', count: null,
      n_patterns: null }), grid)).toEqual({ kind: 'code', code: 'apexPatternCountMismatch' });
  });
  it('is null for a file without a problem', () => {
    expect(refusalOf(file('a.up1'), grid)).toBeNull();
    expect(refusalOf(null, grid)).toBeNull();
  });
});

describe('candidateAsChoice', () => {
  it('makes a pairing by count out of a file of the folder that names no measurement', () => {
    const c = candidateAsChoice({ path: '/d/l.up1', name: 'l.up1', n_patterns: 15, fits: ['guid-1'] });
    expect(c).toMatchObject({ state: 'ok', path: '/d/l.up1', name: 'l.up1' });
    expect(c.entry).toMatchObject({ pairing: 'count', count: 'fits', matches_grid: true, problem: null });
  });
});

describe('pairingDroppedMessage', () => {
  const tt = (key, opts) => (key === 'apexDialog.pairingDropped' ? `dropped(${opts.reason})` : `text(${key})`);
  it('is a sentence only when the stored pairing was not used, naming the reason', () => {
    expect(pairingDroppedMessage({ apex_pairing_dropped: true, apex_pairing_dropped_reason: 'apexUnreadable' }, tt))
      .toBe('dropped(text(errors.apexUnreadable))');
    expect(pairingDroppedMessage({ apex_pairing_dropped: false }, tt)).toBeNull();
    expect(pairingDroppedMessage({}, tt)).toBeNull();
    expect(pairingDroppedMessage(undefined, tt)).toBeNull();
  });
});

describe('stoppedRun', () => {
  const jobs = [{ label: 'A' }, { label: 'B' }, { label: 'C' }];
  it('names the one that failed and those that stayed closed', () => {
    expect(stoppedRun(jobs, 1)).toEqual({ failed: 'B', notOpened: ['C'] });
    expect(stoppedRun(jobs, 0)).toEqual({ failed: 'A', notOpened: ['B', 'C'] });
    expect(stoppedRun(jobs, 2)).toEqual({ failed: 'C', notOpened: [] });
  });
  it('is null when everything was opened', () => {
    expect(stoppedRun(jobs, 3)).toBeNull();
  });
});

describe('routeLoad: where a picked path leads', () => {
  const inspect = (data) => vi.fn(async () => ({ data }));

  it('loads anything that is not an .h5 without asking the backend', async () => {
    const ask = inspect({ kind: 'apex' });
    for (const p of ['/d/a.up1', '/d/Pattern.dat', '/d/Setting.txt', '/d/a.ebsp']) {
      expect(await routeLoad(p, ask)).toEqual({ action: 'load' });
    }
    expect(ask).not.toHaveBeenCalled();
  });

  it('opens the dialog for an APEX metadata file, with the whole answer', async () => {
    const answer = { kind: 'apex', measurements: [{ group: 'g' }] };
    const ask = inspect(answer);
    expect(await routeLoad(' /d/s.h5 ', ask)).toEqual({
      action: 'dialog', kind: 'apex', inspection: answer, dialog: { h5Path: '/d/s.h5', inspection: answer } });
    expect(ask).toHaveBeenCalledWith('/d/s.h5');
  });

  it('loads an .h5 that is not APEX the ordinary way', async () => {
    expect(await routeLoad('/d/a.h5', inspect({ kind: 'plain' }))).toEqual({ action: 'load' });
    expect(await routeLoad('/d/a.h5', inspect({}))).toEqual({ action: 'load' });
  });

  it('says why an .h5 cannot be read, and does not offer the dialog', async () => {
    expect(await routeLoad('/d/a.h5', inspect({ kind: 'unreadable', problem: 'apexFileUnreadable',
      problem_detail: 'a.h5 is held open' }))).toEqual({
      action: 'unreadable', problem: 'apexFileUnreadable', detail: 'a.h5 is held open' });
  });

  it('loads the ordinary way when the question fails, so the load reports its own error', async () => {
    const fail = vi.fn(async () => { throw new Error('400 apexFileNotFound'); });
    expect(await routeLoad('/d/missing.h5', fail)).toEqual({ action: 'load' });
    expect(fail).toHaveBeenCalledTimes(1);
  });
});

describe('mergeFolderSearch: a folder search only ever adds', () => {
  const alt = (name, over = {}) => file(name, { path: `/e/more/${name}`, ...over });
  const listing = (...rows) => ({ measurements: rows, unmatched: [] });
  const answer = (rows, over = {}) => ({ measurements: rows, unmatched_pattern_files: [], folder: '/e/more', ...over });

  it('keeps the file a row already has and offers the new folder\'s files as alternatives', () => {
    const a = ready(1, 'a.up1');
    const fresh = m(1, { loadable: true, problem: null, pattern_files: [alt('a_new.up1')] });
    const merged = mergeFolderSearch(listing(a), answer([fresh]), {});
    const row = rowOf(merged.measurements[0], rowContext({}, merged.unmatched));
    expect(row).toMatchObject({ state: 'ready', loadable: true });
    expect(row.file.path).toBe('/d/a.up1');
    expect(row.alternatives.map((f) => f.path)).toEqual(['/e/more/a_new.up1']);
    expect(merged.measurements[0].alternativesFolder).toBe('/e/more');
  });

  it('offers a loose file of the new folder that fits as an alternative as well', () => {
    const a = ready(1, 'a.up1');
    const loose = { path: '/e/more/l.up1', name: 'l.up1', n_patterns: 15, fits: ['guid-1'] };
    const merged = mergeFolderSearch(listing(a), answer([m(1)], { unmatched_pattern_files: [loose] }), {});
    expect(merged.measurements[0].alternatives.map((f) => f.path)).toEqual(['/e/more/l.up1']);
    expect(merged.unmatched.map((u) => u.path)).toEqual(['/e/more/l.up1']);
  });

  it('does not offer the file in use as an alternative, nor a file twice', () => {
    const a = ready(1, 'a.up1');
    const same = { ...file('a.up1'), path: '/d/a.up1' };
    const merged = mergeFolderSearch(listing(a), answer([m(1, { pattern_files: [same, alt('b.up1'), alt('b.up1')] })]), {});
    expect(merged.measurements[0].alternatives.map((f) => f.path)).toEqual(['/e/more/b.up1']);
  });

  it('fills a row that had no file with what the new folder found', () => {
    const fresh = m(2, { loadable: true, problem: null, pattern_files: [alt('b.up1')] });
    const merged = mergeFolderSearch(listing(m(2)), answer([fresh]), {});
    const row = rowOf(merged.measurements[0], rowContext({}, merged.unmatched));
    expect(row).toMatchObject({ state: 'ready', loadable: true });
    expect(row.file.path).toBe('/e/more/b.up1');
    expect(merged.measurements[0].alternatives).toBeUndefined();
  });

  it('keeps a file the user chose, whatever the new folder holds', () => {
    const b = m(2);
    const chosen = { [b.group]: candidateAsChoice({ path: '/d/loose.up1', name: 'loose.up1', n_patterns: 15 }) };
    const fresh = m(2, { loadable: true, problem: null, pattern_files: [alt('b.up1')] });
    const merged = mergeFolderSearch(listing(b), answer([fresh]), chosen);
    const row = rowOf(merged.measurements[0], rowContext(chosen, merged.unmatched));
    expect(row).toMatchObject({ state: 'chosen', loadable: true });
    expect(row.file.path).toBe('/d/loose.up1');
    expect(row.alternatives.map((f) => f.path)).toEqual(['/e/more/b.up1']);
  });

  it('keeps a file that carries the ID but does not fit when the new folder has nothing for the row', () => {
    const wrong = file('w.up1', { matches_grid: false, problem: 'apexPatternCountMismatch', count: 'too_few', n_patterns: 3 });
    const w = m(3, { pattern_files: [wrong], problem: 'apexPatternCountMismatch' });
    const merged = mergeFolderSearch(listing(w), answer([m(3)]), {});
    expect(rowOf(merged.measurements[0], rowContext({}, merged.unmatched)).state).toBe('wrong');
  });

  it('keeps a row the new folder does not list, and never turns an openable row into one that is not', () => {
    const rows = [ready(1), m(2), ready(3)];
    const merged = mergeFolderSearch(listing(...rows), answer([m(1), m(2)]), {});
    expect(merged.measurements.map((x) => x.group)).toEqual(rows.map((x) => x.group));
    const ctx = rowContext({}, merged.unmatched);
    expect(merged.measurements.map((x) => rowOf(x, ctx).loadable)).toEqual([true, false, true]);
  });

  it('keeps the loose files of the first folder next to those of the second', () => {
    const first = { measurements: [m(2)], unmatched: [{ path: '/d/l1.up1', name: 'l1.up1', fits: ['guid-2'] }] };
    const second = { path: '/e/more/l2.up1', name: 'l2.up1', fits: ['guid-2'] };
    const merged = mergeFolderSearch(first, answer([m(2)], { unmatched_pattern_files: [second] }), {});
    expect(merged.unmatched.map((u) => u.path)).toEqual(['/d/l1.up1', '/e/more/l2.up1']);
  });

  it('a file of the new folder that carries the row\'s ID keeps its pairing by ID when used', () => {
    const c = candidateAsChoice(alt('b.up1'));
    expect(c.entry).toMatchObject({ pairing: 'guid', count: 'fits', matches_grid: true });
    expect(candidateAsChoice({ path: '/e/l.up1', name: 'l.up1' }).entry.pairing).toBe('count');
  });
});

describe('loadedAs and reopenPath: reopening goes through the measurement dialog again', () => {
  const apex = { h5_path: '/d/series/session.h5', measurement_id: 'guid-1' };
  it('records the metadata file, not the pattern file, for a dialog load', () => {
    expect(loadedAs('/d/series/map_a.up1', { apex })).toBe('/d/series/session.h5');
    expect(loadedAs('/d/a.h5oina', {})).toBe('/d/a.h5oina');
    expect(loadedAs('/d/a.h5oina', undefined)).toBe('/d/a.h5oina');
    expect(loadedAs('/d/series/map_a.up1', { apex: null })).toBe('/d/series/map_a.up1');
  });

  it('Recent -> reopen: what was recorded leads to the dialog, not to a load without metadata', async () => {
    const recent = [loadedAs('/d/series/map_a.up1', { apex })];
    const inspect = vi.fn(async () => ({ data: { kind: 'apex', measurements: [] } }));
    expect((await routeLoad(recent[0], inspect)).action).toBe('dialog');
    // the pattern file itself would have loaded the ordinary way (no inspect, no pairing)
    const bare = vi.fn();
    expect((await routeLoad('/d/series/map_a.up1', bare)).action).toBe('load');
    expect(bare).not.toHaveBeenCalled();
  });

  it('asks about the metadata file when a pattern file of a paired entry is opened again', () => {
    const files = [{ path: '/d/series/map_a.up1', name: 'a', apex }, { path: '/d/c.h5oina', name: 'c' }];
    expect(reopenPath('/d/series/map_a.up1', files)).toBe('/d/series/session.h5');
    expect(reopenPath('D:\\series\\x.up1', [{ path: 'D:/series/x.up1', apex }])).toBe('/d/series/session.h5');
    expect(reopenPath('/d/c.h5oina', files)).toBe('/d/c.h5oina');
    expect(reopenPath('/d/other.up1', files)).toBe('/d/other.up1');
    expect(reopenPath('/d/other.up1', undefined)).toBe('/d/other.up1');
  });
});

describe('singleFlight: one question at a time', () => {
  it('answers a second call that comes while the first is pending with the busy marker, once', async () => {
    let finish;
    const slow = vi.fn(() => new Promise((resolve) => { finish = resolve; }));
    const once = singleFlight(slow, 'busy');
    const first = once('a');
    expect(await once('b')).toBe('busy');
    expect(slow).toHaveBeenCalledTimes(1);
    finish('answer');
    expect(await first).toBe('answer');
  });
  it('lets the next call through when the first has finished, also after a failure', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce('second');
    const once = singleFlight(fn, 'busy');
    await expect(once()).rejects.toThrow('boom');
    expect(await once()).toBe('second');
  });
});

describe('runLoads: the dialog\'s run of loads', () => {
  const jobs = [{ label: 'A' }, { label: 'B' }, { label: 'C' }];
  it('loads in order, announces each, and reports everything done', async () => {
    const order = [];
    const started = [];
    const out = await runLoads(jobs, async (j) => { order.push(j.label); return true; }, (i, j) => started.push([i, j.label]));
    expect(order).toEqual(['A', 'B', 'C']);
    expect(started).toEqual([[0, 'A'], [1, 'B'], [2, 'C']]);
    expect(out).toEqual({ done: 3, stopped: null });
  });
  it('stops at the first failure: the failed one and those after it stayed closed', async () => {
    const order = [];
    const out = await runLoads(jobs, async (j) => { order.push(j.label); return j.label !== 'B'; });
    expect(order).toEqual(['A', 'B']);                       // C was never tried
    expect(out).toEqual({ done: 1, stopped: { failed: 'B', notOpened: ['C'] } });
  });
  it('names the first one when it fails at once, and nothing when the last one fails', async () => {
    expect((await runLoads(jobs, async () => false)).stopped).toEqual({ failed: 'A', notOpened: ['B', 'C'] });
    expect((await runLoads(jobs, async (j) => j.label !== 'C')).stopped).toEqual({ failed: 'C', notOpened: [] });
  });
});

describe('sidebarLabels: the labels of the loaded files', () => {
  const apex = (n, name) => ({ path: `/d/${name}.up1`, name, apex: { measurement_name: `Area 1 / OIM Map ${n}` } });
  // stand-in for disambiguateNames: marks what it was given, in order
  const shorten = (names) => names.map((n) => `short(${n})`);

  it('names a measurement by its measurement and shortens only the other files, each at its own place', () => {
    const files = [{ path: '/d/x.h5oina', name: 'scan X' }, apex(1, 'a'), { path: '/d/y.h5oina', name: 'scan Y' }, apex(2, 'b')];
    expect(sidebarLabels(files, shorten)).toEqual([
      'short(scan X)', 'Area 1 / OIM Map 1 (a)', 'short(scan Y)', 'Area 1 / OIM Map 2 (b)']);
  });
  it('keeps the whole name of a file opened as another variant, so the variant stays visible', () => {
    const variant = { path: '/d/x.h5oina', name: 'scan X · unprocessed', variant: { measurement: '1', pattern_source: 'unprocessed' } };
    const files = [{ path: '/d/y.h5oina', name: 'scan Y' }, variant, { path: '/d/z.h5oina', name: 'scan Z' }];
    expect(sidebarLabels(files, shorten)).toEqual(['short(scan Y)', 'scan X · unprocessed', 'short(scan Z)']);
    expect(sidebarLabels(files, shorten, (k) => `T(${k})`)[1]).toBe('scan X · T(loadDialog.unprocessedTag)');
  });
  it('handles a list without measurements and an empty list', () => {
    expect(sidebarLabels([{ name: 'p' }, { name: 'q' }], shorten)).toEqual(['short(p)', 'short(q)']);
    expect(sidebarLabels([], shorten)).toEqual([]);
    expect(sidebarLabels(undefined, shorten)).toEqual([]);
  });
});

describe('announcePairingDropped', () => {
  const tt = (key, opts) => (key === 'apexDialog.pairingDropped' ? `dropped(${opts.reason})` : `text(${key})`);
  it('tells both the log and the user when the pairing was dropped', () => {
    const log = vi.fn();
    const warn = vi.fn();
    const out = announcePairingDropped({ apex_pairing_dropped: true, apex_pairing_dropped_reason: 'apexNoMeasurement' }, tt, { log, warn });
    expect(out).toBe('dropped(text(errors.apexNoMeasurement))');
    expect(log).toHaveBeenCalledWith(out);
    expect(warn).toHaveBeenCalledWith(out);
  });
  it('says nothing when the pairing was used, and needs no log sink', () => {
    const warn = vi.fn();
    expect(announcePairingDropped({ apex_pairing_dropped: false }, tt, { warn })).toBeNull();
    expect(warn).not.toHaveBeenCalled();
    expect(announcePairingDropped({ apex_pairing_dropped: true, apex_pairing_dropped_reason: 'x' }, tt, { warn })).toBe('dropped(text(errors.x))');
  });
});

describe('the name of the open measurement: status bar and suggested export name', () => {
  const entry = (title, file, active, h5 = 'D:\\Scans\\Session.h5') => ({
    path: `D:\\Scans\\${file}`, name: file.replace(/\.up1$/, ''), active,
    apex: { h5_path: h5, measurement_id: 'x', measurement_name: title },
  });
  const files = [entry('Area 1 / OIM Map 1', 'a.up1', false), entry('Area 2 / OIM Map 1', 'b.up1', true)];
  const box = 'd:/scans/session.h5';

  it('finds the active measurement of the APEX file in the path box, comparing paths the way the registry does', () => {
    expect(activeApexEntry(files, box).name).toBe('b');
    expect(activeApexEntry(files, ' D:\\Scans\\Session.h5 ').name).toBe('b');
    expect(activeApexEntry(files, '/x/typed.h5')).toBeNull();
    expect(activeApexEntry(files, '')).toBeNull();
    expect(activeApexEntry([entry('Area 1 / OIM Map 1', 'a.up1', false)], box)).toBeNull();
    expect(activeApexEntry([{ path: 'D:\\Scans\\Session.h5', name: 'Session', active: true }], box)).toBeNull();
  });

  it('names the measurement as the list of loaded files does, with the pattern file\'s path as tooltip', () => {
    expect(statusFileName(box, files)).toEqual({ text: 'Area 2 / OIM Map 1 (b)', title: 'D:\\Scans\\b.up1' });
  });

  it('names an ordinary file by its file name, as before', () => {
    expect(statusFileName('/data/scans/sample_01.h5oina', [{ path: '/data/scans/sample_01.h5oina', name: 'sample_01', active: true }]))
      .toEqual({ text: 'sample_01.h5oina', title: '/data/scans/sample_01.h5oina' });
    expect(statusFileName('C:\\d\\x.h5', [])).toEqual({ text: 'x.h5', title: 'C:\\d\\x.h5' });
    expect(statusFileName('', undefined)).toEqual({ text: '', title: '' });
  });

  it('turns a measurement name into something Windows accepts as a file name', () => {
    expect(fileNameSafe('Area 1 / OIM Map 1')).toBe('Area 1_OIM Map 1');
    expect(fileNameSafe('a<b>c:d"e/f\\g|h?i*j')).toBe('a_b_c_d_e_f_g_h_i_j');
    expect(fileNameSafe('Map 1. ')).toBe('Map 1');
    expect(fileNameSafe('tab\there\u0001')).toBe('tabhere');
    expect(fileNameSafe('///')).toBe('_');
    expect(fileNameSafe(null)).toBe('');
  });

  it('suggests <metadata file stem>_<measurement> for a measurement and the plain stem for everything else', () => {
    expect(exportStem(box, null, files)).toBe('session_Area 2_OIM Map 1');
    expect(exportStem('/data/scans/sample_01.h5oina', 'ds', [])).toBe('sample_01');
    expect(exportStem('', 'dataset_a', [])).toBe('dataset_a');
    expect(exportStem('', null, [])).toBe('ebsd');
    expect(exportStem('/x/typed.h5', null, files)).toBe('typed');
  });
});

describe('the measurement\'s own pattern size against a file\'s', () => {
  const sizedM = (n, shape) => m(n, { pattern_shape: shape });

  it('reads [rows, cols] as width x height, or nothing', () => {
    expect(sizeOf({ pattern_shape: [128, 156] })).toEqual({ width: 156, height: 128 });
    for (const bad of [undefined, null, [], [128], [0, 156], [128, -1], ['a', 'b']]) expect(sizeOf({ pattern_shape: bad })).toBeNull();
    expect(sizeOf(null)).toBeNull();
  });

  it('finds a mismatch only when both sizes are known and differ', () => {
    expect(sizeMismatch(sizedM(1, [128, 156]), { pattern_shape: [64, 64] }))
      .toEqual({ have: { width: 64, height: 64 }, need: { width: 156, height: 128 } });
    expect(sizeMismatch(sizedM(1, [128, 156]), { pattern_shape: [128, 156] })).toBeNull();
    expect(sizeMismatch(sizedM(1, [128, 156]), { pattern_shape: [156, 128] })).not.toBeNull();   // transposed is not equal
    expect(sizeMismatch(m(1), { pattern_shape: [64, 64] })).toBeNull();
    expect(sizeMismatch(sizedM(1, [128, 156]), { pattern_shape: null })).toBeNull();
  });

  it('takes a count-only file with another size out of the candidates and lists it as not offered', () => {
    const loose = (name, shape) => ({ path: `/d/${name}`, name, n_patterns: 15, pattern_shape: shape, fits: ['guid-2'],
      names_measurement: null, problem: null });
    const row2 = rowOf(sizedM(2, [128, 156]), rowContext({}, [loose('ok.up1', [128, 156]), loose('big.up1', [64, 64]), loose('unknown.up1', null)]));
    expect(row2.candidates.map((c) => c.name)).toEqual(['ok.up1', 'unknown.up1']);
    expect(row2.sizeRejected.map((c) => c.name)).toEqual(['big.up1']);
    const plain = rowOf(m(2), rowContext({}, [loose('big.up1', [64, 64])]));
    expect(plain.candidates.map((c) => c.name)).toEqual(['big.up1']);
    expect(plain.sizeRejected).toEqual([]);
  });

  it('a folder searched later offers no loose file of another size as an alternative either', () => {
    const own = m(2, { pattern_shape: [128, 156], loadable: true, problem: null, pattern_files: [file('have.up1')] });
    const loose = (name, shape) => ({ path: `/e/${name}`, name, n_patterns: 15, pattern_shape: shape, fits: ['guid-2'],
      names_measurement: null, names_group: null, problem: null });
    const merged = mergeFolderSearch({ measurements: [own], unmatched: [] },
      { folder: '/e', measurements: [{ ...own, pattern_files: [] }],
        unmatched_pattern_files: [loose('same.up1', [128, 156]), loose('other.up1', [64, 64])] }, {});
    expect(rowOf(merged.measurements[0], rowContext({}, merged.unmatched)).alternatives.map((f) => f.name)).toEqual(['same.up1']);
  });

  it('refusalOf words a chosen file of another size as a size mismatch', () => {
    const entry = { name: 'x.up1', problem: 'patternSizeMismatch', pattern_shape: [64, 64] };
    expect(refusalOf(entry, sizedM(1, [128, 156]), [])).toEqual({ kind: 'size',
      have: { width: 64, height: 64 }, need: { width: 156, height: 128 } });
  });
});
