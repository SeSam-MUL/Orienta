// @vitest-environment jsdom
//
// The pure pieces of the generic load dialog: the rows of a `choices` answer, which variant is
// preselected, the job a choice becomes, and the last choice remembered per file. The answer is the
// backend's own (__fixtures__/inspectChoices.json is held against the real route by
// tests/test_inspect_choices.py::test_the_fixture_matches_the_backend). Expected values are written by hand.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import choices from '../__fixtures__/inspectChoices.json';
import {
  choiceRows, defaultChoice, choiceJobs, canonicalPath, variantStorageKey, readRememberedVariant,
  rememberVariant, sourceBits, limitConcurrent,
} from '../loadChoices';

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('choiceRows: one row per measurement, its sources inside', () => {
  const rows = choiceRows(choices);

  it('lists the two measurements of the answer, in its order', () => {
    expect(rows.map((r) => r.name)).toEqual(['1', '2']);
    expect(rows.map((r) => r.sources.map((s) => s.key))).toEqual([['processed', 'unprocessed'], ['processed']]);
  });

  it('carries what the user compares: count, size (width x height), data type, background, loadable', () => {
    const [first] = rows;
    expect(first.nRows).toBe(2);
    expect(first.nCols).toBe(3);
    expect(first.sources[0]).toMatchObject({
      nPatterns: 6, size: { width: 5, height: 4 }, dtype: 'uint8', hasBackground: true, loadable: true, problem: null });
    expect(first.sources[1]).toMatchObject({ dtype: 'int16', hasBackground: true });
    expect(rows[1].sources[0].hasBackground).toBe(false);
  });

  it('has no size for a source that states none, and no row for an answer without measurements', () => {
    const odd = { ...choices, measurements: [{ name: 'x', label: 'x', n_rows: 1, n_cols: 1,
      sources: [{ key: 'single', n_patterns: 1, pattern_shape: null, dtype: 'uint8', has_background: false,
        loadable: true, problem: null }] }] };
    expect(choiceRows(odd)[0].sources[0].size).toBeNull();
    expect(choiceRows({ kind: 'choices' })).toEqual([]);
    expect(choiceRows(null)).toEqual([]);
  });
});

describe('sourceBits: the bit depth line of a source follows the measured data type', () => {
  it('says "16-bit counts" for any 16-bit integer source, whatever the source is called', () => {
    expect(sourceBits({ key: 'unprocessed', dtype: 'int16' })).toEqual({ kind: 'unprocessed' });
    expect(sourceBits({ key: 'unprocessed', dtype: 'uint16' })).toEqual({ kind: 'unprocessed' });
    expect(sourceBits({ key: 'single', dtype: 'uint16' })).toEqual({ kind: 'unprocessed' });
    expect(sourceBits({ key: 'processed', dtype: 'int16' })).toEqual({ kind: 'unprocessed' });
  });
  it('claims a background correction only for 8-bit patterns that are the processed source', () => {
    expect(sourceBits({ key: 'processed', dtype: 'uint8' })).toEqual({ kind: 'processed' });
    // an 8-bit source that is not the processed one gets the plain depth, no claim
    expect(sourceBits({ key: 'unprocessed', dtype: 'uint8' })).toEqual({ kind: 'bits', bits: 8 });
    expect(sourceBits({ key: 'single', dtype: 'uint8' })).toEqual({ kind: 'bits', bits: 8 });
  });
  it('does not guess from the name of the source: its data type decides', () => {
    expect(sourceBits({ key: 'processed', dtype: 'float32' })).toEqual({ kind: 'bits', bits: 32 });
    expect(sourceBits({ key: 'unprocessed', dtype: 'float64' })).toEqual({ kind: 'bits', bits: 64 });
    expect(sourceBits({ key: 'processed', dtype: '' })).toEqual({ kind: 'none' });
    expect(sourceBits({ key: 'unprocessed' })).toEqual({ kind: 'none' });
    expect(sourceBits({ key: 'single', dtype: 'weird' })).toEqual({ kind: 'none' });
    expect(sourceBits(null)).toEqual({ kind: 'none' });
  });
});

describe('defaultChoice: what is preselected', () => {
  it('is the first measurement with its processed patterns when nothing is remembered (today\'s load)', () => {
    expect(defaultChoice(choices, null)).toEqual({ measurement: '1', pattern_source: 'processed', remembered: false });
  });

  it('is the remembered variant when the file still offers it, and says so', () => {
    expect(defaultChoice(choices, { measurement: '1', pattern_source: 'unprocessed' }))
      .toEqual({ measurement: '1', pattern_source: 'unprocessed', remembered: true });
    expect(defaultChoice(choices, { measurement: '2', pattern_source: 'processed' }))
      .toEqual({ measurement: '2', pattern_source: 'processed', remembered: true });
  });

  it('falls back to the default when the remembered variant is gone or cannot be opened', () => {
    const gone = { measurement: '2', pattern_source: 'unprocessed' };
    expect(defaultChoice(choices, gone)).toEqual({ measurement: '1', pattern_source: 'processed', remembered: false });
    const blocked = { ...choices, measurements: choices.measurements.map((m) => (m.name === '1'
      ? { ...m, sources: m.sources.map((s) => (s.key === 'unprocessed' ? { ...s, loadable: false, problem: 'loadNoPatterns' } : s)) }
      : m)) };
    expect(defaultChoice(blocked, { measurement: '1', pattern_source: 'unprocessed' }).remembered).toBe(false);
  });

  it('takes the variant a cropped file was cut from before the plain default, and a memory before that', () => {
    const cropped = { ...choices, suggested: { measurement: '2', pattern_source: 'processed' } };
    expect(defaultChoice(cropped, null)).toEqual({ measurement: '2', pattern_source: 'processed', remembered: false });
    expect(defaultChoice(cropped, { measurement: '1', pattern_source: 'unprocessed' }).pattern_source).toBe('unprocessed');
  });

  it('is null when no variant can be opened', () => {
    const none = { ...choices, measurements: [{ ...choices.measurements[1],
      sources: [{ ...choices.measurements[1].sources[0], loadable: false, problem: 'loadNoPatterns' }] }] };
    expect(defaultChoice(none, null)).toBeNull();
  });
});

describe('choiceJobs: the one load a choice becomes', () => {
  it('is one job that carries the file, the measurement and the pattern source', () => {
    expect(choiceJobs('/d/two.h5oina', choices, { measurement: '1', pattern_source: 'unprocessed' })).toEqual([
      { path: '/d/two.h5oina', label: '1', variant: { measurement: '1', pattern_source: 'unprocessed' } }]);
  });

  it('is no job for a variant the file does not offer or cannot open', () => {
    expect(choiceJobs('/d/two.h5oina', choices, { measurement: '2', pattern_source: 'unprocessed' })).toEqual([]);
    expect(choiceJobs('/d/two.h5oina', choices, null)).toEqual([]);
  });
});

describe('the last choice, remembered per file', () => {
  it('writes under orienta.loadVariant.<canonical path> and reads it back', () => {
    expect(rememberVariant('C:\\Data\\Two.h5oina', { measurement: '1', pattern_source: 'unprocessed' })).toBe(true);
    expect(variantStorageKey('C:\\Data\\Two.h5oina')).toBe('orienta.loadVariant.c:/data/two.h5oina');
    expect(JSON.parse(localStorage.getItem('orienta.loadVariant.c:/data/two.h5oina')))
      .toEqual({ measurement: '1', pattern_source: 'unprocessed' });
    expect(readRememberedVariant('c:/data/TWO.h5oina')).toEqual({ measurement: '1', pattern_source: 'unprocessed' });
  });

  it('knows two spellings of one path as one, and two files as two', () => {
    expect(canonicalPath(' C:\\A\\b.h5 ')).toBe('c:/a/b.h5');
    expect(variantStorageKey('/d/a.h5')).not.toBe(variantStorageKey('/d/b.h5'));
  });

  it('reads nothing for a file never chosen for, and nothing for text that is not a variant', () => {
    expect(readRememberedVariant('/d/never.h5oina')).toBeNull();
    localStorage.setItem(variantStorageKey('/d/bad.h5oina'), '{not json');
    expect(readRememberedVariant('/d/bad.h5oina')).toBeNull();
    localStorage.setItem(variantStorageKey('/d/bad2.h5oina'), JSON.stringify({ measurement: 3 }));
    expect(readRememberedVariant('/d/bad2.h5oina')).toBeNull();
  });

  it('never throws when the storage is blocked or full', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('full'); });
    expect(readRememberedVariant('/d/a.h5oina')).toBeNull();
    expect(rememberVariant('/d/a.h5oina', { measurement: '1', pattern_source: 'processed' })).toBe(false);
  });
});

describe('limitConcurrent: a few questions at a time, in the order they were asked', () => {
  // A task that finishes when the test says so, and records when it started.
  function jobs(n, log) {
    return Array.from({ length: n }, (_, i) => {
      let done;
      const task = () => { log.push(i); return new Promise((resolve, reject) => { done = { resolve, reject }; }); };
      return { task, finish: (v) => done.resolve(v), fail: (e) => done.reject(e) };
    });
  }
  const flush = () => new Promise((r) => setTimeout(r, 0));

  it('starts two of five at once and the next one only when one has finished', async () => {
    const run = limitConcurrent(2);
    const started = [];
    const j = jobs(5, started);
    const answers = j.map((x) => run(x.task));
    await flush();
    expect(started).toEqual([0, 1]);
    j[0].finish('a');
    await flush();
    expect(started).toEqual([0, 1, 2]);
    j[2].finish('c');
    j[1].finish('b');
    await flush();
    expect(started).toEqual([0, 1, 2, 3, 4]);
    j[3].finish('d');
    j[4].finish('e');
    expect(await Promise.all(answers)).toEqual(['a', 'b', 'c', 'd', 'e']);
  });

  it('gives the place of a task that failed to the next one, and passes the failure on', async () => {
    const run = limitConcurrent(1);
    const started = [];
    const j = jobs(2, started);
    const first = run(j[0].task);
    const second = run(j[1].task);
    await flush();
    expect(started).toEqual([0]);
    j[0].fail(new Error('no'));
    await expect(first).rejects.toThrow('no');
    await flush();
    expect(started).toEqual([0, 1]);
    j[1].finish('ok');
    expect(await second).toBe('ok');
  });

  it('never starts a task whose signal is aborted before its turn, and rejects it', async () => {
    const run = limitConcurrent(1);
    const started = [];
    const j = jobs(3, started);
    const ctl = new AbortController();
    run(j[0].task);
    const waiting = run(j[1].task, ctl.signal);
    const third = run(j[2].task);
    await flush();
    ctl.abort();
    j[0].finish('a');
    await expect(waiting).rejects.toMatchObject({ name: 'AbortError' });
    await flush();
    expect(started).toEqual([0, 2]);
    j[2].finish('c');
    expect(await third).toBe('c');
  });

  it('gives the place back at once when a running task is aborted, though its answer never comes', async () => {
    const run = limitConcurrent(1);
    const started = [];
    const j = jobs(2, started);
    const ctl = new AbortController();
    run(j[0].task, ctl.signal).catch(() => {});
    const next = run(j[1].task);
    await flush();
    expect(started).toEqual([0]);
    ctl.abort();
    await flush();
    expect(started).toEqual([0, 1]);
    j[1].finish('b');
    expect(await next).toBe('b');
  });
});
