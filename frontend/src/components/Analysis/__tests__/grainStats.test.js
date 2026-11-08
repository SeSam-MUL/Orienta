import { describe, it, expect } from 'vitest';
import {
  grainStatsFrom, taskOutcome, pollErrorOutcome, errorMessageFrom, mergeGrainSize, edgeSummary,
  EDGE_RULES, ECD_BASES, reconstructArgs, DEFAULT_GRAIN_STATS_SETTINGS,
} from '../grainStats';

const STATS = { mean_ecd: 12.5, median_ecd: 11.0, std_ecd: 3.2, mean_aspect_ratio: 1.6 };

describe('grainStatsFrom', () => {
  it('reads the nested result of a task payload', () => {
    const payload = { status: 'completed', grain_count: 98, result: { grain_count: 98, n_grains: 98, ...STATS } };
    expect(grainStatsFrom(payload)).toEqual({ count: 98, meanEcd: 12.5, medianEcd: 11.0, stdEcd: 3.2, meanAr: 1.6 });
  });

  it('reads the flat shape of the synchronous answer', () => {
    expect(grainStatsFrom({ grain_count: 7, ...STATS }))
      .toEqual({ count: 7, meanEcd: 12.5, medianEcd: 11.0, stdEcd: 3.2, meanAr: 1.6 });
  });

  it('falls back to the top-level grain_count when result has none', () => {
    const s = grainStatsFrom({ grain_count: 5, result: {} });
    expect(s.count).toBe(5);
    expect(s.meanEcd).toBeUndefined();
  });

  it('keeps the count when the statistics are absent (n_grains = 0 or summary failed)', () => {
    const s = grainStatsFrom({ result: { grain_count: 0, n_grains: 0 } });
    expect(s).toEqual({ count: 0, meanEcd: undefined, medianEcd: undefined, stdEcd: undefined, meanAr: undefined });
  });

  it('counts from result.n_grains, then 0, when grain_count is missing', () => {
    expect(grainStatsFrom({ status: 'completed', result: { n_grains: 12 } }).count).toBe(12);
    expect(grainStatsFrom({ status: 'completed', result: {} }).count).toBe(0);
  });

  it('tolerates an empty payload', () => {
    expect(grainStatsFrom(undefined).count).toBe(0);
    expect(grainStatsFrom(null).meanEcd).toBeUndefined();
  });
});

describe('taskOutcome', () => {
  it('a running task is neither done nor failed', () => {
    expect(taskOutcome({ status: 'running', progress: 0.3 }))
      .toMatchObject({ done: false, failed: false, message: null, stats: null });
  });

  it.each(['completed', 'done'])('%s carries the nested statistics', (status) => {
    const out = taskOutcome({ status, progress: 1, grain_count: 98, result: { grain_count: 98, ...STATS } });
    expect(out.done).toBe(true);
    expect(out.failed).toBe(false);
    expect(out.stats).toEqual({ count: 98, meanEcd: 12.5, medianEcd: 11.0, stdEcd: 3.2, meanAr: 1.6 });
  });

  it('failed uses the error text', () => {
    const out = taskOutcome({ status: 'failed', progress: 1, error: 'no grains today' });
    expect(out).toMatchObject({ done: false, failed: true, message: 'no grains today', stats: null });
  });

  it('legacy "error" status with a message still counts as failed', () => {
    expect(taskOutcome({ status: 'error', message: 'old style' }))
      .toMatchObject({ failed: true, message: 'old style' });
  });

  it('error wins over message; no text gives a null message for the caller to localise', () => {
    expect(taskOutcome({ status: 'failed', error: 'a', message: 'b' }).message).toBe('a');
    expect(taskOutcome({ status: 'failed' }).message).toBeNull();
  });
});

describe('pollErrorOutcome', () => {
  it('stops on a 404: the backend no longer knows the task', () => {
    expect(pollErrorOutcome({ response: { status: 404 } })).toEqual({ stop: true });
  });

  it.each([
    [{ response: { status: 500 } }],
    [{ response: { status: 503 } }],
    [new Error('Network Error')],
    [undefined],
  ])('keeps polling on other errors (%#)', (err) => {
    expect(pollErrorOutcome(err)).toEqual({ stop: false });
  });
});

describe('errorMessageFrom', () => {
  it('returns a string detail as it is', () => {
    expect(errorMessageFrom({ response: { data: { detail: 'No dataset loaded' } } }, 'fb')).toBe('No dataset loaded');
  });

  it('formats a list detail (pydantic 422) as "field: message" entries', () => {
    const err = {
      response: {
        status: 422,
        data: {
          detail: [
            { type: 'float_type', loc: ['body', 'alpha'], msg: 'Input should be a valid number', input: null },
            { type: 'greater_than_equal', loc: ['body', 'misorientation_threshold'], msg: 'Input should be >= 0' },
          ],
        },
      },
    };
    const text = errorMessageFrom(err, 'fb');
    expect(typeof text).toBe('string');
    expect(text).toBe('alpha: Input should be a valid number; misorientation_threshold: Input should be >= 0');
  });

  it('keeps an entry without a loc readable and tolerates plain strings in the list', () => {
    expect(errorMessageFrom({ response: { data: { detail: [{ msg: 'bad request' }, 'plain'] } } }, 'fb'))
      .toBe('bad request; plain');
  });

  it('falls back to the error message, then to the fallback, when there is no response', () => {
    expect(errorMessageFrom(new Error('Network Error'), 'fb')).toBe('Network Error');
    expect(errorMessageFrom({}, 'fb')).toBe('fb');
    expect(errorMessageFrom(undefined, 'fb')).toBe('fb');
  });

  it('never returns an object, whatever detail holds', () => {
    expect(typeof errorMessageFrom({ response: { data: { detail: { a: 1 } } } }, 'fb')).toBe('string');
    expect(errorMessageFrom({ response: { data: { detail: [] } } }, 'fb')).toBe('fb');
  });
});

const EDGE = {
  edge_rule: 'mtex', ecd_basis: 'measured_pixels',
  n_grains_total: 98, n_edge_grains: 34, n_grains_used: 64, edge_safeguard_applied: false,
};
const EDGE_READ = {
  edgeRule: 'mtex', ecdBasis: 'measured_pixels',
  nGrainsTotal: 98, nEdgeGrains: 34, nGrainsUsed: 64, edgeSafeguardApplied: false,
};

describe('grainStatsFrom: edge-grain and ECD-basis fields', () => {
  it('reads them from the nested result of a task payload', () => {
    const s = grainStatsFrom({ status: 'completed', result: { grain_count: 98, ...STATS, ...EDGE } });
    expect(s).toMatchObject(EDGE_READ);
  });

  it('reads them from the flat shape', () => {
    expect(grainStatsFrom({ grain_count: 98, ...STATS, ...EDGE })).toMatchObject(EDGE_READ);
  });

  it('carries the error of a failed statistics step, with no edge counts', () => {
    const s = grainStatsFrom({ result: { grain_count: 98, grain_stats_error: 'no measured pixels' } });
    expect(s.statsError).toBe('no measured pixels');
    expect(s.count).toBe(98);
    expect(s.nEdgeGrains).toBeUndefined();
    expect(s.edgeRule).toBeUndefined();
  });

  it('the fields are absent, not zero, when the answer has none', () => {
    const s = grainStatsFrom({ grain_count: 7, ...STATS });
    expect(s.nGrainsTotal).toBeUndefined();
    expect(s.edgeSafeguardApplied).toBeUndefined();
  });
});

describe('mergeGrainSize', () => {
  const prev = grainStatsFrom({ result: { grain_count: 98, ...STATS, ...EDGE } });

  it('replaces the numbers and the edge fields, keeps the grain count', () => {
    const next = mergeGrainSize(prev, {
      mean_ecd: 20.5, std_ecd: 5.5, mean_aspect_ratio: 1.25,
      ...EDGE, edge_rule: 'outer_row', n_edge_grains: 10, n_grains_used: 88,
    });
    expect(next).toMatchObject({
      count: 98, meanEcd: 20.5, stdEcd: 5.5, meanAr: 1.25,
      edgeRule: 'outer_row', nEdgeGrains: 10, nGrainsUsed: 88,
    });
  });

  it('drops the median of the earlier basis (the answer has none) and an earlier error', () => {
    const next = mergeGrainSize({ ...prev, statsError: 'old' }, { mean_ecd: 1, std_ecd: 1, mean_aspect_ratio: 1, ...EDGE });
    expect(next.medianEcd).toBeUndefined();
    expect(next.statsError).toBeUndefined();
  });

  it('with no earlier statistics the count is the total of the answer, never 0', () => {
    expect(mergeGrainSize(null, { mean_ecd: 1, ...EDGE }).count).toBe(98);
    expect(mergeGrainSize(undefined, { mean_ecd: 1, n_grains_used: 64 }).count).toBe(64);
  });

  it('leaves the count undefined (the page omits the line) when nothing says it', () => {
    expect(mergeGrainSize(null, { mean_ecd: 1 }).count).toBeUndefined();
  });

  it('keeps the count of the earlier statistics over the total of the answer', () => {
    expect(mergeGrainSize({ count: 120 }, { mean_ecd: 1, ...EDGE }).count).toBe(120);
  });
});

describe('edgeSummary', () => {
  const read = (extra) => grainStatsFrom({ grain_count: 98, ...STATS, ...EDGE, ...extra });

  it('says how many edge grains a rule left out', () => {
    expect(edgeSummary(read({}))).toEqual({ kind: 'removed', rule: 'mtex', edge: 34, total: 98, used: 64 });
  });

  it('reports the safeguard, worded as a minimum for two rules and as a share for outer_row', () => {
    expect(edgeSummary(read({ n_edge_grains: 60, edge_safeguard_applied: true }))).toMatchObject({ kind: 'safeguard', reason: 'few' });
    expect(edgeSummary(read({ edge_rule: 'outer_row', edge_safeguard_applied: true })))
      .toMatchObject({ kind: 'safeguard', reason: 'many' });
    expect(edgeSummary(read({ n_edge_grains: 60, edge_safeguard_applied: true }))).not.toHaveProperty('many');
  });

  it('names the condition that fired: under 50 left, else under 15 % of the total, and for outer_row the share', () => {
    // 98 total, 60 edge: 38 left, under 50
    expect(edgeSummary(read({ n_edge_grains: 60, edge_safeguard_applied: true })))
      .toMatchObject({ kind: 'safeguard', left: 38, reason: 'few' });
    // 1000 total, 900 edge: 100 left, not under 50 but under 15 % of 1000
    expect(edgeSummary(read({ n_grains_total: 1000, n_edge_grains: 900, edge_safeguard_applied: true })))
      .toMatchObject({ kind: 'safeguard', left: 100, reason: 'share' });
    expect(edgeSummary(read({ edge_rule: 'outer_row', n_grains_total: 20, n_edge_grains: 19, edge_safeguard_applied: true })))
      .toMatchObject({ kind: 'safeguard', left: 1, reason: 'many' });
  });

  it('keep_all has its own kind', () => {
    expect(edgeSummary(read({ edge_rule: 'keep_all', n_edge_grains: 0 }))).toEqual({ kind: 'keep_all' });
  });

  it('is empty when the statistics failed or the answer has no edge fields', () => {
    expect(edgeSummary(grainStatsFrom({ result: { grain_count: 98, grain_stats_error: 'x' } }))).toEqual({ kind: null });
    expect(edgeSummary(grainStatsFrom({ grain_count: 98, ...STATS }))).toEqual({ kind: null });
    expect(edgeSummary(null)).toEqual({ kind: null });
  });
});

describe('offered options', () => {
  it('are the four rules and two bases the backend accepts, the script rule and cell area first', () => {
    expect(EDGE_RULES).toEqual(['matlab_script', 'mtex', 'outer_row', 'keep_all']);
    expect(ECD_BASES).toEqual(['cell_area', 'measured_pixels']);
  });
});

describe('real shapes of the grain-size answers', () => {
  it('statistics failed: rule and basis are present (the route sets them first), the counts are absent', () => {
    const s = grainStatsFrom({
      result: { grain_count: 98, n_grains: 98, edge_rule: 'matlab_script', ecd_basis: 'cell_area', grain_stats_error: 'boom' },
    });
    expect(s).toMatchObject({ count: 98, edgeRule: 'matlab_script', ecdBasis: 'cell_area', statsError: 'boom' });
    expect(edgeSummary(s)).toEqual({ kind: null });
  });

  it('no grains: the counts are zero and there is nothing to say about edge grains', () => {
    const s = grainStatsFrom({
      result: {
        grain_count: 0, n_grains: 0, edge_rule: 'matlab_script', ecd_basis: 'cell_area',
        n_grains_total: 0, n_edge_grains: 0, n_grains_used: 0, edge_safeguard_applied: false,
      },
    });
    expect(s.count).toBe(0);
    expect(edgeSummary(s)).toEqual({ kind: null });
  });

  it('a /grain-size answer with a median brings the median line back', () => {
    const prev = grainStatsFrom({ result: { grain_count: 98, ...STATS, ...EDGE } });
    expect(mergeGrainSize(prev, { mean_ecd: 20, median_ecd: 9.5, ...EDGE }).medianEcd).toBe(9.5);
  });
});

describe('reconstructArgs', () => {
  it('turns the stored fields into the arguments of reconstructGrains, with the Grain tab defaults', () => {
    expect(reconstructArgs(DEFAULT_GRAIN_STATS_SETTINGS)).toEqual([4, 3, 1.5, 3.1, true, 'matlab_script', 'cell_area', null]);
  });

  it('carries what was typed and chosen', () => {
    expect(reconstructArgs({
      gbThreshold: '6.5', minPixels: '5', minIntercept: '2', alpha: '0', fill: false,
      edgeRule: 'mtex', ecdBasis: 'measured_pixels',
    })).toEqual([6.5, 5, 2, 0, false, 'mtex', 'measured_pixels', null]);
  });

  it.each(['', '-2', 'abc', ' '])('a min intercept of "%s" is left out, not a reason to refuse the reconstruction', (typed) => {
    const args = reconstructArgs({ ...DEFAULT_GRAIN_STATS_SETTINGS, minIntercept: typed });
    expect(args).not.toBeNull();
    expect(args[2]).toBeUndefined();
    expect(args.slice(0, 2)).toEqual([4, 3]);
    expect(args.slice(3)).toEqual([3.1, true, 'matlab_script', 'cell_area', null]);
  });

  it('a min intercept of 0 is a value', () => {
    expect(reconstructArgs({ ...DEFAULT_GRAIN_STATS_SETTINGS, minIntercept: '0' })[2]).toBe(0);
  });

  it('is null when a field is empty or not a number', () => {
    expect(reconstructArgs({ ...DEFAULT_GRAIN_STATS_SETTINGS, alpha: '' })).toBeNull();
    expect(reconstructArgs({ ...DEFAULT_GRAIN_STATS_SETTINGS, gbThreshold: 'x' })).toBeNull();
    expect(reconstructArgs(undefined)).toBeNull();
  });
});
