// Reading of the grain reconstruction answers, kept out of AnalysisPage.jsx so
// the shapes can be tested without rendering the page.
//
// Two shapes reach the page:
//   task payload (GET /api/analysis/task/{id}):
//     { status, progress, grain_count, result: { grain_count, mean_ecd, median_ecd,
//       std_ecd, mean_aspect_ratio, ... }, error }
//   flat answer of a synchronous reconstruction: { grain_count, mean_ecd, ... }
// Both carry what the statistics are made of: edge_rule, ecd_basis, n_grains_total,
// n_edge_grains, n_grains_used, edge_safeguard_applied; or grain_stats_error when the
// statistics failed (then no edge counts).

/** The reconstruction parameters the user types on the Grain tab: the five fields of the settings object. */
export const RECON_FIELDS = ['gbThreshold', 'minPixels', 'minIntercept', 'alpha', 'fill'];

/** The edge-grain rules and ECD bases the backend accepts; the first of each is its default. */
export const EDGE_RULES = ['matlab_script', 'mtex', 'outer_row', 'keep_all'];
export const ECD_BASES = ['cell_area', 'measured_pixels'];
export const DEFAULT_EDGE_RULE = 'matlab_script';
export const DEFAULT_ECD_BASIS = 'cell_area';
/**
 * What the backend does with the grain-size statistics. `statsPhase`: null = all phases, a phase id of
 * the loaded map, or 'most_measured' (the phase with the most measured pixels; the batch offers it).
 */
export const DEFAULT_STATS_FIELDS = Object.freeze({
  edgeRule: DEFAULT_EDGE_RULE,
  ecdBasis: DEFAULT_ECD_BASIS,
  statsPhase: null,
});
/**
 * The page-level object the Grain tab and "Run Complete Analysis" both read: the reconstruction
 * parameters (strings, as typed in the fields) and the statistics settings above.
 */
export const DEFAULT_GRAIN_STATS_SETTINGS = Object.freeze({
  gbThreshold: '4.0',
  minPixels: '3',
  minIntercept: '1.5',
  alpha: '3.1',
  fill: true,
  ...DEFAULT_STATS_FIELDS,
});

/**
 * The minimum intercept as a number of 0 or more, else undefined. It is only recorded in the workbook,
 * so a field that is empty or not such a number is never a reason to refuse anything: it is left out of
 * the request and the workbook cell stays empty.
 */
export function minInterceptOf(settings) {
  const typed = parseFloat((settings || {}).minIntercept);
  return Number.isFinite(typed) && typed >= 0 ? typed : undefined;
}

/** A typed number: NaN for an empty or half-typed field (Number('') would be 0). */
const typedNumber = (text) => {
  if (typeof text === 'number') return text;
  const s = String(text ?? '').trim();
  return s === '' ? NaN : Number(s);
};

/**
 * The three numbers the reconstruction needs, inside the limits of the backend (the batch route answers a 422 for the
 * others): the threshold more than 0, the minimum grain size a whole number of 1 or more ("2.7" is not silently 2), the
 * alpha radius factor 0 to 20. Null when one of them is empty, not a number or outside that.
 */
function reconNumbers(settings) {
  const s = settings || {};
  const threshold = typedNumber(s.gbThreshold);
  const minPixels = typedNumber(s.minPixels);
  const alpha = typedNumber(s.alpha);
  if (!(Number.isFinite(threshold) && threshold > 0)) return null;
  if (!(Number.isInteger(minPixels) && minPixels >= 1)) return null;
  if (!(Number.isFinite(alpha) && alpha >= 0 && alpha <= 20)) return null;
  return { threshold, minPixels, alpha };
}

/**
 * The arguments of `analysisApi.reconstructGrains` from the settings object, or null when a field the
 * reconstruction needs is empty, not a number or outside the backend's limits (it would reach the API as null or come
 * back as a 422). The minimum intercept is not one of them: it is undefined when not given.
 */
export function reconstructArgs(settings) {
  const nums = reconNumbers(settings);
  if (!nums) return null;
  const s = settings || {};
  return [nums.threshold, nums.minPixels, minInterceptOf(s), nums.alpha, s.fill !== false, s.edgeRule, s.ecdBasis, s.statsPhase ?? null];
}

/**
 * The reconstruction parameters of the batch request from the settings object: the numbers the Grain tab
 * holds, as numbers, inside the limits of the batch route (threshold more than 0, minimum size a whole number of 1 or
 * more, alpha 0 to 20); or null when a field is empty, not a number or outside them. `minIntercept` is only recorded in
 * the workbook (the reconstruction does not use it): undefined when it is not a number of 0 or more.
 */
export function batchReconstruction(settings) {
  const nums = reconNumbers(settings);
  if (!nums) return null;
  const s = settings || {};
  // recorded only: never a reason to refuse a batch. A value that is not a number of 0 or more is left out
  // (undefined: not sent, the workbook cell stays empty).
  return { ...nums, minIntercept: minInterceptOf(s), fill: s.fill !== false };
}

/** A file name without its extension, for the default dataset name of an export; '' when there is none. */
export function datasetNameFromFile(fileName) {
  if (typeof fileName !== 'string') return '';
  const base = fileName.split(/[\\/]/).filter(Boolean).pop() || '';
  const stem = base.replace(/\.[^.]*$/, '');
  return (stem || base).trim();
}

/**
 * The backend's task progress is a fraction (0 to 1); the page shows percent (0 to 100). Converted once, where the
 * value is taken. Not a number: undefined.
 */
export function progressPercent(fraction) {
  const n = Number(fraction);
  if (fraction === null || fraction === undefined || !Number.isFinite(n)) return undefined;
  return Math.max(0, Math.min(100, n * 100));
}

/**
 * The same file path? Backslashes and slashes are the same, a trailing slash does not matter, and on Windows
 * (`platform === 'win32'`, or a drive-letter path) the case does not matter either. The backend answers a
 * path as `str(Path(...))`, which on Windows turns `C:/data/x.xlsx` into `C:\\data\\x.xlsx`.
 */
export function samePath(a, b, platform) {
  if (typeof a !== 'string' || typeof b !== 'string') return a === b;
  const norm = (p) => p.replace(/\\/g, '/').replace(/\/+$/, '');
  const x = norm(a);
  const y = norm(b);
  const windows = platform === 'win32' || /^[A-Za-z]:\//.test(x);
  return windows ? x.toLowerCase() === y.toLowerCase() : x === y;
}

/** The same phases (ids and names, in order)? */
export function samePhases(a, b) {
  const x = a || [];
  const y = b || [];
  return x.length === y.length && x.every((p, i) => p.id === y[i].id && p.name === y[i].name);
}

const sameNumber = (typed, value) => Number.isFinite(parseFloat(typed)) && parseFloat(typed) === Number(value);

/**
 * The settings object after `GET /status` said what the backend holds for the grains it has: the
 * statistics settings, and the reconstruction parameters when it knows them (`recon_params` is null for
 * grains not made in this session). A field already showing the same number stays as typed ("4.0").
 * `keepRecon`: the user has typed a reconstruction parameter since the last reconstruction, so the backend's five
 * parameters (the old grains', not the user's choice) are not taken.
 * With no grains the statistics settings are not the backend's to give (nothing uses them yet): the
 * choices stay, except a phase id the loaded map does not have.
 */
export function settingsFromStatus(settings, status, phaseOptions, keepRecon = false) {
  const d = status || {};
  const next = { ...settings };
  if (d.has_grains) {
    if (EDGE_RULES.includes(d.edge_rule)) next.edgeRule = d.edge_rule;
    if (ECD_BASES.includes(d.ecd_basis)) next.ecdBasis = d.ecd_basis;
    if (d.stats_phase === null || d.stats_phase === 'most_measured' || Number.isInteger(d.stats_phase)) {
      next.statsPhase = d.stats_phase;
    }
    const r = d.recon_params;
    if (r && typeof r === 'object' && !keepRecon) {
      const put = (field, value) => {
        if (value === null || value === undefined || sameNumber(next[field], value)) return;
        next[field] = String(value);
      };
      put('gbThreshold', r.threshold);
      put('minPixels', r.min_pixels);
      put('minIntercept', r.min_intercept);
      put('alpha', r.alpha);
      if (typeof r.fill === 'boolean') next.fill = r.fill;
    }
  } else if (Number.isInteger(next.statsPhase) && !(phaseOptions || []).some((p) => p.id === next.statsPhase)) {
    next.statsPhase = null;
  }
  return next;
}

/**
 * Grain statistics from either shape: the nested `result` when present, else the top level.
 * `count` falls back to `result.n_grains`, then 0, so a message never reads "undefined".
 */
export function grainStatsFrom(payload) {
  const p = payload || {};
  const r = p.result && typeof p.result === 'object' ? p.result : p;
  return {
    count: r.grain_count ?? p.grain_count ?? r.n_grains ?? 0,
    meanEcd: r.mean_ecd,
    medianEcd: r.median_ecd,
    stdEcd: r.std_ecd,
    meanAr: r.mean_aspect_ratio,
    edgeRule: r.edge_rule,
    ecdBasis: r.ecd_basis,
    statsPhase: r.stats_phase,
    statsPhaseName: r.stats_phase_name,
    nGrainsTotal: r.n_grains_total,
    nEdgeGrains: r.n_edge_grains,
    nGrainsUsed: r.n_grains_used,
    edgeSafeguardApplied: r.edge_safeguard_applied,
    statsError: r.grain_stats_error,
  };
}

/**
 * Readable text for a failed request.
 * FastAPI answers a string `detail` for its own errors and a LIST of
 * `{ loc, msg, type }` objects for a rejected request body (422); a list rendered
 * as a React child crashes the page. Each entry becomes "field: message"
 * (the last element of `loc`); with no usable detail the error's own message,
 * then `fallback`. Always returns a string.
 */
export function errorMessageFrom(err, fallback) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((entry) => {
        if (typeof entry === 'string') return entry;
        const msg = entry && typeof entry.msg === 'string' ? entry.msg : null;
        if (!msg) return null;
        const loc = Array.isArray(entry.loc) && entry.loc.length ? String(entry.loc[entry.loc.length - 1]) : null;
        return loc ? `${loc}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return (typeof err?.message === 'string' && err.message) || fallback;
}

/**
 * Interpret a reconstruction task payload.
 * Returns { done, failed, message, stats }; `message` is the backend's error text
 * (`error`, then `message`) or null so the caller can supply a localised fallback.
 */
export function taskOutcome(payload) {
  const p = payload || {};
  const done = p.status === 'completed' || p.status === 'done';
  const failed = p.status === 'failed' || p.status === 'error';
  return {
    done,
    failed,
    message: failed ? (p.error || p.message || null) : null,
    stats: done ? grainStatsFrom(p) : null,
  };
}

/**
 * What the polling loop does with a failed status request.
 * A 404 means the backend no longer knows the task (restart, eviction): keep
 * polling would spin forever. Anything else (network blip) is retried.
 */
export function pollErrorOutcome(err) {
  return { stop: err?.response?.status === 404 };
}

/**
 * The answer of POST /grain-size laid over the statistics already shown: the numbers and the
 * edge/ECD fields are replaced, the grain count stays. With no statistics shown yet (tab opened
 * again, reconstruction failed) the count is the answer's own total; if that is missing too the
 * count stays undefined and the page leaves the line out (never "0"). The answer has no median,
 * so a median of the earlier basis is dropped rather than shown next to the new numbers.
 * `count` is the number of grains of the whole map; with a phase chosen the answer's `n_grains_total` is
 * that of the phase, so it is not a total and the count stays unknown.
 */
export function mergeGrainSize(prev, d) {
  const s = grainStatsFrom({ ...d, grain_count: undefined });
  const wholeMap = d?.stats_phase === null || d?.stats_phase === undefined;
  const count = prev?.count ?? (wholeMap ? (d?.n_grains_total ?? d?.n_grains_used) : undefined);
  return { ...(prev || {}), ...s, count, medianEcd: d?.median_ecd, statsError: undefined };
}

/**
 * What the edge-grain rule did, for the statistics block.
 * `kind`: 'keep_all' | 'safeguard' | 'removed' | null (no edge fields, e.g. the statistics failed).
 * The safeguard of `outer_row` is a share (more than 85 %), that of the two others a minimum
 * (fewer than 50 grains or under 15 %), so the page words them differently: `reason` is 'many'
 * (outer_row), 'few' (fewer than 50 would be left) or 'share' (under 15 % of the total would be);
 * `left` is the number of grains the rule would have left.
 */
export function edgeSummary(stats) {
  const s = stats || {};
  // no edge line when the statistics failed (the counts are absent) or there are no grains to count
  if (s.statsError || !s.edgeRule || s.nGrainsTotal === 0) return { kind: null };
  if (s.edgeRule === 'keep_all') return { kind: 'keep_all' };
  if (s.nGrainsTotal === undefined || s.nEdgeGrains === undefined) return { kind: null };
  const base = { rule: s.edgeRule, edge: s.nEdgeGrains, total: s.nGrainsTotal, used: s.nGrainsUsed };
  if (s.edgeSafeguardApplied) {
    // the condition that fired: outer_row keeps all above 85 % left out; the two others when fewer than 50
    // grains would be left, else when under 15 % of the total would be
    const left = s.nGrainsTotal - s.nEdgeGrains;
    const reason = s.edgeRule === 'outer_row' ? 'many' : (left < 50 ? 'few' : 'share');
    return { kind: 'safeguard', left, reason, ...base };
  }
  return { kind: 'removed', ...base };
}

/**
 * The name of the phase the statistics are made of, for the statistics block: the answer's own name, else
 * the name in the phase list of the loaded map, else null.
 */
export function statsPhaseLabel(stats, phaseOptions) {
  const id = stats?.statsPhase;
  if (id === null || id === undefined) return null;
  if (stats.statsPhaseName) return stats.statsPhaseName;
  return (phaseOptions || []).find((p) => p.id === id)?.name ?? null;
}

/**
 * The warnings of an Excel download: the header `X-Orienta-Warnings` holds a percent-encoded JSON list of
 * strings. Returns { warnings: [string], unreadable: boolean }; a header that cannot be read is
 * `unreadable`, never silently dropped.
 */
export function warningsFromHeader(value) {
  if (value === null || value === undefined || value === '') return { warnings: [], unreadable: false };
  try {
    const list = JSON.parse(decodeURIComponent(String(value)));
    if (Array.isArray(list) && list.every((w) => typeof w === 'string')) return { warnings: list, unreadable: false };
  } catch { /* falls through */ }
  return { warnings: [], unreadable: true };
}

/**
 * The Batch tab's own statistics settings the first time they are needed: the Grain tab's edge rule and ECD basis,
 * and "all phases" unless the Grain tab has "the phase with the most measured pixels" (a single phase of the
 * loaded map means nothing for the other files of a batch, and the batch offers only all / most measured).
 */
export function batchStatsFromGrain(settings) {
  const s = settings || DEFAULT_GRAIN_STATS_SETTINGS;
  return {
    edgeRule: s.edgeRule ?? DEFAULT_EDGE_RULE,
    ecdBasis: s.ecdBasis ?? DEFAULT_ECD_BASIS,
    statsPhase: s.statsPhase === 'most_measured' ? 'most_measured' : null,
  };
}
