// The four RX criteria of the Recrystallization tab, in one place: what the page holds, what it sends and
// how it takes the backend's answer back.
//
// The page holds the criteria as typed (strings, like the reconstruction parameters). The backend names are the
// ones of POST /api/analysis/rx, GET /api/analysis/status and the batch request; tests/test_analysis_frontend_contract.py
// holds `RX_WIRE`, the defaults and the ranges against the backend's.

/** The page's names of the criteria, in the order of the tab. */
export const RX_FIELDS = ['gosThreshold', 'gbcFraction', 'gkamThreshold', 'minRadius'];

/** The backend's name of each criterion (request, /status and answer fields). */
export const RX_WIRE = Object.freeze({
  gosThreshold: 'rx_gos_threshold',      // degrees: a grain is RX below this GOS
  gbcFraction: 'rx_gbc_fraction',        // fraction of the largest gBC: a grain is RX above this gBC
  gkamThreshold: 'rx_gkam_threshold',    // degrees: a grain is RX below this gKAM
  minRadius: 'rx_min_radius_um',         // micrometres: a grain is RX above this equivalent radius
});

/** The values of the script; what the backend holds until the first "Classify". */
export const DEFAULT_RX_CRITERIA = Object.freeze({
  gosThreshold: '1.25',
  gbcFraction: '0.7',
  gkamThreshold: '0.55',
  minRadius: '0.4',
});

/** The range the backend accepts for each criterion (inclusive); outside it the request is a 422. */
export const RX_RANGES = Object.freeze({
  gosThreshold: [0, 20],
  gbcFraction: [0, 1],
  gkamThreshold: [0, 10],
  minRadius: [0, 100],
});

/**
 * The criteria as numbers, ready to send, or null when a field is empty, not a number or outside its range
 * (it would come back as a 422).
 */
export function rxCriteriaArgs(criteria) {
  const c = criteria || {};
  const out = {};
  for (const field of RX_FIELDS) {
    const text = c[field];
    const n = typeof text === 'number' ? text : (String(text ?? '').trim() === '' ? NaN : Number(text));
    const [low, high] = RX_RANGES[field];
    if (!Number.isFinite(n) || n < low || n > high) return null;
    out[field] = n;
  }
  return out;
}

const sameNumber = (typed, value) => Number.isFinite(parseFloat(typed)) && parseFloat(typed) === Number(value);

/**
 * The criteria after the backend said which ones it holds (an answer of POST /rx or GET /status): a field that
 * shows the same number already stays as typed ("0.70"). The same object comes back when nothing changes, so a
 * store update with it is no update. An answer without the four numbers changes nothing.
 */
export function rxCriteriaFromBackend(criteria, answer) {
  const d = answer || {};
  let next = criteria;
  for (const field of RX_FIELDS) {
    const value = d[RX_WIRE[field]];
    if (typeof value !== 'number' || !Number.isFinite(value) || sameNumber(criteria[field], value)) continue;
    if (next === criteria) next = { ...criteria };
    next[field] = String(value);
  }
  return next;
}

/** The numbers of the criteria as the line next to the results and the batch settings print them. */
export function rxCriteriaLabels(numbers) {
  return { gos: numbers.gosThreshold, gbc: numbers.gbcFraction, gkam: numbers.gkamThreshold, radius: numbers.minRadius };
}

/** The phase the RX analysis works on: 'most_measured' (the phase with the most indexed pixels of the map) or a phase id. */
export const DEFAULT_RX_PHASE = 'most_measured';

/** A phase choice the map can take: a phase id of the map, else the default (an id of another map means nothing). */
export function rxPhaseOf(value, phaseOptions) {
  return Number.isInteger(value) && (phaseOptions || []).some((p) => p.id === value) ? value : DEFAULT_RX_PHASE;
}
