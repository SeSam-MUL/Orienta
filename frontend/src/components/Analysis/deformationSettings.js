// The three parameters of the Deformation tab, in one place: what the page holds, what it sends and how it takes the
// backend's answer back.
//
// The page holds them as typed (strings, like the reconstruction parameters). The backend names are the ones of
// POST /api/analysis/deformation, GET /api/analysis/status (`deformation_params`) and the batch request.

/** The page's names of the parameters, in the order of the tab. */
export const DEFORMATION_FIELDS = ['kamOrder', 'kamThreshold', 'hagbMin'];

/** The backend's name of each parameter (request, `deformation_params` and batch fields). */
export const DEFORMATION_WIRE = Object.freeze({
  kamOrder: 'kam_order',               // neighbour shell the KAM is averaged over
  kamThreshold: 'kam_threshold_deg',   // degrees: misorientations above this are not part of the KAM
  hagbMin: 'hagb_min_deg',             // degrees: a boundary from here is a high-angle boundary; 3 degrees up to here is a sub-boundary
});

/** The values of the script; what the backend holds until the first "Analyze Deformation". */
export const DEFAULT_DEFORMATION_PARAMS = Object.freeze({
  kamOrder: '2',
  kamThreshold: '7.0',
  hagbMin: '15.0',
});

/**
 * The range the backend accepts for each parameter: `[low, high, lowIncluded, whole]`. The order is a whole number of
 * 1 to 5; the angles are more than the lower value (the threshold: more than 0, the HAGB limit: more than 3) and at most
 * the upper one. Outside it the request is a 422.
 */
export const DEFORMATION_RANGES = Object.freeze({
  kamOrder: [1, 5, true, true],
  kamThreshold: [0, 30, false, false],
  hagbMin: [3, 90, false, false],
});

const asNumber = (text) => {
  if (typeof text === 'number') return text;
  return String(text ?? '').trim() === '' ? NaN : Number(text);
};

/** Is the text a usable value of this parameter? */
export function deformationFieldValid(field, text) {
  const n = asNumber(text);
  const [low, high, lowIncluded, whole] = DEFORMATION_RANGES[field];
  if (!Number.isFinite(n) || n > high || (lowIncluded ? n < low : n <= low)) return false;
  return !whole || Number.isInteger(n);
}

/**
 * The parameters as numbers, ready to send, or null when a field is empty, not a number or outside its range
 * (it would come back as a 422).
 */
export function deformationArgs(params) {
  const p = params || {};
  const out = {};
  for (const field of DEFORMATION_FIELDS) {
    if (!deformationFieldValid(field, p[field])) return null;
    out[field] = asNumber(p[field]);
  }
  return out;
}

/** The request body (or batch fields) from the numbers: the backend's names. */
export function deformationBody(numbers) {
  const body = {};
  for (const field of DEFORMATION_FIELDS) body[DEFORMATION_WIRE[field]] = numbers[field];
  return body;
}

const sameNumber = (typed, value) => Number.isFinite(parseFloat(typed)) && parseFloat(typed) === Number(value);

/**
 * The parameters after the backend said which ones it holds (`deformation_params` of an answer of POST /deformation or
 * of GET /status): a field that shows the same number already stays as typed ("7"). The same object comes back when
 * nothing changes, so a store update with it is no update. An answer without the three numbers changes nothing.
 */
export function deformationFromBackend(params, held) {
  if (!held || typeof held !== 'object') return params;
  let next = params;
  for (const field of DEFORMATION_FIELDS) {
    const value = held[DEFORMATION_WIRE[field]];
    if (typeof value !== 'number' || !Number.isFinite(value) || sameNumber(params[field], value)) continue;
    if (next === params) next = { ...params };
    next[field] = String(value);
  }
  return next;
}
