/**
 * The three grain-boundary classes, and the rule that keeps them a partition.
 *
 * Sub-boundaries, low-angle and high-angle boundaries are ADJACENT ranges, not
 * three independent ones: every misorientation angle belongs to exactly one of
 * them. So the classes share their cut points — raising where the low-angle
 * class ends is the same act as raising where the high-angle class begins, and
 * the panel must not let the user state otherwise (a "5–20°" low-angle class
 * next to a "≥15°" high-angle class describes nothing).
 *
 * Hence: edit any bound, and the neighbour follows.
 */

// Below this, an angle is orientation noise rather than a boundary. It is the
// floor for the first class, not a class in itself.
export const FLOOR_DEG = 0.5;
export const CEILING_DEG = 62.8;   // the largest misorientation cubic symmetry allows

export const BAND_IDS = ['sub', 'lagb', 'hagb'];

export function defaultBands() {
  return [
    { id: 'sub',  min: FLOOR_DEG, max: 5,    color: '#8be9fd', width: 1, on: true },
    { id: 'lagb', min: 5,         max: 15,   color: '#f1fa8c', width: 2, on: true },
    { id: 'hagb', min: 15,        max: null, color: '#ffffff', width: 3, on: true },
  ];
}

const round1 = (v) => Math.round(v * 10) / 10;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// The narrowest a class may get. Without it, dragging one cut point past
// another would produce an empty class — a colour in the legend that can never
// appear on the map.
export const MIN_SPAN_DEG = 0.1;

/**
 * The whole partition is three numbers: the noise floor and two cut points.
 * Every edit is an edit to one of them, which is why a change to "where the
 * low-angle class ends" IS a change to "where the high-angle class begins".
 */
function toCuts(bands) {
  return { floor: bands[0].min, c1: bands[0].max, c2: bands[1].max };
}

function fromCuts(bands, { floor, c1, c2 }) {
  const out = bands.map((b) => ({ ...b }));
  out[0].min = floor; out[0].max = c1;
  out[1].min = c1;    out[1].max = c2;
  out[2].min = c2;    out[2].max = null;
  return out;
}

/**
 * Apply one edit and restore the partition.
 *
 * A bound that is pushed past its neighbour takes the neighbour with it — the
 * user stated what they wanted, and the panel's job is to make the rest agree
 * rather than to swallow the edit.
 *
 * @param bands  current list, ascending
 * @param index  which class was edited
 * @param field  'min' | 'max' | 'color' | 'width' | 'on'
 * @param value  the new value
 */
export function applyBandEdit(bands, index, field, value) {
  if (!Array.isArray(bands) || bands.length !== 3) return bands;
  const band = bands[index];
  if (!band) return bands;

  if (field === 'color' || field === 'on') {
    const out = bands.map((b) => ({ ...b }));
    out[index][field] = value;
    return out;
  }
  if (field === 'width') {
    const out = bands.map((b) => ({ ...b }));
    out[index].width = clamp(Math.round(Number(value) || 1), 1, 8);
    return out;
  }

  const raw = Number(value);
  if (!Number.isFinite(raw)) return bands;
  const v = round1(clamp(raw, FLOOR_DEG, CEILING_DEG));

  let { floor, c1, c2 } = toCuts(bands);

  // Which of the three numbers did this edit touch?
  if (index === 0 && field === 'min') {
    floor = clamp(v, FLOOR_DEG, c1 - MIN_SPAN_DEG);
  } else if ((index === 0 && field === 'max') || (index === 1 && field === 'min')) {
    c1 = clamp(v, floor + MIN_SPAN_DEG, CEILING_DEG - MIN_SPAN_DEG);
    if (c2 < c1 + MIN_SPAN_DEG) c2 = round1(c1 + MIN_SPAN_DEG);      // push the one above
  } else if ((index === 1 && field === 'max') || (index === 2 && field === 'min')) {
    c2 = clamp(v, FLOOR_DEG + 2 * MIN_SPAN_DEG, CEILING_DEG);
    if (c1 > c2 - MIN_SPAN_DEG) {                                     // pull the one below
      c1 = round1(c2 - MIN_SPAN_DEG);
      if (floor > c1 - MIN_SPAN_DEG) floor = round1(Math.max(FLOOR_DEG, c1 - MIN_SPAN_DEG));
    }
  } else {
    // The last class is open-ended by definition; there is no upper end to set.
    return bands;
  }

  return fromCuts(bands, { floor: round1(floor), c1: round1(c1), c2: round1(c2) });
}

/** True when the list still describes three adjacent, non-empty ranges. */
export function bandsArePartition(bands) {
  if (!Array.isArray(bands) || bands.length < 2) return false;
  for (let i = 0; i < bands.length; i += 1) {
    const b = bands[i];
    const upper = b.max ?? Infinity;
    if (!(upper > b.min)) return false;
    if (i > 0 && bands[i - 1].max !== b.min) return false;
  }
  return true;
}
