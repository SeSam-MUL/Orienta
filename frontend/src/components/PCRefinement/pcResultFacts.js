// What the Result panel of the PC page reads out of a search result (the answer of /api/pc/optimize).
// Pure functions, no page state: the page decides how to word and lay them out.

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * The one warning a result carries, from the backend's `quality.warning`:
 * `{kind: 'edge', axes}` | `{kind: 'notAMinimum'}` | `{kind: 'secondPosition'}` | null.
 * A result without `quality` is judged from the search record it carries.
 */
export function warningOf(result) {
  const q = result?.quality;
  const s = result?.search;
  const warning = q ? q.warning : null;
  if (warning === 'edge') return { kind: 'edge', axes: Array.isArray(q.axes) ? q.axes : [] };
  if (warning === 'notAMinimum') return { kind: 'notAMinimum' };
  if (warning === 'secondPosition') return { kind: 'secondPosition' };
  if (q) return null;
  if (Array.isArray(s?.on_edge) && s.on_edge.length > 0) return { kind: 'edge', axes: s.on_edge };
  if (Array.isArray(s?.confidence?.reasons) && s.confidence.reasons.includes('notAMinimum')) {
    return { kind: 'notAMinimum' };
  }
  return null;
}

/** A result on the edge of the search area or without a lowest point is not handed over with a plain "in use". */
export function needsDecision(result) {
  const w = warningOf(result);
  return !!w && (w.kind === 'edge' || w.kind === 'notAMinimum');
}

/** `{pc: [x, y, z], cost}` of a second position the search found, else null. */
export function secondPositionOf(result) {
  const s = result?.quality?.second_position ?? result?.search?.second_position;
  if (!s || !Array.isArray(s.pc) || s.pc.length !== 3 || !s.pc.every(finite)) return null;
  return { pc: s.pc, cost: finite(s.cost) ? s.cost : null };
}

/** The fit contrast (a number) or null. */
export function contrastOf(result) {
  const c = result?.quality ? result.quality.contrast : result?.search?.confidence?.contrast;
  return finite(c) ? c : null;
}

/** How many calibration patterns the result was computed from, or null. */
export function patternCountOf(result) {
  const n = result?.quality?.n_patterns ?? result?.search?.n_patterns;
  return finite(n) ? n : null;
}

/**
 * How far the patterns disagree: for each axis the range (largest minus smallest) of the PC each pattern
 * prefers on its own (`search.per_pattern`). `value` is the largest range in PC units; `pixels` the largest
 * range in pixels on this detector (x counts in columns, y and z in rows), or null when the pattern size is
 * unknown. Null when fewer than two patterns have a PC of their own.
 */
export function patternSpread(result) {
  const per = result?.search?.per_pattern;
  if (!Array.isArray(per)) return null;
  const rows = per.filter((r) => Array.isArray(r) && r.length === 3 && r.every(finite));
  if (rows.length < 2) return null;
  const range = [0, 1, 2].map((k) => Math.max(...rows.map((r) => r[k])) - Math.min(...rows.map((r) => r[k])));
  const size = result?.pattern_size;
  const hasSize = Array.isArray(size) && size.length === 2 && size.every((v) => finite(v) && v > 0);
  const pixels = hasSize
    ? Math.round(Math.max(range[0] * size[1], range[1] * size[0], range[2] * size[0]))
    : null;
  return { value: Math.max(...range), pixels };
}
