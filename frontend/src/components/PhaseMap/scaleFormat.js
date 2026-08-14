/**
 * How a value scale is written down and coloured.
 *
 * Two pure functions, kept apart from the React legend because the canvas
 * exporter needs exactly the same numbers and the same colour stops. A second
 * implementation on the drawing side is how a bar ends up disagreeing with the
 * picture it belongs to.
 */

/** A tick label: readable at a glance, never more precision than it has. */
export function formatScaleValue(v) {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(a >= 1 ? 2 : 3);
}

/** The stops as they arrived from the backend, or a black→white stand-in. */
export function scaleStops(stops) {
  return Array.isArray(stops) && stops.length >= 2 ? stops : ['#000000', '#ffffff'];
}

/** CSS gradient string for a set of colour stops, bottom = min. */
export function stopsToGradient(stops, direction = 'to top') {
  return `linear-gradient(${direction}, ${scaleStops(stops).join(', ')})`;
}
