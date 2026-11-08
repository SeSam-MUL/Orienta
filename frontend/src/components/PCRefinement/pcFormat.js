// How the PC page writes a PC: four decimals, Bruker numbers, the way the fields and the result rows do.

/** "0.5026, 0.3272, 0.8458" for three numbers; '' for anything else. */
export function formatPc(pc) {
  if (!Array.isArray(pc) || pc.length !== 3 || pc.some((v) => typeof v !== 'number' || !Number.isFinite(v))) return '';
  return pc.map((v) => v.toFixed(4)).join(', ');
}

/** True when two PCs agree to the four decimals the page shows (half a unit of the last one). */
export function samePc(a, b, tolerance = 5e-5) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 3 || b.length !== 3) return false;
  return a.every((v, i) => Number.isFinite(v) && Number.isFinite(b[i]) && Math.abs(v - b[i]) <= tolerance);
}
