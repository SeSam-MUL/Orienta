/**
 * Pure utilities for the SVG spectrum plot.
 * No DOM, no React — testable in vitest.
 */
export function downsampleForWidth(values, targetPoints) {
  if (values.length <= targetPoints) return [...values];
  const step = values.length / targetPoints;
  const out = [];
  for (let i = 0; i < targetPoints; i++) {
    out.push(values[Math.floor(i * step)]);
  }
  // Ensure last point is included so the plot extends to the right edge
  if (out[out.length - 1] !== values[values.length - 1]) {
    out[out.length - 1] = values[values.length - 1];
  }
  return out;
}

export function energyToX(energy, eMin, eMax, width) {
  return ((energy - eMin) / (eMax - eMin)) * width;
}

export function xToEnergy(x, eMin, eMax, width) {
  return eMin + (x / width) * (eMax - eMin);
}

export function buildPolyline(counts, energyAxis, eMin, eMax, width, height, logScale = false) {
  const max = Math.max(...counts) || 1;
  const points = [];
  for (let i = 0; i < counts.length; i++) {
    const e = energyAxis[i];
    if (e < eMin || e > eMax) continue;
    const x = energyToX(e, eMin, eMax, width);
    const c = counts[i];
    const yNorm = logScale ? Math.log10(Math.max(1, c)) / Math.log10(Math.max(1, max)) : c / max;
    const y = height - yNorm * height;
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return points.join(' ');
}
