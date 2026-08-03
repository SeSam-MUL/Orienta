/**
 * Rough runtime estimate for spherical indexing WITHOUT a CUDA GPU.
 *
 * The PyTorch spherical backend falls back to the CPU when no NVIDIA GPU is
 * present. It is numerically identical to the CUDA path (verified: max 0.0001°
 * disorientation over real SampleB patterns) — just far slower. Users hitting
 * that fallback deserve to know what they are signing up for BEFORE a run that
 * can take hours, so we show an order-of-magnitude estimate.
 *
 * Rates below were MEASURED on the full-map streaming path (backend.index_h5,
 * the one a full scan uses) on a 24-thread desktop CPU with refine=on, and
 * cross-checked against the in-memory path. They are deliberately reported as
 * "rough": a laptop CPU is easily 2-4x slower, and throughput also depends on
 * pattern size. The estimate exists to separate "minutes" from "hours", not to
 * be accurate to the minute.
 *
 * Bandwidth does NOT scale as a clean L^3: the correlation grid size is
 * FFT-rounded, so 68 and 88 land close together while 128 is ~6x more
 * expensive. Hence a measured bucket table rather than a formula.
 */

// patterns/second on the streaming path, per bandwidth bucket.
const CPU_RATES = [
  { maxBandwidth: 68, patternsPerSecond: 30 },
  { maxBandwidth: 96, patternsPerSecond: 25 },
  { maxBandwidth: Infinity, patternsPerSecond: 4.5 },
];

/** Reference GPU rate (RTX 4070, L=88, refine=on) — for the "vs GPU" comparison. */
export const GPU_REFERENCE_RATE = 600;

export function cpuPatternsPerSecond(bandwidth) {
  const bw = Number(bandwidth) || 88;
  return CPU_RATES.find(r => bw <= r.maxBandwidth).patternsPerSecond;
}

/** Rough seconds for `nPatterns` on the CPU fallback. 0 when unknown. */
export function estimateCpuSphericalSeconds(nPatterns, bandwidth) {
  const n = Number(nPatterns);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return n / cpuPatternsPerSecond(bandwidth);
}

/**
 * Human-readable duration, deliberately coarse ("~12 min", "~2.5 h") so it
 * never reads as a precise promise. Returns '' for 0/unknown.
 */
export function formatRoughDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)} min`;
  const hours = minutes / 60;
  return hours < 10 ? `${hours.toFixed(1)} h` : `${Math.round(hours)} h`;
}
