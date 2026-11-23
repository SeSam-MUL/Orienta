/**
 * Pure helpers of the particle comparison panel (ParticleNccPanel.jsx).
 */

/** The next free particle name: "P1", "P2", … — or the given base with a running number. */
export function nextParticleName(particles, base = 'P') {
  const used = new Set((particles || []).map((p) => p.name));
  for (let i = 1; i < 10000; i += 1) {
    const name = `${base}${i}`;
    if (!used.has(name)) return name;
  }
  return `${base}${Date.now()}`;
}

/**
 * Add a particle centred on `pixelIndex`. Refuses (returns `{ error }`) a name that is
 * empty or already used, so the request the backend gets never names a particle twice.
 */
export function addParticle(particles, { name, pixelIndex, radius }) {
  const clean = String(name ?? '').trim();
  if (!clean) return { error: 'particleNameMissing' };
  if ((particles || []).some((p) => p.name === clean)) return { error: 'particleNameTaken' };
  if (!Number.isInteger(pixelIndex) || pixelIndex < 0) return { error: 'particlePixelMissing' };
  const r = Math.max(0, Math.min(10, Math.round(Number(radius) || 0)));
  return { particles: [...(particles || []), { name: clean, center_index: pixelIndex, radius_px: r }] };
}

/** How many patterns a disc of `radius` around `index` holds on an nRows × nCols scan. */
export function discSize(index, radius, nRows, nCols) {
  if (!(nRows > 0 && nCols > 0)) return 0;
  const r0 = Math.floor(index / nCols);
  const c0 = index % nCols;
  let n = 0;
  for (let r = Math.max(0, r0 - radius); r <= Math.min(nRows - 1, r0 + radius); r += 1) {
    for (let c = Math.max(0, c0 - radius); c <= Math.min(nCols - 1, c0 + radius); c += 1) {
      if ((r - r0) ** 2 + (c - c0) ** 2 <= radius * radius) n += 1;
    }
  }
  return n;
}

/** The request body of /particle-phase-ncc/start. Phase keys in the order the picker lists them. */
export function particleRequest({ particles, phases, selectedKeys, topN, settings }) {
  const keys = (phases || []).map((p) => p.key).filter((k) => selectedKeys.has(k));
  return {
    particles: particles.map((p) => ({ name: p.name, center_index: p.center_index, radius_px: p.radius_px })),
    phase_keys: keys,
    top_n: topN,
    max_bandwidth: settings.bandwidth,
    bg_remove: settings.bgRemove,
    use_pixel_pc: settings.usePixelPc,
    aperture: settings.aperture,
    aperture_radius: settings.apertureRadius,
  };
}

/** A number with three decimals in the reader's locale; "—" for none. */
export function fmtNcc(x, lang, digits = 3) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return '—';
  return new Intl.NumberFormat(lang || 'en', {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }).format(x);
}

/** The phase row of `key` in a particle summary. */
export function phaseRow(particle, key) {
  return (particle?.phases || []).find((r) => r.phase_key === key) || null;
}
