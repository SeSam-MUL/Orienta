/**
 * dictionaryTargets.js — which master belongs to which selected phase.
 *
 * Generating a dictionary needs a MASTER PATTERN, but the Selected-Phases list
 * holds whatever entry the user picked (a master, or an existing dictionary).
 * These pure helpers resolve "phase -> its master", so the per-phase Generate
 * button on a card and the phase dropdown in the dialog can never disagree
 * about what will actually be simulated.
 *
 * Kept free of React so the resolution rules are unit-testable on their own.
 */

/** All discovered master-pattern entries belonging to `phase` (matched by formula). */
export function mastersForPhase(phase, allFiles = []) {
  const formula = (phase?.formula || '').toLowerCase();
  if (!formula) return [];
  return allFiles.filter(
    f => f.file_type === 'master' && (f.formula || '').toLowerCase() === formula
  );
}

/** All discovered dictionary entries belonging to `phase` (matched by formula). */
export function dictsForPhase(phase, allFiles = []) {
  const formula = (phase?.formula || '').toLowerCase();
  if (!formula) return [];
  return allFiles.filter(
    f => f.file_type === 'dictionary' && (f.formula || '').toLowerCase() === formula
  );
}

/**
 * The master pattern to simulate from for `phase`, or '' when there is none.
 *
 * Prefers a discovered master of the same formula; falls back to the phase
 * entry itself when the user selected the master directly (a `.h5` that is not
 * a `_dict_` file). Returns '' for a phase that only has dictionaries — we
 * cannot generate a new dictionary without its master.
 */
export function masterPathForPhase(phase, allFiles = []) {
  const masters = mastersForPhase(phase, allFiles);
  if (masters.length && masters[0].path) return masters[0].path;

  const own = phase?.path || '';
  if (!own) return '';
  if (phase?.file_type === 'master') return own;
  if (/\.(h5|hdf5)$/i.test(own) && !/_dict_/i.test(own)) return own;
  return '';
}

/**
 * One entry per selected phase, for the dialog's phase dropdown.
 *
 * Entries without a master are kept (not hidden) and flagged, so the dialog can
 * say *why* a phase cannot be generated instead of silently omitting it.
 */
export function dictionaryTargets(phases = [], allFiles = []) {
  return phases.map((phase, index) => {
    const masterPath = masterPathForPhase(phase, allFiles);
    return {
      index,
      phasePath: phase?.path || '',
      label: phase?.display_label || phase?.formula || phase?.filename || '—',
      formula: phase?.formula || '',
      masterPath,
      masterFilename: masterPath.split(/[\\/]/).pop() || '',
      hasMaster: !!masterPath,
    };
  });
}

// ---------------------------------------------------------------------------
// Which dictionary a phase will actually be indexed with
// ---------------------------------------------------------------------------

/** % difference between two PC vectors (Euclidean distance / sqrt(3) * 100). */
export function pcDeltaPercent(pcA, pcB) {
  if (!pcA || !pcB || pcA.length < 3 || pcB.length < 3) return null;
  const dx = pcA[0] - pcB[0];
  const dy = pcA[1] - pcB[1];
  const dz = pcA[2] - pcB[2];
  return (Math.sqrt(dx * dx + dy * dy + dz * dz) / Math.sqrt(3)) * 100;
}

/** Does this dictionary's detector match the loaded patterns? */
export function dictShapeMatches(entry, detectorShape) {
  if (!detectorShape) return true;          // unknown dataset -> can't judge
  const s = entry?.detector_shape;
  if (!Array.isArray(s) || s.length < 2) return true;   // unknown dict -> can't judge
  return Number(s[0]) === Number(detectorShape[0]) && Number(s[1]) === Number(detectorShape[1]);
}

/** Largest camera-tilt error we treat as harmless, in degrees.
 *  Measured on the real Si master at 118x118: a 1 deg error already drops the
 *  NCC against the correct pattern to 0.45, 2 deg to 0.23. Half a degree is
 *  the point where the match is still clearly the best one. */
export const TILT_TOLERANCE_DEG = 0.5;

/**
 * Was this dictionary simulated for the camera geometry we are indexing with?
 *
 * Older sidecars have no `detector_tilt` at all — reported as `null` by
 * discovery. That is "unknown", not "flat": we cannot judge it, so we let it
 * through rather than hiding a possibly fine dictionary.
 */
export function dictGeometryMatches(entry, geom) {
  if (!geom) return true;
  const t = entry?.detector_tilt;
  if (t === null || t === undefined) return true;       // unknown -> can't judge
  const want = Number(geom.detectorTilt ?? 0);
  if (!Number.isFinite(want)) return true;
  return Math.abs(Number(t) - want) <= TILT_TOLERANCE_DEG;
}

/**
 * Dictionary whose camera tilt we cannot verify, on a tilted detector.
 *
 * Every dictionary this app wrote before the tilt was plumbed through was
 * simulated at 0 deg — so on a tilted detector a missing `detector_tilt` is a
 * strong hint the file is unusable. It is not proof (the legacy CPU generator
 * was handed a real detector object), so this does not block the run: it ranks
 * such a dictionary BELOW one with a verified matching tilt, and the UI flags
 * it so the user knows to regenerate.
 */
export function dictGeometryUnknown(entry, geom) {
  if (!geom) return false;
  const want = Number(geom.detectorTilt ?? 0);
  if (!Number.isFinite(want) || Math.abs(want) <= TILT_TOLERANCE_DEG) return false;
  const t = entry?.detector_tilt;
  return t === null || t === undefined;
}

/**
 * Best dictionary for a phase: one that FITS the detector first, then lowest
 * PC deviation, then finest angular resolution.
 *
 * Shape comes first because a mismatch is not "worse", it is unusable — the
 * indexer rejects it outright (backend/dict_gpu/pipeline/indexer.py). Sorting
 * by PC alone would happily pick a 128x156 dictionary for 118x118 data.
 */
export function selectBestDict(dicts = [], currentPc = null, detectorShape = null, geom = null) {
  if (!dicts.length) return null;
  const usable = dicts.filter(
    d => dictShapeMatches(d, detectorShape) && dictGeometryMatches(d, geom)
  );
  // A verified-matching camera tilt always beats an unverifiable one, whatever
  // its PC. Only fall back to the unverifiable ones if there is nothing else.
  const verified = usable.filter(d => !dictGeometryUnknown(d, geom));
  const pool = verified.length ? verified : usable;
  if (!pool.length) return null;
  return pool.reduce((best, d) => {
    const bestDelta = pcDeltaPercent(currentPc, best?.pc);
    const dDelta = pcDeltaPercent(currentPc, d?.pc);
    if (dDelta !== null && (bestDelta === null || dDelta < bestDelta)) return d;
    if (dDelta === bestDelta || (dDelta === null && bestDelta === null)) {
      const bestRes = best?.resolution ?? best?.resolution_deg ?? Infinity;
      const dRes = d?.resolution ?? d?.resolution_deg ?? Infinity;
      return dRes < bestRes ? d : best;
    }
    return best;
  });
}

/**
 * The dictionary path a phase will be INDEXED with, or '' when there is none.
 *
 * `explicit` is the user's clicked choice keyed by phase path. Everything else
 * falls back to selectBestDict. Both the phase card's "chosen" badge and the
 * indexing request must go through here: they used to disagree, because the
 * card computed the fallback locally while the request read only `explicit` —
 * so a phase whose dictionary was generated AFTER it was added showed "chosen"
 * while the run silently shipped its MASTER path instead, and died with
 * "phase_list required for Dictionary indexing".
 */
export function resolveDictPathForPhase(phase, allFiles = [], opts = {}) {
  const { explicit = {}, currentPc = null, detectorShape = null, geom = null } = opts;
  const chosen = explicit[phase?.path];
  if (chosen) return chosen;
  return selectBestDict(dictsForPhase(phase, allFiles), currentPc, detectorShape, geom)?.path || '';
}

/**
 * Phase paths exactly as they must land in `master_h5_paths`, with each
 * phase's master replaced by the dictionary it will be indexed with.
 *
 * `phaseFiles` and `phases` are index-aligned by the page's toggle handlers.
 * A phase with no usable dictionary keeps its own path so the caller can spot
 * it (see unresolvedDictPhases) instead of it vanishing from the request.
 */
export function resolvePhasePaths(phaseFiles = [], phases = [], allFiles = [], opts = {}) {
  return phaseFiles.map((p, i) => {
    const phase = phases[i]?.path === p ? phases[i] : phases.find(x => x?.path === p);
    if (!phase) return p;
    return resolveDictPathForPhase(phase, allFiles, opts) || p;
  });
}

/**
 * Phases that would be sent WITHOUT a dictionary — the run cannot work for
 * these. Returns [{label, reason}] so the page can say which and why before
 * starting, rather than failing per-phase deep inside the backend.
 */
export function unresolvedDictPhases(phases = [], allFiles = [], opts = {}) {
  const { detectorShape = null, geom = null } = opts;
  const out = [];
  for (const phase of phases) {
    if (resolveDictPathForPhase(phase, allFiles, opts)) continue;
    const all = dictsForPhase(phase, allFiles);
    const label = phase?.display_label || phase?.formula || phase?.filename || '—';
    // Distinguish the three cases so the message can name the actual problem.
    let reason = 'none';
    if (all.length) {
      const shapeOk = all.filter(d => dictShapeMatches(d, detectorShape));
      reason = shapeOk.length ? 'tilt' : 'shape';
    }
    out.push({
      label,
      reason,
      // What the existing dictionaries offer, for the "wrong detector" message.
      shapes: all
        .map(d => (Array.isArray(d.detector_shape) ? d.detector_shape.join('x') : null))
        .filter(Boolean),
      needed: detectorShape ? detectorShape.join('x') : '',
      tilts: all
        .map(d => (d.detector_tilt === null || d.detector_tilt === undefined
          ? null : `${Number(d.detector_tilt).toFixed(2)}°`))
        .filter(Boolean),
      neededTilt: geom ? `${Number(geom.detectorTilt ?? 0).toFixed(2)}°` : '',
    });
  }
  return out;
}

/**
 * Detector shape (rows, cols) from a dataset's `signal_shape`.
 *
 * hyperspy reports `axes_manager.signal_shape` as (width, height); the
 * dictionary pipeline's `detector_shape` is (nrows, ncols) — see
 * backend/dictionary_gpu/detector.py. So this reverses. Getting this backwards
 * on a non-square detector silently produces a transposed dictionary that
 * correlates against nothing.
 */
export function detectorShapeFromSignalShape(signalShape) {
  if (!Array.isArray(signalShape) || signalShape.length < 2) return null;
  const w = Number(signalShape[0]);
  const h = Number(signalShape[1]);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
  return [h, w];
}
