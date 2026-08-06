/**
 * Should the Pattern-Match dialog offer to adopt a re-indexed orientation?
 *
 * Extracted from PhaseMapPage on 2026-08-06 because the inline version had no
 * test and was wrong: it offered adoption whenever the re-indexed orientation
 * differed by more than 0.05 degrees, WITHOUT comparing how well either one
 * actually renders. Measured on ProbeB pixel (323,144): stored orientation
 * R = 0.262, re-indexed candidate R = 0.140, 44.13 degrees apart — the dialog
 * offered to replace the good orientation with the bad one, and applying it
 * degraded the grain.
 *
 * Why the candidate can be worse at all: the re-index maximises the SHT
 * spherical correlation (0.457 vs 0.347 at that pixel), which is NOT the same
 * ranking as the rendered detector pattern. Orientation candidates must be
 * judged by render-NCC — a lesson this project already paid for once during the
 * pseudo-symmetry work.
 */

/**
 * Minimum render-NCC improvement before a swap is offered. Same value the
 * map-wide variant unifier uses for "a clear margin"
 * (backend/spherical_gpu/pipeline/variant_unification.py, margin_clear).
 */
export const R_ADOPT_MARGIN = 0.03;

/** Below this the two orientations are the same one, not a variant flip. */
export const MIN_DISORIENTATION_DEG = 0.05;

/**
 * @param {object}  a
 * @param {boolean} a.samePhase          candidate and stored are the same phase
 * @param {*}       a.quat               candidate quaternion (must be an array)
 * @param {?number} a.disorientationDeg  angle between candidate and stored
 * @param {?number} a.rCandidate         render-NCC of the candidate
 * @param {?number} a.rStored            render-NCC of the stored orientation
 * @param {number}  [a.margin]
 * @returns {{canAdopt: boolean, rejected: boolean, reason: string}}
 *   `rejected` means: there IS a candidate, but it does not match better — the
 *   UI should say so with both numbers rather than silently show nothing.
 */
export function adoptDecision({
  samePhase,
  quat,
  disorientationDeg,
  rCandidate,
  rStored,
  margin = R_ADOPT_MARGIN,
} = {}) {
  const hasCandidate =
    !!samePhase &&
    Array.isArray(quat) &&
    (disorientationDeg ?? 0) > MIN_DISORIENTATION_DEG;

  if (!hasCandidate) {
    return { canAdopt: false, rejected: false, reason: 'no-candidate' };
  }
  if (rCandidate == null || rStored == null || Number.isNaN(rCandidate) || Number.isNaN(rStored)) {
    // No comparable evidence — refuse rather than guess. Silently adopting on
    // an unknown match is exactly the failure this function exists to prevent.
    return { canAdopt: false, rejected: true, reason: 'no-r-score' };
  }
  if (rCandidate >= rStored + margin) {
    return { canAdopt: true, rejected: false, reason: 'improves' };
  }
  return { canAdopt: false, rejected: true, reason: 'not-better' };
}
