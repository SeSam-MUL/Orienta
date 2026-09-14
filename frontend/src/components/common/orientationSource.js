/**
 * How a finished map's orientation provenance is shown.
 *
 * The backend used to report only "spherical" or "hough", and "hough" the
 * moment the pseudo-symmetry resolver substituted anything. Since the render
 * arbitration (2026-09-09) and the decode fix (2026-09-10) a run routinely
 * keeps BOTH: the resolver substitutes on the z_rot==2 phases, then the arbiter
 * hands disputed pixels back to the sphere wherever the sphere renders better
 * (on the ICAA20 crop: all 676 of them). The backend now counts what survived
 * and can report "mixed"; the counts are in orientation_source_reason.
 *
 * One resolver so the badge and the "this pixel may sit on a wrong variant"
 * hint cannot disagree about what a source value means.
 */

/** Sources for which the provenance badge is shown, and their i18n keys. */
export const ORIENTATION_SOURCE_BADGES = {
  hough: { labelKey: 'matchesDialog.orientationFromHough',
           tipKey: 'matchesDialog.orientationHoughTip' },
  mixed: { labelKey: 'matchesDialog.orientationFromMixed',
           tipKey: 'matchesDialog.orientationMixedTip' },
};

/**
 * @param {string|undefined} source - metadata.orientation_source
 * @returns {{labelKey: string, tipKey: string}|null} null = no badge
 */
export function orientationSourceBadge(source) {
  return ORIENTATION_SOURCE_BADGES[source] || null;
}

/**
 * Does this map carry orientations that did NOT come from the spherical
 * correlation? Then the clicked pixel may sit on a wrong pseudo-variant and
 * the manual flip entry is worth highlighting.
 */
export function orientationSourceIsSuspicious(source) {
  return source === 'hough' || source === 'mixed';
}
