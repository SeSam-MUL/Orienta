/**
 * edsPriorParams.js
 *
 * Request-shape helper for the EDS chemistry prior at indexing.
 *
 * History (2026-08-05): this used to be a PER-PHASE slider (0-100 %). The
 * backend weight is `w_p = (1 - s_p) + s_p * chemistry_fit`, so a phase left at
 * 0 keeps weight EXACTLY 1.0 and is immune while the phases the user raised get
 * penalised. Raising it on Al+Si only was measured to inflate a third phase from
 * 1.5 % to 10 % of the map. There are only two meaningful states — off, and on
 * for EVERY phase — so the UI is one switch and this helper emits 1.0 across the
 * board.
 *
 * Kept as a pure function (not inlined into buildParams) so the on/off request
 * shape is testable without mounting the whole Indexing page.
 */

/**
 * @param {object}   opts
 * @param {boolean}  opts.enabled       the global "use EDS chemistry" switch
 * @param {boolean}  opts.edsAvailable  the loaded file actually has EDS data
 * @param {string[]} opts.phasePaths    paths as they land in cif_paths /
 *                                      master_h5_paths / sht_paths (already
 *                                      remapped to selected dictionary paths)
 * @returns {Object<string, number>|null} `{path: 1.0}` for every phase, or null
 *          when the field must be omitted entirely (keeps an off-run's payload
 *          byte-identical to a build without this feature).
 */
export function buildEdsPhaseStrengths({ enabled, edsAvailable, phasePaths } = {}) {
  if (!enabled || !edsAvailable) return null;
  const paths = (phasePaths || []).filter(Boolean);
  if (paths.length === 0) return null;
  return Object.fromEntries(paths.map((p) => [p, 1.0]));
}

/**
 * Would a run with these settings actually ship the chemistry prior?
 * Used to decide whether a blocking pre-flight failure may hold Start back —
 * a blocked pre-flight is irrelevant when the field is not sent at all.
 */
export function edsPriorActive({ enabled, edsAvailable, phasePaths } = {}) {
  return buildEdsPhaseStrengths({ enabled, edsAvailable, phasePaths }) !== null;
}
