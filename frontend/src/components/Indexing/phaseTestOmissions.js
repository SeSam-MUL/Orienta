/**
 * Why a phase you asked for is not in the phase tester's table.
 *
 * THERE ARE TWO REASONS AND THEY ARE NOT THE SAME ANSWER.
 *
 *   excluded     the EDS chemistry prefilter ruled it out. The test ran
 *                and said no; the number beside it is the fit.
 *   unsupported  it could not be tested at all -- for `no_sht`, because
 *                there is no simulated master to compare the measured
 *                pattern against. Nothing was tried.
 *
 * The backend separated these into two fields on purpose (c1). The dialog
 * showed only the first, so a phase like `Al` -- in the library, ticked by
 * the user, with no `.sht` -- was silently absent from its own results.
 * That reads as "the test rejected it", when the truth is "there is a file
 * you can go and make". The two send a reader to opposite places.
 *
 * A MODULE RATHER THAN A BRANCH IN THE DIALOG, because the dialog needs
 * api clients, three stores and five child components before it will
 * render, so the decision could not otherwise be tested at all -- and an
 * untested branch is how the first reason came to be shown alone.
 */

/** Reasons we can put into words. Anything else is passed through as is. */
export const KNOWN_REASONS = ['no_sht'];

/**
 * `{excluded, unsupported}` from a phase-test result, never merged.
 *
 * Both lists are returned even when empty, so a caller cannot accidentally
 * treat "no exclusions" and "no answer yet" alike.
 */
export function omissions(result) {
  const r = result || {};
  return {
    excluded: (r.excluded || []).map((e) => ({
      key: e.phase_key ?? e.key ?? null,
      label: e.display_formula || e.formula || e.phase_key || e.key || null,
      fit: typeof e.chemistry_fit === 'number' ? e.chemistry_fit : null,
    })),
    unsupported: (r.unsupported || []).map((u) => ({
      key: u.key ?? null,
      reason: u.reason || 'unknown',
      // `known` decides whether the screen can explain it or must quote
      // it. Quoting an unknown reason is better than inventing a sentence
      // for it, and better than dropping the row.
      known: KNOWN_REASONS.includes(u.reason),
    })),
  };
}

/** True when the tester left something out for either reason. */
export function anyOmitted(result) {
  const { excluded, unsupported } = omissions(result);
  return excluded.length > 0 || unsupported.length > 0;
}
