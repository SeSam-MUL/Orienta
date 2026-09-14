/**
 * Why the phase suggestion has nothing to offer.
 *
 * A tester reported on 2026-08-25 that a pixel reading 75 at% Mg / 25 at% Si
 * showed no Mg2Si although the CIF was there. It WAS there, it scored 0.92 —
 * and was then rejected by the enrichment gate, which asks whether Si is
 * enriched over the map's own median. On a map acquired with only Mg and Si
 * that median IS Mg2Si, so the phase filling the map is the one phase that
 * can never clear the bar. The panel showed "no phases suggested" and not one
 * word more, so there was nothing to act on and nothing to report.
 *
 * The backend now answers with a stable `code` plus the numbers behind it
 * (`cif_no_match`). Translating the code is what makes this a sentence in the
 * user's language; `message` is the English prose the backend always sends,
 * used only when a code arrives that this build has no string for — a new
 * backend against an old frontend must still say something true.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

import { colors, alpha, spacing } from '../../theme/components';

/** i18n key per backend code. Anything not listed falls back to `message`. */
const KEY_FOR_CODE = {
  not_enriched_over_background: 'suggest.noMatch.notEnriched',
  below_min_score: 'suggest.noMatch.belowMinScore',
  no_candidate_phase: 'suggest.noMatch.noCandidate',
  empty_library: 'suggest.noMatch.emptyLibrary',
  no_chemistry: 'suggest.noMatch.noChemistry',
};

/**
 * The sentence for one reason, in the caller's language.
 *
 * Numbers are formatted for `lang` rather than interpolated raw: the German
 * file writes 0,01 and 1,3 throughout, so a raw 0.30 in the middle of a
 * German sentence would be the only decimal point on the page.
 */
export function reasonText(reason, t, lang = 'en') {
  if (!reason) return '';
  const key = KEY_FOR_CODE[reason.code];
  if (!key) return reason.message || '';
  const num = (v, digits) => (v == null || Number.isNaN(Number(v))
    ? v
    : new Intl.NumberFormat(lang, {
      minimumFractionDigits: 0, maximumFractionDigits: digits,
    }).format(Number(v)));
  return t(key, {
    phase: reason.phase,
    element: reason.element,
    measured: num(reason.measured_at_pct, 1),
    background: num(reason.background_at_pct, 1),
    required: num(reason.required_at_pct, 1),
    score: num(reason.score_without_map ?? reason.score, 2),
    min: num(reason.min_score, 2),
    elements: (reason.elements || []).join(', '),
    defaultValue: reason.message || '',
  });
}

export default function SuggestNoMatch({ reason }) {
  const { t, i18n } = useTranslation('eds');
  const text = reasonText(reason, t, i18n?.language || 'en');
  if (!text) return null;
  return (
    <div
      data-testid="suggest-no-match"
      style={{
        marginTop: spacing.innerSpacing,
        padding: '6px 8px',
        borderRadius: 4,
        fontSize: '8pt',
        lineHeight: 1.45,
        color: colors.textSecondary,
        background: alpha(colors.orange, 8),
        border: `1px solid ${alpha(colors.orange, 25)}`,
      }}
    >
      {text}
    </div>
  );
}
