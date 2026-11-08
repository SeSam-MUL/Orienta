/**
 * Pure helpers for what the Crystal Database says about a CIF.
 *
 * The backend sends warnings as CODES (`fit_warning_codes`, in the same order
 * as the English sentences in `fit_warnings`). The interface translates the
 * code; the English sentence is only the fallback for a code this build does
 * not know yet, so a newer backend never shows a bare identifier.
 */

/**
 * Longest DOI or citation the backend accepts (`MAX_CITATION_CHARS`): the text ends
 * up in the header of every SHT file, whose length field is a signed 16-bit integer.
 */
export const MAX_CITATION_CHARS = 2000;

/** Longest DOI the backend accepts (`MAX_DOI_CHARS`). */
export const MAX_DOI_CHARS = 200;

/**
 * A DOI as typed or pasted: bare, `doi:` prefixed, or as a doi.org link.
 * The same rule as `normalise_doi` in the backend (10.<3-9 digits>/...).
 */
const DOI_PATTERN = /^(?:https?:\/\/(?:dx\.)?doi\.org\/|doi:\s*)?(10\.\d{3,9}\/\S+)$/i;

/**
 * One text box, two meanings: a DOI goes to the CIF's DOI field, anything else
 * is a citation and goes to its reference text.
 */
export function splitDoiOrCitation(text) {
  const value = String(text ?? '').trim();
  if (!value) return { doi: '', reference: '' };
  const m = DOI_PATTERN.exec(value);
  return m ? { doi: m[1], reference: '' } : { doi: '', reference: value };
}

/** `0.01 Å: Cm (8); 0.05 Å: Fm-3m (225)` for the warning text. */
export function ladderText(ladder) {
  return (ladder || [])
    .map(r => `${r.symprec} \u00c5: ${r.number == null ? '?' : `${r.symbol} (${r.number})`}`)
    .join('; ');
}

/**
 * The tolerances worth offering as a choice: the first tolerance of every distinct
 * group above P 1 on the ladder, in ladder order.
 */
export function ladderChoices(ladder) {
  const seen = new Set();
  const out = [];
  for (const r of ladder || []) {
    if (r.number > 1 && !seen.has(r.number)) {
      seen.add(r.number);
      out.push(r);
    }
  }
  return out;
}

/** Translated warning sentences of a `cif_info` answer, in backend order. */
export function translateFitWarnings(t, info) {
  const codes = info?.fit_warning_codes;
  const english = info?.fit_warnings || [];
  if (!Array.isArray(codes)) return english;       // older backend: sentences only
  const detected = info?.detected_space_group || {};
  const header = info?.header_contradiction || {};
  return codes.map((code, i) => t(`fitWarnings.${code}`, {
    defaultValue: english[i] ?? code,
    symbol: detected.symbol ?? '',
    number: detected.number ?? '',
    symprec: detected.symprec ?? '',
    headerNumber: header.number ?? '',
    detail: info?.p1_check_error ?? '',
    headerSymbol: header.symbol ?? '',
    ladder: ladderText(info?.symmetry_ladder),
  }));
}

/** The `detail` of a failed request as one line of text. */
export function errorText(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail.message === 'string') return detail.message;
  return err?.message || '';
}

/** The machine code of a failed request (`reference_missing`, ...), or ''. */
export function errorCode(err) {
  const detail = err?.response?.data?.detail;
  return detail && typeof detail === 'object' && typeof detail.code === 'string' ? detail.code : '';
}

/** An extra field of a coded refusal (`existing` for same_content_exists), or ''. */
export function errorField(err, name) {
  const detail = err?.response?.data?.detail;
  return detail && typeof detail === 'object' && detail[name] != null ? String(detail[name]) : '';
}
