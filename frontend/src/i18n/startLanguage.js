/**
 * Which language to start in, given what is stored and what the setup handed over.
 *
 * With nothing stored the app starts in English. The setup wizard speaks the
 * user's language — its own or the one they picked — and until this existed a
 * German user went from a German setup straight into an English program. The
 * shell now hands the wizard's language over ONCE, in the URL of the first
 * load: `?lang=de`, plus `&langExplicit=1` when the user picked it in the
 * wizard rather than the wizard starting in it.
 *
 * The rule:
 *   - nothing stored            -> the handed-over language
 *   - stored, handed explicit   -> the handed-over language (the newest choice)
 *   - stored, handed automatic  -> the stored one; a repair must not quietly
 *                                  undo a choice made inside the app because
 *                                  Windows speaks something else
 *
 * Pure: no window, no storage. Returns the language, whether to persist it, and
 * the query string with the hand-over removed, so a reload or a bookmark does
 * not apply it a second time.
 */
export function resolveStartLanguage({ stored, search, supported }) {
  const params = new URLSearchParams(search || '');
  const handed = params.get('lang');
  const explicit = params.get('langExplicit') === '1';
  const hadHandOver = params.has('lang') || params.has('langExplicit');
  params.delete('lang');
  params.delete('langExplicit');
  const rest = params.toString();
  const cleanSearch = rest ? `?${rest}` : '';

  const valid = (code) => Boolean(code) && supported.includes(code);
  const storedOk = valid(stored) ? stored : null;

  if (valid(handed) && (explicit || !storedOk)) {
    return { lang: handed, persist: true, cleanSearch, hadHandOver };
  }
  return { lang: storedOk || 'en', persist: false, cleanSearch, hadHandOver };
}
