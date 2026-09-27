/**
 * System-status warnings, in the user's language where we have one.
 *
 * The backend sends `warnings` (English prose, every producer) and
 * `warning_items` (the subset that carries `{code, values, message}`). The
 * Settings → System Status used to show the English sentence because prose was
 * all there was — reported from the 0.4.5 laptop test, where the GPU blocksize
 * warning appeared in English mid-German. (The standalone installer wizard,
 * electron/setup/, has its own warningCode scheme and never shows these.)
 *
 * Rules, in order:
 *   1. a warning that matches an item WITH a translation -> the translation;
 *   2. an item whose code has no translation yet -> its English message, so a
 *      new backend code shows up as prose instead of as a missing-key string;
 *   3. everything else -> unchanged prose.
 *
 * Matching is by message text, because that is the only thing the two lists
 * share. Both come from the same call, so the strings are identical.
 */

/** i18n key for a warning code, under the settings namespace. */
export const warningKey = (code) => `settings:systemStatus.warnings.${code}`;

/**
 * @param {string[]} warnings        prose warnings, in display order
 * @param {Array<{code: string, values: object, message: string}>} warningItems
 * @param {(key: string, opts?: object) => string} t
 * @param {(key: string) => boolean} [exists]  i18n `exists`, when available
 * @returns {string[]} what to display
 */
export function translateSystemWarnings(warnings, warningItems, t, exists) {
  const byMessage = new Map(
    (warningItems || [])
      .filter((it) => it && typeof it.message === 'string' && it.code)
      .map((it) => [it.message, it]),
  );
  return (warnings || []).map((message) => {
    const item = byMessage.get(message);
    if (!item) return message;
    const key = warningKey(item.code);
    if (typeof exists === 'function' && !exists(key)) return item.message;
    const translated = t(key, { ...(item.values || {}) });
    // i18next returns the key itself when nothing is registered for it.
    return !translated || translated === key ? item.message : translated;
  });
}
