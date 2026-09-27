/**
 * What an API error says, in the reader's language.
 *
 * The backend's error prose reaches the screen directly: 159 places read
 * `err.response.data.detail` and show it. That text is English, so the M5
 * tester met "No indexing result or analysis dataset available. Load data
 * first." in an otherwise German window.
 *
 * Putting a `{code, message}` object in `detail` would have changed the shape
 * all 159 of those readers see, turning each of them into "[object Object]"
 * on the converted routes. So the prose stays exactly where it was and the
 * machine-readable code travels in the `X-Orienta-Code` header. A reader that
 * wants to translate calls this; every reader that does not is unaffected.
 *
 * The fallback chain is the whole point: a code this build has no text for, a
 * backend older than this frontend, a network error with no response at all —
 * each ends in something true rather than in a translation key.
 */

export const CODE_HEADER = 'x-orienta-code';

/** The stable code an error carries, or null. */
export function problemCode(err) {
  const headers = err?.response?.headers;
  if (!headers) return null;
  try {
    // axios lower-cases header names; a raw fetch Headers object needs .get().
    const raw = typeof headers.get === 'function'
      ? headers.get(CODE_HEADER)
      : (headers[CODE_HEADER] ?? headers[CODE_HEADER.toUpperCase()]);
    return raw ? String(raw) : null;
  } catch {
    return null;
  }
}

/** The backend's own English sentence, or ''. */
export function problemProse(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  // Some routes answer with a dict; show its message rather than "[object Object]".
  if (detail && typeof detail === 'object' && typeof detail.message === 'string') {
    return detail.message;
  }
  return err?.message || '';
}

/**
 * @param {Error}    err  an axios error
 * @param {Function} t    i18next's t, bound to any namespace
 * @param {string}   ns   the namespace holding `errors.<code>`
 */
export function problemText(err, t, ns = 'common') {
  // Every caller is inside a `catch`, and its job is to put something on the
  // screen. A throw here would escape that handler and leave the panel empty:
  // a blank failure instead of an English one. So the body is guarded, and
  // the docstring's promise that each path "ends in something true" holds.
  try {
    const code = problemCode(err);
    if (code && typeof t === 'function') {
      // An empty defaultValue is not a value when returnEmptyString is false,
      // so i18next would hand back the key. A sentinel no translation can
      // equal is the only reliable "missing".
      const MISS = '\u0000miss';
      const out = t(`${ns}:errors.${code}`, { defaultValue: MISS });
      if (out !== MISS) return out;
    }
    return problemProse(err);
  } catch {
    return err?.message || '';
  }
}
