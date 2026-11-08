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
 * Where a sentence that is the same on every page is kept, after the page's own namespace: `common`
 * (a hexagonal scan met code that assumes a square one; a measurement or pattern source a step
 * cannot read), then `ebsdviewer`, which holds the APEX refusals (a pairing that conflicts with the
 * pattern file, a metadata file that is gone). Any route that loads a file can answer with one of
 * those, whichever page asked: the header switcher, the Indexing dropdown, the Phase Map gallery.
 */
const SHARED_NAMESPACES = ['common', 'ebsdviewer'];

/**
 * The words for a code in the reader's language, or null when this build has none.
 * Looks in the page's namespace first, then in the shared ones (`SHARED_NAMESPACES`).
 * @param {string}   code
 * @param {Function} t    i18next's t, bound to any namespace
 * @param {string}   ns   the namespace holding the page's own `errors.<code>`
 */
export function codedText(code, t, ns = 'common') {
  if (!code || typeof t !== 'function') return null;
  // An empty defaultValue is not a value when returnEmptyString is false,
  // so i18next would hand back the key. A sentinel no translation can
  // equal is the only reliable "missing".
  const MISS = '\u0000miss';
  const out = t(`${ns}:errors.${code}`, { defaultValue: MISS });
  if (out !== MISS) return out;
  for (const shared of SHARED_NAMESPACES) {
    if (shared === ns) continue;
    const text = t(`${shared}:errors.${code}`, { defaultValue: MISS });
    if (text !== MISS) return text;
  }
  return null;
}

/**
 * The translated text of the code an error carries, or null when it has no code or this build has
 * no text for it. For a reader that keeps its own fallback sentence (what it showed before codes
 * existed) and only wants to put the coded text in front of it.
 */
export function codedProblemText(err, t, ns = 'common') {
  try {
    return codedText(problemCode(err), t, ns);
  } catch {
    return null;
  }
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
    const coded = codedText(problemCode(err), t, ns);
    if (coded !== null) return coded;
    return problemProse(err);
  } catch {
    return err?.message || '';
  }
}

/** The text of a Blob, or '' (jsdom's Blob has no `.text()`, a browser's has). */
async function blobText(blob) {
  if (typeof blob.text === 'function') return blob.text();
  return new Promise((resolve) => {
    try {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ''));
      reader.onerror = () => resolve('');
      reader.readAsText(blob);
    } catch {
      resolve('');
    }
  });
}

/**
 * `problemText` for a request whose response type is `blob` (a file download).
 *
 * On such a response axios hands the error body over as a Blob, so `response.data.detail` is
 * undefined and the user is left with "Request failed with status code 400". The header with the
 * code is readable all the same, so a coded refusal is worded exactly as `problemText` words it.
 * Without a code, a body that is JSON with a string `detail` (or a `detail.message`) is read and
 * shown; anything else ends in the error's own message.
 */
export async function blobProblemText(err, t, ns = 'common') {
  try {
    const coded = codedText(problemCode(err), t, ns);
    if (coded !== null) return coded;
    const data = err?.response?.data;
    if (data && typeof data === 'object' && typeof data.size === 'number'
        && typeof data.slice === 'function') {
      try {
        const body = JSON.parse(await blobText(data));
        const detail = body?.detail;
        if (typeof detail === 'string' && detail) return detail;
        if (detail && typeof detail === 'object' && typeof detail.message === 'string'
            && detail.message) {
          return detail.message;
        }
      } catch {
        // not JSON (or unreadable): the error's own message below
      }
      return err?.message || '';
    }
    return problemProse(err);
  } catch {
    return err?.message || '';
  }
}

/**
 * The reason a batch file/job failed, in the reader's language: the translation of its
 * `error_code` (under `errors.<code>` of `ns`, then of the shared namespaces: a file that fails to
 * load can carry any APEX refusal code), else the backend's own sentence.
 * @param {{error_code?: string|null, error?: string|null, error_msg?: string|null}} entry
 */
export function failureText(entry, t, ns = 'common') {
  try {
    const coded = codedText(entry?.error_code, t, ns);
    if (coded !== null) return coded;
  } catch {
    // fall through to the prose
  }
  return entry?.error || entry?.error_msg || '';
}
