/**
 * Dates and times in the language the app is set to.
 *
 * `toLocaleTimeString()` with no argument uses the BROWSER's locale, not the
 * app's. The M5 tester ran Orienta in German on a Mac set to English and read
 * "9/25/2026, 10:10:02 AM" in the upload history and in every log line
 * (report, section 5, point 4). The app knew it was speaking German; the
 * timestamps did not ask.
 *
 * These take the language from i18n, so the one setting that is visible to
 * the user decides. They never throw: an unparseable value comes back as an
 * em dash rather than "Invalid Date", and a locale the runtime does not know
 * falls back to the browser default rather than crashing the render.
 */
import i18n from './index';

/** The BCP-47 tag for the app's current language. */
export function currentLocale() {
  // i18next stores 'de', 'ja', 'zh'; all are valid BCP-47 tags on their own.
  return i18n?.language || undefined;
}

function safe(value, fn) {
  // new Date(null) is the epoch, not an invalid date, and new Date('') is
  // invalid on some runtimes and the epoch on others. A missing value is a
  // missing value in both cases: a history row with no start time must not
  // read "1/1/1970".
  if (value === null || value === undefined || value === '') return '—';
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return '—';
  try {
    return fn(d, currentLocale());
  } catch {
    // An unsupported locale tag: better the browser default than no render.
    try {
      return fn(d, undefined);
    } catch {
      return '—';
    }
  }
}

/** Time only, e.g. for a log line. */
export function formatTime(value = new Date()) {
  return safe(value, (d, loc) => d.toLocaleTimeString(loc));
}

/** Date and time, e.g. for a row in a history table. */
export function formatDateTime(value) {
  return safe(value, (d, loc) => d.toLocaleString(loc));
}

/** Date only. */
export function formatDate(value) {
  return safe(value, (d, loc) => d.toLocaleDateString(loc));
}

/**
 * Numbers inside a translated sentence, formatted for that sentence.
 *
 * The backend sends raw numbers in `params`. Formatting them there means
 * "median 12,345 counts/pixel" arrives in a German sentence where a comma is
 * the decimal separator, so the reader sees twelve point three four five.
 * Strings pass through untouched.
 */
export function localiseParams(params) {
  if (!params) return {};
  const out = {};
  for (const [k, v] of Object.entries(params)) {
    out[k] = typeof v === 'number' && Number.isFinite(v)
      ? v.toLocaleString(currentLocale())
      : v;
  }
  return out;
}
