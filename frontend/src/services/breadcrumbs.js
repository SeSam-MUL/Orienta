/**
 * Breadcrumb trail — the few dozen things that happened before an error.
 *
 * A bug report that says only "TypeError: x is undefined" costs a round of
 * questions. The same report with "opened Indexing → loaded SampleB →
 * POST /api/indexing/start → 500" is usually diagnosable on sight. Every
 * error we report carries this trail with it.
 *
 * In-memory ring buffer, MAX_CRUMBS entries, no persistence: it exists to be
 * attached to an error report, not to be a second log.
 */

const MAX_CRUMBS = 60;

/**
 * High-frequency paths that would otherwise BE the trail. The health poll
 * alone (every 5 s) would push out everything interesting within a minute.
 * Same reasoning as HTTPTimingMiddleware._EXCLUDED_PREFIXES in the backend.
 */
const IGNORED_PATH_PARTS = [
  '/api/health',
  '/api/system/frontend-error',
  '/progress',
  '/status',
];

const trail = [];

function isIgnoredPath(url) {
  if (!url) return false;
  return IGNORED_PATH_PARTS.some((part) => url.includes(part));
}

/**
 * Record one event.
 * @param {string} type - 'nav' | 'http' | 'http-error' | 'ui-error' | 'action'
 * @param {string} message - human-readable one-liner
 * @param {object} [data] - optional extra fields (kept small)
 */
export function addBreadcrumb(type, message, data = {}) {
  try {
    trail.push({ time: Date.now(), type, message: String(message), ...data });
    if (trail.length > MAX_CRUMBS) trail.shift();
  } catch {
    // A breadcrumb must never break the thing it is observing.
  }
}

/** Record an HTTP call. Returns false when the path was ignored. */
export function addHttpBreadcrumb({ method, url, status, durationMs, detail }) {
  if (isIgnoredPath(url)) return false;
  const failed = status === undefined || status === null || status >= 400;
  const parts = [`${(method || 'GET').toUpperCase()} ${url}`];
  parts.push(status ? `→ ${status}` : '→ (no response)');
  if (durationMs !== undefined) parts.push(`${Math.round(durationMs)}ms`);
  if (detail) parts.push(`· ${String(detail).slice(0, 200)}`);
  addBreadcrumb(failed ? 'http-error' : 'http', parts.join(' '));
  return true;
}

/** Raw copy of the trail, oldest first. */
export function getBreadcrumbs() {
  return trail.slice();
}

/**
 * Render the trail as text, newest last, with times relative to `now`.
 * This is what ends up in logs/orienta.log under the error.
 */
export function formatBreadcrumbs(now = Date.now()) {
  return trail
    .map((c) => {
      const ago = ((now - c.time) / 1000).toFixed(1);
      return `  -${ago.padStart(6)}s [${c.type}] ${c.message}`;
    })
    .join('\n');
}

/** Test hook. */
export function _clearBreadcrumbs() {
  trail.length = 0;
}
