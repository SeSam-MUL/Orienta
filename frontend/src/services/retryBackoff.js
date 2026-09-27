/**
 * One schedule for "keep asking until the backend answers".
 *
 * Both places that need it are the same situation: the first start after a
 * runtime update, when the window is up and uvicorn is not. A failed request
 * is not an answer, so the caller retries — doubling the wait so a genuinely
 * dead backend costs one request a minute instead of a flood.
 *
 * Extracted from `hooks/useAppVersion.js`, which measured the numbers: on a
 * backend that stays down, one mount makes 15 requests in 10 minutes and
 * settles at roughly 1.5 per minute. Anything on this schedule must also be
 * on the breadcrumb ignore list (`services/breadcrumbs.js`), or the retries
 * fill the 60-entry ring that a bug report attaches.
 */
export const FIRST_DELAY_MS = 1000;
export const MAX_DELAY_MS = 60000;

/** The wait after the one that just failed. */
export function nextDelay(delay) {
  return Math.min(delay * 2, MAX_DELAY_MS);
}
