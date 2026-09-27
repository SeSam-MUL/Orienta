/**
 * What the top banner should say while the backend is not answering.
 *
 * The old banner had one message for every case: "Backend not connected —
 * start the server with python -m uvicorn …". Shown in red during a normal
 * cold start (measured 33-36 s after installation, longer on a slow disk),
 * it read as "the backend is not loading at all" to a user who had just
 * started the app — the launcher WAS starting it. Three cases instead:
 *
 *   starting — never connected yet and still inside the start-up grace
 *              period: reassure, count the seconds, no instructions;
 *   down     — never connected and the grace period is over: something is
 *              wrong, show how to start the server by hand;
 *   lost     — was connected and is not any more: the backend died or was
 *              restarted; the poll keeps trying every 2 s.
 *
 * Pure function so the wording rules are testable without rendering App.
 *
 * @param {object} o
 * @param {'checking'|'connected'|'disconnected'} o.status
 * @param {boolean} o.everConnected  has this page ever seen the backend?
 * @param {number} o.sinceLoadSec    seconds since the page loaded
 * @param {number} [o.sinceFirstFailSec] seconds the health poll has been
 *   failing without interruption. Undefined means "not tracked" and keeps the
 *   old, immediate rule.
 * @returns {{ kind: 'none'|'starting'|'down'|'lost', elapsed?: number }}
 */
export const STARTUP_GRACE_SEC = 180; // matches start_app.py / electron main.js

/**
 * How long the health poll must keep failing before "lost" is shown.
 *
 * A single failed request is not evidence that the backend is gone: it times
 * out after 4 s (services/api.js) and the event loop can be GIL-blocked for
 * longer than that while the process is perfectly alive. Reported from the
 * 0.4.5 laptop test, where a red "connection lost" appeared during normal use.
 *
 * Measured from the FIRST FAILURE, not from the last successful reply. While
 * connected the poll only runs every 30 s (App.jsx), so "no reply for 10 s" is
 * never true at the moment we find out — we were not asking. The first failure
 * is the first instant we have any evidence at all, and from there App.jsx
 * retries every 2 s, so the window covers about five attempts.
 *
 * That is the same reasoning as the loader's STALE_THRESHOLD_MS (services/api.js,
 * session 2026-05-27) — a wall clock over a period we are actually watching —
 * and deliberately NOT a count of consecutive failures.
 *
 * Cost, stated plainly: a backend that dies right after a successful poll is
 * still reported up to ~30 s later (unchanged — that is the poll gap), and now
 * ~10 s after that. What it buys is that a 4 s stall no longer turns the banner
 * red.
 */
export const LOST_AFTER_FAILING_SEC = 10;

export function backendBanner({ status, everConnected, sinceLoadSec, sinceFirstFailSec }) {
  if (status !== 'disconnected') return { kind: 'none' };
  if (everConnected) {
    // Failing for less than the threshold: a hiccup, not a loss.
    if (typeof sinceFirstFailSec === 'number' && sinceFirstFailSec < LOST_AFTER_FAILING_SEC) {
      return { kind: 'none' };
    }
    return { kind: 'lost' };
  }
  const elapsed = Math.max(0, Math.floor(sinceLoadSec || 0));
  if (elapsed < STARTUP_GRACE_SEC) return { kind: 'starting', elapsed };
  return { kind: 'down', elapsed };
}

/**
 * "This tab has seen the backend" survives a reload in sessionStorage, so a
 * reloaded page (F5, Vite hot reload, Electron crash recovery) reports a
 * dead backend as "lost", not as a fresh "starting … 0 s" count. Session
 * scoped on purpose: a new window is a new start. Storage can be missing or
 * throw (privacy mode, thumbnail capture) — then it simply forgets.
 */
const EVER_CONNECTED_KEY = 'orienta.backendEverConnected';

export function readEverConnected(storage = typeof sessionStorage !== 'undefined' ? sessionStorage : null) {
  try {
    return storage?.getItem(EVER_CONNECTED_KEY) === '1';
  } catch {
    return false;
  }
}

export function rememberEverConnected(storage = typeof sessionStorage !== 'undefined' ? sessionStorage : null) {
  try {
    storage?.setItem(EVER_CONNECTED_KEY, '1');
  } catch {
    /* forgetting is the fallback */
  }
}

/**
 * Monotonic clock for the failure window.
 *
 * `performance.now()` does not move when the system clock is corrected (NTP,
 * a manual change), so a backwards jump cannot make the elapsed time negative
 * and suppress the banner forever. Falls back to Date.now() where it is absent.
 */
export const monoNow = () =>
  (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
