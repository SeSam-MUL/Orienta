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
 * @returns {{ kind: 'none'|'starting'|'down'|'lost', elapsed?: number }}
 */
export const STARTUP_GRACE_SEC = 180; // matches start_app.py / electron main.js

export function backendBanner({ status, everConnected, sinceLoadSec }) {
  if (status !== 'disconnected') return { kind: 'none' };
  if (everConnected) return { kind: 'lost' };
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
