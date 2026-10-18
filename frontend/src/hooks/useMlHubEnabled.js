/**
 * Is the ML Hub switched on in the backend?
 *
 * The ML Hub is an internal, experimental page. The backend answers 404 on
 * every /api/ml route except /status unless it was started with
 * ORIENTA_EXPERIMENTAL_ML=1. /status always answers 200 (a 404 would show up
 * as a console error on every start) with {"enabled": true|false}, so
 * GET /api/ml/status is the one question: the tile and the sidebar entry may be
 * shown only on a 200 whose body says enabled === true.
 *
 * Three answers, not two:
 *   'enabled'   200 with enabled === true
 *   'disabled'  200 with anything else (enabled: false) -- final, nothing is
 *               asked again; a 404 (an older backend) is read the same way
 *   'unknown'   anything else (no response, 5xx, timeout) -- keep asking
 *
 * "Unknown" must not be read as "disabled": the first start after a runtime
 * update is exactly when the window is up and uvicorn is not, and a one-shot
 * fetch at mount that took that failure as the answer would hide the entry for good.
 * While unknown the entry stays hidden,
 * because showing a page that may not exist is the worse mistake.
 *
 * One request loop is shared by every caller (Dashboard, Sidebar, App): the
 * state lives at module level and the loop runs while somebody is subscribed
 * and the answer is not final yet. The schedule is services/retryBackoff.js.
 * /api/ml/status is on the breadcrumb ignore list via its '/status' suffix.
 */
import { useSyncExternalStore } from 'react';
import { mlApi } from '../services/api';
import { FIRST_DELAY_MS, nextDelay } from '../services/retryBackoff';

let state = 'unknown';
const listeners = new Set();
let timer = null;
let running = false;
let delay = FIRST_DELAY_MS;
let generation = 0; // bumped on stop, so a late reply of a stopped loop is dropped

function setState(next) {
  if (next === state) return;
  state = next;
  listeners.forEach((l) => l());
}

function attempt(gen) {
  mlApi.status()
    .then((res) => {
      if (gen !== generation) return;
      setState(res?.data?.enabled === true ? 'enabled' : 'disabled');
    })
    .catch((err) => {
      if (gen !== generation) return;
      if (err?.response?.status === 404) { setState('disabled'); return; }
      timer = setTimeout(() => attempt(gen), delay);
      delay = nextDelay(delay);
    });
}

function start() {
  if (state !== 'unknown' || running) return;
  running = true;
  delay = FIRST_DELAY_MS;
  attempt(generation);
}

function stop() {
  generation += 1;
  running = false;
  if (timer) clearTimeout(timer);
  timer = null;
}

function subscribe(listener) {
  listeners.add(listener);
  if (listeners.size === 1) start();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) stop();
  };
}

/** 'unknown' | 'enabled' | 'disabled' */
export function useMlHubState() {
  return useSyncExternalStore(subscribe, () => state, () => 'unknown');
}

/** True only when /api/ml/status answered 200 with enabled === true. */
export default function useMlHubEnabled() {
  return useMlHubState() === 'enabled';
}

/** Test seam: forget the answer and stop any loop. */
export function __resetMlHubEnabledForTests() {
  stop();
  state = 'unknown';
  delay = FIRST_DELAY_MS;
}
