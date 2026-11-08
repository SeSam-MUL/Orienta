import { useEffect, useState } from 'react';
import { edsApi } from '../../../services/api';

/** How often the state is asked while the fit runs. */
export const FIT_POLL_MS = 500;
/** How long to wait before asking again after a request that got no answer. */
export const FIT_RETRY_MS = 1000;

const STATES = new Set(['none', 'running', 'ready', 'failed']);

/** "Not known yet": no answer for this file has arrived. Callers treat it as not ready. */
const UNKNOWN = Object.freeze({
  state: null, progress: 0, stage: null, fit: null, error: null, unreachable: false,
  runner: null, runnerReason: null,
});

function normalise(data) {
  const p = Number(data?.progress);
  return {
    state: STATES.has(data?.state) ? data.state : 'none',
    progress: Number.isFinite(p) ? p : 0,
    stage: data?.stage ?? null,
    fit: data?.fit ?? null,
    error: data?.error ?? null,
    unreachable: false,
    // Where the backend runs the fit: "child" (its own process),
    // "in_process", or "in_process_fallback" (the child could not be
    // started; the backend reads the cube itself and answers slowly).
    runner: data?.runner ?? null,
    runnerReason: data?.runner_reason ?? null,
  };
}

/**
 * Did the backend ANSWER with this error? A 4xx is an answer: the status
 * route refuses with 400 "No file open" when there is nothing to evaluate,
 * and asking again would not change that. No response at all (a timeout, a
 * dropped connection), a 5xx, a 408 or a 429 is not an answer about the fit.
 */
function isDefinitiveRefusal(err) {
  const status = Number(err?.response?.status);
  return status >= 400 && status < 500 && status !== 408 && status !== 429;
}

/**
 * The spectrum fit of the open file, as `GET /api/eds/spectrum-fit/status`
 * reports it: `{state, progress, stage, fit, error, unreachable, runner,
 * runnerReason}`.
 *
 * `state` is `"none"` for a file without a spectrum cube (window numbers),
 * else `"running"` / `"ready"` / `"failed"`, and `null` until the first answer
 * for THIS file has arrived -- a new file never inherits the old file's
 * "ready". While the fit runs the state is asked every {@link FIT_POLL_MS};
 * the poll stops on any other answer, so a page that stays mounted (every page
 * in this app does) is not left asking twice a second for good.
 *
 * `active: false` (a hidden page, a closed dialog) asks nothing and polls
 * nothing; turning active asks once more, which is also how a file reloaded
 * elsewhere -- a failed fit is retried on reload -- reaches this page.
 *
 * A request that gets NO answer is never read as one. Turning it into
 * "none" would mount the EDS tabs on a file whose fit may be running: the
 * element list and the default layers load empty on the 503s and nothing
 * recovers -- the very failure the gate exists to prevent. So the last known
 * state stays (null before the first answer), `unreachable` turns true for
 * the page to say so, and the request is repeated every {@link FIT_RETRY_MS}
 * until the backend answers. The status call has a short timeout of its own
 * (`edsApi.spectrumFitStatus`), so "no answer" is known within seconds rather
 * than after the client's five-minute default. A 4xx IS an answer (no file
 * open, nothing to wait for) and reads as `"none"`.
 *
 * `recheck` (any value; default 0): a change forgets what is known for this
 * file and asks again. For the page that LOADS the file: loading the same
 * path again can start a new fit (the file changed on disk, its result was
 * dropped for newer files, a failed fit is retried), and the path alone does
 * not change.
 */
export function useSpectrumFitStatus(filePath, { active = true, recheck = 0 } = {}) {
  const key = filePath || null;
  const [entry, setEntry] = useState({ key: null, recheck: 0, status: UNKNOWN });

  useEffect(() => {
    if (!key || !active) return undefined;
    let cancelled = false;
    let timer = null;
    const tick = async () => {
      timer = null;
      let next = null;
      try {
        const res = await edsApi.spectrumFitStatus();
        next = normalise(res?.data);
      } catch (err) {
        if (isDefinitiveRefusal(err)) next = { ...UNKNOWN, state: 'none' };
      }
      if (cancelled) return;
      if (next === null) {
        // No answer: keep what is known, say the backend is not answering,
        // and ask again.
        setEntry((prev) => ({
          key,
          recheck,
          status: {
            ...(prev.key === key && prev.recheck === recheck ? prev.status : UNKNOWN),
            unreachable: true,
          },
        }));
        timer = setTimeout(tick, FIT_RETRY_MS);
        return;
      }
      setEntry({ key, recheck, status: next });
      if (next.state === 'running') timer = setTimeout(tick, FIT_POLL_MS);
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [key, active, recheck]);

  return key && entry.key === key && entry.recheck === recheck ? entry.status : UNKNOWN;
}

/**
 * How many times `state` (a {@link useSpectrumFitStatus} state) has turned
 * `"ready"` since mount -- from the first answer, from `"running"`, or from
 * `"failed"` after a retry. A widget that fetched EDS data once (on file
 * load, on dialog open) puts this in its fetch's dependencies, so the fetch
 * runs again when the numbers exist: a 503 during the fit used to stay the
 * widget's answer for good. Mounting on a fit
 * that is already ready counts nothing: the widget's first fetch had it.
 */
export function useFitReadyCount(state) {
  // "Storing information from previous renders" (react.dev): the state seen
  // last, updated during render, so the new count is there in this render.
  const [seen, setSeen] = useState({ state, count: 0 });
  if (seen.state !== state) {
    const count = seen.count + (state === 'ready' ? 1 : 0);
    setSeen({ state, count });
    return count;
  }
  return seen.count;
}

export default useSpectrumFitStatus;
