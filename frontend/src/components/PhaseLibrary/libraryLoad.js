/**
 * Loading the phase library, with the four states the page must be able to
 * show (spec §2.8).
 *
 * WHY THIS IS NOT `useEffect(() => { fetch().then(setPhases) }, [])`:
 *
 * A failed request is not an answer. The two places in this app that treated
 * it as one both broke at the same moment -- the first start after a runtime
 * update, when the window is up and uvicorn is not -- and neither recovered:
 * Settings → About said "unknown (no git information found)" while the
 * endpoint was answering, and the Indexing phase list said "No phases found"
 * for a library of 36 phases. The second is the worse shape, and it is this
 * page's shape: an empty library and a library we could not ask for look
 * identical on screen, and the empty one invites the user to go and rebuild
 * something that is already there.
 *
 * So: four states, and the page says which one it is in.
 *
 *   idle     nothing asked for yet
 *   loading  asked, waiting
 *   loaded   the backend answered; `phases` is what it said, empty or not
 *   error    the request failed; we are still asking, on the shared backoff
 *
 * The retry schedule is `services/retryBackoff.js` -- the same one, because
 * a second schedule would be a second thing to measure. Its docstring carries
 * an obligation: anything on it must also be on the breadcrumb ignore list,
 * or the retries fill the 60-entry ring a bug report attaches. The path is on
 * that list, and this module instead notes the FIRST failure and the recovery
 * by hand, so an outage costs two breadcrumbs rather than fifteen. A user who
 * reports "my phases are gone" needs that in the trail; they do not need it
 * ninety times.
 */
import { FIRST_DELAY_MS, nextDelay } from '../../services/retryBackoff';

/**
 * @param {object} o
 * @param {() => Promise<object>} o.fetchIndex  resolves to the index payload
 * @param {(state: 'loading'|'loaded'|'error') => void} o.onState
 * @param {(payload: object) => void} o.onData  called only on success
 * @param {(type: string, message: string) => void} [o.note]  breadcrumb sink
 * @param {typeof setTimeout} [o.setTimer]   injected so tests own the clock
 * @param {typeof clearTimeout} [o.clearTimer]
 */
export function createLibraryLoader({
  fetchIndex, onState, onData, note = () => {},
  setTimer = setTimeout, clearTimer = clearTimeout,
}) {
  let seq = 0;
  let timer = null;
  let delay = FIRST_DELAY_MS;
  // Only the first failure of a run of them is worth a breadcrumb, and only
  // then is the recovery worth one.
  let failing = false;

  function cancelPending() {
    if (timer !== null) { clearTimer(timer); timer = null; }
  }

  function run() {
    cancelPending();
    // Bumping the sequence discards any reply still in flight. Without it a
    // slow first request can land after a fast second one and overwrite the
    // newer library with the older.
    const mine = ++seq;
    onState('loading');
    return Promise.resolve()
      .then(fetchIndex)
      .then((payload) => {
        if (mine !== seq) return;
        if (failing) { note('info', 'Phase library reachable again'); failing = false; }
        delay = FIRST_DELAY_MS;
        onData(payload);
        onState('loaded');
      })
      .catch((err) => {
        if (mine !== seq) return;
        if (!failing) {
          failing = true;
          note('http-error', 'Phase library index unreachable: '
            + String(err && err.message ? err.message : err).slice(0, 200));
        }
        onState('error');
        timer = setTimer(run, delay);
        delay = nextDelay(delay);
      });
  }

  /** The user pressed "try again": ask now, and start the waits over. */
  function retry() {
    delay = FIRST_DELAY_MS;
    return run();
  }

  function dispose() {
    seq += 1;          // in-flight replies are now stale and will be dropped
    cancelPending();
  }

  return { run, retry, dispose };
}

/**
 * Which of the four states the page is in, given what it knows.
 *
 * Kept out of the component because the fourth one is the one that gets
 * forgotten: "loaded, and the filters match nothing". Fassung 1 of the spec
 * had three states and that case rendered as a grey rectangle -- with every
 * facet count at 0, so the one way out was itself invisible.
 *
 * @returns {'loading'|'error'|'empty'|'filtered-empty'|'ready'}
 */
export function pageState({ loadState, total, shown, filtersActive }) {
  if (loadState === 'idle' || loadState === 'loading') return 'loading';
  if (loadState === 'error') return 'error';
  if (!total) return 'empty';
  if (!shown) return filtersActive ? 'filtered-empty' : 'empty';
  return 'ready';
}
