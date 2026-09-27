/**
 * Fetching the phase list, with the one rule that was missing: a failed
 * request is not an answer.
 *
 * The old code lived inline in IndexingPage and ended on
 *
 *     } catch { setDiscoveredFiles([]); setDiscoveredGroups([]); }
 *
 * so a backend that had not started yet — the first start after an update,
 * the case Sebastian reported — produced exactly the state "the library is
 * empty", and the dropdown then said **"No phases found"**. That is a claim
 * about the library, made at the one moment the program had failed to ask
 * it. `useCollectionStore.load()` has always done the opposite, with the
 * reason in its own comment: "a failed refresh must not empty the pickers
 * and thereby widen every run to the whole library."
 *
 * It lives in its own file so the behaviour can be tested. IndexingPage is
 * ~4000 lines and no test mounts it; anything left inline there is, in
 * practice, untested.
 *
 * `schedule`/`cancel` are injected so tests drive the clock without fake
 * timers leaking into an async fetch.
 */
import { FIRST_DELAY_MS, nextDelay } from '../../services/retryBackoff';

/**
 * @param {object} o
 * @param {(method: any) => Promise<{files: any[], groups: any[]}>} o.discover
 * @param {(files: any[], groups: any[]) => void} o.onLoaded  called ONLY on success
 * @param {(state: 'loading'|'loaded'|'error') => void} o.onState
 */
export function createPhaseDiscovery({
  discover,
  onLoaded,
  onState,
  schedule = setTimeout,
  cancel = clearTimeout,
}) {
  let seq = 0;
  let timer = null;
  let delay = FIRST_DELAY_MS;

  function clearPending() {
    if (timer !== null) { cancel(timer); timer = null; }
  }

  async function run(method) {
    // A newer call wins. Switching method twice quickly must not let the
    // first reply land on top of the second's list.
    const mine = ++seq;
    clearPending();
    onState('loading');
    try {
      const { files = [], groups = [] } = (await discover(method)) || {};
      if (mine !== seq) return;
      onLoaded(files, groups);
      onState('loaded');
      delay = FIRST_DELAY_MS;
    } catch {
      if (mine !== seq) return;
      // Deliberately no onLoaded: whatever is on screen stays.
      onState('error');
      const wait = delay;
      delay = nextDelay(wait);
      timer = schedule(() => { timer = null; run(method); }, wait);
    }
  }

  /** "Try again now": back to the short delay, ask immediately. */
  function retry(method) {
    delay = FIRST_DELAY_MS;
    return run(method);
  }

  /** Stop asking — the page is gone. */
  function dispose() {
    seq += 1;
    clearPending();
  }

  return { run, retry, dispose };
}

/**
 * May a run start, as far as the phase list is concerned?
 *
 * Only once the list has actually arrived. An unloaded list is not "no
 * restrictions" — `SinglePixelPhaseTestDialog`'s own comment records what
 * that cost there: with a collection active, "still loading" and "failed to
 * load" both read as "nothing selected", the button offered to auto-run, and
 * a click ran the entire library.
 */
export function phaseListBlocksRun(discoverState) {
  return discoverState !== 'loaded';
}
