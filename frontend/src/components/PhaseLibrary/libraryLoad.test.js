// @vitest-environment node
import { describe, it, expect, vi } from 'vitest';
import { createLibraryLoader, pageState } from './libraryLoad';
import { FIRST_DELAY_MS, MAX_DELAY_MS } from '../../services/retryBackoff';

/** The loader calls fetchIndex on a microtask, so a test that wants to hold a
 *  request open has to let that tick happen first. */
const tick = () => Promise.resolve().then(() => {}).then(() => {});

/** A loader whose clock the test owns; `fire()` runs the pending timer. */
function harness(fetchIndex) {
  const states = [], data = [], notes = [], waits = [];
  let pending = null;
  const loader = createLibraryLoader({
    fetchIndex,
    onState: (s) => states.push(s),
    onData: (d) => data.push(d),
    note: (type, msg) => notes.push(type + ': ' + msg),
    setTimer: (fn, ms) => { waits.push(ms); pending = fn; return waits.length; },
    clearTimer: () => { pending = null; },
  });
  return {
    loader, states, data, notes, waits,
    hasPending: () => pending !== null,
    fire: () => { const f = pending; pending = null; return f && f(); },
  };
}

describe('createLibraryLoader', () => {
  it('reports loading, then loaded, and hands the payload over once', async () => {
    const h = harness(async () => ({ phases: [{ key: 'Al' }] }));
    await h.loader.run();
    expect(h.states).toEqual(['loading', 'loaded']);
    expect(h.data).toEqual([{ phases: [{ key: 'Al' }] }]);
    expect(h.hasPending()).toBe(false);          // nothing to retry
  });

  it('a failed request is an error, never an empty library', async () => {
    const h = harness(async () => { throw new Error('Network Error'); });
    await h.loader.run();
    expect(h.states).toEqual(['loading', 'error']);
    expect(h.data).toEqual([]);                  // NOT an empty list
  });

  it('keeps asking on the shared backoff, doubling to the cap', async () => {
    const h = harness(async () => { throw new Error('down'); });
    await h.loader.run();
    for (let i = 0; i < 8; i += 1) await h.fire();
    expect(h.waits[0]).toBe(FIRST_DELAY_MS);
    expect(h.waits[1]).toBe(FIRST_DELAY_MS * 2);
    expect(h.waits[2]).toBe(FIRST_DELAY_MS * 4);
    expect(h.waits.at(-1)).toBe(MAX_DELAY_MS);
    expect(Math.max(...h.waits)).toBe(MAX_DELAY_MS);
  });

  it('notes the outage once and the recovery once, not every attempt', async () => {
    let up = false;
    const h = harness(async () => {
      if (!up) throw new Error('down');
      return { phases: [] };
    });
    await h.loader.run();
    for (let i = 0; i < 5; i += 1) await h.fire();
    expect(h.notes).toHaveLength(1);             // six failures, one breadcrumb
    expect(h.notes[0]).toContain('unreachable');
    up = true;
    await h.fire();
    expect(h.notes).toHaveLength(2);
    expect(h.notes[1]).toContain('reachable again');
  });

  it('a manual retry asks now and starts the waits over', async () => {
    const h = harness(async () => { throw new Error('down'); });
    await h.loader.run();
    await h.fire(); await h.fire();              // waits are at 4 s
    await h.loader.retry();
    expect(h.waits.at(-1)).toBe(FIRST_DELAY_MS);
  });

  it('a slow first reply never overwrites a newer one', async () => {
    const resolvers = [];
    const h = harness(() => new Promise((res) => resolvers.push(res)));
    const first = h.loader.run();
    await tick();
    const second = h.loader.run();
    await tick();
    resolvers[1]({ phases: [{ key: 'new' }] });  // the newer request answers first
    resolvers[0]({ phases: [{ key: 'old' }] });
    await Promise.all([first, second]);
    expect(h.data).toEqual([{ phases: [{ key: 'new' }] }]);
  });

  it('dispose stops the retrying and drops what is in flight', async () => {
    let resolve;
    const h = harness(() => new Promise((res) => { resolve = res; }));
    const p = h.loader.run();
    await tick();
    h.loader.dispose();
    resolve({ phases: [{ key: 'Al' }] });
    await p;
    expect(h.data).toEqual([]);
    expect(h.hasPending()).toBe(false);
  });

  it('dispose after a failure leaves no timer behind', async () => {
    const h = harness(async () => { throw new Error('down'); });
    await h.loader.run();
    expect(h.hasPending()).toBe(true);
    h.loader.dispose();
    expect(h.hasPending()).toBe(false);
  });
});

describe('pageState — the fourth one is the one that gets forgotten', () => {
  const S = (o) => pageState({ loadState: 'loaded', total: 36, shown: 36,
                               filtersActive: false, ...o });

  it('says loading before anything has come back', () => {
    expect(S({ loadState: 'idle' })).toBe('loading');
    expect(S({ loadState: 'loading' })).toBe('loading');
  });

  it('says error rather than empty when the request failed', () => {
    expect(S({ loadState: 'error', total: 0, shown: 0 })).toBe('error');
  });

  it('distinguishes "nothing in the library" from "nothing matches"', () => {
    expect(S({ total: 0, shown: 0 })).toBe('empty');
    expect(S({ shown: 0, filtersActive: true })).toBe('filtered-empty');
    // A library with phases and no filters cannot show nothing; if it does,
    // calling it "filtered to nothing" would send the user to reset filters
    // that are not set.
    expect(S({ shown: 0, filtersActive: false })).toBe('empty');
  });

  it('says ready when there is something to look at', () => {
    expect(S({})).toBe('ready');
    expect(S({ shown: 1 })).toBe('ready');
  });
});
