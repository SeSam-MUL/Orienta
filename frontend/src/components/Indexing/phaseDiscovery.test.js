// @vitest-environment node
import { describe, it, expect, vi } from 'vitest';
import { createPhaseDiscovery, phaseListBlocksRun } from './phaseDiscovery';
import { FIRST_DELAY_MS, MAX_DELAY_MS } from '../../services/retryBackoff';

/** A clock we hold: records every delay, fires on demand. */
function fakeClock() {
  // Keyed by handle, and `cancel` HONOURS its argument -- an earlier version
  // did `pending.length = 0` and so could not tell "cancelled the right
  // timer" from "cancelled everything", which is exactly the property the
  // stale-chain test below needs.
  const pending = new Map();
  let next = 1;
  return {
    delays: [],
    schedule(fn, ms) { this.delays.push(ms); const h = next++; pending.set(h, fn); return h; },
    cancel(h) { pending.delete(h); },
    async fire() {
      const [h, f] = pending.entries().next().value || [];
      if (f) { pending.delete(h); f(); await Promise.resolve(); await Promise.resolve(); }
    },
    get waiting() { return pending.size; },
  };
}

function harness(discover) {
  const clock = fakeClock();
  const loaded = [];
  const states = [];
  const d = createPhaseDiscovery({
    discover,
    onLoaded: (files, groups) => loaded.push({ files, groups }),
    onState: (s) => states.push(s),
    schedule: (fn, ms) => clock.schedule(fn, ms),
    cancel: (t) => clock.cancel(t),
  });
  return { d, clock, loaded, states };
}

describe('phase discovery', () => {
  // THE defect. Before this, the catch ran `setDiscoveredFiles([])` and the
  // dropdown announced "No phases found" — a claim about the library, made
  // when the library had not been reached.
  it('a failed request never empties the list', async () => {
    const { d, loaded, states } = harness(() => Promise.reject(new Error('ECONNREFUSED')));
    await d.run('hough');
    expect(loaded).toEqual([]);          // nothing was written over the list
    expect(states).toEqual(['loading', 'error']);
  });

  it('retries, and the list fills once the backend is up', async () => {
    const files = [{ path: 'D:/lib/Al.cif', filename: 'Al.cif' }];
    const discover = vi.fn()
      .mockRejectedValueOnce(new Error('backend not up yet'))
      .mockResolvedValueOnce({ files, groups: ['Al'] });
    const { d, clock, loaded, states } = harness(discover);

    await d.run('hough');
    expect(states).toEqual(['loading', 'error']);
    expect(clock.waiting).toBe(1);

    await clock.fire();
    expect(discover).toHaveBeenCalledTimes(2);
    expect(loaded).toEqual([{ files, groups: ['Al'] }]);
    // Never a state in between that would let the UI say "none found":
    // 'loaded' appears exactly once, at the end, and only after real files.
    expect(states).toEqual(['loading', 'error', 'loading', 'loaded']);
  });

  // The other half of the contract. If the backend really answers with an
  // empty library, "No phases found" is true and must still be said.
  it('an empty answer IS an answer', async () => {
    const { d, loaded, states } = harness(() => Promise.resolve({ files: [], groups: [] }));
    await d.run('hough');
    expect(loaded).toEqual([{ files: [], groups: [] }]);
    expect(states).toEqual(['loading', 'loaded']);
  });

  it('doubles the wait, and stops doubling at the cap', async () => {
    const { d, clock } = harness(() => Promise.reject(new Error('down')));
    await d.run('hough');
    for (let i = 0; i < 8; i += 1) await clock.fire();
    expect(clock.delays.slice(0, 4)).toEqual([1000, 2000, 4000, 8000]);
    expect(clock.delays[0]).toBe(FIRST_DELAY_MS);
    expect(Math.max(...clock.delays)).toBeLessThanOrEqual(MAX_DELAY_MS);
    expect(clock.delays[clock.delays.length - 1]).toBe(MAX_DELAY_MS);
  });

  it('"try again now" goes back to the short wait', async () => {
    const { d, clock } = harness(() => Promise.reject(new Error('down')));
    await d.run('hough');
    await clock.fire();
    await clock.fire();
    expect(clock.delays[clock.delays.length - 1]).toBe(4000);
    await d.retry('hough');
    expect(clock.delays[clock.delays.length - 1]).toBe(FIRST_DELAY_MS);
  });

  it('stops asking once disposed', async () => {
    const discover = vi.fn(() => Promise.reject(new Error('down')));
    const { d, clock } = harness(discover);
    await d.run('hough');
    expect(clock.waiting).toBe(1);
    d.dispose();
    expect(clock.waiting).toBe(0);
    await clock.fire();
    expect(discover).toHaveBeenCalledTimes(1);
  });

  // Switching method twice quickly: the first reply must not land on the
  // second's list. Seen in the wild as the Hough CIF library appearing under a
  // Spherical run -- the comment at the discover call site says the same thing.
  it('a slow reply from an older call is discarded', async () => {
    let releaseFirst;
    const discover = vi.fn()
      .mockImplementationOnce(() => new Promise((res) => { releaseFirst = res; }))
      .mockResolvedValueOnce({ files: [{ path: 'new.sht' }], groups: [] });
    const { d, loaded } = harness(discover);

    d.run('hough');                       // deliberately not awaited
    await d.run('spherical');
    expect(loaded).toEqual([{ files: [{ path: 'new.sht' }], groups: [] }]);

    releaseFirst({ files: [{ path: 'stale.cif' }], groups: [] });
    await Promise.resolve(); await Promise.resolve();
    expect(loaded).toHaveLength(1);       // still only the newer one
  });

  // `run()` clears any pending retry before starting. Without that, a manual
  // retry (or a method switch) while a retry is already booked leaves TWO
  // chains ticking, each doubling on its own. A review found this property
  // untested -- the backoff test only ever looked at the last delay.
  it('never leaves two retry chains ticking', async () => {
    const { d, clock } = harness(() => Promise.reject(new Error('down')));
    await d.run('hough');
    expect(clock.waiting).toBe(1);
    await d.retry('hough');
    expect(clock.waiting).toBe(1);        // the old one was cancelled, not added to
    await d.run('spherical');
    expect(clock.waiting).toBe(1);
  });

  // A recovery must reset the wait. Otherwise the next failure resumes at a
  // minute, however long ago the trouble was.
  it('a success puts the wait back to the start', async () => {
    let fail = true;
    const { d, clock } = harness(() => (fail
      ? Promise.reject(new Error('down'))
      : Promise.resolve({ files: [], groups: [] })));
    await d.run('hough');
    await clock.fire();                   // 1 s
    await clock.fire();                   // 2 s
    expect(clock.delays).toEqual([1000, 2000, 4000]);
    fail = false;
    await d.run('hough');                 // succeeds
    fail = true;
    await d.run('hough');                 // fails again
    expect(clock.delays[clock.delays.length - 1]).toBe(1000);
  });

  it('survives a reply with no body at all', async () => {
    const { d, loaded, states } = harness(() => Promise.resolve(undefined));
    await d.run('hough');
    expect(loaded).toEqual([{ files: [], groups: [] }]);
    expect(states).toEqual(['loading', 'loaded']);
  });
});

describe('the run waits for the phase list', () => {
  it('only a list that actually arrived releases Start', () => {
    expect(phaseListBlocksRun('loaded')).toBe(false);
    expect(phaseListBlocksRun('loading')).toBe(true);
    expect(phaseListBlocksRun('error')).toBe(true);
    // "Never asked" is the state that used to read as "nothing to exclude".
    expect(phaseListBlocksRun('idle')).toBe(true);
  });
});
