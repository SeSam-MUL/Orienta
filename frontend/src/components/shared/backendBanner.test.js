import { describe, it, expect } from 'vitest';
import { backendBanner, LOST_AFTER_FAILING_SEC, STARTUP_GRACE_SEC, readEverConnected, rememberEverConnected, monoNow } from './backendBanner';

describe('everConnected memory across a reload', () => {
  const fakeStorage = () => {
    const m = new Map();
    return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v) };
  };

  it('a reloaded page that had seen the backend reports "lost", not "starting"', () => {
    const s = fakeStorage();
    expect(readEverConnected(s)).toBe(false);
    rememberEverConnected(s);
    expect(readEverConnected(s)).toBe(true);
    expect(backendBanner({ status: 'disconnected', everConnected: readEverConnected(s), sinceLoadSec: 1 }).kind).toBe('lost');
  });

  it('missing or throwing storage means "never connected"', () => {
    expect(readEverConnected(null)).toBe(false);
    const broken = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
    expect(readEverConnected(broken)).toBe(false);
    expect(() => rememberEverConnected(broken)).not.toThrow();
  });
});

describe('backendBanner', () => {
  it('shows nothing while checking or connected', () => {
    expect(backendBanner({ status: 'checking', everConnected: false, sinceLoadSec: 5 }).kind).toBe('none');
    expect(backendBanner({ status: 'connected', everConnected: true, sinceLoadSec: 500 }).kind).toBe('none');
  });

  it('a cold start is "starting", not an error, for the whole grace period', () => {
    // Measured: 33-36 s after a fresh installation. The old red banner with
    // "start the server with python -m uvicorn" appeared here.
    for (const s of [0, 1, 33, 36, 120, STARTUP_GRACE_SEC - 1]) {
      const b = backendBanner({ status: 'disconnected', everConnected: false, sinceLoadSec: s });
      expect(b.kind).toBe('starting');
      expect(b.elapsed).toBe(s);
    }
  });

  it('after the grace period a backend that never answered is "down" with instructions', () => {
    expect(backendBanner({ status: 'disconnected', everConnected: false, sinceLoadSec: STARTUP_GRACE_SEC }).kind).toBe('down');
    expect(backendBanner({ status: 'disconnected', everConnected: false, sinceLoadSec: 9999 }).kind).toBe('down');
  });

  it('a backend that was there and went away is "lost", regardless of the clock', () => {
    expect(backendBanner({ status: 'disconnected', everConnected: true, sinceLoadSec: 3 }).kind).toBe('lost');
    expect(backendBanner({ status: 'disconnected', everConnected: true, sinceLoadSec: 9999 }).kind).toBe('lost');
  });

  it('one missed health reply is a hiccup, not a loss', () => {
    // 0.4.5 laptop test: a red "connection lost" during normal use. The
    // mechanism is a stall longer than the 4 s request timeout while the
    // process is alive; what triggered it there is not established.
    expect(backendBanner({
      status: 'disconnected', everConnected: true, sinceLoadSec: 300,
      sinceFirstFailSec: 0,
    }).kind).toBe('none');
    expect(backendBanner({
      status: 'disconnected', everConnected: true, sinceLoadSec: 300,
      sinceFirstFailSec: 9.9,
    }).kind).toBe('none');
  });

  it('failing past the threshold is "lost"', () => {
    expect(backendBanner({
      status: 'disconnected', everConnected: true, sinceLoadSec: 300,
      sinceFirstFailSec: LOST_AFTER_FAILING_SEC,  // pins < against <=
    }).kind).toBe('lost');
    expect(backendBanner({
      status: 'disconnected', everConnected: true, sinceLoadSec: 300,
      sinceFirstFailSec: 60,
    }).kind).toBe('lost');
  });

  it('without a tracked failure time the old, immediate rule stands', () => {
    // A reloaded tab knows it saw the backend (sessionStorage) but has not
    // polled yet; a dead backend must still say "lost".
    expect(backendBanner({
      status: 'disconnected', everConnected: true, sinceLoadSec: 3,
    }).kind).toBe('lost');
  });

  it('the threshold is a wall clock, not a failure count', () => {
    // services/api.js STALE_THRESHOLD_MS = 10000 (session 2026-05-27).
    expect(LOST_AFTER_FAILING_SEC).toBe(10);
  });

  it('the clock does not move when the system clock does', () => {
    // A backwards jump (NTP) on Date.now() would make the elapsed time
    // negative and suppress the banner indefinitely.
    const a = monoNow();
    const b = monoNow();
    expect(typeof a).toBe('number');
    expect(b).toBeGreaterThanOrEqual(a);
  });

  /**
   * The rule alone cannot show whether it ever fires: that depends on WHEN
   * App.jsx asks. This replays its clock -- 30 s apart while connected, 2 s
   * apart once failing, each failure costing the 4 s request timeout
   * (App.jsx, services/api.js) -- and reports every banner the user would see.
   *
   * Written after a review found the first version of this fix measured from
   * the last successful reply. The first failure is then always >= 30 s old,
   * so the suppression could never fire and the constant was dead code.
   */
  function replayPoll({ deadFrom, deadUntil = Infinity, horizon = 300 }) {
    const TIMEOUT = 4, CONNECTED_GAP = 30, FAILING_GAP = 2;
    const alive = (t) => t < deadFrom || t >= deadUntil;
    const seen = [];
    let t = 0, firstFailAt = 0;
    while (t <= horizon) {
      if (alive(t)) {
        firstFailAt = 0;
        t += CONNECTED_GAP;
      } else {
        const observedAt = t + TIMEOUT;
        if (!firstFailAt) firstFailAt = observedAt;
        seen.push({
          at: observedAt,
          kind: backendBanner({
            status: 'disconnected', everConnected: true, sinceLoadSec: 300,
            sinceFirstFailSec: observedAt - firstFailAt,
          }).kind,
        });
        t = observedAt + FAILING_GAP;
      }
    }
    return seen;
  }

  it('a 7 s stall never turns the banner red', () => {
    const seen = replayPoll({ deadFrom: 30, deadUntil: 37 });
    expect(seen.length).toBeGreaterThan(0);          // it DID miss replies
    expect(seen.map(s => s.kind)).not.toContain('lost');
  });

  it('a backend that really died is still reported', () => {
    const seen = replayPoll({ deadFrom: 30 });
    const lost = seen.find(s => s.kind === 'lost');
    expect(lost).toBeDefined();
    // Death at 30 s, first evidence at 34 s, red at 46 s: the 30 s poll gap
    // is unchanged, the threshold adds the rest.
    expect(lost.at).toBeLessThanOrEqual(30 + 4 + LOST_AFTER_FAILING_SEC + 2);
  });

  it('the grace period matches the launchers', () => {
    // start_app.py BACKEND_START_TIMEOUT_S and electron/main.js retries*0.5 s
    expect(STARTUP_GRACE_SEC).toBe(180);
  });
});
