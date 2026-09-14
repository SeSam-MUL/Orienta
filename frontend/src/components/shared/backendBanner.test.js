import { describe, it, expect } from 'vitest';
import { backendBanner, STARTUP_GRACE_SEC, readEverConnected, rememberEverConnected } from './backendBanner';

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

  it('the grace period matches the launchers', () => {
    // start_app.py BACKEND_START_TIMEOUT_S and electron/main.js retries*0.5 s
    expect(STARTUP_GRACE_SEC).toBe(180);
  });
});
