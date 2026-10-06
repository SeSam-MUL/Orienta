// @vitest-environment jsdom
/**
 * "Is a file being loaded or switched right now?" is answered at the one place
 * every request passes: the axios interceptors. A store resync that runs while
 * a load is in flight reads half-changed backend state, so it asks this first.
 */
import { describe, it, expect, afterEach } from 'vitest';

import api, { isFileLoadInFlight } from './api';

const original = api.defaults.adapter;
afterEach(() => { api.defaults.adapter = original; });

const ok = (config) => Promise.resolve({ status: 200, data: {}, statusText: 'OK', headers: {}, config });
const fail = (config) => Promise.reject(Object.assign(new Error('Network Error'), { isAxiosError: true, config }));

function gated(then) {
  let release;
  const gate = new Promise((r) => { release = r; });
  return { adapter: (config) => gate.then(() => then(config)), release };
}

describe('isFileLoadInFlight', () => {
  it('is false at rest', () => {
    expect(isFileLoadInFlight()).toBe(false);
  });

  it.each(['/api/ebsd/load', '/api/ebsd/switch-file', '/api/h5/open'])(
    'is true while a POST to %s is pending and false after it succeeds', async (url) => {
      const g = gated(ok);
      api.defaults.adapter = g.adapter;
      const p = api.post(url, { path: 'x' });
      await Promise.resolve();
      expect(isFileLoadInFlight()).toBe(true);
      g.release();
      await p;
      expect(isFileLoadInFlight()).toBe(false);
    });

  it('is false again after the load fails', async () => {
    const g = gated(fail);
    api.defaults.adapter = g.adapter;
    const p = api.post('/api/ebsd/load', { path: 'x' }).catch(() => {});
    await Promise.resolve();
    expect(isFileLoadInFlight()).toBe(true);
    g.release();
    await p;
    expect(isFileLoadInFlight()).toBe(false);
  });

  it('ignores other requests, including GETs of the same endpoint family', async () => {
    const g = gated(ok);
    api.defaults.adapter = g.adapter;
    const p = api.get('/api/ebsd/info');
    await Promise.resolve();
    expect(isFileLoadInFlight()).toBe(false);
    g.release();
    await p;
  });
});
