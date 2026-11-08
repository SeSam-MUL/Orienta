// @vitest-environment jsdom
/**
 * "Is a file being loaded or switched right now?" is answered at the one place
 * every request passes: the axios interceptors. A store resync that runs while
 * a load is in flight reads half-changed backend state, so it asks this first.
 */
import { describe, it, expect, afterEach } from 'vitest';

import api, { isFileLoadInFlight, isSupersededLoad } from './api';

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

  it.each([
    '/api/ebsd/loaded-files/remove', '/api/ebsd/loaded-files/clear', '/api/ebsd/loaded-files/clear-all',
    '/api/ebsd/load-something-else', '/api/ebsd/switch-file/extra', '/api/h5/open-other',
  ])('does not count a POST to %s: it does not change which file is loaded', async (url) => {
    const g = gated(ok);
    api.defaults.adapter = g.adapter;
    const p = api.post(url, { path: 'x' });
    await Promise.resolve();
    expect(isFileLoadInFlight()).toBe(false);
    g.release();
    await p;
  });

  it('counts a POST whose URL carries a query string, and one with a full URL', async () => {
    for (const url of ['/api/ebsd/load?request_id=1', 'http://127.0.0.1:8000/api/ebsd/switch-file']) {
      const g = gated(ok);
      api.defaults.adapter = g.adapter;
      const p = api.post(url, { path: 'x' });
      await Promise.resolve();
      expect(isFileLoadInFlight()).toBe(true);
      g.release();
      await p;
      expect(isFileLoadInFlight()).toBe(false);
    }
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

describe('isSupersededLoad', () => {
  // A switch answer that arrives after a later load/switch was started is the older answer: the
  // later one will describe the file the backend ends on, so the older must not be written.
  it('is false for the newest request, true for one that a later load/switch has overtaken', async () => {
    const gA = gated(ok);
    const gB = gated(ok);
    api.defaults.adapter = (config) => (config.data && String(config.data).includes('"a"') ? gA.adapter(config) : gB.adapter(config));
    const pA = api.post('/api/ebsd/switch-file', { path: 'a' });
    await Promise.resolve();
    const pB = api.post('/api/ebsd/load', { path: 'b' });
    await Promise.resolve();
    gB.release();
    gA.release();
    const [rA, rB] = await Promise.all([pA, pB]);
    expect(await isSupersededLoad(rA)).toBe(true);
    expect(await isSupersededLoad(rB)).toBe(false);
  });

  it('is false for an answer that carries no load stamp (nothing to compare)', async () => {
    expect(await isSupersededLoad({ config: {} })).toBe(false);
    expect(await isSupersededLoad(undefined)).toBe(false);
  });

  it('does not count requests that do not change the loaded file', async () => {
    const g = gated(ok);
    api.defaults.adapter = (config) => (config.method === 'get' ? ok(config) : g.adapter(config));
    const pA = api.post('/api/ebsd/switch-file', { path: 'a' });
    await Promise.resolve();
    await api.get('/api/ebsd/info');
    g.release();
    expect(await isSupersededLoad(await pA)).toBe(false);
  });

  // A request that does not change which file is loaded is not a load: a switch answer that
  // arrives after the user removed or cleared list entries is still the answer to write.
  it.each([
    ['/api/ebsd/loaded-files/remove', { path: 'other' }],
    ['/api/ebsd/loaded-files/clear', undefined],
    ['/api/ebsd/loaded-files/clear-all', undefined],
  ])('a switch pending while %s is called is not superseded by it', async (url, body) => {
    const g = gated(ok);
    api.defaults.adapter = (config) => (String(config.url).includes('loaded-files') ? ok(config) : g.adapter(config));
    const pA = api.post('/api/ebsd/switch-file', { path: 'a' });
    await Promise.resolve();
    await api.post(url, body);
    g.release();
    expect(await isSupersededLoad(await pA)).toBe(false);
  });

  // The two switches, "a" first and "b" later. Each is answered by its own gate.
  function twoSwitches(thenB) {
    const gA = gated(ok);
    const gB = gated(thenB);
    api.defaults.adapter = (config) => (String(config.data).includes('"a"') ? gA.adapter(config) : gB.adapter(config));
    const pA = api.post('/api/ebsd/switch-file', { path: 'a' });
    const pB = api.post('/api/ebsd/switch-file', { path: 'b' });
    return { gA, gB, pA, pB: pB.catch((e) => e) };
  }

  it('a newer load that FAILS does not supersede the older one (the newer answered first)', async () => {
    const { gA, gB, pA, pB } = twoSwitches(fail);
    gB.release();
    await pB;
    gA.release();
    expect(await isSupersededLoad(await pA)).toBe(false);
  });

  it('a newer load that FAILS does not supersede the older one (the older answered first)', async () => {
    const { gA, gB, pA, pB } = twoSwitches(fail);
    gA.release();
    const verdict = isSupersededLoad(await pA);        // waits for the newer one to settle
    gB.release();
    await pB;
    expect(await verdict).toBe(false);
  });

  it('a newer load that succeeds supersedes the older one (the newer answered first)', async () => {
    const { gA, gB, pA, pB } = twoSwitches(ok);
    gB.release();
    await pB;
    gA.release();
    expect(await isSupersededLoad(await pA)).toBe(true);
  });

  it('a newer load that succeeds supersedes the older one (the older answered first: the verdict waits)', async () => {
    const { gA, gB, pA, pB } = twoSwitches(ok);
    gA.release();
    const verdict = isSupersededLoad(await pA);
    let settled = false;
    verdict.then(() => { settled = true; });
    await new Promise((r) => setTimeout(r, 10));
    expect(settled).toBe(false);                       // the newer one is still pending: not decided yet
    gB.release();
    await pB;
    expect(await verdict).toBe(true);
  });

  it('the newest of three is the only one that is not superseded', async () => {
    const g = [gated(ok), gated(ok), gated(ok)];
    api.defaults.adapter = (config) => g[JSON.parse(config.data).i].adapter(config);
    const ps = [0, 1, 2].map((i) => api.post('/api/ebsd/switch-file', { path: 'x', i }));
    g[2].release(); g[0].release(); g[1].release();
    const rs = await Promise.all(ps);
    expect(await Promise.all(rs.map(isSupersededLoad))).toEqual([true, true, false]);
  });
});
