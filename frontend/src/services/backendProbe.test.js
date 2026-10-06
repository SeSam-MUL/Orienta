// @vitest-environment jsdom
/**
 * The socket heartbeat asks "is the backend busy, or gone?" through one HTTP
 * request. Only a TIMEOUT means busy (the event loop is blocked and cannot
 * answer anything); a reply or a refused connection both mean the socket is dead.
 */
import { describe, it, expect, afterEach } from 'vitest';

import api, { probeBackend } from './api';

const original = api.defaults.adapter;
afterEach(() => { api.defaults.adapter = original; });

const err = (config, extra) =>
  Promise.reject(Object.assign(new Error('x'), { isAxiosError: true, config, ...extra }));

describe('probeBackend', () => {
  it("is 'ok' when the backend answers", async () => {
    api.defaults.adapter = (config) =>
      Promise.resolve({ status: 200, data: {}, statusText: 'OK', headers: {}, config });
    expect(await probeBackend()).toBe('ok');
  });

  it("is 'ok' for an HTTP error status too: something answered", async () => {
    api.defaults.adapter = (config) => err(config, { response: { status: 500, data: {}, config } });
    expect(await probeBackend()).toBe('ok');
  });

  it("is 'timeout' when the request times out", async () => {
    api.defaults.adapter = (config) => err(config, { code: 'ECONNABORTED' });
    expect(await probeBackend()).toBe('timeout');
  });

  it("is 'error' when the connection is refused", async () => {
    api.defaults.adapter = (config) => err(config, { code: 'ERR_NETWORK' });
    expect(await probeBackend()).toBe('error');
  });
});
