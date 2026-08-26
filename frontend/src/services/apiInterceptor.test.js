// @vitest-environment jsdom
/**
 * The axios interceptor observes every request. Its most important property
 * is that it changes NOTHING for the ~463 existing catch blocks: the same
 * error object must come back out, with response/status/detail intact.
 *
 * Uses a stub adapter rather than axios-mock-adapter — no new dependency,
 * and the request interceptor still runs so config.metadata is exercised.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

const reportError = vi.fn();
vi.mock('./errorReporter', () => ({
  reportError: (...a) => reportError(...a),
  reportUiError: vi.fn(),
  installGlobalErrorReporter: vi.fn(),
}));

import api, { healthCheck } from './api';
import { getBreadcrumbs, _clearBreadcrumbs } from './breadcrumbs';

/** Adapter that resolves with the given status/data. */
function respondWith(status, data) {
  return (config) =>
    status >= 200 && status < 300
      ? Promise.resolve({ status, data, statusText: 'OK', headers: {}, config })
      : Promise.reject(
          Object.assign(new Error(`Request failed with status code ${status}`), {
            isAxiosError: true,
            config,
            response: { status, data, statusText: '', headers: {}, config },
          }),
        );
}

/** Adapter that fails before any response arrives. */
function networkError() {
  return (config) =>
    Promise.reject(
      Object.assign(new Error('Network Error'), { isAxiosError: true, config }),
    );
}

beforeEach(() => {
  reportError.mockClear();
  _clearBreadcrumbs();
});

describe('api response interceptor', () => {
  it('passes a failed request through unchanged', async () => {
    api.defaults.adapter = respondWith(400, { detail: 'No file loaded' });
    let caught = null;
    try {
      await api.get('/api/ebsd/info');
    } catch (err) {
      caught = err;
    }
    // Exactly what existing call sites read:
    expect(caught).toBeTruthy();
    expect(caught.response.status).toBe(400);
    expect(caught.response.data.detail).toBe('No file loaded');
    expect(typeof caught.message).toBe('string');
  });

  it('returns successful responses untouched', async () => {
    api.defaults.adapter = respondWith(200, { rows: 90, cols: 120 });
    const res = await api.get('/api/ebsd/info');
    expect(res.status).toBe(200);
    expect(res.data).toEqual({ rows: 90, cols: 120 });
  });

  it('records a breadcrumb for a successful call', async () => {
    api.defaults.adapter = respondWith(200, {});
    await api.get('/api/ebsd/load');
    const [crumb] = getBreadcrumbs();
    expect(crumb.type).toBe('http');
    expect(crumb.message).toContain('GET /api/ebsd/load → 200');
  });

  it('reports a failed call with status and detail', async () => {
    api.defaults.adapter = respondWith(500, { detail: 'phase_list required' });
    await expect(api.post('/api/indexing/start')).rejects.toBeTruthy();
    expect(reportError).toHaveBeenCalledTimes(1);
    const [kind, label] = reportError.mock.calls[0];
    expect(kind).toBe('http-error');
    expect(label).toContain('POST /api/indexing/start');
    expect(label).toContain('500');
    expect(label).toContain('phase_list required');
  });

  it('unwraps a nested detail.error payload', async () => {
    api.defaults.adapter = respondWith(404, { detail: { error: 'not computed yet' } });
    await expect(api.get('/api/forward-diagnostics/summary')).rejects.toBeTruthy();
    expect(reportError.mock.calls[0][1]).toContain('not computed yet');
  });

  it('reports a network failure with no response', async () => {
    api.defaults.adapter = networkError();
    await expect(api.get('/api/ebsd/load')).rejects.toBeTruthy();
    expect(reportError).toHaveBeenCalledTimes(1);
    expect(reportError.mock.calls[0][1]).toContain('no response');
  });

  it('names the loaded file — "which dataset was it?" is the 2nd question', async () => {
    api.defaults.adapter = respondWith(200, { success: true });
    await api.post('/api/ebsd/load', { path: 'D:/data/EBSD_SampleB.h5oina' });
    const crumbs = getBreadcrumbs();
    const fileCrumb = crumbs.find((c) => c.type === 'file');
    expect(fileCrumb).toBeTruthy();
    expect(fileCrumb.message).toBe('loaded EBSD_SampleB.h5oina');
    expect(fileCrumb.path).toBe('D:/data/EBSD_SampleB.h5oina');
  });

  it('notes a file switch too, and ignores calls without a path', async () => {
    api.defaults.adapter = respondWith(200, {});
    await api.post('/api/ebsd/switch-file', { path: '/home/x/Ni.h5' });
    await api.post('/api/indexing/start', { method: 'hough' });
    const files = getBreadcrumbs().filter((c) => c.type === 'file');
    expect(files).toHaveLength(1);
    expect(files[0].message).toBe('loaded Ni.h5');
  });

  it('stays silent for polling endpoints so a down backend cannot flood', async () => {
    api.defaults.adapter = networkError();
    await expect(healthCheck()).rejects.toBeTruthy();
    expect(reportError).not.toHaveBeenCalled();
    expect(getBreadcrumbs()).toHaveLength(0);
  });
});
