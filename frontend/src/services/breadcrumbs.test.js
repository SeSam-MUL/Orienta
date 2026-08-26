import { describe, it, expect, beforeEach } from 'vitest';
import {
  addBreadcrumb,
  addHttpBreadcrumb,
  getBreadcrumbs,
  formatBreadcrumbs,
  _clearBreadcrumbs,
} from './breadcrumbs';

beforeEach(() => _clearBreadcrumbs());

describe('breadcrumbs', () => {
  it('keeps events in order', () => {
    addBreadcrumb('nav', 'page → indexing');
    addBreadcrumb('action', 'clicked start');
    const trail = getBreadcrumbs();
    expect(trail.map((c) => c.message)).toEqual([
      'page → indexing',
      'clicked start',
    ]);
    expect(trail[0].type).toBe('nav');
  });

  it('caps the ring buffer and drops the oldest', () => {
    for (let i = 0; i < 200; i++) addBreadcrumb('action', `event ${i}`);
    const trail = getBreadcrumbs();
    expect(trail.length).toBe(60);
    expect(trail[trail.length - 1].message).toBe('event 199');
    expect(trail[0].message).toBe('event 140');
  });

  it('records HTTP calls with status and duration', () => {
    addHttpBreadcrumb({
      method: 'post',
      url: '/api/indexing/start',
      status: 200,
      durationMs: 1234.6,
    });
    const [crumb] = getBreadcrumbs();
    expect(crumb.type).toBe('http');
    expect(crumb.message).toBe('POST /api/indexing/start → 200 1235ms');
  });

  it('marks failures as http-error and includes the detail', () => {
    addHttpBreadcrumb({
      method: 'get',
      url: '/api/phasemap/layer',
      status: 400,
      detail: 'No indexing result loaded',
    });
    const [crumb] = getBreadcrumbs();
    expect(crumb.type).toBe('http-error');
    expect(crumb.message).toContain('→ 400');
    expect(crumb.message).toContain('No indexing result loaded');
  });

  it('marks a missing response explicitly', () => {
    addHttpBreadcrumb({ method: 'get', url: '/api/ebsd/info' });
    const [crumb] = getBreadcrumbs();
    expect(crumb.type).toBe('http-error');
    expect(crumb.message).toContain('(no response)');
  });

  it('ignores polling paths so they cannot flood the trail', () => {
    expect(addHttpBreadcrumb({ url: '/api/health', status: 200 })).toBe(false);
    expect(addHttpBreadcrumb({ url: '/api/indexing/progress', status: 200 })).toBe(false);
    expect(addHttpBreadcrumb({ url: '/api/system/frontend-error', status: 200 })).toBe(false);
    expect(getBreadcrumbs()).toHaveLength(0);
    // ...but real calls still land
    expect(addHttpBreadcrumb({ url: '/api/ebsd/load', status: 200 })).toBe(true);
    expect(getBreadcrumbs()).toHaveLength(1);
  });

  it('formats with relative times, newest last', () => {
    const now = 10_000;
    addBreadcrumb('nav', 'first');
    addBreadcrumb('ui-error', 'second');
    const trail = getBreadcrumbs();
    trail[0].time = now - 5000;
    trail[1].time = now - 1200;
    const text = formatBreadcrumbs(now);
    const lines = text.split('\n');
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain('5.0s');
    expect(lines[0]).toContain('[nav] first');
    expect(lines[1]).toContain('1.2s');
    expect(lines[1]).toContain('[ui-error] second');
  });

  it('never throws on odd input', () => {
    expect(() => addBreadcrumb('action', undefined)).not.toThrow();
    expect(() => addHttpBreadcrumb({})).not.toThrow();
    expect(() => formatBreadcrumbs()).not.toThrow();
  });
});
