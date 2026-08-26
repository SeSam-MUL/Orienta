// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  installGlobalErrorReporter,
  reportError,
  reportUiError,
  _resetForTests,
} from './errorReporter';
import { addBreadcrumb, getBreadcrumbs, _clearBreadcrumbs } from './breadcrumbs';

let fetchMock;

function sentPayloads() {
  return fetchMock.mock.calls.map(([, opts]) => JSON.parse(opts.body));
}

beforeEach(() => {
  fetchMock = vi.fn(() => Promise.resolve({ ok: true }));
  global.fetch = fetchMock;
  _resetForTests();
  _clearBreadcrumbs();
});

describe('reportError', () => {
  it('posts kind, message, stack and page to the backend', () => {
    reportError('react-boundary', new Error('boom'), {
      componentStack: 'in PhaseMapPage',
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/system/frontend-error');
    expect(opts.method).toBe('POST');
    const payload = JSON.parse(opts.body);
    expect(payload.kind).toBe('react-boundary');
    expect(payload.message).toBe('boom');
    expect(payload.stack).toContain('boom');
    expect(payload.componentStack).toBe('in PhaseMapPage');
  });

  it('attaches the breadcrumb trail so the log shows what came before', () => {
    addBreadcrumb('nav', 'page → indexing');
    addBreadcrumb('http-error', 'POST /api/indexing/start → 500');
    reportError('react-boundary', new Error('later crash'));
    const [payload] = sentPayloads();
    expect(payload.breadcrumbs).toContain('page → indexing');
    expect(payload.breadcrumbs).toContain('POST /api/indexing/start → 500');
  });

  it('dedupes identical errors within a session', () => {
    reportError('window-error', new Error('same thing'));
    reportError('window-error', new Error('same thing'));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('caps the number of reports per session', () => {
    for (let i = 0; i < 200; i++) reportError('window-error', new Error(`e${i}`));
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(60);
  });

  it('survives a failing transport', () => {
    fetchMock.mockImplementation(() => Promise.reject(new Error('offline')));
    expect(() => reportError('window-error', new Error('x'))).not.toThrow();
  });

  it('handles non-Error values', () => {
    reportError('unhandled-rejection', 'just a string');
    expect(sentPayloads()[0].message).toBe('just a string');
  });
});

describe('reportUiError', () => {
  it('reports a message the user was shown, with its location', () => {
    reportUiError('Dictionary not in memory', 'IndexingPage');
    const [payload] = sentPayloads();
    expect(payload.kind).toBe('ui-error');
    expect(payload.message).toBe('Dictionary not in memory');
    expect(payload.where).toBe('IndexingPage');
  });

  it('also drops a breadcrumb so later errors carry the context', () => {
    reportUiError('Load failed', 'EBSDViewer');
    const [crumb] = getBreadcrumbs();
    expect(crumb.type).toBe('ui-error');
    expect(crumb.message).toBe('EBSDViewer: Load failed');
  });

  it('ignores empty messages', () => {
    reportUiError('');
    reportUiError(null);
    reportUiError('   ');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('installGlobalErrorReporter', () => {
  it('reports window error events', () => {
    installGlobalErrorReporter();
    window.dispatchEvent(
      new ErrorEvent('error', {
        error: new Error('render exploded'),
        filename: 'chunk-abc.js',
        lineno: 42,
      }),
    );
    const [payload] = sentPayloads();
    expect(payload.kind).toBe('window-error');
    expect(payload.message).toBe('render exploded');
    expect(payload.source).toBe('chunk-abc.js:42');
  });

  it('reports unhandled promise rejections', () => {
    installGlobalErrorReporter();
    const event = new Event('unhandledrejection');
    event.reason = new Error('promise died');
    window.dispatchEvent(event);
    const [payload] = sentPayloads();
    expect(payload.kind).toBe('unhandled-rejection');
    expect(payload.message).toBe('promise died');
  });

  it('is idempotent — double install does not double-report', () => {
    installGlobalErrorReporter();
    installGlobalErrorReporter();
    window.dispatchEvent(new ErrorEvent('error', { error: new Error('once') }));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
