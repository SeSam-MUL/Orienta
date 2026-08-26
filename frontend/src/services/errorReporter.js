/**
 * Frontend error capture → backend log file.
 *
 * Covers three classes, all of which used to be invisible outside DevTools:
 *   1. uncaught runtime errors and unhandled promise rejections,
 *   2. React render crashes (via the ErrorBoundary),
 *   3. errors the app HANDLES and shows to the user — the panel messages and
 *      toasts people screenshot. Those never reached any log before.
 *
 * Every report carries the breadcrumb trail, so the log shows what happened
 * before the failure, not just the failure.
 *
 * Transport is raw fetch on purpose: api.js imports this module for its
 * interceptor, so importing api.js back would be a cycle — and error
 * reporting should not depend on the axios instance that may itself be the
 * thing failing.
 */

import { addBreadcrumb, formatBreadcrumbs } from './breadcrumbs';

const API_BASE = import.meta.env.VITE_API_URL || '';
const ENDPOINT = `${API_BASE}/api/system/frontend-error`;

const MAX_REPORTS_PER_SESSION = 60;
const seen = new Set();
let reportCount = 0;

function currentPage() {
  try {
    return window.location.hash || window.location.pathname;
  } catch {
    return '';
  }
}

function send(payload) {
  try {
    fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true, // survives a page teardown right after the crash
    }).catch(() => {});
  } catch {
    // Reporting must never become a second error.
  }
}

/**
 * Send one error to the backend log. Deduped by kind+message and capped per
 * session, so an error that fires on every render cannot flood the backend.
 * Safe to call from anywhere — never throws.
 */
export function reportError(kind, error, extra = {}) {
  try {
    const message = String(error?.message ?? error ?? 'unknown error');
    const key = `${kind}|${message}`;
    if (seen.has(key) || reportCount >= MAX_REPORTS_PER_SESSION) return;
    seen.add(key);
    reportCount += 1;
    send({
      kind,
      message,
      stack: typeof error?.stack === 'string' ? error.stack : '',
      page: currentPage(),
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
      breadcrumbs: formatBreadcrumbs(),
      ...extra,
    });
  } catch {
    // ignore
  }
}

/**
 * Report an error message that was SHOWN to the user (panel text, toast,
 * status line). `where` names the component or page so the log says which
 * screen the user was looking at.
 *
 * Also lands in the breadcrumb trail, so a later, unrelated report still
 * shows that this message was on screen beforehand.
 */
export function reportUiError(message, where = '', extra = {}) {
  const text = String(message ?? '').trim();
  if (!text) return;
  addBreadcrumb('ui-error', where ? `${where}: ${text}` : text);
  reportError('ui-error', text, { where, ...extra });
}

/** Install window-level handlers. Idempotent. Call once, before render. */
export function installGlobalErrorReporter() {
  if (window.__orientaErrorReporterInstalled) return;
  window.__orientaErrorReporterInstalled = true;

  window.addEventListener('error', (event) => {
    // Resource-load errors (script/img) have no .error and little signal.
    const err = event.error ?? event.message;
    reportError('window-error', err, {
      source: `${event.filename || ''}:${event.lineno || 0}`,
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    reportError('unhandled-rejection', event.reason);
  });
}

/** Test hook: reset session state. */
export function _resetForTests() {
  seen.clear();
  reportCount = 0;
  delete window.__orientaErrorReporterInstalled;
}
