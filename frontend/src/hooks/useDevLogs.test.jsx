// @vitest-environment jsdom
/**
 * useDevLogs: closing the socket, and reconnecting only when we did not.
 *
 * The hook stopped calling ws.close() in onerror (that tore down a socket
 * mid-handshake and logged a warning on every load) and now closes through
 * closeWebSocket() from services/api, which marks the socket
 * __closingIntentionally so onclose does not schedule a reconnect. That is a
 * contract split across two modules, and it had no test: a refactor of
 * closeWebSocket would silently bring back a reconnect loop after unmount.
 */
import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import useDevLogs from './useDevLogs';

const sockets = [];

class FakeWS {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url) {
    this.url = url;
    this.readyState = FakeWS.CONNECTING;
    this.closeCalls = 0;
    this.listeners = {};
    sockets.push(this);
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }

  close() {
    this.closeCalls += 1;
    this.readyState = FakeWS.CLOSED;
    this.onclose?.({});
  }

  // --- test helpers, not part of the WebSocket API
  finishHandshake() {
    this.readyState = FakeWS.OPEN;
    this.onopen?.({});
    (this.listeners.open || []).forEach((fn) => fn({}));
  }

  failHandshake() {
    this.onerror?.({});          // the browser fires error ...
    this.readyState = FakeWS.CLOSED;
    this.onclose?.({});          // ... and then always close
  }
}

beforeEach(() => {
  sockets.length = 0;
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', FakeWS);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('useDevLogs socket lifecycle', () => {
  it('reconnects after a failed handshake', () => {
    renderHook(() => useDevLogs());
    expect(sockets).toHaveLength(1);

    act(() => { sockets[0].failHandshake(); });
    expect(sockets).toHaveLength(1);           // not immediately

    act(() => { vi.advanceTimersByTime(3000); });
    expect(sockets).toHaveLength(2);           // the reconnect
  });

  it('does not close a socket that is still connecting, and does not reconnect after unmount', () => {
    const { unmount } = renderHook(() => useDevLogs());
    const ws = sockets[0];
    expect(ws.readyState).toBe(FakeWS.CONNECTING);

    act(() => { unmount(); });
    expect(ws.closeCalls).toBe(0);             // closing mid-handshake warns in Chrome
    expect(ws.__closingIntentionally).toBe(true);

    act(() => { ws.finishHandshake(); });      // closeWebSocket waits for 'open'
    expect(ws.closeCalls).toBe(1);

    act(() => { vi.advanceTimersByTime(10000); });
    expect(sockets).toHaveLength(1);           // no reconnect loop after unmount
  });

  it('closes an open socket once on unmount and stays closed', () => {
    const { unmount } = renderHook(() => useDevLogs());
    const ws = sockets[0];
    act(() => { ws.finishHandshake(); });

    act(() => { unmount(); });
    expect(ws.closeCalls).toBe(1);

    act(() => { vi.advanceTimersByTime(10000); });
    expect(sockets).toHaveLength(1);
  });

  it('closes the reconnected socket too', () => {
    const { unmount } = renderHook(() => useDevLogs());
    act(() => { sockets[0].failHandshake(); });
    act(() => { vi.advanceTimersByTime(3000); });
    expect(sockets).toHaveLength(2);

    act(() => { sockets[1].finishHandshake(); });
    act(() => { unmount(); });
    expect(sockets[1].closeCalls).toBe(1);     // the ref follows the newest socket
  });
});
