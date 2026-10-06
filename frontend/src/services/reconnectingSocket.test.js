// @vitest-environment jsdom
/**
 * The backend socket must come back by itself.
 *
 * The page keeps one WebSocket to /ws. The backend can drop it (a restart, a
 * broadcast that finds the peer gone, a laptop waking from sleep) and the
 * browser can drop it (a throttled background tab, a network change). Whatever
 * the reason, the user must not have to reload the page to get live messages
 * back, and a socket that silently stops delivering (half-open) must be
 * noticed and replaced as well.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { backoffDelayMs, openReconnectingSocket } from './reconnectingSocket';

const sockets = [];

class FakeWS {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url) {
    this.url = url;
    this.readyState = FakeWS.CONNECTING;
    this.sent = [];
    this.listeners = {};
    sockets.push(this);
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  send(data) { this.sent.push(JSON.parse(data)); }

  close() {
    if (this.readyState === FakeWS.CLOSED) return;
    this.readyState = FakeWS.CLOSED;
    this.onclose?.({});
  }

  // --- test helpers
  open() { this.readyState = FakeWS.OPEN; this.onopen?.({}); }
  receive(obj) { this.onmessage?.({ data: JSON.stringify(obj) }); }
  // The server (or the network) ends the connection: the browser fires close.
  dropFromServer() { this.readyState = FakeWS.CLOSED; this.onclose?.({}); }
  failHandshake() { this.onerror?.({}); this.readyState = FakeWS.CLOSED; this.onclose?.({}); }
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

const make = (extra = {}) => {
  const onMessage = vi.fn();
  const onStatus = vi.fn();
  const handle = openReconnectingSocket({
    url: () => 'ws://backend.test/ws', onMessage, onStatus, ...extra,
  });
  return { handle, onMessage, onStatus };
};

describe('backoffDelayMs', () => {
  it('doubles from the base and stops at the cap', () => {
    const seq = [0, 1, 2, 3, 4, 5, 9].map((n) => backoffDelayMs(n, { baseMs: 1000, maxMs: 15000 }));
    expect(seq).toEqual([1000, 2000, 4000, 8000, 15000, 15000, 15000]);
  });
});

describe('openReconnectingSocket', () => {
  it('connects once and reports open', () => {
    const { onStatus } = make();
    expect(sockets).toHaveLength(1);
    expect(sockets[0].url).toBe('ws://backend.test/ws');
    sockets[0].open();
    expect(onStatus).toHaveBeenLastCalledWith('open');
  });

  it('reconnects after the server drops the socket, and messages flow again', () => {
    const { onMessage, onStatus } = make();
    sockets[0].open();
    sockets[0].receive({ type: 'progress', progress: 0.1 });
    expect(onMessage).toHaveBeenCalledTimes(1);

    sockets[0].dropFromServer();
    expect(onStatus).toHaveBeenLastCalledWith('reconnecting');
    expect(sockets).toHaveLength(1);              // waits first, no tight loop

    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(2);
    sockets[1].open();
    expect(onStatus).toHaveBeenLastCalledWith('open');
    sockets[1].receive({ type: 'progress', progress: 0.5 });
    expect(onMessage).toHaveBeenCalledTimes(2);
    expect(onMessage).toHaveBeenLastCalledWith({ type: 'progress', progress: 0.5 });
  });

  it('backs off while the backend stays away and starts over after a success', () => {
    make();
    sockets[0].failHandshake();
    vi.advanceTimersByTime(1000); expect(sockets).toHaveLength(2);
    sockets[1].failHandshake();
    vi.advanceTimersByTime(1999); expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(1);    expect(sockets).toHaveLength(3);   // 2 s after the second failure
    sockets[2].open();
    sockets[2].dropFromServer();
    vi.advanceTimersByTime(1000);                                    // back to the base delay
    expect(sockets).toHaveLength(4);
  });

  it('does not reconnect after close(), and does not call back afterwards', () => {
    const { handle, onStatus } = make();
    sockets[0].open();
    handle.close();
    onStatus.mockClear();
    vi.advanceTimersByTime(60000);
    expect(sockets).toHaveLength(1);
    expect(onStatus).not.toHaveBeenCalled();
  });

  it('close() during the handshake leaves no reconnect behind', () => {
    const { handle } = make();
    handle.close();
    sockets[0].failHandshake();
    vi.advanceTimersByTime(60000);
    expect(sockets).toHaveLength(1);
  });

  it('replaces a half-open socket: a ping with no answer forces a reconnect', () => {
    const { onStatus } = make({ heartbeatMs: 20000, pongTimeoutMs: 5000 });
    sockets[0].open();
    vi.advanceTimersByTime(20000);
    expect(sockets[0].sent).toEqual([{ type: 'ping' }]);
    vi.advanceTimersByTime(5000);                  // no pong, no traffic at all
    expect(onStatus).toHaveBeenLastCalledWith('reconnecting');
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(2);
  });

  it('keeps a socket that answers: any message counts as proof of life', () => {
    make({ heartbeatMs: 20000, pongTimeoutMs: 5000 });
    sockets[0].open();
    vi.advanceTimersByTime(20000);
    sockets[0].receive({ type: 'pong' });
    vi.advanceTimersByTime(5000);
    expect(sockets).toHaveLength(1);
    expect(sockets[0].readyState).toBe(FakeWS.OPEN);
  });

  it('retries at once when the tab becomes visible or the network returns', () => {
    make();
    sockets[0].open();
    sockets[0].dropFromServer();
    expect(sockets).toHaveLength(1);            // still waiting for the 1 s timer
    window.dispatchEvent(new Event('online'));
    expect(sockets).toHaveLength(2);
    // a second trigger while a socket is already connecting must not stack one
    window.dispatchEvent(new Event('online'));
    expect(sockets).toHaveLength(2);
  });

  it('ignores frames that are not JSON instead of dying', () => {
    const { onMessage } = make();
    sockets[0].open();
    expect(() => sockets[0].onmessage?.({ data: 'not json' })).not.toThrow();
    sockets[0].receive({ type: 'x' });
    expect(onMessage).toHaveBeenCalledTimes(1);
  });
});
