/**
 * A WebSocket to the backend that comes back by itself.
 *
 * Why this exists: the page holds one socket to /ws (live log lines, progress).
 * It can end for reasons the user cannot see or influence - the backend was
 * restarted, a server-side broadcast found the peer gone, the machine slept, the
 * browser throttled a background tab - and a socket that is gone for good means
 * a page that has to be reloaded by hand. This module owns the whole life cycle:
 *
 *   - reconnect with exponential backoff (1 s, 2 s, 4 s ... capped), reset by
 *     the first successful open;
 *   - a heartbeat: a socket can be half-open (the peer vanished without a close
 *     frame) and then neither delivers nor closes. Every `heartbeatMs` we send
 *     {"type":"ping"}; if nothing at all arrives within `pongTimeoutMs`, the
 *     socket is replaced. The backend answers a ping with {"type":"pong"};
 *     any inbound frame counts as proof of life;
 *   - an immediate retry when the tab becomes visible again or the network
 *     comes back, because timers in a background tab are throttled to a crawl
 *     and a backoff that has grown to 15 s would otherwise be paid in full;
 *   - a clean `close()` that leaves no timer, listener or socket behind.
 *
 * Callers get a status ('connecting' | 'open' | 'reconnecting') through
 * `onStatus` so they can resynchronise their state after a gap.
 */

export const BACKOFF_BASE_MS = 1000;
export const BACKOFF_MAX_MS = 15000;
export const HEARTBEAT_MS = 25000;
export const PONG_TIMEOUT_MS = 10000;

/** Delay before reconnect attempt number `attempt` (0 = first retry). */
export function backoffDelayMs(attempt, { baseMs = BACKOFF_BASE_MS, maxMs = BACKOFF_MAX_MS } = {}) {
  return Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
}

/**
 * @param {object} opts
 * @param {() => string} opts.url        evaluated on every attempt
 * @param {(msg: object) => void} [opts.onMessage]  parsed JSON frames only
 * @param {(status: string) => void} [opts.onStatus]
 * @returns {{ close: () => void, reconnectNow: () => void }}
 */
export function openReconnectingSocket({
  url,
  onMessage = () => {},
  onStatus = () => {},
  baseMs = BACKOFF_BASE_MS,
  maxMs = BACKOFF_MAX_MS,
  heartbeatMs = HEARTBEAT_MS,
  pongTimeoutMs = PONG_TIMEOUT_MS,
}) {
  let ws = null;
  let attempt = 0;
  let stopped = false;
  let retryTimer = null;
  let pingTimer = null;
  let pongTimer = null;
  let lastStatus = null;

  const emit = (status) => {
    if (stopped || status === lastStatus) return;
    lastStatus = status;
    try { onStatus(status); } catch (err) { console.error('WebSocket onStatus threw:', err); }
  };

  const clearHeartbeat = () => {
    if (pingTimer) { clearTimeout(pingTimer); pingTimer = null; }
    if (pongTimer) { clearTimeout(pongTimer); pongTimer = null; }
  };

  const scheduleRetry = () => {
    if (stopped || retryTimer) return;
    const delay = backoffDelayMs(attempt, { baseMs, maxMs });
    attempt += 1;
    retryTimer = setTimeout(() => { retryTimer = null; connect(); }, delay);
  };

  // The one place a socket is declared dead. Detaching the handlers first means
  // the close event of the socket we are abandoning cannot schedule a second
  // retry on top of this one.
  const abandon = (socket) => {
    clearHeartbeat();
    socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
    try { socket.close(); } catch { /* already closed */ }
    if (ws === socket) ws = null;
    emit('reconnecting');
    scheduleRetry();
  };

  const armHeartbeat = (socket) => {
    clearHeartbeat();
    pingTimer = setTimeout(() => {
      pingTimer = null;
      if (stopped || ws !== socket || socket.readyState !== WebSocket.OPEN) return;
      try { socket.send(JSON.stringify({ type: 'ping' })); } catch { abandon(socket); return; }
      pongTimer = setTimeout(() => { pongTimer = null; if (ws === socket) abandon(socket); },
        pongTimeoutMs);
    }, heartbeatMs);
  };

  function connect() {
    if (stopped) return;
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    emit(attempt === 0 && lastStatus === null ? 'connecting' : 'reconnecting');
    const socket = new WebSocket(url());
    ws = socket;

    socket.onopen = () => {
      if (stopped || ws !== socket) return;
      attempt = 0;
      emit('open');
      armHeartbeat(socket);
    };

    socket.onmessage = (event) => {
      if (stopped || ws !== socket) return;
      armHeartbeat(socket);               // proof of life: restart both timers
      let data;
      try { data = JSON.parse(event.data); } catch { return; }   // not a JSON frame
      try { onMessage(data); } catch (err) { console.error('WebSocket onMessage threw:', err); }
    };

    // The error event carries no detail and is always followed by close, which
    // owns the retry. Closing here too tore down sockets mid-handshake.
    socket.onerror = () => {};

    socket.onclose = () => {
      if (ws === socket) ws = null;
      clearHeartbeat();
      if (stopped) return;
      emit('reconnecting');
      scheduleRetry();
    };
  }

  // Skip the wait: a visible tab or a returning network is the moment to try.
  const reconnectNow = () => {
    if (stopped) return;
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
    connect();
  };
  const onVisible = () => { if (document.visibilityState === 'visible') reconnectNow(); };

  window.addEventListener('online', reconnectNow);
  document.addEventListener('visibilitychange', onVisible);

  connect();

  return {
    reconnectNow,
    close() {
      if (stopped) return;
      stopped = true;
      window.removeEventListener('online', reconnectNow);
      document.removeEventListener('visibilitychange', onVisible);
      if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
      clearHeartbeat();
      const socket = ws;
      ws = null;
      if (!socket) return;
      socket.onmessage = socket.onclose = socket.onerror = null;
      // Closing a socket that is still connecting makes the browser log a
      // warning on every page load; wait for the open, then close.
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => { try { socket.close(); } catch { /* ignore */ } };
      } else {
        socket.onopen = null;
        try { socket.close(); } catch { /* ignore */ }
      }
    },
  };
}
