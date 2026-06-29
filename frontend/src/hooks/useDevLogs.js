/**
 * useDevLogs — manages WebSocket subscription for Dev-Panel log entries.
 *
 * Listens to the existing /ws WebSocket for messages with type "dev_log"
 * and maintains a ring buffer of log entries. Updates are batched over a
 * 200 ms window so a busy backend (e.g. status polling 10 Hz during a
 * 12 h batch) doesn't trigger a Root re-render on every single message.
 *
 * Pass `enabled=false` to keep the WebSocket open but drop messages —
 * useful when the Dev Panel is collapsed and the logs would only pile
 * pressure on React without anyone seeing them.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

const MAX_ENTRIES = 500;
const FLUSH_INTERVAL_MS = 200;

/**
 * @param {{ enabled?: boolean }} [opts] — set enabled=false to drop incoming
 *   messages instead of buffering them. WebSocket itself stays open.
 * @returns {{ logs: Array, clearLogs: Function, isConnected: boolean }}
 */
export default function useDevLogs(opts = {}) {
  const enabled = opts.enabled !== false;  // default true for back-compat
  const [logs, setLogs] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  // Inbox holds incoming entries until the flush timer fires. setLogs runs
  // at most once per FLUSH_INTERVAL_MS regardless of how many messages
  // arrived in that window — keeps the React tree calm during heavy polling.
  const inboxRef = useRef([]);
  const flushTimerRef = useRef(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  const clearLogs = useCallback(() => {
    inboxRef.current = [];
    setLogs([]);
  }, []);

  useEffect(() => {
    let mounted = true;

    const flushInbox = () => {
      flushTimerRef.current = null;
      const pending = inboxRef.current;
      if (pending.length === 0) return;
      inboxRef.current = [];
      setLogs(prev => {
        const merged = prev.concat(pending);
        return merged.length > MAX_ENTRIES ? merged.slice(-MAX_ENTRIES) : merged;
      });
    };

    const connect = () => {
      // api.js uses VITE_API_URL as the single source of truth for the
      // backend host. Read the same variable here so a custom backend URL
      // (e.g. pointing at a remote dev server) applies to both HTTP and
      // WebSocket traffic instead of just HTTP.
      const base = import.meta.env.VITE_API_URL
        || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
      const wsUrl = base.replace(/^http/, 'ws') + '/ws';

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (mounted) setIsConnected(true);
      };

      ws.onmessage = (event) => {
        if (!mounted) return;
        if (!enabledRef.current) return;  // Dev Panel collapsed — drop log
        try {
          const data = JSON.parse(event.data);
          if (data.type !== 'dev_log') return;
          inboxRef.current.push(data);
          if (flushTimerRef.current === null) {
            flushTimerRef.current = setTimeout(flushInbox, FLUSH_INTERVAL_MS);
          }
        } catch {
          // Ignore non-JSON or malformed messages
        }
      };

      ws.onclose = () => {
        if (mounted) {
          setIsConnected(false);
          reconnectTimer.current = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      mounted = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { logs, clearLogs, isConnected };
}
