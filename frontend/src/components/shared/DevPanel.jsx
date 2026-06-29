/**
 * DevPanel — Global collapsible developer log panel.
 *
 * Shows real-time backend logs, timing data, and HTTP requests streamed
 * via WebSocket. Sits on the right edge of the app, toggleable via click
 * or Ctrl+Shift+D.
 *
 * Props:
 *   logs       — array of log entries from useDevLogs()
 *   onClear    — callback to clear the log buffer
 *   isConnected — whether the WebSocket is connected
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

const PANEL_WIDTH = 350;
const TAB_WIDTH = 24;

const CATEGORIES = ['all', 'log', 'timing', 'http', 'err'];

const CAT_COLORS = {
  log:    colors.text,
  timing: colors.green,
  http:   colors.yellow,
  err:    colors.red,
};

const LEVEL_IS_ERROR = new Set(['WARNING', 'ERROR', 'CRITICAL']);

function categorize(entry) {
  if (entry.category === 'timing') return 'timing';
  if (entry.category === 'http') return 'http';
  if (LEVEL_IS_ERROR.has(entry.level)) return 'err';
  return 'log';
}

function formatEntry(entry) {
  const ts = entry.timestamp || '';
  if (entry.category === 'timing') {
    return `${ts} [TIME] ${entry.step} — ${entry.duration_ms}ms`;
  }
  if (entry.category === 'http') {
    const status = entry.status >= 400 ? `${entry.status} !!` : entry.status;
    return `${ts} [HTTP] ${entry.method} ${entry.path} ${status} ${entry.duration_ms}ms`;
  }
  const lvl = entry.level || 'INFO';
  const tag = LEVEL_IS_ERROR.has(lvl) ? `[${lvl}]` : '[LOG]';
  return `${ts} ${tag} ${entry.message}`;
}

export default function DevPanel({ logs, onClear, isConnected, open: openProp, onOpenChange }) {
  const { t } = useTranslation('shell');
  // Controlled when both props are supplied, otherwise fall back to local
  // state so old call sites keep working unchanged.
  const [openLocal, setOpenLocal] = useState(false);
  const open = openProp !== undefined ? openProp : openLocal;
  const setOpen = (next) => {
    const value = typeof next === 'function' ? next(open) : next;
    if (onOpenChange) onOpenChange(value);
    else setOpenLocal(value);
  };
  const [activeFilter, setActiveFilter] = useState('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);
  const scrollRef = useRef(null);

  // Keyboard shortcut: Ctrl+Shift+D
  useEffect(() => {
    const handler = (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        setOpen(prev => !prev);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll, open]);

  const filtered = useMemo(() => {
    if (activeFilter === 'all') return logs;
    return logs.filter(e => categorize(e) === activeFilter);
  }, [logs, activeFilter]);

  const handleCopy = () => {
    const text = filtered.map(formatEntry).join('\n');
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }).catch(() => {});
  };

  // --- Collapsed tab ---
  if (!open) {
    return (
      <div
        onClick={() => setOpen(true)}
        title={t('hoverTips.devOpen')}
        style={{
          width: TAB_WIDTH,
          minWidth: TAB_WIDTH,
          height: '100%',
          background: colors.bgSecondary,
          borderLeft: `1px solid ${colors.border}`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          userSelect: 'none',
          transition: 'background 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = colors.bgTertiary; }}
        onMouseLeave={e => { e.currentTarget.style.background = colors.bgSecondary; }}
      >
        <span style={{
          writingMode: 'vertical-rl',
          textOrientation: 'mixed',
          fontSize: '9pt',
          fontWeight: 700,
          color: isConnected ? colors.green : colors.red,
          letterSpacing: 2,
        }}>
          DEV
        </span>
        {logs.length > 0 && (
          <span style={{
            fontSize: '7pt',
            color: colors.textSecondary,
            marginTop: 4,
          }}>
            {logs.length}
          </span>
        )}
      </div>
    );
  }

  // --- Expanded panel ---
  return (
    <div style={{
      width: PANEL_WIDTH,
      minWidth: PANEL_WIDTH,
      height: '100%',
      background: colors.bgSecondary,
      borderLeft: `1px solid ${colors.border}`,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '6px 8px',
        borderBottom: `1px solid ${colors.border}`,
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: '9pt', fontWeight: 700, color: colors.accent }}>DEV</span>
        <span style={{
          width: 6, height: 6, borderRadius: '50%',
          background: isConnected ? colors.green : colors.red,
          flexShrink: 0,
        }} />
        <div style={{ flex: 1 }} />
        <button onClick={handleCopy} title={t('hoverTips.copyLogs')} style={btnStyle}>
          {copied ? '\u2713' : '\u2398'}
        </button>
        <button onClick={onClear} title={t('hoverTips.clearLogs')} style={btnStyle}>
          {'\u2715'}
        </button>
        <button onClick={() => setOpen(false)} title={t('hoverTips.closeDevPanel')} style={btnStyle}>
          {'\u25B6'}
        </button>
      </div>

      {/* Filter bar */}
      <div style={{
        display: 'flex',
        gap: 2,
        padding: '4px 8px',
        borderBottom: `1px solid ${colors.border}`,
        flexShrink: 0,
      }}>
        {CATEGORIES.map(cat => (
          <button
            key={cat}
            onClick={() => setActiveFilter(cat)}
            title={t('hoverTips.devLogFilter')}
            style={{
              background: activeFilter === cat ? `${colors.accent}33` : 'transparent',
              border: activeFilter === cat ? `1px solid ${colors.accent}66` : `1px solid transparent`,
              borderRadius: 3,
              color: cat === 'all' ? colors.text : (CAT_COLORS[cat] || colors.text),
              fontSize: '7.5pt',
              fontWeight: activeFilter === cat ? 700 : 400,
              padding: '1px 6px',
              cursor: 'pointer',
              textTransform: 'uppercase',
            }}
          >
            {cat}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button
          onClick={() => setAutoScroll(v => !v)}
          title={autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
          style={{
            ...btnStyle,
            color: autoScroll ? colors.green : colors.textSecondary,
          }}
        >
          {autoScroll ? '\u25BC' : '\u25A0'}
        </button>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '4px 8px',
          fontFamily: 'Consolas, "Fira Code", monospace',
          fontSize: '8pt',
          lineHeight: 1.5,
        }}
      >
        {filtered.length === 0 && (
          <div style={{ color: colors.textSecondary, fontStyle: 'italic', padding: '16px 0', textAlign: 'center' }}>
            {isConnected ? 'Waiting for backend logs...' : 'WebSocket disconnected'}
          </div>
        )}
        {filtered.map((entry, i) => {
          const cat = categorize(entry);
          return (
            <div key={i} style={{
              color: CAT_COLORS[cat] || colors.text,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              borderBottom: `1px solid ${colors.border}22`,
              padding: '1px 0',
            }}>
              {formatEntry(entry)}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div style={{
        padding: '2px 8px',
        borderTop: `1px solid ${colors.border}`,
        fontSize: '7pt',
        color: colors.textSecondary,
        textAlign: 'center',
        flexShrink: 0,
      }}>
        {filtered.length} / {logs.length} entries | Ctrl+Shift+D to toggle
      </div>
    </div>
  );
}

const btnStyle = {
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  color: colors.textSecondary,
  fontSize: '9pt',
  padding: '1px 4px',
  borderRadius: 3,
};
