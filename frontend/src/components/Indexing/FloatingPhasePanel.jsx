/**
 * FloatingPhasePanel — Draggable floating window for phase selection.
 *
 * Can be toggled open/closed. When open, floats above the page content
 * and can be dragged by its title bar. Remembers position.
 *
 * Props:
 *   open: boolean
 *   onClose: () => void
 *   title: string
 *   children: React content
 *   defaultX: initial X position (default: 100)
 *   defaultY: initial Y position (default: 100)
 */

import { useRef, useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';

export default function FloatingPhasePanel({
  open,
  onClose,
  title,
  children,
  defaultX = 100,
  defaultY = 80,
}) {
  const { t } = useTranslation('indexing');
  const panelTitle = title ?? t('floatingPanel.defaultTitle');
  const panelRef = useRef(null);
  const [pos, setPos] = useState({ x: defaultX, y: defaultY });
  const [dragging, setDragging] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const [minimized, setMinimized] = useState(false);

  const handleMouseDown = useCallback((e) => {
    // Only drag from title bar
    if (e.target.closest('[data-no-drag]')) return;
    setDragging(true);
    const rect = panelRef.current?.getBoundingClientRect();
    if (rect) {
      dragOffset.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
    e.preventDefault();
  }, []);

  useEffect(() => {
    if (!dragging) return;
    function handleMouseMove(e) {
      setPos({
        x: e.clientX - dragOffset.current.x,
        y: e.clientY - dragOffset.current.y,
      });
    }
    function handleMouseUp() {
      setDragging(false);
    }
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragging]);

  if (!open) return null;

  return (
    <div
      ref={panelRef}
      data-floating-panel
      style={{
        position: 'fixed',
        left: pos.x,
        top: pos.y,
        zIndex: 500,
        width: minimized ? 260 : 380,
        maxHeight: minimized ? 'auto' : '70vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: C.bgSecondary,
        border: `1px solid ${C.border}`,
        borderRadius: 6,
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        overflow: 'hidden',
        userSelect: dragging ? 'none' : 'auto',
      }}
    >
      {/* Title bar — draggable */}
      <div
        onMouseDown={handleMouseDown}
        title={t('hoverTips.floatingPanelDragHeader')}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 10px',
          background: C.bg,
          borderBottom: `1px solid ${C.border}`,
          cursor: dragging ? 'grabbing' : 'grab',
          flexShrink: 0,
        }}
      >
        <span style={{ color: C.cyan, fontSize: '9pt', fontWeight: 700 }}>
          {panelTitle}
        </span>
        <div style={{ display: 'flex', gap: 4 }} data-no-drag>
          <TitleBarBtn
            label={minimized ? '▢' : '—'}
            title={minimized ? t('floatingPanel.restore') : t('floatingPanel.minimize')}
            onClick={() => setMinimized(m => !m)}
          />
          <TitleBarBtn
            label="✕"
            title={t('floatingPanel.close')}
            onClick={onClose}
            hoverColor={C.red}
          />
        </div>
      </div>

      {/* Content */}
      {!minimized && (
        <div style={{ overflowY: 'auto', flex: 1, padding: 8 }}>
          {children}
        </div>
      )}
    </div>
  );
}

function TitleBarBtn({ label, title, onClick, hoverColor }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        background: 'none',
        border: 'none',
        color: C.textSecondary,
        fontSize: '10pt',
        cursor: 'pointer',
        padding: '0 4px',
        lineHeight: 1,
        borderRadius: 2,
      }}
      onMouseEnter={e => e.currentTarget.style.color = hoverColor || C.text}
      onMouseLeave={e => e.currentTarget.style.color = C.textSecondary}
    >
      {label}
    </button>
  );
}
