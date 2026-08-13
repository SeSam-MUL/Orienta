import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { colors } from '../../theme/components';

/**
 * A small right-click menu anchored at a viewport point.
 *
 * The caller owns the open/closed state: it stores `{x, y}` on contextmenu and
 * clears it in `onClose`. This component only positions itself, keeps itself on
 * screen, and closes on the usual gestures (Escape, click elsewhere, scroll,
 * window blur, resize).
 *
 * items: [{ id, label, onSelect, disabled?, danger? }]
 */
export default function ContextMenu({ x, y, items = [], onClose }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ left: x, top: y });

  // Measure after mount and nudge back inside the viewport. useLayoutEffect so
  // the correction happens before paint — otherwise the menu visibly jumps.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const pad = 6;
    const left = Math.max(pad, Math.min(x, window.innerWidth - r.width - pad));
    const top = Math.max(pad, Math.min(y, window.innerHeight - r.height - pad));
    setPos({ left, top });
  }, [x, y]);

  useEffect(() => {
    const close = () => onClose?.();
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
    // `true` (capture) so the menu closes even when the click lands on an
    // element that stops propagation on its way up.
    const onDown = (e) => { if (!ref.current?.contains(e.target)) close(); };
    document.addEventListener('mousedown', onDown, true);
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('blur', close);
    window.addEventListener('resize', close);
    // Capture phase again: scrolling happens on inner panels, not on window.
    document.addEventListener('scroll', close, true);
    return () => {
      document.removeEventListener('mousedown', onDown, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('blur', close);
      window.removeEventListener('resize', close);
      document.removeEventListener('scroll', close, true);
    };
  }, [onClose]);

  return createPortal(
    <div
      ref={ref}
      role="menu"
      data-context-menu
      style={{
        position: 'fixed', left: pos.left, top: pos.top, zIndex: 4000,
        minWidth: 180,
        background: colors.bgSecondary,
        border: `1px solid ${colors.border}`,
        borderRadius: 6,
        boxShadow: '0 10px 30px rgba(0,0,0,.45)',
        padding: 4,
        fontSize: '9.5pt',
        userSelect: 'none',
      }}
      onContextMenu={(e) => e.preventDefault()}
      onClick={(e) => e.stopPropagation()}
    >
      {items.map((it) => (
        <div
          key={it.id}
          role="menuitem"
          tabIndex={it.disabled ? -1 : 0}
          aria-disabled={!!it.disabled}
          data-context-item={it.id}
          onClick={() => { if (!it.disabled) { onClose?.(); it.onSelect?.(); } }}
          onKeyDown={(e) => {
            if ((e.key === 'Enter' || e.key === ' ') && !it.disabled) {
              e.preventDefault();
              onClose?.();
              it.onSelect?.();
            }
          }}
          onMouseEnter={(e) => { if (!it.disabled) e.currentTarget.style.background = colors.sidebarActive; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
          style={{
            padding: '6px 12px',
            borderRadius: 4,
            cursor: it.disabled ? 'default' : 'pointer',
            color: it.disabled ? colors.textSecondary : (it.danger ? colors.red : colors.text),
            opacity: it.disabled ? 0.5 : 1,
            whiteSpace: 'nowrap',
          }}
        >
          {it.label}
        </div>
      ))}
    </div>,
    document.body,
  );
}
