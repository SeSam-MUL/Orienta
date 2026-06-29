/**
 * Modal — generic overlay dialog with focus trap, Esc-to-close, backdrop click-to-close.
 *
 * Behaviour change vs the previous inline implementation: initial focus is now
 * applied via useEffect-on-mount rather than setTimeout(50). This removes the
 * race where the dialog content rendered after the timer fired (giving
 * inconsistent focus on slow renders) and aligns with the React lifecycle:
 * on mount the children are already in the DOM, so querying for the first
 * focusable element returns the correct node deterministically.
 *
 * Esc-to-close, backdrop click-to-close, and Tab/Shift+Tab focus-trapping
 * behaviour are unchanged.
 */
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';

export default function Modal({ title, onClose, children }) {
  const { t } = useTranslation('hdf5viewer');
  const dialogRef = useRef(null);

  // Initial focus when modal mounts. Deps are [] (mount-only) so a parent
  // re-render that passes a new inline onClose arrow does NOT rip focus out
  // of whatever element the user is currently interacting with inside the
  // dialog. Children are already in the DOM by the time this effect runs.
  useEffect(() => {
    const focusable = dialogRef.current?.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    focusable?.focus();
  }, []);

  // Esc-to-close + Tab focus-trap. Re-binds when onClose identity changes;
  // re-binding a single keydown listener is cheap and behaviourally invisible.
  useEffect(() => {
    const h = (e) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'Tab') {
        const focusable = dialogRef.current?.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable || focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) { e.preventDefault(); last.focus(); }
        } else {
          if (document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      }
    };
    window.addEventListener('keydown', h);
    return () => { window.removeEventListener('keydown', h); };
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: 8,
          padding: 20,
          minWidth: 340, maxWidth: 520,
          maxHeight: '80vh', overflowY: 'auto',
          boxShadow: '0 8px 40px rgba(0,0,0,0.7)',
          animation: 'fadeSlideIn 0.2s ease-out',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
          <span style={{ flex: 1, fontSize: '11pt', fontWeight: 600, color: colors.text }}>
            {title}
          </span>
          <button
            aria-label={t('modal.closeTooltip')}
            title={t('modal.closeTooltip')}
            onClick={onClose}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: colors.textSecondary, fontSize: 18, lineHeight: 1, padding: '0 4px',
              borderRadius: 3, transition: 'color 0.12s, background 0.12s',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = colors.text; e.currentTarget.style.background = `${colors.border}55`; }}
            onMouseLeave={e => { e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.background = 'none'; }}
          >
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
