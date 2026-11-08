// The frame of the calibration windows (the same look as "Take the pattern centre from..."): a dark backdrop, a
// panel, a title, and Escape closes it.
import React, { useEffect } from 'react';
import { colors } from '../../../theme/components';

export default function ModalShell({ title, label, testId, onClose, children, zIndex = 1000 }) {
  useEffect(() => {
    // Escape closes the window in front only: a window opened over this one has the higher z-index
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      const zs = [...document.querySelectorAll('[data-modal-z]')].map((n) => Number(n.dataset.modalZ));
      if (zs.length && Math.max(...zs) > zIndex) return;
      onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose, zIndex]);
  return (
    <div
      role="dialog" aria-modal="true" aria-label={label || title} data-testid={testId} data-modal-z={zIndex}
      style={{
        position: 'fixed', inset: 0, zIndex, background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        className="thin-scrollbar"
        style={{
          background: colors.bgSecondary || colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8, padding: 16,
          width: 'min(640px, 94vw)', maxHeight: '90vh', overflowY: 'auto', color: colors.text,
        }}
      >
        <h3 style={{ margin: '0 0 6px', fontSize: '12pt', overflowWrap: 'anywhere' }}>{title}</h3>
        {children}
      </div>
    </div>
  );
}
