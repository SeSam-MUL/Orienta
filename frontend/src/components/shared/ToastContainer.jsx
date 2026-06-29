/**
 * ToastContainer — renders floating notification toasts.
 * Positioned bottom-right, auto-dismiss with manual close.
 */
import { useTranslation } from 'react-i18next';
import useToastStore from '../../stores/useToastStore';
import { colors } from '../../theme/tokens';

const TYPE_STYLES = {
  success: { bg: `${colors.green}22`, border: colors.green, icon: '\u2714' },
  error:   { bg: `${colors.red}22`,   border: colors.red,   icon: '\u2716' },
  info:    { bg: `${colors.cyan}22`,   border: colors.cyan,  icon: '\u2139' },
  warning: { bg: `${colors.orange}22`, border: colors.orange, icon: '\u26A0' },
};

export default function ToastContainer() {
  const { t } = useTranslation('shell');
  const toasts = useToastStore((s) => s.toasts);
  const removeToast = useToastStore((s) => s.removeToast);

  if (toasts.length === 0) return null;

  return (
    <div style={{
      position: 'fixed', bottom: 36, right: 16, zIndex: 9999,
      display: 'flex', flexDirection: 'column-reverse', gap: 6,
      pointerEvents: 'none', maxWidth: 360,
    }}>
      {toasts.map((toast) => {
        const s = TYPE_STYLES[toast.type] || TYPE_STYLES.info;
        return (
          <div
            key={toast.id}
            role={toast.type === 'error' ? 'alert' : 'status'}
            style={{
              pointerEvents: 'auto',
              background: s.bg, border: `1px solid ${s.border}`,
              borderLeft: `3px solid ${s.border}`,
              borderRadius: 6, padding: '8px 12px',
              display: 'flex', alignItems: 'flex-start', gap: 8,
              fontSize: '10pt', color: colors.text,
              animation: 'pageFadeIn 0.2s ease-out',
              boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
              backdropFilter: 'blur(8px)',
            }}
          >
            <span style={{ fontSize: '12pt', flexShrink: 0 }}>{s.icon}</span>
            <span style={{ flex: 1, lineHeight: 1.4 }}>{toast.message}</span>
            <button
              aria-label={t('hoverTips.dismissNotification')}
              onClick={() => removeToast(toast.id)}
              style={{
                background: 'transparent', border: 'none', color: colors.textSecondary,
                cursor: 'pointer', fontSize: '11pt', padding: '0 2px', lineHeight: 1,
                borderRadius: 3, transition: 'color 0.12s, background 0.12s',
              }}
              onMouseEnter={e => { e.currentTarget.style.color = colors.text; e.currentTarget.style.background = `${colors.border}55`; }}
              onMouseLeave={e => { e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.background = 'transparent'; }}
            >&times;</button>
          </div>
        );
      })}
    </div>
  );
}
