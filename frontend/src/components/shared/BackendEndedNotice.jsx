/**
 * A notice that the backend stopped on its own (a crash, a kill from outside).
 *
 * The desktop shell restarts a backend that ends unexpectedly (electron/main.js,
 * maybeRestartBackend). Before 0.4.8 that happened in silence: the page lost
 * the file it had loaded, showed "reconnecting" for a moment and came back
 * empty, and nothing said that the backend had died or why. The shell now sends
 * every unexpected end (`window.electronAPI.onBackendEnded`); this notice says
 * that it happened, how it ended, what the backend was working on, and whether
 * it is being started again. It stays until it is dismissed.
 *
 * Only in the desktop app: a browser page has no shell to tell it.
 */
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

const DISMISSED_KEY = 'orienta.backendEndDismissed';

function dismissedAt() {
  try { return sessionStorage.getItem(DISMISSED_KEY); } catch { return null; }
}

function rememberDismissed(at) {
  try { sessionStorage.setItem(DISMISSED_KEY, at); } catch { /* forgetting is the fallback */ }
}

export default function BackendEndedNotice({ api = typeof window !== 'undefined' ? window.electronAPI : null }) {
  const { t } = useTranslation('shell');
  const [notice, setNotice] = useState(null);

  useEffect(() => {
    if (!api || typeof api.onBackendEnded !== 'function') return undefined;
    let alive = true;
    // An end that happened before this page loaded (a reload after the crash) is shown
    // too, unless it was dismissed already.
    Promise.resolve(api.lastBackendEnd ? api.lastBackendEnd() : null)
      .then((n) => { if (alive && n && n.at !== dismissedAt()) setNotice(n); })
      .catch(() => {});
    const off = api.onBackendEnded((n) => { if (alive) setNotice(n); });
    return () => { alive = false; if (typeof off === 'function') off(); };
  }, [api]);

  if (!notice) return null;
  const time = (() => {
    try { return new Date(notice.at).toLocaleTimeString(); } catch { return ''; }
  })();
  return (
    <div role="alert" data-backend-ended style={{
      padding: '6px 12px',
      background: `${colors.red}18`,
      borderBottom: `1px solid ${colors.red}44`,
      fontSize: '9pt',
      color: colors.red,
      display: 'flex',
      gap: 12,
      alignItems: 'flex-start',
      flexShrink: 0,
    }}>
      <div style={{ flex: 1 }}>
        <div>{t('backendEnded.title', { time, exit: notice.exit })}</div>
        {notice.lastRequest && (
          <div>{t('backendEnded.lastRequest', { request: notice.lastRequest })}</div>
        )}
        <div>{notice.restarting ? t('backendEnded.restarting') : t('backendEnded.notRestarting')}</div>
        <div>{t('backendEnded.details')}</div>
      </div>
      <button
        type="button"
        onClick={() => { rememberDismissed(notice.at); setNotice(null); }}
        style={{ background: 'none', border: `1px solid ${colors.red}66`, color: colors.red,
                 borderRadius: 4, cursor: 'pointer', fontSize: '9pt', padding: '2px 8px' }}
      >
        {t('backendEnded.dismiss')}
      </button>
    </div>
  );
}
