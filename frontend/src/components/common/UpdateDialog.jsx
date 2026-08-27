/**
 * Update dialog — shown on start-up when a newer release exists.
 *
 * Three shapes, because three situations are genuinely different:
 *  - a git install can update itself: offer it,
 *  - an unzipped install cannot: say so and how to get the new version,
 *  - an update is running: show the steps, and never let the window close
 *    mid-way (a half-applied update is the one thing that breaks the app).
 */

import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import { startUpdate, getUpdateProgress } from '../../services/api';

const POLL_MS = 1000;
const REPO_URL = 'https://github.com/SeSam-MUL/Orienta';

const STEP_KEYS = {
  preflight: 'stepPreflight',
  starting: 'stepPreflight',
  fetch: 'stepFetch',
  checkout: 'stepCheckout',
  dependencies: 'stepDependencies',
  node_modules: 'stepNodeModules',
  build: 'stepBuild',
  done: 'stepDone',
};

export default function UpdateDialog({ info, onClose, onSkip }) {
  const { t } = useTranslation('settings');
  const [phase, setPhase] = useState('offer'); // offer | running | done | failed
  const [progress, setProgress] = useState(null);
  const timerRef = useRef(null);
  const logRef = useRef(null);

  const canSelfUpdate = info?.install_kind === 'git';

  useEffect(() => () => clearTimeout(timerRef.current), []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [progress]);

  const poll = async () => {
    try {
      const p = await getUpdateProgress();
      setProgress(p);
      if (p.state === 'done') { setPhase('done'); return; }
      if (p.state === 'failed') { setPhase('failed'); return; }
    } catch {
      // A restarting backend drops requests; keep polling.
    }
    timerRef.current = setTimeout(poll, POLL_MS);
  };

  const handleInstall = async () => {
    setPhase('running');
    try {
      await startUpdate(info.latest);
      poll();
    } catch (err) {
      setProgress({
        state: 'failed',
        error: err?.response?.data?.detail || err.message,
      });
      setPhase('failed');
    }
  };

  const handleRestart = () => {
    if (window.electronAPI?.relaunch) {
      window.electronAPI.relaunch();
    } else {
      window.location.reload();
    }
  };

  const busy = phase === 'running';
  const stepKey = STEP_KEYS[progress?.step] || null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('settings:update.title')}
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose?.(); }}
      style={{
        position: 'fixed', inset: 0, zIndex: 4100,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onMouseDown={(e) => e.stopPropagation()}
        style={{
          width: 'min(620px, 92vw)', maxHeight: '86vh', overflowY: 'auto',
          background: colors.bgSecondary, border: `1px solid ${colors.border}`,
          borderRadius: 6, padding: 20, color: colors.text, fontSize: 12,
          display: 'flex', flexDirection: 'column', gap: 12,
        }}
      >
        <h2 style={{ margin: 0, fontSize: 15 }}>{t('settings:update.title')}</h2>

        <div>
          {t('settings:update.versions', {
            current: info?.current || '—',
            latest: info?.latest || '—',
          })}
        </div>

        {phase === 'offer' && !canSelfUpdate && (
          <>
            <div style={{ color: colors.yellow }}>
              {t('settings:update.manualOnly')}
            </div>
            <div style={{ color: colors.textSecondary, fontSize: 11, lineHeight: 1.6 }}>
              {t('settings:update.manualHow')}
            </div>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              style={{ color: colors.cyan, fontSize: 11 }}
            >
              {REPO_URL}
            </a>
          </>
        )}

        {phase === 'offer' && canSelfUpdate && (
          <div style={{ color: colors.textSecondary, fontSize: 11, lineHeight: 1.6 }}>
            {t('settings:update.whatHappens')}
          </div>
        )}

        {info?.notes && phase === 'offer' && (
          <pre style={{
            margin: 0, padding: 10, maxHeight: 260, overflowY: 'auto',
            background: colors.bg, border: `1px solid ${colors.border}`,
            borderRadius: 4, fontSize: 11, lineHeight: 1.5,
            whiteSpace: 'pre-wrap', color: colors.textSecondary,
          }}>
            {info.notes}
          </pre>
        )}

        {(phase === 'running' || phase === 'failed' || phase === 'done') && (
          <>
            <div style={{ fontWeight: 600 }}>
              {phase === 'done'
                ? t('settings:update.doneMessage')
                : stepKey
                  ? t(`settings:update.${stepKey}`)
                  : t('settings:update.stepPreflight')}
            </div>
            {phase === 'running' && (
              <div style={{ color: colors.textSecondary, fontSize: 11 }}>
                {t('settings:update.doNotClose')}
              </div>
            )}
            {progress?.error && (
              <div style={{ color: colors.red, fontSize: 11, lineHeight: 1.6 }}>
                {progress.error}
                <div style={{ color: colors.textSecondary, marginTop: 6 }}>
                  {t('settings:update.failedSafe')}
                </div>
              </div>
            )}
            {progress?.log?.length > 0 && (
              <pre
                ref={logRef}
                style={{
                  margin: 0, padding: 8, height: 180, overflowY: 'auto',
                  background: colors.bg, border: `1px solid ${colors.border}`,
                  borderRadius: 4, fontSize: 10, lineHeight: 1.45,
                  whiteSpace: 'pre-wrap', color: colors.textSecondary,
                }}
              >
                {progress.log.join('\n')}
              </pre>
            )}
          </>
        )}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4, flexWrap: 'wrap' }}>
          {phase === 'offer' && (
            <>
              <button onClick={onSkip} style={ghost(colors)}>
                {t('settings:update.skipVersion')}
              </button>
              <button onClick={onClose} style={ghost(colors)}>
                {t('settings:update.later')}
              </button>
              {canSelfUpdate && (
                <button onClick={handleInstall} style={primary(colors)}>
                  {t('settings:update.install')}
                </button>
              )}
            </>
          )}
          {phase === 'failed' && (
            <button onClick={onClose} style={ghost(colors)}>
              {t('settings:update.close')}
            </button>
          )}
          {phase === 'done' && (
            <button onClick={handleRestart} style={primary(colors)}>
              {t('settings:update.restart')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const ghost = (c) => ({
  padding: '7px 16px', background: 'transparent',
  border: `1px solid ${c.border}`, borderRadius: 4,
  color: c.textSecondary, fontSize: 12, cursor: 'pointer',
});

const primary = (c) => ({
  padding: '7px 16px', background: c.accent, border: 'none', borderRadius: 4,
  color: c.textOnAccent, fontWeight: 600, fontSize: 12, cursor: 'pointer',
});
