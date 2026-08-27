/**
 * "Report a problem" — turns a screenshot into an actionable report.
 *
 * The logs answer what happened; only the user can answer what they were
 * trying to do and what they expected instead. That sentence is what makes
 * "the result looks wrong" diagnosable, and no amount of logging replaces
 * it — so the report bundles both.
 *
 * After the report is saved it offers to open a pre-filled GitHub issue.
 * That is the step that turns a pile of individual reports into a list that
 * can be worked through in order: the issue carries the error id, so the same
 * fault reported by three people is visibly one problem.
 */

import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import { exportDiagnostics, getAppVersion } from '../../services/api';
import { getLastFingerprint } from '../../services/errorReporter';
import { formatBreadcrumbs } from '../../services/breadcrumbs';
import { downloadBlob } from './imageExport';
import { newIssueUrl } from './githubIssue';

export default function ProblemReportDialog({ onClose, screenshot = null }) {
  const { t } = useTranslation('settings');
  const [description, setDescription] = useState('');
  const [includeShot, setIncludeShot] = useState(Boolean(screenshot));
  const [state, setState] = useState('idle'); // idle | busy | done | failed
  const [versionInfo, setVersionInfo] = useState(null);
  const textareaRef = useRef(null);
  const fingerprint = getLastFingerprint();

  useEffect(() => {
    textareaRef.current?.focus();
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    document.addEventListener('keydown', onKey);
    let alive = true;
    getAppVersion()
      .then((v) => { if (alive) setVersionInfo(v); })
      .catch(() => {});
    return () => {
      alive = false;
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  const handleCreate = async () => {
    setState('busy');
    try {
      const blob = await exportDiagnostics({
        description,
        fingerprint,
        screenshot: includeShot ? screenshot : null,
      });
      const stamp = new Date().toISOString().slice(0, 10);
      downloadBlob(blob, `orienta-problem-report-${stamp}.zip`);
      setState('done');
    } catch {
      setState('failed');
    }
  };

  const handleOpenIssue = () => {
    const url = newIssueUrl(versionInfo?.repo_url, {
      description,
      version: versionInfo?.version,
      page: typeof window !== 'undefined' ? window.location.hash : '',
      fingerprint,
      breadcrumbs: formatBreadcrumbs(),
      environment: versionInfo?.branch ? `branch: ${versionInfo.branch}` : '',
      hasScreenshot: includeShot && Boolean(screenshot),
    });
    if (url) window.open(url, '_blank', 'noopener');
  };

  const canOpenIssue = Boolean(versionInfo?.repo_url);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('settings:report.title')}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose?.(); }}
      style={{
        position: 'fixed', inset: 0, zIndex: 4000,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onMouseDown={(e) => e.stopPropagation()}
        style={{
          width: 'min(600px, 92vw)', maxHeight: '88vh', overflowY: 'auto',
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`, borderRadius: 6,
          padding: 20, display: 'flex', flexDirection: 'column', gap: 12,
          color: colors.text, fontSize: 12,
        }}
      >
        <h2 style={{ margin: 0, fontSize: 15, color: colors.text }}>
          {t('settings:report.title')}
        </h2>

        <label htmlFor="problem-report-text" style={{ color: colors.textSecondary }}>
          {t('settings:report.prompt')}
        </label>
        <textarea
          id="problem-report-text"
          ref={textareaRef}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder={t('settings:report.placeholder')}
          rows={6}
          style={{
            width: '100%', boxSizing: 'border-box', resize: 'vertical',
            background: colors.bg, color: colors.text,
            border: `1px solid ${colors.border}`, borderRadius: 4,
            padding: 8, fontSize: 12, fontFamily: 'inherit', lineHeight: 1.5,
          }}
        />

        {screenshot && (
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '11px' }}>
            <input
              type="checkbox"
              checked={includeShot}
              onChange={(e) => setIncludeShot(e.target.checked)}
            />
            {t('settings:report.includeScreenshot')}
          </label>
        )}

        {fingerprint && (
          <div style={{ color: colors.textSecondary, fontSize: 11 }}>
            {t('settings:report.errorId')}{': '}
            <span style={{ fontFamily: 'monospace' }}>{fingerprint}</span>
          </div>
        )}

        <div style={{ color: colors.textSecondary, fontSize: 11, lineHeight: 1.6 }}>
          {t('settings:report.contents')}
        </div>

        {state === 'done' && (
          <div style={{ color: colors.green, fontSize: 11, lineHeight: 1.6 }}>
            {t('settings:report.done')}
            {canOpenIssue && <> {t('settings:report.doneIssueHint')}</>}
          </div>
        )}
        {state === 'failed' && (
          <div style={{ color: colors.red, fontSize: 11 }}>
            {t('settings:about.exportDiagnosticsFailed')}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4, flexWrap: 'wrap' }}>
          <button
            onClick={onClose}
            style={{
              padding: '7px 16px', background: 'transparent',
              border: `1px solid ${colors.border}`, borderRadius: 4,
              color: colors.textSecondary, fontSize: 12, cursor: 'pointer',
            }}
          >
            {state === 'done' ? t('settings:report.close') : t('settings:report.cancel')}
          </button>
          {state === 'done' && canOpenIssue && (
            <button
              onClick={handleOpenIssue}
              style={{
                padding: '7px 16px', background: 'transparent',
                border: `1px solid ${colors.accent}`, borderRadius: 4,
                color: colors.accent, fontWeight: 600, fontSize: 12, cursor: 'pointer',
              }}
            >
              {t('settings:report.openIssue')}
            </button>
          )}
          {state !== 'done' && (
            <button
              onClick={handleCreate}
              disabled={state === 'busy'}
              style={{
                padding: '7px 16px', background: colors.accent, border: 'none',
                borderRadius: 4, color: colors.textOnAccent, fontWeight: 600,
                fontSize: 12, cursor: state === 'busy' ? 'wait' : 'pointer',
              }}
            >
              {state === 'busy'
                ? t('settings:about.exportDiagnosticsBusy')
                : t('settings:report.create')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
