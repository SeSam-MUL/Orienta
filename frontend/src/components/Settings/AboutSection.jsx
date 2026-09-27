/**
 * About / legal notice section — the GPL §5d "Appropriate Legal Notices":
 * copyright, the ABSOLUTELY-NO-WARRANTY statement, the license grant, and
 * where to read the full license. Plus the app's version identity (git
 * commit based) and the diagnostics-bundle export for bug reports.
 */

import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing, GroupBox, Label } from '../../theme/components';
import { getAppVersion } from '../../services/api';
import DiagnosticsExportButton from '../common/DiagnosticsExportButton';
import UpdateDialog from '../common/UpdateDialog';
import useUpdateCheck, { isCheckEnabled, setCheckEnabled } from '../../hooks/useUpdateCheck';

const LICENSE_URL = 'https://www.gnu.org/licenses/gpl-3.0.html';
const REPO_URL = 'https://github.com/SeSam-MUL/Orienta';

function versionText(info, t) {
  if (!info || !info.version || info.version === 'unknown') {
    return t('settings:about.versionUnknown');
  }
  return info.branch ? `${info.version} · ${info.branch}` : info.version;
}

export default function AboutSection() {
  const { t } = useTranslation('settings');
  const [versionInfo, setVersionInfo] = useState(null);
  const [autoCheck, setAutoCheck] = useState(isCheckEnabled);
  // idle | busy | uptodate | <reason>. A manual check must SAY why it found
  // nothing — on start-up we stay silent, but here the user asked.
  const [checkState, setCheckState] = useState('idle');
  const { updateInfo, checkNow, close: closeUpdate, skip: skipUpdate } = useUpdateCheck();

  const handleCheckNow = async () => {
    setCheckState('busy');
    const result = await checkNow();
    if (result?.available) setCheckState('idle');
    else if (!result) setCheckState('check_failed');
    else setCheckState(result.reason || 'up_to_date');
  };

  const CHECK_MESSAGES = {
    up_to_date: t('settings:update.upToDate'),
    auth_required: t('settings:update.authRequired'),
    remote_unreachable: t('settings:update.unreachable'),
    not_a_git_install: t('settings:update.manualOnly'),
    no_local_release: t('settings:update.upToDate'),
    check_failed: t('settings:update.unreachable'),
    // An installed copy asks GitHub directly, so it can hear things a git
    // checkout never hears.
    rate_limited: t('settings:update.rateLimited'),
    unreachable: t('settings:update.unreachable'),
    not_found: t('settings:update.unreachable'),
    no_release_tag: t('settings:update.unreachable'),
  };

  // A reason nobody mapped used to render NOTHING: the user pressed "check for
  // updates" and the line stayed empty, which reads as "it is fine". Say the
  // raw reason rather than say nothing.
  const checkMessage = checkState === 'idle' || checkState === 'busy'
    ? ''
    : (CHECK_MESSAGES[checkState] || t('settings:update.checkIncomplete', { reason: checkState }));

  useEffect(() => {
    let mounted = true;
    getAppVersion()
      .then((info) => { if (mounted) setVersionInfo(info); })
      .catch(() => { if (mounted) setVersionInfo(null); });
    return () => { mounted = false; };
  }, []);

  return (
    <GroupBox title={t('settings:about.title')}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.innerSpacing,
        fontSize: '12px',
        color: colors.text,
      }}>
        <div>
          {t('settings:about.versionLabel')}{': '}
          <span style={{ fontFamily: 'monospace' }}>{versionText(versionInfo, t)}</span>
        </div>
        <div>{t('settings:about.copyright')}</div>
        <div>{t('settings:about.licenseLine')}</div>
        <div style={{ fontWeight: 600 }}>{t('settings:about.noWarranty')}</div>
        <Label secondary style={{ fontSize: '11px' }}>
          {t('settings:about.dataNote')}
        </Label>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <a
            href={LICENSE_URL}
            target="_blank"
            rel="noreferrer"
            style={{ color: colors.cyan, fontSize: '11px' }}
          >
            {t('settings:about.viewLicense')}
          </a>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            style={{ color: colors.cyan, fontSize: '11px' }}
          >
            {t('settings:about.sourceCode')}
          </a>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <button
            onClick={handleCheckNow}
            disabled={checkState === 'busy'}
            title={t('settings:hoverTips.checkForUpdates')}
            style={{
              padding: '6px 14px', background: colors.border, border: 'none',
              borderRadius: 4, color: colors.text, fontSize: '12px',
              cursor: checkState === 'busy' ? 'wait' : 'pointer',
            }}
          >
            {checkState === 'busy'
              ? t('settings:update.checking')
              : t('settings:update.checkNow')}
          </button>
          {checkMessage && (
            <span style={{
              color: checkState === 'up_to_date' || checkState === 'no_local_release'
                ? colors.textSecondary : colors.yellow,
              fontSize: '11px',
            }}>
              {checkMessage}
            </span>
          )}
          <DiagnosticsExportButton />
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '11px' }}>
          <input
            type="checkbox"
            checked={autoCheck}
            onChange={(e) => {
              const next = e.target.checked;
              setAutoCheck(next);
              setCheckEnabled(next);
            }}
          />
          {t('settings:update.checkOnStart')}
        </label>
        {updateInfo && (
          <UpdateDialog info={updateInfo} onClose={closeUpdate} onSkip={skipUpdate} />
        )}
        <Label secondary style={{ fontSize: '10px', opacity: 0.7 }}>
          {t('settings:about.exportDiagnosticsHint')}
        </Label>
        <Label secondary style={{ fontSize: '10px', opacity: 0.7 }}>
          {t('settings:about.authoritative')}
        </Label>
      </div>
    </GroupBox>
  );
}
