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
          <DiagnosticsExportButton />
        </div>
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
