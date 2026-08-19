/**
 * About / legal notice section — the GPL §5d "Appropriate Legal Notices":
 * copyright, the ABSOLUTELY-NO-WARRANTY statement, the license grant, and
 * where to read the full license.
 */

import { useTranslation } from 'react-i18next';
import { colors, spacing, GroupBox, Label } from '../../theme/components';

const LICENSE_URL = 'https://www.gnu.org/licenses/gpl-3.0.html';
const REPO_URL = 'https://github.com/SeSam-MUL/Orienta';

export default function AboutSection() {
  const { t } = useTranslation('settings');

  return (
    <GroupBox title={t('settings:about.title')}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.innerSpacing,
        fontSize: '12px',
        color: colors.text,
      }}>
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
        <Label secondary style={{ fontSize: '10px', opacity: 0.7 }}>
          {t('settings:about.authoritative')}
        </Label>
      </div>
    </GroupBox>
  );
}
