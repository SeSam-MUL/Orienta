/**
 * "Report a problem" — opens the report dialog, which bundles the user's
 * own description together with version, environment and log files.
 *
 * Shared between the Settings/About section and the crash screen, because
 * the moment a user is most willing to send a report is the moment the app
 * just broke in front of them.
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import ProblemReportDialog from './ProblemReportDialog';

export default function DiagnosticsExportButton({ variant = 'default', style }) {
  const { t } = useTranslation('settings');
  const [open, setOpen] = useState(false);

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, ...style }}>
      <button
        onClick={() => setOpen(true)}
        title={t('settings:hoverTips.exportDiagnostics')}
        style={{
          padding: variant === 'crash' ? '8px 20px' : '6px 14px',
          background: colors.border,
          border: 'none',
          borderRadius: 4,
          color: colors.text,
          fontSize: '12px',
          fontWeight: variant === 'crash' ? 500 : 400,
          cursor: 'pointer',
        }}
      >
        {t('settings:report.open')}
      </button>
      {open && <ProblemReportDialog onClose={() => setOpen(false)} />}
    </span>
  );
}
