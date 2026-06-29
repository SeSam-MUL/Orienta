/**
 * MasterSelector — paste-a-path picker for an EMsoft master pattern .h5 file.
 *
 * Minimal stand-alone tool, no shared store. Parent owns `masterPath` and a
 * single setter. Hitting Enter triggers `onCommit` (used by the page to
 * normalise / validate later if we want it).
 */

import { useTranslation } from 'react-i18next';
import { colors, spacing, Button, Input, GroupBox, FormRow } from '../../theme/components';

export default function MasterSelector({ masterPath, onChange, onCommit, disabled = false }) {
  const { t } = useTranslation('dictionarygpu');
  return (
    <GroupBox title={t('master.groupTitle')}>
      <FormRow label={t('master.fieldLabel')}>
        <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
          <Input
            value={masterPath || ''}
            onChange={(e) => onChange(e.target.value)}
            placeholder={t('master.placeholder')}
            title={t('master.inputTooltip')}
            disabled={disabled}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && onCommit) onCommit();
            }}
            style={{ flex: 1, fontFamily: "'Courier New', monospace", fontSize: '9pt' }}
          />
          {onCommit && (
            <Button small onClick={onCommit} disabled={disabled || !masterPath} title={t('master.useTooltip')}>
              {t('master.use')}
            </Button>
          )}
        </div>
      </FormRow>
      <div style={{ fontSize: '9pt', color: colors.textSecondary, marginTop: 4 }}>
        {t('master.hint')}
      </div>
    </GroupBox>
  );
}
