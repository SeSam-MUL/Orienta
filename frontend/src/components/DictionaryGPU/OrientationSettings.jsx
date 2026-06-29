/**
 * OrientationSettings — resolution_deg + normalize toggle.
 *
 * `resolutionDeg` controls how dense the SO(3) sampling is. Smaller = more
 * patterns generated (quadratically). The slider mirrors the NumberInput.
 */

import { useTranslation } from 'react-i18next';
import { colors, spacing, NumberInput, GroupBox, FormRow } from '../../theme/components';

export default function OrientationSettings({ orientation, onChange, disabled = false }) {
  const { t } = useTranslation('dictionarygpu');
  const set = (key, val) => onChange({ ...orientation, [key]: val });

  // Coerce for the range slider only — keep raw string in state for the input.
  const sliderVal = Number(orientation.resolutionDeg);
  const sliderSafe = Number.isFinite(sliderVal) ? sliderVal : 5.0;

  return (
    <GroupBox title={t('orientation.groupTitle')}>
      <FormRow label={t('orientation.resolution')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing }}>
          <input
            type="range"
            min={1}
            max={15}
            step={0.5}
            value={sliderSafe}
            onChange={(e) => set('resolutionDeg', e.target.value)}
            disabled={disabled}
            title={t('orientation.sliderTooltip')}
            style={{ flex: 1, accentColor: colors.accent, cursor: disabled ? 'not-allowed' : 'pointer' }}
          />
          <NumberInput
            value={orientation.resolutionDeg}
            onChange={(e) => set('resolutionDeg', e.target.value)}
            min={0.5}
            max={30}
            step={0.5}
            disabled={disabled}
            title={t('orientation.numberTooltip')}
            style={{ width: 80 }}
          />
          <span style={{ fontSize: '9pt', color: colors.textSecondary, minWidth: 24 }}>{t('orientation.resolutionUnit')}</span>
        </div>
      </FormRow>

      <FormRow label={t('orientation.normalize')}>
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: '9pt',
            color: colors.text,
            cursor: disabled ? 'not-allowed' : 'pointer',
            userSelect: 'none',
          }}
          title={t('orientation.normalizeTooltip')}
        >
          <input
            type="checkbox"
            checked={!!orientation.normalize}
            onChange={(e) => set('normalize', e.target.checked)}
            disabled={disabled}
            style={{ accentColor: colors.accent }}
          />
          {t('orientation.normalizeCheckboxLabel')}
        </label>
      </FormRow>

      <div style={{ fontSize: '8pt', color: colors.textSecondary, opacity: 0.8 }}>
        {t('orientation.hint')}
      </div>
    </GroupBox>
  );
}
