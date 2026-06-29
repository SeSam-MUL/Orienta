/**
 * DetectorSettings — H, W, PC (x,y,z), sample tilt.
 *
 * Defaults (per Task 16 spec): 60x60, PC (0.5, 0.5, 0.5), tilt 70deg.
 * Values are floats kept as strings in state for friendly editing; parent is
 * responsible for parseFloat at submit time.
 */

import { useTranslation } from 'react-i18next';
import { spacing, NumberInput, GroupBox, FormRow, Label } from '../../theme/components';

export default function DetectorSettings({ detector, onChange, disabled = false }) {
  const { t } = useTranslation('dictionarygpu');
  const set = (key, val) => onChange({ ...detector, [key]: val });

  return (
    <GroupBox title={t('detector.groupTitle')}>
      <FormRow label={t('detector.patternSize')}>
        <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
          <NumberInput
            value={detector.H}
            onChange={(e) => set('H', e.target.value)}
            min={16}
            max={2048}
            step={1}
            disabled={disabled}
            title={t('detector.heightTooltip')}
            style={{ width: 90 }}
          />
          <Label secondary small>×</Label>
          <NumberInput
            value={detector.W}
            onChange={(e) => set('W', e.target.value)}
            min={16}
            max={2048}
            step={1}
            disabled={disabled}
            title={t('detector.widthTooltip')}
            style={{ width: 90 }}
          />
          <Label secondary small>{t('detector.patternSizeUnit')}</Label>
        </div>
      </FormRow>

      <FormRow label={t('detector.pcLabel')}>
        <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
          <NumberInput
            value={detector.pcx}
            onChange={(e) => set('pcx', e.target.value)}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
            title={t('detector.pcxTooltip')}
            style={{ width: 80 }}
          />
          <NumberInput
            value={detector.pcy}
            onChange={(e) => set('pcy', e.target.value)}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
            title={t('detector.pcyTooltip')}
            style={{ width: 80 }}
          />
          <NumberInput
            value={detector.pcz}
            onChange={(e) => set('pcz', e.target.value)}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
            title={t('detector.pczTooltip')}
            style={{ width: 80 }}
          />
        </div>
      </FormRow>

      <FormRow label={t('detector.sampleTilt')}>
        <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
          <NumberInput
            value={detector.sampleTilt}
            onChange={(e) => set('sampleTilt', e.target.value)}
            min={0}
            max={90}
            step={0.1}
            disabled={disabled}
            title={t('detector.sampleTiltTooltip')}
            style={{ width: 90 }}
          />
          <Label secondary small>{t('detector.sampleTiltUnit')}</Label>
        </div>
      </FormRow>
    </GroupBox>
  );
}
