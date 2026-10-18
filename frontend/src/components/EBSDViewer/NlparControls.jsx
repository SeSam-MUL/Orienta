/**
 * NLPAR controls for the Pattern Processing box, and the Cancel button that
 * has to live in the busy overlay (the overlay covers the whole page, so a
 * button in the panel could not be clicked while the run is going).
 *
 * No state here: everything comes from `useNlpar`.
 */
import { useTranslation } from 'react-i18next';
import InfoTooltip from '../common/InfoTooltip';
import {
  colors, spacing, Button, NumberInput, FormRow, Checkbox,
} from '../../theme/components';
import { MIN_RADIUS, MAX_RADIUS } from './useNlpar';

const STAGES = ['compile', 'max', 'sigma', 'lambda', 'filter'];

/** What the advice says, and in which colour. `null` while it is being worked out. */
export function adviceView(advice, t) {
  if (!advice) return { key: 'checking', color: colors.textSecondary, text: t('nlpar.advice.checking') };
  switch (advice.recommendation) {
    case 'applied':
      return { key: 'applied', color: colors.textSecondary,
        text: t('nlpar.advice.applied', { radius: advice.applied_radius ?? '?' }) };
    case 'recommended':
    case 'not_recommended': {
      const key = advice.recommendation === 'recommended' ? 'recommended' : 'notRecommended';
      return { key, color: key === 'recommended' ? colors.green : colors.orange,
        text: t(`nlpar.advice.${key}`, {
          lam: Number(advice.lam).toFixed(2), sigma: Number(advice.sigma).toFixed(1),
          threshold: Number(advice.threshold).toFixed(1),
        }) };
    }
    default: {
      const reasons = { too_small: 'unavailableTooSmall', hex: 'unavailableHex', uniform: 'unavailableUniform' };
      return { key: 'unavailable', color: colors.textSecondary,
        text: t(`nlpar.advice.${reasons[advice.reason] || 'unavailable'}`) };
    }
  }
}

export function NlparControls({ nlpar, disabled, onRun }) {
  const { t } = useTranslation('ebsdviewer');
  const view = adviceView(nlpar.advice, t);
  return (
    <>
      <FormRow label={(
        <span style={{ display: 'inline-flex', alignItems: 'center' }}>
          {t('nlpar.radiusLabel')}
          <InfoTooltip>
            <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
              {t('nlpar.tooltipTitle')}
            </div>
            <p style={{ margin: '4px 0' }}>{t('nlpar.tooltipP1')}</p>
            <p style={{ margin: '4px 0' }}>{t('nlpar.tooltipP2')}</p>
            <p style={{ margin: '4px 0' }}>{t('nlpar.tooltipP3')}</p>
            <p style={{ margin: '4px 0', fontSize: 10, color: colors.textSecondary }}>
              {t('nlpar.tooltipP4')}
            </p>
          </InfoTooltip>
        </span>
      )}>
        <div style={{ display: 'flex', gap: spacing.innerSpacing, alignItems: 'center' }}>
          <NumberInput
            value={nlpar.radius}
            onChange={(e) => nlpar.setRadius(e.target.value)}
            min={MIN_RADIUS}
            max={MAX_RADIUS}
            title={t('nlpar.radiusTooltip')}
            aria-label={t('nlpar.radiusLabel')}
            style={{ width: 60 }}
          />
          <Button
            onClick={onRun}
            disabled={disabled || !nlpar.canRun}
            title={nlpar.applied ? t('nlpar.appliedTooltip') : t('nlpar.buttonTooltip')}
            small
            style={{ flex: 1 }}
          >
            {t('nlpar.button')}
          </Button>
        </div>
      </FormRow>
      <div
        data-testid="nlpar-advice"
        data-advice={view.key}
        title={t('nlpar.advice.basis')}
        style={{ fontSize: '9pt', color: view.color, margin: `0 0 ${spacing.innerSpacing}px`, lineHeight: 1.35 }}
      >
        {view.text}
      </div>
      <div style={{ marginBottom: spacing.innerSpacing }}>
        <Checkbox
          checked={nlpar.denoiseFirst}
          onChange={(e) => nlpar.setDenoiseFirst(e.target.checked)}
          disabled={disabled || !nlpar.canRun}
          label={t('nlpar.pipelineFirst')}
          title={t('nlpar.pipelineFirstTooltip')}
        />
      </div>
    </>
  );
}

/** Stage line and Cancel button, drawn inside the busy overlay while NLPAR runs. */
export function NlparOverlayExtras({ nlpar, phase }) {
  const { t } = useTranslation('ebsdviewer');
  if (!nlpar.running) return null;
  const stage = STAGES.includes(phase) ? phase : 'compile';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <div style={{ fontSize: '9pt', color: colors.textSecondary, textAlign: 'center', maxWidth: 360 }}>
        {t(`nlpar.stage.${stage}`)}
      </div>
      <Button
        onClick={nlpar.cancel}
        disabled={nlpar.cancelling}
        small
      >
        {nlpar.cancelling ? t('nlpar.cancelling') : t('nlpar.cancel')}
      </Button>
    </div>
  );
}
