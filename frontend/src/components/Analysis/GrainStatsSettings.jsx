/**
 * The settings of the grain-size statistics: which edge grains are left out, which basis the ECD is
 * measured on and, when `phaseOptions` is given, which phase the statistics are made of. Used by the Grain
 * Analysis tab and the Batch & Export tab.
 *
 * A native <select> cannot show a tooltip per option reliably, so the sentence that states the
 * rule of the CURRENT option is on the select's `title` and, for those who do not hover, under
 * the select as a small help line.
 */
import { useId } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { EDGE_RULES, ECD_BASES } from './grainStats';

const selectStyle = {
  width: '100%',
  background: colors.bg,
  border: `1px solid ${colors.border}`,
  borderRadius: 3,
  color: colors.text,
  fontSize: 11,
  padding: '3px 4px',
};

function SettingSelect({ label, value, options, optionLabel, optionTip, onChange, disabled }) {
  const id = useId();
  return (
    <div style={{ marginBottom: 8 }}>
      <label htmlFor={id} style={{ display: 'block', fontSize: 11, color: colors.textSecondary, marginBottom: 3 }}>
        {label}
      </label>
      <select
        id={id}
        aria-describedby={`${id}-help`}
        value={value}
        disabled={disabled}
        title={optionTip(value)}
        onChange={(e) => onChange(e.target.value)}
        style={{ ...selectStyle, cursor: disabled ? 'default' : 'pointer' }}
      >
        {options.map((o) => (
          <option key={o} value={o} title={optionTip(o)}>{optionLabel(o)}</option>
        ))}
      </select>
      <div id={`${id}-help`} style={{ fontSize: 10, color: colors.textSecondary, marginTop: 3, lineHeight: 1.35 }}>
        {optionTip(value)}
      </div>
    </div>
  );
}

const ALL_PHASES = 'all';
const MOST_MEASURED = 'most_measured';

/**
 * The sentence of a phase choice. Its second half follows the CURRENT edge rule: the MATLAB-script rule draws
 * its strip around the phase's own extent, MTEX's and the outermost-row rule look at the whole map, keep_all
 * has no edge rule. "All phases" and "Phase with the most measured pixels" each have their own.
 */
function phaseTip(key, edgeRule, t) {
  if (key === ALL_PHASES) return t('analysis:grain.statsPhaseTip.all');
  const family = edgeRule === 'matlab_script' ? 'matlab_script' : (edgeRule === 'keep_all' ? 'keep_all' : 'whole_map');
  const sentence = t(`analysis:grain.statsPhaseTip.${family}`);
  return key === MOST_MEASURED ? t('analysis:grain.statsPhaseTip.mostMeasured', { sentence }) : sentence;
}

/**
 * The phase choice as the select's string keys: 'all', one key per phase id, and 'most_measured' where it
 * is offered (the batch, which runs on other files) or already chosen. A phase id the list does not have
 * (the list is not there yet) is offered under its number so that the select never shows a wrong value.
 */
function phaseChoices(value, phaseOptions, allowMostMeasured, t) {
  const choices = [{ key: ALL_PHASES, label: t('analysis:grain.statsPhaseAll'), value: null }];
  for (const p of phaseOptions) choices.push({ key: String(p.id), label: p.name, value: p.id });
  if (allowMostMeasured || value === MOST_MEASURED) {
    choices.push({ key: MOST_MEASURED, label: t('analysis:grain.statsPhaseMostMeasured'), value: MOST_MEASURED });
  }
  if (Number.isInteger(value) && !choices.some((c) => c.value === value)) {
    choices.push({ key: String(value), label: t('analysis:grain.statsPhaseUnknown', { id: value }), value });
  }
  return choices;
}

/**
 * `settings` is the object `{ edgeRule, ecdBasis, statsPhase }`; `onChange(patch)` gets only the field that
 * changed. `phaseOptions` is `[{ id, name }]` (the phases of the loaded map); without it there is no phase
 * select. `allowMostMeasured` adds "Phase with the most measured pixels".
 */
export default function GrainStatsSettings({
  settings, onChange, disabled = false, phaseOptions = null, allowMostMeasured = false,
}) {
  const { t } = useTranslation('analysis');
  const phases = phaseOptions ? phaseChoices(settings.statsPhase ?? null, phaseOptions, allowMostMeasured, t) : null;
  const phaseKey = settings.statsPhase === null || settings.statsPhase === undefined ? ALL_PHASES : String(settings.statsPhase);
  return (
    <div role="group" aria-label={t('analysis:grain.statsSettingsTitle')} style={{ margin: '8px 0 4px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: colors.cyan, marginBottom: 5 }}>
        {t('analysis:grain.statsSettingsTitle')}
      </div>
      <div style={{ fontSize: 10, color: colors.textSecondary, marginBottom: 6, lineHeight: 1.35 }}>
        {t('analysis:grain.statsSettingsHelp')}
      </div>
      <SettingSelect
        label={t('analysis:grain.edgeRuleLabel')}
        value={settings.edgeRule}
        options={EDGE_RULES}
        optionLabel={(o) => t(`analysis:grain.edgeRule.${o}`)}
        optionTip={(o) => t(`analysis:grain.edgeRuleTip.${o}`)}
        onChange={(value) => onChange({ edgeRule: value })}
        disabled={disabled}
      />
      <SettingSelect
        label={t('analysis:grain.ecdBasisLabel')}
        value={settings.ecdBasis}
        options={ECD_BASES}
        optionLabel={(o) => t(`analysis:grain.ecdBasis.${o}`)}
        optionTip={(o) => t(`analysis:grain.ecdBasisTip.${o}`)}
        onChange={(value) => onChange({ ecdBasis: value })}
        disabled={disabled}
      />
      {phases && (
        <SettingSelect
          label={t('analysis:grain.statsPhaseLabel')}
          value={phaseKey}
          options={phases.map((c) => c.key)}
          optionLabel={(key) => phases.find((c) => c.key === key)?.label ?? key}
          optionTip={(key) => phaseTip(key, settings.edgeRule, t)}
          onChange={(key) => onChange({ statsPhase: phases.find((c) => c.key === key)?.value ?? null })}
          disabled={disabled}
        />
      )}
    </div>
  );
}
