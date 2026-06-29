/**
 * Material preset dropdown — pre-populates element list + provides
 * "expected phases" rank boost.
 */

import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

const S = {
  label: {
    fontSize: 11,
    color: colors.textSecondary,
    marginRight: 6,
  },
  select: {
    background: colors.bgSecondary,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    padding: '4px 10px',
    fontSize: 12,
    minWidth: 220,
    cursor: 'pointer',
  },
};

export default function MaterialPresetPicker({ presets, loading, value, onChange }) {
  const { t } = useTranslation('crystalhint');
  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <span style={S.label}>{t('preset.material')}</span>
      <select
        style={S.select}
        value={value}
        disabled={loading}
        onChange={(e) => onChange(e.target.value)}
        title={t('preset.tooltip')}
      >
        {loading && <option value="">{t('preset.loading')}</option>}
        {!loading && Object.entries(presets).map(([key, p]) => (
          <option key={key} value={key}>{p.label || key}</option>
        ))}
      </select>
    </div>
  );
}
