/**
 * Simple horizontal-bar histogram for symmetry / lattice / system distributions.
 *
 * Data is a {key: count} dict. Bars are sorted by count desc (or by
 * `orderKeys` if supplied — useful for ordered bins like lattice ranges).
 */

import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

const S = {
  card: {
    padding: 12,
    background: colors.bgSecondary,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
  },
  title: {
    fontSize: 13, fontWeight: 600, color: colors.accent,
    marginBottom: 8,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '90px 1fr 60px',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
    fontSize: 11,
  },
  label: { color: colors.text, fontFamily: 'monospace' },
  barTrack: {
    height: 14, background: colors.bg, borderRadius: 3, overflow: 'hidden',
    border: `1px solid ${colors.border}`,
  },
  barFill: (frac) => ({
    height: '100%', width: `${frac * 100}%`,
    background: colors.accent, transition: 'width 0.2s',
  }),
  count: { color: colors.textSecondary, fontFamily: 'monospace', textAlign: 'right' },
  empty: {
    fontSize: 11,
    color: colors.textSecondary,
    fontStyle: 'italic',
    padding: '4px 0',
  },
};

export default function HistogramBars({ title, data, labelKey, total, orderKeys }) {
  const { t } = useTranslation('crystalhint');
  const entries = Object.entries(data || {});
  if (entries.length === 0) {
    return (
      <div style={S.card}>
        <div style={S.title}>{title}</div>
        <div style={S.empty}>{t('histogram.noData')}</div>
      </div>
    );
  }

  // Determine ordering
  let sorted;
  if (orderKeys) {
    sorted = orderKeys
      .filter(k => k in data)
      .map(k => [k, data[k]])
      .concat(entries.filter(([k]) => !orderKeys.includes(k)));
  } else {
    sorted = [...entries].sort((a, b) => b[1] - a[1]);
  }

  const maxCount = Math.max(...sorted.map(([, v]) => v), 1);
  const denom = total > 0 ? total : maxCount;

  return (
    <div style={S.card}>
      <div style={S.title}>{title}</div>
      {sorted.map(([key, count]) => {
        const label = labelKey ? labelKey(key) : key;
        const frac = count / maxCount;
        const pct = total > 0 ? (count / denom) * 100 : 0;
        return (
          <div key={key} style={S.row}>
            <span style={S.label}>{label}</span>
            <span style={S.barTrack}>
              <span style={S.barFill(frac)} />
            </span>
            <span style={S.count}>
              {count}
              {total > 0 && <span style={{ opacity: 0.6 }}>{t('histogram.percentSuffix', { pct: pct.toFixed(0) })}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}
