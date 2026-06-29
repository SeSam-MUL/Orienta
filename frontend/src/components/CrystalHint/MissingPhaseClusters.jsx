/**
 * Missing-phase clusters panel — ranks signatures (system + lattice + fold)
 * that occur in many pixels but have NO local library match. These are
 * the prime candidates for "phases you should add to your SHT library".
 *
 * Each cluster gets a row showing the structural signature + pixel count
 * + preset-aware hint about which expected phases match the signature.
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
    fontSize: 13,
    fontWeight: 600,
    color: colors.accent,
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  item: {
    padding: 8,
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    fontSize: 11,
  },
  rank: { color: colors.textSecondary, fontFamily: 'monospace', marginRight: 6 },
  signature: { fontWeight: 600, color: colors.text },
  meta: { fontSize: 10, color: colors.textSecondary, marginTop: 2 },
  hint: {
    fontSize: 11,
    color: colors.accent,
    marginTop: 4,
    fontStyle: 'italic',
  },
  empty: {
    fontSize: 11,
    color: colors.textSecondary,
    fontStyle: 'italic',
    padding: '4px 0',
  },
};

function expectedPhaseHint(cluster, presetEntry) {
  if (!presetEntry?.expected_phases?.length) return null;
  // Find expected phases whose 'a' lies within ±20% of the cluster lattice
  const a = cluster.lattice_a_A;
  if (!a) return null;
  const matches = presetEntry.expected_phases.filter(p => {
    if (!p.a_A) return false;
    return Math.abs(p.a_A - a) / p.a_A < 0.20;
  });
  if (matches.length === 0) return null;
  return matches.map(p => p.name);
}

export default function MissingPhaseClusters({ clusters = [], presetEntry }) {
  const { t } = useTranslation('crystalhint');
  return (
    <div style={S.card}>
      <div style={S.title}>
        <span>{t('clusters.title')}</span>
        <span style={{ fontSize: 11, fontWeight: 400, color: colors.textSecondary }}>
          {t('clusters.count', { count: clusters.length })}
        </span>
      </div>
      {clusters.length === 0 && (
        <div style={S.empty}>
          {t('clusters.empty')}
        </div>
      )}
      <div style={S.list}>
        {clusters.map((c, idx) => {
          const hintNames = expectedPhaseHint(c, presetEntry);
          return (
            <div key={idx} style={S.item}>
              <span style={S.rank}>#{idx + 1}</span>
              <span style={S.signature}>
                {t('clusters.signature', {
                  system: c.crystal_system,
                  a: c.lattice_a_A?.toFixed?.(1) ?? '?',
                  nFold: c.n_fold,
                })}
              </span>
              <div style={S.meta}>
                {t('clusters.pixelCount', { count: c.pixel_count })}
                {c.fraction_of_analyzed > 0 && (
                  <>{t('clusters.fractionSuffix', { pct: Math.round(c.fraction_of_analyzed * 100) })}</>
                )}
              </div>
              {hintNames && hintNames.length > 0 && (
                <div style={S.hint}>
                  {t('clusters.matchHint', { count: hintNames.length, names: hintNames.join(', ') })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
