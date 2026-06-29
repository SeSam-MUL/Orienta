/**
 * DegeneracyWarningBanner — non-blocking warning shown above the selected
 * phase list when crystallographically degenerate / duplicate phases are
 * selected. Hard clusters get a one-click "reduce to one"; soft clusters are
 * informational only.
 */

import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import { planReduction } from './phaseDegeneracy';

export default function DegeneracyWarningBanner({ degeneracy, onReducePhases }) {
  const { t } = useTranslation('indexing');
  const clusters = degeneracy?.clusters ?? [];
  if (clusters.length === 0) return null;

  return (
    <div role="alert" style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
      {clusters.map((cluster) => {
        const isHard = cluster.tier === 'hard';
        const accent = isHard ? C.red : C.yellow;
        return (
          <div
            key={cluster.members.join('-') + cluster.tier}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '5px 8px', borderRadius: 4,
              border: `1px solid ${accent}`,
              background: 'rgba(255,255,255,0.04)',
              borderLeft: `3px solid ${accent}`,
            }}
          >
            <span aria-hidden="true" style={{ flexShrink: 0 }}>{isHard ? '🔴' : '🟡'}</span>
            <span style={{ flex: 1, minWidth: 0, color: C.text, fontSize: '9pt' }}>
              {isHard
                ? t('degeneracy.redundant', { reason: cluster.reason })
                : t('degeneracy.softHint', { reason: cluster.reason })}
            </span>
            {isHard && (
              <button
                onClick={() => onReducePhases?.(planReduction(cluster))}
                title={t('hoverTips.degeneracyReduceToOne')}
                style={{
                  flexShrink: 0, padding: '3px 9px', fontSize: '8pt', fontWeight: 600,
                  background: accent, color: C.bg, border: 'none',
                  borderRadius: 3, cursor: 'pointer',
                }}
              >
                {t('degeneracy.reduceToOne')}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
