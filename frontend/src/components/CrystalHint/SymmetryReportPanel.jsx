/**
 * Symmetry Report panel — shows detected n-fold + confidence + compatible systems.
 */

import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import InfoTooltip from '../common/InfoTooltip';

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
  },
  bigStat: {
    fontSize: 36,
    fontWeight: 700,
    color: colors.accent,
    lineHeight: 1.0,
  },
  bigStatLabel: {
    fontSize: 10,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  scores: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 6,
    marginTop: 8,
  },
  scoreBox: (highlighted) => ({
    padding: '6px 4px',
    textAlign: 'center',
    background: highlighted ? `${colors.accent}22` : colors.bg,
    border: `1px solid ${highlighted ? colors.accent : colors.border}`,
    borderRadius: 4,
  }),
  scoreFold: { fontSize: 14, fontWeight: 600, color: colors.text },
  scoreVal: { fontSize: 10, color: colors.textSecondary },
  systemChip: {
    display: 'inline-block',
    padding: '2px 6px',
    fontSize: 11,
    background: `${colors.accent}33`,
    color: colors.accent,
    border: `1px solid ${colors.accent}66`,
    borderRadius: 4,
    marginRight: 4,
  },
  warning: {
    fontSize: 11,
    color: colors.orange,
    background: `${colors.orange}11`,
    padding: '4px 8px',
    borderRadius: 4,
    marginTop: 6,
  },
  methodTag: {
    display: 'inline-block',
    padding: '2px 6px',
    fontSize: 10,
    fontFamily: 'monospace',
    background: colors.bg,
    color: colors.textSecondary,
    border: `1px solid ${colors.border}`,
    borderRadius: 3,
    marginLeft: 6,
  },
};

function _methodLabel(method, t) {
  if (!method) return '';
  if (method === 'image') return t('symmetry.methodAImage');
  if (method === 'spherical') return t('symmetry.methodBSpherical');
  // spherical_avg9 etc.
  if (method.startsWith('spherical_avg')) {
    const n = method.slice('spherical_avg'.length);
    return t('symmetry.methodBAvg', { n });
  }
  if (method.startsWith('image')) return t('symmetry.methodAImage');
  return method;
}

const CONFIDENCE_COLORS = {
  high: colors.green,
  medium: colors.accent,
  low: colors.orange,
  none: colors.red,
};

export default function SymmetryReportPanel({ symmetry, indexedPhase = null }) {
  const { t } = useTranslation('crystalhint');
  if (!symmetry) return null;
  const {
    detected_n_fold,
    n_fold_scores = {},
    confidence,
    compatible_systems = [],
    warnings = [],
    zone_axis_yx,
    method,
    n_patterns_averaged,
  } = symmetry;

  const confidenceColor = CONFIDENCE_COLORS[confidence] || colors.textSecondary;

  return (
    <div style={S.card}>
      <div style={S.title}>
        {t('symmetry.title')}
        {method && (
          <InfoTooltip inline content={(
            <>
              <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
                {_methodLabel(method, t)}
              </div>
              {method.startsWith('spherical') ? (
                <>
                  <p style={{ margin: '4px 0' }}>
                    {t('symmetry.methodSphericalBody')}
                  </p>
                  {n_patterns_averaged > 1 && (
                    <p style={{ margin: '4px 0' }}>
                      {t('symmetry.methodSphericalAveraged', { count: n_patterns_averaged })}
                    </p>
                  )}
                  <p style={{ margin: '6px 0 0 0', fontSize: 10,
                    color: colors.textSecondary }}>
                    {t('symmetry.methodSphericalThresholds')}
                  </p>
                </>
              ) : method.startsWith('image') ? (
                <>
                  <p style={{ margin: '4px 0' }}>
                    {t('symmetry.methodImageBody')}
                  </p>
                  <p style={{ margin: '6px 0 0 0', fontSize: 10,
                    color: colors.textSecondary }}>
                    {t('symmetry.methodImageFallback')}
                  </p>
                </>
              ) : null}
            </>
          )}>
            <span style={S.methodTag}>
              {_methodLabel(method, t)}
              {n_patterns_averaged > 1 && t('symmetry.methodTagSuffix', { n: n_patterns_averaged })}
            </span>
          </InfoTooltip>
        )}
      </div>
      <div style={S.row}>
        <div>
          <div style={S.bigStat}>
            {detected_n_fold ? t('symmetry.foldLabel', { n: detected_n_fold }) : '—'}
          </div>
          <div style={S.bigStatLabel}>{t('symmetry.detectedRotation')}</div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 4 }}>
            {t('symmetry.confidence')}&nbsp;
            <span style={{ color: confidenceColor, fontWeight: 600 }}>
              {confidence}
            </span>
          </div>
          {zone_axis_yx && (
            <div style={{ fontSize: 11, color: colors.textSecondary }}>
              {t('symmetry.zoneAxis', { y: zone_axis_yx[0], x: zone_axis_yx[1] })}
            </div>
          )}
          <div style={{ marginTop: 4 }}>
            {compatible_systems.length > 0 ? (
              compatible_systems.map(sys => (
                <span key={sys} style={S.systemChip}>{sys}</span>
              ))
            ) : (
              <span style={{ fontSize: 11, color: colors.textSecondary }}>
                {t('symmetry.noCompatibleSystem')}
              </span>
            )}
          </div>
        </div>
      </div>

      <div style={S.scores}>
        {Object.entries(n_fold_scores).map(([fold, score]) => {
          const numFold = parseInt(fold, 10);
          const highlighted = numFold === detected_n_fold;
          return (
            <div key={fold} style={S.scoreBox(highlighted)}>
              <div style={S.scoreFold}>{t('symmetry.foldLabel', { n: fold })}</div>
              <div style={S.scoreVal}>{t('symmetry.nccLabel', { value: Number(score).toFixed(3) })}</div>
            </div>
          );
        })}
      </div>

      {indexedPhase && (
        <div style={{
          marginTop: 8,
          padding: '6px 8px',
          background: colors.bg,
          border: `1px solid ${colors.border}`,
          borderRadius: 4,
          fontSize: 11,
        }}>
          <span style={{ color: colors.textSecondary }}>{t('symmetry.indexedPhase')}</span>{' '}
          {indexedPhase.is_unindexed ? (
            <span style={{ color: colors.textSecondary, fontStyle: 'italic' }}>
              {t('symmetry.unindexed')}
            </span>
          ) : (
            <>
              <span style={{ color: colors.text, fontWeight: 600 }}>
                {indexedPhase.phase_name || t('symmetry.phaseNumber', { id: indexedPhase.phase_id })}
              </span>{' '}
              <span style={{ color: colors.textSecondary }}>
                ({indexedPhase.crystal_system}
                {indexedPhase.space_group ? `, ${indexedPhase.space_group}` : ''})
              </span>
              {indexedPhase.consistent_with_symmetry === true && (
                <span style={{
                  marginLeft: 6,
                  color: colors.green,
                  fontWeight: 600,
                }}>{t('symmetry.consistent')}</span>
              )}
              {indexedPhase.consistent_with_symmetry === false && (
                <span style={{
                  marginLeft: 6,
                  color: colors.orange,
                  fontWeight: 600,
                }}>{t('symmetry.mismatch')}</span>
              )}
            </>
          )}
        </div>
      )}

      {warnings.length > 0 && warnings.map((w, i) => (
        <div key={i} style={S.warning}>⚠ {w}</div>
      ))}
    </div>
  );
}
