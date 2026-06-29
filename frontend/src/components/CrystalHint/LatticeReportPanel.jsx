/**
 * Lattice Report panel — d-spacings + lattice param estimate + best system.
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
  twoCol: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 12,
  },
  stat: {
    fontSize: 22,
    fontWeight: 600,
    color: colors.accent,
  },
  statLabel: {
    fontSize: 10,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  dList: {
    fontSize: 11,
    color: colors.text,
    marginTop: 6,
  },
  hklChip: {
    display: 'inline-block',
    padding: '1px 6px',
    fontSize: 11,
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 3,
    marginRight: 4,
    marginBottom: 2,
    fontFamily: 'monospace',
  },
  warning: {
    fontSize: 11,
    color: colors.orange,
    background: `${colors.orange}11`,
    padding: '4px 8px',
    borderRadius: 4,
    marginTop: 6,
  },
};

const CONFIDENCE_COLORS = {
  high: colors.green,
  medium: colors.accent,
  low: colors.orange,
  none: colors.red,
};

export default function LatticeReportPanel({ lattice }) {
  const { t } = useTranslation('crystalhint');
  if (!lattice) return null;
  const {
    d_spacings_A = [],
    a_estimate_A,
    a_range_A,
    confidence,
    crystal_system_best,
    candidate_hkls = [],
    warnings = [],
  } = lattice;

  const confidenceColor = CONFIDENCE_COLORS[confidence] || colors.textSecondary;

  return (
    <div style={S.card}>
      <div style={{ ...S.title, display: 'flex', alignItems: 'center' }}>
        {t('lattice.title')}
        <InfoTooltip>
          <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
            {t('lattice.infoTitle')}
          </div>
          <p style={{ margin: '4px 0' }}>
            {t('lattice.infoIntro')}
          </p>
          <div style={{ fontWeight: 700, marginTop: 6, marginBottom: 2 }}>
            {t('lattice.infoConfidenceTitle')}
          </div>
          <ul style={{ margin: 0, paddingLeft: 16 }}>
            <li><b style={{ color: colors.green }}>{t('lattice.infoHighMediumPre')}</b>{t('lattice.infoHighMediumBody')}</li>
            <li><b style={{ color: colors.red }}>{t('lattice.infoNonePre')}</b>{t('lattice.infoNoneBody')}</li>
          </ul>
          <div style={{ fontWeight: 700, marginTop: 6, marginBottom: 2 }}>
            {t('lattice.infoWhatToDoTitle')}
          </div>
          <p style={{ margin: '4px 0' }}>
            <b>{t('lattice.infoWhatToDoBold')}</b>{t('lattice.infoWhatToDoBody')}
          </p>
          <p style={{ margin: '4px 0', fontSize: 10, color: colors.textSecondary }}>
            {t('lattice.infoRealMeasurement')}
          </p>
        </InfoTooltip>
      </div>
      <div style={S.twoCol}>
        <div>
          <div style={S.stat}>
            {a_estimate_A ? `${a_estimate_A.toFixed(2)} Å` : '—'}
          </div>
          <div style={S.statLabel}>{t('lattice.estimatedA')}</div>
          {a_range_A && (
            <div style={{ fontSize: 11, color: colors.textSecondary, marginTop: 4 }}>
              {t('lattice.range', { min: a_range_A[0].toFixed(2), max: a_range_A[1].toFixed(2) })}
            </div>
          )}
        </div>
        <div>
          <div style={{ fontSize: 11, color: colors.textSecondary }}>
            {t('lattice.confidence')}&nbsp;
            <span style={{ color: confidenceColor, fontWeight: 600 }}>
              {confidence}
            </span>
          </div>
          {crystal_system_best && (
            <div style={{ fontSize: 11, color: colors.text, marginTop: 4 }}>
              {t('lattice.bestFit')}<strong>{crystal_system_best}</strong>
            </div>
          )}
          {candidate_hkls.length > 0 && (
            <div style={{ marginTop: 6 }}>
              {candidate_hkls.map((hkl, i) => (
                <span key={i} style={S.hklChip}>
                  ({hkl.join('')})
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {d_spacings_A.length > 0 && (
        <div style={S.dList}>
          <strong>{t('lattice.dSpacings')}</strong>{' '}
          {d_spacings_A.map(d => Number(d).toFixed(2)).join(', ')}{t('lattice.dSpacingsUnit')}
        </div>
      )}

      {warnings.length > 0 && warnings.map((w, i) => (
        <div key={i} style={S.warning}>⚠ {w}</div>
      ))}
    </div>
  );
}
