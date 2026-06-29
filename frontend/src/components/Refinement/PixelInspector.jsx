import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import { Button } from '../../theme/components';

// ---------------------------------------------------------------------------
// CI bar — horizontal bar showing confidence index visually
// ---------------------------------------------------------------------------
function CiBar({ value, max = 1 }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const barColor = value >= 0.4 ? colors.green : value >= 0.2 ? colors.yellow : colors.red;
  return (
    <div style={{
      height: 6, borderRadius: 3,
      background: `${colors.border}55`,
      flex: 1, overflow: 'hidden',
    }}>
      <div style={{
        height: '100%', width: `${pct}%`,
        background: barColor, borderRadius: 3,
        transition: 'width 0.2s',
      }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// PixelInspector
// ---------------------------------------------------------------------------
export default function PixelInspector({ selectedPixel, onAssignPhase }) {
  const { t } = useTranslation('refinement');
  if (!selectedPixel) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: 120, color: colors.textSecondary, fontSize: '10pt',
        textAlign: 'center', padding: spacing.innerMargin,
      }}>
        {t('refinement:inspector.prompt')}
      </div>
    );
  }

  const { row, col, phases } = selectedPixel;

  // Sort descending by CI
  const sorted = phases
    ? [...phases].sort((a, b) => (b.ci ?? 0) - (a.ci ?? 0))
    : [];

  const maxCi = sorted.length > 0 ? (sorted[0].ci ?? 1) : 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
      {/* Pixel coordinates */}
      <div style={{ fontSize: '9pt', color: colors.textSecondary }}>
        {t('refinement:inspector.pixelCoords', { row, col })}
      </div>

      {sorted.length === 0 ? (
        <div style={{ fontSize: '10pt', color: colors.textSecondary }}>{t('refinement:inspector.noPhaseData')}</div>
      ) : (
        sorted.map((p, i) => (
          <div
            key={i}
            style={{
              display: 'flex', flexDirection: 'column', gap: 4,
              padding: '6px 8px',
              background: i === 0 ? `${colors.accent}11` : colors.bgSecondary,
              border: `1px solid ${i === 0 ? colors.accent : colors.border}22`,
              borderRadius: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '10pt', fontWeight: i === 0 ? 700 : 400, color: colors.text }}>
                {p.name ?? t('refinement:inspector.phaseFallback', { id: p.phase_id ?? i })}
              </span>
              <span style={{ fontSize: '9pt', fontFamily: 'monospace', color: colors.textSecondary }}>
                {t('refinement:inspector.ciLabel', { value: (p.ci ?? 0).toFixed(3) })}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <CiBar value={p.ci ?? 0} max={maxCi} />
              <Button
                onClick={() => onAssignPhase?.({ row, col, phaseId: p.phase_id ?? i, name: p.name })}
                variant={i === 0 ? 'primary' : 'default'}
                small
                title={t('refinement:tooltips.assign')}
              >
                {t('refinement:inspector.assign')}
              </Button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
