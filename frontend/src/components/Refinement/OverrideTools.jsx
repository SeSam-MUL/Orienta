import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import { Button } from '../../theme/components';
import useRefinementStore from '../../stores/useRefinementStore';

// Only the per-pixel ("Click") override mode is implemented end-to-end.
// Auto / Region / Threshold modes have no handler wired (they would just
// disable pixel-clicking with no alternative), so they are hidden for launch.
const MODES = [
  { id: 'click', labelKey: 'refinement:tools.modePixel', tipKey: 'refinement:tooltips.modePixel' },
];

// ---------------------------------------------------------------------------
// OverrideTools
// ---------------------------------------------------------------------------
export default function OverrideTools({ onExport }) {
  const { t } = useTranslation(['refinement', 'common']);
  const { mode, setMode, ciThreshold, setCiThreshold, overrideHistory } = useRefinementStore();

  const overrideCount = overrideHistory.reduce((sum, op) => sum + (op.pixels?.length ?? 1), 0);

  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'center',
      gap: 8,
      padding: `${spacing.innerSpacing}px ${spacing.innerMargin}px`,
      borderTop: `1px solid ${colors.border}`,
      background: colors.bgSecondary,
    }}>
      {/* Mode buttons */}
      <div style={{ display: 'flex', gap: 4 }}>
        {MODES.map((m) => (
          <button
            key={m.id}
            title={t(m.tipKey)}
            onClick={() => setMode(m.id)}
            style={{
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: '9pt',
              fontWeight: mode === m.id ? 700 : 400,
              cursor: 'pointer',
              border: `1px solid ${mode === m.id ? colors.accent : colors.border}`,
              background: mode === m.id ? `${colors.accent}22` : 'transparent',
              color: mode === m.id ? colors.accent : colors.textSecondary,
              transition: 'all 0.12s',
            }}
          >
            {t(m.labelKey)}
          </button>
        ))}
      </div>

      {/* CI threshold slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: '9pt', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
          {t('refinement:tools.ciLessThan')}
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={ciThreshold}
          onChange={(e) => setCiThreshold(parseFloat(e.target.value))}
          title={t('refinement:tooltips.ciThreshold')}
          style={{ width: 80, accentColor: colors.accent }}
        />
        <span style={{ fontSize: '9pt', fontFamily: 'monospace', color: colors.text, minWidth: 32 }}>
          {ciThreshold.toFixed(2)}
        </span>
      </div>

      {/* Override count */}
      {overrideCount > 0 && (
        <span style={{ fontSize: '9pt', color: colors.yellow }}>
          {t('refinement:tools.overrideCount', { count: overrideCount })}
        </span>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Export. Save is hidden for launch — overrides are persisted
          automatically after each assignment (handleSave is an empty stub,
          there is no dedicated save endpoint). */}
      <Button onClick={onExport} variant="default" small title={t('refinement:tooltips.export')}>
        {t('common:export')}
      </Button>
    </div>
  );
}
