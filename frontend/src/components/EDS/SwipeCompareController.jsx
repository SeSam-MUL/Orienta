import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

/**
 * UI for picking Layer A and Layer B from the visible layers in the stack.
 * Reports the choice via onChange({ a, b }).
 *
 * `value` shape: { a: <layerId>|null, b: <layerId>|null }.
 * When BOTH are non-null, EDSPage will route OverlayCard into swipe mode.
 */
export default function SwipeCompareController({ layers, value, onChange }) {
  const { t } = useTranslation('eds');
  const visible = (layers || [])
    .filter((l) => l.visible)
    .map((l) => ({ id: l.id, label: l.label || l.id }));
  const a = value?.a || '';
  const b = value?.b || '';
  // A <select> sizes itself to its LONGEST option and, as a flex item,
  // refuses to shrink below that (min-width: auto). Two of them plus labels
  // outgrew the ~400 px overlay box and the row ran under the splitter into
  // the tile grid. Let both share the leftover width instead; the browser
  // truncates the shown option text itself.
  const selectStyle = {
    // Longhands, not `flex: '1 1 0'`: jsdom drops the shorthand, so the
    // test could not see it.
    flexGrow: 1, flexShrink: 1, flexBasis: 0, minWidth: 0,
    background: colors.bg, color: colors.text,
    border: `1px solid ${colors.border}`, padding: '2px 6px', borderRadius: 3,
  };
  return (
    <div data-swipe-controls style={{
      display: 'flex', alignItems: 'center', gap: 6, fontSize: '9pt',
      padding: '4px 6px', background: colors.bgSecondary,
      border: `1px solid ${colors.border}`, borderRadius: 4,
      minWidth: 0,
    }}>
      <label style={{ color: colors.textSecondary, flexShrink: 0 }}>{t('swipe.labelA')}</label>
      <select
        aria-label={t('swipe.selectA')}
        value={a}
        onChange={(e) => onChange?.({ a: e.target.value || null, b: value?.b || null })}
        style={selectStyle}>
        <option value="">{t('swipe.none')}</option>
        {visible.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
      </select>
      <label style={{ color: colors.textSecondary, flexShrink: 0 }}>{t('swipe.labelB')}</label>
      <select
        aria-label={t('swipe.selectB')}
        value={b}
        onChange={(e) => onChange?.({ a: value?.a || null, b: e.target.value || null })}
        style={selectStyle}>
        <option value="">{t('swipe.none')}</option>
        {visible.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
      </select>
      {a && b && (
        <button
          onClick={() => onChange?.({ a: null, b: null })}
          style={{
            background: 'transparent', border: `1px solid ${colors.cyan}`,
            color: colors.cyan, borderRadius: 3, padding: '2px 8px',
            fontSize: '8.5pt', cursor: 'pointer',
          }}
          title={t('swipe.exitTooltip')}
        >
          {t('swipe.exit')}
        </button>
      )}
    </div>
  );
}
