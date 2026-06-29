import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

/**
 * Fixed-position tooltip near the cursor showing multi-layer values at the
 * hovered pixel: BC, all EDS elements at the current displayMode, and phase
 * id+name. Pure presentational — all data comes from props.
 *
 * `visible=false` → renders null. `probe=null` and `visible=true` → shows
 * a "Loading…" placeholder. Errors are red.
 */
export default function HoverProbeOverlay({ probe, error, displayMode, visible, pos }) {
  const { t } = useTranslation('eds');
  if (!visible || !pos) return null;

  // Position offset so tooltip never lands under the cursor (which would
  // immediately trigger mouseleave on the overlay/tile underneath).
  const left = (pos.x ?? 0) + 16;
  const top = (pos.y ?? 0) + 16;

  return (
    <div
      data-hover-probe-overlay
      role="tooltip"
      style={{
        position: 'fixed', left, top, zIndex: 50,
        background: colors.bgSecondary,
        border: `1px solid ${error ? colors.red : colors.purple}`,
        borderRadius: 5,
        padding: '6px 10px',
        fontSize: '9pt', color: colors.text,
        boxShadow: '0 10px 24px rgba(0,0,0,.55)',
        pointerEvents: 'none',
        minWidth: 200, maxWidth: 320,
      }}
    >
      {error ? (
        <div style={{ color: colors.red }}>{error}</div>
      ) : probe ? (
        <>
          <div style={{ color: colors.cyan, fontWeight: 700, marginBottom: 4 }}>
            ({probe.row}, {probe.col})
          </div>
          {probe.bc != null && <Row k={t('probe.bc')} v={probe.bc.toFixed(0)} />}
          {Object.entries(probe.elements || {}).map(([el, v]) => (
            <Row key={el} k={el} v={fmtElement(v, displayMode, t)} />
          ))}
          {probe.phase?.name && <Row k={t('probe.phase')} v={probe.phase.name} accent />}
        </>
      ) : (
        <div style={{ color: colors.textSecondary, fontStyle: 'italic' }}>{t('probe.loading')}</div>
      )}
    </div>
  );
}

function fmtElement(v, mode, t) {
  const value = v?.[mode];
  if (typeof value !== 'number') return t('probe.empty');
  if (mode === 'counts') return Math.round(value).toLocaleString('en-US');
  return value.toFixed(1) + '%';
}

function Row({ k, v, accent }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
      <span style={{ color: colors.textSecondary }}>{k}</span>
      <span style={{ color: accent ? colors.green : colors.text, fontWeight: 600 }}>{v}</span>
    </div>
  );
}
