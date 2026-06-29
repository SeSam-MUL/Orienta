import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

/**
 * Fixed-position tooltip near the cursor showing the indexing result at the
 * hovered pixel: phase id+name (if any) plus a sorted list of scalar layers
 * (ci, bc, kam, gos, uncertainty, per-phase ci_<id>, ...). Mirror of the
 * EDS `HoverProbeOverlay` — pure presentational, all data from props.
 *
 * `visible=false` or `pos=null` → renders null. `probe=null` and
 * `visible=true` → "Loading…" placeholder. Errors render red.
 *
 * Scalar formatting:
 *   - `ci`, `uncertainty`: 3 decimals
 *   - `bc`: rounded integer
 *   - `kam`, `gos`: 2 decimals + "°"
 *   - everything else (incl. per-phase ci_<id>): 2 decimals
 */
export default function PhaseMapProbeOverlay({ probe, error, visible, pos }) {
  const { t } = useTranslation('phasemap');
  if (!visible || !pos) return null;

  const left = (pos.x ?? 0) + 16;
  const top = (pos.y ?? 0) + 16;

  const scalarEntries = probe?.scalars
    ? Object.entries(probe.scalars)
        .filter(([, v]) => v !== null && v !== undefined)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    : [];

  return (
    <div
      data-phasemap-probe-overlay
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
          {probe.phase?.name && (
            <Row k={t('phasemap:probe.phase')} v={probe.phase.name} accent />
          )}
          {scalarEntries.map(([k, v]) => (
            <Row key={k} k={k} v={fmtScalar(k, v)} />
          ))}
        </>
      ) : (
        <div style={{ color: colors.textSecondary, fontStyle: 'italic' }}>{t('phasemap:probe.loading')}</div>
      )}
    </div>
  );
}

function fmtScalar(key, value) {
  if (typeof value !== 'number' || !isFinite(value)) return '—';
  if (key === 'ci' || key === 'uncertainty') return value.toFixed(3);
  if (key === 'bc') return Math.round(value).toString();
  if (key === 'kam' || key === 'gos') return value.toFixed(2) + '°';
  return value.toFixed(2);
}

function Row({ k, v, accent }) {
  return (
    <div
      data-phasemap-probe-row
      data-key={k}
      style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}
    >
      <span style={{ color: colors.textSecondary }}>{k}</span>
      <span style={{ color: accent ? colors.green : colors.text, fontWeight: 600 }}>{v}</span>
    </div>
  );
}
