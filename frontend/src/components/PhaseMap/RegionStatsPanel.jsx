import { useTranslation } from 'react-i18next';
import { GroupBox, colors } from '../../theme/components';

/**
 * Right-rail summary for a rectangular pixel selection on the phase map.
 *
 * Renders nothing when there is no data, no loading state, and no error.
 * Otherwise shows:
 *   - a header "Region (r=r0..r1, c=c0..c1) · N px"
 *   - Phases table: name, count, percent — sorted by fraction desc, with
 *     id=-1 ("unindexed") forced to the bottom and rendered greyed out.
 *   - Scalars table: layer, "mean ± σ" — skips null entries.
 *
 * Backend shape (see `phasemap_region_stats`):
 *   {
 *     row_start, row_end, col_start, col_end, n_pixels,
 *     phases: [{ id, name, count, fraction }, ...],
 *     scalars: { <key>: {mean, std} | null }
 *   }
 */
export default function RegionStatsPanel({ data, error, loading }) {
  const { t } = useTranslation('phasemap');
  if (!data && !loading && !error) return null;

  return (
    <GroupBox title={t('phasemap:regionStats.title')}>
      {loading ? (
        <div style={{ color: colors.textSecondary, fontStyle: 'italic' }}>
          {t('phasemap:regionStats.computing')}
        </div>
      ) : error ? (
        <div
          data-region-stats-error
          style={{ color: colors.red }}
        >
          {error}
        </div>
      ) : (
        <RegionBody data={data} />
      )}
    </GroupBox>
  );
}

function RegionBody({ data }) {
  const { t } = useTranslation('phasemap');
  const phases = sortPhases(data.phases || []);
  const scalarEntries = Object.entries(data.scalars || {})
    .filter(([, v]) => v != null)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '9pt' }}>
      <div style={{ color: colors.textSecondary }}>
        {t('phasemap:regionStats.header', {
          r0: data.row_start, r1: data.row_end, c0: data.col_start, c1: data.col_end,
        })}
        {' · '}
        <span style={{ color: colors.text, fontWeight: 600 }}>{t('phasemap:regionStats.pixels', { n: data.n_pixels })}</span>
      </div>

      {phases.length > 0 && (
        <div data-region-phases-table>
          <div style={headerRow}>
            <span style={{ flex: 1 }}>{t('phasemap:regionStats.phase')}</span>
            <span style={{ width: 50, textAlign: 'right' }}>{t('phasemap:regionStats.count')}</span>
            <span style={{ width: 50, textAlign: 'right' }}>{t('phasemap:regionStats.percent')}</span>
          </div>
          {phases.map((p) => {
            const isUnindexed = p.id === -1;
            const rowColor = isUnindexed ? colors.textSecondary : colors.text;
            return (
              <div
                key={`phase-${p.id}`}
                data-region-phase-row
                data-name={p.name}
                style={{
                  display: 'flex', gap: 6, alignItems: 'center',
                  padding: '2px 0',
                  color: rowColor,
                  opacity: isUnindexed ? 0.7 : 1,
                }}
              >
                <span style={{ flex: 1, fontWeight: isUnindexed ? 400 : 600 }}>
                  {p.name}
                </span>
                <span style={{ width: 50, textAlign: 'right' }}>{p.count}</span>
                <span style={{ width: 50, textAlign: 'right' }}>
                  {`${(p.fraction * 100).toFixed(1)}%`}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {scalarEntries.length > 0 && (
        <div data-region-scalars-table>
          <div style={headerRow}>
            <span style={{ flex: 1 }}>{t('phasemap:regionStats.layer')}</span>
            <span style={{ textAlign: 'right' }}>{t('phasemap:regionStats.meanStd')}</span>
          </div>
          {scalarEntries.map(([key, v]) => (
            <div
              key={`scalar-${key}`}
              data-region-scalar-row
              data-key={key}
              style={{
                display: 'flex', gap: 6, alignItems: 'center',
                padding: '2px 0', color: colors.text,
              }}
            >
              <span style={{ flex: 1, color: colors.textSecondary }}>{key}</span>
              <span style={{ textAlign: 'right', fontWeight: 600 }}>
                {`${v.mean.toFixed(2)} ± ${v.std.toFixed(2)}`}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const headerRow = {
  display: 'flex', gap: 6, alignItems: 'center',
  padding: '2px 0',
  color: colors.textSecondary,
  fontSize: '8.5pt',
  fontWeight: 700,
  textTransform: 'uppercase',
  letterSpacing: 0.4,
  borderBottom: `1px solid ${colors.border}`,
};

/** Sort phases by `fraction` desc, but always push id=-1 ("unindexed") last. */
function sortPhases(rows) {
  const unindexed = rows.filter((r) => r.id === -1);
  const indexed = rows.filter((r) => r.id !== -1)
    .slice()
    .sort((a, b) => (b.fraction ?? 0) - (a.fraction ?? 0));
  return [...indexed, ...unindexed];
}
