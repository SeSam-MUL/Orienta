/**
 * The per-pixel composition table, and the one mapping from the backend's
 * answer to its rows.
 *
 * Lifted out of EDSPage so both can be driven by a test with a real response
 * body. The mapping is the piece that matters: `/quantify/pixel` answers
 * `wt_pct: null` for a window the quantification could not price (see
 * eds_utils._resolve_k_factors), and the page used to coerce that with
 * `?? 0` — turning "we could not measure this" into a confident 0.00 %,
 * which is the one reading the backend change exists to prevent. The table
 * renders null as `t('table.empty')` and leaves it out of the sums.
 */
import { useTranslation } from 'react-i18next';

import { colors, alpha } from '../../theme/components';
import { compositionMethodKey } from './compositionMethod';
import { regionHeaderKeys } from './regionAverage';

/**
 * `[{element, counts, wt_pct, at_pct}]` from a /quantify/pixel response.
 *
 * Missing values stay null. Never 0: a zero is a measurement.
 */
export function quantRowsFromResponse(payload) {
  const raw = payload?.data || payload?.elements || {};
  if (Array.isArray(raw)) return raw;
  return Object.entries(raw).map(([element, vals]) => ({
    element,
    counts: vals?.counts ?? null,
    wt_pct: vals?.wt_pct ?? null,
    at_pct: vals?.at_pct ?? null,
  }));
}

/**
 * `source` is the quantification's own `source` ("spectrum" | "window").
 * On the spectrum source the counts are NET line counts from the spectrum
 * fit -- background subtracted, so a pixel without the element scatters
 * around 0 and can be negative -- and the column says so. Under a plain
 * "Counts" a -3 reads as a broken number.
 */
export default function QuantTable({ data, source = null }) {
  const { t } = useTranslation('eds');
  if (!data || data.length === 0) return null;
  const th = { padding: '5px 8px', textAlign: 'left', color: colors.textSecondary, fontWeight: 600, fontSize: '9pt', borderBottom: `1px solid ${colors.border}`, whiteSpace: 'nowrap' };
  const td = (i) => ({ padding: '4px 8px', fontSize: '9pt', borderBottom: `1px solid ${alpha(colors.border, 13)}`, background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20) });
  const net = source === 'spectrum';
  const heads = [
    [t('table.element')],
    net ? [t('table.netCounts'), t('table.netCountsTooltip')] : [t('table.counts')],
    [t('table.wtPct')],
    [t('table.atPct')],
  ];
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead><tr>{heads.map(([h, tip]) => <th key={h} style={th} title={tip}>{h}</th>)}</tr></thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={row.element || i} className="table-row-hover">
            <td style={{ ...td(i), color: colors.cyan, fontWeight: 600 }}>{row.element}</td>
            <td style={td(i)}>{typeof row.counts === 'number' ? Math.round(row.counts).toLocaleString('en-US') : row.counts ?? t('table.empty')}</td>
            <td style={td(i)}>{typeof row.wt_pct === 'number' ? row.wt_pct.toFixed(2) : row.wt_pct ?? t('table.empty')}</td>
            <td style={td(i)}>{typeof row.at_pct === 'number' ? row.at_pct.toFixed(2) : row.at_pct ?? t('table.empty')}</td>
          </tr>
        ))}
      </tbody>
      {(() => {
        const atSum = data.reduce((s, r) => s + (typeof r.at_pct === 'number' ? r.at_pct : 0), 0);
        const wtSum = data.reduce((s, r) => s + (typeof r.wt_pct === 'number' ? r.wt_pct : 0), 0);
        if (atSum === 0 && wtSum === 0) return null;
        const ok = Math.abs(atSum - 100) < 1;
        return (
          <tfoot>
            <tr>
              <td style={{ ...th, fontWeight: 700 }} title={t('table.sumTooltip')}>{'Σ'}</td>
              <td style={th}></td>
              <td style={{ ...th, fontWeight: 600 }}>{wtSum.toFixed(1)}%</td>
              <td style={{
                ...th, fontWeight: 700, color: ok ? colors.green : colors.orange,
                background: ok ? `${colors.green}11` : `${colors.orange}11`,
                borderRadius: 3,
              }}>
                {ok ? '✓ ' : '⚠ '}{atSum.toFixed(1)}%
              </td>
            </tr>
          </tfoot>
        );
      })()}
    </table>
  );
}

/**
 * The Region Average table: mean ± σ per element for a rectangle
 * (`/region-quantify`).
 *
 * Two things the header used to leave unsaid. On the spectrum source the
 * counts are NET counts (see QuantTable). And the "mean" of wt%/at% is not
 * always a mean: on the spectrum source it is the composition of the
 * region's SUMMED net counts, quantified once (`composition_method:
 * "summed_net_counts"`), while ± σ, min and max stay the pixels'. The line
 * under the summary says which.
 */
export function RegionQuantTable({ data }) {
  const { t } = useTranslation('eds');
  if (!data?.data || Object.keys(data.data).length === 0) return null;
  const th = { padding: '5px 8px', textAlign: 'left', color: colors.textSecondary, fontWeight: 600, fontSize: '8pt', borderBottom: `1px solid ${colors.border}`, whiteSpace: 'nowrap' };
  const td = (i) => ({ padding: '4px 8px', fontSize: '8pt', borderBottom: `1px solid ${alpha(colors.border, 13)}`, background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20) });
  const fmtStat = (s) => {
    if (!s) return t('table.empty');
    if (typeof s === 'number') return s.toFixed(2);
    return `${s.mean?.toFixed(1) ?? t('table.empty')} ± ${s.std?.toFixed(1) ?? t('table.empty')}`;
  };
  const entries = Object.entries(data.data);
  const net = data.quantification?.source === 'spectrum';
  const methodKey = compositionMethodKey(data.composition_method);
  const headerKeys = regionHeaderKeys(data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ fontSize: '8pt', color: colors.textSecondary, marginBottom: 4 }}>
        {t('region.summary', { rowStart: data.row_start, colStart: data.col_start, rowEnd: data.row_end, colEnd: data.col_end, nPixels: data.n_pixels })}
      </div>
      {methodKey && (
        <div
          data-composition-method={data.composition_method}
          style={{ fontSize: '7.5pt', color: colors.textSecondary, marginBottom: 4, lineHeight: 1.35 }}
        >
          {t(methodKey)}
        </div>
      )}
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {headerKeys.map((k, i) => [
              t(k), i === 1 && net ? t('table.netCountsTooltip') : undefined,
            ]).map(([h, tip]) => (
              <th key={h} style={th} title={tip}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map(([el, vals], i) => (
            <tr key={el}>
              <td style={{ ...td(i), color: colors.cyan, fontWeight: 600 }}>{el}</td>
              <td style={td(i)}>{fmtStat(vals.counts)}</td>
              <td style={td(i)}>{fmtStat(vals.wt_pct)}</td>
              <td style={td(i)}>{fmtStat(vals.at_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
