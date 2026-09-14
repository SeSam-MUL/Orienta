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

export default function QuantTable({ data }) {
  const { t } = useTranslation('eds');
  if (!data || data.length === 0) return null;
  const th = { padding: '5px 8px', textAlign: 'left', color: colors.textSecondary, fontWeight: 600, fontSize: '9pt', borderBottom: `1px solid ${colors.border}`, whiteSpace: 'nowrap' };
  const td = (i) => ({ padding: '4px 8px', fontSize: '9pt', borderBottom: `1px solid ${alpha(colors.border, 13)}`, background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20) });
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead><tr>{[t('table.element'), t('table.counts'), t('table.wtPct'), t('table.atPct')].map((h) => <th key={h} style={th}>{h}</th>)}</tr></thead>
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
