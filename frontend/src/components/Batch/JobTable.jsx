import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

// ↺ (U+21BA) = circular arrow — "resumed / reused" is clearer than the
// em-dash we used to render for jobs whose result came from a pre-existing
// checkpoint rather than fresh indexing in the current run. Tooltip
// (title attr) on the status cell explains each symbol on hover.
const STATUS_TIP_KEY = {
  pending: 'jobTable.status.pending',
  running: 'jobTable.status.running',
  done:    'jobTable.status.done',
  failed:  'jobTable.status.failed',
  skipped: 'jobTable.status.skipped',
};
const STATUS_STYLE = {
  pending: { color: colors.textSecondary, label: '\u25CB' },
  running: { color: colors.cyan,          label: '\u23F3' },
  done:    { color: colors.green,         label: '\u2713' },
  failed:  { color: colors.red,           label: '\u2717' },
  skipped: { color: colors.yellow,        label: '\u2014' },
};

export default function JobTable({ jobs }) {
  const { t } = useTranslation('batch');
  return (
    <div style={{ maxHeight: 300, overflow: 'auto', fontSize: '9pt', fontFamily: 'monospace' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
            <th style={{ textAlign: 'left',   padding: '4px 8px' }}>{t('jobTable.colFile')}</th>
            <th style={{ textAlign: 'left',   padding: '4px 8px' }}>{t('jobTable.colPhase')}</th>
            <th style={{ textAlign: 'center', padding: '4px 8px' }}>{t('jobTable.colStatus')}</th>
            <th style={{ textAlign: 'right',  padding: '4px 8px' }}>{t('jobTable.colCi')}</th>
            <th style={{ textAlign: 'right',  padding: '4px 8px' }}>{t('jobTable.colTime')}</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((j, i) => {
            const s = STATUS_STYLE[j.status] || STATUS_STYLE.pending;
            return (
              <tr key={i} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                <td
                  style={{
                    padding: '3px 8px', color: colors.text,
                    maxWidth: 160, overflow: 'hidden',
                    textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}
                  title={j.file_name}
                >
                  {j.file_name}
                </td>
                <td style={{ padding: '3px 8px', color: colors.accent }}>{j.phase_name}</td>
                <td style={{ padding: '3px 8px', textAlign: 'center', color: s.color, fontWeight: 'bold' }}
                  title={STATUS_TIP_KEY[j.status] ? t(STATUS_TIP_KEY[j.status]) : ''}
                >
                  {j.status === 'skipped' ? '↺' : s.label}
                </td>
                <td style={{ padding: '3px 8px', textAlign: 'right', color: colors.text }}>
                  {j.ci_mean != null ? j.ci_mean.toFixed(3) : '\u2014'}
                </td>
                <td style={{ padding: '3px 8px', textAlign: 'right', color: colors.textSecondary }}>
                  {j.duration_sec != null ? `${j.duration_sec.toFixed(0)}s` : '\u2014'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
