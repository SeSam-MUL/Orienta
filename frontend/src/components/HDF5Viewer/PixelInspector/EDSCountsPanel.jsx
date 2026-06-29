import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { h5Api } from '../../../services/api';
import { colors, alpha, Select } from '../../../theme/components';

export default function EDSCountsPanel({ isFileOpen, currentRow, currentCol }) {
  const { t } = useTranslation('hdf5viewer');
  const [data, setData] = useState(null);
  const [unit, setUnit] = useState('counts'); // 'counts' | 'wt'
  const seq = useRef(0);

  useEffect(() => {
    if (!isFileOpen || currentRow == null || currentCol == null) { setData(null); return; }
    const s = ++seq.current;
    h5Api.getEDSPixel(currentRow, currentCol)
      .then((res) => { if (s === seq.current) setData(res.data); })
      .catch((err) => { if (s === seq.current) console.warn('[EDSCountsPanel]', err); });
  }, [isFileOpen, currentRow, currentCol]);

  if (!isFileOpen) return <div style={{ color: colors.textSecondary }}>{t('counts.noFile')}</div>;
  if (!data) return <div style={{ color: colors.textSecondary }}>{t('counts.loading')}</div>;

  const counts = data.counts || {};
  const total = Object.values(counts).reduce((s, v) => s + v, 0) || 1;
  // wt% is a simplified counts-normalization until FEAT-10 K-factors land.
  // At% requires atomic-mass weighting and is hidden for now.
  const rows = Object.entries(counts).map(([el, c]) => ({
    el,
    counts: c,
    wt_pct: (c / total) * 100,
  }));

  if (rows.length === 0) {
    return <div style={{ color: colors.textSecondary, fontSize: '8pt' }}>
      {t('counts.noData')}
    </div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span title={t('counts.unitTooltip')} style={{ display: 'inline-flex' }}>
          <Select value={unit} onChange={(e) => setUnit(e.target.value)}
            options={[
              { value: 'counts', label: t('counts.unitCounts') },
              { value: 'wt', label: t('counts.unitWt') },
            ]} style={{ width: 100 }} />
        </span>
        <span style={{ fontSize: '8pt', color: colors.textSecondary }}>{t('counts.total', { value: total.toFixed(0) })}</span>
      </div>
      {rows.map((r) => (
        <div key={r.el} style={{
          display: 'flex', justifyContent: 'space-between',
          padding: '3px 0', fontSize: '9pt',
          borderBottom: `1px solid ${alpha(colors.border, 19)}`,
        }}>
          <span style={{ color: colors.cyan, fontFamily: 'monospace' }}>{r.el}</span>
          <span style={{ color: colors.text, fontFamily: 'monospace' }}>
            {unit === 'counts' ? r.counts.toFixed(0) : `${r.wt_pct.toFixed(2)}%`}
          </span>
        </div>
      ))}
    </div>
  );
}
