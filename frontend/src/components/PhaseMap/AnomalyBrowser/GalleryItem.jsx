/**
 * GalleryItem — one row in the Anomaly Browser gallery.
 */
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';
import { forwardDiagApi } from '../../../services/api';

// A metric that is NaN for a pixel arrives as null (the route cannot send NaN):
// show a dash, not a crash.
const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : '–');

export default function GalleryItem({ item, resultId, visible, onClick, onHover, onLeave }) {
  const { t } = useTranslation('phasemap');
  const [thumbs, setThumbs] = useState(null);
  const [error, setError] = useState(null);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (!visible || fetchedRef.current) return;
    fetchedRef.current = true;
    forwardDiagApi.thumbnail(resultId, item.row, item.col, 64)
      .then(setThumbs)
      .catch((e) => setError(e.message || String(e)));
  }, [visible, resultId, item.row, item.col]);

  return (
    <div
      onClick={() => onClick?.(item.row, item.col)}
      onMouseEnter={() => onHover?.(item.row, item.col)}
      onMouseLeave={() => onLeave?.()}
      title={t('phasemap:hoverTips.anomalyItem')}
      style={{
        display: 'flex', gap: 8, alignItems: 'center',
        padding: 8, borderBottom: `1px solid ${colors.border}`,
        cursor: 'pointer', minHeight: 80,
      }}
    >
      <div style={{ width: 32, color: colors.textSecondary, fontVariantNumeric: 'tabular-nums' }}>
        #{item.rank}
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {thumbs && thumbs.experimental_b64 ? (
          <img src={`data:image/png;base64,${thumbs.experimental_b64}`} width={64} height={64} alt={t('phasemap:anomalyBrowser.itemExpAlt')} />
        ) : (
          <div style={{ width: 64, height: 64, background: colors.bgSecondary }} />
        )}
        {thumbs && thumbs.simulated_b64 ? (
          <img src={`data:image/png;base64,${thumbs.simulated_b64}`} width={64} height={64} alt={t('phasemap:anomalyBrowser.itemSimAlt')} />
        ) : (
          <div style={{ width: 64, height: 64, background: colors.bgSecondary }} />
        )}
      </div>
      <div style={{ flex: 1, fontSize: 12 }}>
        <div>{t('phasemap:anomalyBrowser.itemRowCol', { row: item.row, col: item.col })} <strong>{item.phase_name}</strong></div>
        <div>{t('phasemap:anomalyBrowser.itemNccPrefix')}<strong>{fmt(item.ncc)}</strong></div>
        <div>{t('phasemap:anomalyBrowser.itemMetrics', { anom: fmt(item.local_anomaly), pcs: fmt(item.pc_sensitivity), res: fmt(item.pattern_residual) })}</div>
        {error && <div style={{ color: colors.red, fontSize: 11 }}>{t('phasemap:anomalyBrowser.thumbnailError', { error })}</div>}
      </div>
    </div>
  );
}
