/**
 * HistogramPanel — SVG bar chart + drag-to-select range brush.
 */
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';

const W = 440;
const H = 110;
const PAD_L = 24;
const PAD_B = 18;
const PAD_T = 8;
const PAD_R = 8;

export default function HistogramPanel({ metricKey, summary, loading, range, onChange }) {
  const { t } = useTranslation('phasemap');
  // Hooks must run unconditionally — declare them before any early return.
  const svgRef = useRef(null);
  const [dragging, setDragging] = useState(null);

  // Distinguish the three falsy-summary states so the message is honest:
  //   loading              → request in flight
  //   summary === null     → backend 404 (diagnostics not computed for result)
  //   summary === undefined→ not yet requested / no active result
  if (loading) {
    return <div style={{ padding: 12, color: colors.textSecondary }}>{t('phasemap:anomalyBrowser.histLoading')}</div>;
  }
  if (summary === null) {
    return (
      <div style={{ padding: 12, color: colors.textSecondary, fontSize: 12 }}>
        {t('phasemap:anomalyBrowser.histNotComputed')}
      </div>
    );
  }
  if (!summary) {
    return <div style={{ padding: 12, color: colors.textSecondary }}>{t('phasemap:anomalyBrowser.histLoading')}</div>;
  }
  const m = summary[metricKey];
  if (!m || !m.histogram) return <div style={{ padding: 12 }}>{t('phasemap:anomalyBrowser.histNoData')}</div>;

  const { bins, counts } = m.histogram;
  const xMin = bins[0];
  const xMax = bins[bins.length - 1];
  const yMax = Math.max(1, ...counts);
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const xToPx = (x) => PAD_L + ((x - xMin) / (xMax - xMin)) * innerW;
  const pxToX = (px) => xMin + ((px - PAD_L) / innerW) * (xMax - xMin);
  const yToPx = (y) => PAD_T + innerH - (y / yMax) * innerH;

  const onMouseDown = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    const px = e.clientX - r.left;
    setDragging({ startPx: px });
    onChange?.([pxToX(px), pxToX(px)]);
  };
  const onMouseMove = (e) => {
    if (!dragging) return;
    const r = svgRef.current.getBoundingClientRect();
    const px = e.clientX - r.left;
    const lo = Math.min(dragging.startPx, px);
    const hi = Math.max(dragging.startPx, px);
    onChange?.([pxToX(lo), pxToX(hi)]);
  };
  const onMouseUp = () => setDragging(null);

  const selX1 = range ? xToPx(range[0]) : null;
  const selX2 = range ? xToPx(range[1]) : null;

  return (
    <div style={{ background: colors.bgSecondary, padding: 6 }}>
      <svg ref={svgRef} width={W} height={H}
           onMouseDown={onMouseDown}
           onMouseMove={onMouseMove}
           onMouseUp={onMouseUp}
           onMouseLeave={onMouseUp}
           style={{ cursor: 'crosshair', userSelect: 'none' }}>
        <line x1={PAD_L} y1={PAD_T + innerH} x2={PAD_L + innerW} y2={PAD_T + innerH} stroke={colors.border} />
        {counts.map((c, i) => {
          const x0 = xToPx(bins[i]);
          const x1 = xToPx(bins[i + 1]);
          return (
            <rect key={i}
                  x={x0} y={yToPx(c)}
                  width={Math.max(1, x1 - x0 - 1)}
                  height={(PAD_T + innerH) - yToPx(c)}
                  fill={colors.accent}
                  opacity={range ? 0.4 : 0.9} />
          );
        })}
        {selX1 !== null && (
          <rect x={selX1} y={PAD_T}
                width={Math.max(1, selX2 - selX1)} height={innerH}
                fill={colors.accent} opacity={0.25} />
        )}
        <text x={PAD_L} y={H - 4} fill={colors.textSecondary} fontSize={10}>{xMin.toFixed(2)}</text>
        <text x={PAD_L + innerW - 24} y={H - 4} fill={colors.textSecondary} fontSize={10}>{xMax.toFixed(2)}</text>
      </svg>
      <div style={{ fontSize: 11, color: colors.textSecondary }}>
        {range
          ? t('phasemap:anomalyBrowser.rangeSelected', { min: range[0].toFixed(3), max: range[1].toFixed(3) })
          : t('phasemap:anomalyBrowser.rangeDragHint')}
        <button onClick={() => onChange?.(null)} title={t('phasemap:hoverTips.histClearRange')} style={{
          background: 'none', border: 'none', color: colors.accent, cursor: 'pointer',
        }}>{t('phasemap:anomalyBrowser.rangeClear')}</button>
      </div>
    </div>
  );
}
