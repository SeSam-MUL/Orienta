import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { useThresholdHistogram } from './hooks/useThresholdHistogram';

const SVG_W = 200;
const SVG_H = 50;

/**
 * Small histogram preview + two-handle range slider. Pure presentational:
 * the histogram comes from the hook, the threshold value + setter come from
 * the parent.
 *
 * threshold is `{min: number, max: number}` in 0..255 luminance space.
 */
export default function ThresholdHistogram({ bitmap, threshold, onThresholdChange }) {
  const { t } = useTranslation('eds');
  const data = useThresholdHistogram(bitmap);
  if (!data) return null;
  const { hist } = data;
  const maxBin = Math.max(...hist) || 1;
  const lo = threshold?.min ?? 0;
  const hi = threshold?.max ?? 255;
  return (
    <div data-threshold-block style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
      <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ background: colors.bg, borderRadius: 3 }}>
        {Array.from(hist).map((v, i) => {
          const x = i * (SVG_W / hist.length);
          const h = (v / maxBin) * SVG_H;
          return <rect key={i} x={x} y={SVG_H - h} width={(SVG_W / hist.length) - 0.5} height={h} fill={colors.purple} />;
        })}
        <rect
          x={lo / 255 * SVG_W}
          y={0}
          width={Math.max(0, (hi - lo) / 255 * SVG_W)}
          height={SVG_H}
          fill="rgba(139,233,253,.18)"
          stroke={colors.cyan}
          strokeWidth="0.5"
        />
      </svg>
      <div style={{ display: 'flex', gap: 4 }}>
        <input
          type="range" min={0} max={255} value={lo}
          aria-label={t('threshold.minAria')}
          onChange={(e) => onThresholdChange?.({ min: Number(e.target.value), max: hi })}
          style={{ flex: 1, accentColor: colors.cyan }}
        />
        <input
          type="range" min={0} max={255} value={hi}
          aria-label={t('threshold.maxAria')}
          onChange={(e) => onThresholdChange?.({ min: lo, max: Number(e.target.value) })}
          style={{ flex: 1, accentColor: colors.cyan }}
        />
      </div>
    </div>
  );
}
