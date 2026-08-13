import { colors } from '../../theme/components';

/**
 * The colour bar for a layer whose colours mean a number.
 *
 * The range is the one the backend actually painted with, not a nominal one:
 * a confidence map stretched over 0.18..0.47 is drawn with the full colormap,
 * so a bar labelled 0..1 would misstate every colour on it.
 *
 * `stops` are sampled from the very colormap the map was rendered with, so the
 * bar cannot drift from the picture (no second implementation of viridis).
 */

export function formatScaleValue(v) {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(a >= 1 ? 2 : 3);
}

/** CSS gradient string for a set of colour stops, bottom = min. */
export function stopsToGradient(stops, direction = 'to top') {
  const list = Array.isArray(stops) && stops.length >= 2 ? stops : ['#000000', '#ffffff'];
  return `linear-gradient(${direction}, ${list.join(', ')})`;
}

/**
 * `height` accepts a number of pixels or 'fill' — the latter stretches the bar
 * over whatever room the parent gives it, which is what a bar beside a map
 * should do: the map is the height, the bar reads against it.
 */
export default function ScaleLegend({ label, scale, height = 96, width = 14 }) {
  if (!scale || !Number.isFinite(scale.min) || !Number.isFinite(scale.max)) return null;
  const mid = (scale.min + scale.max) / 2;
  const fill = height === 'fill';
  return (
    <div
      data-scale-legend={label || ''}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
        padding: '4px 2px',
        ...(fill ? { flex: 1, minHeight: 0 } : null),
      }}
    >
      <div style={{
        fontSize: '7.5pt', color: colors.text, maxWidth: 74,
        textAlign: 'center', lineHeight: 1.15,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>
        {label}
      </div>
      <div style={{
        display: 'flex', alignItems: 'stretch', gap: 4,
        ...(fill ? { flex: 1, minHeight: 0 } : { height }),
      }}>
        <div style={{
          width, borderRadius: 2, border: `1px solid ${colors.border}`,
          background: stopsToGradient(scale.stops),
        }} />
        {/* Three ticks: the two ends, which are the claim, and the middle as a
            reading aid. More would not fit beside a 96 px bar. */}
        <div style={{
          display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
          fontSize: '7pt', color: colors.textSecondary, fontVariantNumeric: 'tabular-nums',
        }}>
          <span>{formatScaleValue(scale.max)}</span>
          <span>{formatScaleValue(mid)}</span>
          <span>{formatScaleValue(scale.min)}</span>
        </div>
      </div>
      {scale.unit ? (
        <div style={{ fontSize: '7pt', color: colors.textSecondary }}>{scale.unit}</div>
      ) : null}
    </div>
  );
}
