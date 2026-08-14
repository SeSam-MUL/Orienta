import { colors } from '../../theme/components';

/**
 * How much of a colour comes through — shown, not just numbered.
 *
 * A plain range input said "60 %" and nothing else: what 60 % of THIS colour
 * looks like could only be found out on the map. Here the track is the answer.
 * It runs from fully transparent on the left to the colour at full strength on
 * the right, over the checkerboard that means "you can see through this", so
 * the handle sits at the shade it will produce.
 *
 * The control underneath is a real <input type="range">, so arrow keys,
 * Home/End, click-to-jump and screen readers keep working.
 */

/** '#rgb' / '#rrggbb' → [r, g, b]; anything unreadable → black. */
export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(hex ?? '').trim());
  if (!m) return [0, 0, 0];
  const s = m[1].length === 3 ? m[1].split('').map((c) => c + c).join('') : m[1];
  const n = parseInt(s, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * The two background layers of the track: the colour ramp on top, the
 * checkerboard underneath.
 *
 * Exported so a test can assert the ramp really ends at the chosen colour
 * rather than at some fixed swatch.
 */
export function trackBackground(color) {
  const [r, g, b] = hexToRgb(color);
  const CHECK_A = 'rgba(255,255,255,0.16)';
  const CHECK_B = 'rgba(0,0,0,0.30)';
  return {
    backgroundImage: [
      `linear-gradient(to right, rgba(${r},${g},${b},0), rgba(${r},${g},${b},1))`,
      `conic-gradient(${CHECK_A} 0 25%, ${CHECK_B} 0 50%, ${CHECK_A} 0 75%, ${CHECK_B} 0)`,
    ].join(', '),
    backgroundSize: '100% 100%, 10px 10px',
    backgroundRepeat: 'no-repeat, repeat',
  };
}

export default function OpacitySlider({
  value, onChange, color = '#000000',
  step = 0.05, title, 'aria-label': ariaLabel, ...rest
}) {
  const v = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
  const pct = Math.round(v * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <input
        type="range"
        className="alpha-slider"
        min={0}
        max={1}
        step={step}
        value={v}
        onChange={(e) => onChange?.(Number(e.target.value))}
        title={title}
        aria-label={ariaLabel || title}
        aria-valuetext={`${pct}%`}
        style={trackBackground(color)}
        {...rest}
      />
      {/* Fixed width and tabular figures: the track must not jump sideways
          while dragging because the number went from 5 % to 100 %. */}
      <span style={{
        fontSize: '8pt', color: colors.textSecondary,
        width: 32, textAlign: 'right', flexShrink: 0,
        fontVariantNumeric: 'tabular-nums',
      }}>
        {pct}%
      </span>
    </div>
  );
}
