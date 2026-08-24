/**
 * Orienta brand marks for in-app use — the wordmark and the bare IPF triangle.
 *
 * These are inline SVG rather than <img src="…"> on purpose: the ink colour has
 * to follow the active theme (five of them, not just light/dark), which a flat
 * file cannot do. The geometry below is the SAME as the shipped design files in
 * `branding/` — `Brand.test.jsx` reads those files and fails if the two drift,
 * so treat the constants as mirrored, not as a free copy.
 *
 * Geometry note: the triangle is the dot of a dotless i (U+0131) and its x is
 * measured from the rendered glyph — see branding/BRAND.md. Do not nudge it.
 */

import { useId } from 'react';
import { useTheme } from '../../theme/ThemeProvider';
import { getThemeColors } from '../../theme/tokens';

// --- geometry, mirrored from branding/wordmark-*.svg -------------------------
export const WORDMARK_VIEWBOX = '0 0 228 86';
export const WORDMARK_ASPECT = 228 / 86;
export const WORDMARK_TRIANGLE = 'M84.5 12 100.5 40 68.5 40Z';
export const WORDMARK_GRAD_BOX = { x: 60.5, y: 6, width: 48, height: 40 };
export const WORDMARK_FONT = 'Segoe UI,Inter,system-ui,sans-serif';

// --- geometry, mirrored from branding/icon.svg -------------------------------
export const ICON_VIEWBOX = '0 0 120 120';
export const ICON_TRIANGLE = 'M60 25 98 91 22 91Z';

/** IPF corner hues. The light set is darkened so it survives a white backdrop. */
const IPF = {
  dark:  { base: '#0f1830', red: '#ff3b3b', green: '#22dd66', blue: '#2f7bff', blend: 'screen' },
  light: { base: '#cfd6e0', red: '#e23b3b', green: '#159a4c', blue: '#2766cc', blend: 'multiply' },
};

/** Relative luminance of a #rrggbb string (0 = black, 1 = white). */
function luminance(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ''));
  if (!m) return 0;
  const n = parseInt(m[1], 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Which IPF treatment fits the current theme.
 * Derived from the theme's own background luminance rather than a hard-coded
 * list of theme names, so a theme added later is handled without touching this.
 */
export function useBrandVariant(override = 'auto') {
  const { theme } = useTheme();
  if (override === 'dark' || override === 'light') return override;
  return luminance(getThemeColors(theme)['--bg-primary']) > 0.5 ? 'light' : 'dark';
}

/** The three corner gradients + the base plate, clipped to `path`. */
function IpfFill({ ids, variant, path, box, gradient }) {
  const p = IPF[variant];
  const stops = (color) => (
    <>
      <stop offset="0" stopColor={color} />
      {gradient === 'icon' && <stop offset=".42" stopColor={color} stopOpacity=".18" />}
      <stop offset={gradient === 'icon' ? '.66' : '.6'} stopColor={color} stopOpacity="0" />
    </>
  );
  const r = gradient === 'icon' ? '105%' : '120%';
  const topY = gradient === 'icon' ? '4%' : '0%';
  const botY = gradient === 'icon' ? '98%' : '100%';
  return (
    <>
      <defs>
        <clipPath id={ids.clip}><path d={path} /></clipPath>
        <radialGradient id={ids.r} cx="50%" cy={topY} r={r}>{stops(p.red)}</radialGradient>
        <radialGradient id={ids.g} cx="92%" cy={botY} r={r}>{stops(p.green)}</radialGradient>
        <radialGradient id={ids.b} cx="8%" cy={botY} r={r}>{stops(p.blue)}</radialGradient>
      </defs>
      {/* isolate: keeps the screen/multiply blend inside this group instead of
          letting it mix with whatever the page paints behind the SVG. */}
      <g clipPath={`url(#${ids.clip})`} style={{ isolation: 'isolate' }}>
        <rect {...box} fill={p.base} />
        <rect {...box} fill={`url(#${ids.r})`} style={{ mixBlendMode: p.blend }} />
        <rect {...box} fill={`url(#${ids.g})`} style={{ mixBlendMode: p.blend }} />
        <rect {...box} fill={`url(#${ids.b})`} style={{ mixBlendMode: p.blend }} />
      </g>
    </>
  );
}

/**
 * The "orienta" wordmark. Ink follows the theme; the triangle keeps its fixed
 * IPF hues (they are the EBSD signature and must not be recoloured).
 *
 * @param {number} height   rendered height in px (width follows the aspect)
 * @param {string} variant  'auto' | 'dark' | 'light'
 * @param {string} ink      CSS colour for the letters
 */
export function Wordmark({
  height = 28,
  variant = 'auto',
  ink = 'var(--text-primary)',
  title = 'Orienta',
  style,
  ...rest
}) {
  const uid = useId().replace(/:/g, '');
  const ids = { clip: `wm-c-${uid}`, r: `wm-r-${uid}`, g: `wm-g-${uid}`, b: `wm-b-${uid}` };
  const v = useBrandVariant(variant);
  return (
    <svg
      viewBox={WORDMARK_VIEWBOX}
      height={height}
      width={Math.round(height * WORDMARK_ASPECT)}
      role="img"
      aria-label={title}
      style={{ display: 'block', ...style }}
      {...rest}
    >
      <title>{title}</title>
      <text
        x="20" y="74" fontFamily={WORDMARK_FONT} fontWeight="680"
        fontSize="58" letterSpacing="-1" fill={ink}
      >
        or&#x131;enta
      </text>
      <IpfFill ids={ids} variant={v} path={WORDMARK_TRIANGLE} box={WORDMARK_GRAD_BOX} gradient="wordmark" />
    </svg>
  );
}

/**
 * The bare IPF triangle — for tight spots (collapsed sidebar, headers) where
 * the full wordmark does not fit.
 *
 * @param {boolean} plate  draw the dark rounded square of the app icon behind it
 */
export function BrandMark({ size = 24, variant = 'auto', plate = false, title = 'Orienta', style, ...rest }) {
  const uid = useId().replace(/:/g, '');
  const ids = { clip: `bm-c-${uid}`, r: `bm-r-${uid}`, g: `bm-g-${uid}`, b: `bm-b-${uid}` };
  const v = useBrandVariant(variant);
  return (
    <svg
      viewBox={ICON_VIEWBOX} width={size} height={size}
      role="img" aria-label={title}
      style={{ display: 'block', ...style }}
      {...rest}
    >
      <title>{title}</title>
      {plate && <rect x="5" y="5" width="110" height="110" rx="27" fill="#0f141b" />}
      <IpfFill
        ids={ids} variant={v} path={ICON_TRIANGLE}
        box={{ x: 0, y: 0, width: 120, height: 120 }} gradient="icon"
      />
    </svg>
  );
}

export default Wordmark;
