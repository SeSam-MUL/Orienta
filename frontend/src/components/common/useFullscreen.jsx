import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

/**
 * Put one preview into real fullscreen.
 *
 * The browser's Fullscreen API rather than a fixed-position overlay: Escape
 * leaves it without any handling of ours, and the page behind it is genuinely
 * gone rather than merely covered.
 *
 * Two things every caller needs to know:
 *  - Fullscreen changes the element's box but NOT the window, and Plotly's
 *    resize handling keys on window resize. A `resize` event is therefore
 *    dispatched after each transition, so plots re-lay out. A ResizeObserver
 *    (the crystal scene) reacts on its own and ignores the extra event.
 *  - The element must be told to fill the screen while active — a preview with
 *    a fixed pixel height would otherwise sit as a small box on black. Use the
 *    returned `active` for that.
 */
export function useFullscreen(ref) {
  const [active, setActive] = useState(false);

  useEffect(() => {
    const onChange = () => {
      const el = document.fullscreenElement || document.webkitFullscreenElement || null;
      setActive(!!el && !!ref.current && (el === ref.current || el.contains(ref.current)));
      // Let size-dependent renderers catch up, after the browser has settled
      // the new box.
      requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    };
    document.addEventListener('fullscreenchange', onChange);
    document.addEventListener('webkitfullscreenchange', onChange);
    return () => {
      document.removeEventListener('fullscreenchange', onChange);
      document.removeEventListener('webkitfullscreenchange', onChange);
    };
  }, [ref]);

  const toggle = useCallback(async () => {
    const el = ref.current;
    if (!el) return;
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        await (document.exitFullscreen?.() ?? document.webkitExitFullscreen?.());
      } else {
        await (el.requestFullscreen?.() ?? el.webkitRequestFullscreen?.());
      }
    } catch {
      // A denied request (no user gesture, iframe policy) leaves everything as
      // it was; the button simply does nothing rather than breaking the view.
    }
  }, [ref]);

  return { active, toggle };
}

/**
 * Corner brackets, in the 1000x1000 box Plotly's icons use. The same paths
 * serve the plain button, so the toggle looks identical whether it sits in a
 * Plotly modebar or on a bare canvas.
 */
export const FULLSCREEN_ICON_PATHS = {
  // Arrows pointing out — "make this big".
  enter: 'M0,420 L0,0 L420,0 L420,130 L130,130 L130,420 Z '
    + 'M580,0 L1000,0 L1000,420 L870,420 L870,130 L580,130 Z '
    + 'M1000,580 L1000,1000 L580,1000 L580,870 L870,870 L870,580 Z '
    + 'M420,1000 L0,1000 L0,580 L130,580 L130,870 L420,870 Z',
  // Arrows pointing in — "give me the page back".
  exit: 'M420,420 L420,60 L290,60 L290,290 L60,290 L60,420 Z '
    + 'M580,420 L580,60 L710,60 L710,290 L940,290 L940,420 Z '
    + 'M580,580 L580,940 L710,940 L710,710 L940,710 L940,580 Z '
    + 'M420,580 L420,940 L290,940 L290,710 L60,710 L60,580 Z',
};

/** The icon in the shape Plotly's `modeBarButtonsToAdd` expects. */
export function fullscreenIcon(active) {
  return {
    width: 1000,
    height: 1000,
    path: active ? FULLSCREEN_ICON_PATHS.exit : FULLSCREEN_ICON_PATHS.enter,
  };
}

/**
 * The toggle for viewers whose controls are not a Plotly modebar.
 *
 * The rule everywhere is the same: the toggle joins the row of controls the
 * viewer already has, in that row's own style. So there are two looks.
 *
 *  - `variant="plate"` matches a button row like the crystal viewer's
 *    "⟲ Reset · ⤓ PNG": plate, border, icon plus label.
 *  - `variant="flat"` is the bare icon, for a viewer with no button row at all.
 */
export function FullscreenButton({ active, onToggle, style, variant = 'flat' }) {
  const { t } = useTranslation(['common']);
  const [hover, setHover] = useState(false);
  const label = active ? t('common:fullscreen.exit') : t('common:fullscreen.enter');
  const plate = variant === 'plate';
  return (
    <button
      type="button"
      data-fullscreen-toggle
      data-fullscreen-variant={variant}
      onClick={(e) => { e.stopPropagation(); onToggle(); }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={label}
      aria-label={label}
      aria-pressed={active}
      style={plate ? {
        background: colors.bgTertiary,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        color: colors.text,
        cursor: 'pointer',
        fontSize: '8pt',
        padding: '2px 8px',
        // Match the row's height instead of guessing it. The neighbours here
        // are "⟲ Reset" and "⤓ PNG", whose glyphs give them a 1px taller line
        // box than plain Latin text — measured 19.00 vs 18.00. Stretching
        // follows whatever they are, so this cannot drift when their label or
        // font changes. Harmless outside a flex row.
        alignSelf: 'stretch',
        // Deliberately NOT a flex container and no line-height of our own, so
        // the text sets the line box exactly as it does in those buttons and
        // the smaller icon rides along inline.
        ...style,
      } : {
        background: 'transparent',
        border: 'none',
        padding: 3,
        display: 'flex',
        alignItems: 'center',
        cursor: 'pointer',
        color: hover ? colors.text : colors.textSecondary,
        opacity: hover ? 1 : 0.75,
        lineHeight: 0,
        ...style,
      }}
    >
      <svg
        viewBox="0 0 1000 1000"
        width={plate ? 10 : 16}
        height={plate ? 10 : 16}
        aria-hidden="true"
        focusable="false"
        style={plate ? { verticalAlign: -1, marginRight: 5 } : undefined}
      >
        <path d={active ? FULLSCREEN_ICON_PATHS.exit : FULLSCREEN_ICON_PATHS.enter} fill="currentColor" />
      </svg>
      {plate && label}
    </button>
  );
}

export default useFullscreen;
