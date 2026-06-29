/**
 * Pure-React tooltip with a `?` icon trigger.
 *
 * Why custom (vs the HTML `title=""` attribute)?
 *   - HTML title has a ~1.5 s hover delay before showing
 *   - No styling control — looks bad against our dark theme
 *   - No HTML formatting (paragraphs, lists, bold)
 *   - Disappears after ~5 s in some browsers
 *
 * Usage:
 *   <InfoTooltip>
 *     <p>First paragraph</p>
 *     <p>Second paragraph with <b>bold</b></p>
 *   </InfoTooltip>
 *
 * Or with `inline` mode (no `?` icon, the children become the trigger):
 *   <InfoTooltip inline content={<p>Explanation</p>}>
 *     <span>Hover-me text</span>
 *   </InfoTooltip>
 */
import { useState, useRef, useCallback, useLayoutEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

const ICON_STYLE = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 14,
  height: 14,
  borderRadius: '50%',
  background: `${colors.accent}33`,
  color: colors.accent,
  border: `1px solid ${colors.accent}66`,
  fontSize: 10,
  fontWeight: 700,
  cursor: 'help',
  marginLeft: 4,
  verticalAlign: 'middle',
  userSelect: 'none',
};

const POPUP_STYLE = {
  position: 'fixed',  // fixed avoids overflow:hidden clipping in sidebar
  background: colors.bgSecondary,
  color: colors.text,
  border: `1px solid ${colors.accent}`,
  borderRadius: 6,
  padding: '10px 12px',
  fontSize: 11,
  lineHeight: 1.5,
  width: 320,
  // Cap height to fit the viewport with a 16 px margin; if content is
  // taller, the popup scrolls internally. Without this, tall tooltips
  // (CLAHE's 8-line list) ran off the bottom of the screen on smaller
  // windows and the user couldn't read the last bullet.
  maxHeight: 'calc(100vh - 32px)',
  overflowY: 'auto',
  boxShadow: '0 8px 24px rgba(0,0,0,0.7)',
  zIndex: 10000,
  // pointerEvents: 'auto' so the user can scroll within the popup. The
  // popup is dismissed on trigger-mouseleave anyway, so the cursor never
  // enters the popup naturally — only when the user explicitly moves into
  // it to scroll, which we now allow.
  pointerEvents: 'auto',
};

export default function InfoTooltip({ children, content, inline = false }) {
  const { t } = useTranslation('common');
  // `children` = popup body when inline=false (default). With inline=true,
  // `content` is the popup and `children` is the wrapped trigger.
  const [pos, setPos] = useState(null);
  const triggerRef = useRef(null);
  const popupRef = useRef(null);

  const positionPopup = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // Anchor below-right of the trigger; flip if it would overflow viewport.
    const x = r.right + 6;
    const y = r.top - 4;
    const viewportW = window.innerWidth;
    const viewportH = window.innerHeight;
    const popupW = 320;  // matches POPUP_STYLE.width
    const finalX = x + popupW > viewportW - 10
      ? Math.max(10, r.left - popupW - 6)  // flip left of trigger
      : x;
    // Initial Y guess — will be refined by useLayoutEffect once the popup
    // has actual rendered height.
    const finalY = Math.min(viewportH - 10, Math.max(10, y));
    setPos({ x: finalX, y: finalY, triggerTop: r.top, triggerBottom: r.bottom });
  }, []);

  // After the popup mounts, measure its actual height and re-position if
  // it would overflow the viewport bottom. Flip up so the bottom-of-popup
  // sits just above the trigger when there's more room above than below.
  useLayoutEffect(() => {
    if (!pos || !popupRef.current) return;
    const popupH = popupRef.current.offsetHeight;
    const viewportH = window.innerHeight;
    // Available space below the trigger vs above
    const spaceBelow = viewportH - pos.triggerBottom - 16;
    const spaceAbove = pos.triggerTop - 16;
    if (popupH > spaceBelow && spaceAbove > spaceBelow) {
      // Flip up: anchor bottom of popup to top of trigger - 4 px
      const newTop = Math.max(8, pos.triggerTop - 4 - popupH);
      if (newTop !== pos.y) {
        setPos({ ...pos, y: newTop });
      }
    } else if (pos.y + popupH > viewportH - 8) {
      // Still overflows below — clamp upward so bottom sits at viewport
      const newTop = Math.max(8, viewportH - 8 - popupH);
      if (newTop !== pos.y) {
        setPos({ ...pos, y: newTop });
      }
    }
  }, [pos]);

  // Deferred-close handler: keeps the popup alive while the cursor is on
  // either the trigger OR the popup, with a 150 ms grace period to let
  // the user move between them.
  const closeTimerRef = useRef(null);
  const cancelClose = useCallback(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);
  const scheduleClose = useCallback(() => {
    cancelClose();
    closeTimerRef.current = setTimeout(() => setPos(null), 150);
  }, [cancelClose]);

  const handleEnter = useCallback(() => {
    cancelClose();
    positionPopup();
  }, [positionPopup, cancelClose]);
  const handleLeave = scheduleClose;
  // Focus/blur for keyboard accessibility — close immediately on blur
  // since there's no "move to popup" path for keyboard users.
  const handleBlur = useCallback(() => setPos(null), []);

  const popupBody = inline ? content : children;
  const trigger = inline ? (
    <span
      ref={triggerRef}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      onFocus={handleEnter}
      onBlur={handleBlur}
      tabIndex={0}
      style={{ display: 'inline-block', cursor: 'help' }}
    >
      {children}
    </span>
  ) : (
    <span
      ref={triggerRef}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      onFocus={handleEnter}
      onBlur={handleBlur}
      tabIndex={0}
      role="button"
      aria-label={t('moreInfo')}
      style={ICON_STYLE}
    >
      ?
    </span>
  );

  return (
    <>
      {trigger}
      {pos && (
        <div
          ref={popupRef}
          style={{ ...POPUP_STYLE, left: pos.x, top: pos.y }}
          role="tooltip"
          onMouseEnter={cancelClose}
          onMouseLeave={scheduleClose}
        >
          {popupBody}
        </div>
      )}
    </>
  );
}
