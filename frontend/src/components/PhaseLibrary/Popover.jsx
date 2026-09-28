/**
 * The popover the profile card lives in (spec §2.3, task 9).
 *
 * A POPOVER AND NOT A HOVER CARD, and the spec is unusually firm about it:
 * Radix's own documentation says a hover card is "intended for sighted users
 * only", and this card contains a link. Something you can only reach by
 * holding a mouse over it is not reachable at all for part of the audience,
 * and §2.9 exists because the mockups had no keyboard path.
 *
 * So: it opens on click, it takes the focus, Escape closes it and gives the
 * focus back to the button that opened it, and a click outside closes it
 * too. `@floating-ui/react` does the positioning -- `flip` when there is no
 * room below, `shift` to keep it on screen, `autoUpdate` so it follows a
 * scroll -- and the portal keeps it out of the overflow-hidden ancestors
 * that a facet column and a band list are full of.
 *
 * THREE THINGS HERE ARE THE LIBRARY'S DEFAULTS AND NOT OUR CODE, which
 * mutation testing established by deleting each one and watching nothing
 * fail: returning the focus to the trigger on close, `aria-expanded` on the
 * trigger (`useRole` sets it), and making the panel focusable. They were
 * written out as props at first, which reads as "we do this" and would have
 * let a reader believe the props were what held them up. The behaviour is
 * required and is pinned by the tests; the props are gone.
 *
 * WHAT THIS FILE'S TESTS CANNOT TELL YOU: jsdom has no layout, so every
 * element is 0x0 and nothing ever collides. Flip and shift are configured
 * here and verified by reading the configuration, which is not the same as
 * seeing them work; that measurement belongs in a real browser and is
 * written down as not done rather than implied.
 */
import { cloneElement, useState } from 'react';
import {
  useFloating, autoUpdate, offset, flip, shift,
  useClick, useDismiss, useRole, useInteractions,
  FloatingPortal, FloatingFocusManager,
} from '@floating-ui/react';
import { colors } from '../../theme/components';

/** Distance from the trigger, and the margin kept to the viewport edge. */
export const GAP_PX = 6;
export const EDGE_PADDING_PX = 8;

export const MIDDLEWARE_NAMES = ['offset', 'flip', 'shift'];

const S = {
  panel: {
    background: colors.bgSecondary,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    boxShadow: '0 6px 24px rgba(0,0,0,0.45)',
    padding: 12,
    maxWidth: 420,
    // A card with everything §2.3 asks for is taller than a short window.
    maxHeight: '70vh',
    overflowY: 'auto',
    zIndex: 60,
    fontSize: '9pt',
  },
};

/**
 * @param {object} o
 * @param {React.ReactElement} o.trigger  the control that opens it; it is
 *   cloned with the props floating-ui needs, so it must forward them
 * @param {string} o.label      accessible name for the dialog
 * @param {Function} o.children called with `{ close }` once open -- and only
 *   once open, so a card never fetches anything for a popover nobody opened
 */
export default function Popover({ trigger, label, children, placement = 'right-start' }) {
  const [open, setOpen] = useState(false);

  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: setOpen,
    placement,
    whileElementsMounted: autoUpdate,
    middleware: [
      offset(GAP_PX),
      flip({ padding: EDGE_PADDING_PX }),
      shift({ padding: EDGE_PADDING_PX }),
    ],
  });

  const { getReferenceProps, getFloatingProps } = useInteractions([
    useClick(context),
    // Escape and outside clicks. `outsidePressEvent: 'mousedown'` so a drag
    // that starts inside and ends outside does not count as "outside" --
    // selecting text in the card would otherwise close it.
    useDismiss(context, { outsidePressEvent: 'mousedown' }),
    useRole(context, { role: 'dialog' }),
  ]);

  return (
    <>
      {cloneElement(trigger, getReferenceProps({
        ref: refs.setReference,
        ...trigger.props,
      }))}
      {open && (
        <FloatingPortal>
          {/* `initialFocus` on the panel itself, not on its first tabbable.
              THIS one is not a default: without it the focus jumps to the
              first link INSIDE the card, so a screen reader starts in the
              middle of a card of facts. Mutation-tested -- removing it fails
              the focus test. */}
          <FloatingFocusManager
            context={context}
            modal={false}
            initialFocus={refs.floating}
          >
            <div
              ref={refs.setFloating}
              style={{ ...floatingStyles, ...S.panel }}
              aria-label={label}
              data-testid="phase-popover"
              {...getFloatingProps()}
            >
              {children({ close: () => setOpen(false) })}
            </div>
          </FloatingFocusManager>
        </FloatingPortal>
      )}
    </>
  );
}
