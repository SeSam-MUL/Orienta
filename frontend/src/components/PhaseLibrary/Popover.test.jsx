// @vitest-environment jsdom
/**
 * The popover's behaviour, and an honest note about what jsdom cannot show.
 *
 * TESTED HERE: opens on click, takes the focus, Escape closes it AND returns
 * the focus to the trigger, an outside click closes it, the trigger says
 * whether it is open, the content is not built until it is.
 *
 * NOT TESTED HERE, and not pretended otherwise: flip and shift. jsdom gives
 * every element a zero-sized rect, so nothing ever collides with a viewport
 * edge and a test of "it flips above when there is no room below" would pass
 * with the middleware deleted. The configuration is asserted -- that the
 * middleware is present and padded -- and the behaviour is left for a real
 * browser. A test that cannot fail is worse than a missing one, because it
 * reads as coverage.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import Popover, { GAP_PX, EDGE_PADDING_PX, MIDDLEWARE_NAMES } from './Popover';

function harness({ onRender = () => {} } = {}) {
  return render(
    <div>
      <button type="button">before</button>
      <Popover trigger={<button type="button">open card</button>} label="Al">
        {({ close }) => {
          onRender();
          return (
            <div>
              <a href="https://doi.org/10.1063/1.4812323">the source</a>
              <button type="button" onClick={close}>close</button>
            </div>
          );
        }}
      </Popover>
    </div>,
  );
}

const trigger = () => screen.getByRole('button', { name: /open card/i });
const panel = () => screen.queryByTestId('phase-popover');

afterEach(cleanup);

describe('opening and closing', () => {
  it('is closed until it is asked for, and builds nothing meanwhile', () => {
    const onRender = vi.fn();
    harness({ onRender });
    expect(panel()).toBeNull();
    // The card's content is a function, so nothing is computed for a popover
    // nobody opened -- which matters when it grows a request.
    expect(onRender).not.toHaveBeenCalled();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
  });

  it('opens on click and says so on the trigger', async () => {
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(panel().getAttribute('role')).toBe('dialog');
    expect(panel().getAttribute('aria-label')).toBe('Al');
  });

  it('takes the focus when it opens, at the card and not inside it', async () => {
    // Measured before this assertion existed: the focus landed on the first
    // LINK in the card, so a screen reader started in the middle of a card
    // of facts. It also arrives a tick late, which is why this waits --
    // asserting immediately after the click found <body> and would have
    // been "fixed" by deleting the assertion.
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    await waitFor(() => expect(document.activeElement).toBe(panel()));
  });

  it('Escape closes it and hands the focus back', async () => {
    // The half that gets forgotten: without returnFocus the focus lands on
    // <body> and a keyboard user starts again from the top of the page.
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(panel()).toBeNull());
    expect(document.activeElement).toBe(trigger());
  });

  it('a click outside closes it', async () => {
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    fireEvent.mouseDown(screen.getByRole('button', { name: /before/i }));
    await waitFor(() => expect(panel()).toBeNull());
  });

  it('a click inside does not', async () => {
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    fireEvent.mouseDown(screen.getByRole('link', { name: /the source/i }));
    fireEvent.click(screen.getByRole('link', { name: /the source/i }));
    expect(panel()).toBeTruthy();
  });

  it('the card can close itself', async () => {
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    fireEvent.click(screen.getByRole('button', { name: /^close$/i }));
    await waitFor(() => expect(panel()).toBeNull());
  });

  it('holds a real link, which is why it is not a hover card', async () => {
    // Radix's own note: a hover card is "intended for sighted users only".
    harness();
    fireEvent.click(trigger());
    await waitFor(() => expect(panel()).toBeTruthy());
    const link = screen.getByRole('link', { name: /the source/i });
    expect(link.getAttribute('href')).toContain('doi.org');
    // Reachable by keyboard: it is inside the focus-managed panel.
    link.focus();
    expect(document.activeElement).toBe(link);
  });
});

describe('positioning — configured here, provable only in a browser', () => {
  it('keeps a gap from the trigger and a margin from the edge', () => {
    expect(GAP_PX).toBeGreaterThan(0);
    expect(EDGE_PADDING_PX).toBeGreaterThan(0);
  });

  it('names the middleware it relies on, so a deletion is visible', () => {
    // Weak on purpose, and labelled: jsdom has no layout, so flip and shift
    // cannot be exercised. This at least fails if someone removes the names
    // from the contract without meaning to.
    expect(MIDDLEWARE_NAMES).toEqual(['offset', 'flip', 'shift']);
  });
});
