// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';
import InfoTooltip from './InfoTooltip';

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#111', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
  },
}));

afterEach(() => cleanup());

describe('InfoTooltip', () => {
  it('renders a `?` icon by default and no popup initially', () => {
    const { container, queryByRole } = render(
      <InfoTooltip>
        <p>Test content</p>
      </InfoTooltip>
    );
    expect(container.textContent).toContain('?');
    expect(queryByRole('tooltip')).toBeNull();
  });

  it('shows the popup on mouse-enter', () => {
    const { container, queryByRole } = render(
      <InfoTooltip>
        <p>Hover me explanation</p>
      </InfoTooltip>
    );
    const icon = container.querySelector('[role="button"]');
    expect(icon).toBeTruthy();
    fireEvent.mouseEnter(icon);
    const popup = queryByRole('tooltip');
    expect(popup).toBeTruthy();
    expect(popup.textContent).toMatch(/Hover me explanation/);
  });

  it('hides the popup on mouse-leave (after 150 ms grace)', async () => {
    const { container, queryByRole } = render(
      <InfoTooltip>
        <p>Disappearing content</p>
      </InfoTooltip>
    );
    const icon = container.querySelector('[role="button"]');
    fireEvent.mouseEnter(icon);
    expect(queryByRole('tooltip')).toBeTruthy();
    fireEvent.mouseLeave(icon);
    // The deferred-close timeout keeps the popup alive for 150 ms so the
    // user can mouse into a scrollable popup. Wait long enough.
    await new Promise(r => setTimeout(r, 200));
    expect(queryByRole('tooltip')).toBeNull();
  });

  it('cancels close when user mouses into the popup', async () => {
    const { container, queryByRole } = render(
      <InfoTooltip>
        <p>Keep me alive</p>
      </InfoTooltip>
    );
    const icon = container.querySelector('[role="button"]');
    fireEvent.mouseEnter(icon);
    const popup = queryByRole('tooltip');
    expect(popup).toBeTruthy();
    fireEvent.mouseLeave(icon);   // starts the 150 ms close timer
    fireEvent.mouseEnter(popup);  // cancels it
    await new Promise(r => setTimeout(r, 200));
    expect(queryByRole('tooltip')).toBeTruthy();  // still alive
  });

  it('supports inline mode (wraps an arbitrary trigger)', () => {
    const { container, queryByRole } = render(
      <InfoTooltip inline content={<p>Wrap target tooltip</p>}>
        <span data-testid="inner">some-text</span>
      </InfoTooltip>
    );
    // No `?` icon in inline mode
    expect(container.textContent).not.toContain('?');
    expect(container.textContent).toContain('some-text');
    // The wrapper span (with cursor:help) is the trigger
    const wrapper = container.querySelector('[style*="cursor: help"]');
    expect(wrapper).toBeTruthy();
    fireEvent.mouseEnter(wrapper);
    const popup = queryByRole('tooltip');
    expect(popup).toBeTruthy();
    expect(popup.textContent).toMatch(/Wrap target tooltip/);
  });

  it('responds to keyboard focus for accessibility', () => {
    const { container, queryByRole } = render(
      <InfoTooltip>
        <p>Accessibility content</p>
      </InfoTooltip>
    );
    const icon = container.querySelector('[role="button"]');
    expect(icon.getAttribute('tabIndex')).toBe('0');
    fireEvent.focus(icon);
    expect(queryByRole('tooltip')).toBeTruthy();
    fireEvent.blur(icon);
    expect(queryByRole('tooltip')).toBeNull();
  });
});
