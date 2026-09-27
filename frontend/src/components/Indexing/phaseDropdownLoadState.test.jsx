// @vitest-environment jsdom
//
// What the dropdown is allowed to SAY. "No phases found" is a statement about
// the user's library; before this it was also what appeared when the request
// had failed or had not come back — the same shape as the About box blaming
// git while the version endpoint was answering.
import React from 'react';
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import PhaseDropdown from './PhaseDropdown';

const FILES = [
  { path: 'D:/lib/CIF_Library/Al.cif', filename: 'Al.cif', formula: 'Al', element_group: 'Al-Fe-Si' },
];

function mount(props = {}) {
  return render(
    <PhaseDropdown
      discoveredFiles={props.discoveredFiles ?? []}
      groups={['Al-Fe-Si']}
      selectedPaths={[]}
      onTogglePath={() => {}}
      onSetAll={() => {}}
      method="hough"
      open
      onClose={() => {}}
      {...props}
    />
  );
}

afterEach(() => cleanup());

describe('what the phase dropdown says while the list is not there', () => {
  it('says it is loading, and does NOT claim the library is empty', () => {
    mount({ loadState: 'loading' });
    expect(screen.getByText('Loading phases…')).toBeTruthy();
    expect(screen.queryByText('No phases found')).toBeNull();
  });

  it('treats "never asked" like "asking" — never like "nothing there"', () => {
    mount({ loadState: 'idle' });
    expect(screen.getByText('Loading phases…')).toBeTruthy();
    expect(screen.queryByText('No phases found')).toBeNull();
  });

  it('names the backend, not the library, when the request failed', () => {
    mount({ loadState: 'error' });
    expect(screen.getByText(/Could not reach the phase library/)).toBeTruthy();
    expect(screen.queryByText('No phases found')).toBeNull();
  });

  it('offers a way to ask again, and calls it', () => {
    const onRetry = vi.fn();
    mount({ loadState: 'error', onRetry });
    fireEvent.click(screen.getByText('Try again now'));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  // The other half: when the backend really answers with nothing, the old
  // sentence is true and must survive.
  it('still says "No phases found" when the library answered and is empty', () => {
    mount({ loadState: 'loaded', discoveredFiles: [] });
    expect(screen.getByText('No phases found')).toBeTruthy();
  });

  it('"No matches" is also a claim about data — it waits for the list too', () => {
    const { container } = mount({ loadState: 'loading' });
    const search = container.querySelector('input');
    fireEvent.change(search, { target: { value: 'zzz' } });
    expect(screen.getByText('Loading phases…')).toBeTruthy();
    expect(screen.queryByText('No matches')).toBeNull();
  });

  it('but says "No matches" once the list is really there', () => {
    const { container } = mount({ loadState: 'loaded', discoveredFiles: FILES });
    fireEvent.change(container.querySelector('input'), { target: { value: 'zzz' } });
    expect(screen.getByText('No matches')).toBeTruthy();
  });

  // Every other caller and every existing test passes no loadState at all.
  it('without the prop, behaves exactly as before', () => {
    mount({ discoveredFiles: [] });
    expect(screen.getByText('No phases found')).toBeTruthy();
  });

  // A failed refresh keeps the old list on screen, which means the
  // empty-state block never renders and the failure was invisible — while
  // Start went grey with no reason. A review found this; the banner is the
  // only place that says it when there is still something to see.
  it('warns that a shown list may be stale after a failed refresh', () => {
    const onRetry = vi.fn();
    mount({ loadState: 'error', discoveredFiles: FILES, onRetry });
    expect(screen.getByText(/This list may be out of date/)).toBeTruthy();
    expect(screen.getByText(/Al\.cif/)).toBeTruthy();   // the list is still there
    fireEvent.click(screen.getByText('Try again now'));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('no stale warning when the list is current', () => {
    mount({ loadState: 'loaded', discoveredFiles: FILES });
    expect(screen.queryByText(/This list may be out of date/)).toBeNull();
  });

  it('shows the phases when there are phases', () => {
    mount({ loadState: 'loaded', discoveredFiles: FILES });
    expect(screen.getByText(/Al\.cif/)).toBeTruthy();
    expect(screen.queryByText('Loading phases…')).toBeNull();
  });
});
