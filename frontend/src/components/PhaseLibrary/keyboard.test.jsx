// @vitest-environment jsdom
/**
 * The keyboard pass (spec §2.9).
 *
 * "Die Mockups haben KEINEN Tastaturpfad" -- so this file checks the things
 * that are easy to leave out and impossible to notice with a mouse: that the
 * search has the focus when the page opens, that every control is a real
 * control rather than a styled div, and that a disabled chip is out of the
 * tab order instead of being a stop that does nothing.
 *
 * It also pins the consistency the pass turned up: a phase was selectable in
 * the band view and not in the flat list, while the selection is supposed to
 * survive the switch between them.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';
import i18n from '../../i18n';
import fixture from './__fixtures__/library.json';
import PhaseLibraryPage from './PhaseLibraryPage';

const gives = (phases) => () => Promise.resolve(
  { phases, unassigned_masters: fixture.unassigned_masters });

async function show(phases = fixture.phases) {
  await act(async () => { await i18n.changeLanguage('en'); });
  const view = render(<PhaseLibraryPage loadIndex={gives(phases)} />);
  await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
  return view;
}

const searchBox = () => screen.getByLabelText(/search the phase library/i);
const toList = () => fireEvent.click(screen.getByRole('button', { name: /^list$/i }));
const boxFor = (key) => document.querySelector(`[data-phase-key="${key}"] input`);

afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('the search is where you land', () => {
  it('has the focus as soon as the page opens', async () => {
    await show();
    expect(document.activeElement).toBe(searchBox());
  });

  it('and it is a real search field, not a div with a border', async () => {
    await show();
    expect(searchBox().tagName).toBe('INPUT');
    expect(searchBox().getAttribute('type')).toBe('search');
  });
});

describe('every control is a control', () => {
  it('chips, methods and band headers are buttons', async () => {
    await show();
    for (const name of ['Al', 'Spherical']) {
      const b = screen.getAllByRole('button')
        .find((x) => x.textContent.startsWith(name));
      expect(b, name).toBeTruthy();
      expect(b.tagName, name).toBe('BUTTON');
      expect(b.getAttribute('aria-pressed'), name).toBe('false');
    }
    const head = document.querySelector('[data-band="Al"] button');
    expect(head.getAttribute('aria-expanded')).toBe('true');
  });

  it('a chip that would empty the list is out of the tab order, not a dead stop',
     async () => {
    await show();
    const chip = (sym) => screen.getAllByRole('button')
      .find((b) => new RegExp('^' + sym + '[0-9]+$').test(b.textContent));
    fireEvent.click(chip('Ni'));
    await waitFor(() => expect(chip('Cu').disabled).toBe(true));
    // A disabled button is skipped by the browser's own tab handling; an
    // enabled one that does nothing is a stop that lies.
    expect(chip('Ni').disabled).toBe(false);          // still undoable
  });

  it('a phase row is a labelled checkbox', async () => {
    await show();
    const box = boxFor('Al');
    expect(box.tagName).toBe('INPUT');
    expect(box.getAttribute('type')).toBe('checkbox');
    expect(box.getAttribute('aria-label')).toMatch(/select/i);
    // Space on a checkbox is the browser's job; what this asserts is that
    // there IS a checkbox for it to act on.
    fireEvent.click(box);
    await waitFor(() => expect(boxFor('Al').checked).toBe(true));
  });
});

describe('one phase, one behaviour, in both views', () => {
  it('a phase is selectable in the flat list too', async () => {
    // It was not: the list rendered plain divs, so the same phase was
    // selectable in one view and not the other, decided by a button at the
    // top of the page.
    await show();
    toList();
    await waitFor(() => expect(screen.queryByTestId('phase-bands')).toBeNull());
    const box = boxFor('Al');
    expect(box).toBeTruthy();
    fireEvent.click(box);
    await waitFor(() => expect(boxFor('Al').checked).toBe(true));
  });

  it('a selection survives the switch between views', async () => {
    await show();
    fireEvent.click(boxFor('Al'));
    await waitFor(() => expect(boxFor('Al').checked).toBe(true));
    toList();
    await waitFor(() => expect(screen.queryByTestId('phase-bands')).toBeNull());
    expect(boxFor('Al').checked).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: /^systems$/i }));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    expect(boxFor('Al').checked).toBe(true);
  });
});
