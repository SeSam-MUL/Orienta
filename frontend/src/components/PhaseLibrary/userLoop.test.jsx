// @vitest-environment jsdom
/**
 * What the user loop after M1 changed, and why.
 *
 * Four readers walked the built page against the real library without a
 * manual. Three of them independently concluded the page was broken, for
 * the same reason, and the quotes are kept here because they are the
 * argument: "I pressed Clear and literally nothing changed", "that poisons
 * everything after it", "I have no way to learn the state of the library
 * except by triggering the empty state".
 *
 * Each test below pins one of their findings. They are in this file rather
 * than spread through the others so that the next person can see what the
 * loop bought, and so that undoing any of it fails a test that says who
 * asked for it.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';
import i18n from '../../i18n';
import fixture from './__fixtures__/library.json';
import PhaseLibraryPage from './PhaseLibraryPage';
import cards from './__fixtures__/cards.json';
import useDatabaseReveal from '../../stores/useDatabaseReveal';

const gives = () => Promise.resolve(
  { phases: fixture.phases, unassigned_masters: fixture.unassigned_masters });

const loadCard = (key) => Promise.resolve(cards[key]);

async function show() {
  await act(async () => { await i18n.changeLanguage('en'); });
  render(<PhaseLibraryPage loadIndex={gives} loadCard={loadCard} />);
  await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
}

const chip = (sym) => screen.getAllByRole('button')
  .find((b) => new RegExp('^' + sym + '[0-9]+$').test(b.textContent));
const method = (name) => screen.getAllByRole('button')
  .find((b) => new RegExp('^' + name + '[0-9]+$').test(b.textContent));
const shown = () => new Set([...document.querySelectorAll('[data-phase-key]')]
  .map((n) => n.getAttribute('data-phase-key'))).size;

afterEach(() => { cleanup(); vi.useRealTimers(); useDatabaseReveal.getState().clear(); });

describe('what is filtering, while it is filtering', () => {
  // "The dangerous case is not zero results, it is seventeen
  // plausible-looking results that are secretly the wrong seventeen."
  // The page named the active filters ONLY in the empty state.
  it('names the filters on screen as soon as any are on', async () => {
    await show();
    expect(screen.queryByTestId('active-filters')).toBeNull();

    fireEvent.click(chip('Fe'));
    fireEvent.click(method('Spherical'));
    const bar = await screen.findByTestId('active-filters');
    expect(bar.textContent).toMatch(/Fe/);
    expect(bar.textContent).toMatch(/Spherical/);
  });

  it('and says them out loud, not just as a number', async () => {
    // A keyboard reader pressing the wrong chip heard a number change and
    // nothing else.
    await show();
    fireEvent.click(chip('Fe'));
    await waitFor(() => {
      expect(screen.getByRole('status').textContent).toMatch(/Fe/);
    });
  });

  it('one control undoes all of it, from the bar itself', async () => {
    await show();
    fireEvent.click(chip('Fe'));
    fireEvent.click(method('Spherical'));
    await waitFor(() => expect(shown()).toBe(17));
    const bar = screen.getByTestId('active-filters');
    fireEvent.click(bar.querySelector('button'));
    await waitFor(() => expect(shown()).toBe(36));
    expect(screen.queryByTestId('active-filters')).toBeNull();
  });
});

describe('the two resets', () => {
  it('say on the button which one they are', async () => {
    await show();
    fireEvent.click(chip('Fe'));
    fireEvent.click(method('Spherical'));
    await waitFor(() => expect(shown()).toBe(17));
    expect(screen.getByRole('button', { name: /^clear elements$/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^clear methods$/i })).toBeTruthy();
  });

  it('and each leaves the other alone', async () => {
    await show();
    fireEvent.click(chip('Fe'));
    fireEvent.click(method('Spherical'));
    await waitFor(() => expect(shown()).toBe(17));
    fireEvent.click(screen.getByRole('button', { name: /^clear methods$/i }));
    await waitFor(() => expect(shown()).toBe(19));      // Fe alone
    fireEvent.click(screen.getByRole('button', { name: /^clear elements$/i }));
    await waitFor(() => expect(shown()).toBe(36));
  });
});

describe('getting past a band without a mouse', () => {
  // 65 Tab stops to reach Mg, ~110 to cross the page. Collapsing every band
  // turns it into a nine-line table of contents -- which the reader found
  // by himself in 18 keystrokes, and nothing said it was possible.
  it('collapses and expands every band from one control', async () => {
    await show();
    const bandRows = () => document.querySelectorAll('[data-band] [data-phase-key]').length;
    expect(bandRows()).toBe(93);

    fireEvent.click(screen.getByRole('button', { name: /collapse all bands/i }));
    await waitFor(() => expect(bandRows()).toBe(0));
    // The counts stay: they are the reason to open one again.
    expect(document.querySelector('[data-band="Al"]').textContent).toContain('28');
    expect(document.querySelectorAll('[data-band]')).toHaveLength(9);

    fireEvent.click(screen.getByRole('button', { name: /expand all bands/i }));
    await waitFor(() => expect(bandRows()).toBe(93));
  });
});

describe('names a screen reader can tell apart', () => {
  it('the Al band header is not the Al chip', async () => {
    // Both read "Al 28" -- same name, same number, two different actions.
    await show();
    const header = document.querySelector('[data-band="Al"] button');
    // The chip has no aria-label: its accessible name IS its text, "Al28".
    // The header's is its label. What matters is that the two differ, so a
    // list of buttons does not show the same entry twice.
    const headerName = header.getAttribute('aria-label');
    const chipName = chip('Al').getAttribute('aria-label') || chip('Al').textContent;
    expect(headerName).toMatch(/band/i);
    expect(headerName).toMatch(/28/);
    expect(headerName).not.toBe(chipName);
    expect(chipName).not.toMatch(/band/i);
  });

  it('a method row carries what the method needs, not just a number', async () => {
    // `title` is invisible to keyboard focus, so "Hough 35, button" was all
    // a keyboard reader ever got. Thirty-five what?
    await show();
    const label = method('Spherical').getAttribute('aria-label');
    expect(label).toMatch(/\.sht/);
    expect(label).toMatch(/29 of 36/);
  });

  it('a phase ticked in three bands has three distinguishable names', async () => {
    await show();
    const boxes = [...document.querySelectorAll(
      '[data-phase-key="Al3Fe2Si_mp-1190708_symmetrized"] input')];
    expect(boxes).toHaveLength(3);
    const names = boxes.map((b) => b.getAttribute('aria-label'));
    expect(new Set(names).size).toBe(3);
    for (const el of ['Al', 'Fe', 'Si']) {
      expect(names.some((n) => n.includes(el)), el).toBe(true);
    }
  });

  it('the caret is not part of the header name', async () => {
    await show();
    expect(document.querySelector('[data-band="Al"] [aria-hidden="true"]').textContent)
      .toMatch(/[▾▸]/);
  });
});

describe('what a tick is for', () => {
  it('nothing says it until something is ticked, then it says it plainly', async () => {
    await show();
    expect(screen.queryByTestId('selection-note')).toBeNull();
    fireEvent.click(document.querySelector('[data-phase-key="Al"] input'));
    const note = await screen.findByTestId('selection-note');
    expect(note.textContent).toMatch(/1 phase selected/i);
    expect(note.textContent).toMatch(/stays on this page/i);
  });

  it('and can be undone without hunting for the ticks again', async () => {
    await show();
    fireEvent.click(document.querySelector('[data-phase-key="Al"] input'));
    fireEvent.click(document.querySelector('[data-phase-key="Al13Fe4"] input'));
    await waitFor(() => {
      expect(screen.getByTestId('selection-note').textContent)
        .toMatch(/2 phases selected/i);
    });
    fireEvent.click(screen.getByRole('button', { name: /clear selection/i }));
    await waitFor(() => expect(screen.queryByTestId('selection-note')).toBeNull());
  });
});

describe('two rows with the same name', () => {
  // Every reader in the loop got stuck here. "I'd click the first one and
  // hope, and I'd have a coin-flip in my results without knowing it."
  // MEASURED before choosing what the short line carries: the twins share
  // their label, their Pearson symbol (cI168), their space group (Im-3) and
  // their atom count (168). Formula and Pearson would have left them
  // exactly as indistinguishable. What differs is the lattice parameter --
  // 12.5 against 12.56 Å -- and the file.
  const twinRows = () => [...document.querySelectorAll('[data-phase-key]')]
    .filter((n) => ['sd_0302719', 'sd_1401510']
      .includes(n.getAttribute('data-phase-key')));

  it('are told apart in the BAND view, where the reader met them', async () => {
    await show();
    expect(screen.getByTestId('phase-bands')).toBeTruthy();
    const rows = twinRows();
    expect(rows.length).toBeGreaterThanOrEqual(2);

    const texts = rows.map((n) => n.textContent);
    // Both still carry the same name, which is the honest thing -- it IS
    // their name -- so the discriminator has to be beside it.
    for (const text of texts) expect(text).toContain('Mn0.5Fe0.5Al5Si0.68');
    const first = texts.find((x) => x.includes('sd_0302719'));
    const second = texts.find((x) => x.includes('sd_1401510'));
    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    expect(first).not.toBe(second);
  });

  it('by the cell, which is the thing that actually differs', async () => {
    await show();
    const byKey = (k) => [...document.querySelectorAll(`[data-phase-key="${k}"]`)]
      .map((n) => n.textContent).join(' ');
    expect(byKey('sd_0302719')).toContain('12.5');
    expect(byKey('sd_1401510')).toContain('12.56');
    // And not by Pearson or space group, which they share.
    expect(byKey('sd_0302719')).toContain('cI168');
    expect(byKey('sd_1401510')).toContain('cI168');
  });
});

describe('the headline counts what it can count', () => {
  it('says entries, not phases', async () => {
    // The library holds 36 FILES. A lab head counted at least four pairs
    // that are one phase twice and put the number of distinct physical
    // phases "somewhere near 30". "36 phases" states something the library
    // has not established.
    await show();
    const live = screen.getByRole('status').textContent;
    expect(live).toMatch(/36 of 36 entries/i);
    expect(live).not.toMatch(/36 phases/i);
  });
});

describe('the deep link, end to end on the page', () => {
  // CAUGHT IN THE RUNNING APP, not here: the page's handler called
  // `onNavigate` and the page never destructured it, so clicking "show in
  // the database browser" threw `ReferenceError: onNavigate is not defined`
  // and went nowhere. The card's own tests passed -- they hand it a mock
  // and assert the call -- and the seam between card and page had nothing
  // on it. This test is that seam.
  it('asks the shell to change page, and says which', async () => {
    const onNavigate = vi.fn();
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<PhaseLibraryPage loadIndex={gives} onNavigate={onNavigate}
                              loadCard={loadCard} />);
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    fireEvent.click(document.querySelector('[data-phase-key="Al"] button'));
    const link = await waitFor(() => {
      const b = [...document.querySelectorAll('[data-testid="phase-card"] button')]
        .find((x) => /database browser/i.test(x.textContent));
      expect(b).toBeTruthy();
      return b;
    });
    fireEvent.click(link);
    expect(onNavigate).toHaveBeenCalledWith('database');
    expect(useDatabaseReveal.getState().pending).toMatchObject({
      category: 'cif_library', tab: 'cif',
    });
  });

  it('and does not throw when the shell gave it nowhere to go', async () => {
    // The page is rendered without `onNavigate` in several tests, and a
    // handler that assumed it was there is how the ReferenceError happened.
    await show();
    fireEvent.click(document.querySelector('[data-phase-key="Al"] button'));
    const link = await waitFor(() => {
      const b = [...document.querySelectorAll('[data-testid="phase-card"] button')]
        .find((x) => /database browser/i.test(x.textContent));
      expect(b).toBeTruthy();
      return b;
    });
    expect(() => fireEvent.click(link)).not.toThrow();
  });
});

describe('the sentence that read like a fault report', () => {
  it('says what it means for the reader', async () => {
    // Two of four readers took "N simulated masters on disk belong to no
    // phase in this library" for an error, on arrival, before touching
    // anything -- one would have screenshotted it and asked a postdoc.
    await show();
    const facets = screen.getByTestId('phase-library-facets').textContent;
    expect(facets).toMatch(/4 simulated masters/i);
    expect(facets).toMatch(/nothing is missing/i);
  });
});

describe('the twin a reader would never have found', () => {
  /**
   * Two structure models of one phase, thirty years apart. In a list they
   * are indistinguishable -- same label composition, same `Im-3`, same
   * `cI168`, 12.50 A against 12.56 A -- and typing `Al Fe Si` returns one
   * of them FIRST and the other LAST of fifteen hits. Clicking the first
   * and carrying on is the natural thing to do.
   *
   * These go through the page, opening a real card, because the point is
   * that the card knows about a phase it was not asked about.
   */
  const open = async (key) => {
    const row = document.querySelector(`[data-phase-key="${key}"] button`);
    fireEvent.click(row);
    return waitFor(() => expect(screen.getByTestId('phase-card')).toBeTruthy());
  };

  it('each of the pair names the other, with its key', async () => {
    await show();
    await open('sd_0302719');
    const note = screen.getByTestId('sibling-note');
    expect(note.textContent).toMatch(/same space group and the same compound/);
    expect(note.querySelector('[data-sibling-key="sd_1401510"]')).toBeTruthy();
    // The key as well as the name: the two are called `α-Al(Fe,Mn)Si` and
    // `α-Al(Fe,Mn)`, two characters apart.
    expect(note.textContent).toMatch(/sd_1401510/);
  });

  it('and says what differs, starting with what you can index with', async () => {
    await show();
    await open('sd_0302719');
    const kinds = [...screen.getByTestId('sibling-note')
      .querySelectorAll('[data-difference]')]
      .map((n) => n.getAttribute('data-difference'));
    expect(kinds[0]).toBe('notIndexableWith');
    expect(kinds).toContain('composition');
    expect(kinds).toContain('reference');
  });

  it('a phase with no twin gets no note at all', async () => {
    await show();
    await open('Al');
    expect(screen.queryByTestId('sibling-note')).toBe(null);
  });
});
