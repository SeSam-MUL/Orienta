// @vitest-environment jsdom
/**
 * The card on screen, and the one interaction question it raises.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';
import i18n from '../../i18n';
import cards from './__fixtures__/cards.json';
import PhaseCard from './PhaseCard';
import PhaseRow from './PhaseRow';

const load = (key) => Promise.resolve(cards[key]);
const never = () => new Promise(() => {});
const notFound = () => Promise.reject({ response: { status: 404 } });
const broke = () => Promise.reject(new Error('Network Error'));

async function showCard(key, loader = load) {
  await act(async () => { await i18n.changeLanguage('en'); });
  render(<PhaseCard phaseKey={key} load={loader} />);
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
}

afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('the four states, again and for the same reason', () => {
  it('says it is reading', async () => {
    await showCard('Al', never);
    expect(screen.getByTestId('card-loading')).toBeTruthy();
  });

  it('distinguishes "no such phase" from "could not ask"', async () => {
    await showCard('nope', notFound);
    expect(screen.getByTestId('card-missing')).toBeTruthy();
    cleanup();
    await showCard('Al', broke);
    expect(screen.getByTestId('card-error')).toBeTruthy();
  });
});

describe('what one card shows', () => {
  it('the whole of §2.3 for a phase that has everything', async () => {
    await showCard('Al');
    const card = await screen.findByTestId('phase-card');
    expect(card.getAttribute('data-card-key')).toBe('Al');

    expect(screen.getByTestId('card-composition').textContent).toContain('Al');
    // Every lattice constant, not a selection of them.
    const cell = screen.getByTestId('card-cell').textContent;
    for (const bit of ['4.049 Å', '90°']) expect(cell).toContain(bit);
    // A tick or a cross per file, and the repo-relative path.
    const files = screen.getByTestId('card-files').textContent;
    expect(files).toContain('Al.cif');
    expect(files).toContain('Database/CIF_Library/Al.cif');
    // The simulation parameters behind master and SHT.
    expect(screen.getByTestId('card-master').textContent).toContain('20 kV');
    expect(screen.getByTestId('card-sht').textContent).toContain('384');
  });

  it('shows a cross and a word, never only a colour', async () => {
    // `beta-AlFeSi` has a CIF and nothing else. Red alone reads as an
    // error, and a phase without a dictionary is not an error.
    await showCard('beta-AlFeSi');
    await screen.findByTestId('phase-card');
    const dict = document.querySelector('[data-file="dictionary"]');
    expect(dict.textContent).toContain('✗');
    expect(dict.textContent).toMatch(/not here/i);
    expect(document.querySelector('[data-file="cif"]').textContent).toContain('✓');
  });

  it('never prints an absolute path', async () => {
    for (const key of ['Al', 'Si', 'sd_1816951']) {
      cleanup();
      await showCard(key);
      const text = (await screen.findByTestId('phase-card')).textContent;
      expect(text, key).not.toMatch(/[A-Za-z]:[\\/]/);
      expect(text.toLowerCase(), key).not.toContain('sebas');
    }
  });
});

describe('the cards that are not straightforward', () => {
  it('an unreadable CIF says so and still shows what the other files give', async () => {
    await showCard('sd_1816951');
    await screen.findByTestId('phase-card');
    const notice = document.querySelector('[data-notice="parseError"]');
    expect(notice).toBeTruthy();
    expect(notice.textContent).toMatch(/2 different compositions/);
    // Not an empty card: it has an xtal, an sht and a master.
    expect(document.querySelector('[data-file="sht"]').textContent).toContain('✓');
    expect(screen.getByTestId('card-cell').textContent).toContain('7.0309');
  });

  it('a name that claims an element the structure lacks is flagged', async () => {
    await showCard('beta-AlFeSi');
    await screen.findByTestId('phase-card');
    expect(document.querySelector('[data-notice="elementsDisagree"]')).toBeTruthy();
    const comp = screen.getByTestId('card-composition').textContent;
    expect(comp).toMatch(/from the structure/i);
    expect(comp).toMatch(/from the label/i);
  });

  it('says which cell setting is on screen when the file declares several', async () => {
    await showCard('Si');
    await screen.findByTestId('phase-card');
    expect(document.querySelector('[data-notice="severalSettings"]')).toBeTruthy();
  });

  it('the twins are told apart by their cell, on the card as in the row', async () => {
    await showCard('sd_0302719');
    expect((await screen.findByTestId('card-cell')).textContent).toContain('12.5 Å');
    cleanup();
    await showCard('sd_1401510');
    expect((await screen.findByTestId('card-cell')).textContent).toContain('12.56 Å');
  });
});

describe('showing a file in the database browser', () => {
  it('offers the link for a file that is there, and not for one that is not', async () => {
    const onReveal = vi.fn();
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<PhaseCard phaseKey="beta-AlFeSi" load={load} onReveal={onReveal} />);
    await screen.findByTestId('phase-card');
    // This phase has a CIF and nothing else.
    const cif = document.querySelector('[data-file="cif"]');
    const dict = document.querySelector('[data-file="dictionary"]');
    expect(cif.querySelector('button')).toBeTruthy();
    expect(dict.querySelector('button')).toBeNull();
  });

  it('asks with the category the browser needs, not with a path', async () => {
    const onReveal = vi.fn();
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<PhaseCard phaseKey="Al" load={load} onReveal={onReveal} />);
    await screen.findByTestId('phase-card');
    fireEvent.click(document.querySelector('[data-file="master"] button'));
    expect(onReveal).toHaveBeenCalledWith('h5_cache', 'Al_master_E20kV_npx500.h5');
  });

  it('and shows no link at all when nobody can act on it', async () => {
    await showCard('Al');
    await screen.findByTestId('phase-card');
    expect(screen.getByTestId('card-files').querySelectorAll('button')).toHaveLength(0);
  });
});

describe('opening it from a row', () => {
  const phase = { key: 'Al', formula_label: 'Al' };

  async function showRow() {
    await act(async () => { await i18n.changeLanguage('en'); });
    const onToggle = vi.fn();
    render(
      <PhaseRow
        phase={phase}
        label="Al"
        checked={false}
        onToggle={onToggle}
        selectLabel="Select Al"
        withCard
        loadCard={load}
      />,
    );
    return onToggle;
  }

  it('the name opens the card', async () => {
    await showRow();
    fireEvent.click(screen.getByRole('button', { name: 'Al' }));
    await waitFor(() => expect(screen.getByTestId('phase-card')).toBeTruthy());
  });

  it('and does NOT tick the checkbox on the way', async () => {
    // The name is a button INSIDE the row's `<label>`. HTML says a label
    // does not forward its activation when the click landed on interactive
    // content -- which is a rule worth testing rather than reading.
    const onToggle = await showRow();
    fireEvent.click(screen.getByRole('button', { name: 'Al' }));
    await waitFor(() => expect(screen.getByTestId('phase-card')).toBeTruthy());
    expect(onToggle).not.toHaveBeenCalled();
    expect(screen.getByRole('checkbox').checked).toBe(false);
  });

  it('and the checkbox still ticks when IT is clicked', async () => {
    const onToggle = await showRow();
    fireEvent.click(screen.getByRole('checkbox'));
    expect(onToggle).toHaveBeenCalledWith('Al');
  });

  it('nothing is fetched for a card nobody opened', async () => {
    const asked = vi.fn(load);
    await act(async () => { await i18n.changeLanguage('en'); });
    render(
      <PhaseRow phase={phase} label="Al" checked={false} onToggle={() => {}}
                selectLabel="Select Al" withCard loadCard={asked} />,
    );
    await act(async () => { await Promise.resolve(); });
    expect(asked).not.toHaveBeenCalled();
  });
});

describe('the byline after a rename, which the two sides spell differently', () => {
  // THE DEFECT: the card reads `name_author`/`name_updated` -- what
  // `build_phase_detail` emits -- while `PUT /names` answers with
  // `author`/`updated`. `{...cur, ...stored}` therefore left the OLD
  // byline in place, so after Bob renamed a phase Alice had named, the
  // card showed Bob's name credited to Alice until the page was read
  // again. In a shared library that is somebody's name on a decision they
  // did not make, which is the thing this byline exists to prevent.
  const namedCard = {
    ...cards.Al,
    display_name: 'Aluminium (Alice)',
    name_author: 'alice',
    name_updated: '2026-09-01T10:00:00',
  };

  async function renameIt() {
    await act(async () => { await i18n.changeLanguage('en'); });
    const save = vi.fn(() => Promise.resolve({
      key: 'Al',
      display_name: 'Aluminium (Bob)',
      search_terms: [],
      author: 'bob',
      updated: '2026-09-27T12:00:00',
    }));
    render(<PhaseCard phaseKey="Al" load={() => Promise.resolve(namedCard)}
                      saveNames={save} />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    fireEvent.click(screen.getByTestId('name-editor-open'));
    fireEvent.change(screen.getByLabelText('Name'),
      { target: { value: 'Aluminium (Bob)' } });
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(save).toHaveBeenCalled());
    return save;
  }

  it('credits whoever just saved, not whoever saved last time', async () => {
    await renameIt();
    const by = await screen.findByTestId('name-author');
    expect(by.textContent).toMatch(/bob/);
    expect(by.textContent).not.toMatch(/alice/);
  });

  it('and moves the date with it', async () => {
    await renameIt();
    const by = await screen.findByTestId('name-author');
    expect(by.textContent).toMatch(/2026-09-27/);
  });
});

describe('the dictionary slot, when a master can make one', () => {
  // b9's finding: the row's H S D says Dictionary is possible (a master
  // is here, and a dictionary is generated from one) while the file list
  // says the dictionary slot is empty. Both true; the card showed only
  // the emptier of the two, so a reader comparing them found a
  // contradiction and no way to tell which half was wrong.
  //
  // Five of the eight cards in this library are in exactly this state,
  // including both twins, so this is measured rather than invented.

  it('says a master is here and one can be generated', async () => {
    await showCard('sd_1401510');
    const row = (await screen.findByTestId('card-files'))
      .querySelector('[data-file="dictionary"]');
    expect(row.textContent).toMatch(/master pattern is here/);
    expect(row.textContent).not.toMatch(/^.*Absent/);
  });

  it('and a phase that cannot do Dictionary at all still says absent', async () => {
    // `beta-AlFeSi` has no master, so there is nothing to generate from.
    await showCard('beta-AlFeSi');
    const row = (await screen.findByTestId('card-files'))
      .querySelector('[data-file="dictionary"]');
    expect(row.textContent).not.toMatch(/master pattern is here/);
  });

  it('and a phase that HAS a dictionary shows the file, not the offer', async () => {
    await showCard('Al');
    const row = (await screen.findByTestId('card-files'))
      .querySelector('[data-file="dictionary"]');
    expect(row.textContent).not.toMatch(/can be generated/);
  });
});
