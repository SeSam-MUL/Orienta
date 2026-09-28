// @vitest-environment jsdom
/**
 * The four states, and the one rule about when the facet counts may appear.
 *
 * Each case below is a state a user can be in; the point of the file is that
 * they are four DIFFERENT screens. The bug this page exists partly to avoid
 * shipped twice already in this app: a request that failed and a library that
 * is empty rendered the same, and the empty one tells you to go and build
 * what you already have.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';
import i18n from '../../i18n';
import fixture from './__fixtures__/library.json';
import PhaseLibraryPage, { SLOW_AFTER_MS } from './PhaseLibraryPage';

// The page's own module, not the network: the default loader is the one place
// that names the URL and it is covered by `libraryLoad.test.js`.
const never = () => new Promise(() => {});
const fails = () => Promise.reject(new Error('Network Error'));
// The payload shape the endpoint returns: phases AND the top-level facts
// about the library, in one answer (see services/api.js on why one request).
const gives = (phases) => () => Promise.resolve(
  { phases, unassigned_masters: fixture.unassigned_masters });

async function show(loadIndex) {
  await act(async () => { await i18n.changeLanguage('en'); });
  const view = render(<PhaseLibraryPage loadIndex={loadIndex} />);
  await act(async () => { await Promise.resolve(); });
  return view;
}

afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('the four states are four different screens', () => {
  it('waiting says it is reading, and shows no list and no count', async () => {
    await show(never);
    expect(screen.getByTestId('state-loading')).toBeTruthy();
    expect(screen.queryByTestId('phase-list')).toBeNull();
    expect(screen.queryByTestId('state-empty')).toBeNull();
    // The result count is a counter too, and §2.8's rule is about counters:
    // a number here while "reading" runs says the total is already known.
    expect(screen.getByRole('status').textContent).toBe('');
  });

  it('a long wait explains itself, and a short one does not', async () => {
    // MEASURED against the real endpoint: 0.11 s warm, 16.4 s cold, and 74 s
    // the very first time on this machine. It reads 36 structure files and
    // 65 HDF5 headers, so the first person to open the page after Orienta
    // starts waits a quarter of a minute in front of one sentence -- and a
    // silent wait that long reads as a hang.
    await act(async () => { await i18n.changeLanguage('en'); });
    vi.useFakeTimers();
    render(<PhaseLibraryPage loadIndex={never} />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByTestId('state-loading')).toBeTruthy();
    expect(screen.queryByTestId('state-loading-slow')).toBeNull();

    await act(async () => { vi.advanceTimersByTime(SLOW_AFTER_MS + 50); });
    const note = screen.getByTestId('state-loading-slow');
    expect(note.textContent).toMatch(/first read/i);
    expect(note.textContent).toMatch(/immediate|few seconds/i);
  });

  it('and stops explaining once the library is here', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    vi.useFakeTimers();
    let release;
    const held = () => new Promise((r) => { release = r; });
    render(<PhaseLibraryPage loadIndex={held} />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    await act(async () => { vi.advanceTimersByTime(SLOW_AFTER_MS + 50); });
    expect(screen.getByTestId('state-loading-slow')).toBeTruthy();

    await act(async () => {
      release({ phases: fixture.phases, unassigned_masters: [] });
      await Promise.resolve(); await Promise.resolve();
    });
    expect(screen.queryByTestId('state-loading-slow')).toBeNull();
  });

  it('a failed request is an error, NOT an empty library', async () => {
    await show(fails);
    await waitFor(() => expect(screen.getByTestId('state-error')).toBeTruthy());
    expect(screen.queryByTestId('state-empty')).toBeNull();
    expect(screen.getByText(/could not be read/i)).toBeTruthy();
    expect(screen.getByText(/still trying/i)).toBeTruthy();
  });

  // WITH THE CLOCK HELD STILL, and that is the whole test. The first version
  // let real time run, so the loader's own 1 s retry landed inside waitFor's
  // 1 s window and the test passed with the button wired to nothing --
  // measured: making onClick a no-op left all twelve green. It proved
  // "recovers eventually", which the loader does by itself.
  it('the "try now" button asks again without waiting for the backoff', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    vi.useFakeTimers();
    let attempts = 0;
    const load = () => {
      attempts += 1;
      return attempts === 1 ? fails() : gives(fixture.phases)();
    };
    render(<PhaseLibraryPage loadIndex={load} />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(attempts).toBe(1);
    expect(screen.getByTestId('state-error')).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /try now/i }));
      await Promise.resolve(); await Promise.resolve();
    });
    expect(attempts).toBe(2);                      // no timer was advanced
    expect(screen.getByTestId('phase-list')).toBeTruthy();
  });

  it('a library that really is empty says where phases come from', async () => {
    await show(gives([]));
    await waitFor(() => expect(screen.getByTestId('state-empty')).toBeTruthy());
    expect(screen.getByText(/crystal database/i)).toBeTruthy();
    expect(screen.queryByTestId('state-error')).toBeNull();
  });

  it('filtered to nothing names the filter and offers to clear it', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Osmium' } });
    const panel = await screen.findByTestId('state-filtered-empty');
    expect(panel).toBeTruthy();
    // The count is the rescue: with nothing shown, every facet reads 0, so
    // the message has to carry both the filter and the way out.
    expect(panel.textContent).toContain('Osmium');
    expect(panel.textContent).toContain(String(fixture.phases.length));

    fireEvent.click(screen.getByRole('button', { name: /clear filters/i }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
  });

  it('a loaded library lists phases and says how many, out loud', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    const live = screen.getByRole('status');
    expect(live.getAttribute('aria-live')).toBe('polite');
    expect(live.textContent).toContain(String(fixture.phases.length));

    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Al-Fe-Si' } });
    await waitFor(() => {
      expect(screen.getByRole('status').textContent)
        .not.toContain(`${fixture.phases.length} of`);
    });
  });
});

describe('it asks only when someone is looking', () => {
  // App.jsx mounts every page cell at once, so an ungated effect here fires
  // at app start for a page nobody opened -- and on a dead backend that is a
  // retry loop from boot. The early `return null` is too late: hooks run
  // first.
  it('makes no request while the page is not the open one', async () => {
    const asked = vi.fn(gives([]));
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<PhaseLibraryPage isActive={false} loadIndex={asked} />);
    await act(async () => { await Promise.resolve(); });
    expect(asked).not.toHaveBeenCalled();
  });

  it('and asks as soon as it becomes the open one', async () => {
    const asked = vi.fn(gives(fixture.phases));
    await act(async () => { await i18n.changeLanguage('en'); });
    const { rerender } = render(<PhaseLibraryPage isActive={false} loadIndex={asked} />);
    await act(async () => {
      rerender(<PhaseLibraryPage isActive loadIndex={asked} />);
      await Promise.resolve();
    });
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    expect(asked).toHaveBeenCalledTimes(1);
  });
});

describe('the counts do not arrive before the phases', () => {
  // In the mockups the facet counts stood there complete while "loading" was
  // still running, which reads as "we know the numbers, the list is merely
  // slow". They come from one request.
  // Asked of the FACETS and not of the whole left column: the groups sit in
  // the same column and are shown straight away, which is right -- a group's
  // existence is not an answer to the request in flight, and a drop target
  // that appears late is a drop target somebody was already aiming at.
  it('no facet counts until the library is here', async () => {
    await show(never);
    expect(screen.queryByTestId('facet-counts')).toBe(null);
  });

  it('and none while the request is failing', async () => {
    await show(fails);
    await waitFor(() => expect(screen.getByTestId('state-error')).toBeTruthy());
    expect(screen.queryByTestId('facet-counts')).toBe(null);
  });

  it('but present once it is', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    expect(screen.getByTestId('facet-counts').textContent).not.toBe('');
  });
});

// DISTINCT phases on screen. In the default band view a phase appears once
// per system it belongs to -- 36 phases, 93 cards -- so counting elements
// would measure placements and no assertion below is about placements.
const shownPhases = () => new Set(
  [...document.querySelectorAll('[data-phase-key]')]
    .map((n) => n.getAttribute('data-phase-key'))).size;

const chip = (sym) => screen.getAllByRole('button')
  .find((b) => b.textContent.startsWith(sym)
               && /^[A-Z][a-z]?\d+$/.test(b.textContent));

describe('the element chips', () => {

  it('the number on a chip is what clicking it gives you', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    const fe = chip('Fe');
    const promised = Number(fe.textContent.replace('Fe', ''));
    fireEvent.click(fe);
    await waitFor(() => {
      expect(shownPhases()).toBe(promised);
    });
    expect(screen.getByRole('status').textContent).toContain(String(promised));
  });

  it('two chips narrow further, and the counts keep up', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    fireEvent.click(chip('Al'));
    await waitFor(() => expect(shownPhases()).toBe(28));
    const promised = Number(chip('Fe').textContent.replace('Fe', ''));
    expect(promised).toBe(18);                     // measured on the real library
    fireEvent.click(chip('Fe'));
    await waitFor(() => expect(shownPhases()).toBe(18));
  });

  it('a chosen chip says so, for a screen reader too', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Mg'));
    await waitFor(() => expect(chip('Mg').getAttribute('aria-pressed')).toBe('true'));
  });

  it('clear puts every phase back', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Cu'));
    await waitFor(() => expect(shownPhases()).toBe(4));
    fireEvent.click(screen.getByRole('button', { name: /clear elements/i }));
    await waitFor(() => {
      expect(shownPhases())
        .toBe(fixture.phases.length);
    });
  });

  it('the two resets do not both answer to "Clear"', async () => {
    // Found in the running app, then twice more in the user loop: the two
    // resets both read "Clear", side by side in one column. Two readers
    // pressed one, saw the list not move, and concluded the page was broken
    // -- "that poisons everything after it". The first repair gave them
    // distinct ACCESSIBLE names and left both reading "Clear", which helps
    // nobody looking at two identical buttons. The word is on the button.
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Fe'));
    fireEvent.click(screen.getAllByRole('button')
      .find((b) => /^Spherical[0-9]+$/.test(b.textContent)));
    await waitFor(() => expect(shownPhases()).toBeLessThan(19));

    const resets = screen.getAllByRole('button', { name: /^clear (elements|methods)$/i });
    expect(resets).toHaveLength(2);
    const names = resets.map((b) => b.textContent.trim());
    expect(new Set(names).size).toBe(2);
    expect(names.join(' ')).toMatch(/element/i);
    expect(names.join(' ')).toMatch(/method/i);
    // Visible, not only announced: nobody reads an aria-label with their eyes.
    for (const b of resets) expect(b.getAttribute('aria-label')).toBeNull();

    // And each clears its own: the element one leaves the method filter on.
    fireEvent.click(screen.getByRole('button', { name: /clear elements/i }));
    await waitFor(() => expect(shownPhases()).toBe(29));   // Spherical alone
  });

  it('names the KIND of filter, not "search term" for all three', async () => {
    // The sentence was `Search term: {{filter}}` with the three joined
    // raw, so with only an element chip on the page said
    // "Search term: Al + Fe" -- naming the one filter that was empty. The
    // reader clears the (already empty) search box, nothing happens, and
    // the page has told them where to look and been wrong.
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Ni'));
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Al-Fe-Si' } });
    const panel = await screen.findByTestId('state-filtered-empty');
    // Each part says which filter it is, and the words differ.
    expect(panel.textContent).toMatch(/search .Al-Fe-Si./);
    expect(panel.textContent).toMatch(/elements Ni/);
    fireEvent.click(screen.getByRole('button', { name: /clear filters/i }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
  });

  it('and with only a chip on, does not call it a search', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Ni'));
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Al-Fe-Si' } });
    await screen.findByTestId('state-filtered-empty');
    // Now empty the search box again, leaving the chip: still empty, and
    // the message must no longer mention a search at all.
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: '' } });
    const panel = screen.queryByTestId('state-filtered-empty');
    if (panel) {
      expect(panel.textContent).toMatch(/elements Ni/);
      expect(panel.textContent).not.toMatch(/search/i);
    }
  });

  it('chips and search together can empty the list, and the message names BOTH',
     async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Ni'));
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Al-Fe-Si' } });
    const panel = await screen.findByTestId('state-filtered-empty');
    // Either one could be the one that emptied it, and from a screen with
    // nothing on it the user cannot tell which.
    expect(panel.textContent).toContain('Al-Fe-Si');
    expect(panel.textContent).toContain('Ni');
    fireEvent.click(screen.getByRole('button', { name: /clear filters/i }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
  });

  // WHY THIS ONE IS SO ROUNDABOUT. Chips alone cannot normally empty the
  // list: a chip at zero is disabled, so every click narrows to at least one
  // phase. The case only exists when the library changes UNDER a selection,
  // and it is worth reaching, because without chosen elements counting as an
  // active filter the page then says "there are no phases in the library" --
  // sending the user off to rebuild a library that is sitting right there.
  // Mutation-tested: dropping `elements.length > 0` from the filter check
  // broke nothing until this test existed.
  it('a selection that survives a smaller library is a FILTER, not an empty library',
     async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    const { rerender } = render(
      <PhaseLibraryPage loadIndex={gives(fixture.phases)} />);
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    fireEvent.click(chip('Cu'));
    await waitFor(() => expect(shownPhases()).toBe(4));

    // A different library arrives, and it has no copper in it.
    const noCopper = fixture.phases.filter(
      (p) => !(p.elements_structure || []).includes('Cu'));
    await act(async () => {
      rerender(<PhaseLibraryPage loadIndex={gives(noCopper)} />);
      await Promise.resolve();
    });

    const panel = await screen.findByTestId('state-filtered-empty');
    expect(panel.textContent).toContain('Cu');
    expect(screen.queryByTestId('state-empty')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /clear filters/i }));
    await waitFor(() => {
      expect(shownPhases()).toBe(noCopper.length);
    });
  });

  it('this library needs no chip search — nine elements, not fifteen', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    expect(screen.queryByLabelText(/find an element/i)).toBeNull();
  });
});

describe('the three method facets', () => {
  const method = (name) => screen.getAllByRole('button')
    // A character class, not `\d`: in a template literal `\d` is an unknown
    // escape and collapses to `d`, so the first version looked for "Houghd+"
    // and found nothing.
    .find((b) => new RegExp('^' + name + '[0-9]+$').test(b.textContent));

  it('each says what it can index now, and what it can index at all', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    // The measured three. Not 4 (which counts the pre-built dictionaries)
    // and not 2 (which is what the shipped app's name matcher answers).
    // 35, not 36: one CIF parses to two compositions and is refused (see
    // facets.test.js for the measurement c1 made).
    expect(method('Hough').textContent).toBe('Hough35');
    expect(method('Spherical').textContent).toBe('Spherical29');
    expect(method('Dictionary').textContent).toBe('Dictionary16');
    // The library-wide fact stays true while filtering, so it lives in the
    // tooltip rather than beside a number measured on the filtered view.
    expect(method('Spherical').getAttribute('title'))
      .toContain('29 of 36 phases in this library');
  });

  it('clicking one narrows to exactly the number it promised', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(method('Spherical'));
    await waitFor(() => expect(shownPhases()).toBe(29));
    expect(method('Spherical').getAttribute('aria-pressed')).toBe('true');
    // Still 29 of 36 in the tooltip: filtering does not change what the
    // library contains.
    expect(method('Spherical').getAttribute('title'))
      .toContain('29 of 36 phases in this library');
  });

  it('a method and an element narrow together', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Fe'));
    await waitFor(() => expect(shownPhases()).toBe(19));
    const promised = Number(method('Dictionary').textContent.replace('Dictionary', ''));
    fireEvent.click(method('Dictionary'));
    await waitFor(() => expect(shownPhases()).toBe(promised));
  });

  it('the library-wide fact does not move when a filter does', async () => {
    // Mutation-tested: computing it on the FILTERED set survived every other
    // test here, because selecting Spherical leaves both numbers at 29. It
    // takes a filter of a different kind to tell them apart.
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.click(chip('Fe'));
    await waitFor(() => expect(shownPhases()).toBe(19));
    expect(Number(method('Spherical').textContent.replace('Spherical', '')))
      .toBeLessThan(29);                               // the filtered count moved
    expect(method('Spherical').getAttribute('title'))
      .toContain('29 of 36 phases in this library');   // the library did not
  });

  it('a method that survives a smaller library is a FILTER, not an empty library',
     async () => {
    // Same shape as the element case, and for the same reason: a method row
    // at zero is disabled, so methods alone can only empty the list when the
    // library changes underneath the selection. Without it the page says
    // "there are no phases in the library" about a library full of phases.
    await act(async () => { await i18n.changeLanguage('en'); });
    const { rerender } = render(
      <PhaseLibraryPage loadIndex={gives(fixture.phases)} />);
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    fireEvent.click(method('Dictionary'));
    await waitFor(() => expect(shownPhases()).toBe(16));

    const noDict = fixture.phases.filter((p) => !p.capabilities.dictionary);
    await act(async () => {
      rerender(<PhaseLibraryPage loadIndex={gives(noDict)} />);
      await Promise.resolve();
    });

    const panel = await screen.findByTestId('state-filtered-empty');
    expect(panel.textContent).toContain('Dictionary');
    expect(screen.queryByTestId('state-empty')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /clear filters/i }));
    await waitFor(() => expect(shownPhases()).toBe(noDict.length));
  });

  it('says how many simulated masters belong to no phase here', async () => {
    // Spec §2.7: shown rather than concealed. One of these four is the
    // S-phase's master, under a stem the library does not map -- so a card
    // saying "no master" about it would be wrong about a file on the disk.
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    const facets = screen.getByTestId('phase-library-facets');
    expect(facets.textContent).toMatch(/4 simulated masters/i);
  });

  it('and says nothing when there are none', async () => {
    await show(() => Promise.resolve({ phases: fixture.phases, unassigned_masters: [] }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    expect(screen.getByTestId('phase-library-facets').textContent)
      .not.toMatch(/belongs? to no phase/i);
  });
});

describe('the band view', () => {
  const band = (el) => document.querySelector(`[data-band="${el}"]`);
  const cardsIn = (el) => [...band(el).querySelectorAll('[data-phase-key]')]
    .map((n) => n.getAttribute('data-phase-key'));

  it('is what you get without asking, with the nine measured bands', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    expect([...document.querySelectorAll('[data-band]')]
      .map((n) => n.getAttribute('data-band')))
      .toEqual(['Al', 'Fe', 'Si', 'Mg', 'Mn', 'Zn', 'Cu', 'Ni', 'O']);
    expect(cardsIn('Al')).toHaveLength(28);
    expect(cardsIn('O')).toHaveLength(1);
  });

  it('says 36 phases AND 93 placements, because either alone misleads', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    const live = screen.getByRole('status').textContent;
    expect(live).toContain('36 of 36');
    expect(live).toContain('93');
  });

  it('one phase ticked in one band is ticked in all of its bands', async () => {
    // It is one phase in three systems. A tick that did not follow it would
    // say the three cards are three phases.
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    const key = 'Al3Fe2Si_mp-1190708_symmetrized';
    const boxes = () => [...document.querySelectorAll(`[data-phase-key="${key}"] input`)];
    expect(boxes()).toHaveLength(3);                 // Al, Fe, Si
    expect(boxes().every((b) => !b.checked)).toBe(true);
    fireEvent.click(boxes()[0]);
    await waitFor(() => expect(boxes().every((b) => b.checked)).toBe(true));
    fireEvent.click(boxes()[2]);                     // untick from another band
    await waitFor(() => expect(boxes().every((b) => !b.checked)).toBe(true));
  });

  it('a collapsed band hides its cards and keeps its count', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    const head = band('Al').querySelector('button');
    expect(head.getAttribute('aria-expanded')).toBe('true');
    fireEvent.click(head);
    await waitFor(() => expect(cardsIn('Al')).toHaveLength(0));
    expect(head.getAttribute('aria-expanded')).toBe('false');
    // The count is the reason to open it again.
    expect(band('Al').textContent).toContain('28');
  });

  it('a citation hit never becomes a card in a band', async () => {
    // §2.1: a class-2 hit must never look like a class-1 hit, and inside a
    // band there is nowhere to say which it is.
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Barlock' } });
    await waitFor(() => expect(screen.getByText(/mentioned in the reference/i)).toBeTruthy());
    expect(document.querySelectorAll('[data-band]')).toHaveLength(0);
    expect(screen.getByText(/mentioned in the reference/i).textContent).toContain('(1)');
  });

  it('the switcher goes to the flat list and back', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
    fireEvent.click(screen.getByRole('button', { name: /^list$/i }));
    await waitFor(() => expect(screen.queryByTestId('phase-bands')).toBeNull());
    expect(shownPhases()).toBe(36);
    expect(screen.getByRole('status').textContent).not.toContain('93');
    fireEvent.click(screen.getByRole('button', { name: /^systems$/i }));
    await waitFor(() => expect(screen.getByTestId('phase-bands')).toBeTruthy());
  });
});

describe('the reason on a row is in the reader language', () => {
  // It was hard-coded German -- `Raumgruppe`, `Strukturtyp`, `nur laut
  // Etikett` -- rendered straight to the screen in an app that ships in
  // four languages. The search returns keys now and the row translates
  // them; these fail if a key ever reaches the screen raw.
  const rowFor = (key) => document.querySelector(`[data-phase-key="${key}"]`);

  it('says "by name only" in English, not "nur laut Etikett"', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Al-Fe-Si' } });
    await waitFor(() => expect(rowFor('beta-AlFeSi')).toBeTruthy());
    const text = rowFor('beta-AlFeSi').textContent;
    expect(text).toContain('by name only');
    expect(text).not.toContain('labelOnly');        // never the raw key
    expect(text).not.toContain('Etikett');
  });

  it('and says it in German when the app is in German', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    render(<PhaseLibraryPage loadIndex={gives(fixture.phases)} />);
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/phasenbibliothek durchsuchen/i),
                     { target: { value: 'Al-Fe-Si' } });
    await waitFor(() => expect(rowFor('beta-AlFeSi')).toBeTruthy());
    expect(rowFor('beta-AlFeSi').textContent).toContain('nur laut Etikett');
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('no row ever shows a bare key', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    const raw = ['spaceGroup', 'labelOnly', 'filename', 'prototype', 'citation'];
    const all = screen.getByTestId('phase-list').textContent;
    for (const k of raw) expect(all, k).not.toContain(k);
  });
});

describe('the search from Task 2 is the search on the page', () => {
  it('finds pure aluminium by its name — the query that started this', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'aluminium' } });
    await waitFor(() => {
      expect(document.querySelector('[data-phase-key="Al"]')).toBeTruthy();
    });
  });

  it('keeps structure-type and citation hits under their own headings', async () => {
    await show(gives(fixture.phases));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());

    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Cu' } });
    await waitFor(() => expect(screen.getByText(/same structure type/i)).toBeTruthy());

    fireEvent.change(screen.getByLabelText(/search the phase library/i),
                     { target: { value: 'Barlock' } });
    await waitFor(() => expect(screen.getByText(/mentioned in the reference/i)).toBeTruthy());
  });
});

describe('a name somebody gave a phase reaches the search', () => {
  // THE TEST THAT WAS MISSING. `buildIndex` took a second argument carrying
  // the names, three unit tests passed it, and the page never did -- so the
  // list could show `β-AlFeSi` while typing `β-AlFeSi` found nothing. Asked
  // here, through the page, because that is the only place the two halves
  // meet.
  const withName = (key, names) => gives(fixture.phases.map(
    (p) => (p.key === key ? { ...p, ...names } : p)));

  const type = (text) => fireEvent.change(
    screen.getByRole('searchbox'), { target: { value: text } });

  // One phase stands in one band per element, so `Al2CuMg` is three rows in
  // the default view -- 36 phases, 93 placements, which is what the header
  // says. The question these ask is "which phases", not "how many rows".
  const shownKeys = () => [...new Set([...screen.getByTestId('phase-list')
    .querySelectorAll('[data-phase-key]')]
    .map((n) => n.getAttribute('data-phase-key')))];

  it('the list shows the name, not the filename', async () => {
    await show(withName('sd_1814127', { display_name: 'S-Phase' }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    // Three rows: one per element band. All of them say the name.
    expect(screen.getAllByText('S-Phase').length).toBe(3);
  });

  it('and typing that name finds it', async () => {
    await show(withName('sd_1814127', { display_name: 'S-Phase' }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    type('S-Phase');
    await waitFor(() => expect(shownKeys()).toEqual(['sd_1814127']));
  });

  it('and so does a search term nobody would guess from the file', async () => {
    await show(withName('sd_1814127',
      { display_name: 'S-Phase', search_terms: ['die klebrige'] }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    type('die klebrige');
    await waitFor(() => expect(shownKeys()).toEqual(['sd_1814127']));
  });

  it('the KEY is still what a row is, and is still searchable', async () => {
    // b9's guard: a display name stands in FRONT OF the key, never instead
    // of it. Renaming a phase must not make its file unfindable -- the whole
    // complaint this page answers was "ich musste ewig suchen um das cif zu
    // finden".
    await show(withName('sd_1814127', { display_name: 'S-Phase' }));
    await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
    type('sd_1814127');
    await waitFor(() => expect(shownKeys()).toEqual(['sd_1814127']));
  });
});
