// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import NameEditor, { parseTerms } from './NameEditor';
import useGroups from './useGroups';

afterEach(() => { cleanup(); useGroups.setState({ author: '' }); });

const card = (over = {}) => ({
  key: 'sd_1814127', formula_label: 'Al2CuMg',
  display_name: null, search_terms: [], name_author: null, name_updated: null,
  ...over,
});

function show(over = {}, cardOver = {}) {
  const save = vi.fn(() => Promise.resolve({
    key: 'sd_1814127', display_name: 'S-Phase', search_terms: ['s phase'],
    author: 'seb', updated: '2026-09-27T18:00:00Z',
  }));
  const onSaved = vi.fn();
  render(<NameEditor card={card(cardOver)} save={save} onSaved={onSaved} {...over} />);
  return { save, onSaved };
}

const open = () => fireEvent.click(screen.getByTestId('name-editor-open'));

describe('reading terms people actually type', () => {
  it('splits on commas and drops the spaces and the blanks', () => {
    expect(parseTerms(' s phase , s-phase ,, Al2CuMg ')).toEqual(
      ['s phase', 's-phase', 'Al2CuMg']);
  });

  it('a single term is a term', () => {
    expect(parseTerms('S-Phase')).toEqual(['S-Phase']);
  });

  it('nothing typed is no terms, not one empty one', () => {
    expect(parseTerms('')).toEqual([]);
    expect(parseTerms('   ,  ,')).toEqual([]);
    expect(parseTerms(null)).toEqual([]);
  });
});

describe('the invitation', () => {
  it('offers to name a phase that has no name', () => {
    show();
    expect(screen.getByTestId('name-editor-open').textContent)
      .toBe('Give this phase a name…');
  });

  it('offers to change one that has', () => {
    show({}, { display_name: 'S-Phase' });
    expect(screen.getByTestId('name-editor-open').textContent)
      .toBe('Change this name…');
  });

  it('says who named it and when, because the library is shared', () => {
    show({}, { display_name: 'S-Phase', name_author: 'irmgard',
      name_updated: '2026-09-20T08:11:00Z' });
    expect(screen.getByTestId('name-author').textContent)
      .toBe('named by irmgard, 2026-09-20');
  });

  it('says nothing about an author when nobody signed it', () => {
    show({}, { display_name: 'S-Phase' });
    expect(screen.queryByTestId('name-author')).toBe(null);
  });
});

describe('naming', () => {
  it('sends the name, the terms and who typed them', async () => {
    const { save } = show();
    open();
    fireEvent.change(screen.getByLabelText('Name'),
      { target: { value: ' S-Phase ' } });
    fireEvent.change(screen.getByLabelText('Also find by'),
      { target: { value: 's phase, Al2CuMg' } });
    fireEvent.change(screen.getByLabelText('Your name'),
      { target: { value: 'seb' } });
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(save).toHaveBeenCalledWith({
      key: 'sd_1814127',
      displayName: 'S-Phase',
      searchTerms: ['s phase', 'Al2CuMg'],
      author: 'seb',
    }));
  });

  it('a name cleared to nothing is sent as the EMPTY STRING, which clears it', async () => {
    // THIS TEST USED TO ASSERT `null`, AND WAS NAMED "sent as nothing,
    // which is how you undo it". Both halves were wrong against the
    // service: `null` means "leave unchanged" there (`phase_synonyms.py`:
    // `if display_name is not None:`, pinned by
    // `tests/test_phase_synonyms.py`), the empty string is what clears.
    // Measured against the real service before the fix: emptying the field
    // and saving left the name in place, and the editor then repopulated
    // the box with it, so the screen argued with the user. The only
    // documented way to take a name away could not work, and this test
    // said it did -- against a stubbed save, which never saw the meaning.
    const { save } = show({}, { display_name: 'S-Phase' });
    open();
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: '  ' } });
    fireEvent.submit(screen.getByTestId('name-editor'));
    // The send is a microtask behind the submit; asserting straight away
    // read an empty mock and passed for the wrong reason on the first run.
    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(save.mock.calls[0][0].displayName).toBe('');
  });

  it('hands the STORED record on, not the text that was typed', async () => {
    // The store trims, drops blanks and records the author, so echoing the
    // form back would put a name on screen that the library does not have.
    const { onSaved } = show();
    open();
    fireEvent.change(screen.getByLabelText('Name'),
      { target: { value: 'whatever' } });
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({ display_name: 'S-Phase', search_terms: ['s phase'] })));
  });

  it('closes on success and shows what was stored', async () => {
    show();
    open();
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(screen.queryByTestId('name-editor')).toBe(null));
    expect(screen.getByTestId('name-editor-open')).toBeTruthy();
  });

  it('keeps the author for the next phase, so it is typed once', async () => {
    const { save } = show();
    open();
    fireEvent.change(screen.getByLabelText('Your name'), { target: { value: 'seb' } });
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(useGroups.getState().author).toBe('seb');
  });

  it('says so, and stays open, when the store refused', async () => {
    // The store lives in the library folder, which may be read-only or on a
    // share that is not there. A name that was not written is worse than one
    // that was refused: the screen would go on showing it.
    const save = vi.fn(() => Promise.reject({
      response: { data: { detail: 'Could not save the name: read-only' } } }));
    render(<NameEditor card={card()} save={save} />);
    open();
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(screen.getByTestId('name-problem').textContent)
      .toMatch(/read-only/));
    expect(screen.getByTestId('name-editor')).toBeTruthy();
  });

  it('names a plain network failure too, rather than going quiet', async () => {
    const save = vi.fn(() => Promise.reject(new Error('Network Error')));
    render(<NameEditor card={card()} save={save} />);
    open();
    fireEvent.submit(screen.getByTestId('name-editor'));
    await waitFor(() => expect(screen.getByTestId('name-problem').textContent)
      .toMatch(/Network Error/));
  });

  it('cancel writes nothing', () => {
    const { save } = show();
    open();
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'X' } });
    fireEvent.click(screen.getByText('Cancel'));
    expect(save).not.toHaveBeenCalled();
    expect(screen.queryByTestId('name-editor')).toBe(null);
  });
});

describe('the key is still the key', () => {
  it('the form says so, with the key in it', () => {
    show();
    open();
    expect(screen.getByText(/sd_1814127 still finds this phase/)).toBeTruthy();
  });
});
