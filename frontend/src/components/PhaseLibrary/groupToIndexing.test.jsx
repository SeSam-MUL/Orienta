// @vitest-environment jsdom
/**
 * A group in the library, used for a run.
 *
 * Sebastian, on the collections this replaces: "ich will sie aus jeder
 * Anzeige heraus benutzen". The way to honour that was NOT a second path.
 * Indexing and the phase tester already read the ACTIVE collection through
 * `useCollectionStore` and `collectionsApi.resolve`, and that is where the
 * per-method availability is worked out (`Al -> no_master` for Dictionary).
 * A parallel mechanism could disagree with that one, and then two screens
 * would offer different phases for the same run.
 *
 * So these tests are about a seam, not a feature: that the library feeds
 * the mechanism that is already there, with the right argument, and that a
 * change made here reaches the pickers that read the same folder.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import {
  render, screen, cleanup, fireEvent, act, waitFor, within,
} from '@testing-library/react';

const setActive = vi.fn(() => Promise.resolve());
const load = vi.fn(() => Promise.resolve());
// `active` is set, because the chip and the "use the whole library" button
// both key on it -- a mock with an empty state would render neither and
// the tests would pass over a screen that shows nothing.
const DATA = { collections: [], state: { active: 'Al systems', hidden: [] } };
vi.mock('../../stores/useCollectionStore', () => {
  const store = (sel) => (sel ? sel({ data: DATA, loading: false }) : { data: DATA });
  store.getState = () => ({ setActive, load, data: DATA });
  return { default: store };
});

const groupStore = {
  kind: 'test',
  load: vi.fn(() => Promise.resolve({
    groups: [{ id: 'Al_systems', name: 'Al systems', parent: null,
      members: ['Al', 'Al13Fe4'] },
    { id: 'Empty_one', name: 'Empty one', parent: null, members: [] }],
    journal: [] })),
  apply: vi.fn(() => Promise.resolve({ saved: true })),
};
vi.mock('./groupPersistence', () => ({
  flags: { backendReady: true },
  STORAGE_KEY: 'test',
  activeStore: () => groupStore,
}));

const i18n = (await import('../../i18n')).default;
const fixture = (await import('./__fixtures__/library.json')).default;
const { default: PhaseLibraryPage } = await import('./PhaseLibraryPage');
const { default: useGroups } = await import('./useGroups');

const gives = () => Promise.resolve(
  { phases: fixture.phases, unassigned_masters: fixture.unassigned_masters });

async function show(onNavigate = vi.fn()) {
  await act(async () => { await i18n.changeLanguage('en'); });
  useGroups.setState({ groups: [], journal: [], past: [], loaded: false,
    saveError: null, unavailable: false, author: 'seb' });
  render(<PhaseLibraryPage loadIndex={gives} onNavigate={onNavigate} />);
  await waitFor(() => expect(screen.getByTestId('phase-list')).toBeTruthy());
  await waitFor(() => expect(screen.getByText('Al systems')).toBeTruthy());
  return onNavigate;
}

const menu = (name) => {
  fireEvent.click(screen.getByLabelText(`Actions for ${name}`));
  return screen.getByRole('menu', { name: `Actions for ${name}` });
};

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('using a group for the next run', () => {
  it('makes it the ACTIVE collection, by its id', async () => {
    // The id and not the name: `active` goes through `resolve_ref`, which
    // takes either and STORES the id, so a later rename does not break it.
    await show();
    menu('Al systems');
    fireEvent.click(screen.getByText('Use this group for the next run'));
    await waitFor(() => expect(setActive).toHaveBeenCalledWith('Al_systems'));
  });

  it('and takes the user to Indexing, where the choice has an effect', async () => {
    const go = await show();
    menu('Al systems');
    fireEvent.click(screen.getByText('Use this group for the next run'));
    await waitFor(() => expect(go).toHaveBeenCalledWith('indexing'));
  });

  it('is offered but disabled for a group with nothing in it', async () => {
    // Disabled rather than hidden: a menu whose contents change between
    // visits is a menu people stop trusting.
    await show();
    const m = menu('Empty one');
    const item = within(m).getByText(
      'Use this group for the next run (it is empty)');
    expect(item.disabled).toBe(true);
    fireEvent.click(item);
    expect(setActive).not.toHaveBeenCalled();
  });

  it('survives a page that was given nowhere to navigate', async () => {
    // Half the tests in this feature render the page without `onNavigate`,
    // and a card's deep link once threw for exactly that reason.
    await act(async () => { await i18n.changeLanguage('en'); });
    useGroups.setState({ groups: [], journal: [], past: [], loaded: false,
      saveError: null, unsaved: [], unavailable: false, author: 'seb' });
    render(<PhaseLibraryPage loadIndex={gives} />);
    await waitFor(() => expect(screen.getByText('Al systems')).toBeTruthy());
    menu('Al systems');
    expect(() => fireEvent.click(
      screen.getByText('Use this group for the next run'))).not.toThrow();
    await waitFor(() => expect(setActive).toHaveBeenCalled());
  });
});

describe('a change here reaches the pickers that read the same folder', () => {
  // Two stores over one endpoint while the old collections UI is still
  // standing. The hazard is not the duplication, it is the staleness: a
  // phase filed here that Indexing does not see until it happens to poll.
  it('re-reads the collections after a group changes', async () => {
    await show();
    load.mockClear();
    await act(async () => {
      useGroups.getState().addMany('Al_systems', ['Si']);
    });
    await waitFor(() => expect(load).toHaveBeenCalled());
  });

  it('but not merely because the page rendered', async () => {
    await show();
    // Opening the page must not make every other picker re-fetch.
    expect(load).not.toHaveBeenCalled();
  });
});

describe('the way back to the whole library', () => {
  // There was none. `setActive` had exactly one caller and it always
  // passed a group; the dropdown that used to offer "all phases" is gone;
  // and the blocked-Start message told people to "clear it in the Phase
  // Library" -- an instruction nothing on any screen could carry out. With
  // one group in the library and a stale active reference, every picker
  // was empty and every run blocked, with no action available at all.
  it('is offered when a group is narrowing runs', async () => {
    await show();
    expect(screen.getByTestId('stop-using-group')).toBeTruthy();
  });

  it('and clears the active group', async () => {
    await show();
    fireEvent.click(screen.getByTestId('stop-using-group'));
    await waitFor(() => expect(setActive).toHaveBeenCalledWith(null));
  });

  it('is not offered when nothing is narrowing anything', async () => {
    // Nothing to undo, so nothing to offer. The mocked store is mutated
    // rather than replaced: the module-level mock is what the page reads.
    DATA.state.active = null;
    try {
      await show();
      expect(screen.queryByTestId('stop-using-group')).toBe(null);
    } finally {
      DATA.state.active = 'Al systems';
    }
  });
});
