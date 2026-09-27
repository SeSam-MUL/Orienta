// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('../../services/api', () => ({
  collectionsApi: {
    list: vi.fn(),
    putState: vi.fn().mockResolvedValue({ data: { success: true } }),
  },
}));

import { collectionsApi } from '../../services/api';
import CollectionPicker from './CollectionPicker';
import useCollectionStore from '../../stores/useCollectionStore';

const BODY = {
  collections: [
    { name: 'Intermetallics in Al', parent: null, hidden: false,
      member_count: 3, effective_member_count: 3, exclusive: true,
      members: [{ key: 'Al13Fe4' }, { key: 'Al6Fe' }, { key: 'Al7FeCu2' }] },
    { name: 'Hartmetalle', parent: null, hidden: true,
      member_count: 1, effective_member_count: 1, exclusive: true,
      members: [{ key: 'WC' }] },
  ],
  unassigned: [{ key: 'Ni' }],
  problems: [],
  state: { active: null, hidden: ['Hartmetalle'] },
};

beforeEach(() => {
  collectionsApi.list.mockResolvedValue({ data: BODY });
});

// This project's vitest config does not set `test.globals: true`, so
// @testing-library/react's own auto-cleanup (which checks for a bare global
// `afterEach`) never fires here — every other multi-render test file in this
// codebase (e.g. EDS/ExportDialog.test.jsx) imports `cleanup` explicitly for
// the same reason. Without it, four `render()` calls in this file leave four
// stacked buttons in `document.body` and every findByTestId after the first
// throws "multiple elements".
afterEach(() => {
  cleanup();
});

describe('CollectionPicker', () => {
  it('offers "all phases" and every collection that is not hidden', async () => {
    render(<CollectionPicker />);
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    expect(screen.getByText(/Intermetallics in Al/)).toBeInTheDocument();
    expect(screen.queryByText(/^Hartmetalle$/)).not.toBeInTheDocument();
  });

  it('keeps a standing notice while something is hidden, and can undo it', async () => {
    render(<CollectionPicker />);
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    const notice = screen.getByTestId('collection-hidden-notice');
    expect(notice).toHaveTextContent('1');
    fireEvent.click(notice);
    await waitFor(() => expect(collectionsApi.putState).toHaveBeenCalledWith(
      expect.objectContaining({ hidden: [] }),
    ));
  });

  it('choosing a collection writes it to the server, not just to the tab', async () => {
    render(<CollectionPicker />);
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    fireEvent.click(screen.getByText(/Intermetallics in Al/));
    await waitFor(() => expect(collectionsApi.putState).toHaveBeenCalledWith(
      expect.objectContaining({ active: 'Intermetallics in Al' }),
    ));
  });

  // The store's initial `data` is an empty-but-VALID shape and only
  // `collapsed` is persisted, so on every single start there is a window in
  // which the menu is empty and says so by saying nothing. Sebastian read
  // exactly that as the app having lost his collections: "beim starten der
  // app laden die phasen nicht sofort ... das kann zu missverstaendnissen
  // fuehren". The store has tracked `loading` all along; nobody read it.
  it('says it is loading rather than showing an empty menu', async () => {
    // The store is module-level and the tests above have already filled it.
    // This test is about the FIRST start, so put it back to what a fresh app
    // holds -- and note that the guard being tested is exactly
    // `loading && collections.length === 0`: a background refresh with
    // collections already on screen must NOT flash a loading line.
    useCollectionStore.setState({
      data: { collections: [], unassigned: [], problems: [], state: {} },
      loading: false,
    });
    let release;
    collectionsApi.list.mockReturnValueOnce(new Promise((res) => { release = res; }));
    render(<CollectionPicker />);
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    expect(screen.getByText('Loading collections…')).toBeInTheDocument();
    expect(screen.queryByText(/Intermetallics in Al/)).not.toBeInTheDocument();

    release({ data: BODY });
    await waitFor(() => expect(screen.getByText(/Intermetallics in Al/)).toBeInTheDocument());
    expect(screen.queryByText('Loading collections…')).not.toBeInTheDocument();
  });

  // The other half of `loading && collections.length === 0`, which the test
  // above only claimed in a comment: a background refresh with collections
  // already on screen must NOT flash a loading line.
  it('does not flash "loading" while a refresh runs over a full menu', async () => {
    useCollectionStore.setState({ data: BODY, loading: true, loadFailed: false });
    render(<CollectionPicker />);
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    expect(screen.getByText(/Intermetallics in Al/)).toBeInTheDocument();
    expect(screen.queryByText('Loading collections…')).not.toBeInTheDocument();
  });

  // First start with the backend down: "keep what is on screen" keeps the
  // empty default, so without this the menu presents "all phases (0)" as a
  // fact -- the same sentence this whole change was written against.
  it('says the library was unreachable instead of showing an empty menu', async () => {
    // The mount effect calls load(), so the rejection has to come from the
    // API mock -- setting loadFailed by hand would be overwritten a tick later
    // by the beforeEach mock resolving with BODY.
    collectionsApi.list.mockRejectedValueOnce(new Error('backend down'));
    useCollectionStore.setState({
      data: { collections: [], unassigned: [], problems: [], state: {} },
      loading: false, loadFailed: false,
    });
    render(<CollectionPicker />);
    await waitFor(() => expect(useCollectionStore.getState().loadFailed).toBe(true));
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    expect(screen.getByText('Could not reach the phase library')).toBeInTheDocument();
    expect(screen.getByText('Try again')).toBeInTheDocument();
  });

  it('a failed load leaves the picker on "all phases" rather than empty', async () => {
    collectionsApi.list.mockRejectedValueOnce(new Error('backend down'));
    render(<CollectionPicker />);
    expect(await screen.findByTestId('collection-picker-button')).toBeInTheDocument();
  });
});

/**
 * I3 (whole-branch review): `state.active` can name a collection that
 * `load_all()` no longer returns (its file went missing or failed to parse
 * between saves — `bad_schema`/broken JSON — while `_local.json` still calls
 * it active). Every picker's `activeKeySet` treats an unresolved name as
 * "no filter" (collectionFilter.js: "renamed or deleted: filter nothing"),
 * so EVERY phase picker in the app silently widens to the whole library —
 * and before this fix, the toolbar button still read "Collection: Ghost" as
 * if the narrowing were still in effect. These tests pin that the button
 * now says so instead of lying.
 */
describe('CollectionPicker — active collection resolves to nothing (I3)', () => {
  const DANGLING_BODY = {
    collections: [
      { name: 'Matrix', parent: null, hidden: false, exclusive: true,
        member_count: 1, effective_member_count: 1, members: [{ key: 'Al' }] },
    ],
    unassigned: [],
    problems: [],
    state: { active: 'Ghost', hidden: [] },
  };

  it('marks the toolbar button instead of quietly showing "Ghost" as if it still filtered', async () => {
    collectionsApi.list.mockResolvedValue({ data: DANGLING_BODY });
    render(<CollectionPicker />);
    const button = await screen.findByTestId('collection-picker-button');
    // Still names the dangling collection (so the user can tell WHICH one
    // broke), but now visibly flagged rather than looking like an ordinary,
    // successfully-filtering active collection.
    expect(button).toHaveTextContent('Ghost');
    expect(button).toHaveTextContent('⚠');
    expect(button.title).toMatch(/Ghost/);
  });

  it('does not flag a collection that genuinely resolves', async () => {
    collectionsApi.list.mockResolvedValue({
      data: { ...DANGLING_BODY, state: { active: 'Matrix', hidden: [] } },
    });
    render(<CollectionPicker />);
    const button = await screen.findByTestId('collection-picker-button');
    expect(button).toHaveTextContent('Matrix');
    expect(button).not.toHaveTextContent('⚠');
  });

  it('shows a standing notice in the dropdown, and clicking it switches to "All phases"', async () => {
    collectionsApi.list.mockResolvedValue({ data: DANGLING_BODY });
    render(<CollectionPicker />);
    fireEvent.click(await screen.findByTestId('collection-picker-button'));
    const notice = screen.getByTestId('collection-active-missing-notice');
    expect(notice).toHaveTextContent('Ghost');
    fireEvent.click(notice);
    await waitFor(() => expect(collectionsApi.putState).toHaveBeenCalledWith(
      expect.objectContaining({ active: null }),
    ));
  });

  it('no active name at all is not treated as "missing" — this is the ordinary, unfiltered state', async () => {
    collectionsApi.list.mockResolvedValue({
      data: { ...DANGLING_BODY, state: { active: null, hidden: [] } },
    });
    render(<CollectionPicker />);
    const button = await screen.findByTestId('collection-picker-button');
    expect(button).not.toHaveTextContent('⚠');
    fireEvent.click(button);
    expect(screen.queryByTestId('collection-active-missing-notice')).not.toBeInTheDocument();
  });
});
