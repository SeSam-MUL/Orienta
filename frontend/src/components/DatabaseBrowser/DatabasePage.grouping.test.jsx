// @vitest-environment jsdom
/**
 * Render-level tests for two of the three traps the task-10 brief named —
 * `databaseCollections.test.jsx` pins the maths (`databaseGrouping.js`)
 * directly; these mount the real components the brief actually asked for:
 * "mounts the real table, selects a row, causes a regroup" and "the badge
 * equals the rows across all groups".
 */
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('../../services/api', () => ({
  dbApi: {
    browse: vi.fn(),
    cacheStatus: vi.fn().mockResolvedValue({ data: { used_bytes: 0, max_bytes: 1, file_count: 0 } }),
  },
  h5Api: { open: vi.fn() },
  collectionsApi: {
    list: vi.fn(),
    putState: vi.fn().mockResolvedValue({ data: { success: true } }),
    create: vi.fn(), rename: vi.fn(), remove: vi.fn(), update: vi.fn(),
    addMembers: vi.fn(), removeMembers: vi.fn(), suggest: vi.fn(), applySuggest: vi.fn(),
  },
  settingsApi: { get: vi.fn().mockResolvedValue({ data: {} }) },
}));

import { dbApi, collectionsApi } from '../../services/api';
import useCollectionStore from '../../stores/useCollectionStore';
import { FileTable, TAB_DEFS } from './DatabasePage';
import DatabasePage from './DatabasePage';

afterEach(() => cleanup());

// `useCollectionStore` is a real, module-level singleton (not mocked) — its
// `collapsed` map survives `cleanup()` (that only unmounts the React tree),
// so the "collapse Matrix" test above would otherwise leave Matrix collapsed
// for every test that runs after it in this file.
beforeEach(() => {
  useCollectionStore.setState({
    collapsed: {},
    data: { collections: [], unassigned: [], problems: [], state: {} },
  });
});

// ---------------------------------------------------------------------------
// Gap 1: selectedRow resets on a REAL regroup.
// `regroup()` in databaseCollections.test.jsx is a reconstruction of the
// rule, not a call into the shipped component — it would still pass with
// DatabasePage.jsx's `useEffect(() => setSelectedRow(-1), [orderSignature])`
// deleted. This mounts the real `FileTable` and drives it through props, the
// way its actual parent (`DatabasePage`) would on a real regroup.
// ---------------------------------------------------------------------------
const CIF_TAB = TAB_DEFS.find((t) => t.id === 'cif');

function fileTableProps(overrides = {}) {
  return {
    tabDef: CIF_TAB,
    entries: [{ name: 'Ni.cif', file_type: 'cif' }, { name: 'Al.cif', file_type: 'cif' }],
    onRowClick: () => {},
    loading: false,
    selectedFiles: new Set(),
    onToggleSelect: () => {},
    onSelectAll: () => {},
    onDownload: () => {},
    collections: [{ name: 'Matrix', members: [{ key: 'Al' }] }],
    ...overrides,
  };
}

const rowFor = (text) => screen.getByText(text).closest('tr');

describe('FileTable — selectedRow resets on a real regroup', () => {
  it('clears the highlighted row when the row order actually changes', () => {
    const { rerender } = render(<FileTable {...fileTableProps()} />);

    // Grouped order: Matrix claims "Al" -> [Al.cif, Ni.cif]. Ni.cif sits at
    // index 1. Select it.
    fireEvent.click(rowFor('Ni.cif'));
    expect(rowFor('Ni.cif')).toHaveAttribute('aria-selected', 'true');
    expect(rowFor('Al.cif')).toHaveAttribute('aria-selected', 'false');

    // Regroup: Matrix no longer claims anything -> flat order is the raw
    // `entries` order, [Ni.cif, Al.cif] — Ni.cif's OWN index moves from 1 to
    // 0. If the highlight followed the index instead of resetting, Al.cif
    // (now at index 1) would incorrectly light up instead.
    rerender(<FileTable {...fileTableProps({ collections: [] })} />);

    expect(rowFor('Ni.cif')).toHaveAttribute('aria-selected', 'false');
    expect(rowFor('Al.cif')).toHaveAttribute('aria-selected', 'false');
  });
});

// ---------------------------------------------------------------------------
// Gap 2: the tab badge counts the same rows the table shows — across every
// group, expanded or collapsed.
// ---------------------------------------------------------------------------
const CIF_ENTRIES = [
  { name: 'Al.cif', filename: 'Al.cif', file_type: 'CIF', material: 'Al', location: 'local', size_bytes: 1 },
  { name: 'Ni.cif', filename: 'Ni.cif', file_type: 'CIF', material: 'Ni', location: 'local', size_bytes: 2 },
  { name: 'WC.cif', filename: 'WC.cif', file_type: 'CIF', material: 'WC', location: 'local', size_bytes: 3 },
];

// The exact filename from the review: derived key `Ni (Ni) [cF4] {20kV}`,
// nothing close to the library key `Ni`.
const SHT_ENTRY = {
  name: 'Ni (Ni) [cF4] {20kV}.sht', filename: 'Ni (Ni) [cF4] {20kV}.sht',
  file_type: 'sht', material: 'Ni', location: 'local', size_bytes: 4,
};

const COLLECTIONS_BODY = {
  collections: [{ name: 'Matrix', parent: null, exclusive: true, member_count: 1, effective_member_count: 1,
                  members: [{ key: 'Al', present: true }] }],
  unassigned: [{ key: 'Ni' }, { key: 'WC' }],
  problems: [], state: { active: null, hidden: [] },
};

beforeEach(() => {
  vi.clearAllMocks();
  dbApi.browse.mockResolvedValue({ data: { entries: [...CIF_ENTRIES, SHT_ENTRY] } });
  dbApi.cacheStatus.mockResolvedValue({ data: { used_bytes: 0, max_bytes: 1, file_count: 0 } });
  collectionsApi.list.mockResolvedValue({ data: COLLECTIONS_BODY });
});

async function openCifTab() {
  render(<DatabasePage isActive />);
  await waitFor(() => expect(dbApi.browse).toHaveBeenCalled());
  fireEvent.click(await screen.findByRole('tab', { name: /^CIF/ }));
  await screen.findByText('Al.cif');
}

/** Data rows in the visible table body — group-header rows carry a single
 *  `colspan` cell and are excluded. */
function visibleDataRows() {
  const tbody = document.querySelector('tbody');
  return within(tbody).getAllByRole('row').filter((tr) => !tr.querySelector('td[colspan]'));
}

describe('DatabasePage — the CIF tab badge equals the rows the table shows', () => {
  it('with nothing collapsed, the badge equals every group\'s rows summed', async () => {
    await openCifTab();
    const tab = screen.getByRole('tab', { name: /^CIF/ });
    expect(tab).toHaveTextContent('CIF (3)');
    expect(visibleDataRows()).toHaveLength(3);
  });

  it('DELIBERATE: a collapsed group hides its rows, but the badge still counts them', async () => {
    // Correct, not a bug — the badge counts what the tab HOLDS, not what is
    // scrolled/expanded into view. Pinned so a future "fix" that makes the
    // badge track only expanded rows is a deliberate, visible decision.
    await openCifTab();
    expect(screen.getByRole('tab', { name: /^CIF/ })).toHaveTextContent('CIF (3)');
    expect(visibleDataRows()).toHaveLength(3);

    // Collapse "Matrix" (holds Al.cif) — the first collapse triangle, since
    // `groupEntriesByCollection` orders Matrix before the trailing
    // "Unassigned" bucket.
    fireEvent.click(screen.getAllByTitle('Collapse this collection.')[0]);

    expect(visibleDataRows()).toHaveLength(2);                          // Al.cif's row is hidden
    expect(screen.getByRole('tab', { name: /^CIF/ })).toHaveTextContent('CIF (3)');   // badge unchanged
  });
});

// ---------------------------------------------------------------------------
// CRITICAL fix, wired: selecting a row from the SHT (or Master/MC/
// Dictionary) tab must not offer a "move to collection" that silently writes
// a permanently-dead member. `databaseCollections.test.jsx` proves
// `movableMemberKeys` refuses such a row in isolation; this proves
// `DatabasePage.jsx`'s own control actually consults it.
// ---------------------------------------------------------------------------
describe('DatabasePage — "move to collection" refuses a non-phase row', () => {
  it('disables the control, with a category-error reason, for an SHT-only selection', async () => {
    render(<DatabasePage isActive />);
    await waitFor(() => expect(dbApi.browse).toHaveBeenCalled());
    // SHT is the default tab.
    await screen.findByText('Ni (Ni) [cF4] {20kV}.sht');

    fireEvent.click(screen.getByTitle('Select this file for Upload to server or Delete Selected.'));

    const move = await screen.findByTitle(
      'Not a phase — file the CIF (or XTAL); derived files (SHT, master, MC, dictionary) follow automatically.');
    expect(move.tagName).toBe('SELECT');
    expect(move).toBeDisabled();
    expect(within(move).getByRole('option', { selected: true })).toHaveTextContent(
      'Not a phase — file the CIF (or XTAL); derived files (SHT, master, MC, dictionary) follow automatically.');
    expect(collectionsApi.addMembers).not.toHaveBeenCalled();
  });

  it('the same phase is movable once selected from the CIF tab', async () => {
    await openCifTab();
    // Grouped order puts Al.cif (Matrix's member) first — the row-select
    // checkbox tooltip is shared by every row, so pick it by position.
    fireEvent.click(screen.getAllByTitle('Select this file for Upload to server or Delete Selected.')[0]);

    const move = await screen.findByTitle(
      'Move the selected phases into this collection. A phase leaves any other exclusive collection it was filed in — moving into the working set does not.');
    expect(move).not.toBeDisabled();

    fireEvent.change(move, { target: { value: 'Matrix' } });
    await waitFor(() => expect(collectionsApi.addMembers).toHaveBeenCalledWith('Matrix', ['Al']));
  });
});
