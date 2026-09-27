// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('../../services/api', () => ({
  collectionsApi: {
    list: vi.fn(),
    create: vi.fn().mockResolvedValue({ data: { success: true } }),
    rename: vi.fn().mockResolvedValue({ data: { success: true } }),
    remove: vi.fn().mockResolvedValue({ data: { success: true } }),
    update: vi.fn().mockResolvedValue({ data: { success: true } }),
    addMembers: vi.fn().mockResolvedValue({ data: { success: true } }),
    removeMembers: vi.fn().mockResolvedValue({ data: { success: true } }),
    suggest: vi.fn(),
    applySuggest: vi.fn(),
  },
}));

import { collectionsApi } from '../../services/api';
import useCollectionStore from '../../stores/useCollectionStore';
import CollectionManager from './CollectionManager';

const BODY = {
  collections: [
    {
      name: 'Matrix', parent: null, exclusive: true, hidden: false,
      member_count: 2, effective_member_count: 2,
      members: [
        { key: 'Al', label: 'Al (Al)', formula: 'Al', present: true },
        { key: 'Ghost', label: 'Ghost', formula: 'Ghost', present: false },
      ],
    },
    {
      name: 'Intermetallics', parent: null, exclusive: true, hidden: false,
      member_count: 1, effective_member_count: 1,
      members: [{ key: 'Al6Fe', label: 'Al6Fe', formula: 'Al6Fe', present: true }],
    },
  ],
  unassigned: [{ key: 'Ni', label: 'Ni (Ni)', formula: 'Ni' }],
  problems: [],
  state: { active: null, hidden: [] },
};

// This project's vitest config does not set `test.globals: true`, so
// @testing-library/react's own auto-cleanup never fires here — see
// CollectionPicker.test.jsx for the same note.
afterEach(() => cleanup());

beforeEach(() => {
  vi.clearAllMocks();
  collectionsApi.list.mockResolvedValue({ data: BODY });
  useCollectionStore.setState({ data: { collections: [], unassigned: [], problems: [], state: {} } });
});

async function openManager() {
  render(<CollectionManager onClose={() => {}} />);
  await waitFor(() => expect(collectionsApi.list).toHaveBeenCalled());
  // Not `findByText('Matrix')`: every collection's name ALSO appears as an
  // `<option>` in the "parent" select in the create form, so a plain text
  // query is ambiguous. "Expand" is unique to a card.
  await waitFor(() => expect(screen.getAllByTitle('Expand')).toHaveLength(2));
}

describe('CollectionManager — reading the store', () => {
  it('lists every top-level collection with its member count', async () => {
    await openManager();
    // `{ selector: 'span' }`: the name also appears as an `<option>` in the
    // create form's "parent" select, which a plain text query would match too.
    expect(screen.getByText('Matrix', { selector: 'span' })).toBeInTheDocument();
    expect(screen.getByText('Intermetallics', { selector: 'span' })).toBeInTheDocument();
    expect(screen.getByText('(2)')).toBeInTheDocument();   // Matrix
    expect(screen.getByText('(1)')).toBeInTheDocument();   // Intermetallics
  });

  it('a member the library has lost renders struck through, and only there', async () => {
    // "This is the only place it is rendered" — pinned by checking the
    // missing-count text exists exactly once in the whole document.
    await openManager();
    fireEvent.click(screen.getAllByTitle('Expand')[0]);   // Matrix
    const ghost = screen.getByText('Ghost');
    expect(ghost).toHaveStyle({ textDecoration: 'line-through' });
    expect(screen.getByText('not in library')).toBeInTheDocument();
    expect(screen.getAllByText(/phase is not in the library/)).toHaveLength(1);
  });

  it('a present member is NOT struck through', async () => {
    await openManager();
    fireEvent.click(screen.getAllByTitle('Expand')[0]);
    expect(screen.getByText('Al (Al)')).not.toHaveStyle({ textDecoration: 'line-through' });
  });

  it('lists unassigned phases with an "add to" control', async () => {
    await openManager();
    expect(screen.getByText('Unassigned (1)')).toBeInTheDocument();
    expect(screen.getByText('Ni (Ni)')).toBeInTheDocument();
  });

  it('with no collections at all, says so instead of showing an empty list', async () => {
    collectionsApi.list.mockResolvedValue({
      data: { collections: [], unassigned: [], problems: [], state: {} },
    });
    render(<CollectionManager onClose={() => {}} />);
    await waitFor(() => expect(collectionsApi.list).toHaveBeenCalled());
    expect(await screen.findByText(/No collections yet/)).toBeInTheDocument();
  });
});

describe('CollectionManager — create', () => {
  it('creates a top-level collection from the name field', async () => {
    await openManager();
    fireEvent.change(screen.getByPlaceholderText('New collection name…'), { target: { value: 'Carbides' } });
    fireEvent.click(screen.getByText('Create'));
    await waitFor(() => expect(collectionsApi.create).toHaveBeenCalledWith({ name: 'Carbides', parent: null }));
  });

  it('the Create button is disabled until a name is typed', async () => {
    await openManager();
    expect(screen.getByText('Create')).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText('New collection name…'), { target: { value: 'X' } });
    expect(screen.getByText('Create')).not.toBeDisabled();
  });

  it('offers only top-level collections as a parent (one level only) — a PARENT WITH A CHILD IS STILL OFFERED', async () => {
    // I4: the one-level rule (phase_collections.py#save) forbids a
    // grandchild — a `parent` whose own `parent` is set, or a collection
    // that already has children becoming a CHILD itself. It does NOT limit
    // a parent to one child. "Child" (itself a child: `parent: 'Parent'`)
    // must be excluded; "Parent" (a top-level collection that HAS a child)
    // must still be offered, so a second sibling can be filed under it —
    // spec §4.2's own worked example is exactly this: "Fe-haltig" and
    // "Mg-haltig" both under "Intermetallics in Al".
    collectionsApi.list.mockResolvedValue({
      data: {
        collections: [
          { name: 'Parent', parent: null, exclusive: true, members: [] },
          { name: 'Child', parent: 'Parent', exclusive: true, members: [] },
          { name: 'Standalone', parent: null, exclusive: true, members: [] },
        ],
        unassigned: [], problems: [], state: {},
      },
    });
    render(<CollectionManager onClose={() => {}} />);
    await waitFor(() => expect(collectionsApi.list).toHaveBeenCalled());
    const parentSelect = await screen.findByTitle('Parent (optional)');
    // Placeholder + "Parent" + "Standalone" — "Child" is the only exclusion.
    await waitFor(() => expect(within(parentSelect).getAllByRole('option')).toHaveLength(3));
    const options = within(parentSelect).getAllByRole('option').map((o) => o.textContent);
    expect(options).toContain('Parent');
    expect(options).toContain('Standalone');
    expect(options).not.toContain('Child');
  });

  it('I4 REGRESSION: two siblings can be created under the same parent', async () => {
    // The exact scenario the review report says was unreachable: with one
    // child ("Fe-haltig") already filed under "Intermetallics in Al", the
    // parent dropdown offered NO parent at all, so a second sibling
    // ("Mg-haltig") could never be created through the UI.
    collectionsApi.list.mockResolvedValue({
      data: {
        collections: [
          { name: 'Intermetallics in Al', parent: null, exclusive: true, members: [] },
          { name: 'Fe-haltig', parent: 'Intermetallics in Al', exclusive: true, members: [] },
        ],
        unassigned: [], problems: [], state: {},
      },
    });
    render(<CollectionManager onClose={() => {}} />);
    await waitFor(() => expect(collectionsApi.list).toHaveBeenCalled());

    fireEvent.change(screen.getByPlaceholderText('New collection name…'),
      { target: { value: 'Mg-haltig' } });
    fireEvent.change(screen.getByTitle('Parent (optional)'),
      { target: { value: 'Intermetallics in Al' } });
    fireEvent.click(screen.getByText('Create'));
    await waitFor(() => expect(collectionsApi.create).toHaveBeenCalledWith(
      { name: 'Mg-haltig', parent: 'Intermetallics in Al' },
    ));
  });
});

describe('CollectionManager — the working set', () => {
  it('offers the star button when none exists yet', async () => {
    await openManager();
    expect(screen.getByText('★ Create working set')).toBeInTheDocument();
  });

  it('posts {name, exclusive:false} once, and the button is gone afterward', async () => {
    await openManager();
    // `run()` calls `create()` THEN `refresh()` (-> `list()`) in the same
    // tick it awaits; queuing this with `mockResolvedValueOnce` BEFORE the
    // click, rather than reacting to `create` having been called, avoids a
    // race against that internal refresh already having consumed the
    // still-old default reply.
    collectionsApi.list.mockResolvedValueOnce({
      data: {
        collections: [...BODY.collections,
          { name: 'Working set', parent: null, exclusive: false, members: [] }],
        unassigned: BODY.unassigned, problems: [], state: {},
      },
    });
    fireEvent.click(screen.getByText('★ Create working set'));
    // The persisted NAME is a stable "Working set" regardless of UI
    // language — not `t('picker.workingSet')`, which would stop matching
    // its own heading after a language switch.
    await waitFor(() => expect(collectionsApi.create).toHaveBeenCalledWith(
      { name: 'Working set', exclusive: false },
    ));
    await waitFor(() => expect(collectionsApi.list).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText('★ Create working set')).not.toBeInTheDocument());
  });

  it('a working-set collection has no delete button — "not deletable in the UI"', async () => {
    collectionsApi.list.mockResolvedValue({
      data: {
        collections: [{ name: 'Working set', parent: null, exclusive: false,
                         members: [{ key: 'Al', present: true }] }],
        unassigned: [], problems: [], state: {},
      },
    });
    render(<CollectionManager onClose={() => {}} />);
    await screen.findByText('Working set');
    // Rename stays available; Delete must not.
    expect(screen.getByText('Rename')).toBeInTheDocument();
    expect(screen.queryByText('Delete')).not.toBeInTheDocument();
  });
});

describe('CollectionManager — the suggestion run', () => {
  it('proposes without writing anything, shows a count per proposal', async () => {
    collectionsApi.suggest.mockResolvedValue({
      data: { suggestions: [{ name: 'Matrix and pure metals', keys: ['Al', 'Ni', 'WC'] }] },
    });
    await openManager();
    fireEvent.click(screen.getByText('Suggest collections'));
    await screen.findByText('Matrix and pure metals');
    expect(screen.getByText('3 phases')).toBeInTheDocument();
    expect(collectionsApi.applySuggest).not.toHaveBeenCalled();
  });

  it('applies only the checked proposals', async () => {
    collectionsApi.suggest.mockResolvedValue({
      data: {
        suggestions: [
          { name: 'Group A', keys: ['Al'] },
          { name: 'Group B', keys: ['Ni'] },
        ],
      },
    });
    collectionsApi.applySuggest.mockResolvedValue({ data: { created: ['Group A'], skipped: [] } });
    await openManager();
    fireEvent.click(screen.getByText('Suggest collections'));
    await screen.findByText('Group A');

    // Uncheck "Group B" before applying.
    fireEvent.click(screen.getByText('Group B').closest('label').querySelector('input'));
    fireEvent.click(screen.getByText('Create selected'));

    await waitFor(() => expect(collectionsApi.applySuggest).toHaveBeenCalledWith(['Group A']));
  });

  it('a name that already existed comes back under skipped, and is shown — never dropped', async () => {
    collectionsApi.suggest.mockResolvedValue({
      data: { suggestions: [{ name: 'Matrix', keys: ['Al'] }] },
    });
    collectionsApi.applySuggest.mockResolvedValue({ data: { created: [], skipped: ['Matrix'] } });
    await openManager();
    fireEvent.click(screen.getByText('Suggest collections'));
    // "1 phase" (suggestCount for a single-key proposal) is unique — the
    // proposal's own name "Matrix" collides with the existing collection's
    // card and the parent-select option of the same name.
    await screen.findByText('1 phase');
    fireEvent.click(screen.getByText('Create selected'));

    expect(await screen.findByText(/already existed and was skipped/)).toBeInTheDocument();
    expect(screen.getByText(/already existed and was skipped/)).toHaveTextContent('Matrix');
  });

  it('no proposals is shown as an explicit empty state, not a blank panel', async () => {
    collectionsApi.suggest.mockResolvedValue({ data: { suggestions: [] } });
    await openManager();
    fireEvent.click(screen.getByText('Suggest collections'));
    expect(await screen.findByText(/No suggestions/)).toBeInTheDocument();
  });
});

describe('CollectionManager — rename and delete', () => {
  it('renames through the prompt dialog', async () => {
    await openManager();
    fireEvent.click(screen.getAllByText('Rename')[0]);   // Matrix's row
    const dialog = screen.getByRole('dialog', { name: /Rename/ });
    const input = within(dialog).getByRole('textbox');
    fireEvent.change(input, { target: { value: 'Matrix Phases' } });
    fireEvent.click(within(dialog).getByText('Rename'));
    await waitFor(() => expect(collectionsApi.rename).toHaveBeenCalledWith('Matrix', 'Matrix Phases'));
  });

  it('deletes only after the confirm dialog is accepted', async () => {
    await openManager();
    fireEvent.click(screen.getAllByText('Delete')[0]);   // Matrix's row
    expect(collectionsApi.remove).not.toHaveBeenCalled();
    const dialog = screen.getByRole('alertdialog');
    fireEvent.click(within(dialog).getByText('Delete'));
    await waitFor(() => expect(collectionsApi.remove).toHaveBeenCalledWith('Matrix'));
  });
});

describe('CollectionManager — member reorder, remove, move', () => {
  it('the up arrow swaps a member with its predecessor', async () => {
    await openManager();
    fireEvent.click(screen.getAllByTitle('Expand')[0]);   // Matrix: Al, Ghost
    // Two members -> two "Move up" buttons (Al's is disabled at index 0);
    // click Ghost's (index 1) to swap it up with Al.
    fireEvent.click(screen.getAllByTitle('Move up')[1]);
    await waitFor(() => expect(collectionsApi.update).toHaveBeenCalledWith({
      name: 'Matrix', member_keys: ['Ghost', 'Al'],
    }));
  });

  it('the × button removes a member from just this collection', async () => {
    await openManager();
    fireEvent.click(screen.getAllByTitle('Expand')[0]);
    fireEvent.click(screen.getAllByTitle('Remove from this collection (the phase stays in the library).')[0]);
    await waitFor(() => expect(collectionsApi.removeMembers).toHaveBeenCalledWith('Matrix', ['Al']));
  });

  it('"move to" adds to the target — and does not call removeMembers itself', async () => {
    // I1: `handleMoveMember` used to follow `addMembers` with its own
    // `removeMembers(fromCollection.name, ...)` call, unconditionally — that
    // overrides the backend's `assign()`, which already strips the key from
    // every OTHER exclusive collection ONLY when the target is exclusive,
    // and leaves everything untouched when the target is the non-exclusive
    // working set. The frontend must trust `assign()` and never issue its
    // own strip; see the two tests below (exclusive target / working-set
    // target) for the two cases this now covers correctly.
    await openManager();
    fireEvent.click(screen.getAllByTitle('Expand')[0]);   // Matrix
    const moveSelect = screen.getAllByTitle(
      /^Move this phase into another collection\./)[0];
    fireEvent.change(moveSelect, { target: { value: 'Intermetallics' } });
    await waitFor(() => expect(collectionsApi.addMembers).toHaveBeenCalledWith('Intermetallics', ['Al']));
    expect(collectionsApi.removeMembers).not.toHaveBeenCalled();
  });

  it('I1 REGRESSION: moving a member into the working set does not un-file it from its exclusive home', async () => {
    // The exact scenario from the review report: starring a phase into the
    // working set must be purely additive. Before the fix, `handleMoveMember`
    // called `removeMembers('Matrix', ['Al'])` unconditionally after
    // `addMembers`, un-filing Al from Matrix even though "Working set" is
    // `exclusive: false` — overriding `assign()` and contradicting
    // docs/user-guide/Phase-Collections.md's promise that starring never
    // empties a phase's home collection.
    collectionsApi.list.mockResolvedValue({
      data: {
        collections: [
          ...BODY.collections,
          { name: 'Working set', parent: null, exclusive: false, hidden: false,
            member_count: 0, effective_member_count: 0, members: [] },
        ],
        unassigned: BODY.unassigned, problems: [], state: { active: null, hidden: [] },
      },
    });
    render(<CollectionManager onClose={() => {}} />);
    await waitFor(() => expect(collectionsApi.list).toHaveBeenCalled());
    fireEvent.click(screen.getAllByTitle('Expand')[0]);   // Matrix
    const moveSelect = screen.getAllByTitle(
      /^Move this phase into another collection\./)[0];
    fireEvent.change(moveSelect, { target: { value: 'Working set' } });
    await waitFor(() => expect(collectionsApi.addMembers).toHaveBeenCalledWith('Working set', ['Al']));
    expect(collectionsApi.removeMembers).not.toHaveBeenCalled();
  });

  it('adding an unassigned phase to a collection calls addMembers', async () => {
    await openManager();
    const addSelect = screen.getByDisplayValue('Add to…');
    fireEvent.change(addSelect, { target: { value: 'Matrix' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(collectionsApi.addMembers).toHaveBeenCalledWith('Matrix', ['Ni']));
  });
});

/**
 * I3 (whole-branch review): `GET /api/phase-collections/` has always
 * returned `data.problems` (bad JSON / unknown schema, a key filed in two
 * exclusive collections, an orphaned `parent`, a renamed-file repair
 * candidate) — nothing in the frontend ever rendered it, so a structural
 * problem the SERVER already detected and named was invisible on screen.
 * These tests pin that CollectionManager, which already has an error strip
 * for its own failed actions, now surfaces the server's own findings too —
 * for every `kind` the backend emits, not just the ones an English reader
 * would guess at from a generic `JSON.stringify` fallback.
 */
describe('CollectionManager — surfaces data.problems (I3)', () => {
  it('renders a readable sentence for each problem kind, not the raw object', async () => {
    collectionsApi.list.mockResolvedValue({
      data: {
        collections: BODY.collections, unassigned: BODY.unassigned,
        problems: [
          { kind: 'bad_schema', detail: 'Weird.json: Expecting value: line 1 column 1' },
          { kind: 'orphan_parent', detail: 'Impurities', parent: 'Deleted Parent' },
          { kind: 'duplicate_key', key: 'Al', collections: ['Matrix', 'Extras'] },
          { kind: 'missing_phase', collection: 'Matrix', missing_key: 'Al2Cu_old',
            candidates: ['Al2Cu_mp-985806'] },
        ],
        state: { active: null, hidden: [] },
      },
    });
    render(<CollectionManager onClose={() => {}} />);
    await waitFor(() => expect(collectionsApi.list).toHaveBeenCalled());

    // Each message names the specific thing that is wrong — proof this is
    // the real `describeProblem` formatting, not "[object Object]" (what a
    // naive `{problems.map(p => <div>{p}</div>)}` would render) and not a
    // silently dropped, unhandled `kind`.
    screen.getByText(/Weird\.json/);
    screen.getByText(/Impurities/);
    screen.getByText(/Deleted Parent/);
    screen.getByText((_, node) => node?.textContent === '“Al” is filed in more than one collection: Matrix, Extras.');
    screen.getByText(/Al2Cu_old/);
    screen.getByText(/Al2Cu_mp-985806/);
  });

  it('shows nothing extra when there are no problems', async () => {
    await openManager();   // BODY has problems: [] implicitly via the mock default
    expect(screen.queryByText(/is filed in more than one collection/)).toBeNull();
  });
});
