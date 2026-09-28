// @vitest-environment jsdom
/**
 * The state the screen holds, and the change it hands to the store.
 *
 * The store itself is replaced here. Where the groups end up is
 * `groupPersistence.test.js`'s question; this one asks what the screen does
 * and, just as importantly, WHAT IT SAYS IT DID -- because with the library
 * folder behind it, the change is the whole message. An `added` where a
 * `moved` belonged leaves a phase in two groups on eight machines.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const store = {
  kind: 'test',
  load: vi.fn(() => Promise.resolve({ groups: [], journal: [] })),
  apply: vi.fn(() => Promise.resolve({ saved: true })),
};

vi.mock('./groupPersistence', () => ({
  flags: { backendReady: true },
  STORAGE_KEY: 'test',
  activeStore: () => store,
}));

const { default: useGroups, newGroupId, UNDO_DEPTH, invert, remapId } =
  await import('./useGroups');

const s = () => useGroups.getState();
const lastChange = () => store.apply.mock.calls.at(-1)[0];
/** The change is sent on a microtask behind the state update. */
const settle = () => new Promise((r) => { setTimeout(r, 0); });

beforeEach(() => {
  store.load.mockResolvedValue({ groups: [], journal: [] });
  store.apply.mockResolvedValue({ saved: true });
  store.load.mockClear();
  store.apply.mockClear();
  useGroups.setState({
    groups: [], journal: [], past: [], loaded: false,
    saveError: null, unsaved: [], unavailable: false, author: 'seb',
  });
});
afterEach(() => { vi.restoreAllMocks(); });

describe('identity', () => {
  it('two groups never share an id', () => {
    expect(new Set(Array.from({ length: 500 }, newGroupId)).size).toBe(500);
  });

  it('an id survives a rename', () => {
    const id = s().create('Al-Fe');
    s().rename(id, 'Al-Fe-Si');
    expect(s().groups[0].id).toBe(id);
    expect(s().groups[0].name).toBe('Al-Fe-Si');
  });

  it('works where the browser has no randomUUID', () => {
    const real = globalThis.crypto.randomUUID;
    globalThis.crypto.randomUUID = undefined;
    try {
      expect(new Set(Array.from({ length: 200 }, newGroupId)).size).toBe(200);
    } finally {
      globalThis.crypto.randomUUID = real;
    }
  });
});

describe('reading what is there', () => {
  it('takes the groups the store gives it', async () => {
    store.load.mockResolvedValueOnce({
      groups: [{ id: 'g1', name: 'Al-Fe', members: ['Al'] }], journal: [] });
    await s().load();
    expect(s().groups).toHaveLength(1);
    expect(s().loaded).toBe(true);
  });

  it('a read that FAILED is not an empty library', async () => {
    // Saying "no groups" after a failed request is the mistake this whole
    // page was built to stop making.
    store.load.mockRejectedValueOnce(new Error('Network Error'));
    await s().load();
    expect(s().unavailable).toBe(true);
    expect(s().saveError).toMatch(/Network Error/);
  });

  it('a store that says it is unavailable is passed through as such', async () => {
    store.load.mockResolvedValueOnce({ groups: [], journal: [], unavailable: true });
    await s().load();
    expect(s().unavailable).toBe(true);
  });
});

describe('the change that is handed over', () => {
  it('creating says created, with the name', async () => {
    s().create('  Al-Fe  ');
    await settle();
    expect(lastChange()).toMatchObject({ verb: 'created', groupName: 'Al-Fe' });
  });

  it('renaming carries the OLD name, because that is what the store knows it by', async () => {
    const id = s().create('Al-Fe');
    s().rename(id, 'Al-Fe-Si');
    await settle();
    expect(lastChange()).toMatchObject(
      { verb: 'renamed', fromName: 'Al-Fe', groupName: 'Al-Fe-Si' });
  });

  it('deleting carries the members, so an undo can put them back', async () => {
    const id = s().create('Al-Fe');
    s().addMany(id, ['Al', 'Si']);
    s().remove(id);
    await settle();
    expect(lastChange()).toMatchObject(
      { verb: 'deleted', groupName: 'Al-Fe', keys: ['Al', 'Si'] });
  });

  it('a dragged selection is ONE request and one journal line per phase', async () => {
    const id = s().create('Al-Fe');
    // `await settle()` BEFORE clearing: the store is called a microtask
    // behind the state update, so clearing straight after `create` clears
    // nothing and the count comes out one too high.
    await settle();
    store.apply.mockClear();
    s().addMany(id, ['Al', 'Si', 'Zn']);
    await settle();
    expect(store.apply).toHaveBeenCalledTimes(1);
    expect(lastChange()).toMatchObject({ verb: 'added', keys: ['Al', 'Si', 'Zn'] });
    expect(s().journal.filter((e) => e.verb === 'added').map((e) => e.key))
      .toEqual(['Al', 'Si', 'Zn']);
  });

  it('and only the phases that were not already in it', async () => {
    const id = s().create('Al-Fe');
    s().addMany(id, ['Al']);
    await settle();
    store.apply.mockClear();
    s().addMany(id, ['Al', 'Si']);
    await settle();
    expect(lastChange().keys).toEqual(['Si']);
  });

  it('a move says where it came from, which is what makes it a move', async () => {
    const a = s().create('A');
    const b = s().create('B');
    s().add(a, 'Al');
    await settle();
    store.apply.mockClear();
    s().move(b, 'Al');
    await settle();
    expect(lastChange()).toMatchObject(
      { verb: 'moved', groupName: 'B', keys: ['Al'], fromNames: ['A'],
        // WHICH phase came from WHICH group, not merely the set of
        // groups: the union cannot be undone (see `invert`).
        fromMap: { A: ['Al'] }, wasHereKeys: [] });
  });
});

describe('removing several phases at once', () => {
  // The ⋯ menu ran `drop` in a `forEach`: one gesture became N changes and
  // N unawaited `DELETE /members`. Against the real service before c1's
  // folder lock that lost a removal in 30 of 30 trials; the lock closed
  // that, and this keeps the gesture one change all the same -- otherwise
  // undo takes back one fifth of it with nothing saying so.

  it('is ONE change carrying every phase', async () => {
    const id = s().create('G');
    s().addMany(id, ['Al', 'Si', 'Fe']);
    await settle();
    store.apply.mockClear();
    s().dropMany(id, ['Al', 'Fe']);
    await settle();
    expect(store.apply).toHaveBeenCalledTimes(1);
    expect(lastChange()).toMatchObject(
      { verb: 'removed', groupName: 'G', keys: ['Al', 'Fe'] });
    expect(s().groups[0].members).toEqual(['Si']);
  });

  it('ignores phases that are not in the group', async () => {
    const id = s().create('G');
    s().addMany(id, ['Al']);
    await settle();
    store.apply.mockClear();
    s().dropMany(id, ['Al', 'Nb']);
    await settle();
    expect(lastChange()).toMatchObject({ keys: ['Al'] });
  });

  it('removing nothing is not a change', async () => {
    const id = s().create('G');
    await settle();
    store.apply.mockClear();
    expect(s().dropMany(id, ['Al'])).toBe(false);
    await settle();
    expect(store.apply).not.toHaveBeenCalled();
  });

  it('and one undo puts all of them back', async () => {
    const id = s().create('G');
    s().addMany(id, ['Al', 'Si', 'Fe']);
    await settle();
    s().dropMany(id, ['Al', 'Fe']);
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(lastChange()).toMatchObject(
      { verb: 'added', groupName: 'G', keys: ['Al', 'Fe'] });
    expect(s().groups[0].members).toEqual(['Al', 'Si', 'Fe']);
  });
});

describe('a change that changes nothing is not sent', () => {
  it('adding a phase twice', async () => {
    const id = s().create('Al-Fe');
    s().add(id, 'Al');
    await settle();
    store.apply.mockClear();
    expect(s().add(id, 'Al')).toBe(false);
    await settle();
    expect(store.apply).not.toHaveBeenCalled();
  });

  it('removing one that is not there', () => {
    const id = s().create('Al-Fe');
    expect(s().drop(id, 'Al')).toBe(false);
  });

  it('moving a phase into the only group it is in', () => {
    const a = s().create('A');
    s().add(a, 'Al');
    expect(s().move(a, 'Al')).toBe(false);
  });

  it('anything aimed at a group that is gone', async () => {
    expect(s().add('nope', 'Al')).toBe(false);
    expect(s().remove('nope')).toBe(false);
    expect(s().move('nope', 'Al')).toBe(false);
    expect(s().rename('nope', 'X')).toBe(false);
    await settle();
    expect(store.apply).not.toHaveBeenCalled();
  });
});

describe('a refusal is put on screen, not swallowed', () => {
  it('names what the store said', async () => {
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().create('Al-Fe');
    await settle();
    expect(s().saveError).toBe('read-only');
    // The change still happened on screen; what failed is keeping it.
    expect(s().groups).toHaveLength(1);
  });

  it('names a thrown failure too', async () => {
    store.apply.mockRejectedValueOnce(new Error('Network Error'));
    s().create('Al-Fe');
    await settle();
    expect(s().saveError).toMatch(/Network Error/);
  });

  it('a later change that lands clears the complaint', async () => {
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().create('A');
    await settle();
    expect(s().saveError).toBe('read-only');
    s().create('B');
    await settle();
    expect(s().saveError).toBe(null);
  });
});

describe('undo', () => {
  // Two different things at once, on purpose: the SCREEN gets the whole
  // previous state (2026-08-26, an undo here rebuilt its snapshot field by
  // field, left two out and saved the emptied version); the STORE gets the
  // inverse CHANGE, because writing a snapshot back to a shared folder
  // reverts whatever a colleague did in between.
  it('the screen goes back to the whole previous state', () => {
    const id = s().create('Al-Fe');
    s().addMany(id, ['Al', 'Si']);
    s().remove(id);
    s().undo();
    expect(s().groups[0].members).toEqual(['Al', 'Si']);
    expect(s().groups[0].id).toBe(id);
  });

  it('and the journal goes back with it', () => {
    const id = s().create('Al-Fe');
    const n = s().journal.length;
    s().add(id, 'Al');
    s().undo();
    expect(s().journal).toHaveLength(n);
  });

  it('the store gets the inverse change, never a snapshot', async () => {
    const id = s().create('Al-Fe');
    s().add(id, 'Al');
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(lastChange()).toMatchObject(
      { verb: 'removed', groupName: 'Al-Fe', keys: ['Al'] });
  });

  it('undoing a delete asks for the group AND its contents back', async () => {
    const id = s().create('Al-Fe');
    s().addMany(id, ['Al', 'Si']);
    s().remove(id);
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(lastChange()).toMatchObject(
      { verb: 'recreated', groupName: 'Al-Fe', keys: ['Al', 'Si'] });
  });

  it('undoing a move puts it back where it came from', async () => {
    const a = s().create('A');
    const b = s().create('B');
    s().add(a, 'Al');
    s().move(b, 'Al');
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(lastChange()).toMatchObject({ verb: 'unmoved', groupName: 'B',
      keys: ['Al'], fromMap: { A: ['Al'] }, removeHereKeys: ['Al'] });
  });

  it('undoing a MULTI-phase move is exact per phase', async () => {
    // The defect this replaces: one boolean and one union for a list.
    // G held Al, X held Si, both were dragged onto G. The old inverse
    // added BOTH to X -- Al had never been there -- and removed BOTH
    // from G -- Al had been there before the drag. One membership
    // destroyed, one invented, and no test could express it because
    // every fixture used one key and one source group.
    const g = s().create('G');
    const x = s().create('X');
    s().add(g, 'Al');
    s().add(x, 'Si');
    await settle();
    s().moveMany(g, ['Al', 'Si']);
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(lastChange()).toMatchObject({
      verb: 'unmoved', groupName: 'G',
      fromMap: { X: ['Si'] },     // Al came from nowhere; it was already here
      removeHereKeys: ['Si'],     // ... so only Si leaves again
    });
  });

  it('does nothing, and says so, when there is nothing to take back', () => {
    expect(s().undo()).toBe(false);
  });

  it('keeps a bounded history rather than the whole session', () => {
    const id = s().create('G');
    for (let i = 0; i < UNDO_DEPTH + 10; i += 1) s().add(id, `k${i}`);
    expect(s().past).toHaveLength(UNDO_DEPTH);
  });

  it('records no step for a change that did not happen', () => {
    const id = s().create('Al-Fe');
    s().add(id, 'Al');
    const depth = s().past.length;
    s().add(id, 'Al');
    expect(s().past).toHaveLength(depth);
  });
});

describe('inverting a change, on its own', () => {
  it('a move back into the group it was already in does not evict it', () => {
    // Taking it out would be a loss the user never asked for.
    expect(invert({ verb: 'moved', groupId: 'b', groupName: 'B',
      keys: ['Al'], fromMap: { A: ['Al'] }, wasHereKeys: ['Al'] })
      .removeHereKeys).toEqual([]);
  });

  it('and evicts only the phases the move actually brought in', () => {
    expect(invert({ verb: 'moved', groupId: 'g', groupName: 'G',
      keys: ['Al', 'Si'], fromMap: { X: ['Si'] }, wasHereKeys: ['Al'] })
      .removeHereKeys).toEqual(['Si']);
  });

  it('undoing a delete carries the shape, not just the contents', () => {
    // Without these the folder keeps the children at the top level while
    // the screen redraws the tree.
    expect(invert({ verb: 'deleted', groupId: 'g', groupName: 'Al systems',
      keys: ['Al'], parentName: 'Metals', childNames: ['Al-Fe'] }))
      .toMatchObject({ verb: 'recreated', parentName: 'Metals',
        childNames: ['Al-Fe'] });
  });

  it('a rename inverts to a rename the other way', () => {
    expect(invert({ verb: 'renamed', groupId: 'g', groupName: 'New',
      fromName: 'Old' })).toMatchObject({ groupName: 'Old', fromName: 'New' });
  });

  it('each verb has an inverse, and an unknown one has none', () => {
    for (const verb of ['created', 'deleted', 'renamed', 'added', 'removed',
      'moved']) {
      expect(invert({ verb, groupId: 'g', groupName: 'N', keys: [] }))
        .not.toBe(null);
    }
    expect(invert({ verb: 'teleported' })).toBe(null);
  });
});

describe('naming', () => {
  it('refuses an empty name without touching anything', async () => {
    expect(() => s().create('   ')).toThrow(/needs a name/);
    expect(s().groups).toEqual([]);
    await settle();
    expect(store.apply).not.toHaveBeenCalled();
  });

  it('refuses to rename to nothing', () => {
    const id = s().create('Al-Fe');
    expect(() => s().rename(id, '')).toThrow(/needs a name/);
    expect(s().groups[0].name).toBe('Al-Fe');
  });
});

describe('the journal', () => {
  it('records who and when, and the name as well as the id', () => {
    const id = s().create('Al-Fe');
    s().add(id, 'Al');
    expect(s().journal.at(-1)).toMatchObject({
      verb: 'added', groupId: id, groupName: 'Al-Fe', key: 'Al', author: 'seb',
    });
    expect(Date.parse(s().journal.at(-1).at)).not.toBeNaN();
  });

  it('a delete entry still reads once the group is gone', () => {
    const id = s().create('Al-Fe');
    s().remove(id);
    expect(s().journal.at(-1)).toMatchObject(
      { verb: 'deleted', groupName: 'Al-Fe' });
  });
});

describe('the folder mints the identity, not this browser', () => {
  // FOUND LIVE, not by reading: the screen carried `g_8c114cca-…` for a
  // group the folder called `zz-m5-live-AlFe`. Nothing broke at the time,
  // because every write goes by name today -- it breaks the moment the
  // writing routes take ids, and `data-group-id` was already a lie.
  const folderAnswers = (name, id, members = []) => {
    store.apply.mockResolvedValueOnce({ saved: true, reload: true });
    store.load.mockResolvedValueOnce({
      groups: [{ id, name, parent: null, members }], journal: [] });
  };

  it('takes the folder id after a create', async () => {
    folderAnswers('Al systems', 'Al_systems');
    s().create('Al systems');
    await settle();
    expect(s().groups.map((g) => g.id)).toEqual(['Al_systems']);
  });

  it('and does not leave the placeholder in the journal', async () => {
    folderAnswers('Al systems', 'Al_systems');
    const placeholder = s().create('Al systems');
    await settle();
    expect(placeholder).toMatch(/^g_/);
    expect(s().journal.map((e) => e.groupId)).toEqual(['Al_systems']);
  });

  it('nor in the undo history, which would put it back on screen', async () => {
    // Stepping back past the create would otherwise restore the invented id.
    const first = s().create('A');
    await settle();
    folderAnswers('B', 'B_id');
    s().create('B');
    await settle();
    expect(s().past.at(-1).change.groupId).not.toMatch(/^g_/);
    expect(s().past.at(-1).change.groupId).toBe('B_id');
    expect(first).toMatch(/^g_/);   // the first create had no folder answer
  });

  it('and does not stop undo from writing to the folder', async () => {
    // THE DEFECT: `remapId` rebuilt each undo snapshot as
    // `{groups, journal, change}`, dropping `landed` -- the flag `undo`
    // gates on. It runs after every successful create (a create answers
    // `reload: true`), so from the first group a session made, every undo
    // was screen-only and SILENT: `undo` clears `saveError` and never
    // touches `unsaved`, so the one button that re-reads the folder does
    // not even appear.
    //
    // Delete a group with 20 phases, create any group, undo twice: the
    // screen showed the group restored with its phases and the folder had
    // neither, until the next restart.
    //
    // This asserts what reached the STORE, not what is on screen -- the
    // screen was right the whole time, which is why nothing caught it.
    useGroups.setState({
      groups: [{ id: 'Matrix', name: 'Matrix', parent: null,
        members: ['Al', 'Si'], author: 'seb', updated: null }],
    });
    s().remove('Matrix');
    await settle();
    folderAnswers('Later', 'Later_id');
    s().create('Later');
    await settle();

    store.apply.mockClear();
    s().undo();                      // takes back the create
    await settle();
    s().undo();                      // takes back the delete
    await settle();

    const verbs = store.apply.mock.calls.map((c) => c[0].verb);
    expect(verbs).toEqual(['deleted', 'recreated']);
    expect(store.apply.mock.calls.at(-1)[0]).toMatchObject({
      verb: 'recreated', groupName: 'Matrix', keys: ['Al', 'Si'] });
  });

  it('takes the folder COPY, so members it already had are on screen', async () => {
    // Undoing a delete recreates the group and refills it; the folder is
    // then the only one that knows what landed.
    folderAnswers('Al-Fe', 'Al-Fe', ['Al', 'Si']);
    s().create('Al-Fe');
    await settle();
    expect(s().groups[0].members).toEqual(['Al', 'Si']);
  });

  it('keeps the placeholder when the folder has no such group', async () => {
    // A failed re-read must not make the group vanish from the screen.
    store.apply.mockResolvedValueOnce({ saved: true, reload: true });
    store.load.mockResolvedValueOnce({ groups: [], journal: [] });
    s().create('Al-Fe');
    await settle();
    expect(s().groups).toHaveLength(1);
    expect(s().groups[0].name).toBe('Al-Fe');
  });

  it('survives a re-read that fails outright', async () => {
    store.apply.mockResolvedValueOnce({ saved: true, reload: true });
    store.load.mockRejectedValueOnce(new Error('Network Error'));
    s().create('Al-Fe');
    await settle();
    expect(s().groups).toHaveLength(1);
  });

  it('a change that needs no new identity does NOT re-read', async () => {
    const id = s().create('Al-Fe');
    await settle();
    store.load.mockClear();
    s().addMany(id, ['Al']);
    await settle();
    expect(store.load).not.toHaveBeenCalled();
  });
});

describe('remapping an identity, on its own', () => {
  const state = () => ({
    groups: [{ id: 'old', name: 'N', members: ['Al'] },
      { id: 'other', name: 'O', members: [] }],
    journal: [{ verb: 'created', groupId: 'old' },
      { verb: 'added', groupId: 'other' }],
    past: [{ groups: [{ id: 'old', name: 'N', members: [] }], journal: [],
      change: { verb: 'created', groupId: 'old' } }],
  });

  it('reaches the groups, the journal AND the undo history', () => {
    const r = remapId(state(), 'old', 'new');
    expect(r.groups.map((g) => g.id)).toEqual(['new', 'other']);
    expect(r.journal.map((e) => e.groupId)).toEqual(['new', 'other']);
    expect(r.past[0].groups[0].id).toBe('new');
    expect(r.past[0].change.groupId).toBe('new');
  });

  it('leaves every other group alone', () => {
    expect(remapId(state(), 'old', 'new').groups[1]).toEqual(
      { id: 'other', name: 'O', members: [] });
  });

  it('does nothing when there is nothing to do', () => {
    const st = state();
    expect(remapId(st, 'old', 'old')).toBe(st);
    expect(remapId(st, null, 'new')).toBe(st);
    expect(remapId(st, 'old', null)).toBe(st);
  });
});

describe('undo must not undo something that never happened', () => {
  /**
   * THE WORST DEFECT FOUND IN THIS FEATURE, and it needs one user and two
   * clicks.
   *
   *   1. A group "Matrix" with 20 phases exists.
   *   2. The user makes a new group and types "Matrix". The endpoint
   *      refuses: 409, "a collection named 'Matrix' already exists". The
   *      panel says the change was not saved.
   *   3. The user does the obvious thing and presses Undo.
   *   4. The inverse of `created` is `deleted` -- and the folder deletes
   *      the REAL Matrix, with its twenty memberships, permanently. The
   *      screen shows Matrix still there, because the snapshot has it.
   *      Nothing says a thing was destroyed.
   *
   * The cause: `undo` inverted a change without asking whether the forward
   * change had ever reached the folder. An inverse is only meaningful for
   * something that happened.
   */
  it('a change the store REFUSED is taken back on screen and nowhere else', async () => {
    store.apply.mockResolvedValueOnce({ saved: false, reason: '409 already exists' });
    s().create('Matrix');
    await settle();
    expect(s().saveError).toMatch(/already exists/);
    store.apply.mockClear();

    expect(s().undo()).toBe(true);
    await settle();
    expect(s().groups).toEqual([]);          // the screen goes back
    expect(store.apply).not.toHaveBeenCalled();   // the folder is untouched
  });

  it('the same for a change that threw', async () => {
    store.apply.mockRejectedValueOnce(new Error('Network Error'));
    s().create('Matrix');
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(store.apply).not.toHaveBeenCalled();
  });

  it('and a change that DID land is still inverted, as before', async () => {
    const id = s().create('Al-Fe');
    s().add(id, 'Al');
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(lastChange()).toMatchObject({ verb: 'removed', keys: ['Al'] });
  });

  it('a refused change does not hide an earlier one that landed', async () => {
    // Undo twice: the first step takes back something that never reached
    // the folder, the second takes back something that did.
    const id = s().create('Al-Fe');
    await settle();
    s().add(id, 'Al');
    await settle();
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().add(id, 'Si');
    await settle();
    store.apply.mockClear();

    s().undo();                       // the refused one: screen only
    await settle();
    expect(store.apply).not.toHaveBeenCalled();

    s().undo();                       // the one that landed: inverted
    await settle();
    expect(lastChange()).toMatchObject({ verb: 'removed', keys: ['Al'] });
  });
});

describe('a move is one change, however many phases', () => {
  it('twelve phases are one request and one undo', async () => {
    const a = s().create('A');
    const b = s().create('B');
    const keys = Array.from({ length: 12 }, (_, i) => `k${i}`);
    s().addMany(a, keys);
    await settle();
    store.apply.mockClear();

    s().moveMany(b, keys);
    await settle();
    expect(store.apply).toHaveBeenCalledTimes(1);
    expect(lastChange()).toMatchObject({ verb: 'moved', groupName: 'B', keys });
    expect(s().groups.find((g) => g.id === a).members).toEqual([]);

    // One press, the whole gesture.
    s().undo();
    expect(s().groups.find((g) => g.id === a).members).toEqual(keys);
    expect(s().groups.find((g) => g.id === b).members).toEqual([]);
  });

  it('names every group the phases have to leave, each once', async () => {
    const a = s().create('A');
    const b = s().create('B');
    const c = s().create('C');
    s().addMany(a, ['Al', 'Si']);
    s().addMany(b, ['Al']);
    await settle();
    store.apply.mockClear();
    s().moveMany(c, ['Al', 'Si']);
    await settle();
    expect(lastChange().fromNames.sort()).toEqual(['A', 'B']);
  });

  it('a move that would change nothing is still not a change', () => {
    const a = s().create('A');
    s().addMany(a, ['Al', 'Si']);
    expect(s().moveMany(a, ['Al', 'Si'])).toBe(false);
  });

  it('but moving phases the group only partly holds IS a change', () => {
    const a = s().create('A');
    const b = s().create('B');
    s().addMany(a, ['Al']);
    s().addMany(b, ['Si']);
    expect(s().moveMany(a, ['Al', 'Si'])).toBe(true);
    expect(s().groups.find((g) => g.id === a).members).toEqual(['Al', 'Si']);
    expect(s().groups.find((g) => g.id === b).members).toEqual([]);
  });

  it('an empty list moves nothing', () => {
    const a = s().create('A');
    expect(s().moveMany(a, [])).toBe(false);
  });
});

describe('what did not get written is remembered until somebody reads again', () => {
  // There was one slot, cleared at the top of every change. A failure
  // followed by a success erased the warning while the unsaved change was
  // still on screen, and after two failures the first was unnameable --
  // so the screen and the folder could disagree indefinitely with nothing
  // saying so, and there was no way to re-read short of restarting.
  it('a refusal is kept, with the group and the reason', async () => {
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().create('Matrix');
    await settle();
    expect(s().unsaved).toEqual([
      { verb: 'created', groupName: 'Matrix', reason: 'read-only' },
    ]);
  });

  it('a LATER success does not erase it', async () => {
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().create('A');
    await settle();
    s().create('B');
    await settle();
    expect(s().unsaved).toHaveLength(1);
    expect(s().unsaved[0].groupName).toBe('A');
  });

  it('two refusals are both kept, in order', async () => {
    store.apply.mockResolvedValue({ saved: false, reason: 'read-only' });
    s().create('A');
    await settle();
    s().create('B');
    await settle();
    expect(s().unsaved.map((u) => u.groupName)).toEqual(['A', 'B']);
  });

  it('only a successful read clears them', async () => {
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().create('A');
    await settle();
    await s().load();
    expect(s().unsaved).toEqual([]);
  });

  it('a read that FAILED does not clear them', async () => {
    // Forgetting on a failed read would be the same lie in a new place.
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().create('A');
    await settle();
    store.load.mockRejectedValueOnce(new Error('Network Error'));
    await s().load();
    expect(s().unsaved).toHaveLength(1);
  });

  it('a refused move is NOT landed, so undo sends nothing', async () => {
    // The opposite of what this test used to assert, because the thing it
    // asserted cannot happen any more: a move is one request since c1's
    // `move: true`, so a refusal means nothing reached the folder, and
    // sending the inverse would take the phase out of a group the server
    // never put it in.
    const a = s().create('A');
    const b = s().create('B');
    s().addMany(a, ['Al']);
    await settle();
    store.apply.mockResolvedValueOnce({ saved: false, reason: 'read-only' });
    s().moveMany(b, ['Al']);
    await settle();
    store.apply.mockClear();
    s().undo();
    await settle();
    expect(store.apply).not.toHaveBeenCalled();
  });
});
