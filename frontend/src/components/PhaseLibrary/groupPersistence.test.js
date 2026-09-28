// @vitest-environment jsdom
//
// jsdom, because the transitional store talks to `window.localStorage` and
// part of what this file pins is what happens when that throws.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../services/api', () => ({
  collectionsApi: {
    list: vi.fn(),
    create: vi.fn(() => Promise.resolve({ data: { success: true } })),
    rename: vi.fn(() => Promise.resolve({ data: { success: true } })),
    remove: vi.fn(() => Promise.resolve({ data: { success: true } })),
    addMembers: vi.fn(() => Promise.resolve({ data: { success: true } })),
    removeMembers: vi.fn(() => Promise.resolve({ data: { success: true } })),
    // Absent until now, which is why NOTHING in this file had ever
    // exercised `nested` -- the one verb that goes through this call.
    update: vi.fn(() => Promise.resolve({ data: { success: true } })),
  },
}));

const { collectionsApi } = await import('../../services/api');
const {
  flags, STORAGE_KEY, transitionalStore, backendStore, activeStore, toGroup,
  toGroups,
} = await import('./groupPersistence');

// `fileURLToPath`, not `new URL(...).pathname`: this checkout lives under a
// directory with spaces, and a raw pathname hands you `%20`. The first run of
// this file failed on exactly that, and the test below that asserts the walk
// SEES something is why it failed loudly instead of walking nothing and
// passing.
const HERE = path.dirname(fileURLToPath(import.meta.url));

const WAS = flags.backendReady;
afterEach(() => { flags.backendReady = WAS; vi.clearAllMocks(); });

describe('the stopgap must not survive the switch', () => {
  // b9's request, in the form he asked for: something that trips when groups
  // move to the library folder if the browser copy is still wired up.

  it('the transitional store refuses to read once the backend is live', () => {
    flags.backendReady = true;
    expect(() => transitionalStore.load()).toThrow(/after the backend went live/);
  });

  it('and refuses to write', () => {
    flags.backendReady = true;
    expect(() => transitionalStore.apply({ verb: 'added' },
      { groups: [], journal: [] })).toThrow(/after the backend went live/);
  });

  it('its message names the file with the switch in it', () => {
    flags.backendReady = true;
    // Whoever hits this at three in the afternoon should not have to search.
    expect(() => transitionalStore.load()).toThrow(/groupPersistence\.js/);
  });

  it('the active store follows the switch, both ways', () => {
    flags.backendReady = true;
    expect(activeStore()).toBe(backendStore);
    flags.backendReady = false;
    expect(activeStore()).toBe(transitionalStore);
  });

  it('the switch is ON -- groups live in the library folder', () => {
    // Stated as a test so that turning it back is a decision somebody makes
    // on purpose, with this line in the diff.
    expect(WAS).toBe(true);
  });

  it('exactly one shipped file in the feature touches browser storage', () => {
    // "State held twice with one copy updated" is the defect this branch has
    // shipped five times. The sixth would be a second reader nobody
    // remembered when the switch was flipped -- and a second reader does not
    // throw, it just quietly disagrees.
    const offenders = [];
    const walk = (dir) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) { walk(p); continue; }
        if (!/\.(js|jsx)$/.test(e.name)) continue;
        if (e.name === 'groupPersistence.js') continue;
        // Tests are allowed to poke storage -- they set it up and clear it,
        // and none of them ships.
        if (/\.test\.(js|jsx)$/.test(e.name)) continue;
        const src = fs.readFileSync(p, 'utf-8');
        if (src.includes('localStorage') || src.includes(STORAGE_KEY)) {
          offenders.push(path.relative(HERE, p).replace(/\\/g, '/'));
        }
      }
    };
    walk(HERE);
    expect(offenders).toEqual([]);
  });

  it('the walk can actually see the files it claims to check', () => {
    // A tree-walk that silently reads nothing passes forever.
    const seen = [];
    const walk = (dir) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) walk(p);
        else if (/\.(js|jsx)$/.test(e.name)) seen.push(e.name);
      }
    };
    walk(HERE);
    expect(seen).toContain('groupPersistence.js');
    expect(seen).toContain('phaseSearch.js');
    expect(seen.length).toBeGreaterThan(15);
  });
});

describe('the library folder', () => {
  it('turns a collection into a group, members and all', () => {
    expect(toGroup({
      id: 'g1', name: 'Al-Fe', parent: null,
      members: [{ key: 'Al', formula: 'Al' }, { key: 'Al13Fe4' }],
    })).toEqual({
      id: 'g1', name: 'Al-Fe', parent: null, members: ['Al', 'Al13Fe4'],
      author: null, updated: null,
    });
  });

  it('copes with a collection that has no members yet', () => {
    expect(toGroup({ id: 'g', name: 'N' }).members).toEqual([]);
  });

  it('reads the groups out of one request', async () => {
    collectionsApi.list.mockResolvedValueOnce({ data: { collections: [
      { id: 'g1', name: 'Al-Fe', members: [{ key: 'Al' }] },
    ] } });
    const got = await backendStore.load();
    expect(got.groups).toEqual([{ id: 'g1', name: 'Al-Fe', parent: null,
      members: ['Al'], author: null, updated: null }]);
  });

  it('a change is ONE request against the group it names, not a whole write', async () => {
    // A whole-state write to a shared folder reverts whatever a colleague
    // did between the read and the save, without a word.
    await backendStore.apply({ verb: 'added', groupName: 'Al-Fe',
      keys: ['Al', 'Si'] });
    expect(collectionsApi.addMembers).toHaveBeenCalledWith('Al-Fe', ['Al', 'Si']);
    expect(collectionsApi.list).not.toHaveBeenCalled();
  });

  it('each verb reaches its own route', async () => {
    await backendStore.apply({ verb: 'created', groupName: 'New' });
    expect(collectionsApi.create).toHaveBeenCalledWith({ name: 'New' });
    await backendStore.apply({ verb: 'renamed', fromName: 'A', groupName: 'B' });
    expect(collectionsApi.rename).toHaveBeenCalledWith('A', 'B');
    await backendStore.apply({ verb: 'deleted', groupName: 'A' });
    expect(collectionsApi.remove).toHaveBeenCalledWith('A');
    await backendStore.apply({ verb: 'removed', groupName: 'A', keys: ['Al'] });
    expect(collectionsApi.removeMembers).toHaveBeenCalledWith('A', ['Al']);
  });

  it('a move is ONE call, carrying the move flag', async () => {
    // It was an add plus one remove per source group, which could
    // half-succeed. c1's `move: true` (c1ac120d) made it atomic on the
    // server, and this test is what says the frontend actually uses it --
    // dropping the flag would silently turn every Alt-drag back into a
    // plain add, leaving the phase in both groups.
    await backendStore.apply({ verb: 'moved', groupName: 'Cubic',
      keys: ['Al'], fromNames: ['Al-Fe', 'Working'] });
    expect(collectionsApi.addMembers).toHaveBeenCalledWith(
      'Cubic', ['Al'], null, { move: true });
    expect(collectionsApi.removeMembers).not.toHaveBeenCalled();
  });

  it('undoing a delete puts the container back AND its contents', async () => {
    await backendStore.apply({ verb: 'recreated', groupName: 'Al-Fe',
      keys: ['Al', 'Si'] });
    expect(collectionsApi.create).toHaveBeenCalledWith({ name: 'Al-Fe' });
    expect(collectionsApi.addMembers).toHaveBeenCalledWith('Al-Fe', ['Al', 'Si']);
  });

  it('undoing a delete of an empty group does not add nothing to it', async () => {
    await backendStore.apply({ verb: 'recreated', groupName: 'Empty', keys: [] });
    expect(collectionsApi.addMembers).not.toHaveBeenCalled();
  });

  it('undoing a move puts each phase back where THAT phase came from', async () => {
    await backendStore.apply({ verb: 'unmoved', groupName: 'Cubic',
      keys: ['Al', 'Si'],
      fromMap: { 'Al-Fe': ['Al'], Working: ['Si'] },
      removeHereKeys: ['Al', 'Si'] });
    expect(collectionsApi.addMembers.mock.calls).toEqual([
      ['Al-Fe', ['Al']], ['Working', ['Si']]]);
    expect(collectionsApi.removeMembers).toHaveBeenCalledWith(
      'Cubic', ['Al', 'Si']);
  });

  it('and never puts a phase into a group it was not in', async () => {
    // The union of source groups used to be sent every key. With G={Al},
    // X={Si} and both dragged onto G, undo added Al to X -- a membership
    // the user had never created, in a group they had not touched.
    await backendStore.apply({ verb: 'unmoved', groupName: 'G',
      keys: ['Al', 'Si'],
      fromMap: { X: ['Si'] },
      removeHereKeys: ['Si'] });
    expect(collectionsApi.addMembers.mock.calls).toEqual([['X', ['Si']]]);
    // ... and Al, which WAS here before the drag, stays here.
    expect(collectionsApi.removeMembers).toHaveBeenCalledWith('G', ['Si']);
  });

  it('does NOT use the move flag to put things back', async () => {
    // Two source groups would be two moves, and the second would take the
    // phases out of the first. An inverse of one move is several adds.
    await backendStore.apply({ verb: 'unmoved', groupName: 'Cubic',
      keys: ['Al', 'Si'],
      fromMap: { A: ['Al'], B: ['Si'] }, removeHereKeys: [] });
    collectionsApi.addMembers.mock.calls.forEach((call) => {
      expect(call[3]).toBeUndefined();
    });
  });

  it('and leaves it here when it was here before the move', async () => {
    // Taking it out would be a loss the user never asked for.
    await backendStore.apply({ verb: 'unmoved', groupName: 'Cubic',
      keys: ['Al'], fromMap: { 'Al-Fe': ['Al'] }, removeHereKeys: [] });
    expect(collectionsApi.removeMembers).not.toHaveBeenCalled();
  });

  it('reports what the server said, rather than a shrug', async () => {
    collectionsApi.create.mockRejectedValueOnce({
      response: { data: { detail: "a collection named 'Al-Fe' already exists" } } });
    const r = await backendStore.apply({ verb: 'created', groupName: 'Al-Fe' });
    expect(r).toEqual({ saved: false,
      reason: "a collection named 'Al-Fe' already exists" });
  });

  it('reports a network failure too', async () => {
    collectionsApi.addMembers.mockRejectedValueOnce(new Error('Network Error'));
    const r = await backendStore.apply({ verb: 'added', groupName: 'A', keys: ['Al'] });
    expect(r.saved).toBe(false);
    expect(r.reason).toMatch(/Network Error/);
  });

  it('refuses a verb it does not know rather than reporting success', async () => {
    const r = await backendStore.apply({ verb: 'teleported', groupName: 'A' });
    expect(r.saved).toBe(false);
    expect(r.reason).toMatch(/teleported/);
  });
});

describe('the browser store, while it is still reachable', () => {
  beforeEach(() => { flags.backendReady = false; window.localStorage.clear(); });

  it('an empty browser means no groups, not an error', async () => {
    await expect(transitionalStore.load()).resolves.toEqual(
      { groups: [], journal: [], schema: 1 });
  });

  it('round-trips groups and the journal', async () => {
    const groups = [{ id: 'g1', name: 'Al-Fe', parent: null, members: ['Al'] }];
    const journal = [{ verb: 'added', groupId: 'g1', key: 'Al' }];
    await expect(transitionalStore.apply({ verb: 'added' }, { groups, journal }))
      .resolves.toEqual({ saved: true });
    await expect(transitionalStore.load()).resolves.toEqual(
      { groups, journal, schema: 1 });
  });

  it('treats unreadable contents as no groups rather than phantom ones', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'not json at all');
    expect((await transitionalStore.load()).groups).toEqual([]);
  });

  it('treats the wrong shape as no groups', async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ groups: 'Al-Fe' }));
    expect((await transitionalStore.load()).groups).toEqual([]);
  });

  it('survives a stored record with no journal', async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ groups: [] }));
    expect((await transitionalStore.load()).journal).toEqual([]);
  });

  it('a refused write is reported, not swallowed', async () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });
    const r = await transitionalStore.apply({ verb: 'added' },
      { groups: [], journal: [] });
    expect(r.saved).toBe(false);
    expect(r.reason).toMatch(/Quota/);
    vi.restoreAllMocks();
  });

  it('a refused read says so, and still answers with no groups', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    const r = await transitionalStore.load();
    expect(r.groups).toEqual([]);
    expect(r.unavailable).toBe(true);
    vi.restoreAllMocks();
  });

  it('the spy really does reach the store (else the two above prove nothing)', async () => {
    // Spied on `Storage.prototype` and not on `window.localStorage`: jsdom
    // serves the storage object through a proxy, and an own-property
    // assignment there is accepted and then ignored -- the first version of
    // these tests called a method that never threw.
    const spy = vi.spyOn(Storage.prototype, 'setItem');
    await transitionalStore.apply({ verb: 'added' }, { groups: [], journal: [] });
    expect(spy).toHaveBeenCalledWith(STORAGE_KEY, expect.any(String));
    vi.restoreAllMocks();
  });
});

describe('a parent arrives as a NAME and has to become an id', () => {
  /**
   * The fixture c1 sent, and the reason it is shaped this way: with
   * `id: 'P', name: 'P'` nothing is being checked. On this machine the
   * collection "Al systems" is stored under the id `Al_systems`, so a
   * comparison that mixes the two finds nothing -- which is exactly the
   * defect that had just been found on the other side of this seam, and
   * which would have shown a flat list here where there is a tree.
   */
  const wire = [
    { id: 'Al_systems', name: 'Al systems', parent: null,
      members: [{ key: 'Al' }] },
    { id: 'Al-Fe_phases', name: 'Al-Fe phases', parent: 'Al systems',
      members: [{ key: 'Al13Fe4' }] },
  ];

  it('the child points at its parent by ID once it is in the screen', () => {
    const got = toGroups(wire);
    expect(got[1].parent).toBe('Al_systems');
    expect(got[1].parent).not.toBe('Al systems');
  });

  it('a root keeps no parent', () => {
    expect(toGroups(wire)[0].parent).toBe(null);
  });

  it('a parent that names nothing known becomes null, not a dangling string', () => {
    // A child pointing at nothing would be hidden from the list, and an
    // invisible group is worse than a misplaced one.
    expect(toGroups([{ id: 'c', name: 'C', parent: 'Deleted yesterday',
      members: [] }])[0].parent).toBe(null);
  });

  it('the store own load goes through it', async () => {
    collectionsApi.list.mockResolvedValueOnce({ data: { collections: wire } });
    const got = await backendStore.load();
    expect(got.groups.map((g) => [g.id, g.parent]))
      .toEqual([['Al_systems', null], ['Al-Fe_phases', 'Al_systems']]);
  });

  it('copes with no collections at all', () => {
    expect(toGroups(undefined)).toEqual([]);
    expect(toGroups([])).toEqual([]);
  });
});

describe('a move, now that it is one request', () => {
  // THIS BLOCK USED TO TEST A HALF-DONE MOVE: the add landed, one remove
  // 404'd, and the result carried `partial` and `stillIn` so the screen
  // could say "it is here AND still there". c1's `move: true` made the
  // whole thing one server-side operation, so that state cannot occur any
  // more -- and the handling for it is gone rather than kept as dead code
  // with green tests over it, which is the shape the board warns about.

  it('either lands or does not -- there is no partial result', async () => {
    collectionsApi.addMembers.mockRejectedValueOnce(new Error('Network Error'));
    const r = await backendStore.apply({ verb: 'moved', groupName: 'Cubic',
      keys: ['Al'], fromNames: ['A', 'B'] });
    expect(r.saved).toBe(false);
    expect(r.reason).toMatch(/Network Error/);
    expect(r.partial).toBeUndefined();
    expect(r.stillIn).toBeUndefined();
    // Nothing was taken out of anywhere, because nothing arrived.
    expect(collectionsApi.removeMembers).not.toHaveBeenCalled();
  });

  it('a move that landed is simply saved', async () => {
    const r = await backendStore.apply({ verb: 'moved', groupName: 'Cubic',
      keys: ['Al'], fromNames: ['A'] });
    expect(r).toEqual({ saved: true });
  });
});

describe('nesting -- the one verb that had no test here at all', () => {
  // The mock had no `update`, so nothing in this file could reach the
  // `nested` branch; it was covered only at the store layer, where the
  // store is a double. `PUT /update` is also the one route that still
  // refuses an id, so what this branch sends is worth pinning.

  it('a subgroup sends the parent by NAME', async () => {
    const r = await backendStore.apply({ verb: 'nested', groupId: 'Al_Fe',
      groupName: 'Al-Fe', parentName: 'Al systems' });
    expect(collectionsApi.update).toHaveBeenCalledWith(
      { name: 'Al-Fe', parent: 'Al systems' });
    expect(r).toEqual({ saved: true });
  });

  it('promoting to the top level says so explicitly', async () => {
    // `parent: null` means "leave it alone" on that route, so null cannot
    // also mean "no parent" -- it needs its own flag.
    await backendStore.apply({ verb: 'nested', groupId: 'Al_Fe',
      groupName: 'Al-Fe', parentName: null });
    expect(collectionsApi.update).toHaveBeenCalledWith(
      { name: 'Al-Fe', clear_parent: true });
  });
});

describe('undoing a delete puts the SHAPE back, not just the contents', () => {
  // `pc.delete` promotes the children and rewrites their files. Undo used
  // to send create + addMembers only, so the screen redrew the tree and
  // the folder stayed flat; the next read flattened the screen too, with
  // nothing said.

  it('re-parents the group itself', async () => {
    await backendStore.apply({ verb: 'recreated', groupName: 'Al-Fe',
      keys: [], parentName: 'Al systems', childNames: [] });
    expect(collectionsApi.update).toHaveBeenCalledWith(
      { name: 'Al-Fe', parent: 'Al systems' });
  });

  it('and puts its children back underneath it', async () => {
    await backendStore.apply({ verb: 'recreated', groupName: 'Al systems',
      keys: [], parentName: null, childNames: ['Al-Fe', 'Al-Cu'] });
    expect(collectionsApi.update.mock.calls).toEqual([
      [{ name: 'Al-Fe', parent: 'Al systems' }],
      [{ name: 'Al-Cu', parent: 'Al systems' }]]);
  });

  it('a top-level group with no children asks for no re-parenting', async () => {
    await backendStore.apply({ verb: 'recreated', groupName: 'Loose',
      keys: ['Al'], parentName: null, childNames: [] });
    expect(collectionsApi.update).not.toHaveBeenCalled();
  });
});
