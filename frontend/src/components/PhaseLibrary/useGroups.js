/**
 * The groups, as the screen holds them.
 *
 * `groups.js` says what a group is, `groupPersistence.js` says where it
 * lives, and this binds the two and adds the two things only a running
 * screen needs: undo, and knowing whether the last change reached the store.
 *
 * THE SCREEN IS OPTIMISTIC AND THE STORE IS ASKED AFTERWARDS. A change shows
 * at once and the request follows; a refusal is put on screen in words, and
 * the screen then disagrees with the folder until the next read -- which the
 * message says. The alternative, waiting for the round trip before drawing,
 * makes filing twelve phases feel like twelve separate decisions.
 *
 * UNDO IS TWO DIFFERENT THINGS AT ONCE, and that is deliberate. For the
 * SCREEN it restores the whole previous state -- 2026-08-26, this project
 * shipped an undo that rebuilt its snapshot field by field, left two fields
 * out and silently saved the emptied version, the exact work the feature had
 * been built to keep. For the STORE it sends the INVERSE CHANGE, not the
 * snapshot: eight people share that folder, and writing a whole snapshot
 * back would revert whatever a colleague did in between, without a word.
 */
import { create } from 'zustand';
import {
  makeGroup, addMember, removeMember, moveTo, renameGroup, deleteGroup,
  journalEntry, canNest,
} from './groups';
import { activeStore } from './groupPersistence';

/** How many steps back. Beyond this the oldest is dropped. */
export const UNDO_DEPTH = 30;

/**
 * Identity that survives a rename and never collides.
 *
 * `crypto.randomUUID` where it exists -- it does in Electron and in every
 * browser this ships to -- and a time-plus-random fallback rather than a
 * counter, because a counter restarts at 1 on the next launch and would hand
 * a second group the id of the first.
 */
export function newGroupId() {
  try {
    if (globalThis.crypto && globalThis.crypto.randomUUID) {
      return `g_${globalThis.crypto.randomUUID()}`;
    }
  } catch { /* fall through */ }
  return `g_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * The change that undoes a change.
 *
 * Every field an inverse needs is carried on the change itself, so this is
 * arithmetic and not a lookup into a state that has already moved on.
 */
export function invert(change) {
  switch (change.verb) {
    case 'created':
      return { verb: 'deleted', groupId: change.groupId,
        groupName: change.groupName };
    case 'deleted':
      // Four steps in the store, because a group, its contents, where it
      // sat and what sat under it are four things there. The screen has
      // already put all of them back in one, from the snapshot.
      //
      // IT USED TO CARRY ONLY THE MEMBERS, and the children were the half
      // nobody restored: deleting a parent promotes its children (both
      // sides do this, correctly), undo redrew them indented under the
      // restored parent and left them at the top level in the folder. The
      // next read flattened the tree again with nothing said. A change
      // object cannot restore what it was never told.
      return { verb: 'recreated', groupId: change.groupId,
        groupName: change.groupName, keys: change.keys || [],
        parentName: change.parentName || null,
        childNames: change.childNames || [] };
    case 'renamed':
      return { verb: 'renamed', groupId: change.groupId,
        groupName: change.fromName, fromName: change.groupName };
    case 'added':
      return { verb: 'removed', groupId: change.groupId,
        groupName: change.groupName, keys: change.keys };
    case 'removed':
      return { verb: 'added', groupId: change.groupId,
        groupName: change.groupName, keys: change.keys };
    case 'nested':
      return { verb: 'nested', groupId: change.groupId,
        groupName: change.groupName, parentName: change.fromParentName || null,
        fromParentName: change.parentName || null };
    case 'moved':
      // Put each phase back in the groups THAT PHASE came from, and take out
      // of here only the ones that were not here to begin with.
      //
      // BOTH HALVES USED TO BE ONE ANSWER FOR A WHOLE LIST, and a
      // multi-phase Alt-drag then undid into a state nobody had ever been
      // in. `fromNames` was the union of every source group and
      // `alsoRemoveHere` a single boolean, so with G={Al}, X={Si}, both
      // dragged onto G:
      //
      //   undo -> addMembers('X', ['Al','Si'])   Al was never in X
      //           removeMembers('G', ['Al','Si']) Al was in G before the drag
      //   folder: G={}, X={Si,Al}      screen: G={Al}, X={Si}
      //
      // One membership destroyed, one invented, silently, until restart.
      // Per phase it is exact, and the map is what the change already knew
      // and threw away.
      return { verb: 'unmoved', groupId: change.groupId,
        groupName: change.groupName, keys: change.keys,
        fromMap: change.fromMap || {},
        // Only the phases the move actually brought in leave again.
        removeHereKeys: (change.keys || [])
          .filter((k) => !(change.wasHereKeys || []).includes(k)) };
    default:
      return null;
  }
}

/**
 * Swap a placeholder identity for the real one, everywhere it appears.
 *
 * A group made here gets an id from this browser while the request is in
 * flight, and the FOLDER mints its own -- "Al systems" is stored as
 * `Al_systems`. Measured live: the screen carried `g_8c114cca-...` for a
 * group the folder called `zz-m5-live-AlFe`. Nothing broke at the time,
 * because every write goes by name today; it breaks the moment the writing
 * routes take ids, and it is already a lie in `data-group-id`.
 *
 * The undo snapshots are remapped too. Leaving them alone would put the
 * placeholder back on screen the first time somebody stepped back past the
 * create -- state held twice with one copy updated, for the sixth time.
 *
 * SPREAD THE SNAPSHOT, NEVER REBUILD IT. This function used to write the
 * new snapshot out field by field -- `{groups, journal, change}` -- which
 * silently dropped `landed`, the flag that decides whether undo writes to
 * the folder at all. It ran after every successful create, because
 * `created` returns `reload: true`, so from the first group a session made,
 * every undo went screen-only:
 *
 *   delete "Matrix" (20 phases) -> create any group -> undo, undo
 *   screen: Matrix back, with its phases.  folder: neither.
 *
 * and nothing said so, because `undo` clears `saveError` and never touches
 * `unsaved`, so the one button that re-reads the folder does not appear.
 * Restarting was the first time you found out.
 *
 * A snapshot is a bag of state whose contents will grow again. Copying it
 * whole and overriding the three fields that hold ids keeps the next added
 * field from disappearing the same way.
 */
export function remapId(state, from, to) {
  if (!from || !to || from === to) return state;
  const fix = (g) => (g.id === from ? { ...g, id: to } : g);
  const fixEntry = (e) => (e.groupId === from ? { ...e, groupId: to } : e);
  const fixChange = (c) => (c && c.groupId === from ? { ...c, groupId: to } : c);
  return {
    groups: state.groups.map(fix),
    journal: state.journal.map(fixEntry),
    past: state.past.map((snap) => ({
      ...snap,
      groups: snap.groups.map(fix),
      journal: snap.journal.map(fixEntry),
      change: fixChange(snap.change),
    })),
  };
}

const useGroups = create((set, get) => ({
  groups: [],
  journal: [],
  //: {groups, journal, change} snapshots, newest last
  past: [],
  loaded: false,
  //: null, or a sentence saying the last change did not reach the store
  saveError: null,
  /**
   * EVERY change that did not reach the folder, until somebody reads it
   * again.
   *
   * There was one slot, cleared at the top of every change -- so a failure
   * followed by a success erased the warning while the unsaved change was
   * still on screen, and after two failures the first was unnameable. The
   * screen and the folder could then disagree indefinitely with nothing
   * saying so, and there was no way to re-read short of restarting the app.
   *
   * Entries are {verb, groupName, reason}. Only a successful `load` clears
   * them, because only a read makes the screen true again.
   */
  unsaved: [],
  //: true when the store could not be read at all
  unavailable: false,
  //: who the journal credits. Typed once, kept for the session.
  author: '',

  setAuthor: (author) => set({ author: String(author || '') }),

  load: () => Promise.resolve()
    .then(() => activeStore().load())
    .then((got) => {
      set({
        groups: got.groups,
        journal: got.journal,
        loaded: true,
        unavailable: Boolean(got.unavailable),
        saveError: null,
        // A read is the only thing that makes the screen true again, so it
        // is the only thing that may forget what did not get written.
        unsaved: [],
        past: [],
      });
      return true;
    })
    .catch((err) => {
      // A failed read is not an empty library. Saying "no groups" here is
      // the mistake this whole page was built to stop making.
      set({ loaded: true, unavailable: true, groups: [], journal: [],
        saveError: String(err && err.message ? err.message : err) });
      return false;
    }),

  /**
   * Apply a change, record it, and tell the store.
   *
   * One funnel, so "push undo, append journal, write, report a refusal"
   * cannot be got right in five places and wrong in the sixth. `mutate`
   * returning the SAME array means nothing happened -- and then nothing is
   * recorded and nothing is sent either.
   */
  apply: (mutate, change, entryKeys) => {
    const { groups, journal, past, author } = get();
    const now = new Date().toISOString();
    const next = mutate(groups, now);
    if (next === groups) return false;

    // One journal line per phase, because the journal answers "when did Al
    // get into Al-Fe" -- and one request per change, because the store's
    // routes take a list. They are not the same granularity.
    const entries = (entryKeys && entryKeys.length
      ? entryKeys.map((key) => journalEntry(change.verb, {
        ...change, key, author, now }))
      : [journalEntry(change.verb, { ...change, key: null, author, now })]);

    const nextJournal = [...journal, ...entries];
    // `landed` says whether this change ever reached the store. An undo of
    // something that was REFUSED must take it back on screen and send
    // nothing -- see the note on `undo`. The entry is mutated in place
    // because the write finishes after the state is set, and the entry is
    // not React state: nothing renders it.
    const entry = { groups, journal, change, landed: false };
    set({
      groups: next,
      journal: nextJournal,
      past: [...past, entry].slice(-UNDO_DEPTH),
      saveError: null,
    });
    Promise.resolve()
      .then(() => activeStore().apply(change, { groups: next, journal: nextJournal }))
      .then((res) => {
        if (!res || res.saved === false) {
          set((st) => ({
            saveError: (res && res.reason) || 'unknown',
            unsaved: [...st.unsaved, {
              verb: change.verb,
              groupName: change.groupName,
              reason: (res && res.reason) || 'unknown',
            }],
          }));
          // `landed` stays false: with every verb now a single request, a
          // refusal means nothing reached the folder, so the inverse must
          // not be sent. The `partial` case this used to carry belonged to
          // the three-request move and went with it.
          return null;
        }
        entry.landed = true;
        // The store knows an identity this screen invented. Read it back and
        // take the folder's answer -- matched on the NAME, the only thing
        // both sides agreed on while the request was in flight.
        if (res.reload) return get().adoptIdentity(change);
        return null;
      })
      .catch((err) => set({
        saveError: String(err && err.message ? err.message : err) }));
    return true;
  },

  /** Re-read, and swap this change's placeholder id for the folder's. */
  adoptIdentity: (change) => Promise.resolve()
    .then(() => activeStore().load())
    .then((got) => {
      const real = got.groups.find((g) => g.name === change.groupName);
      if (!real) return false;
      const { groups, journal, past } = get();
      // The folder's copy of the group replaces the placeholder wholesale:
      // it carries the members, the author and the timestamp that the
      // screen could only guess at.
      const merged = groups.map(
        (g) => (g.id === change.groupId ? { ...real } : g));
      set(remapId({ groups: merged, journal, past }, change.groupId, real.id));
      return true;
    })
    .catch(() => false),

  /**
   * Take the last change back.
   *
   * ON SCREEN it restores the whole previous state. IN THE FOLDER it sends
   * the inverse change -- but ONLY if the forward change ever got there.
   *
   * That condition is not a nicety. Without it: a group "Matrix" exists, a
   * user makes a new group and types "Matrix", the endpoint refuses with
   * 409, the user presses Undo -- and the inverse of `created` is
   * `deleted`, which deletes the REAL Matrix and its twenty memberships,
   * permanently, while the screen goes on showing it because the snapshot
   * has it. Nothing says a thing was destroyed. Found by an adversarial
   * read; it needs one user and two clicks, and the second click is the
   * recovery gesture.
   *
   * An inverse is only meaningful for something that happened.
   */
  undo: () => {
    const { past } = get();
    if (!past.length) return false;
    const prev = past[past.length - 1];
    set({
      groups: prev.groups,
      journal: prev.journal,
      past: past.slice(0, -1),
      saveError: null,
    });
    if (!prev.landed) return true;
    const back = invert(prev.change);
    if (!back) return true;
    Promise.resolve()
      .then(() => activeStore().apply(back, prev))
      .then((res) => {
        if (!res || res.saved === false) {
          set({ saveError: (res && res.reason) || 'unknown' });
        }
      })
      .catch((err) => set({
        saveError: String(err && err.message ? err.message : err) }));
    return true;
  },

  // ---- the named changes -------------------------------------------------

  create: (name) => {
    const id = newGroupId();
    const clean = String(name || '').trim();
    return get().apply(
      (groups, now) => [...groups,
        makeGroup({ id, name, author: get().author, now })],
      { verb: 'created', groupId: id, groupName: clean },
    ) ? id : null;
  },

  rename: (id, name) => {
    const g = get().groups.find((x) => x.id === id);
    if (!g) return false;
    const clean = String(name || '').trim();
    return get().apply(
      (groups, now) => renameGroup(groups, id, name, { now }),
      { verb: 'renamed', groupId: id, groupName: clean, fromName: g.name },
    );
  },

  remove: (id) => {
    const { groups: before } = get();
    const g = before.find((x) => x.id === id);
    if (!g) return false;
    const parent = before.find((x) => x.id === g.parent);
    return get().apply(
      (groups) => {
        const next = deleteGroup(groups, id);
        return next.length === groups.length ? groups : next;
      },
      // Everything an undo needs, by NAME, because once the group is gone
      // `g_8f2c` is unreadable: the contents, where this group sat, and
      // which groups sat under it and were promoted by the delete.
      { verb: 'deleted',
        groupId: id,
        groupName: g.name,
        keys: [...g.members],
        parentName: parent ? parent.name : null,
        childNames: before.filter((x) => x.parent === id).map((x) => x.name) },
    );
  },

  add: (groupId, key) => get().addMany(groupId, [key]),

  drop: (groupId, key) => {
    const g = get().groups.find((x) => x.id === groupId);
    if (!g || !g.members.includes(key)) return false;
    return get().apply(
      (groups, now) => removeMember(groups, groupId, key, { now }),
      { verb: 'removed', groupId, groupName: g.name, keys: [key] },
      [key],
    );
  },

  /**
   * Take SEVERAL phases out of one group -- the ⋯ menu's "remove selected".
   *
   * That menu ran `drop` in a `forEach`, so removing five phases was five
   * changes and five unawaited `DELETE /members`. Measured against the real
   * service on the tree before c1's folder lock (`fdbc1f5b`): 30 of 30
   * trials left behind a phase the user had removed, 21 of 30 also raised
   * out of `os.replace`. The lock closed that -- the same harness now reads
   * 0 of 30 -- so this is no longer the fix for a lost update.
   *
   * It is still the right shape, for the two reasons that have nothing to
   * do with the race: undo takes back one fifth of a gesture otherwise,
   * with nothing on screen saying four presses are left; and a refusal
   * reports once instead of five times for one thing the user did.
   */
  dropMany: (groupId, keys) => {
    const g = get().groups.find((x) => x.id === groupId);
    if (!g) return false;
    const wanted = (keys || []).filter((k) => g.members.includes(k));
    if (!wanted.length) return false;
    return get().apply(
      (groups, now) => wanted.reduce(
        (acc, k) => removeMember(acc, groupId, k, { now }), groups),
      { verb: 'removed', groupId, groupName: g.name, keys: wanted },
      wanted,
    );
  },

  move: (groupId, key) => get().moveMany(groupId, [key]),

  /**
   * Alt-drag, and the named entry in the menu -- ALL the phases at once.
   *
   * This ran once per phase. Twelve Alt-dragged phases were twelve separate
   * changes, which meant two things and both were bad: Undo took back one
   * twelfth of the gesture with nothing on screen saying eleven presses
   * were left, and the twelve `POST /members` calls went to the server in
   * parallel, where each one is a read-modify-write of the same file with
   * no lock anywhere in `phase_collections.py`. Two that read the same
   * members and both write leave one phase silently unfiled -- twelve on
   * screen, nine in the folder, every request 200, nothing to see.
   *
   * One change, one request, one undo.
   */
  moveMany: (groupId, keys) => {
    const g = get().groups.find((x) => x.id === groupId);
    if (!g) return false;
    const { groups } = get();
    const wanted = keys.filter(Boolean);
    if (!wanted.length) return false;
    // WHICH phases each group loses, not merely which groups are touched.
    // The union was enough to send the move and not enough to undo it --
    // see `invert`'s `moved` branch for what that cost.
    const fromMap = {};
    groups
      .filter((x) => x.id !== groupId)
      .forEach((x) => {
        const held = wanted.filter((k) => x.members.includes(k));
        if (held.length) fromMap[x.name] = held;
      });
    const fromNames = Object.keys(fromMap);
    const wasHereKeys = wanted.filter((k) => g.members.includes(k));
    if (wasHereKeys.length === wanted.length && !fromNames.length) {
      return false;                                        // nothing to move
    }
    return get().apply(
      (gs, now) => wanted.reduce(
        (acc, k) => moveTo(acc, groupId, k, { now }), gs),
      { verb: 'moved', groupId, groupName: g.name, keys: wanted,
        fromNames, fromMap, wasHereKeys },
      wanted,
    );
  },

  /**
   * Make a group a child of another, or promote it back to the top.
   *
   * One level only, and `canNest` is the judge -- it is used here AND by the
   * drop target, so the menu and the drag cannot disagree about what is
   * allowed. `parentId = null` promotes.
   */
  setParent: (id, parentId) => {
    const { groups } = get();
    const g = groups.find((x) => x.id === id);
    if (!g) return false;
    const parent = parentId ? groups.find((x) => x.id === parentId) : null;
    if (parentId && !parent) return false;
    if (parentId && !canNest(groups, id, parentId)) return false;
    if ((g.parent || null) === (parentId || null)) return false;
    const was = groups.find((x) => x.id === g.parent);
    return get().apply(
      (gs, now) => gs.map((x) => (x.id === id
        ? { ...x, parent: parentId || null, updated: now } : x)),
      // NAMES in the change, because that is what the route takes; the ids
      // stay on screen. The two are not the same string -- "Al systems" is
      // stored as `Al_systems`.
      { verb: 'nested', groupId: id, groupName: g.name,
        parentName: parent ? parent.name : null,
        fromParentName: was ? was.name : null },
    );
  },

  /** Several phases at once -- the selection, dragged as one. */
  addMany: (groupId, keys) => {
    const g = get().groups.find((x) => x.id === groupId);
    if (!g) return false;
    // The ones that are NOT already in. Dragging twelve phases onto a group
    // that holds nine is one change to three, and the journal has to say
    // three -- the rule `addMember` enforces for one phase, which describing
    // the whole drag would undo.
    const fresh = keys.filter((k) => !g.members.includes(k));
    if (!fresh.length) return false;
    return get().apply(
      (groups, now) => fresh.reduce(
        (acc, k) => addMember(acc, groupId, k, { now }), groups),
      { verb: 'added', groupId, groupName: g.name, keys: fresh },
      fresh,
    );
  },
}));

export default useGroups;
