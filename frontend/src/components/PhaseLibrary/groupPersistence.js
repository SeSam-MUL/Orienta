/**
 * Where groups live -- and the one seam that has to disappear cleanly.
 *
 * Groups are a library-wide fact: eight people share `Database/`, and a group
 * one of them makes has to be a group the next one opens. c1's schema 2 is
 * that, and it is merged; the browser store below is what the screen used
 * before it landed and is kept only so the flip has something to flip from.
 *
 * THE SEAM CHANGED SHAPE, and that is worth writing down. The first version
 * was `{load, save}` -- save the whole state, the way a file in a browser
 * wants. The backend is not a file: it has one route per change, and for a
 * good reason. In a shared library a whole-state write is a write that
 * silently reverts whatever a colleague did between your read and your save.
 * So the seam is a list of CHANGES, each of which says what it did, and each
 * store answers them its own way.
 *
 * b9 asked for a test that trips if the stopgap survives the switch. The
 * mechanism is here and not only in a test, because a test only fails on a
 * machine that runs it:
 *
 *   - `flags.backendReady` is the switch. One line, one file.
 *   - `transitionalStore` REFUSES TO RUN once it is on. A stray caller does
 *     not quietly keep a second copy of the groups in `localStorage` while
 *     the real ones are on disk -- it throws, on the first call, with a
 *     message naming this file.
 *   - `STORAGE_KEY` is exported so a test can walk the feature and assert
 *     that no other SHIPPED file names it. "State held twice with one copy
 *     updated" is the defect this branch has shipped five times; the sixth
 *     would be a second reader nobody remembered.
 */
import { collectionsApi } from '../../services/api';

/** The switch. */
export const flags = { backendReady: true };

/** The only key this feature may put in browser storage. */
export const STORAGE_KEY = 'orienta.phaseLibrary.groups.transitional';

const EMPTY = { groups: [], journal: [], schema: 1 };

function refuseIfBackendIsLive(what) {
  if (flags.backendReady) {
    throw new Error(
      `phase-library groups: ${what} reached the transitional browser store `
      + 'after the backend went live. Groups now live in the library folder; '
      + 'see `flags.backendReady` in groupPersistence.js. This is refused '
      + 'rather than kept, because a second copy in this browser would '
      + "silently disagree with everyone else's.",
    );
  }
}

// ==========================================================================
// The browser store. Transitional.
// ==========================================================================

function readRaw() {
  let raw = null;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return { ...EMPTY, unavailable: true };
  }
  if (!raw) return { ...EMPTY };
  try {
    const parsed = JSON.parse(raw);
    // Anything that is not the shape we wrote is treated as nothing: a
    // half-parsed group list would show phantom groups whose members cannot
    // be resolved, which reads on screen as data loss in the library itself.
    if (!parsed || !Array.isArray(parsed.groups)) return { ...EMPTY };
    return {
      schema: 1,
      groups: parsed.groups,
      journal: Array.isArray(parsed.journal) ? parsed.journal : [],
    };
  } catch {
    return { ...EMPTY };
  }
}

function writeRaw({ groups, journal }) {
  try {
    window.localStorage.setItem(
      STORAGE_KEY, JSON.stringify({ schema: 1, groups, journal }));
    return { saved: true };
  } catch (err) {
    return { saved: false,
      reason: String(err && err.message ? err.message : err) };
  }
}

export const transitionalStore = {
  kind: 'browser',

  load() {
    refuseIfBackendIsLive('load()');
    return Promise.resolve(readRaw());
  },

  /**
   * A change, applied to the browser's copy.
   *
   * The browser store has no per-operation storage, so every change is a
   * whole write -- which is safe HERE, because nobody else is writing to
   * this browser. The caller hands the state it has just computed; this
   * only has to keep it.
   */
  apply(change, next) {
    refuseIfBackendIsLive('apply()');
    return Promise.resolve(writeRaw(next));
  },
};

// ==========================================================================
// The library folder, over c1's routes.
// ==========================================================================

/** A collection as the endpoint sends it -> a group as the screen holds it. */
export function toGroup(c) {
  return {
    id: c.id,
    name: c.name,
    // As it arrives: a NAME. `toGroups` below turns it into an id.
    parent: c.parent || null,
    members: (c.members || []).map((m) => m.key),
    // The endpoint does not send these per collection yet; the screen shows
    // them only where it has them rather than inventing a blank author.
    author: c.author || null,
    updated: c.updated || null,
  };
}

/**
 * The whole list, with `parent` translated from a name into an id.
 *
 * ONE CURRENCY INSIDE THE SCREEN. The folder refers to a parent by NAME --
 * `parent: "Al systems"` -- and everything in here keys on id, because a
 * name is what the user edits. Comparing the two directly is the defect
 * that had just been found on the other side of this seam: on this machine
 * `Al systems` is stored under the id `Al_systems`, so a comparison that
 * mixes them finds nothing and quietly shows a flat list where there is a
 * tree. Translated once, at the boundary, where it can be tested.
 *
 * A parent that names no known collection is dropped to null rather than
 * kept: a child pointing at nothing would be hidden from the list, and an
 * invisible group is worse than a misplaced one.
 */
export function toGroups(collections) {
  const list = (collections || []).map(toGroup);
  const idOfName = new Map(list.map((g) => [g.name, g.id]));
  return list.map((g) => (g.parent
    ? { ...g, parent: idOfName.get(g.parent) || null } : g));
}

export const backendStore = {
  kind: 'backend',

  load() {
    return collectionsApi.list().then((r) => ({
      schema: 2,
      groups: toGroups(r.data.collections),
      // The journal lives with the groups on disk and this build does not
      // read it back yet: the screen keeps the entries it made this session
      // so an undo can say what it is undoing, and does not pretend to know
      // what anyone else did.
      journal: [],
    }));
  },

  /**
   * One change, one request -- never a whole-state write.
   *
   * A whole-state write to a shared folder reverts whatever a colleague did
   * between your read and your save, without a word. Each change below
   * touches exactly the group it names.
   *
   * `move` IS ONE REQUEST since c1's `move: true` (c1ac120d). It was three,
   * and the half-succeeded state that cost is gone with it.
   *
   * WHAT THE FLAG DOES THAT THIS SCREEN CANNOT SEE: the service removes the
   * phase from EVERY other collection, while `fromMap` in the change knows
   * only the groups this screen has read. A group a colleague created since
   * the last read is emptied of that phase and the undo will not restore
   * it. Bounded -- it needs a concurrent writer -- and the same staleness
   * as the name hazard below, with the same answer.
   *
   * ADDRESSED BY NAME, AND FOR ONE ROUTE THAT IS STILL FORCED. `delete`,
   * `assign`, `unassign_from` and `rename` now go through `resolve_ref` and
   * take either an id or a name (c1, 21f58a73). `PUT /update` -- the
   * NESTING route, used by `nested` and by `recreated` above -- does not:
   * `routes/phase_collections.py` looks it up in a name-keyed dict and 404s
   * on an id, not even casefolded. So this file keeps sending names
   * throughout rather than sending ids to four routes and a name to the
   * fifth, which is the arrangement most likely to be half-migrated by
   * somebody reading only one branch. Reported to c1.
   *
   * THE HAZARD, written down rather than hidden: a name is what the user
   * edits, so if somebody renames a group between this screen's read and
   * this write, the write lands on the wrong group or on none. The answer is
   * ids, which is why they exist -- available on four of the five routes
   * today, and worth switching to as soon as the fifth takes them.
   */
  apply(change) {
    const ok = () => ({ saved: true });
    const failed = (err) => {
      const detail = err && err.response && err.response.data
        && err.response.data.detail;
      return { saved: false,
        reason: detail || String(err && err.message ? err.message : err) };
    };
    // A new file gets its identity from the FOLDER, not from this browser:
    // the collection "Al systems" is stored as `Al_systems`, and the id the
    // screen minted while waiting is a placeholder. `reload` is how the
    // store says "read me again, I know something you do not".
    const remade = () => ({ saved: true, reload: true });
    let work;
    switch (change.verb) {
      case 'created':
        return collectionsApi.create({ name: change.groupName })
          .then(remade, failed);
      case 'renamed':
        work = collectionsApi.rename(change.fromName, change.groupName);
        break;
      case 'deleted':
        work = collectionsApi.remove(change.groupName);
        break;
      case 'added':
        work = collectionsApi.addMembers(change.groupName, change.keys);
        break;
      case 'removed':
        work = collectionsApi.removeMembers(change.groupName, change.keys);
        break;
      // ONE REQUEST, since c1's `move: true` landed (c1ac120d). It used to
      // be an add plus N removes, which could half-succeed, so this branch
      // carried a `partial` state the screen had to explain. A single
      // request either lands or does not, so that whole state -- and the
      // `stillIn` list, and the `landed`-on-partial rule in `useGroups` --
      // is gone rather than kept as dead handling with tests over it.
      case 'moved':
        work = collectionsApi.addMembers(
          change.groupName, change.keys, null, { move: true });
        break;
      // The two inverses. They exist because undoing a delete or a move is
      // ONE thing on screen and two in the folder, and the screen has
      // already done its half.
      case 'recreated':
        return collectionsApi.create({ name: change.groupName })
          .then(() => (change.keys && change.keys.length
            ? collectionsApi.addMembers(change.groupName, change.keys)
            : null))
          // Where it sat, and what sat under it. A delete promotes the
          // children (`pc.delete` rewrites their files), so putting the
          // container back is only a third of the undo; without these two
          // the screen showed the tree restored and the folder kept it
          // flat.
          .then(() => (change.parentName
            ? collectionsApi.update({ name: change.groupName,
              parent: change.parentName })
            : null))
          .then(() => (change.childNames || []).reduce(
            (chain, child) => chain.then(() => collectionsApi.update(
              { name: child, parent: change.groupName })),
            Promise.resolve()))
          .then(remade, failed);
      // Nesting. `parent: null` means "leave it alone" on this route, so
      // promoting a group to the top level has its own flag -- null cannot
      // say both things, and the route says so in its own comment.
      case 'nested':
        work = change.parentName
          ? collectionsApi.update({ name: change.groupName,
            parent: change.parentName })
          : collectionsApi.update({ name: change.groupName,
            clear_parent: true });
        break;
      // PER PHASE, and deliberately NOT with `move: true`.
      //
      // Each source group gets back exactly the phases IT held, from
      // `fromMap`; the union used to be sent to all of them, which put
      // phases into groups they had never been in. And the flag is wrong
      // here by construction: moving back into two source groups would be
      // two moves, and the second would take the phases out of the first.
      // An inverse of one move is several adds.
      case 'unmoved':
        work = Object.entries(change.fromMap || {}).reduce(
          (chain, [name, keys]) => chain.then(
            () => collectionsApi.addMembers(name, keys)),
          Promise.resolve())
          .then(() => ((change.removeHereKeys || []).length
            ? collectionsApi.removeMembers(
              change.groupName, change.removeHereKeys)
            : null));
        break;
      default:
        return Promise.resolve({ saved: false,
          reason: `unknown change ${change.verb}` });
    }
    return work.then(ok, failed);
  },
};

/** The store in force. One call site, so there is one answer. */
export function activeStore() {
  return flags.backendReady ? backendStore : transitionalStore;
}
