/**
 * Groups: membership is a TAG, not a folder (spec §2.5).
 *
 * The decision behind every function here, made by b9 after the personas
 * disagreed: a phase can be in several groups at once. Dragging ADDS;
 * removing is something you say explicitly. That is why there is no `move`
 * in this file except the one that is named `moveTo` and documents itself
 * as the exception -- Alt-drag, the named action in the ⋯ menu.
 *
 * WHY Alt AND NOT Ctrl OR SHIFT: under Windows those two already mean copy
 * and move in every file manager, and here they would mean something else.
 *
 * A GROUP HAS AN IDENTITY SEPARATE FROM ITS NAME. Today the shipped
 * collections key on `PhaseCollection.name` and the file is named after it,
 * so renaming on two machines breaks. An id that never changes is the whole
 * reason this is not just the old collections with a new screen.
 *
 * NOTHING HERE TOUCHES THE DISK. The store decides where a group lives; this
 * decides what a group IS, so the rules can be tested without one.
 */

/** A new group. `id` is supplied so the caller owns identity generation. */
export function makeGroup({ id, name, parent = null, author = null, now = null }) {
  const clean = String(name || '').trim();
  // An empty name is refused rather than defaulted: a group called
  // "Untitled" is a group nobody can find again, and the spec says so.
  if (!clean) throw new Error('a group needs a name');
  return {
    id,
    name: clean,
    parent,                    // one level of nesting, no more (spec §2)
    members: [],               // library keys, in the order they were added
    created: now,
    updated: now,
    author,
  };
}

/** Membership, as a set of group ids per phase -- the tag view of the data. */
export function groupsOfPhase(groups, key) {
  return groups.filter((g) => g.members.includes(key)).map((g) => g.id);
}

/**
 * Add a phase to a group. Adding twice is one membership, not two.
 *
 * Returns the same array when nothing changed, so a caller can tell a real
 * change from a no-op without comparing contents -- which matters for the
 * journal: an entry saying "added Al" when Al was already there is a lie in
 * a record eight people share.
 */
export function addMember(groups, groupId, key, { now = null } = {}) {
  let changed = false;
  const next = groups.map((g) => {
    if (g.id !== groupId || g.members.includes(key)) return g;
    changed = true;
    return { ...g, members: [...g.members, key], updated: now };
  });
  return changed ? next : groups;
}

/** Remove a phase from ONE group. The others keep it -- that is the point. */
export function removeMember(groups, groupId, key, { now = null } = {}) {
  let changed = false;
  const next = groups.map((g) => {
    if (g.id !== groupId || !g.members.includes(key)) return g;
    changed = true;
    return { ...g, members: g.members.filter((m) => m !== key), updated: now };
  });
  return changed ? next : groups;
}

/**
 * The exception: put a phase HERE and take it out of every other group.
 *
 * Alt-drag, and the named action in the ⋯ menu. It exists because the
 * personas split on it -- some file, some tag -- and the answer was to make
 * tagging the default and moving a thing you ask for by name.
 */
export function moveTo(groups, groupId, key, { now = null } = {}) {
  return groups.map((g) => {
    if (g.id === groupId) {
      return g.members.includes(key)
        ? g
        : { ...g, members: [...g.members, key], updated: now };
    }
    return g.members.includes(key)
      ? { ...g, members: g.members.filter((m) => m !== key), updated: now }
      : g;
  });
}

/** Rename, by id. The name is a label; the id is the thing. */
export function renameGroup(groups, groupId, name, { now = null } = {}) {
  const clean = String(name || '').trim();
  if (!clean) throw new Error('a group needs a name');
  return groups.map((g) => (
    g.id === groupId ? { ...g, name: clean, updated: now } : g));
}

/**
 * Delete a group. ITS CHILDREN ARE PROMOTED, not deleted with it.
 *
 * This said the opposite, and argued for it, and was wrong: the folder
 * promotes them (`phase_collections.delete` sets `c.parent = None` and
 * rewrites each child). So the screen removed a child that the folder
 * kept. The user saw "Al systems" and its child "Al-Fe phases (12)" both
 * vanish with no message, and the child came back at the top level on the
 * next start -- or, worse, made a later "already exists" refusal
 * incomprehensible for a group the screen said was gone.
 *
 * Promoting is also the kinder rule on its own merits: deleting a container
 * is not a statement about what was inside it, and twelve memberships are
 * not something to throw away as a side effect. One truth, and it is the
 * folder's.
 */
export function deleteGroup(groups, groupId) {
  return groups
    .filter((g) => g.id !== groupId)
    .map((g) => (g.parent === groupId ? { ...g, parent: null } : g));
}

/**
 * The counts beside each group, measured on what is CURRENTLY SHOWN.
 *
 * §2.5: "Zähler reagieren auf Suche und Facetten. Heute bleibt `Mg-Systeme
 * 3` stehen, während die Liste 0 zeigt." So the count takes the visible
 * keys, not the library.
 *
 * `total` is also returned, because "2 of 7 shown" is a different sentence
 * from "2", and a group that has gone to zero under a filter must not read
 * the same as a group that is empty.
 */
export function groupCounts(groups, visibleKeys) {
  const visible = new Set(visibleKeys);
  return groups.map((g) => ({
    id: g.id,
    shown: g.members.filter((k) => visible.has(k)).length,
    total: g.members.length,
  }));
}

/** Phases in no group at all, out of what is shown. */
export function ungrouped(groups, visibleKeys) {
  const filed = new Set(groups.flatMap((g) => g.members));
  return visibleKeys.filter((k) => !filed.has(k));
}

/**
 * One journal entry per change, because eight people share a library.
 *
 * The verb, the group, the phase, who and when -- and nothing else. A
 * journal that also carried the whole group would be a second copy of the
 * state, which is the defect this branch has shipped four times.
 */
export function journalEntry(verb, { groupId, groupName, key, author, now }) {
  return { verb, groupId, groupName, key: key || null, author, at: now };
}

/** One level of nesting, checked rather than assumed. */
export function canNest(groups, childId, parentId) {
  if (childId === parentId) return false;
  const parent = groups.find((g) => g.id === parentId);
  if (!parent) return false;
  // The parent must be a root: nesting under a child would be two levels.
  if (parent.parent) return false;
  // And the child must not itself be a parent.
  return !groups.some((g) => g.parent === childId);
}
