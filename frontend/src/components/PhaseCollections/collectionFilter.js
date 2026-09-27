/**
 * Pure collection maths, shared by all four consumers.
 *
 * `null` means "no collection active" everywhere in this file, and it is NOT
 * the same as an empty Set. An empty Set is "the active collection has no
 * phases here" — a real, reportable state. Collapsing the two is how a picker
 * ends up silently showing everything when the user asked for one collection,
 * or nothing when they asked for none.
 */

/** Basename without the final extension. Library stems contain dots, so only
 *  the LAST dot is dropped. */
export function keyForPath(path) {
  const s = String(path ?? '');
  if (!s) return '';
  const base = s.split(/[\\/]/).pop();
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(0, dot) : base;
}

/** Keys of the active collection, its children included. `null` = no filter. */
export function activeKeySet(collections, activeName) {
  if (!activeName) return null;
  const list = Array.isArray(collections) ? collections : [];
  const self = list.find((c) => c.name === activeName);
  if (!self) return null;         // renamed or deleted: filter nothing
  const keys = new Set((self.members || []).map((m) => m.key));
  for (const child of list) {
    if (child.parent === activeName) {
      for (const m of child.members || []) keys.add(m.key);
    }
  }
  return keys;
}

/**
 * A cheap, order-independent fingerprint of `activeKeySet`'s result — for a
 * `useEffect` dependency, so the "follow the active collection" effect in
 * `SinglePixelPhaseTestDialog.jsx` and `PhaseMapPanel.jsx` also reseeds when
 * the STILL-active collection's own membership changes, not only when its
 * NAME changes.
 *
 * `collections` gets a brand-new array reference from `useCollectionStore`
 * every time anything reloads it — a poll on a completely different page, an
 * edit to a completely unrelated collection (e.g. the database browser's own
 * periodic refresh). Depending on `collections` itself would reseed the
 * follow effect on every one of those, even though the ACTIVE collection's
 * members never moved. This string changes only when they actually did — the
 * same content, in any order, back-to-back, produces the same signature.
 */
export function activeKeySignature(collections, activeName) {
  const keys = activeKeySet(collections, activeName);
  if (!keys) return '';
  return [...keys].sort().join('\u0001');
}

/**
 * The selection a picker should hold, plus the two numbers it must display.
 *
 * `allKeys` is every key the picker knows; `availableKeys` is the subset this
 * particular place can use (the Phase Tester needs an .sht, Hough needs a CIF).
 */
export function narrowSelection(allKeys, collectionKeys, availableKeys) {
  const all = Array.isArray(allKeys) ? allKeys : [];
  const avail = new Set(availableKeys ?? all);
  if (!collectionKeys) {
    return { keys: all.slice(), inCollection: all.length, usableHere: all.length };
  }
  const keys = all.filter((k) => collectionKeys.has(k) && avail.has(k));
  return { keys, inCollection: collectionKeys.size, usableHere: keys.length };
}
