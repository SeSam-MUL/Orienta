/**
 * Pure collection maths, shared by all four consumers.
 *
 * `null` means "no collection active" everywhere in this file, and it is NOT
 * the same as an empty Set. An empty Set is "the active collection has no
 * phases here" — a real, reportable state. Collapsing the two is how a picker
 * ends up silently showing everything when the user asked for one collection,
 * or nothing when they asked for none.
 */

/**
 * The library key behind a path, a file name, or a key that is already one.
 *
 * ONLY A KNOWN EXTENSION IS DROPPED. This used to cut at the last dot,
 * whatever was there -- and its own comment, "library stems contain dots",
 * states the very observation that makes that wrong. Exactly one key in
 * this library contains a dot, `Al4Fe1.7Si (τ11)`, and handed to this
 * function WITHOUT an extension it came back as `Al4Fe1`: a key that
 * matches nothing, so the phase drops out of its group in silence.
 *
 * Reachable, not hypothetical: `PhaseMapPanel.jsx:140` calls this on
 * `p.key` rather than on a path, precisely because an EDS key may or may
 * not carry `.cif`. Both spellings have to survive, and only a
 * known-extension rule does that.
 *
 * An unknown extension is KEPT rather than guessed away: a key that is a
 * little too long fails visibly when it matches nothing, while a key cut
 * in the middle looks like missing data.
 *
 * The same rule was already written, correctly, in `PhaseLibrary/phaseKey.js`
 * -- a module whose only importer was its own test. Rather than keep two
 * spellings of one idea, the live one takes the better rule and that file
 * goes.
 */
const KNOWN_EXTENSIONS = ['.cif', '.xtal', '.sht', '.h5oina', '.h5', '.nml',
                          '.str', '.json'];

export function keyForPath(path) {
  const s = String(path ?? '');
  if (!s) return '';
  const base = s.split(/[\\/]/).pop();
  const hit = KNOWN_EXTENSIONS.find((e) => base.toLowerCase().endsWith(e));
  return hit ? base.slice(0, -hit.length) : base;
}

/**
 * Whether the active reference names a collection that is actually there.
 *
 * `false` for "nothing is active" as well -- the caller asks this only to
 * decide whether to SAY something is wrong, and nothing active is not
 * wrong.
 */
export function activeMissing(collections, activeName) {
  if (!activeName) return false;
  const list = Array.isArray(collections) ? collections : [];
  return !list.some((c) => c.name === activeName);
}

/**
 * Keys of the active collection, its children included.
 *
 * `null` = NO COLLECTION IS ACTIVE, and only that. An active collection
 * that cannot be found returns an EMPTY SET -- "this group offers nothing"
 * -- and never `null`.
 *
 * That distinction is the whole point, and this function got it wrong for
 * one release. The old line read
 *     if (!self) return null;   // renamed or deleted: filter nothing
 * so a reference that resolved to nothing widened the run to the WHOLE
 * LIBRARY while the toolbar went on saying "Collection: Al systems". It
 * shipped the day `GET /` began sending ids where this compares names, and
 * every group whose name has a space was affected -- which is every name
 * this machine suggests. The same structure had bitten once before: the
 * phase tester running over the whole library while its button offered
 * "(0)".
 *
 * Widening is the dangerous direction. Too few phases is a run that comes
 * back and says so; too many is a run that quietly indexes against
 * structures the user excluded, and nothing on the screen disagrees.
 * Callers pair this with `activeMissing` to say WHY nothing is on offer.
 */
export function activeKeySet(collections, activeName) {
  if (!activeName) return null;
  const list = Array.isArray(collections) ? collections : [];
  const self = list.find((c) => c.name === activeName);
  if (!self) return new Set();
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
