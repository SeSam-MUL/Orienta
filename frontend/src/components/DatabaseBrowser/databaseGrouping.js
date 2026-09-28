/**
 * Pure grouping maths for the database browser's file table.
 *
 * Extracted so the two identity traps the plan found can be pinned with real
 * tests against the function the table actually calls, not against a copy of
 * the rule re-typed into a test file (see `databaseCollections.test.jsx`).
 */
import { keyForPath } from '../PhaseCollections/collectionFilter';

/**
 * The identity a browser row is addressed by everywhere in this page: the
 * bulk-selection Set (`selectedFiles`), the "move to collection" action, and
 * now grouping. `DatabasePage.jsx` used `entry.name || entry.filename` for
 * selection but `entry.filename || entry.name` for the React `key` — two
 * different rows for a row where only one of the two fields is set and they
 * differ. Collection membership has to agree with the FIRST one (selection),
 * or ticking a row and moving it into a collection can silently move a
 * different row than the one the checkbox showed as checked.
 */
export function entryIdentity(entry) {
  return entry?.name || entry?.filename || '';
}

/**
 * The collection-membership key for a browser row: the identity's filename
 * stem, in the same key space `PhaseMember.key` and every picker already use
 * (see `collectionFilter.js#keyForPath`).
 *
 * This is an EXACT-match key, deliberately not the substring match the
 * backend's SHT-to-CIF attachment uses. `crystal_hint_local_library.py`'s
 * third pass checks whether a LIBRARY KEY appears INSIDE an SHT filename
 * (`if k in sht_stem`, e.g. `Al` inside `Al (Al) [cF4] {20kV}.sht`), and it
 * deliberately iterates candidate keys LONGEST-FIRST across the WHOLE key set
 * precisely so a short key like `Al` cannot win against a longer, unrelated
 * one like `AlFeMnSi` whose own SHT filename also happens to contain "Al"
 * (`crystal_hint_local_library.py:347-358`). Reproducing that safely needs
 * that same whole-library, longest-first ordering; checking one row against
 * one collection's keys in isolation has no such ordering to borrow and would
 * reintroduce exactly the collision the backend's ordering exists to defeat.
 * Exact match sidesteps the question entirely and reliably groups CIF and
 * XTAL rows (their stems ARE the library key,
 * `crystal_hint_local_library.py` `key = cif.stem` / `key = xtal.stem`);
 * SHT/Master/MC/Dictionary rows, whose filenames carry extra simulation
 * parameters, will not resolve to a real library key this way at all — see
 * `libraryKeySet`/`movableMemberKeys` below, which exist because of that.
 */
export function entryMemberKey(entry) {
  return keyForPath(entryIdentity(entry));
}

/**
 * Every key the library actually knows about: every collection's members,
 * plus every unfiled phase (`unassigned`, from `GET /api/phase-collections/`).
 * This is the set `movableMemberKeys` refuses a row against — a phase (CIF or
 * XTAL) always resolves into it via `entryMemberKey`; a derived artefact
 * (SHT, master `.h5`, MC `.h5`, dictionary `.h5`) essentially never does,
 * because its filename embeds simulation parameters the library key does not
 * carry.
 */
export function libraryKeySet(collections, unassigned) {
  const keys = new Set();
  for (const c of Array.isArray(collections) ? collections : []) {
    for (const m of c.members || []) keys.add(m.key);
  }
  for (const m of Array.isArray(unassigned) ? unassigned : []) keys.add(m.key);
  return keys;
}

/**
 * Which of `entries` can actually be filed into a collection: those whose
 * derived key is a REAL library key, deduplicated.
 *
 * A collection's member is a phase, and a phase is its crystal structure —
 * the CIF (or the XTAL derived from it). An `.sht`, a master `.h5` and an MC
 * `.h5` are derived ARTEFACTS of a phase, not phases in their own right,
 * and their filenames do not equal the phase's library key (see
 * `entryMemberKey`'s own docstring). Filing one under its own filename stem
 * would silently create a member the library can never resolve — permanently
 * `present: false`, inflating `counts.missingFromLibrary`, and reported by
 * `GET /resolve` as `not_in_library` for every indexing method, shrinking the
 * collection for indexing without any error anywhere. This function is the
 * refusal: a row whose key is not in `validKeys` is left out, silently to
 * the caller but never silently to the user — `DatabasePage.jsx` uses an
 * empty result here to disable the "move to collection" control and explain
 * why, rather than perform a move that would only look like it worked.
 */
export function movableMemberKeys(entries, validKeys) {
  const seen = new Set();
  for (const entry of Array.isArray(entries) ? entries : []) {
    const key = entryMemberKey(entry);
    if (key && validKeys.has(key)) seen.add(key);
  }
  return [...seen];
}

/**
 * Group `filtered` rows by the collection whose OWN members contain the
 * row's key (children are their own group, not folded into the parent — the
 * server already returns children immediately after their parent, sorted by
 * `(parent or name).lower()`, so keeping that order reads as a simple nested
 * list without this function having to know about parents at all).
 *
 * A row's key is looked up against every group's member set, and the FIRST
 * in `collections` order that claims it wins, so a phase in several groups
 * prints once rather than several times. Being in several is the ordinary
 * state now -- membership is a tag -- and this table has one row per FILE,
 * so it can only show one of them. Which groups a phase is really in is a
 * question the phase library answers, on the phase's card.
 *
 * A row matching no collection's members lands in the trailing group with
 * `collection: null` ("unassigned").
 *
 * Returns `[{ collection, entries }]`, omitting empty groups, in this order:
 * every collection that claimed at least one of THIS tab's filtered rows (in
 * `collections` order), then unassigned last (if
 * non-empty). When `filtered` is empty, or no collection claims anything, the
 * single unassigned group still carries every row — callers use its
 * presence/absence to decide whether grouping is worth rendering at all
 * (`collections.length === 0`).
 */
export function groupEntriesByCollection(filtered, collections) {
  const rows = Array.isArray(filtered) ? filtered : [];
  const list = Array.isArray(collections) ? collections : [];

  /**
   * ONE ROW PER FILE, so a phase in two groups is shown under the FIRST of
   * them in server order. That is a limitation of this table, not a
   * statement about membership -- the phase library is where a phase's
   * groups are listed, and its card names all of them.
   *
   * This used to run in two passes, exclusive groups first, so that a
   * phase in an old-style "working set" appeared under its folder instead.
   * Under the tag model there are no exclusive groups -- but files written
   * before schema 2 still carry `exclusive: false`, and `from_dict` keeps
   * it. So the row a phase appeared under DEPENDED ON THE AGE OF THE FILES
   * in the library: an old working set sorted second, a new group of the
   * same shape did not. Behaviour that turns on a field nobody sets any
   * more, in a way nobody can see, is worse than a rule somebody can read.
   * One pass, server order, stated here (found by c1).
   */
  const collectionForKey = new Map();
  for (const c of list) {
    for (const m of c.members || []) {
      if (!collectionForKey.has(m.key)) collectionForKey.set(m.key, c);
    }
  }

  const bucketByName = new Map();
  const unassigned = { collection: null, entries: [] };

  for (const entry of rows) {
    const key = entryMemberKey(entry);
    const collection = key ? collectionForKey.get(key) : undefined;
    if (!collection) {
      unassigned.entries.push(entry);
      continue;
    }
    let bucket = bucketByName.get(collection.name);
    if (!bucket) {
      bucket = { collection, entries: [] };
      bucketByName.set(collection.name, bucket);
    }
    bucket.entries.push(entry);
  }

  const ordered = list
    .map((c) => bucketByName.get(c.name))
    .filter(Boolean);
  if (unassigned.entries.length > 0) ordered.push(unassigned);
  return ordered;
}

/** Flatten `groupEntriesByCollection`'s output back into one row order. */
export function flattenGroups(groups) {
  const out = [];
  for (const g of groups) out.push(...g.entries);
  return out;
}

/**
 * A cheap signature of a grouping's actual row order, for a `useEffect`
 * dependency: identical order/membership -> identical string, so the effect
 * that resets the highlighted row only fires when the order really changed,
 * not on every render. Row identity, not the whole entry object — the size/
 * location fields the 5 s poll updates must NOT count as "regrouped".
 */
export function groupOrderSignature(groups) {
  return groups
    .map((g) => `${g.collection ? g.collection.name : ''}:${g.entries.map(entryIdentity).join(',')}`)
    .join('|');
}
