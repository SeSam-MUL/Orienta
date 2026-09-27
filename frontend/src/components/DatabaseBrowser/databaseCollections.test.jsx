// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import {
  entryIdentity, entryMemberKey, groupEntriesByCollection, flattenGroups,
  groupOrderSignature, libraryKeySet, movableMemberKeys,
} from './databaseGrouping';

/**
 * The two identity traps the task brief pins as prose, restated here as the
 * literal rule they came from — kept so the failure they describe is legible
 * without cross-referencing the brief. The tests that follow constrain the
 * SHIPPED function (`entryIdentity`, `groupEntriesByCollection`), not this
 * restatement.
 */

/** DatabasePage's bulk selection is keyed `entry.name || entry.filename`
 *  (line 324, unified with the React key at line 312 by this task) — this
 *  used to be the opposite precedence. Collection membership must use the
 *  SAME identity as the selection, or "move selected into collection" moves
 *  the wrong rows. */
const selectionId = (e) => e.name || e.filename;

/** `selectedRow` is an index into the FILTERED array (line 192). Any grouping
 *  that reorders rows must reset it, or the next click opens a different file
 *  than the one under the cursor. */
function regroup(filtered, groupOf) {
  const sorted = [...filtered].sort(
    (a, b) => String(groupOf(a)).localeCompare(String(groupOf(b))));
  return { rows: sorted, selectedRow: -1 };
}

describe('database browser grouping (the rule, as prose)', () => {
  it('uses the same identity as the bulk selection', () => {
    expect(selectionId({ name: 'Al.cif', filename: 'other.cif' })).toBe('Al.cif');
    expect(selectionId({ filename: 'Al.cif' })).toBe('Al.cif');
  });

  it('resets the selected row when the rows are regrouped', () => {
    const rows = [{ name: 'b' }, { name: 'a' }];
    const r = regroup(rows, (e) => e.name);
    expect(r.rows.map((e) => e.name)).toEqual(['a', 'b']);
    expect(r.selectedRow).toBe(-1);
  });
});

// ---------------------------------------------------------------------------
// Real tests against the shipped module — these constrain the actual code
// FileTable calls, not a restatement of it.
// ---------------------------------------------------------------------------

describe('entryIdentity', () => {
  it('prefers `name` over `filename`, matching the bulk-selection key', () => {
    expect(entryIdentity({ name: 'Al.cif', filename: 'other.cif' })).toBe('Al.cif');
    expect(entryIdentity({ filename: 'Al.cif' })).toBe('Al.cif');
    expect(entryIdentity({})).toBe('');
    expect(entryIdentity(null)).toBe('');
  });
});

describe('entryMemberKey', () => {
  it('is the filename stem, the same key space collection members use', () => {
    expect(entryMemberKey({ name: 'Al.cif' })).toBe('Al');
    expect(entryMemberKey({ filename: 'Al.xtal' })).toBe('Al');
    expect(entryMemberKey({ name: 'Al_Cu.mixed.cif' })).toBe('Al_Cu.mixed');
  });

  it('does NOT substring-match — no key can win off-order the way the backend defeats it', () => {
    // `crystal_hint_local_library.py`'s third pass checks whether a LIBRARY
    // KEY sits inside an SHT filename (`if k in sht_stem`), and deliberately
    // iterates keys LONGEST-FIRST across the whole set so `Al` cannot beat
    // `AlFeMnSi` for a filename that contains both (lines 347-358). This
    // function has no such whole-library ordering to borrow — it checks one
    // row in isolation — so it does not attempt the substring match at all,
    // exact-matching the stem instead. `AlFeMnSi.cif`'s own stem is
    // `AlFeMnSi`, never `Al`, regardless of what any collection contains.
    expect(entryMemberKey({ name: 'AlFeMnSi.cif' })).toBe('AlFeMnSi');
    expect(entryMemberKey({ name: 'AlFeMnSi.cif' })).not.toBe('Al');
  });
});

describe('libraryKeySet', () => {
  it('is the union of every collection member and every unassigned key', () => {
    const collections = [
      { name: 'Matrix', members: [{ key: 'Al' }, { key: 'Ni' }] },
      { name: 'Carbides', members: [{ key: 'WC' }] },
    ];
    const unassigned = [{ key: 'Fe' }];
    const keys = libraryKeySet(collections, unassigned);
    expect([...keys].sort()).toEqual(['Al', 'Fe', 'Ni', 'WC']);
  });

  it('is total: no collections, no unassigned, no crash', () => {
    expect(libraryKeySet(undefined, undefined).size).toBe(0);
    expect(libraryKeySet([], []).size).toBe(0);
  });
});

describe('movableMemberKeys — the CRITICAL fix: refuse a row that is not a real phase', () => {
  const validKeys = libraryKeySet(
    [{ name: 'Matrix', members: [{ key: 'Al' }, { key: 'Ni' }] }], []);

  it('a CIF/XTAL row whose stem IS a library key is movable', () => {
    expect(movableMemberKeys([{ name: 'Al.cif' }], validKeys)).toEqual(['Al']);
    expect(movableMemberKeys([{ name: 'Al.xtal' }], validKeys)).toEqual(['Al']);
  });

  it('an SHT row is refused: its stem is never a bare library key', () => {
    // The exact filename from the review: derived key `Ni (Ni) [cF4]
    // {20kV}`, nothing close to the library key `Ni`.
    const sht = { name: 'Ni (Ni) [cF4] {20kV}.sht' };
    expect(entryMemberKey(sht)).not.toBe('Ni');
    expect(movableMemberKeys([sht], validKeys)).toEqual([]);
  });

  it('a master h5 / MC h5 / dictionary row is refused the same way', () => {
    expect(movableMemberKeys([{ name: 'Ni_master_20kV_npx500.h5' }], validKeys)).toEqual([]);
    expect(movableMemberKeys([{ name: 'Ni_E20kV_sig70_n501_o0.h5' }], validKeys)).toEqual([]);
    expect(movableMemberKeys([{ name: 'Ni_master_20kV_npx500.dict.h5' }], validKeys)).toEqual([]);
  });

  it('a mix files only the movable rows, deduplicated', () => {
    const rows = [{ name: 'Al.cif' }, { name: 'Al.xtal' }, { name: 'Al_master.h5' }];
    expect(movableMemberKeys(rows, validKeys)).toEqual(['Al']);
  });

  it('an empty selection, or one with nothing movable, refuses cleanly', () => {
    expect(movableMemberKeys([], validKeys)).toEqual([]);
    expect(movableMemberKeys([{ name: 'unknown_master.h5' }], validKeys)).toEqual([]);
  });
});

describe('groupEntriesByCollection', () => {
  const alCif = { name: 'Al.cif', file_type: 'cif' };
  const alXtal = { name: 'Al.xtal', file_type: 'xtal' };
  const alFeMnSi = { name: 'AlFeMnSi.cif', file_type: 'cif' };
  const niCif = { name: 'Ni.cif', file_type: 'cif' };
  const collections = [
    { name: 'Matrix', members: [{ key: 'Al' }] },
    { name: 'Intermetallics', members: [{ key: 'AlFeMnSi' }] },
  ];

  it('assigns each row to the collection whose members contain its key', () => {
    const groups = groupEntriesByCollection([alCif, alXtal, alFeMnSi, niCif], collections);
    expect(groups.map((g) => g.collection?.name ?? null)).toEqual(
      ['Matrix', 'Intermetallics', null]);
    expect(groups[0].entries).toEqual([alCif, alXtal]);
    expect(groups[1].entries).toEqual([alFeMnSi]);
    expect(groups[2].entries).toEqual([niCif]);   // Ni is nobody's member -> unassigned
  });

  it('a key claimed by an earlier collection does not print a second time', () => {
    const dup = [
      { name: 'Both', members: [{ key: 'Al' }] },
      { name: 'AlsoBoth', members: [{ key: 'Al' }] },
    ];
    const groups = groupEntriesByCollection([alCif], dup);
    expect(groups).toHaveLength(1);
    expect(groups[0].collection.name).toBe('Both');
  });

  it('omits an unassigned group when nothing is unassigned', () => {
    const groups = groupEntriesByCollection([alCif], collections);
    expect(groups.map((g) => g.collection?.name ?? null)).toEqual(['Matrix']);
  });

  it('with no collections at all, every row is unassigned in one group', () => {
    const groups = groupEntriesByCollection([alCif, niCif], []);
    expect(groups).toHaveLength(1);
    expect(groups[0].collection).toBeNull();
    expect(groups[0].entries).toEqual([alCif, niCif]);
  });

  it('preserves collection order (children already adjacent to their parent,\n' +
     '   as the server sorts them), not first-row-seen order', () => {
    const parentFirst = [
      { name: 'Alloys', parent: null, members: [{ key: 'Ni' }] },
      { name: 'Alloys / Rare', parent: 'Alloys', members: [{ key: 'Al' }] },
    ];
    // alCif (key Al, the CHILD's member) appears in the input before niCif
    // (key Ni, the PARENT's member) — output order must still follow
    // `collections` order, not row-encounter order.
    const groups = groupEntriesByCollection([alCif, niCif], parentFirst);
    expect(groups.map((g) => g.collection.name)).toEqual(['Alloys', 'Alloys / Rare']);
  });

  it('flattenGroups is the exact concatenation, entries in group order', () => {
    const groups = groupEntriesByCollection([alCif, alXtal, alFeMnSi, niCif], collections);
    expect(flattenGroups(groups)).toEqual([alCif, alXtal, alFeMnSi, niCif]);
  });

  describe('IMPORTANT 2 — an exclusive collection always wins over the working set', () => {
    // Dual membership is the designed state: `phase_collections.py#assign`
    // deliberately leaves a non-exclusive (working-set) collection holding a
    // key when that same key is filed into an exclusive one. The browser
    // exists to show where a phase is ACTUALLY filed, so the real, exclusive
    // collection must win the grouping regardless of name-sort order.
    it('wins when the working set sorts BEFORE the real collection', () => {
      const cols = [
        { name: 'Arbeitsauswahl', exclusive: false, members: [{ key: 'Al' }] },
        { name: 'Matrix', exclusive: true, members: [{ key: 'Al' }] },
      ];
      const groups = groupEntriesByCollection([alCif], cols);
      expect(groups.map((g) => g.collection.name)).toEqual(['Matrix']);
    });

    it('wins when the working set sorts AFTER the real collection too', () => {
      const cols = [
        { name: 'Matrix', exclusive: true, members: [{ key: 'Al' }] },
        { name: 'Zeta working set', exclusive: false, members: [{ key: 'Al' }] },
      ];
      const groups = groupEntriesByCollection([alCif], cols);
      expect(groups.map((g) => g.collection.name)).toEqual(['Matrix']);
    });

    it('a key ONLY the working set holds still gets its own group', () => {
      const cols = [
        { name: 'Matrix', exclusive: true, members: [{ key: 'Ni' }] },
        { name: 'Arbeitsauswahl', exclusive: false, members: [{ key: 'Al' }] },
      ];
      const groups = groupEntriesByCollection([alCif], cols);
      expect(groups.map((g) => g.collection.name)).toEqual(['Arbeitsauswahl']);
    });
  });
});

describe('groupOrderSignature', () => {
  const collections = [{ name: 'Matrix', members: [{ key: 'Al' }] }];

  it('is identical for an identical grouping', () => {
    const g1 = groupEntriesByCollection([{ name: 'Al.cif' }], collections);
    const g2 = groupEntriesByCollection([{ name: 'Al.cif' }], collections);
    expect(groupOrderSignature(g1)).toBe(groupOrderSignature(g2));
  });

  it('changes when the order WITHIN a group changes', () => {
    // Two rows in the SAME collection: grouping cannot mask a within-group
    // reorder the way it would if each landed in its own bucket.
    const twoInOne = [{ name: 'Al.cif' }, { name: 'Al.xtal' }];
    const before = groupOrderSignature(groupEntriesByCollection(twoInOne, collections));
    const afterReorder = groupOrderSignature(
      groupEntriesByCollection([...twoInOne].reverse(), collections));
    expect(afterReorder).not.toBe(before);
  });

  it('changes when membership changes (a row moves group)', () => {
    const rows = [{ name: 'Al.cif' }, { name: 'Ni.cif' }];
    const before = groupOrderSignature(groupEntriesByCollection(rows, collections));
    const afterRegroup = groupOrderSignature(groupEntriesByCollection(rows, []));
    expect(afterRegroup).not.toBe(before);
  });

  it('does NOT change when only a non-identity field changes (a 5 s poll)', () => {
    const before = groupOrderSignature(
      groupEntriesByCollection([{ name: 'Al.cif', location: 'local' }], collections));
    const after = groupOrderSignature(
      groupEntriesByCollection([{ name: 'Al.cif', location: 'both' }], collections));
    expect(after).toBe(before);
  });
});
