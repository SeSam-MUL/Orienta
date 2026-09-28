import { describe, it, expect } from 'vitest';
import {
  keyForPath, activeKeySet, activeKeySignature, narrowSelection, activeMissing,
} from './collectionFilter';

describe('keyForPath', () => {
  it('strips directory and extension, and keeps dots inside the name', () => {
    // Real library names. `Al4Fe1.7Si (τ11).cif` must not lose the ".7".
    expect(keyForPath('C:/x/Database/CIF_Library/Al4Fe1.7Si (τ11).cif'))
      .toBe('Al4Fe1.7Si (τ11)');
    expect(keyForPath('/mnt/d/lib/α-(AlFeSi) (Fe23Al81Si15).sht'))
      .toBe('α-(AlFeSi) (Fe23Al81Si15)');
    expect(keyForPath('Al.cif')).toBe('Al');
  });

  it('splits on a genuine Windows backslash path, not just forward slashes', () => {
    // `resolve` returns str(Path(...)) from a backend running on the Windows
    // machine this app ships for — that is backslash-separated, not the
    // drive-letter-with-forward-slashes or /mnt/... shapes used above.
    // An escaped-but-forward-slash-only split (`[\/]`) passes both of those
    // and still returns the whole path here.
    expect(keyForPath('C:\\Users\\x\\Database\\CIF_Library\\Al4Fe1.7Si (τ11).cif'))
      .toBe('Al4Fe1.7Si (τ11)');
  });

  it('leaves a key that is ALREADY a key alone, dots and all', () => {
    // The defect: it cut at the last dot whatever was there, so the one
    // key in this library that contains a dot came back as `Al4Fe1` --
    // matching nothing, so the phase silently left its group. Reachable
    // because `PhaseMapPanel.jsx:140` calls this on `p.key`, not on a
    // path: an EDS key may or may not carry `.cif`, and both spellings
    // have to survive.
    expect(keyForPath('Al4Fe1.7Si (τ11)')).toBe('Al4Fe1.7Si (τ11)');
    expect(keyForPath('Al')).toBe('Al');
    expect(keyForPath('sd_0302719')).toBe('sd_0302719');
  });

  it('and still strips the extension when the key also has a dot', () => {
    // Both spellings of the same phase must land on the same key, or the
    // two callers disagree about whether it is in the group.
    expect(keyForPath('Al4Fe1.7Si (τ11).cif'))
      .toBe(keyForPath('Al4Fe1.7Si (τ11)'));
  });

  it('keeps an extension it does not know rather than guessing it away', () => {
    // A key a little too long fails visibly when it matches nothing; a
    // key cut in the middle looks like missing data.
    expect(keyForPath('Al.wombat')).toBe('Al.wombat');
  });

  it('is total: no path, no crash', () => {
    expect(keyForPath('')).toBe('');
    expect(keyForPath(undefined)).toBe('');
  });
});

describe('activeKeySet', () => {
  const cols = [
    { name: 'P', parent: null, members: [{ key: 'Al' }] },
    { name: 'C', parent: 'P', members: [{ key: 'Si' }] },
    { name: 'Other', parent: null, members: [{ key: 'Ni' }] },
  ];

  it('returns null when nothing is active, meaning "do not filter"', () => {
    expect(activeKeySet(cols, null)).toBeNull();
    expect(activeKeySet(cols, '')).toBeNull();
  });

  it('includes the children of the active parent', () => {
    expect([...activeKeySet(cols, 'P')].sort()).toEqual(['Al', 'Si']);
  });

  it('a child on its own does not pull in its parent', () => {
    expect([...activeKeySet(cols, 'C')]).toEqual(['Si']);
  });

  it('an active name that no longer exists offers NOTHING, and is not null', () => {
    // This test used to say the opposite, with the reason "a renamed or
    // deleted collection must not empty every picker" -- a real concern,
    // and the cure was worse than the disease: it widened the run to the
    // whole library while the toolbar went on naming the collection. Too
    // few phases is a run that comes back and says so; too many is a run
    // that quietly indexes against structures the user excluded.
    //
    // Emptying is only acceptable BECAUSE it is now said out loud:
    // `activeMissing` is true, the toolbar says the group is gone, and the
    // start button is blocked with that reason.
    expect([...activeKeySet(cols, 'Geloescht')]).toEqual([]);
    expect(activeMissing(cols, 'Geloescht')).toBe(true);
  });

  it('and "nothing active" keeps its own answer, which is null', () => {
    expect(activeKeySet(cols, null)).toBe(null);
    expect(activeKeySet(cols, '')).toBe(null);
    expect(activeMissing(cols, null)).toBe(false);
  });
});

describe('activeKeySignature', () => {
  const cols = [
    { name: 'P', parent: null, members: [{ key: 'Al' }] },
    { name: 'C', parent: 'P', members: [{ key: 'Si' }] },
    { name: 'Other', parent: null, members: [{ key: 'Ni' }] },
  ];

  it('is empty when nothing is active — matches activeKeySet returning null', () => {
    expect(activeKeySignature(cols, null)).toBe('');
  });

  it('is identical across two DIFFERENT array instances with the same members', () => {
    // The whole point: a fresh `collections` array from another `load()`
    // must not look like a change if the active collection's own members
    // did not move.
    const reloaded = JSON.parse(JSON.stringify(cols));
    expect(activeKeySignature(reloaded, 'P')).toBe(activeKeySignature(cols, 'P'));
  });

  it('does not change when an UNRELATED collection changes', () => {
    const before = activeKeySignature(cols, 'P');
    const otherChanged = cols.map((c) =>
      c.name === 'Other' ? { ...c, members: [{ key: 'Ni' }, { key: 'W' }] } : c);
    expect(activeKeySignature(otherChanged, 'P')).toBe(before);
  });

  it('changes when the ACTIVE collection (or its child) gains or loses a member', () => {
    const before = activeKeySignature(cols, 'P');
    const grown = cols.map((c) =>
      c.name === 'C' ? { ...c, members: [{ key: 'Si' }, { key: 'Cu' }] } : c);
    expect(activeKeySignature(grown, 'P')).not.toBe(before);
  });

  it('is order-independent', () => {
    const shuffled = [
      { name: 'P', parent: null, members: [{ key: 'Al' }, { key: 'Zn' }] },
    ];
    const reversedMembers = [
      { name: 'P', parent: null, members: [{ key: 'Zn' }, { key: 'Al' }] },
    ];
    expect(activeKeySignature(shuffled, 'P')).toBe(activeKeySignature(reversedMembers, 'P'));
  });
});

describe('narrowSelection', () => {
  const all = ['Al', 'Si', 'Ni', 'WC'];

  it('keeps everything when no collection is active', () => {
    const r = narrowSelection(all, null, all);
    expect(r.keys).toEqual(all);
  });

  it('narrows to the intersection of collection and what works here', () => {
    const r = narrowSelection(all, new Set(['Al', 'Si', 'WC']), ['Al', 'Si', 'Ni']);
    expect(r.keys).toEqual(['Al', 'Si']);
    expect(r.inCollection).toBe(3);
    expect(r.usableHere).toBe(2);
  });

  it('reports an empty result instead of silently falling back to all', () => {
    // The caller has to be able to say WHY the list is empty.
    const r = narrowSelection(all, new Set(['WC']), ['Al', 'Si']);
    expect(r.keys).toEqual([]);
    expect(r.inCollection).toBe(1);
    expect(r.usableHere).toBe(0);
  });
});

describe('a collection whose id is NOT its name', () => {
  /**
   * Added after a live regression, and the fixtures here are the point.
   *
   * Every other fixture in this file names its collections `P`, `C`,
   * `Other` -- one word, no spaces. Schema 2 mints an id from the name by
   * replacing spaces with underscores, so for a one-word name the id and
   * the name are the SAME STRING and a test cannot tell which one the code
   * used. The round-trip test on the backend side had the same shape: its
   * collection was called "M".
   *
   * What that hid: for one commit `GET /` sent `state.active` and `parent`
   * as ids, `activeKeySet` compares against `name`, found nothing, and took
   * the "filter nothing" fallback -- so a run silently widened to the whole
   * library while the toolbar still read "Collection: Al systems". Every
   * name with a space triggered it, which is every name this machine
   * suggests.
   *
   * These fixtures carry an id that DIFFERS from the name, so the two can
   * never again be confused for each other here.
   */
  const shared = () => [
    { id: 'Al_systems', name: 'Al systems', parent: null,
      members: [{ key: 'Al' }, { key: 'Al13Fe4' }] },
    { id: 'Al_Fe_phases', name: 'Al-Fe phases', parent: 'Al systems',
      members: [{ key: 'Al7FeCu2' }] },
    { id: 'Mg_systems', name: 'Mg systems', parent: null,
      members: [{ key: 'Mg2Si' }] },
  ];

  it('narrows on the NAME, spaces and all', () => {
    expect([...activeKeySet(shared(), 'Al systems')].sort())
      .toEqual(['Al', 'Al13Fe4', 'Al7FeCu2']);
  });

  it('and the child is found by its parent NAME, not its parent id', () => {
    // Two places compare a reference: the active collection, and `parent`.
    // The regression hit the first; this pins the second.
    expect(activeKeySet(shared(), 'Al systems').has('Al7FeCu2')).toBe(true);
  });

  it('an id where a name belongs offers nothing -- it no longer widens', () => {
    // This is where the change announced itself, exactly as the earlier
    // version of this test said it would. It used to expect `null` -- "no
    // filter" -- and that was the regression's whole mechanism.
    expect([...activeKeySet(shared(), 'Al_systems')]).toEqual([]);
    expect(activeMissing(shared(), 'Al_systems')).toBe(true);
  });

  it('a deleted collection is the same case, and is reported the same way', () => {
    expect([...activeKeySet(shared(), 'Gone yesterday')]).toEqual([]);
    expect(activeMissing(shared(), 'Gone yesterday')).toBe(true);
  });

  it('a collection that IS there is never reported missing, even when empty', () => {
    // "This group holds nothing" and "this group is gone" are two
    // different screens, and only one of them is a problem.
    const cols = [{ id: 'Empty_one', name: 'Empty one', parent: null, members: [] }];
    expect([...activeKeySet(cols, 'Empty one')]).toEqual([]);
    expect(activeMissing(cols, 'Empty one')).toBe(false);
  });

  it('the signature follows the same rule, so the follow-effect agrees', () => {
    expect(activeKeySignature(shared(), 'Al systems'))
      .toBe(['Al', 'Al13Fe4', 'Al7FeCu2'].sort().join('\u0001'));
    expect(activeKeySignature(shared(), 'Al_systems')).toBe('');
  });
});
