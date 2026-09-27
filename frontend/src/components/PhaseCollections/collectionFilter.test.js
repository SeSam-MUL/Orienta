import { describe, it, expect } from 'vitest';
import { keyForPath, activeKeySet, activeKeySignature, narrowSelection } from './collectionFilter';

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

  it('an active name that no longer exists does not filter everything away', () => {
    // A renamed or deleted collection must not empty every picker.
    expect(activeKeySet(cols, 'Geloescht')).toBeNull();
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
