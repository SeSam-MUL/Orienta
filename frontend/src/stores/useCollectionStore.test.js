// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { deepEqual } from './useCollectionStore';

vi.mock('../services/api', () => ({
  collectionsApi: {
    list: vi.fn(),
    putState: vi.fn(),
  },
}));

import { collectionsApi } from '../services/api';
import useCollectionStore from './useCollectionStore';

describe('deepEqual', () => {
  it('is true for primitives, including through ===', () => {
    expect(deepEqual(1, 1)).toBe(true);
    expect(deepEqual('a', 'a')).toBe(true);
    expect(deepEqual(null, null)).toBe(true);
    expect(deepEqual(undefined, undefined)).toBe(true);
    expect(deepEqual(1, 2)).toBe(false);
    expect(deepEqual(null, undefined)).toBe(false);
  });

  it('compares objects by content, NOT by key insertion order', () => {
    // The exact class of bug `JSON.stringify` equality would reintroduce —
    // documented once already in DatabasePage.jsx's own `sameEntries`.
    const a = { name: 'x', members: [{ key: 'Al' }] };
    const b = { members: [{ key: 'Al' }], name: 'x' };
    expect(deepEqual(a, b)).toBe(true);
  });

  it('a different value under the same key is a real change', () => {
    expect(deepEqual({ a: 1 }, { a: 2 })).toBe(false);
  });

  it('a different key set is a real change, either direction', () => {
    expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    expect(deepEqual({ a: 1, b: 2 }, { a: 1 })).toBe(false);
  });

  it('arrays compare index-by-index — reordering IS a real change', () => {
    expect(deepEqual([1, 2], [1, 2])).toBe(true);
    expect(deepEqual([1, 2], [2, 1])).toBe(false);
    expect(deepEqual([1, 2], [1, 2, 3])).toBe(false);
  });

  it('recurses through the real collections shape', () => {
    const body = () => ({
      collections: [{ name: 'Matrix', parent: null, exclusive: true,
                       members: [{ key: 'Al', present: true }] }],
      unassigned: [{ key: 'Ni' }],
      problems: [],
      state: { active: 'Matrix', hidden: [] },
    });
    expect(deepEqual(body(), body())).toBe(true);
    const changed = body();
    changed.collections[0].members.push({ key: 'Si', present: true });
    expect(deepEqual(body(), changed)).toBe(false);
  });

  it('a null/undefined mismatch never throws trying to read .keys of it', () => {
    expect(deepEqual(null, { a: 1 })).toBe(false);
    expect(deepEqual({ a: 1 }, null)).toBe(false);
    expect(deepEqual(null, undefined)).toBe(false);
  });
});

describe('useCollectionStore.load — IMPORTANT 3: an unchanged reply is free', () => {
  const BODY = {
    collections: [{ name: 'Matrix', parent: null, exclusive: true,
                     members: [{ key: 'Al', present: true }] }],
    unassigned: [], problems: [], state: { active: null, hidden: [] },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    useCollectionStore.setState({
      data: { collections: [], unassigned: [], problems: [], state: {} }, loading: false,
    });
  });

  it('does not replace `data` when the reply deep-equals what is already there', async () => {
    // Two SEPARATE objects with identical content — exactly what two HTTP
    // replies look like (a fresh parse every time), the case this fix is for.
    collectionsApi.list.mockResolvedValue({ data: JSON.parse(JSON.stringify(BODY)) });
    await useCollectionStore.getState().load();
    const firstData = useCollectionStore.getState().data;

    collectionsApi.list.mockResolvedValue({ data: JSON.parse(JSON.stringify(BODY)) });
    await useCollectionStore.getState().load();
    const secondData = useCollectionStore.getState().data;

    // Reference equality: this is what stops every subscriber (toolbar
    // picker, Phase Tester, EDS panel, Indexing page — all kept mounted) from
    // re-rendering on an unchanged poll.
    expect(secondData).toBe(firstData);
  });

  it('still replaces `data` when the reply actually changed', async () => {
    collectionsApi.list.mockResolvedValue({ data: JSON.parse(JSON.stringify(BODY)) });
    await useCollectionStore.getState().load();
    const firstData = useCollectionStore.getState().data;

    const grown = JSON.parse(JSON.stringify(BODY));
    grown.collections[0].members.push({ key: 'Ni', present: true });
    collectionsApi.list.mockResolvedValue({ data: grown });
    await useCollectionStore.getState().load();

    expect(useCollectionStore.getState().data).not.toBe(firstData);
    expect(useCollectionStore.getState().data.collections[0].members).toHaveLength(2);
  });

  it('always ends with loading: false, on both branches', async () => {
    collectionsApi.list.mockResolvedValue({ data: JSON.parse(JSON.stringify(BODY)) });
    await useCollectionStore.getState().load();
    expect(useCollectionStore.getState().loading).toBe(false);
    await useCollectionStore.getState().load();   // second call: the dedup branch
    expect(useCollectionStore.getState().loading).toBe(false);
  });
});

describe('useCollectionStore.setHidden — MINOR: a failed PUT is not an unhandled rejection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCollectionStore.setState({
      data: { collections: [], unassigned: [], problems: [], state: { active: null, hidden: [] } },
      loading: false,
    });
    collectionsApi.list.mockResolvedValue({
      data: { collections: [], unassigned: [], problems: [], state: { active: null, hidden: [] } },
    });
  });

  it('a rejected putState does not throw out of setHidden, and still reloads', async () => {
    collectionsApi.putState.mockRejectedValueOnce(new Error('network down'));
    await expect(useCollectionStore.getState().setHidden(['Matrix'])).resolves.toBeUndefined();
    expect(collectionsApi.list).toHaveBeenCalled();   // the reload still ran
  });
});

/**
 * Whole-branch review (2026-09-26), "smaller" findings: `setActive` had no
 * `catch` around its `putState` call while its sibling `setHidden` does —
 * the toolbar picker's click handler (`CollectionPicker.jsx`'s
 * `handleSelect`) neither awaits nor catches the promise `setActive`
 * returns, so a failed write surfaced only as an unhandled rejection and the
 * click silently no-opped (the dropdown closes, nothing reverts, nothing is
 * logged). Same fix and same test shape as `setHidden` above.
 */
describe('useCollectionStore.setActive — a failed PUT is not an unhandled rejection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCollectionStore.setState({
      data: { collections: [], unassigned: [], problems: [], state: { active: null, hidden: [] } },
      loading: false,
    });
    collectionsApi.list.mockResolvedValue({
      data: { collections: [], unassigned: [], problems: [], state: { active: null, hidden: [] } },
    });
  });

  it('a rejected putState does not throw out of setActive, and still reloads', async () => {
    collectionsApi.putState.mockRejectedValueOnce(new Error('network down'));
    await expect(useCollectionStore.getState().setActive('Matrix')).resolves.toBeUndefined();
    expect(collectionsApi.list).toHaveBeenCalled();   // the reload still ran
  });
});
