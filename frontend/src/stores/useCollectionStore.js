/**
 * Collections: server state, cached, plus the one thing that is purely local.
 *
 * `active` and `hidden` live on the server (Database/Collections/_local.json)
 * because they decide what a run is offered and must survive a reload of any
 * tab. `collapsed` is the triangle in the browser list — pure view, per
 * browser, so it stays in localStorage and never travels.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { collectionsApi } from '../services/api';

/**
 * Recursive structural equality over plain JSON-shaped data (strings,
 * numbers, booleans, null, arrays, plain objects) — the shape
 * `GET /api/phase-collections/` always returns.
 *
 * NOT `JSON.stringify(a) === JSON.stringify(b)`: this codebase already
 * documents why, in `DatabaseBrowser/DatabasePage.jsx`'s own `sameEntries` —
 * stringify depends on object KEY ORDER, so two objects with identical keys
 * and values inserted in a different order would compare unequal, and every
 * unchanged 5 s poll would look like a change again. Object comparison here
 * is by key set + per-key value, independent of enumeration order; array
 * comparison is index-by-index, because order WITHIN an array (e.g. a
 * collection's `members`, after a reorder) is real data, not incidental
 * serialisation order.
 */
export function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return a === b;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  if (typeof a === 'object') {
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    for (const k of ka) {
      if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
      if (!deepEqual(a[k], b[k])) return false;
    }
    return true;
  }
  return false;   // primitives already handled by `a === b` above (NaN excepted)
}

const useCollectionStore = create(
  persist(
    (set, get) => ({
      data: { collections: [], unassigned: [], problems: [], state: {} },
      loading: false,
      collapsed: {},            // { [name]: true }

      /**
       * `GET /api/phase-collections/` is not cheap — it walks three
       * libraries' filesystems for a signature, rglobs the Dictionary
       * library uncached, and runs a fuzzy master-file search per key
       * (`phase_collections.py#list_collections` -> `_master_map`). Every
       * page that shows a collection stays mounted (App.jsx only
       * `display:none`s an inactive page) and subscribes to `data`, so an
       * unconditional `set` here — a fresh object every single call, even
       * when nothing changed — re-renders the toolbar picker, the Phase
       * Tester, the EDS panel and the Indexing page on every poll of every
       * page that calls `load()`, exactly the cost `DatabasePage.jsx`'s own
       * `sameEntries` exists to avoid for its file list. `deepEqual` makes an
       * unchanged reply free: same content, no new object, no re-render
       * anywhere.
       */
      load: async () => {
        set({ loading: true });
        try {
          const res = await collectionsApi.list();
          set((s) => (deepEqual(s.data, res.data)
            ? { loading: false }
            : { data: res.data, loading: false }));
        } catch {
          // Keep whatever is on screen; a failed refresh must not empty the
          // pickers and thereby widen every run to the whole library.
          set({ loading: false });
        }
      },

      setActive: async (name) => {
        const hidden = get().data?.state?.hidden || [];
        try {
          await collectionsApi.putState({ active: name || null, hidden });
        } catch (err) {
          // Same reasoning as setHidden below: the toolbar picker (the only
          // caller today) awaits neither the click handler's promise nor a
          // rejection — a bare `await collectionsApi.putState(...)` here
          // would surface only as an unhandled promise rejection, with
          // nothing on screen to show it, and the click would silently
          // no-op (the dropdown closes, `load()` never runs to prove
          // anything happened either way). Logged so the failure is not
          // silent to anyone actually looking; `load()` below still runs
          // regardless, so the button reverts to the server's real
          // (unchanged) active collection rather than the UI pretending the
          // click took effect.
          console.error('phase-collections: failed to update active state', err);
        }
        await get().load();
      },

      setHidden: async (hidden) => {
        const active = get().data?.state?.active || null;
        try {
          await collectionsApi.putState({ active, hidden });
        } catch (err) {
          // The database browser's eye toggle (the only caller today) has no
          // error channel of its own — failing loudly here would just be an
          // unhandled promise rejection with nothing to show it to. Logged
          // so a failure is not silent to anyone actually looking; the
          // `load()` below still runs regardless, so on a failed write the
          // toggle visibly reverts to the server's real (unchanged) hidden
          // list rather than the UI pretending the click took effect.
          console.error('phase-collections: failed to update hidden state', err);
        }
        await get().load();
      },

      toggleCollapsed: (name) =>
        set((s) => ({ collapsed: { ...s.collapsed, [name]: !s.collapsed[name] } })),
    }),
    {
      name: 'phase-collections-view',
      partialize: (s) => ({ collapsed: s.collapsed }),
    },
  ),
);

export default useCollectionStore;
