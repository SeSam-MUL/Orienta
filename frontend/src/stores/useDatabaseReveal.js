/**
 * One-shot "show me this file in the Database browser" requests (spec §2.3,
 * task 14).
 *
 * A QUEUE OF ONE, consumed once, modelled on `useAddonLayerRequests` and for
 * the same reason it gives: the browser's tab and selection are its own
 * state, and a second copy here would be the defect this branch has shipped
 * four times -- state held twice with one copy updated. This holds only what
 * has not been handed over yet, and the page empties it.
 *
 * It carries the CATEGORY the phase library knows and the file NAME, not a
 * path: the two pages disagree about names for the same thing -- the library
 * says `h5_cache` and the browser's tab is `master` -- and the translation
 * belongs in one place rather than in every caller.
 */
import { create } from 'zustand';

/**
 * The phase library's category against the Database browser's tab.
 *
 * Not guessed: the categories come from `_CATEGORY_OF_SLOT` in
 * `backend/api/services/phase_library.py` and the tab ids from `TAB_DEFS` in
 * `DatabasePage.jsx`. They disagree on three of five, which is exactly why
 * this map exists instead of a string that happens to match.
 */
export const TAB_OF_CATEGORY = {
  cif_library: 'cif',
  xtal_library: 'xtal',
  h5_cache: 'master',
  dictionary_library: 'dict',
  sht_database: 'sht',
};

const useDatabaseReveal = create((set, get) => ({
  //: {category, name, tab} or null -- at most one, because a second request
  //: before the page has drained would only overwrite the first anyway.
  pending: null,

  reveal: (category, name) => {
    const tab = TAB_OF_CATEGORY[category];
    // An unknown category is a programming error, and sending the browser to
    // a tab that does not exist would leave it on whichever tab it was on,
    // looking as though the link did nothing.
    if (!tab) {
      // eslint-disable-next-line no-console
      console.warn(`[databaseReveal] no tab for category "${category}"`);
      return false;
    }
    set({ pending: { category, name, tab } });
    return true;
  },

  /** Read and clear in ONE step: the drain runs in an effect, and StrictMode
   *  runs effects twice in development. */
  take: () => {
    const p = get().pending;
    if (p) set({ pending: null });
    return p;
  },

  clear: () => set({ pending: null }),
}));

export default useDatabaseReveal;
