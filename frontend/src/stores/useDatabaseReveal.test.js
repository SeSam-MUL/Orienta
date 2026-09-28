// @vitest-environment node
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import useDatabaseReveal, { TAB_OF_CATEGORY } from './useDatabaseReveal';

beforeEach(() => useDatabaseReveal.getState().clear());
afterEach(() => vi.restoreAllMocks());

describe('the two pages disagree about names, and this is where it is settled', () => {
  it('translates every category the phase library can send', () => {
    // From `_CATEGORY_OF_SLOT` in the backend service. Three of the five
    // differ from the browser's tab id, which is why this map exists rather
    // than a string that happens to match.
    expect(TAB_OF_CATEGORY).toEqual({
      cif_library: 'cif',
      xtal_library: 'xtal',
      h5_cache: 'master',
      dictionary_library: 'dict',
      sht_database: 'sht',
    });
    expect(TAB_OF_CATEGORY.h5_cache).not.toBe('h5_cache');
  });

  it('refuses a category it does not know, loudly', () => {
    // Sending the browser to a tab that does not exist would leave it where
    // it was, looking as though the link did nothing.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(useDatabaseReveal.getState().reveal('nonsense', 'x.cif')).toBe(false);
    expect(useDatabaseReveal.getState().pending).toBeNull();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('nonsense'));
  });
});

describe('a queue of one, drained once', () => {
  it('holds what was asked for, with the tab already resolved', () => {
    expect(useDatabaseReveal.getState().reveal('h5_cache', 'Al_master.h5')).toBe(true);
    expect(useDatabaseReveal.getState().pending)
      .toEqual({ category: 'h5_cache', name: 'Al_master.h5', tab: 'master' });
  });

  it('take() reads and clears in one step', () => {
    // The drain runs in an effect, and StrictMode runs effects twice in
    // development: a drain that read from a render snapshot would see the
    // same request again and act on it twice.
    useDatabaseReveal.getState().reveal('cif_library', 'Al.cif');
    const first = useDatabaseReveal.getState().take();
    expect(first.name).toBe('Al.cif');
    expect(useDatabaseReveal.getState().take()).toBeNull();
  });

  it('a second request before the drain replaces the first', () => {
    useDatabaseReveal.getState().reveal('cif_library', 'Al.cif');
    useDatabaseReveal.getState().reveal('sht_database', 'Al.sht');
    expect(useDatabaseReveal.getState().pending.name).toBe('Al.sht');
  });
});
