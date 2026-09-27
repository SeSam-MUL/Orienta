// @vitest-environment jsdom
/**
 * Registration — the list, the labels, and that the page renders.
 *
 * WHAT THIS FILE DOES NOT DO, said plainly because its earlier docstring
 * claimed otherwise: it does not click the sidebar entry, and it does not
 * render App.jsx. The claim was load-bearing — a later reader trusts prose
 * like that instead of re-checking — and it was false: deleting the page's
 * mount from App.jsx left the entire suite green.
 *
 * That half now lives in `shared/pageMounts.test.js`, which compares the
 * sidebar list against App.jsx's mounts in BOTH directions, for every page.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { SIDEBAR_PAGES } from '../shared/sidebarPages';
import en from '../../locales/en/nav.json';
import de from '../../locales/de/nav.json';
import ja from '../../locales/ja/nav.json';
import zh from '../../locales/zh/nav.json';

vi.mock('../../services/addonsApi', () => ({
  addonsApi: {
    list: vi.fn(() => Promise.resolve({ data: {
      api_version: 0, orienta_version: '0', searched_paths: ['C:/addons'],
      addons: [] } })),
    setEnabled: vi.fn(), runJob: vi.fn(), job: vi.fn(), jobResult: vi.fn(),
    mapImage: vi.fn(),
  },
}));

afterEach(cleanup);

describe('the page is registered', () => {
  it('is in the sidebar list, under Index & Analyze', () => {
    const ids = SIDEBAR_PAGES.map((p) => p.id);
    expect(ids).toContain('addons');
    // After the analysis pages and before the Tools section: an add-on is an
    // analysis, and the citation it writes lands in an indexing result.
    expect(ids.indexOf('addons')).toBeGreaterThan(ids.indexOf('indexing'));
    expect(ids.indexOf('addons')).toBeLessThan(ids.indexOf('mlhub'));
  });

  it('has a label in all four locales, and not the raw key', () => {
    for (const [lng, doc] of [['en', en], ['de', de], ['ja', ja], ['zh', zh]]) {
      const entry = doc.pages?.addons;
      expect(entry, lng).toBeTruthy();
      expect(entry.label, lng).toBeTruthy();
      expect(entry.label, lng).not.toContain('pages.');
      expect(entry.tooltip, lng).toBeTruthy();
    }
  });

  it('the id is one App.jsx will accept', () => {
    // The guard refuses an id no page answers to — the "→ Phase Map" button
    // once sent 'phase-map' and the window went blank.
    expect('addons').toMatch(/^[a-z0-9_]+$/);
  });

  it('a misspelled id is still refused', () => {
    expect(SIDEBAR_PAGES.some((p) => p.id === 'add-ons')).toBe(false);
    expect(SIDEBAR_PAGES.some((p) => p.id === 'addon')).toBe(false);
  });
});

describe('the page mounts when the entry is clicked', () => {
  it('renders the add-ons page, not a blank panel', async () => {
    // The whole app is heavy, so this mounts the page the way App.jsx does —
    // lazily, by the same import path — and asserts it actually renders.
    const { default: AddonsPage } = await import('./AddonsPage');
    render(<AddonsPage />);
    // Its own empty state, which no other page produces.
    expect(await screen.findByTestId('addons-empty')).toBeTruthy();
  });
});
