// @vitest-environment jsdom
/**
 * "Show me this file in the database browser", arriving from the phase
 * library's profile card (spec §2.3, task 14).
 *
 * It arrives as a CATEGORY and a NAME, never as a row index: the library
 * does not know this table's ordering, and an index would go stale the
 * moment anything regrouped -- which this page does on every poll, and
 * which it already has a test for (`selectedRow` resets on regroup).
 */
import { describe, it, expect, afterEach, beforeEach } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { render, screen, cleanup, act } from '@testing-library/react';
import { FileTable, TAB_DEFS } from './DatabasePage';
import useDatabaseReveal, { TAB_OF_CATEGORY } from '../../stores/useDatabaseReveal';

const here = path.dirname(fileURLToPath(import.meta.url));
const SOURCE = fs.readFileSync(path.join(here, 'DatabasePage.jsx'), 'utf8');

const ENTRIES = [
  { name: 'Al.cif', filename: 'Al.cif', file_type: 'cif', location: 'local' },
  { name: 'Ni.cif', filename: 'Ni.cif', file_type: 'cif', location: 'local' },
];

beforeEach(() => useDatabaseReveal.getState().clear());
afterEach(cleanup);

describe('the table takes an incoming search', () => {
  it('and shows it in its own search box, so the reader can widen it again', async () => {
    const cifTab = TAB_DEFS.find((t) => t.id === 'cif');
    await act(async () => {
      render(
        <FileTable
          tabDef={cifTab}
          entries={ENTRIES}
          onRowClick={() => {}}
          loading={false}
          selectedFiles={new Set()}
          onToggleSelect={() => {}}
          onSelectAll={() => {}}
          onDownload={() => {}}
          collections={[]}
          incomingSearch="Al.cif"
        />,
      );
    });
    const box = screen.getByPlaceholderText(/search/i);
    expect(box.value).toBe('Al.cif');
  });

  it('and is left alone when nothing came in', async () => {
    const cifTab = TAB_DEFS.find((t) => t.id === 'cif');
    await act(async () => {
      render(
        <FileTable
          tabDef={cifTab}
          entries={ENTRIES}
          onRowClick={() => {}}
          loading={false}
          selectedFiles={new Set()}
          onToggleSelect={() => {}}
          onSelectAll={() => {}}
          onDownload={() => {}}
          collections={[]}
        />,
      );
    });
    expect(screen.getByPlaceholderText(/search/i).value).toBe('');
  });
});

/**
 * The page's half, read from the source.
 *
 * Rendering DatabasePage means its polls, its WebGL previews and its whole
 * fetch surface, and this repo already has one guard written this way
 * (`phaseListWiring.test.js`) for the same reason. What matters here is that
 * three specific things are true of the wiring, and each is one line.
 */
describe('the page hands it to exactly one table', () => {
  it('drains the request only when it is the visible page', () => {
    // Taking it while hidden would set a tab nobody is looking at and lose
    // the request -- `take()` clears.
    expect(SOURCE).toMatch(/if \(!isActive\) return;\s*\n\s*const pending = takeReveal\(\)/);
  });

  it('passes the name to the tab the link pointed at, and to no other', () => {
    expect(SOURCE).toMatch(/incomingSearch=\{revealed && revealed\.tab === tabDef\.id/);
  });

  it('switches to that tab', () => {
    expect(SOURCE).toMatch(/setActiveTab\(pending\.tab\)/);
  });

  it('and every category the library can send names a tab this page has', () => {
    // The two vocabularies differ on three of five; a category that mapped
    // to nothing would leave the browser where it was, looking as though
    // the link did nothing.
    const ids = new Set(TAB_DEFS.map((t) => t.id));
    for (const [category, tab] of Object.entries(TAB_OF_CATEGORY)) {
      expect(ids.has(tab), `${category} -> ${tab}`).toBe(true);
    }
  });
});
