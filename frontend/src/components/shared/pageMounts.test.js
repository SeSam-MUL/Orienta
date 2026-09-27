import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SIDEBAR_PAGES } from './sidebarPages';

const here = path.dirname(fileURLToPath(import.meta.url));
const APP = fs.readFileSync(path.join(here, '..', '..', 'App.jsx'), 'utf8');

/**
 * Every page in the sidebar is actually MOUNTED, and every mount is reachable.
 *
 * Registration is five hand-edited places with no registry, and the failure is
 * silent both ways: a sidebar entry with no mount navigates to a blank panel,
 * and a mount with no entry is dead code nobody can reach. Measured during the
 * final review: deleting the add-ons page's `<div data-page="addons">` from
 * App.jsx left the whole suite green — 230 files, 2222 tests — because no test
 * in this repo renders App.jsx.
 *
 * Read from the source rather than by rendering App: App pulls in every page,
 * every store and i18n, and a test that heavy would be skipped when it got
 * slow. This one is narrow on purpose and states what it checks.
 */
const mounted = new Set(
  [...APP.matchAll(/data-page="([a-z0-9_]+)"/g)].map((m) => m[1]),
);

const listed = SIDEBAR_PAGES.filter((p) => !p.sectionKey).map((p) => p.id);

describe('page registration', () => {
  it('the parser found the mounts at all', () => {
    // A regex that matched nothing would make both assertions below pass for
    // an App.jsx that mounts no pages whatsoever.
    expect(mounted.size).toBeGreaterThan(10);
  });

  it('every sidebar page is mounted in App.jsx', () => {
    const missing = listed.filter((id) => !mounted.has(id));
    expect(missing, 'these navigate to a blank panel').toEqual([]);
  });

  it('every mounted page is in the sidebar', () => {
    // The other direction: a mount nobody can navigate to. `h5viewer` is a
    // modal, not a page, and is handled before the guard in handleNavigate.
    const extra = [...mounted].filter(
      (id) => !listed.includes(id) && id !== 'h5viewer');
    expect(extra, 'these are mounted but unreachable').toEqual([]);
  });
});
