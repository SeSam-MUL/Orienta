import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SIDEBAR_PAGES } from './sidebarPages';

const here = path.dirname(fileURLToPath(import.meta.url));
const SIDEBAR = fs.readFileSync(path.join(here, 'Sidebar.jsx'), 'utf8');

/**
 * Every icon a page NAMES must exist in the sidebar's map.
 *
 * Found in a browser, not by a test: adding a page with `icon: 'Puzzle'` put
 * the literal word "Puzzle" in the navigation, because `ICON_MAP[page.icon]`
 * was undefined and the fallback renders the NAME as text. Two files, each
 * correct on its own — the repo's dominant defect shape — and nothing
 * compared them.
 *
 * Read from the source rather than by importing the map, because the map is
 * a module-private const in a component file. That is a weaker test than
 * importing it, so it is narrow on purpose: it asserts the identifier appears
 * inside the ICON_MAP literal, which is what the lookup uses.
 */
function iconMapNames() {
  const start = SIDEBAR.indexOf('const ICON_MAP = {');
  expect(start, 'ICON_MAP moved or was renamed').toBeGreaterThan(-1);
  const end = SIDEBAR.indexOf('};', start);
  return new Set(
    SIDEBAR.slice(start, end)
      .replace('const ICON_MAP = {', '')
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean),
  );
}

describe('sidebar icons', () => {
  it('every page names an icon the map knows', () => {
    const known = iconMapNames();
    const missing = SIDEBAR_PAGES
      .filter((p) => p.icon && !known.has(p.icon))
      .map((p) => `${p.id} → ${p.icon}`);
    expect(missing, 'these render as their own NAME in the navigation')
      .toEqual([]);
  });

  it('the parser finds a map worth checking', () => {
    // A guard that parsed nothing would make the assertion above vacuously
    // true for every page, forever.
    expect(iconMapNames().size).toBeGreaterThan(10);
  });
});
