/**
 * The hexagonal navigation map is wired into the Indexing page.
 *
 * NavigationCanvas.hex.test.jsx proves the component; nothing there proves the PAGE hands it the map and asks the
 * backend for the raster. The page is not mounted in tests (it needs a dozen API surfaces), so the wiring is read
 * out of the source, one statement per check.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';

const here = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(join(here, 'IndexingPage.jsx'), 'utf8');

const count = (text, needle) => text.split(needle).length - 1;

describe('IndexingPage: navigation map on a hexagonal scan', () => {
  it('asks for the display raster, in one place', () => {
    expect(page).toContain("ebsdApi.overview('bc', { display: 'raster' })");
    expect(page).not.toContain("ebsdApi.overview('bc')");
    // the file load, the file switch, and a geometry the page moved to with no question asked
    expect(count(page, 'loadNavImage();')).toBe(3);
    // the dataset switch asks again even when a question for the same file and geometry is out
    expect(count(page, 'loadNavImage(true);')).toBe(1);
  });

  it('keeps what the backend says it drew next to the picture', () => {
    expect(page).toContain('setNavDisplay(r.data.display || null);');
  });

  it('takes the scan geometry from the store, where the dataset it shows the grid of keeps it', () => {
    expect(page).toContain('const navGeometry = useDataStore((s) => s.scanGeometry);');
    expect(page).toContain('const hexMap = useMemo(() => makeHexMap(navGeometry), [navGeometry]);');
  });

  it('shows no picture that is not the raster the page expects', () => {
    expect(page).toContain('const navMismatched = !!navImage && !navStale && overviewMismatch(hexMap, navDisplay);');
    expect(page).toContain('navImage={navStale || navMismatched ? null : navImage}');
  });

  it('hands the map to the canvas, through the panel', () => {
    expect(page).toMatch(/^ {6}hexMap=\{hexMap\}$/m);                // <NavigationMapPanel ... />
    expect(page).toMatch(/^ {8}hexMap=\{hexMap\}$/m);                // <NavigationCanvas ... /> inside the panel
  });
});
