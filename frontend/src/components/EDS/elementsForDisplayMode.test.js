/**
 * One tile per element in wt% / at%.
 *
 * The spectrum source lists one label per fitted family -- the Oxford reference scan: 12 labels
 * for AZtec's 8 windows. In wt% / at% mode `Fe Kα1` and `Fe Lα1,2` draw the
 * same quantified Fe map, so the tile grid showed it twice (four such pairs
 * on the Oxford reference scan). Counts mode keeps one tile per fitted family; window files are
 * unchanged.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { elementsForDisplayMode } from './edsLayerSources';

/** The Oxford reference scan's composition labels on the spectrum source, and its fit.lines. */
const SAMPLEB = ['Al Kα1', 'C Kα1,2', 'Cu Kα1', 'Cu Lα1,2', 'Fe Kα1', 'Fe Lα1,2',
                 'Mn Kα1', 'Mn Lα1,2', 'O Kα1', 'Si Kα1', 'Zn Kα1', 'Zn Lα1,2'];
const LINES = {
  Al: { label: 'Al Kα1', family: 'K' }, C: { label: 'C Kα1,2', family: 'K' },
  Cu: { label: 'Cu Lα1,2', family: 'L' }, Fe: { label: 'Fe Kα1', family: 'K' },
  Mn: { label: 'Mn Kα1', family: 'K' }, O: { label: 'O Kα1', family: 'K' },
  Si: { label: 'Si Kα1', family: 'K' }, Zn: { label: 'Zn Kα1', family: 'K' },
};

describe('elementsForDisplayMode', () => {
  it('wt% and at%: one label per element, the quantified line', () => {
    for (const mode of ['wt_pct', 'at_pct']) {
      expect(elementsForDisplayMode(SAMPLEB, mode, LINES)).toEqual(
        ['Al Kα1', 'C Kα1,2', 'Cu Lα1,2', 'Fe Kα1', 'Mn Kα1', 'O Kα1', 'Si Kα1', 'Zn Kα1']);
    }
  });

  it('counts: one tile per fitted family, all twelve', () => {
    expect(elementsForDisplayMode(SAMPLEB, 'counts', LINES)).toEqual(SAMPLEB);
  });

  it('a window file (no fit lines) is unchanged in every mode', () => {
    const win = ['Al Kα1', 'Cu Kα1', 'Cu Lα1,2', 'Fe Kα1'];
    for (const mode of ['counts', 'wt_pct', 'at_pct']) {
      expect(elementsForDisplayMode(win, mode, null)).toBe(win);
    }
  });

  it('never drops an element whose chosen label is not on offer', () => {
    expect(elementsForDisplayMode(['Cu Kα1', 'Fe Kα1'], 'at_pct',
                                  { Cu: { label: 'Cu Lα1,2', family: 'L' } }))
      .toEqual(['Cu Kα1', 'Fe Kα1']);
  });
});

describe('the EDS page offers what it lists', () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const src = readFileSync(resolve(here, 'EDSPage.jsx'), 'utf8');
  it('the tile grid, the add-layer picker and the background picker read shownElements', () => {
    expect(src).toMatch(/const shownElements = useMemo\(\s*\(\) => elementsForDisplayMode\(def\.elements/);
    expect(src).toMatch(/allMapsLayersFor\(\{\s*elements: shownElements,/);
    expect(src).toMatch(/for \(const el of \(shownElements \|\| \[\]\)\)/);
    expect(src).toMatch(/const els = \(shownElements \|\| \[\]\)\.map/);
    expect(src).toMatch(/fitStatus\?\.state === 'ready' \? \(fitStatus\.fit\?\.lines \?\? null\)/);
  });
});
