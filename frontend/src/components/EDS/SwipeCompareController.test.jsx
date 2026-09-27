// @vitest-environment jsdom
/**
 * The A/B row sits in the ~400 px overlay box. A <select> sizes itself to
 * its longest option, and as a flex item it will not shrink below that
 * (min-width: auto). With a layer named
 * "SE Image (SE/Elektronenbild 33 (Input1))" both selects came out ~300 px
 * each and the row ran under the splitter into the tile grid (user report
 * 2026-08-31). jsdom has no layout, so this pins the two declarations that
 * let the browser shrink them instead.
 */
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import SwipeCompareController from './SwipeCompareController';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k) => k }),
}));

const layers = [
  { id: 'electron-SE', label: 'SE Image (SE/Elektronenbild 33 (Input1))', visible: true },
  { id: 'eds-Al', label: 'Al Kα1', visible: true },
  { id: 'eds-Fe', label: 'Fe Kα1', visible: false },
];

describe('SwipeCompareController fits its box', () => {
  it('lets both selects shrink below their longest option', () => {
    const { container } = render(
      <SwipeCompareController layers={layers} value={{ a: null, b: null }} onChange={() => {}} />,
    );
    const selects = container.querySelectorAll('select');
    expect(selects).toHaveLength(2);
    for (const s of selects) {
      // Both are needed: `flex: 1` shares the width, `min-width: 0` lifts the
      // implicit min-width: auto that would otherwise win.
      expect(s.style.minWidth).toBe('0px');
      expect(s.style.flexGrow).toBe('1');
      expect(s.style.flexShrink).toBe('1');
      expect(s.style.flexBasis).toBe('0px');
    }
  });

  it('offers only visible layers, so a hidden one cannot be picked as A or B', () => {
    const { container } = render(
      <SwipeCompareController layers={layers} value={{ a: null, b: null }} onChange={() => {}} />,
    );
    const labels = [...container.querySelectorAll('select')[0].options].map((o) => o.textContent);
    expect(labels).toEqual(['swipe.none', 'SE Image (SE/Elektronenbild 33 (Input1))', 'Al Kα1']);
  });
});
