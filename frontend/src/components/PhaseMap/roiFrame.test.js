import { describe, it, expect } from 'vitest';
import { definesRoi, roiDefiningLayers } from './roiFrame';

const L = (id, source, extra = {}) => ({ id, source, visible: true, ...extra });

describe('roiFrame — who decides the frame', () => {
  it('lets the indexing result and what is derived from it decide', () => {
    expect(definesRoi(L('phase', 'result'))).toBe(true);
    expect(definesRoi(L('kam', 'analysis'))).toBe(true);
    expect(definesRoi(L('forward-ncc', 'diagnostics'))).toBe(true);
    expect(definesRoi(L('refined-ncc', 'refinement'))).toBe(true);
  });

  it('gives no vote to layers that always cover the whole scan', () => {
    // These exist at full scan size whatever the indexed region was, so
    // measuring them snapped the view back to the whole map — the report that
    // started this: "it uses the whole thing again, not the crop".
    expect(definesRoi(L('eds:Fe Kα1', 'h5'))).toBe(false);
    expect(definesRoi(L('se:SE/Elektronenbild 1', 'h5'))).toBe(false);
    expect(definesRoi(L('vbse', 'ebsd'))).toBe(false);
  });

  it('ignores a hidden layer — an invisible map frames nothing', () => {
    expect(definesRoi(L('phase', 'result', { visible: false }))).toBe(false);
  });

  it('picks exactly the deciding layers out of a mixed stack', () => {
    const stack = [
      L('eds:Fe Kα1', 'h5'),
      L('ipf-z', 'result'),
      L('bc', 'result'),
      L('se:SE/Elektronenbild 1', 'h5'),
      L('kam', 'analysis', { visible: false }),
    ];
    expect(roiDefiningLayers(stack).map((l) => l.id)).toEqual(['ipf-z', 'bc']);
  });

  it('returns nothing rather than guessing when the result is hidden', () => {
    const stack = [L('eds:Fe Kα1', 'h5'), L('ipf-z', 'result', { visible: false })];
    expect(roiDefiningLayers(stack)).toEqual([]);
  });

  it('survives junk', () => {
    expect(definesRoi(null)).toBe(false);
    expect(definesRoi({})).toBe(false);
    expect(roiDefiningLayers(null)).toEqual([]);
  });
});
