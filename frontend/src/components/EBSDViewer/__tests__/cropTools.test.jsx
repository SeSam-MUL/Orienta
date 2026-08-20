/**
 * The three selection tools, from the viewer's state to the mask that is sent.
 *
 * Two pure steps carry everything the tools decide:
 *   selectionPointsFor(tool, ...)  — which drawn points the ACTIVE tool owns
 *   maskForTool(tool, points, bbox) — which mask builder those points feed
 *
 * The drag itself is browser behaviour and is not pinned here, but the two
 * steps above are: they are where a wrong tool's mask would be sent, and where
 * a lasso would silently leak the rectangle drawn before it.
 *
 * Both are imported from ../navSelection — the very functions the viewer
 * calls — rather than restated here, so a test cannot pass against a copy of
 * the logic that the shipped viewer does not use.
 */
import { describe, it, expect } from 'vitest';
import { boundsOf, maskForTool, selectionPointsFor } from '../navSelection';

const ROI = { startRow: 0, startCol: 0, endRow: 4, endCol: 4 };

describe('selectionPointsFor', () => {
  it('gives the rectangle tool the two corners of the drawn box', () => {
    expect(selectionPointsFor('rect', [], ROI))
      .toEqual([{ r: 0, c: 0 }, { r: 4, c: 4 }]);
  });

  it('gives the ellipse tool the same box — one drag, two readings', () => {
    expect(selectionPointsFor('ellipse', [], ROI))
      .toEqual(selectionPointsFor('rect', [], ROI));
  });

  it('gives the lasso its traced path', () => {
    const path = [{ r: 1, c: 1 }, { r: 2, c: 3 }];
    expect(selectionPointsFor('lasso', path, null)).toBe(path);
  });

  it('never lets the lasso read a rectangle drawn before it', () => {
    // The box is still in `roi` (the image export reads it); the lasso must
    // not adopt it as its own path.
    expect(selectionPointsFor('lasso', [], ROI)).toEqual([]);
  });

  it('is empty when nothing is drawn', () => {
    expect(selectionPointsFor('rect', [], null)).toEqual([]);
    expect(selectionPointsFor('lasso', [], null)).toEqual([]);
  });
});

describe('maskForTool', () => {
  const pts = [{ r: 0, c: 0 }, { r: 4, c: 4 }];
  const bbox = boundsOf(pts);

  it('rectangle carries no mask', () => {
    expect(maskForTool('rect', pts, bbox)).toBeNull();
  });

  it('ellipse leaves the corners out', () => {
    const mask = maskForTool('ellipse', pts, bbox);
    expect(mask[0]).toBe(0);
    expect(mask[2 * 5 + 2]).toBe(1);
  });

  it('lasso follows the drawn path', () => {
    const path = [{ r: 0, c: 0 }, { r: 0, c: 4 }, { r: 4, c: 0 }];
    const b = boundsOf(path);
    const mask = maskForTool('lasso', path, b);
    expect(mask[1 * b.cols + 1]).toBe(1);
    expect(mask[3 * b.cols + 3]).toBe(0);
  });

  it('sends a different mask for each tool on the same box', () => {
    // The cell (0, 2) sits on the ellipse's vertical axis but off the
    // diagonal a lasso traced — so a tool mix-up cannot pass this.
    const ellipse = maskForTool('ellipse', pts, bbox);
    const lasso = maskForTool('lasso', pts, bbox);
    expect(ellipse[0 * 5 + 2]).toBe(1);
    expect(lasso[0 * 5 + 2]).toBe(0);
    expect(lasso[2 * 5 + 2]).toBe(1); // on the traced diagonal
  });

  it('nothing drawn carries no mask, for every tool', () => {
    for (const tool of ['rect', 'ellipse', 'lasso']) {
      expect(maskForTool(tool, [], null)).toBeNull();
    }
  });

  // The degenerate paths: lassoMask itself returns an all-zero mask for fewer
  // than three points, which would earn the user an HTTP 400 for a drag that
  // looked fine. maskForTool is where that is made to agree with the
  // three-point collinear case — what was traced is what is selected.
  it('a two-point lasso selects the line it traced, not nothing', () => {
    const path = [{ r: 0, c: 0 }, { r: 3, c: 3 }];
    const b = boundsOf(path);
    const mask = maskForTool('lasso', path, b);
    expect(mask[0 * b.cols + 0]).toBe(1);
    expect(mask[1 * b.cols + 1]).toBe(1);
    expect(mask[2 * b.cols + 2]).toBe(1);
    expect(mask[3 * b.cols + 3]).toBe(1);
    expect(mask[0 * b.cols + 3]).toBe(0); // off the line
  });

  it('a one-point lasso selects that single pixel', () => {
    const path = [{ r: 7, c: 9 }];
    const b = boundsOf(path);
    const mask = maskForTool('lasso', path, b);
    expect(Array.from(mask)).toEqual([1]);
  });

  it('a three-point collinear lasso still selects its line', () => {
    // Guards the agreement the two-point case was made to match.
    const path = [{ r: 0, c: 0 }, { r: 1, c: 1 }, { r: 2, c: 2 }];
    const b = boundsOf(path);
    const mask = maskForTool('lasso', path, b);
    expect(mask[0]).toBe(1);
    expect(mask[1 * b.cols + 1]).toBe(1);
    expect(mask[2 * b.cols + 2]).toBe(1);
  });

  it('an unknown tool falls back to the rectangle', () => {
    expect(maskForTool('something-else', pts, bbox)).toBeNull();
  });
});
