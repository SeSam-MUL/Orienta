/**
 * The one line whose regression would be silent.
 *
 * `cropTools.test.jsx` proves the tool picks the right mask; nothing there
 * proves the VIEWER hands that mask to CropPanel. Before Task 10 those props
 * were the literals `mask={null}` and `shape="rect"`, and putting either back
 * would break nothing loudly: the ellipse and lasso would keep drawing, the
 * panel would keep reporting, and every crop would quietly cut the bounding
 * box instead of the shape. No test would fail.
 *
 * That is not browser behaviour, so it does not fall under "the drawing itself
 * is not pinned here". Rendering EBSDViewer to assert it would mean standing up
 * the stores, i18n and a dozen API surfaces for two props, so the wiring is
 * read straight out of the source instead. The check is deliberately narrow: it
 * only looks inside the single <CropPanel …/> element, so unrelated edits to a
 * 3 000-line file cannot trip it.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, '..', 'EBSDViewer.jsx'), 'utf8');

/** The text of the single <CropPanel ... /> element in the viewer. */
function cropPanelElement() {
  const open = source.indexOf('<CropPanel');
  expect(open, 'the viewer should render a CropPanel').toBeGreaterThan(-1);
  expect(source.indexOf('<CropPanel', open + 1),
    'exactly one CropPanel, or this test is reading the wrong one').toBe(-1);
  const close = source.indexOf('/>', open);
  expect(close).toBeGreaterThan(open);
  return source.slice(open, close + 2);
}

describe('CropPanel wiring in EBSDViewer', () => {
  const el = cropPanelElement();

  it('is handed the active tool mask, not a hardcoded null', () => {
    expect(el).toContain('mask={cropMask}');
    expect(el).not.toContain('mask={null}');
  });

  it('is handed the active tool, not a hardcoded shape', () => {
    expect(el).toContain('shape={cropTool}');
    expect(el).not.toMatch(/shape="/);
  });

  it('is handed the box derived from the active tool', () => {
    expect(el).toContain('bbox={cropBbox}');
  });
});

describe('selection state in EBSDViewer', () => {
  it('builds the mask from the settled path, never from the live one', () => {
    // maskForTool must be fed `lassoMaskPath`. Passing `lassoPoints` would
    // still be correct on screen and would reintroduce the per-move rebuild
    // that useSettledPath exists to prevent — a silent performance regression.
    expect(source).toContain('selectionPointsFor(cropTool, lassoMaskPath, roi)');
    expect(source).toContain(
      'useSettledPath(lassoPoints, lassoDrawing, LASSO_BACKSTOP_MS)');
  });

  it('draws the outline from the LIVE path, so the drawing never lags', () => {
    expect(source).toMatch(/lassoPoints\.map\(/);
  });

  it('ends the drag on a window mouse-up, not only on the image', () => {
    // Without this a release outside the overview leaves the drag "in
    // progress" and the mask frozen at the shape it had when the hand left.
    expect(source).toContain("window.addEventListener('mouseup', end)");
    expect(source).toContain("window.removeEventListener('mouseup', end)");
  });
});
