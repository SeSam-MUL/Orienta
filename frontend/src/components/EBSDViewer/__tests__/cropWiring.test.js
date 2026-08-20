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

  it('is handed the save-crop handler, so a crop can leave the process', () => {
    // Same class of silent regression as the mask: the panel hides the button
    // when `onExport` is missing, so dropping this prop removes the ONLY way
    // to get a crop onto disk — with nothing failing and nothing on screen to
    // say the feature went away.
    expect(el).toContain('onExport={handleExportCrop}');
    expect(el).toContain('saving={cropSaving}');
  });
});

describe('crop export in EBSDViewer', () => {
  it('asks for a PATH, because the backend does the writing', () => {
    // `saveImage` shows a dialog AND writes renderer bytes; the cropped file
    // is written by Python from a path. Using the wrong channel here would
    // write an empty file over the user's chosen name.
    const open = source.indexOf('const handleExportCrop');
    expect(open, 'the viewer should have a crop-export handler').toBeGreaterThan(-1);
    const body = source.slice(open, source.indexOf('  const switchDataset', open));
    expect(body).toContain('window.electronAPI?.saveFile');
    expect(body).not.toContain('saveImage');
    // ...and it still works without Electron (plain browser via start_app.py).
    expect(body).toContain('askPrompt(');
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

  it('clears BOTH drag sentinels on that window mouse-up', () => {
    // They are set together at mouse-down, so they must be cleared together.
    // Clearing only `lassoDrawing` leaves `roiStartRef` holding the anchor of
    // a finished drag; a later press beginning off-image then extends it.
    const open = source.indexOf('const end = () => {');
    expect(open, 'the window mouse-up handler should be a block').toBeGreaterThan(-1);
    const body = source.slice(open, source.indexOf('};', open));
    expect(body).toContain('setLassoDrawing(false)');
    expect(body).toContain('roiStartRef.current = null');
  });

  it('extends the lasso path only while a drag it started is live', () => {
    // The invariant that makes the per-move rebuild unreachable rather than
    // unlikely: a path can only grow while `lassoDrawing` — which is exactly
    // when useSettledPath is holding the mask still. Appending with the flag
    // false makes the hook a passthrough and rebuilds the full mask on every
    // mouse-move (392 ms each at 361x461).
    const branch = source.indexOf("if (cropTool === 'lasso') {");
    expect(branch).toBeGreaterThan(-1);
    const head = source.slice(branch, branch + 1600);
    expect(head).toContain('if (!lassoDrawing) return;');
    // ...and the guard comes BEFORE the append, not after it.
    expect(head.indexOf('if (!lassoDrawing) return;'))
      .toBeLessThan(head.indexOf('setLassoPoints('));
  });
});
