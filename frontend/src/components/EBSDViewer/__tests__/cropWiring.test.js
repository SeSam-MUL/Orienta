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

/**
 * The selection tool must OVERRULE the crosshair.
 *
 * Reported from the running app: with the tool bar visible, no rectangle,
 * ellipse or lasso could be drawn — the gesture still required Shift, which
 * this feature inherited from the image-export ROI and never re-examined once
 * an explicit tool picker existed. A visible tool that does nothing when you
 * drag on it is the worst of both: it advertises a mode the app is not in.
 *
 * The fix makes the tools armable toggles, so these four facts hang together
 * and none can be changed alone without the others becoming wrong:
 *   - no tool is armed by default (otherwise click-to-navigate is gone for
 *     everyone who never crops),
 *   - an armed tool draws on a PLAIN drag,
 *   - an armed tool suppresses navigate-on-click,
 *   - and pressing the armed tool disarms it, so navigation comes back.
 */
describe('a selection tool overrules the crosshair', () => {
  it('arms no tool by default, so navigation is the untouched default', () => {
    // Anchored on the cropTool declaration itself: the file holds 22 other
    // useState(null) calls, and a bare search for one would pass whatever
    // this line said.
    expect(source).toMatch(/const \[cropTool, setCropTool\] = useState\(null\)/);
    expect(source, 'rect must not be armed on mount')
      .not.toMatch(/const \[cropTool, setCropTool\] = useState\(['"]rect['"]\)/);
  });

  it('draws on a plain drag, not only on Shift+drag', () => {
    expect(source).toContain('e.buttons === 1 && (cropTool || e.shiftKey) && gridShape');
  });

  it('stops the plain drag from scrubbing while a tool is armed', () => {
    expect(source).toContain("e.buttons === 1 && !e.shiftKey && !cropTool");
  });

  it('suppresses navigate-on-click while a tool is armed', () => {
    expect(source).toContain('if (!e.shiftKey && !cropTool) handleOverviewPointer(e);');
  });

  it('starts a drag from the tool alone, with no modifier', () => {
    expect(source).toContain("(cropTool || e.shiftKey) && gridShape && e.button === 0");
  });

  it('keeps a way to pan a zoomed overview once the plain drag is taken', () => {
    expect(source).toContain('ovZoomed && e.ctrlKey && e.button === 0');
  });

  it('disarms when the armed tool is pressed again, and drops the selection', () => {
    const open = source.indexOf('const chooseTool');
    expect(open).toBeGreaterThan(-1);
    const body = source.slice(open, source.indexOf('}, [cropTool, clearSelection]);', open));
    expect(body).toContain('if (tool === cropTool)');
    expect(body).toContain('setCropTool(null)');
    expect(body, 'a disarmed tool must not leave a box nothing is drawing')
      .toMatch(/if \(tool === cropTool\) \{\s*clearSelection\(\);/);
  });
});

/**
 * A successful crop disarms the tool.
 *
 * Asked for from the running app, and it follows from the gesture model rather
 * than being a preference: after the crop the overview shows the CUT-OUT, and
 * an armed tool both takes the plain drag and suppresses navigate-on-click. So
 * the very next thing a user does — click through the patterns of what they
 * just cut — would instead start drawing a crop of the crop.
 */
describe('cropping disarms the selection tool', () => {
  const body = (() => {
    const open = source.indexOf('const handleCrop');
    expect(open, 'the viewer should have a handleCrop').toBeGreaterThan(-1);
    const close = source.indexOf('setCropBusy(false)', open);
    expect(close).toBeGreaterThan(open);
    return source.slice(open, close);
  })();

  it('disarms inside handleCrop, not somewhere a failed crop would also reach', () => {
    expect(body).toContain('setCropTool(null)');
  });

  it('disarms only after the request resolved, so a failed crop keeps the drawing', () => {
    const awaitAt = body.indexOf('await ebsdApi.crop(');
    const disarmAt = body.indexOf('setCropTool(null)');
    expect(awaitAt).toBeGreaterThan(-1);
    expect(disarmAt).toBeGreaterThan(awaitAt);
  });

  it('drops the selection with it, so no box outlives the tool that drew it', () => {
    const clearAt = body.indexOf('clearSelection()');
    const disarmAt = body.indexOf('setCropTool(null)');
    expect(clearAt).toBeGreaterThan(-1);
    expect(Math.abs(disarmAt - clearAt)).toBeLessThan(600);
  });
});
