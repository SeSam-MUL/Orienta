/**
 * The hexagonal overview is wired into the viewer.
 *
 * hexOverview.test.js proves the functions; nothing there proves the VIEWER calls them. The viewer is not mounted
 * in tests (it needs the stores, i18n and a dozen API surfaces), so, like cropWiring.test.js, the wiring is read
 * out of the source. Each check is narrow: it looks at one statement, so an unrelated edit to a 3 000-line file
 * cannot trip it, and the removal of the statement it guards must.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, '..', 'EBSDViewer.jsx'), 'utf8');
const api = readFileSync(join(here, '..', '..', '..', 'services', 'api.js'), 'utf8');

/** The text between `start` and the first `end` after it. */
function between(start, end) {
  const a = source.indexOf(start);
  expect(a, `the viewer should contain: ${start}`).toBeGreaterThan(-1);
  const b = source.indexOf(end, a + start.length);
  expect(b, `... followed by: ${end}`).toBeGreaterThan(a);
  return source.slice(a, b);
}

describe('the overview request', () => {
  it('asks for the display raster, and the api passes the option on', () => {
    expect(between('const fetchOverview', '} catch (e)')).toContain("{ display: 'raster' }");
    expect(api).toMatch(/overview: \(mode = 'mean', options = \{\}\) =>\s*api\.get\('\/api\/ebsd\/overview', \{ params: \{ mode, \.\.\.options \} \}\)/);
  });

  it('keeps what the backend says it drew next to the picture', () => {
    expect(between('const fetchOverview', '} catch (e)')).toContain('setOverviewDisplay(res.data?.display || null)');
  });

  it('shows no picture, and offers no click, that is not the raster this page expects', () => {
    expect(source).toContain('const overviewMismatched = !!overviewImage && !overviewStale && overviewMismatch(hexMap, overviewDisplay);');
    expect(source).toContain('const shownOverview = overviewStale || overviewMismatched ? null : overviewImage;');
    expect(source).toContain('src={`data:image/png;base64,${shownOverview}`}');
    expect(source).not.toContain('src={`data:image/png;base64,${overviewImage}`}');
    expect(source).toContain('{shownOverview ? (');
  });
});

describe('the pointer', () => {
  const calc = () => between('const calcOverviewPos', 'const handleOverviewPointer');

  it('reads a hexagonal picture through the pure module', () => {
    expect(calc()).toContain('if (hexMap) return overviewPointer(hexMap, rx, ry);');
  });

  it('measures the picture in the raster shape before the image has loaded', () => {
    expect(calc()).toContain('img.naturalWidth || frameShape[1]');
    expect(calc()).toContain('img.naturalHeight || frameShape[0]');
  });

  it('keeps the square scan on the rounding it always had', () => {
    expect(calc()).toContain('Math.round(ry * gridShape[0])');
    expect(calc()).toContain('sr: r, sc: c');
  });

  it('navigates to nothing in a corner no hexagon covers', () => {
    expect(between('const handleOverviewPointer', 'if (pos.r !== row')).toContain('pos.r === null');
  });

  it('draws a selection in raster pixels, never in the cell of the pointer', () => {
    const down = between('roiStartRef.current = {', 'else if (ovZoomed');
    expect(down).toContain('{ row: pos.sr, col: pos.sc }');
    expect(down).toContain('setLassoPoints([{ r: pos.sr, c: pos.sc }])');
    const drag = between('if (e.buttons === 1 && (cropTool || e.shiftKey) && gridShape) {', '} else if (e.buttons === 1 && !e.shiftKey');
    expect(drag).toContain('pos.sr');
    expect(drag).not.toMatch(/pos\.r\b/);
    expect(drag).not.toMatch(/pos\.c\b/);
  });
});

describe('the marks on the picture', () => {
  it('puts the crosshair where the raster pixel of the cell is', () => {
    expect(between('const crosshair = gridShape', 'The drawn selection'))
      .toMatch(/\(hexMap\s*\?\s*crosshairPercent\(hexMap, row, col\)/);
  });

  it('measures the selection frame and the lasso in the shape of the picture', () => {
    const box = between('data-roi-box', 'border:');
    expect(box).toContain('frameShape[1]');
    expect(box.match(/\+ boxPad/g)).toHaveLength(2);                 // the width and the height
    expect(box).not.toContain('gridShape');
    const lasso = between('data-lasso-path', 'fill={alpha');
    expect(lasso).toContain('frameShape');
    expect(lasso).not.toContain('gridShape');
  });

  it('prints the cells of the crop window in the caption on a hexagonal scan', () => {
    expect(source).toContain("t('overview.roiLabel', roiLabelBox(hexCrop, roi))");
  });

  it('prints no caption for a frame that holds no cell, instead of raster pixels as if they were cells', () => {
    expect(between('data-roi-box', "t('overview.roiLabel'")).toContain('(!hexMap || hexCrop) && (');
  });
});

describe('the crop window', () => {
  it('is the cell box and mask of the drawn shape on a hexagonal scan, the old hull otherwise', () => {
    expect(source).toContain('const hexCrop = useMemo(');
    expect(source).toContain('selectionCrop(hexMap, cropTool, maskPoints)');
    expect(source).toContain('(hexMap ? (hexCrop ? hexCrop.bbox : null) : boundsOf(selectionPoints))');
    expect(source).toContain('(hexMap ? (hexCrop ? hexCrop.mask : null) : maskForTool(cropTool, maskPoints, cropBbox))');
  });

  it('is computed after the points it depends on', () => {
    expect(source.indexOf('const maskPoints')).toBeGreaterThan(-1);
    expect(source.indexOf('const hexCrop')).toBeGreaterThan(source.indexOf('const maskPoints'));
    expect(source.indexOf('const cropBbox')).toBeGreaterThan(source.indexOf('const hexCrop'));
  });
});

describe('the image export', () => {
  const exp = () => between("if (exportFor === 'overview')", "return {\n      src: pattern");

  it('measures a display pixel, not a grid cell, and marks the cell on the raster', () => {
    // the scale and its unit come from one place (overviewScale, tested in hexOverview.test.js): um on a hexagonal scan
    expect(exp()).toContain('...overviewScale(hexMap, stepSize)');
    expect(exp()).not.toContain('unitLabel: units');
    expect(exp()).not.toContain('hexMap.spec.factor');
    expect(exp()).toContain('hexMap.pixelOf(row, col)');
    expect(exp()).toContain('shownOverview');
  });
});

describe('the four texts', () => {
  it.each(['en', 'de', 'ja', 'zh'])('has overview.rasterMismatch in %s', (lang) => {
    const file = JSON.parse(readFileSync(
      join(here, '..', '..', '..', 'locales', lang, 'ebsdviewer.json'), 'utf8'));
    expect(typeof file.overview.rasterMismatch).toBe('string');
    expect(file.overview.rasterMismatch.length).toBeGreaterThan(10);
  });

  it('addresses the reader formally in German', () => {
    const de = JSON.parse(readFileSync(
      join(here, '..', '..', '..', 'locales', 'de', 'ebsdviewer.json'), 'utf8'));
    expect(de.overview.rasterMismatch).toContain('Sie');
  });
});
