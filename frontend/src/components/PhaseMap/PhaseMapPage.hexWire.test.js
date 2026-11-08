/**
 * The wiring of the hexagonal map through the Phase Map page, read from the source.
 *
 * The component tests (PhaseMapPage.hexPointer.test.jsx) prove that CanvasInteractionLayer and
 * HoverProbeLayer convert correctly WHEN they are given a hexMap, and PhaseMapPage.hexGeometryChange.test.jsx
 * mounts the page itself. This file proves the page gives the map to both layers, and that the callers that
 * magnify data pixels keep asking for the index grid: a missing prop is silent (the old behaviour on the new
 * raster, which is wrong by up to a row and half a column).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const page = read('./PhaseMapPage.jsx');
const count = (text, re) => (text.match(re) || []).length;

describe('PhaseMapPage and the hexagonal map', () => {
  it('builds one hexMap from the geometry of what the routes draw: the shown result, else the dataset', () => {
    expect(page).toContain('const drawnScanGeometry = activeScanGeometry ?? datasetScanGeometry;');
    expect(page).toContain('const hexMap = useMemo(() => makeHexMap(drawnScanGeometry), [drawnScanGeometry]);');
  });

  it('learns the dataset geometry from the available-maps answer', () => {
    expect(page).toContain('setDatasetScanGeometry(parseScanGeometry(res.data?.dataset_scan_geometry));');
    expect(page).toContain('setDatasetScanGeometry(null);');
  });

  it('gives the canvas layer and the hover probe the same map', () => {
    expect(count(page, /hexMap=\{hexMap\}/g)).toBe(2);
  });

  it('tells the layer stack which raster every layer must be on, and when that is not final', () => {
    expect(page).toContain('expectedShape: hexMap ? hexMap.shape : null');
    expect(page).toContain('shapePending: datasetGeometryPending');
    expect(page).toContain('rasterMatches: hexMap ? hexMap.matches : null');
  });

  it('frames the region and marks the hover point in raster pixels', () => {
    expect(page).toContain('roiBbox={roiFrame}');
    expect(page).toContain('hoverPixel={hoverMarker}');
    expect(page).not.toContain('roiBbox={activeRoiBbox}');
  });

  it('converts the grid view and the stacked view the same way', () => {
    expect(page).toContain('onPixelClick={pickCell}');
    expect(page).toContain('if (out) pickCell(out.row, out.col);');
  });
});

describe('the callers that magnify data pixels ask for the index grid', () => {
  it.each([
    ['PhaseMapPage.jsx', page, "phaseMapApi.layer(zoomKind, { display: 'index' })"],
    ['IndexingPage.jsx', read('../Indexing/IndexingPage.jsx'), "phaseMapApi.layer(zoomKind, { display: 'index' })"],
    ['useEdsLayerStack.js', read('../EDS/hooks/useEdsLayerStack.js'), "phaseMapApi.layer(layer.id, { display: 'index' })"],
  ])('%s', (_name, text, call) => {
    expect(text).toContain(call);
  });

  it('the stack asks for the virtual BSE image on the raster', () => {
    expect(read('./hooks/useLayerStack.js')).toContain("ebsdApi.virtualBSE('gray', '', '', { display: 'raster' })");
  });
});
