// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { LAYER_SOURCES, buildAddLayerOptions } from './layerSources';

const PAGE = readFileSync(fileURLToPath(new URL('./PhaseMapPage.jsx', import.meta.url)), 'utf8');

const needsSquare = () => Object.values(LAYER_SOURCES)
  .flatMap((group) => group.layers).filter((l) => l.requiresSquareGrid).map((l) => l.id).sort();

const ALL_IDS = Object.values(LAYER_SOURCES).flatMap((group) => group.layers).map((l) => l.id);

// Every layer the picker can offer has been decided about, by hand. A layer added to LAYER_SOURCES
// without a line in one of these two lists turns the test below red, so "does it look at a point's
// neighbours?" is answered for it before it ships, not discovered by a user on a hexagonal scan.
const NEIGHBOUR = [
  'kam',                // kernel average misorientation: a pixel against its neighbours
  'gos',                // grain orientation spread: needs grains, which are built from neighbours
];
const DECIDED_NOT_NEIGHBOUR = [
  'phase', 'ipf-z', 'ipf-x', 'ipf-y',            // one pixel's own phase / orientation
  'grain-boundaries',   // neighbours, but computed on the six neighbours of a hexagonal scan
  'local-anomaly',      // ditto: the median of the two-step hexagonal disc
  'bc', 'ci', 'uncertainty', 'ci-threshold',     // one pixel's own quality value
  'vbse',                                         // a detector-intensity map, pixel by pixel
  'forward-ncc', 'pc-sensitivity', 'pattern-residual',    // per-pixel comparisons with a simulation
  'phase-margin', 'assignment-source',            // per-pixel phase bookkeeping
  'refined-ncc', 'convergence-status', 'orientation-delta',    // per-pixel refinement results
  'pc-delta-x', 'pc-delta-y', 'pc-delta-l',
];

describe('layers that compare a point with its neighbours', () => {
  it('every layer of the catalogue has been classified, and only the neighbour ones carry the flag', () => {
    const decided = [...NEIGHBOUR, ...DECIDED_NOT_NEIGHBOUR];
    expect(new Set(decided).size).toBe(decided.length);          // nothing is on both lists
    expect([...decided].sort()).toEqual([...ALL_IDS].sort());    // nothing unclassified, nothing stale
    expect(needsSquare()).toEqual([...NEIGHBOUR].sort());
  });

  it('the backend refuses every analysis map of a hexagonal dataset, and no layer route', () => {
    const py = readFileSync(
      fileURLToPath(new URL('../../../../backend/api/services/hex_guard.py', import.meta.url)), 'utf8');
    expect(py).not.toContain('"/api/phasemap/layer"');
    expect(py).not.toContain('LAYER_KINDS');
    expect(py).not.toContain('NEIGHBOUR_MAPS');
    expect(py).toMatch(/"\/api\/analysis\/map\/\{map_type\}",\s*"analysis",\s*ANALYSIS_PAGE,\s*"The maps of the Analysis page",\s*key="analysis_map"/);
    // the picker offers two of the analysis maps (KAM, GOS); both stay held back
    for (const id of ['kam', 'gos']) expect(NEIGHBOUR).toContain(id);
  });

  const options = (squareGrid) => buildAddLayerOptions({
    layers: [], dynamicLayers: [], sourceLinked: true,
    availability: { squareGrid },
  });

  it('are held back, with the reason, on a hexagonal scan', () => {
    const opts = options({ ok: false, reason: 'Not available for a hexagonal scan yet' });
    for (const id of needsSquare()) {
      const o = opts.find((x) => x.value === id);
      expect(o.disabled).toBe(true);
      expect(o.tip).toBe('Not available for a hexagonal scan yet');
    }
  });

  it('leave every other layer alone', () => {
    const opts = options({ ok: false, reason: 'x' });
    expect(opts.find((o) => o.value === 'phase').disabled).toBe(false);
    expect(opts.find((o) => o.value === 'ipf-z').disabled).toBe(false);
    expect(opts.find((o) => o.value === 'ci').disabled).toBe(false);
  });

  it('are offered on a square scan, and when nothing is known', () => {
    for (const squareGrid of [{ ok: true }, undefined]) {
      const opts = options(squareGrid);
      for (const id of needsSquare()) expect(opts.find((o) => o.value === id).disabled).toBe(false);
    }
  });
});

describe('the Phase Map page feeds the gate', () => {
  it('reads the geometry of the shown result wherever it reads its region', () => {
    const regions = PAGE.match(/setActiveRoiBbox\(shown\??\.roi_bbox \?\? null\)/g) || [];
    const geometries = PAGE.match(/setActiveScanGeometry\(parseScanGeometry\(shown\??\.scan_geometry\)\)/g) || [];
    expect(regions.length).toBe(2);
    // plus the three effects that re-read the list for another reason (a new run, a refined result,
    // a refinement job ending) and read no region: they take the geometry with the id
    expect(geometries.length).toBe(regions.length + 3);
  });

  it('turns it into the squareGrid gate of the picker', () => {
    expect(PAGE).toMatch(/squareGrid: \{ ok: !isHexScan\(drawnScanGeometry\), reason: t\('phasemap:layers\.gateHexGrid'\) \}/);
  });
});
