// @vitest-environment node
//
// EBSDViewer cannot be mounted in a test, so the rule "wherever the grid is written, the scan
// geometry is written with it" is pinned in the source: the viewer has ONE place that publishes a
// new navigation grid (`applyGrid`), and it must write the grid of `ebsdInfo` through
// `ebsdInfoWithGrid` (which replaces `scan_geometry` along with `grid_shape`) and hand the store
// the parsed geometry. A second writer, or an old-style spread, turns this red.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(fileURLToPath(new URL('../EBSDViewer.jsx', import.meta.url)), 'utf8');

/** The text of `const applyGrid = useCallback(` up to its dependency array. */
function applyGridBody() {
  const start = SRC.indexOf('const applyGrid = useCallback(');
  expect(start, 'the viewer should define applyGrid').toBeGreaterThan(-1);
  const end = SRC.indexOf('}, [', start);
  return SRC.slice(start, end);
}

describe('the viewer publishes a navigation grid in one place', () => {
  it('has exactly one call of setNavigationGrid, and it is inside applyGrid', () => {
    const calls = [...SRC.matchAll(/setNavigationGrid\(/g)];
    expect(calls).toHaveLength(1);
    expect(applyGridBody()).toContain('setNavigationGrid(');
  });

  it('writes the geometry of ebsdInfo together with its grid', () => {
    expect(applyGridBody()).toContain('setEBSDLoaded(ebsdInfoWithGrid(ebsdInfo, geom))');
  });

  it('hands the store the parsed geometry, not the parent\'s', () => {
    expect(applyGridBody()).toContain('scanGeometry: geom.scanGeometry');
  });

  it('never writes a grid_shape by hand without the scan_geometry of the same dataset', () => {
    // the two literals that carry a grid_shape (entering the viewer with a file already open:
    // setEBSDLoaded and setFileData) each carry the dataset's scan_geometry next to it
    const literals = [...SRC.matchAll(/(?:setEBSDLoaded|setFileData)\(\{[^}]*?grid_shape[^}]*\}/g)];
    expect(literals).toHaveLength(2);
    for (const m of literals) expect(m[0]).toContain('scan_geometry: active.scan_geometry ?? null');
  });

  it('hands the store the resampled flag of the same dataset, wherever it hands it the geometry', () => {
    expect(applyGridBody()).toContain('hexResampled: geom.hexResampled');
    const literals = [...SRC.matchAll(/(?:setEBSDLoaded|setFileData)\(\{[^}]*?grid_shape[^}]*\}/g)];
    for (const m of literals) expect(m[0]).toContain('hex_resampled: active.hex_resampled ?? null');
  });

  it('the crop\'s fallback grid brings the crop\'s own geometry and flag, not the parent\'s', () => {
    // the block that runs when the new crop is not in /datasets: from its guard to the closing `});`
    const start = SRC.indexOf('if (!applied && cropWin) {');
    expect(start, 'the crop handler should have a fallback block').toBeGreaterThan(-1);
    const call = SRC.indexOf('applyGrid({', start);
    expect(call, 'the fallback block should publish through applyGrid').toBeGreaterThan(-1);
    const end = SRC.indexOf('});', call);
    const block = SRC.slice(call, end);
    expect(block).toContain('scanGeometry: parseScanGeometry(res.data?.scan_geometry)');
    expect(block).toContain('scanGeometryRaw: res.data?.scan_geometry ?? null');
    expect(block).toContain('hexResampled: res.data?.hex_resampled ?? null');
  });
});
