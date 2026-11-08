import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';
import { regionFromDrag, regionFrame, maskToRaster, navRasterKey, navGridKey, navMeasurementKey } from './hexNavMap';

// Written by tests/hex_raster_oracle.py from the lattice alone; the other expectations are worked by hand.
const FIXTURE = JSON.parse(
  readFileSync(new URL('../../../../tests/fixtures/hex_raster_cases.json', import.meta.url), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');       // a 12 x 14 raster
const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));
const [H, W] = hexMap.shape;
/** The fractions of the centre of raster pixel (i, j). */
const centre = (i, j) => ({ rx: (j + 0.5) / W, ry: (i + 0.5) / H });

describe('regionFromDrag', () => {
  it('is the box of the cells whose centres the dragged rectangle contains, by hand', () => {
    // 3 x 3 scan, step 2 um, row pitch 1.732 um, odd rows shifted by +1 um; the raster has pixels of 0.5 um from
    // x = -1 um, y = -1.155 um. Pixels (3, 3)..(9, 9): x in [0.5, 4.0), y in [0.345, 3.845) um.
    // Centres inside: row 1 (y = 1.732) at x = 1, 3; row 2 (y = 3.464) at x = 2 -> cells (1,0), (1,1), (2,1).
    expect(regionFromDrag(hexMap, centre(3, 3), centre(9, 9)))
      .toEqual({ rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 });
  });

  it('does not care which corner the drag started in', () => {
    expect(regionFromDrag(hexMap, centre(9, 9), centre(3, 3)))
      .toEqual({ rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 });
  });

  it('selects the one cell under a click, never an empty region', () => {
    const corner = CASE.pixels.find((p) => p.cell !== null);
    expect(regionFromDrag(hexMap, centre(corner.i, corner.j), centre(corner.i, corner.j)))
      .toEqual({
        rowStart: corner.cell[0], rowEnd: corner.cell[0] + 1, colStart: corner.cell[1], colEnd: corner.cell[1] + 1,
      });
  });

  it('selects nothing when the drag starts or ends outside the picture', () => {
    expect(regionFromDrag(hexMap, { rx: -0.01, ry: 0.5 }, centre(9, 9))).toBeNull();
    expect(regionFromDrag(hexMap, centre(3, 3), { rx: 0.5, ry: 1.01 })).toBeNull();
    expect(regionFromDrag(hexMap, centre(3, 3), { rx: NaN, ry: 0.5 })).toBeNull();
    expect(regionFromDrag(hexMap, null, centre(3, 3))).toBeNull();
  });

  it('selects nothing for a drag that stays in the corner no hexagon covers', () => {
    const corner = CASE.pixels.find((p) => p.cell === null);
    expect(regionFromDrag(hexMap, centre(corner.i, corner.j), centre(corner.i, corner.j))).toBeNull();
  });
});

describe('regionFrame', () => {
  it('is the raster box that covers the hexagons of the region, as fractions, by hand', () => {
    // cells (1,0)..(2,1): centre x from 0 (row 2) to 3 (row 1), so the hexagons span x = -1 .. 4 um, pixels 0..10;
    // y from 1.732 - 1.155 = 0.577 um to 3.464 + 1.155 = 4.619 um, pixels 3 .. 12 (floor 3.46, ceil 11.55)
    const f = regionFrame(hexMap, { rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 });
    expect(f.x0).toBeCloseTo(0, 12);
    expect(f.x1).toBeCloseTo(10 / 14, 12);
    expect(f.y0).toBeCloseTo(3 / 12, 12);
    expect(f.y1).toBeCloseTo(1, 12);
  });

  it('is the whole picture for the whole scan', () => {
    const f = regionFrame(hexMap, { rowStart: 0, rowEnd: 3, colStart: 0, colEnd: 3 });
    expect([f.x0, f.y0, f.x1, f.y1]).toEqual([0, 0, 1, 1]);
  });

  it('is a box narrower than the picture for a region that does not reach the right edge', () => {
    // cells (0,0)..(0,1): x from -1 to 3 um -> pixels 0..8 of 14; y from -1.155 to 1.155 um -> pixels 0..5 of 12
    const f = regionFrame(hexMap, { rowStart: 0, rowEnd: 1, colStart: 0, colEnd: 2 });
    expect(f.x0).toBeCloseTo(0, 12);
    expect(f.x1).toBeCloseTo(8 / 14, 12);
    expect(f.y0).toBeCloseTo(0, 12);
    expect(f.y1).toBeCloseTo(5 / 12, 12);
  });
});

describe('maskToRaster', () => {
  const maskData = { mask2d: [1, 0, 0, 0, 1, 0, 0, 0, 1], shape: [3, 3] };      // cells (0,0), (1,1), (2,2)

  it('selects the pixels of the selected cells and skips the pixels no hexagon covers', () => {
    const out = maskToRaster(hexMap, maskData);
    expect(out.shape).toEqual([H, W]);
    let selected = 0;
    let skipped = 0;
    for (const p of CASE.pixels) {
      const k = p.i * W + p.j;
      if (p.cell === null) {
        expect(out.skip[k]).toBe(1);
        expect(out.mask2d[k]).toBe(0);
        skipped += 1;
      } else {
        expect(out.skip[k]).toBe(0);
        expect(out.mask2d[k]).toBe(maskData.mask2d[p.cell[0] * 3 + p.cell[1]]);
        if (out.mask2d[k]) selected += 1;
      }
    }
    // the sample must hold both kinds of pixel, or the checks above could not fail
    expect(selected).toBeGreaterThan(0);
    expect(skipped).toBeGreaterThan(0);
  });
});

describe('navRasterKey and navGridKey', () => {
  const hex = (extra = {}) => parseScanGeometry({ ...CASE.geometry, ...extra });

  it('name the fields that decide a hexagonal raster, by hand', () => {
    // 3 x 3 points, 2 um in a row, row pitch 1.7320508075688772 um, odd rows shifted by half a step
    expect(navRasterKey(hex())).toBe('hex|3|3|2|1.7320508075688772|0.5');
  });

  it('tell two rasters of the same shape apart, but not the grid', () => {
    const shifted = hex({ odd_row_shift: -0.5 });
    expect(navRasterKey(shifted)).not.toBe(navRasterKey(hex()));
    expect(navGridKey(shifted)).toBe(navGridKey(hex()));
    expect(navGridKey(hex())).toBe('hex|3|3');
    expect(navGridKey(hex({ n_cols: 4 }))).toBe('hex|3|4');
  });

  it('bind a square or missing geometry to nothing', () => {
    const square = parseScanGeometry({ kind: 'square', n_rows: 3, n_cols: 3, step_x: 1, step_y: 1 });
    expect(navRasterKey(square)).toBeNull();
    expect(navRasterKey(null)).toBeNull();
    expect(navGridKey(square)).toBe('square|3|3');
    expect(navGridKey(null)).toBeNull();
  });
});

describe('navMeasurementKey', () => {
  it('tells two measurements of one file apart, and a pattern source, and an APEX measurement', () => {
    const one = navMeasurementKey({ variant: { measurement: '1', pattern_source: 'processed' } });
    expect(one).toBe('1/processed/');
    expect(navMeasurementKey({ variant: { measurement: '2', pattern_source: 'processed' } })).toBe('2/processed/');
    expect(navMeasurementKey({ variant: { measurement: '1', pattern_source: 'unprocessed' } })).toBe('1/unprocessed/');
    expect(navMeasurementKey({ apex: { measurement_id: 'GUID-7' } })).toBe('//GUID-7');
  });

  it('is empty where the load answer says nothing', () => {
    expect(navMeasurementKey(null)).toBe('//');
    expect(navMeasurementKey({})).toBe('//');
    expect(navMeasurementKey({ variant: { measurement: null } })).toBe('//');
  });
});
