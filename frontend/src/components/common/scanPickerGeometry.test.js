import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';
import {
  pickerAspect, pickAt, markerAt, rowColOf, shownPicture, pictureEntry, pictureAfterFailure, pictureProblem,
  pickerRect, pickerInner,
} from './scanPickerGeometry';

// Written by tests/hex_raster_oracle.py from the lattice alone; the other expectations are worked by hand.
const FIXTURE = JSON.parse(
  readFileSync(new URL('../../../../tests/fixtures/hex_raster_cases.json', import.meta.url), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');       // a 12 x 14 raster
const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));
// What the backend stamps on the picture of this scan, in its own names (the oracle's description of the raster)
const STAMP = {
  kind: 'hex_raster', factor: CASE.spec.factor, shape: [CASE.spec.height, CASE.spec.width],
  step_um: CASE.spec.step_um, x0: CASE.spec.x0, y0: CASE.spec.y0,
};

describe('pickerAspect', () => {
  it('is the width over the height of the raster on a hexagonal scan, of the grid otherwise', () => {
    expect(pickerAspect(hexMap, 3, 3)).toBeCloseTo(14 / 12, 12);
    expect(pickerAspect(null, 10, 12)).toBeCloseTo(1.2, 12);
    expect(pickerAspect(null, 0, 12)).toBe(12);                      // the old guard against a zero height
  });
});

describe('pickerInner and pickerRect', () => {
  // an element as the browser reports it: a 280 x 240 box at (100, 50) with a 1 px frame
  const el = {
    getBoundingClientRect: () => ({ left: 100, top: 50, width: 280, height: 240 }),
    clientLeft: 1, clientTop: 1, clientWidth: 278, clientHeight: 238,
  };

  it('take the frame off the box on a hexagonal scan: 280 x 240 -> 278 x 238, from (101, 51)', () => {
    expect(pickerInner(hexMap, 280, 240)).toEqual({ w: 278, h: 238 });
    expect(pickerRect(el, hexMap)).toEqual({ left: 101, top: 51, width: 278, height: 238 });
  });

  it('leave the box as it is on a square scan', () => {
    expect(pickerInner(null, 280, 240)).toEqual({ w: 280, h: 240 });
    expect(pickerRect(el, null)).toEqual({ left: 100, top: 50, width: 280, height: 240 });
  });
});

describe('pickAt', () => {
  it('names the point under every sampled pixel of the fixture, and none in a corner', () => {
    expect(CASE.pixels.some((p) => p.cell === null)).toBe(true);     // the fixture does hold corners
    for (const p of CASE.pixels) {
      const got = pickAt(hexMap, 3, 3, (p.j + 0.5) / 14, (p.i + 0.5) / 12);
      expect(got).toEqual(p.cell);
    }
  });

  it('answers null outside the picture on a hexagonal scan', () => {
    expect(pickAt(hexMap, 3, 3, -0.01, 0.5)).toBeNull();
    expect(pickAt(hexMap, 3, 3, 0.5, 1)).toBeNull();
  });

  it('is the old clamped arithmetic on a square scan', () => {
    expect(pickAt(null, 10, 12, 0.5, 0.5)).toEqual([5, 6]);
    expect(pickAt(null, 10, 12, 1, 1)).toEqual([9, 11]);
    expect(pickAt(null, 10, 12, -0.2, 3)).toEqual([9, 0]);
  });
});

describe('markerAt', () => {
  it('puts the marker on the exact centre of the point, by hand', () => {
    // point (1, 1) is at (3, 1.732) um; the raster is 14 x 12 pixels of 0.5 um from (-1, -1.155) um:
    // (3 + 1) / 7 across and (1.732 + 1.155) / 6 down
    const m = markerAt(hexMap, 3, 3, 4);
    expect(m.fx).toBeCloseTo(4 / 7, 12);
    expect(m.fy).toBeCloseTo((5 * Math.sqrt(3)) / 18, 12);
  });

  it('is the middle of the cell on a square scan', () => {
    expect(markerAt(null, 10, 12, 66)).toEqual({ fx: 6.5 / 12, fy: 5.5 / 10 });
  });
});

describe('rowColOf', () => {
  it('splits a flat index', () => {
    expect(rowColOf(12, 66)).toEqual({ row: 5, col: 6 });
    expect(rowColOf(0, 3)).toEqual({ row: 3, col: 0 });
  });
});

describe('shownPicture', () => {
  const held = { image: 'IMG', display: STAMP, hexMap };

  it('shows the picture that carries the raster of the map it was fetched for', () => {
    expect(shownPicture(hexMap, held)).toBe('IMG');
  });

  it('shows nothing before an answer, or an answer without an image', () => {
    expect(shownPicture(hexMap, null)).toBeNull();
    expect(shownPicture(hexMap, { ...held, image: null })).toBeNull();
    expect(shownPicture(null, null)).toBeNull();
  });

  it('withholds a hexagonal picture that carries no raster or another one', () => {
    expect(shownPicture(hexMap, { ...held, display: null })).toBeNull();
    expect(shownPicture(hexMap, { ...held, display: { ...STAMP, shape: [3, 3], factor: 1 } })).toBeNull();
    expect(shownPicture(hexMap, { ...held, display: { ...STAMP, x0: STAMP.x0 + 0.25 } })).toBeNull();   // same shape, other origin
  });

  it('withholds a picture fetched for another map, even one with the same raster', () => {
    // another measurement of the same size: a click on the old picture would name a point of the wrong one
    const other = makeHexMap(parseScanGeometry(CASE.geometry));
    expect(other).not.toBe(hexMap);
    expect(shownPicture(other, held)).toBeNull();
    expect(shownPicture(null, held)).toBeNull();                      // and a square scan shows no raster picture
  });

  it('is the picture itself on a square scan', () => {
    expect(shownPicture(null, { image: 'IMG', display: null, hexMap: null })).toBe('IMG');
  });
});

describe('pictureEntry, pictureAfterFailure and pictureProblem', () => {
  const heldHex = { image: 'IMG', display: STAMP, hexMap };
  const heldSquare = { image: 'IMG', display: null, hexMap: null };

  it('a hexagonal picker names a failure, a missing picture and a foreign raster', () => {
    expect(pictureProblem(hexMap, pictureAfterFailure(null, hexMap))).toBe('failed');
    expect(pictureProblem(hexMap, pictureEntry(null, hexMap, null, STAMP))).toBe('failed');
    expect(pictureProblem(hexMap, pictureEntry(null, hexMap, 'IMG', { ...STAMP, shape: [3, 3] }))).toBe('rasterMismatch');
    expect(pictureProblem(hexMap, pictureEntry(null, hexMap, 'IMG', STAMP))).toBeNull();
    expect(pictureProblem(hexMap, null)).toBeNull();
  });

  it('a square picker that held a hexagonal picture names a failure, and keeps naming it', () => {
    const first = pictureAfterFailure(heldHex, null);
    expect(pictureProblem(null, first)).toBe('failed');
    expect(pictureProblem(null, pictureAfterFailure(first, null))).toBe('failed');
    expect(pictureProblem(null, pictureEntry(heldHex, null, null, null))).toBe('failed');
  });

  it('a square picker that never held one stays silent, and a good square answer ends the failure', () => {
    expect(pictureAfterFailure(heldSquare, null)).toBe(heldSquare);
    expect(pictureAfterFailure(null, null)).toBeNull();
    expect(pictureProblem(null, pictureEntry(null, null, null, null))).toBeNull();
    const good = pictureEntry(pictureAfterFailure(heldHex, null), null, 'SQ', null);
    expect(pictureProblem(null, good)).toBeNull();
    expect(pictureProblem(null, pictureAfterFailure(good, null))).toBeNull();
  });

  it('is silent for an entry of another map', () => {
    expect(pictureProblem(null, heldHex)).toBeNull();
    expect(pictureProblem(hexMap, heldSquare)).toBeNull();
  });
});
