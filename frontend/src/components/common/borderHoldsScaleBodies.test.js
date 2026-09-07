/**
 * The border has to hold the scale bodies that were placed in it.
 *
 * Reported 2026-09-07 with the file attached: a three-phase IPF colour key
 * beside a wide, short phase map, sliced across the top and the bottom — the
 * caption "Al (m-3m)" cut off above, the third key cut off below.
 *
 * Nothing draws it wrongly. `scaleMargins` works out the border the figure
 * needs and `widerMargins` re-asserts it at save time (added 2026-09-03 for
 * this very symptom) — and then an editorial ceiling of half the map per side
 * clamped it away again. The real ceilings are the browser's encode limits,
 * and on this figure they had eight times the headroom needed.
 *
 * The numbers below are measured off the PNG the reporter sent (6354 x 2496,
 * map region 4352 x 1248) and confirmed against the result that produced it:
 * `original_shape [39, 136]` — 3.487 : 1, so the map region is that grid at
 * exactly 32x — and three phases, Al / Al7FeCu2 / MgCuAl2. The key's own shape
 * comes from the backend that draws it (`get_ipf_key`: 2.4 in wide, 2.2 in per
 * phase, `orientation="vertical"`).
 */
import { describe, it, expect } from 'vitest';
import { canvasSizeWithMargins } from './imageExport';
import { scaleMargins } from '../PhaseMap/annotations/exportAnnotEdits';

/** The map region of the reported figure: the 39 x 136 scan at 32x. */
const MAP = { width: 136 * 32, height: 39 * 32 };

/** How many times taller than wide the vertical key is, for n phases. */
const keyAspect = (n) => Math.max(2.4, 2.2 * n) / 2.4;

/**
 * The body you get by sizing the key to a readable width beside the map.
 * `w` is a fraction of the map width — 0.4 is the default the checkbox places.
 */
function keyBody(nPhases, w = 0.4) {
  const h = (w * MAP.width * keyAspect(nPhases)) / MAP.height;
  return { type: 'colorkey', x: 1.04, y: (1 - h) / 2, w, h, props: {} };
}

describe('the border holds the scale bodies placed in it', () => {
  it('reserves the whole height a three-phase IPF key needs', () => {
    const body = keyBody(3);
    // Sanity on the premise: this key really is taller than three maps.
    expect(body.h).toBeGreaterThan(3.8);

    const m = scaleMargins([body]);
    expect(m.top).toBeCloseTo(-body.y + 0.02, 6);
    expect(m.bottom).toBeCloseTo(body.y + body.h - 1 + 0.02, 6);
  });

  it('writes a sheet the key actually fits on', () => {
    const body = keyBody(3);
    const total = canvasSizeWithMargins(MAP, scaleMargins([body]));

    const top = total.origin.y + body.y * MAP.height;
    const bottom = top + body.h * MAP.height;
    expect(top).toBeGreaterThanOrEqual(0);
    expect(bottom).toBeLessThanOrEqual(total.height);
    // Well inside the browser's encode limits — the old ceiling was editorial,
    // not a technical one.
    expect(total.limited).toBe(false);
  });

  it('no longer collapses every wide border onto the same sheet', () => {
    // The reported file was 6354 x 2496 = exactly twice the map height, which
    // is what ANY border of half the map or more used to produce.
    const wide = canvasSizeWithMargins(MAP, { top: 5, bottom: 5, left: 0, right: 0.46 });
    expect(wide.height).toBeGreaterThan(2496);
  });

  it('still asks for nothing when no scale body is placed', () => {
    expect(scaleMargins([])).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
    // A body sitting ON the map needs no border either.
    const onMap = [{ type: 'colorkey', x: 0.5, y: 0.1, w: 0.3, h: 0.3, props: {} }];
    expect(scaleMargins(onMap)).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
  });
});
