// The exported sheet must have room for everything the figure needs, even when
// the border was fitted before the body grew.
//
// The export dialog re-fits its border only when `fitMarginsKey` changes — the
// SET of scale bodies, not their geometry — so that dragging one does not
// resize the picture under the user's hand. The cost was silent: a colour key
// made taller after that point stuck out of the canvas and was cut off by it.
// Measured on a real export (2026-09-03): a three-phase IPF key came out as its
// MIDDLE key alone, sliced at the top and bottom edges, in a file with zero
// vertical border.
import { describe, it, expect } from 'vitest';
import { widerMargins, NO_MARGINS } from './imageExport';
import { scaleMargins } from '../PhaseMap/annotations/exportAnnotEdits';

describe('widerMargins', () => {
  it('takes the larger of each side', () => {
    expect(widerMargins(
      { top: 0.1, right: 0.4, bottom: 0, left: 0.2 },
      { top: 0.3, right: 0.1, bottom: 0.25, left: 0 },
    )).toEqual({ top: 0.3, right: 0.4, bottom: 0.25, left: 0.2 });
  });

  it('treats a missing or partial set as zero rather than throwing', () => {
    expect(widerMargins(null, null)).toEqual(NO_MARGINS);
    expect(widerMargins({ top: 0.2 }, undefined)).toEqual(
      { top: 0.2, right: 0, bottom: 0, left: 0 },
    );
    expect(widerMargins({ top: Number.NaN }, { top: 0.1 })).toEqual(
      { top: 0.1, right: 0, bottom: 0, left: 0 },
    );
  });

  it('never shrinks a border the user widened by hand', () => {
    const byHand = { top: 0.4, right: 0.4, bottom: 0.4, left: 0.4 };
    expect(widerMargins(byHand, { top: 0, right: 0.1, bottom: 0, left: 0 })).toEqual(byHand);
  });

  it('gives a tall colour key its vertical room (the shipped defect)', () => {
    // A key body reaching well above and below the map, as one does on a very
    // flat scan (the reported figure was 136 x 39).
    const tallKey = [{ type: 'colorkey', x: 1.04, y: -0.9, w: 0.4, h: 3.0 }];
    const needed = scaleMargins(tallKey);
    expect(needed.top).toBeGreaterThan(0);
    expect(needed.bottom).toBeGreaterThan(0);

    // Before: the fitted border was whatever it had been (here: none vertically)
    // and the key was cut. After: the file gets the room.
    const fittedEarlier = { top: 0, right: 0.46, bottom: 0, left: 0 };
    const used = widerMargins(fittedEarlier, needed);
    expect(used.top).toBeGreaterThan(0);
    expect(used.bottom).toBeGreaterThan(0);
    expect(used.right).toBeGreaterThanOrEqual(0.46);
  });
});
