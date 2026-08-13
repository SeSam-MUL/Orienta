import { describe, it, expect } from 'vitest';
import { zoomRectPct } from './zoomOverlay';
import { IDENTITY_VIEW, clampView, zoomedRect, zoomAt } from '../EDS/zoomView';

// A letterboxed overview: 156x128 content in a wider box, so there are bars
// left and right (object-fit: contain). Percentages of the untransformed box.
const LETTERBOX = { left: 12.5, top: 0, width: 75, height: 100 };

describe('zoomRectPct', () => {
  it('is a no-op at the identity view', () => {
    expect(zoomRectPct(LETTERBOX, IDENTITY_VIEW)).toEqual(LETTERBOX);
  });

  it('scales about the box centre when the view is centred', () => {
    const r = zoomRectPct(LETTERBOX, { scale: 2, cx: 0.5, cy: 0.5 });
    // Width doubles; the rect stays centred because the letterbox is centred.
    expect(r.width).toBeCloseTo(150, 10);
    expect(r.left).toBeCloseTo(-25, 10);
    expect(r.height).toBeCloseTo(200, 10);
    expect(r.top).toBeCloseTo(-50, 10);
  });

  it('moves the rect opposite to the pan when the centre shifts', () => {
    // Looking further right (cx > 0.5) must push content left.
    const centred = zoomRectPct(LETTERBOX, { scale: 4, cx: 0.5, cy: 0.5 });
    const right = zoomRectPct(LETTERBOX, { scale: 4, cx: 0.6, cy: 0.5 });
    expect(right.left).toBeLessThan(centred.left);
    expect(right.width).toBeCloseTo(centred.width, 10);
  });

  it('clamps the view like the rest of the zoom math', () => {
    // scale 0.25 is below MIN_SCALE, so it must behave exactly like scale 1.
    expect(zoomRectPct(LETTERBOX, { scale: 0.25, cx: 0.5, cy: 0.5 }))
      .toEqual(zoomRectPct(LETTERBOX, IDENTITY_VIEW));
  });

  // The load-bearing property: the overlay is positioned in percent of the
  // untransformed box while the image is moved by a CSS transform. Both must
  // land on the same screen pixels, or the ROI box and crosshair drift off the
  // features they mark. Cross-check against zoomedRect, which is the same math
  // the (already tested) EDS pointer mapping uses.
  it('agrees with zoomedRect on screen coordinates', () => {
    const box = { left: 40, top: 90, width: 800, height: 600 };
    const views = [
      IDENTITY_VIEW,
      { scale: 2, cx: 0.5, cy: 0.5 },
      { scale: 3.7, cx: 0.3, cy: 0.8 },
      zoomAt(IDENTITY_VIEW, 4, 0.2, 0.15),
      clampView({ scale: 16, cx: 0.05, cy: 0.95 }),
    ];

    for (const view of views) {
      const z = zoomedRect(box, view);
      // Where the letterboxed sub-rect ends up inside the zoomed content.
      const expected = {
        left: z.left + (LETTERBOX.left / 100) * z.width,
        top: z.top + (LETTERBOX.top / 100) * z.height,
        width: (LETTERBOX.width / 100) * z.width,
        height: (LETTERBOX.height / 100) * z.height,
      };

      // Where the percent overlay ends up.
      const p = zoomRectPct(LETTERBOX, view);
      const actual = {
        left: box.left + (p.left / 100) * box.width,
        top: box.top + (p.top / 100) * box.height,
        width: (p.width / 100) * box.width,
        height: (p.height / 100) * box.height,
      };

      expect(actual.left).toBeCloseTo(expected.left, 8);
      expect(actual.top).toBeCloseTo(expected.top, 8);
      expect(actual.width).toBeCloseTo(expected.width, 8);
      expect(actual.height).toBeCloseTo(expected.height, 8);
    }
  });
});
