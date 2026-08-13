import { describe, it, expect } from 'vitest';
import { scalebarBoxSize } from './AnnotationLayer';

// The map this is measured against: 145 columns of 0.2 µm, drawn 580 px wide,
// unzoomed. One micrometre is then 4 px on screen.
const CTX = {
  mapNativeSize: { w: 145, h: 174 },
  mapContentBbox: null,
  stepX: 0.2,
  mapBoxWidthPx: 580,
  mapZoomScale: 1,
};

const bar = (props = {}) => ({
  id: 's', type: 'scalebar', x: 0.6, y: 0.9, w: 0.3, h: 0.05,
  props: { lengthUm: 5, fontSize: 12, ...props },
});

describe('scalebarBoxSize — the frame follows the bar', () => {
  it('is as wide as the bar plus a little padding', () => {
    // 5 µm / 0.2 µm per column = 25 columns = 100 px, + 2×4 px padding.
    const box = scalebarBoxSize(bar(), CTX);
    expect(box.w).toBe(108);
    expect(box.h).toBeCloseTo(30.4, 6);
  });

  it('grows with the stated length, not with the stored box', () => {
    const five = scalebarBoxSize(bar({ lengthUm: 5 }), CTX);
    const twenty = scalebarBoxSize(bar({ lengthUm: 20 }), CTX);
    // The old frame was 0.3 of the map (174 px) no matter what the bar said.
    expect(twenty.w - 8).toBeCloseTo((five.w - 8) * 4, 6);
  });

  it('grows with zoom, because the bar does', () => {
    const at1 = scalebarBoxSize(bar(), CTX);
    const at3 = scalebarBoxSize(bar(), { ...CTX, mapZoomScale: 3 });
    expect(at3.w - 8).toBeCloseTo((at1.w - 8) * 3, 6);
  });

  it('leaves room for the label under the bar', () => {
    const small = scalebarBoxSize(bar({ fontSize: 10 }), CTX);
    const large = scalebarBoxSize(bar({ fontSize: 24 }), CTX);
    expect(large.h).toBeGreaterThan(small.h);
    // bar height + gap + one text line + padding, nothing more
    expect(small.h).toBeCloseTo(Math.max(3, 10 * 0.4) + 2 + 10 * 1.3 + 8, 6);
  });

  it('says "no idea" rather than guessing when the geometry is missing', () => {
    // Without a step size there is no µm→px factor; the caller then keeps the
    // stored box instead of collapsing the widget to nothing.
    expect(scalebarBoxSize(bar(), { ...CTX, stepX: null })).toBeNull();
    expect(scalebarBoxSize(bar(), { ...CTX, mapBoxWidthPx: 0 })).toBeNull();
    expect(scalebarBoxSize(bar(), {})).toBeNull();
  });
});
