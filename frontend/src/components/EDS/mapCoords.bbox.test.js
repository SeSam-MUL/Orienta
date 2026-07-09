import { describe, it, expect } from 'vitest';
import {
  pointerToRowCol,
  rowColToContainerPx,
  displayContentRect,
  bboxContentRect,
} from './mapCoords';

// Build a fake DOMRect (only the fields the mappers read).
const mkRect = (width, height, left = 0, top = 0) => ({ left, top, width, height });
// Build a fake pointer event at container-relative (x, y).
const mkEv = (rect, x, y) => ({ clientX: rect.left + x, clientY: rect.top + y });

describe('mapCoords bbox-aware mapping', () => {
  describe('null bbox — byte-for-byte identity with legacy behaviour', () => {
    // shape [natH, natW]
    const shape = [100, 120];
    const rect = mkRect(640, 480);
    // Sample a grid of pointers spanning bars + content.
    const pts = [];
    for (let x = -20; x <= 660; x += 37) {
      for (let y = -20; y <= 500; y += 41) pts.push([x, y]);
    }

    it('pointerToRowCol(...,null) === pointerToRowCol(...) 3-arg', () => {
      for (const [x, y] of pts) {
        const ev = mkEv(rect, x, y);
        expect(pointerToRowCol(ev, rect, shape, null))
          .toEqual(pointerToRowCol(ev, rect, shape));
      }
    });

    it('rowColToContainerPx(...,null) === rowColToContainerPx(...) 4-arg', () => {
      for (let row = 0; row < shape[0]; row += 13) {
        for (let col = 0; col < shape[1]; col += 11) {
          expect(rowColToContainerPx(row, col, rect, shape, null))
            .toEqual(rowColToContainerPx(row, col, rect, shape));
        }
      }
    });

    it('displayContentRect with null bbox fills the letterboxed canvas', () => {
      const c = displayContentRect(rect, shape, null);
      // Source rect always equals the full native grid when there is no bbox.
      expect(c.srcX).toBe(0);
      expect(c.srcY).toBe(0);
      expect(c.srcW).toBe(120);
      expect(c.srcH).toBe(100);
    });
  });

  describe('round-trip pointer -> rowCol -> containerPx with a bbox', () => {
    const shape = [100, 120];
    const bbox = { x: 20, y: 10, w: 40, h: 30 };
    const rect = mkRect(600, 500);

    it('recovers the same integer pixel for every pixel centre in the bbox', () => {
      for (let row = bbox.y; row < bbox.y + bbox.h; row += 3) {
        for (let col = bbox.x; col < bbox.x + bbox.w; col += 3) {
          const px = rowColToContainerPx(row, col, rect, shape, bbox);
          expect(px).not.toBeNull();
          const back = pointerToRowCol(mkEv(rect, px.x, px.y), rect, shape, bbox);
          expect(back).toEqual({ row, col });
        }
      }
    });
  });

  describe('pointer in the letterbox bar returns null under bbox zoom', () => {
    const shape = [100, 120];
    const bbox = { x: 20, y: 10, w: 40, h: 30 };
    const rect = mkRect(600, 500);

    it('a pointer above the content top edge is null', () => {
      // Content for this case: y spans 25..475 (see hand computation), so y=5
      // sits in the top letterbox bar.
      const c = displayContentRect(rect, shape, bbox);
      expect(c.y).toBeGreaterThan(6); // sanity: there IS a top bar
      const ev = mkEv(rect, 300, 5);
      expect(pointerToRowCol(ev, rect, shape, bbox)).toBeNull();
    });

    it('a pointer left of the content is null when the bbox leaves a side bar', () => {
      // Use a portrait-ish bbox against a wide container to force a side bar.
      const sh = [120, 100];
      const bb = { x: 40, y: 10, w: 20, h: 90 };
      const rc = mkRect(800, 400);
      const c = displayContentRect(rc, sh, bb);
      expect(c.x).toBeGreaterThan(1); // there IS a left bar
      const ev = mkEv(rc, Math.max(0, c.x - 5), 200);
      expect(pointerToRowCol(ev, rc, sh, bb)).toBeNull();
    });
  });

  describe('realistic ROI case', () => {
    // Full map 174 rows x 145 cols, ROI bbox, 600x500 viewport.
    const shape = [174, 145];
    const bbox = { x: 48, y: 87, w: 25, h: 22 };
    const rect = mkRect(600, 500);

    it('clicking the visual centre returns a (row,col) INSIDE the bbox', () => {
      const ev = mkEv(rect, 300, 250);
      const out = pointerToRowCol(ev, rect, shape, bbox);
      expect(out).not.toBeNull();
      expect(out.row).toBeGreaterThanOrEqual(bbox.y);
      expect(out.row).toBeLessThan(bbox.y + bbox.h);
      expect(out.col).toBeGreaterThanOrEqual(bbox.x);
      expect(out.col).toBeLessThan(bbox.x + bbox.w);
      // And it should map to (near) the bbox centre.
      expect(out.row).toBe(98);
      expect(out.col).toBe(60);
    });

    it('the bbox mapping DIFFERS from the legacy full-grid mapping (the fix)', () => {
      const ev = mkEv(rect, 300, 250);
      const withBbox = pointerToRowCol(ev, rect, shape, bbox);
      const legacy = pointerToRowCol(ev, rect, shape); // full-grid mapping
      expect(withBbox).not.toBeNull();
      expect(legacy).not.toBeNull();
      // Same physical click resolves to a different pixel once the auto-zoom
      // bbox is accounted for — the legacy result is the wrong pixel.
      expect(withBbox).not.toEqual(legacy);
    });

    it('a click off the bbox centre that is OUTSIDE under legacy mapping', () => {
      // Upper-left of the visual content: legacy maps toward the grid origin
      // (outside this bottom-ish ROI); bbox maps inside the ROI.
      const ev = mkEv(rect, 120, 80);
      const withBbox = pointerToRowCol(ev, rect, shape, bbox);
      const legacy = pointerToRowCol(ev, rect, shape);
      expect(withBbox).not.toBeNull();
      const inBbox = (o) =>
        o && o.row >= bbox.y && o.row < bbox.y + bbox.h &&
        o.col >= bbox.x && o.col < bbox.x + bbox.w;
      expect(inBbox(withBbox)).toBe(true);
      expect(inBbox(legacy)).toBe(false);
    });
  });

  describe('bboxContentRect mirrors the draw math', () => {
    it('landscape bbox is width-constrained inside the buffer', () => {
      // bboxAspect 40/30=1.333 > bufAspect 120/100=1.2 -> dstW=natW, letterbox top/bottom
      const dst = bboxContentRect(120, 100, { x: 20, y: 10, w: 40, h: 30 });
      expect(dst.x).toBe(0);
      expect(dst.w).toBe(120);
      expect(dst.h).toBe(90); // 120/1.3333 = 90
      expect(dst.y).toBe(5);  // (100-90)/2
    });
  });
});
