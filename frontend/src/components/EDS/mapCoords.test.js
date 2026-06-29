import { describe, it, expect } from 'vitest';
import { pointerToRowCol, rowColToContainerPx } from './mapCoords';

describe('pointerToRowCol', () => {
  it('returns center pixel for click at container center when aspect matches', () => {
    const rect = { width: 200, height: 200, left: 0, top: 0 };
    const shape = [100, 100]; // rows, cols
    const out = pointerToRowCol({ clientX: 100, clientY: 100 }, rect, shape);
    expect(out).toEqual({ row: 50, col: 50 });
  });
  it('letterboxes horizontally when container is wider than image aspect', () => {
    const rect = { width: 300, height: 100, left: 0, top: 0 };
    const shape = [100, 100]; // square image -> 100x100 letterboxed inside 300x100
    // canvas drawn at x in [100..200], width 100
    const out = pointerToRowCol({ clientX: 150, clientY: 50 }, rect, shape);
    expect(out).toEqual({ row: 50, col: 50 });
  });
  it('returns null when pointer is outside the image area', () => {
    const rect = { width: 300, height: 100, left: 0, top: 0 };
    const shape = [100, 100];
    expect(pointerToRowCol({ clientX: 10, clientY: 50 }, rect, shape)).toBeNull();
  });
});

describe('rowColToContainerPx', () => {
  it('returns pixel-center offset when aspect matches', () => {
    const rect = { width: 200, height: 200, left: 0, top: 0 };
    const shape = [100, 100];
    // (50 + 0.5) / 100 * 200 = 101
    const out = rowColToContainerPx(50, 50, rect, shape);
    expect(out).toEqual({ x: 101, y: 101 });
  });
  it('applies horizontal letterbox offset', () => {
    const rect = { width: 300, height: 100, left: 0, top: 0 };
    const shape = [100, 100];
    // image drawn as 100x100 centered horizontally -> offX = 100
    // pixel-center x = 100 + (50 + 0.5) / 100 * 100 = 150.5
    // pixel-center y =   0 + (50 + 0.5) / 100 * 100 =  50.5
    const out = rowColToContainerPx(50, 50, rect, shape);
    expect(out).toEqual({ x: 150.5, y: 50.5 });
  });
  it('returns null when shape has a zero dimension', () => {
    const rect = { width: 200, height: 200, left: 0, top: 0 };
    expect(rowColToContainerPx(0, 0, rect, [0, 0])).toBeNull();
  });
});

describe('pointerToRowCol <-> rowColToContainerPx roundtrip', () => {
  it('recovers (row, col) from the pixel-center of each cell', () => {
    const rect = { width: 400, height: 200, left: 0, top: 0 };
    const shape = [50, 100]; // rows, cols
    const cases = [
      { row: 0, col: 0 },
      { row: 25, col: 50 },
      { row: 49, col: 99 },
    ];
    for (const { row, col } of cases) {
      const { x, y } = rowColToContainerPx(row, col, rect, shape);
      const back = pointerToRowCol(
        { clientX: x + rect.left, clientY: y + rect.top },
        rect,
        shape,
      );
      expect(back).toEqual({ row, col });
    }
  });
});
