// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup } from '@testing-library/react';
import { useRef } from 'react';
import { useHeatmapPick } from './useHeatmapPick';

// A 200x100 image on screen showing a 20x10 grid: 10 CSS px per map pixel.
const RECT = { left: 100, top: 50, width: 200, height: 100 };
const DIMS = { rows: 10, cols: 20 };

function Harness({ onPick, dims = DIMS, offset = { row: 0, col: 0 }, onPan = null }) {
  const imgRef = useRef(null);
  const pick = useHeatmapPick({ imgRef, dims, offset, onPick, onPan });
  return (
    <div>
      <img
        ref={imgRef}
        alt="heatmap"
        data-testid="img"
        onMouseDown={pick.onMouseDown}
      />
      <span data-testid="preview">
        {pick.preview ? `${pick.preview.localCol},${pick.preview.localRow}` : 'none'}
      </span>
      <span data-testid="dragging">{String(pick.dragging)}</span>
    </div>
  );
}

const at = (x, y, shiftKey = false) => ({ clientX: x, clientY: y, button: 0, shiftKey });

function down(el, x, y, shiftKey = false) {
  act(() => {
    el.dispatchEvent(new MouseEvent('mousedown', { ...at(x, y, shiftKey), bubbles: true, cancelable: true }));
  });
}
function move(x, y) {
  act(() => {
    window.dispatchEvent(new MouseEvent('mousemove', { ...at(x, y), bubbles: true }));
  });
}
function up(x, y) {
  act(() => {
    window.dispatchEvent(new MouseEvent('mouseup', { ...at(x, y), bubbles: true }));
  });
}

beforeEach(() => {
  // jsdom has no layout; give the image the rect the maths needs.
  Element.prototype.getBoundingClientRect = function rect() {
    return { ...RECT, right: RECT.left + RECT.width, bottom: RECT.top + RECT.height, x: RECT.left, y: RECT.top };
  };
});

afterEach(cleanup);

describe('useHeatmapPick', () => {
  it('commits a plain click, exactly once', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    const img = screen.getByTestId('img');
    // 10 px per map pixel: (145, 75) -> col 4, row 2
    down(img, 145, 75);
    up(145, 75);
    expect(onPick).toHaveBeenCalledTimes(1);
    expect(onPick).toHaveBeenCalledWith({ row: 2, col: 4, localRow: 2, localCol: 4 });
  });

  it('does not commit while the pointer is still down', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    down(screen.getByTestId('img'), 105, 55);
    move(155, 85);
    move(205, 95);
    // This is the whole point: each move would otherwise re-fetch a pattern.
    expect(onPick).not.toHaveBeenCalled();
    expect(screen.getByTestId('preview').textContent).toBe('10,4');
  });

  it('commits once on release, at the release position', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    down(screen.getByTestId('img'), 105, 55);
    move(155, 85);
    up(255, 95);
    expect(onPick).toHaveBeenCalledTimes(1);
    expect(onPick).toHaveBeenCalledWith({ row: 4, col: 15, localRow: 4, localCol: 15 });
    expect(screen.getByTestId('preview').textContent).toBe('none');
    expect(screen.getByTestId('dragging').textContent).toBe('false');
  });

  it('clamps to the grid when the drag leaves the image', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    down(screen.getByTestId('img'), 150, 80);
    move(-500, -500);
    expect(screen.getByTestId('preview').textContent).toBe('0,0');
    move(5000, 5000);
    expect(screen.getByTestId('preview').textContent).toBe('19,9');
    up(5000, 5000);
    expect(onPick).toHaveBeenCalledWith({ row: 9, col: 19, localRow: 9, localCol: 19 });
  });

  it('adds the crop offset to the committed pixel but not to the local one', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} offset={{ row: 45, col: 2 }} />);
    down(screen.getByTestId('img'), 145, 75);
    up(145, 75);
    expect(onPick).toHaveBeenCalledWith({ row: 47, col: 6, localRow: 2, localCol: 4 });
  });

  describe('once zoomed in (onPan given)', () => {
    it('drags the picture instead of the point', () => {
      const onPick = vi.fn();
      const onPan = vi.fn();
      render(<Harness onPick={onPick} onPan={onPan} />);
      down(screen.getByTestId('img'), 150, 80);
      move(170, 95);
      move(190, 110);
      // No preview point is armed, and nothing is committed mid-drag.
      expect(screen.getByTestId('preview').textContent).toBe('none');
      expect(onPick).not.toHaveBeenCalled();
      expect(onPan).toHaveBeenCalledTimes(2);
      // Deltas as fractions of the 200x100 box.
      expect(onPan).toHaveBeenNthCalledWith(1, 20 / 200, 15 / 100);
      up(190, 110);
      // A drag that travelled must NOT also move the point.
      expect(onPick).not.toHaveBeenCalled();
    });

    it('still picks on a click that did not travel', () => {
      const onPick = vi.fn();
      const onPan = vi.fn();
      render(<Harness onPick={onPick} onPan={onPan} />);
      down(screen.getByTestId('img'), 145, 75);
      up(145, 75);
      expect(onPan).not.toHaveBeenCalled();
      expect(onPick).toHaveBeenCalledWith({ row: 2, col: 4, localRow: 2, localCol: 4 });
    });

    it('moves the point again while Shift is held', () => {
      const onPick = vi.fn();
      const onPan = vi.fn();
      render(<Harness onPick={onPick} onPan={onPan} />);
      down(screen.getByTestId('img'), 105, 55, true);   // shift
      move(155, 85);
      // The point follows and the picture stays put.
      expect(screen.getByTestId('preview').textContent).toBe('5,3');
      expect(onPan).not.toHaveBeenCalled();
      up(155, 85);
      expect(onPick).toHaveBeenCalledWith({ row: 3, col: 5, localRow: 3, localCol: 5 });
    });

    it('treats a tiny wobble as a click, not a pan', () => {
      const onPick = vi.fn();
      const onPan = vi.fn();
      render(<Harness onPick={onPick} onPan={onPan} />);
      down(screen.getByTestId('img'), 145, 75);
      move(146, 75);
      up(146, 75);
      expect(onPick).toHaveBeenCalledTimes(1);
    });
  });

  it('ignores the right button so the context menu still works', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    act(() => {
      screen.getByTestId('img').dispatchEvent(new MouseEvent('mousedown', {
        clientX: 145, clientY: 75, button: 2, bubbles: true, cancelable: true,
      }));
    });
    expect(screen.getByTestId('dragging').textContent).toBe('false');
    up(145, 75);
    expect(onPick).not.toHaveBeenCalled();
  });

  it('stays inert without grid dimensions', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} dims={{ rows: 0, cols: 0 }} />);
    down(screen.getByTestId('img'), 145, 75);
    up(145, 75);
    expect(onPick).not.toHaveBeenCalled();
  });

  it('a move without a preceding press does nothing', () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    move(145, 75);
    up(145, 75);
    expect(onPick).not.toHaveBeenCalled();
    expect(screen.getByTestId('preview').textContent).toBe('none');
  });
});
