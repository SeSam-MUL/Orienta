// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useRectangleDrag } from './useRectangleDrag';

function mockEvent(over, props) {
  // pointer event with shift + clientX/Y; currentTarget has a rect
  const target = {
    getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 }),
  };
  return { shiftKey: !!over, clientX: 0, clientY: 0, currentTarget: target, ...props };
}

describe('useRectangleDrag', () => {
  it('emits a region rect on shift-drag pointer-up', () => {
    const onRegion = vi.fn();
    const { result } = renderHook(() => useRectangleDrag({ shape: [100, 100], onRegion }));
    act(() => {
      result.current.onPointerDown(mockEvent(true, { clientX: 10, clientY: 10 }));
      result.current.onPointerMove({ clientX: 40, clientY: 50 });
      result.current.onPointerUp({ clientX: 40, clientY: 50 });
    });
    expect(onRegion).toHaveBeenCalledWith({ rowStart: 10, colStart: 10, rowEnd: 50, colEnd: 40 });
  });

  it('ignores non-shift drag (plain click path stays untouched)', () => {
    const onRegion = vi.fn();
    const { result } = renderHook(() => useRectangleDrag({ shape: [100, 100], onRegion }));
    act(() => {
      result.current.onPointerDown(mockEvent(false, { clientX: 10, clientY: 10 }));
      result.current.onPointerMove({ clientX: 40, clientY: 50 });
      result.current.onPointerUp({ clientX: 40, clientY: 50 });
    });
    expect(onRegion).not.toHaveBeenCalled();
  });

  it('normalises reversed drag (right-to-left or bottom-to-top)', () => {
    const onRegion = vi.fn();
    const { result } = renderHook(() => useRectangleDrag({ shape: [100, 100], onRegion }));
    act(() => {
      result.current.onPointerDown(mockEvent(true, { clientX: 80, clientY: 70 }));
      result.current.onPointerMove({ clientX: 20, clientY: 30 });
      result.current.onPointerUp({ clientX: 20, clientY: 30 });
    });
    expect(onRegion).toHaveBeenCalledWith({ rowStart: 30, colStart: 20, rowEnd: 70, colEnd: 80 });
  });

  it('exposes an overlay rect during the drag', () => {
    const { result } = renderHook(() => useRectangleDrag({ shape: [100, 100], onRegion: () => {} }));
    expect(result.current.overlay).toBeNull();
    act(() => { result.current.onPointerDown(mockEvent(true, { clientX: 10, clientY: 10 })); });
    expect(result.current.overlay).toEqual({ x0: 10, y0: 10, x1: 10, y1: 10 });
    act(() => { result.current.onPointerMove({ clientX: 40, clientY: 50 }); });
    expect(result.current.overlay).toEqual({ x0: 10, y0: 10, x1: 40, y1: 50 });
    act(() => { result.current.onPointerUp({ clientX: 40, clientY: 50 }); });
    expect(result.current.overlay).toBeNull();
  });

  it('no-op when shape is null', () => {
    const onRegion = vi.fn();
    const { result } = renderHook(() => useRectangleDrag({ shape: null, onRegion }));
    act(() => {
      result.current.onPointerDown(mockEvent(true, { clientX: 10, clientY: 10 }));
      result.current.onPointerUp({ clientX: 40, clientY: 50 });
    });
    expect(onRegion).not.toHaveBeenCalled();
  });
});
