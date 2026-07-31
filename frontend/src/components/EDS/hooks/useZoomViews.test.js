// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useZoomViews, SYNC_ALL, SYNC_SINGLE } from './useZoomViews';
import { IDENTITY_VIEW } from '../zoomView';

describe('useZoomViews', () => {
  it('starts unzoomed for every id and disables reset', () => {
    const { result } = renderHook(() => useZoomViews());
    expect(result.current.mode).toBe(SYNC_ALL);
    expect(result.current.viewFor('bc')).toEqual(IDENTITY_VIEW);
    expect(result.current.viewFor('eds-Al')).toEqual(IDENTITY_VIEW);
    expect(result.current.anyZoomed).toBe(false);
  });

  it('sync mode: zooming one map moves every map', () => {
    const { result } = renderHook(() => useZoomViews(SYNC_ALL));
    act(() => result.current.zoomAtPointer('bc', 2, 0.5, 0.5));
    expect(result.current.viewFor('bc').scale).toBe(2);
    expect(result.current.viewFor('eds-Al').scale).toBe(2);
    expect(result.current.anyZoomed).toBe(true);
  });

  it('single mode: zooming one map leaves the others alone', () => {
    const { result } = renderHook(() => useZoomViews(SYNC_SINGLE));
    act(() => result.current.zoomAtPointer('bc', 2, 0.5, 0.5));
    expect(result.current.viewFor('bc').scale).toBe(2);
    expect(result.current.viewFor('eds-Al')).toEqual(IDENTITY_VIEW);
  });

  it('keeps per-map views across a trip through sync mode', () => {
    const { result } = renderHook(() => useZoomViews(SYNC_SINGLE));
    act(() => result.current.zoomAtPointer('bc', 4, 0.5, 0.5));
    act(() => result.current.setMode(SYNC_ALL));
    // Shared view is seeded from the last map touched, so nothing jumps.
    expect(result.current.viewFor('eds-Al').scale).toBe(4);
    act(() => result.current.zoomAtPointer('eds-Al', 2, 0.5, 0.5));
    act(() => result.current.setMode(SYNC_SINGLE));
    expect(result.current.viewFor('bc').scale).toBe(4);       // untouched
    expect(result.current.viewFor('eds-Al')).toEqual(IDENTITY_VIEW);
  });

  it('panning shifts the view centre', () => {
    const { result } = renderHook(() => useZoomViews(SYNC_ALL));
    act(() => result.current.zoomAtPointer('bc', 4, 0.5, 0.5));
    act(() => result.current.pan('bc', 0.2, 0));
    expect(result.current.viewFor('bc').cx).toBeCloseTo(0.45, 6);
  });

  it('resetAll clears both the shared and every per-map view', () => {
    const { result } = renderHook(() => useZoomViews(SYNC_SINGLE));
    act(() => result.current.zoomAtPointer('bc', 3, 0.5, 0.5));
    act(() => result.current.zoomAtPointer('eds-Al', 2, 0.5, 0.5));
    expect(result.current.anyZoomed).toBe(true);
    act(() => result.current.resetAll());
    expect(result.current.viewFor('bc')).toEqual(IDENTITY_VIEW);
    expect(result.current.viewFor('eds-Al')).toEqual(IDENTITY_VIEW);
    expect(result.current.anyZoomed).toBe(false);
  });

  it('resetOne only clears that map in single mode', () => {
    const { result } = renderHook(() => useZoomViews(SYNC_SINGLE));
    act(() => result.current.zoomAtPointer('bc', 3, 0.5, 0.5));
    act(() => result.current.zoomAtPointer('eds-Al', 2, 0.5, 0.5));
    act(() => result.current.resetOne('bc'));
    expect(result.current.viewFor('bc')).toEqual(IDENTITY_VIEW);
    expect(result.current.viewFor('eds-Al').scale).toBe(2);
  });
});
