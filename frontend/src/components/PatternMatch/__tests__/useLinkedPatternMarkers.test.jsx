// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useLinkedPatternMarkers } from '../useLinkedPatternMarkers';

describe('useLinkedPatternMarkers', () => {
  it('adds numbered markers on click', () => {
    const { result } = renderHook(() => useLinkedPatternMarkers());
    act(() => result.current.handlePanelClick({ x: 0.2, y: 0.3 }));
    act(() => result.current.handlePanelClick({ x: 0.8, y: 0.7 }));
    expect(result.current.markers.map(m => m.n)).toEqual([1, 2]);
    expect(result.current.markers[0]).toMatchObject({ n: 1, x: 0.2, y: 0.3 });
  });
  it('clicking an existing marker removes it and renumbers', () => {
    const { result } = renderHook(() => useLinkedPatternMarkers());
    act(() => result.current.handlePanelClick({ x: 0.2, y: 0.2 })); // m1
    act(() => result.current.handlePanelClick({ x: 0.6, y: 0.6 })); // m2
    act(() => result.current.handlePanelClick({ x: 0.205, y: 0.2 })); // hits m1 -> remove
    expect(result.current.markers).toHaveLength(1);
    expect(result.current.markers[0].n).toBe(1); // renumbered
    expect(result.current.markers[0].x).toBeCloseTo(0.6);
  });
  it('clearMarkers empties the list', () => {
    const { result } = renderHook(() => useLinkedPatternMarkers());
    act(() => result.current.handlePanelClick({ x: 0.5, y: 0.5 }));
    act(() => result.current.clearMarkers());
    expect(result.current.markers).toHaveLength(0);
  });
});
