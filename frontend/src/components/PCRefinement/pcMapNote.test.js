import { describe, it, expect } from 'vitest';
import { pcMapInfoOf, pcMapActionOf, pcMapNote } from './pcMapNote';

describe('pcMapInfoOf', () => {
  it('is null for a detector with one PC', () => {
    expect(pcMapInfoOf({ pc_per_pixel: false, pc_map_shape: null })).toBeNull();
    expect(pcMapInfoOf(undefined)).toBeNull();
    expect(pcMapInfoOf({})).toBeNull();
  });
  it('carries the rows and columns of the map', () => {
    expect(pcMapInfoOf({ pc_per_pixel: true, pc_map_shape: [300, 200] })).toEqual({ rows: 300, cols: 200 });
  });
  it('says "12 PCs" for a flat list, never "? x ?"', () => {
    expect(pcMapNote({ count: 12 }, null)).toEqual({ key: 'mapActiveFlat', params: { count: 12 } });
  });
  it('does not invent a size for a flat list of PCs', () => {
    expect(pcMapInfoOf({ pc_per_pixel: true, pc_map_shape: [12] })).toEqual({ count: 12 });
    expect(pcMapInfoOf({ pc_per_pixel: true })).toEqual({ rows: '?', cols: '?' });
  });
});

describe('pcMapActionOf', () => {
  it('keeps the two actions that changed a map', () => {
    expect(pcMapActionOf('shifted')).toBe('shifted');
    expect(pcMapActionOf('replaced')).toBe('replaced');
  });
  it('drops everything else', () => {
    expect(pcMapActionOf('none')).toBeNull();
    expect(pcMapActionOf(undefined)).toBeNull();
    expect(pcMapActionOf('kept')).toBeNull();
  });
});

describe('pcMapNote', () => {
  it('what just happened wins over the standing description', () => {
    expect(pcMapNote({ rows: 3, cols: 4 }, 'shifted').key).toBe('mapShifted');
    expect(pcMapNote({ rows: 3, cols: 4 }, 'replaced').key).toBe('mapReplaced');
  });
  it('describes an active map with its size', () => {
    expect(pcMapNote({ rows: 3, cols: 4 }, null)).toEqual({ key: 'mapActive', params: { rows: 3, cols: 4 } });
  });
  it('says nothing without a map and without an action', () => {
    expect(pcMapNote(null, null)).toBeNull();
  });
});
