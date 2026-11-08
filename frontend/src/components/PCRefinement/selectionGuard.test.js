import { describe, it, expect } from 'vitest';
import { createSelectionGuard, createLatestRequest } from './selectionGuard';

describe('createSelectionGuard', () => {
  it('knows the selected pattern', () => {
    const g = createSelectionGuard();
    expect(g.selected()).toBeNull();
    g.select(2);
    expect(g.selected()).toBe(2);
    expect(g.isSelected(2)).toBe(true);
    expect(g.isSelected(1)).toBe(false);
  });
  it('nothing belongs to a cleared selection', () => {
    const g = createSelectionGuard();
    g.select(0);
    g.select(null);
    expect(g.selected()).toBeNull();
    expect(g.isSelected(0)).toBe(false);
    expect(g.isSelected(null)).toBe(false);
  });
  it('a new selection voids every answer asked before it, even for the same pattern', () => {
    const g = createSelectionGuard();
    const first = g.select(1);
    expect(g.isCurrent(first)).toBe(true);
    const second = g.select(1);
    expect(g.isCurrent(first)).toBe(false);
    expect(g.isCurrent(second)).toBe(true);
  });
});

describe('createLatestRequest', () => {
  it('only the newest request is the latest', () => {
    const r = createLatestRequest();
    const a = r.next();
    const b = r.next();
    expect(r.isLatest(a)).toBe(false);
    expect(r.isLatest(b)).toBe(true);
  });
});
