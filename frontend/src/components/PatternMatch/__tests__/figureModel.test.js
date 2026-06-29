// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { defaultModel, addPanel, addElement, removeElement, updateElement,
  resizeElement, moveElement, alignElements, distributeElements, sameSize,
  niceScaleLength, letterString, reseedIdsFrom } from '../figureModel';

describe('defaultModel', () => {
  it('lays out two panels side by side within [0,1]', () => {
    const m = defaultModel(['experimental', 'simulated']);
    const panels = m.elements.filter(e => e.type === 'panel');
    expect(panels).toHaveLength(2);
    for (const p of panels) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x + p.w).toBeLessThanOrEqual(1.0001);
    }
    expect(panels[0].x).toBeLessThan(panels[1].x);
  });
  it('produces at least one panel even with no sources', () => {
    const m = defaultModel([]);
    expect(m.canvas).toBeTruthy();
    expect(Array.isArray(m.elements)).toBe(true);
  });
});

describe('addPanel/removeElement/updateElement', () => {
  it('adds and removes immutably', () => {
    let m = defaultModel(['experimental']);
    const before = m.elements.length;
    m = addPanel(m, 'ncc');
    expect(m.elements.length).toBe(before + 1);
    const id = m.elements[m.elements.length - 1].id;
    m = updateElement(m, id, { x: 0.25 });
    expect(m.elements.find(e => e.id === id).x).toBe(0.25);
    m = removeElement(m, id);
    expect(m.elements.find(e => e.id === id)).toBeUndefined();
  });
  it('deep-merges nested patches (border)', () => {
    let m = defaultModel(['experimental']);
    const id = m.elements[0].id;
    m = updateElement(m, id, { border: { on: true } });
    const el = m.elements.find(e => e.id === id);
    expect(el.border.on).toBe(true);
    expect(el.border.width).toBe(2); // untouched
  });
});

describe('moveElement / resizeElement', () => {
  it('clamps move within [0,1]', () => {
    const out = moveElement({ x: 0.9, y: 0.9, w: 0.2, h: 0.2 }, 0.5, 0.5);
    expect(out.x).toBeLessThanOrEqual(1);
    expect(out.y).toBeLessThanOrEqual(1);
  });
  it('keeps w/h ratio when aspectLock', () => {
    const el = { x: 0.1, y: 0.1, w: 0.4, h: 0.4 };
    const out = resizeElement(el, 'se', 0.2, 0.05, { aspectLock: true });
    expect(out.w / out.h).toBeCloseTo(1.0, 5);
  });
  it('anchors the opposite corner on NW aspect-locked resize', () => {
    const el = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 }; // SE corner at (0.6,0.6)
    const out = resizeElement(el, 'nw', -0.1, -0.1, { aspectLock: true });
    expect(out.x + out.w).toBeCloseTo(0.6, 5);
    expect(out.y + out.h).toBeCloseTo(0.6, 5);
  });
  it('never produces NaN/Infinity when height starts at 0', () => {
    const out = resizeElement({ x: 0, y: 0, w: 0.4, h: 0 }, 'se', 0.1, 0.1, { aspectLock: true });
    expect(Number.isFinite(out.w)).toBe(true);
    expect(Number.isFinite(out.h)).toBe(true);
    expect(out.h).toBeGreaterThanOrEqual(0.03);
  });
});

describe('alignElements / distributeElements / sameSize', () => {
  it('aligns left to the min x', () => {
    const els = [{ id: 'a', x: 0.2, w: 0.1 }, { id: 'b', x: 0.5, w: 0.1 }];
    const patches = alignElements(els, 'left');
    expect(patches.find(p => p.id === 'b').patch.x).toBe(0.2);
  });
  it('distributes three elements evenly in x', () => {
    const els = [{ id: 'a', x: 0.0 }, { id: 'b', x: 0.1 }, { id: 'c', x: 0.4 }];
    const patches = distributeElements(els, 'x');
    expect(patches.find(p => p.id === 'b').patch.x).toBeCloseTo(0.2);
  });
  it('sameSize copies the first element width', () => {
    const els = [{ id: 'a', w: 0.3 }, { id: 'b', w: 0.1 }];
    const patches = sameSize(els, 'w');
    expect(patches.find(p => p.id === 'b').patch.w).toBe(0.3);
  });
});

describe('niceScaleLength', () => {
  it('returns a 1/2/5 ×10^n value', () => {
    const { valueUm } = niceScaleLength(0.1, 1.0, 200); // 0.1 µm/px, full panel native 200px -> 20 µm wide
    expect([1, 2, 5, 10, 20, 50, 100]).toContain(valueUm);
  });
  it('returns zero (no NaN/Infinity) for a zero step size', () => {
    expect(niceScaleLength(0, 1.0, 200)).toEqual({ valueUm: 0, fracOfPanel: 0 });
  });
  it('bar never exceeds the panel (fracOfPanel <= 1)', () => {
    const { fracOfPanel } = niceScaleLength(0.05, 1.0, 156);
    expect(fracOfPanel).toBeGreaterThan(0);
    expect(fracOfPanel).toBeLessThanOrEqual(1);
  });
});

describe('reseedIdsFrom', () => {
  it('prevents id collisions after loading a preset', () => {
    reseedIdsFrom({ elements: [{ id: 'panel_900' }] });
    const m = addElement({ canvas: {}, elements: [] }, { type: 'panel' });
    const suffix = Number(/_(\d+)$/.exec(m.elements[0].id)[1]);
    expect(suffix).toBeGreaterThan(900);
  });
});

describe('letterString', () => {
  it('formats styles', () => {
    expect(letterString(0, 'paren')).toBe('(a)');
    expect(letterString(1, 'rparen')).toBe('b)');
    expect(letterString(2, 'upper')).toBe('C');
  });
});
