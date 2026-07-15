import { describe, it, expect } from 'vitest';
import { elementSummary, sceneBounds, cellEdges, atomSphereRadius } from './crystalScene';

const atoms = [
  { element: 'Al', color: '#bfa6a6', radius: 1.21, cart: [0, 0, 0] },
  { element: 'Al', color: '#bfa6a6', radius: 1.21, cart: [4, 0, 0] },
  { element: 'Fe', color: '#e06633', radius: 1.32, cart: [2, 2, 2] },
];

describe('elementSummary', () => {
  it('groups by element, keeps colour, orders by count desc', () => {
    const s = elementSummary(atoms);
    expect(s.map((e) => e.element)).toEqual(['Al', 'Fe']);
    expect(s[0]).toMatchObject({ element: 'Al', color: '#bfa6a6', count: 2 });
    expect(s[1]).toMatchObject({ element: 'Fe', count: 1 });
  });
});

describe('sceneBounds', () => {
  it('returns the bbox centre and an enclosing radius', () => {
    const { center, radius } = sceneBounds(atoms.map((a) => a.cart));
    expect(center).toEqual([2, 1, 1]);
    expect(radius).toBeGreaterThanOrEqual(Math.sqrt(4 + 1 + 1)); // encloses farthest atom
  });
  it('never returns a zero radius (single point)', () => {
    expect(sceneBounds([[0, 0, 0]]).radius).toBeGreaterThan(0);
  });
});

describe('cellEdges', () => {
  it('produces 12 edges for a parallelepiped', () => {
    const edges = cellEdges([[4, 0, 0], [0, 4, 0], [0, 0, 4]]);
    expect(edges).toHaveLength(12);
    for (const [p, q] of edges) {
      expect(p).toHaveLength(3);
      expect(q).toHaveLength(3);
    }
    // origin→a edge present
    expect(edges).toContainEqual([[0, 0, 0], [4, 0, 0]]);
  });
});

describe('atomSphereRadius', () => {
  it('scales covalent radius into a ball-and-stick sphere size, clamped', () => {
    expect(atomSphereRadius(1.2)).toBeGreaterThan(0);
    expect(atomSphereRadius(1.2)).toBeLessThan(1.2); // ball-and-stick, not space-filling
    expect(atomSphereRadius(99)).toBeLessThanOrEqual(0.9); // clamped
    expect(atomSphereRadius(0)).toBeGreaterThan(0); // fallback
  });
});
