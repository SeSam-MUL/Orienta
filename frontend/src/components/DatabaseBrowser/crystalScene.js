/**
 * Pure geometry helpers for the Three.js crystal-structure viewer — no three /
 * DOM dependency, so they are unit-testable. The imperative WebGL scene lives in
 * crystalScene3d.js and consumes these.
 */

/** Legend data: one entry per element, ordered by descending atom count. */
export function elementSummary(atoms) {
  const by = new Map();
  for (const a of atoms) {
    const e = by.get(a.element);
    if (e) e.count += 1;
    else by.set(a.element, { element: a.element, color: a.color, radius: a.radius, count: 1 });
  }
  return [...by.values()].sort((x, y) => y.count - x.count);
}

/** Bounding-box centre + a radius that encloses every point (min 1). */
export function sceneBounds(points) {
  if (!points.length) return { center: [0, 0, 0], radius: 1 };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  }
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  let radius = 0;
  for (const p of points) {
    const d = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    if (d > radius) radius = d;
  }
  return { center, radius: Math.max(radius, 1) };
}

/** The 12 edges of the unit cell as [start, end] Cartesian segment pairs. */
export function cellEdges(cellVectors) {
  const [a, b, c] = cellVectors;
  const add = (u, v) => [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
  const O = [0, 0, 0];
  const corner = {
    O, a, b, c, ab: add(a, b), ac: add(a, c), bc: add(b, c), abc: add(add(a, b), c),
  };
  const pairs = [
    ['O', 'a'], ['O', 'b'], ['O', 'c'], ['a', 'ab'], ['a', 'ac'], ['b', 'ab'],
    ['b', 'bc'], ['c', 'ac'], ['c', 'bc'], ['ab', 'abc'], ['ac', 'abc'], ['bc', 'abc'],
  ];
  return pairs.map(([p, q]) => [corner[p], corner[q]]);
}

/**
 * Sphere radius (Å, data space) for ball-and-stick: a fraction of the covalent
 * radius so atoms read as balls with visible gaps, not a space-filling blob.
 * Real 3D spheres scale with zoom, so this is a physical size, not pixels.
 */
export function atomSphereRadius(covalentRadius) {
  const r = covalentRadius > 0 ? covalentRadius : 1.2;
  return Math.min(0.9, Math.max(0.28, r * 0.32));
}
