/**
 * phaseDegeneracy — pure detection of crystallographically degenerate or
 * duplicate phases in a selected phase set.
 *
 * EBSD/Kikuchi indexing distinguishes structure (Laue class + lattice
 * metric), not chemistry. Two phases with the same Laue class + centering and
 * near-equal lattice parameters cannot be told apart by indexing.
 *
 * No React, no I/O — safe to unit-test in isolation.
 */

const HARD_REL_TOL = 0.025;  // 2.5% per-axis -> "EBSD cannot separate"
const SOFT_REL_TOL = 0.06;   // 6% per-axis  -> "maybe hard to separate"
const ANGLE_TOL = 1.5;       // degrees

function hasLattice(p) {
  return !!p && p.lattice_a > 0 && p.lattice_b > 0 && p.lattice_c > 0;
}

function relDiff(x, y) {
  const lo = Math.min(x, y);
  return lo > 0 ? Math.abs(x - y) / lo : Infinity;
}

function latticeRelMax(a, b) {
  return Math.max(
    relDiff(a.lattice_a, b.lattice_a),
    relDiff(a.lattice_b, b.lattice_b),
    relDiff(a.lattice_c, b.lattice_c),
  );
}

function angleMax(a, b) {
  return Math.max(
    Math.abs((a.lattice_alpha ?? 90) - (b.lattice_alpha ?? 90)),
    Math.abs((a.lattice_beta ?? 90) - (b.lattice_beta ?? 90)),
    Math.abs((a.lattice_gamma ?? 90) - (b.lattice_gamma ?? 90)),
  );
}

function phaseLabel(p, idx) {
  return (p && (p.formula || p.filename)) || `Phase ${idx + 1}`;
}

/**
 * Classify the relationship between two phases.
 * @returns {'hard'|'soft'|'none'}
 */
export function classifyPair(a, b) {
  if (!a || !b) return 'none';
  if (a.path && a.path === b.path) return 'hard';

  if (hasLattice(a) && hasLattice(b)) {
    if (!a.laue_class || a.laue_class !== b.laue_class) return 'none';
    if (angleMax(a, b) > ANGLE_TOL) return 'none';
    const relMax = latticeRelMax(a, b);
    const sameCentering = !!a.centering && a.centering === b.centering;
    if (sameCentering && relMax <= HARD_REL_TOL) return 'hard';
    if (relMax <= SOFT_REL_TOL) return 'soft';
    return 'none';
  }

  // Fallback when lattice data is missing on either side.
  if (a.space_group_number != null
      && a.space_group_number === b.space_group_number
      && a.pearson && a.pearson === b.pearson) {
    return 'hard';
  }
  return 'none';
}

/**
 * Detect degeneracy clusters in a selected phase array.
 * @returns {{clusters: Array<{tier, members, reason}>, byPhaseIndex: Object}}
 */
export function detectPhaseDegeneracy(phases) {
  const list = Array.isArray(phases) ? phases : [];
  const n = list.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (x, y) => { parent[find(x)] = find(y); };

  const softPairs = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const rel = classifyPair(list[i], list[j]);
      if (rel === 'hard') union(i, j);
      else if (rel === 'soft') softPairs.push([i, j]);
    }
  }

  // Hard clusters = connected components of size >= 2.
  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(i);
  }
  const clusters = [];
  const hardMembers = new Set();
  for (const members of groups.values()) {
    if (members.length < 2) continue;
    members.forEach((m) => hardMembers.add(m));
    clusters.push({
      tier: 'hard',
      members,
      reason: members.map((m) => phaseLabel(list[m], m)).join(' ≈ '),
    });
  }

  // Soft pairs — skip any pair touching a phase already in a hard cluster.
  for (const [i, j] of softPairs) {
    if (hardMembers.has(i) || hardMembers.has(j)) continue;
    clusters.push({
      tier: 'soft',
      members: [i, j],
      reason: `${phaseLabel(list[i], i)} / ${phaseLabel(list[j], j)}`,
    });
  }

  // Per-phase lookup for the card badges (hard wins over soft).
  const byPhaseIndex = {};
  for (const c of clusters) {
    for (const m of c.members) {
      const partners = c.members
        .filter((x) => x !== m)
        .map((x) => phaseLabel(list[x], x));
      const existing = byPhaseIndex[m];
      if (!existing || (existing.tier === 'soft' && c.tier === 'hard')) {
        byPhaseIndex[m] = { tier: c.tier, partners };
      }
    }
  }

  return { clusters, byPhaseIndex };
}

/**
 * Indices to remove to collapse a hard cluster to one phase (keeps the
 * lowest / first-selected index). Returns [] for soft clusters.
 */
export function planReduction(cluster) {
  if (!cluster || cluster.tier !== 'hard' || !Array.isArray(cluster.members)
      || cluster.members.length < 2) {
    return [];
  }
  return [...cluster.members].sort((a, b) => a - b).slice(1);
}
