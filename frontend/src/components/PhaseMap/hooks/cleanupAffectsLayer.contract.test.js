// Cross-source guard: the frontend's cleanup-invalidation list vs. what the
// backend actually does.
//
// `cleanupAffectsLayer` is a hand-written mirror of a decision that lives in
// backend/api/routes/phase_map.py. Two lists, one truth — and this repo has
// been bitten by exactly that shape before. A unit test of either side alone
// proves nothing here: both were internally consistent on 2026-09-03, and the
// map was still stale. So this test reads the BACKEND SOURCE and holds it
// against the predicate.
//
// The assertion is deliberately ONE-DIRECTIONAL:
//   backend filters a kind  ⇒  the predicate MUST claim it.
// Under-claiming is silent and user-visible ("the sliders do nothing" — the
// layer keeps its cached bitmap forever). Over-claiming only costs one extra
// refetch, so it stays allowed on purpose.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { cleanupAffectsLayer } from './useLayerStack';

const BACKEND = fileURLToPath(
  new URL('../../../../../backend/api/routes/phase_map.py', import.meta.url),
);

/** The `if kind == ...` chain of the layer renderer, split into branches. */
function layerKindBranches(src) {
  const lines = src.split(/\r?\n/);
  const isBranch = (l) => /^ {4}(el)?if kind[ .]/.test(l);
  const starts = [];
  lines.forEach((l, i) => { if (isBranch(l)) starts.push(i); });
  // The renderer's chain is the longest run of such branches in the file.
  const runs = [];
  let run = [starts[0]];
  for (let k = 1; k < starts.length; k += 1) {
    // Same chain = no other 4-space-indented `def` between the two branches.
    const between = lines.slice(starts[k - 1], starts[k]).some((l) => /^def /.test(l));
    if (between) { runs.push(run); run = [starts[k]]; } else { run.push(starts[k]); }
  }
  runs.push(run);
  const chain = runs.sort((a, b) => b.length - a.length)[0];

  // A branch body ends at the next branch, or — for the last one — where the
  // enclosing function ends (first non-empty line back at module indent).
  // Without that second stop the last branch swallowed the helper below it and
  // reported a cleanup dependency that branch does not have.
  const functionEnd = (from) => {
    for (let i = from + 1; i < lines.length; i += 1) {
      if (lines[i].trim() && !/^\s/.test(lines[i])) return i;
    }
    return lines.length;
  };

  return chain.map((start, i) => {
    const end = i + 1 < chain.length ? chain[i + 1] : functionEnd(start);
    const header = lines[start];
    const body = lines.slice(start, Math.min(end, lines.length)).join('\n');
    // Representative layer ids this branch answers for.
    const ids = [
      ...[...header.matchAll(/kind == "([^"]+)"/g)].map((m) => m[1]),
      ...[...header.matchAll(/kind\.startswith\("([^"]+)"\)/g)].map((m) => `${m[1]}z`),
    ];
    return { header: header.trim(), ids, usesCleanup: /effective_pid_2d|ipf_valid_2d/.test(body) };
  });
}

describe('cleanupAffectsLayer mirrors the backend layer renderer', () => {
  const branches = layerKindBranches(readFileSync(BACKEND, 'utf8'));

  it('found the layer renderer branch chain', () => {
    // Guards the parsing itself: if phase_map.py is restructured so this stops
    // matching, the test must fail loudly rather than pass on zero branches.
    expect(branches.length).toBeGreaterThanOrEqual(5);
    expect(branches.flatMap((b) => b.ids)).toContain('phase');
    expect(branches.some((b) => b.usesCleanup)).toBe(true);
  });

  it('claims every kind whose rendering reads the cleanup-filtered arrays', () => {
    const missed = branches
      .filter((b) => b.usesCleanup)
      .flatMap((b) => b.ids.map((id) => ({ id, header: b.header })))
      .filter(({ id }) => !cleanupAffectsLayer(id));

    expect(
      missed.map(({ id, header }) => `${id}  (backend: ${header})`),
    ).toEqual([]);
  });

  it('does not claim non-result layer sources', () => {
    // EDS elements, SE images, analysis maps and virtual BSE are served by
    // other endpoints that never see the cleanup params — flushing them on a
    // slider drag would be pure waste.
    for (const id of ['eds:Al', 'se:SE1', 'vbse', 'kam', 'gos', 'grain-boundaries', 'ci-threshold']) {
      expect(cleanupAffectsLayer(id)).toBe(false);
    }
  });
});
