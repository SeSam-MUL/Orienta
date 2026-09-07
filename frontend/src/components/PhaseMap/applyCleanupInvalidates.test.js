// Guard: an action that MUTATES the stored result must invalidate the layer
// cache, or the map keeps showing the state from before the mutation.
//
// This is a source-level check, and it is weaker than a behaviour test on
// purpose: a full PhaseMapPage mount in jsdom is brittle (the existing
// PhaseMapPage.frame.test.jsx says so and tests the panel in isolation
// instead). It is here because the defect it guards is invisible at runtime —
// "Apply cleanup" reported "moved N pixels" and the picture did not change,
// because the handler refreshed only the legacy /render image, which is not
// what the page displays (measured 2026-09-03: 6 apply-cleanup calls, no
// visible change).
//
// If PhaseMapPage is restructured, replace this with a mounted test rather
// than deleting it.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(
  fileURLToPath(new URL('./PhaseMapPage.jsx', import.meta.url)), 'utf8',
);

/** Body of the applyCleanup success path, up to its catch. */
function applyCleanupBlock() {
  const start = SRC.indexOf('phaseMapApi.applyCleanup(');
  expect(start, 'applyCleanup call site not found — update this guard').toBeGreaterThan(-1);
  const end = SRC.indexOf('} catch', start);
  expect(end, 'no catch after applyCleanup — update this guard').toBeGreaterThan(start);
  return SRC.slice(start, end);
}

describe('Apply cleanup refreshes what the user is actually looking at', () => {
  it('invalidates the cached layer bitmaps after a successful apply', () => {
    const block = applyCleanupBlock();
    expect(
      /cacheFlush\s*\(/.test(block),
      'the applyCleanup success path does not flush the layer cache, so the '
      + 'displayed map keeps its pre-apply bitmaps',
    ).toBe(true);
  });

  it('flushes through the shared predicate rather than a fourth hand-written list', () => {
    expect(applyCleanupBlock()).toMatch(/cacheFlush\(\s*cleanupAffectsLayer\s*\)/);
    expect(SRC).toMatch(/import\s*\{[^}]*cleanupAffectsLayer[^}]*\}\s*from\s*'\.\/hooks\/useLayerStack'/);
  });
});
