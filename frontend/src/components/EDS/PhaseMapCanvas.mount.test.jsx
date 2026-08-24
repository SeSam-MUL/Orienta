// @vitest-environment jsdom
/**
 * Guards the regression that started this work: `PhaseMapCanvas` is fully
 * built (image, letterbox-correct click mapping, rectangle and polygon
 * paint) but lost its only mount point in the FEAT-33 EDSPage rewrite
 * (commit e8f5fd0, 2026-05-14). The backend rendered the PNG, the response
 * carried it, React stored it — and nothing drew it.
 *
 * These are source-level assertions on purpose: the defect is "the
 * component exists but nothing renders it", which a render test of the
 * component itself cannot catch.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(resolve(here, 'EDSPage.jsx'), 'utf8');

describe('EDS phase map is reachable from the page', () => {
  it('imports PhaseMapCanvas from PhaseMapPanel', () => {
    const importLine = page.match(
      /import\s*\{([^}]*)\}\s*from\s*['"]\.\/PhaseMapPanel['"]/,
    );
    expect(importLine, 'EDSPage must import from ./PhaseMapPanel').toBeTruthy();
    expect(importLine[1]).toMatch(/\bPhaseMapCanvas\b/);
  });

  it('mounts PhaseMapCanvas somewhere in the layout', () => {
    expect(page).toMatch(/<PhaseMapCanvas\b/);
  });

  it('passes the shared handle to it, not a fresh one', () => {
    const usage = page.match(/<PhaseMapCanvas\b[^/>]*/);
    expect(usage).toBeTruthy();
    expect(usage[0]).toMatch(/handle=\{phaseMapHandle\}/);
  });
});
