/**
 * The viewer takes its "Open PC Refinement" number from usePcPatternCount, under
 * the page id the router really uses - otherwise the hook never hears that the
 * viewer is shown again and a pattern removed on the PC Refinement page stays
 * counted.
 *
 * Source-level: standing up the whole viewer needs its stores and a backend.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const viewer = readFileSync(resolve(here, '..', 'EBSDViewer.jsx'), 'utf8');
const app = readFileSync(resolve(here, '..', '..', '..', 'App.jsx'), 'utf8');

describe('"Open PC Refinement" number', () => {
  it('comes from the hook, not from a local state only Add sets', () => {
    const m = viewer.match(/const \[pcPatternCount, setPcPatternCount\] = (\w+)\('([^']+)', (\w+)\);/);
    expect(m).not.toBeNull();
    expect(m[1]).toBe('usePcPatternCount');
    // The id the router announces in 'page-changed' and mounts the viewer under.
    expect(app).toContain(`data-page="${m[2]}"`);
    // ...and the viewer's active dataset, so a processed copy made while the
    // viewer stays on screen is counted afresh.
    expect(m[3]).toBe('activeDataset');
    expect(viewer).toMatch(/const \[activeDataset, setActiveDataset\] = useState/);
  });

  it('is still set from the Add reply at once', () => {
    const add = viewer.slice(viewer.indexOf('const handleAddCurrentPattern'));
    expect(add.slice(0, add.indexOf('};'))).toMatch(/setPcPatternCount\(n\)/);
  });
});
