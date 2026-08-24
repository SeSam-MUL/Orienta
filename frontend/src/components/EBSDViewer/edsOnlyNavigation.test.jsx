// @vitest-environment jsdom
/**
 * An EDS-only acquisition (Aztec "Elementverteilungsdaten") carries no
 * diffraction patterns, no detector and no orientations, so the EBSD viewer
 * has literally nothing to show. Loading one used to leave the user staring
 * at empty panes with only a toast to tell them why, and they had to work
 * out for themselves that the data lives on the EDS page.
 *
 * Source-level assertions: the defect is "the branch exists but does not
 * navigate", which a render test of the component cannot reach without
 * standing up the whole viewer, its stores and a backend.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(resolve(here, 'EBSDViewer.jsx'), 'utf8');
const app = readFileSync(resolve(here, '../../App.jsx'), 'utf8');

describe('EDS-only files go straight to the EDS page', () => {
  it('navigates after a load', () => {
    const block = src.slice(
      src.indexOf("content_mode === 'eds_only'"),
      src.indexOf("} else {", src.indexOf("content_mode === 'eds_only'")),
    );
    expect(block).toMatch(/onNavigate\?\.\('eds'\)/);
  });

  it('navigates after a file switch too', () => {
    expect(src).toMatch(
      /info\?\.content_mode === 'eds_only'\s*\)\s*onNavigate\?\.\('eds'\)/,
    );
  });

  it("uses a page id the router actually knows", () => {
    // A previous bug shipped 'phase-map' where the router wanted 'phasemap',
    // producing a blank screen. Pin the id against App.jsx.
    expect(app).toMatch(/\bid:\s*'eds'/);
  });

  it('lists onNavigate in the deps of every callback that navigates', () => {
    // A stale closure here would call an onNavigate from an earlier render.
    // Only the callbacks that actually navigate need it - checking every
    // callback that happens to use fetchAtlas would flag an unrelated one.
    let from = 0;
    let checked = 0;
    for (;;) {
      const at = src.indexOf("onNavigate?.('eds')", from);
      if (at === -1) break;
      from = at + 1;
      const close = src.indexOf('}, [', at);
      if (close === -1) continue;
      const deps = src.slice(close, src.indexOf(']);', close));
      // The render-time onClick handler at the bottom is not a callback.
      if (!deps.includes('fetchAtlas')) continue;
      expect(deps).toMatch(/onNavigate/);
      checked += 1;
    }
    expect(checked).toBe(2);
  });
});
