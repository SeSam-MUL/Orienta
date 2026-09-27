// @vitest-environment jsdom
/**
 * The left column of the "Element maps" tab is `overflow: hidden` on purpose
 * — the composite overlay must stay pinned while the layer list below it is
 * worked on. That is only sound if the layer list brings its own scroll
 * container; without one it is clipped, and with eleven elements every row
 * past the third was unreachable (user report 2026-08-31, 9x12 scan).
 *
 * Source-level on purpose: jsdom has no layout, so `scrollHeight` cannot be
 * measured here. The live measurement is in the commit message.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(resolve(here, 'EDSPage.jsx'), 'utf8');

// The left column starts at its marker comment and ends at the splitter.
const leftColumn = () => {
  const start = page.indexOf('{/* LEFT: Overlay + Layer panel */}');
  const end = page.indexOf('{/* SPLITTER */}', start);
  expect(start, 'left-column marker comment').toBeGreaterThan(-1);
  expect(end, 'splitter marker comment').toBeGreaterThan(start);
  return page.slice(start, end);
};

describe('EDS element-maps left column', () => {
  it('clips (so the overlay stays pinned) ...', () => {
    const col = leftColumn();
    const opening = col.slice(0, col.indexOf('<GroupBox'));
    expect(opening).toMatch(/overflow:\s*'hidden'/);
  });

  it('... and therefore the layer list must scroll on its own', () => {
    const col = leftColumn();
    const panelAt = col.indexOf('<LayerStackPanel');
    expect(panelAt).toBeGreaterThan(-1);
    // The nearest element opened before <LayerStackPanel> is its scroll box.
    const before = col.slice(0, panelAt);
    const lastOpen = before.lastIndexOf('<div');
    const wrapper = before.slice(lastOpen);
    expect(wrapper).toMatch(/overflowY:\s*'auto'/);
    // A scroll box inside a flex column only gets a height to scroll within
    // when it may shrink below its content.
    expect(wrapper).toMatch(/minHeight:\s*0/);
  });
});
