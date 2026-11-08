// @vitest-environment jsdom
//
// Regression: on a small window the selected-pattern list
// and the Remove button of the PC Refinement page could not be seen, and the
// column did not scroll. Its root clipped (`overflow: hidden`), so the left
// panel around it never overflowed and no scrollbar appeared anywhere.
//
// jsdom has no layout, so this pins the property that decides it: between the
// Remove button and the first element that scrolls, nothing may clip. The live
// check was made on a small window.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 1, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 1, col: 2 }],
    }),
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi: quiet, indexApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

const scrolls = (el) => ['auto', 'scroll'].includes(el.style.overflowY) || ['auto', 'scroll'].includes(el.style.overflow);
const clips = (el) => [el.style.overflow, el.style.overflowY].includes('hidden');

describe('PC Refinement left column', () => {
  it('reaches the Remove button by scrolling, with nothing clipping it first', async () => {
    render(<PCRefinement />);
    const remove = (await screen.findAllByTitle(/remove the selected pattern/i))[0];
    const chain = [];
    for (let el = remove.parentElement; el && !scrolls(el); el = el.parentElement) chain.push(el);
    const scroller = chain.length ? chain.at(-1).parentElement : remove.parentElement;
    expect(scroller, 'an element that scrolls above the Remove button').toBeTruthy();
    expect(scroller.dataset.testid).toBe('pc-preview-column');
    expect(chain.filter(clips)).toEqual([]);
    // A scroll box inside a flex column only gets a height to scroll within
    // when it may shrink below its content.
    expect(scroller.style.minHeight).toBe('0px');
  });

  it('keeps the pattern list scrollable on its own with a usable floor', async () => {
    render(<PCRefinement />);
    const item = await screen.findByTitle(/Pattern at row 1, col 2/);
    const list = item.parentElement;
    expect(list.style.overflowY).toBe('auto');
    expect(list.style.minHeight).toBe('80px');
  });
});
