// @vitest-environment node
/**
 * What actually goes on the wire for the two EDS requests whose backend
 * support existed for a while with no caller.
 *
 * Kept apart from the component tests on purpose: those mock
 * `../../services/api` wholesale, so they can only ever prove what the panel
 * asked the client for — never what the client sent. The defect being pinned
 * here was exactly of that shape: the backend accepted `scale_um`,
 * `authored_elements` and `authored_step_um`, the UI never sent them, and
 * every test in the tree was green.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const inst = {
    get: vi.fn(() => Promise.resolve({ data: {} })),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import { edsApi, edsExportApi } from '../../services/api';

const lastPost = (url) => {
  const call = [...axios.__inst.post.mock.calls].reverse().find((c) => c[0] === url);
  return call ? call[1] : null;
};

beforeEach(() => { vi.clearAllMocks(); });

describe('POST /api/eds/auto-classify — one statement about the smoothing box', () => {
  it('sends scale_um, and NOT scale, when a physical width is given', async () => {
    await edsApi.autoClassify({ scale: 7, scaleUm: 2.5 });
    const body = lastPost('/api/eds/auto-classify');
    expect(body.scale_um).toBe(2.5);
    // Two contradictory widths in one request, resolved by a precedence rule
    // invisible to whoever reads it.
    expect('scale' in body).toBe(false);
  });

  it('sends scale when there is no physical width — the pixel path is untouched', async () => {
    await edsApi.autoClassify({ scale: 7 });
    const body = lastPost('/api/eds/auto-classify');
    expect(body.scale).toBe(7);
    expect('scale_um' in body).toBe(false);
  });

  it('sends neither when neither is set, so the backend default still applies', async () => {
    await edsApi.autoClassify({});
    const body = lastPost('/api/eds/auto-classify');
    expect('scale' in body).toBe(false);
    expect('scale_um' in body).toBe(false);
  });
});

describe('POST /api/eds/presets — the metadata the warnings need', () => {
  it('records the element list and step size it was authored against', async () => {
    await edsExportApi.savePreset({
      name: 'Al matrix', settings: {},
      authoredElements: ['Al', 'Si', 'Fe'], authoredStepUm: 0.5,
      materialClass: 'AA6061', matrixElement: 'Al',
    });
    const body = lastPost('/api/eds/presets');
    // `element_set_differs` and `step_size_differs` are gated on exactly
    // these two; without them the two warnings can never fire.
    expect(body.authored_elements).toEqual(['Al', 'Si', 'Fe']);
    expect(body.authored_step_um).toBe(0.5);
    expect(body.material_class).toBe('AA6061');
    expect(body.matrix_element).toBe('Al');
  });

  it('OMITS what this scan cannot supply rather than sending an empty claim', async () => {
    await edsExportApi.savePreset({
      name: 'x', settings: {},
      authoredElements: [], authoredStepUm: null,
      materialClass: '', matrixElement: '',
    });
    const body = lastPost('/api/eds/presets');
    // "Authored against no elements, at a step of nothing" is a false
    // record, and a false record is worse than none: the check would then
    // compare this scan against a claim nobody made.
    expect('authored_elements' in body).toBe(false);
    expect('authored_step_um' in body).toBe(false);
    expect('material_class' in body).toBe(false);
    expect('matrix_element' in body).toBe(false);
    expect(body.name).toBe('x');
  });

  it('treats a zero step size as no step size, not as a measurement', async () => {
    await edsExportApi.savePreset({ name: 'x', settings: {}, authoredStepUm: 0 });
    expect('authored_step_um' in lastPost('/api/eds/presets')).toBe(false);
  });

  // Same shape of defect as the three above, found the same way: the backend
  // has accepted `tags` since the feature shipped, the picker filters on
  // them, and the client never put them on the wire - so nine of ten presets
  // in a real user directory carry `tags: []` and the filter has nothing to
  // filter on.
  it('sends the tags the save form collected', async () => {
    await edsExportApi.savePreset({
      name: 'Al matrix', settings: {}, tags: ['rolled', 'qa'],
    });
    expect(lastPost('/api/eds/presets').tags).toEqual(['rolled', 'qa']);
  });

  it('copies the tag list rather than sending the caller its own array', async () => {
    const tags = ['rolled'];
    await edsExportApi.savePreset({ name: 'x', settings: {}, tags });
    const sent = lastPost('/api/eds/presets').tags;
    expect(sent).toEqual(['rolled']);
    expect(sent).not.toBe(tags);
  });

  it('omits tags when there are none, so the backend default still applies', async () => {
    await edsExportApi.savePreset({ name: 'x', settings: {}, tags: [] });
    expect('tags' in lastPost('/api/eds/presets')).toBe(false);
    await edsExportApi.savePreset({ name: 'y', settings: {} });
    expect('tags' in lastPost('/api/eds/presets')).toBe(false);
  });
});
