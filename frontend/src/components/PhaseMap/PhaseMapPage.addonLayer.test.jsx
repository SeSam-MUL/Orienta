// @vitest-environment jsdom
/**
 * The add-on map survives the page's own re-seed — measured through the PAGE.
 *
 * This file exists because of what the unit tests could not see. The add-on
 * hand-over is three parts: the store queues a request, PhaseMapPage drains it
 * into the stack, and useLayerStack re-seeds that stack when the result id
 * arrives. Each part had tests and each part was right; together they lost the
 * user's first click, and the first person to notice was a human clicking
 * through the built app.
 *
 * So the page is rendered here, with the real store, the real drain, the real
 * hook and the real reducer. Only the network and the leaf components that
 * need a canvas are mocked — mock the hook and this test proves nothing.
 *
 * Its neighbour PhaseMapPage.frame.test.jsx says a full mount is "brittle",
 * which was true and is why nobody had written one. It is bearable here
 * because everything under ../../services/api is a single mock and the
 * assertions are about one label in the layer list.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

const RESULT_ID = 'spherical_123';
const ADDON_LAYER_ID = `addon:bc-gmm/${RESULT_ID}/addon.bc_gmm/component_map#run-1`;
const ADDON_LABEL = 'Mixture component (rank)';

const listResults = vi.fn();
// How this test knows the RE-SEED has happened — the thing whose timing the
// bug was about.
//
// The first version waited for the text "Phase Map" to appear and called that
// the proof. It is not: "Phase Map" is also a fixed option in the A/B compare
// pickers, so it is on screen from the first paint, and the assertion ran
// BEFORE the re-seed. The test passed with the fix reverted — a guard that
// agreed with the bug. Caught by reverting it, which is the only thing that
// catches this.
//
// The seeded phase layer FETCHES ITS BITMAP, so the call is the event rather
// than a word on the screen. Named ('phase'), because `phaseMapApi.layer` has
// a second caller in this page — the neighbourhood zoom — which is closed
// here today and need not stay closed for this test to keep meaning what it
// says.

vi.mock('../../services/api', () => {
  const ok = (data) => Promise.resolve({ data });
  // Every api OBJECT this page touches, each answering with an empty 200.
  // An object missing from the factory throws loudly at first use rather
  // than answering empty, so this cannot turn a regression into a pass.
  //
  // A Proxy and not a hand-written list: PhaseMapPage calls across eight api
  // objects, and naming them one at a time turns this file into a catalogue
  // that goes stale the moment someone adds a call. What the assertions
  // depend on is overridden below; everything else may answer with nothing,
  // because a page that cannot draw a layer still has a layer LIST, and the
  // list is what is under test.
  const anyApi = (overrides = {}) => new Proxy({ ...overrides }, {
    get: (target, prop) => {
      if (prop in target) return target[prop];
      if (typeof prop !== 'string') return undefined;
      target[prop] = vi.fn(() => ok({}));
      return target[prop];
    },
  });
  const layer = () => ok({ image: '', scale: null });
  return {
    default: anyApi(),
    indexApi: anyApi({ listResults: (...a) => listResults(...a) }),
    phaseMapApi: anyApi({ layer: vi.fn(layer) }),
    analysisApi: anyApi(),
    ebsdApi: anyApi(),
    h5Api: anyApi(),
    pcApi: anyApi(),
    // 404 is the honest answer for "not computed yet", and the page has a
    // branch for it.
    forwardDiagApi: anyApi({
      summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    refinementApi: anyApi({
      summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    citationsApi: anyApi({ forResult: vi.fn(() => ok({ steps: [], bibtex: '' })) }),
  };
});

vi.mock('../../services/addonsApi', () => ({
  addonsApi: {
    // Returns no image: the fetch failing or returning nothing must not take
    // the layer off the stack, and this test is about the stack.
    mapImage: vi.fn(() => Promise.resolve({ data: { image: '', scale: null } })),
  },
}));

// Leaves that need a real canvas, a real WebGL context or a 4 MB plotting
// library. None of them can decide what is on the layer stack.
vi.mock('./LayeredCanvas', () => ({ default: () => <div data-testid="canvas" /> }));
vi.mock('../EDS/LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../EDS/MagnifierLens', () => ({ default: () => null }));
vi.mock('../PatternMatch/LinkedPatternImage', () => ({ default: () => null }));
vi.mock('../PoleFigure/openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }));

import { phaseMapApi } from '../../services/api';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';
import PhaseMapPage from './PhaseMapPage';

// `is_active: true` is load-bearing, not decoration: it puts the page on the
// branch the real hand-over takes. An add-on runs against the ACTIVE result,
// so the auto-adopt guard — which would otherwise ask the backend which file
// is loaded — is never reached, exactly as in the app.
const GALLERY_ENTRY = {
  id: RESULT_ID, result_id: RESULT_ID, label: 'spherical — Al', is_active: true,
  source_file: 'D:/data/SampleB.h5oina', phases: ['Al'], shape: [10, 10],
  n_pixels: 100, ci_mean: 0.3,
};

beforeEach(() => {
  vi.clearAllMocks();
  listResults.mockResolvedValue({ data: { results: [GALLERY_ENTRY] } });
  useAddonLayerRequests.setState({ requests: [], superseded: [], citationTick: 0 });
});

afterEach(cleanup);

describe('an add-on map handed to a Phase Maps page that was never opened', () => {
  it('is still on the stack after the page seeds itself', async () => {
    // The exact sequence the user performs: the Add-ons page queues the
    // request and navigates. The page then mounts, learns its result id for
    // the first time, and re-seeds the stack from the default preset.
    useAddonLayerRequests.getState().request(ADDON_LAYER_ID, ADDON_LABEL);

    render(<PhaseMapPage isActive />);

    // The seeded layer proves the re-seed has run — without it this test
    // would pass on a page that never seeded at all, which is the state the
    // stack is in for the first instant of every mount.
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    await waitFor(() => expect(phaseMapApi.layer).toHaveBeenCalledWith('phase', expect.anything()));
    // findAll: the label also names the layer in the A/B compare pickers.
    expect((await screen.findAllByText(ADDON_LABEL)).length)
      .toBeGreaterThan(0);
  });

  it('does not keep a map that belongs to a different result', async () => {
    // The other half, and the reason the keep is keyed on the result rather
    // than on "it is an add-on layer": this map draws another result's pixels
    // and the fetch refuses it by name. Carried across, it would sit on the
    // stack as a permanent error row.
    useAddonLayerRequests.getState().request(
      'addon:bc-gmm/some_other_result/addon.bc_gmm/component_map#run-1',
      'Map of another result');

    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    await waitFor(() => expect(phaseMapApi.layer).toHaveBeenCalledWith('phase', expect.anything()));
    await waitFor(() =>
      expect(screen.queryByText('Map of another result')).toBeNull());
  });
});
