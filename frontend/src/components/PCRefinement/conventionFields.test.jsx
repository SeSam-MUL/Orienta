// @vitest-environment jsdom
//
// The PC fields of the Detector Settings in three conventions, through the real page and the real texts:
// a person picks the kind of numbers (Bruker, EDAX/TSL, Oxford), types the three numbers, sees what Orienta
// takes them as and presses "Use these numbers". Typing alone applies nothing.
//
// The fake backend knows ONE detector, 128 rows x 156 columns (K = 156 / 128 = 1.21875), and converts by a
// table of answers worked out by hand, not by a formula (an unknown request fails the test):
//   Bruker (0.5019, 0.3301, 0.842)   = Oxford (0.5019, (1-0.3301)/K, 0.842/K) = (0.5019, 0.549662, 0.690872)
//                                    = TSL    (0.5019, 1-0.3301, 0.842)       = (0.5019, 0.6699, 0.842)
//   Oxford (0.5, 0.55, 0.71)         = Bruker (0.5, 1-0.55*K, 0.71*K)         = (0.5, 0.3296875, 0.8653125)
//   Oxford (0.5, 0.56, 0.71)         = Bruker (0.5, 1-0.56*K, 0.71*K)         = (0.5, 0.3175, 0.8653125)
//   Bruker (0.5, 0.3296875, 0.8653125) = Oxford (0.5, (1-0.3296875)/K, 0.8653125/K) = (0.5, 0.55, 0.71)
//   Bruker (0.5, 0.3175, 0.8653125)    = Oxford (0.5, (1-0.3175)/K, 0.8653125/K)    = (0.5, 0.56, 0.71)
//   Oxford (0.5019, 0.5497, 0.6909)  = Bruker (0.5019, 1-0.5497*K, 0.6909*K)  = (0.5019, 0.330053125, 0.842034375)
//                                    = TSL    (0.5019, 0.5497*K, 0.6909*K)    = (0.5019, 0.669946875, 0.842034375)
//   TSL    (0.5019, 0.6699, 0.842)   = Bruker (0.5019, 1-0.6699, 0.842)       = (0.5019, 0.3301, 0.842)
//   TSL    (0.5, 0.67, 0.84)         = Bruker (0.5, 1-0.67, 0.84)             = (0.5, 0.33, 0.84)
//   Oxford (0.5, 0.55, 0.71)         = TSL    (0.5, 0.55*K, 0.71*K)           = (0.5, 0.6703125, 0.8653125)
//   TSL    (0.5, 0.6703, 0.8653)     = Bruker (0.5, 1-0.6703, 0.8653)         = (0.5, 0.3297, 0.8653)
// A tall pattern, 156 rows x 128 columns (K = 128 / 156 = 0.820513), and a square one, 60 x 60 (K = 1):
//   Oxford (0.5, 0.55, 0.71) at 156 x 128 = Bruker (0.5, 1-0.55*K, 0.71*K) = (0.5, 0.548718, 0.582564)
//   Oxford (0.5, 0.55, 0.71) at 60 x 60   = Bruker (0.5, 1-0.55, 0.71)     = (0.5, 0.45, 0.71)
//   TSL    (0.5, 0.67, 0.84) at 156 x 128 = Bruker (0.5, 1-0.67, 0.84*K)   = (0.5, 0.33, 0.689231)
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  pc: [0.5019, 0.3301, 0.842], before: null, source: 'header', kind: null, convention: 'oxford', conventionKnown: true,
  shape: [128, 156], dataset: 'scan_a', calls: [], previews: [], unknown: [], hold: null, convertError: null,
  updateError: null, updateAnswer: {}, perPixel: false, patterns: [], statusPc: null, statusGate: null, updateGates: [], hadNoPc: false,
}));

const TABLE = [
  [[[0.5019, 0.3301, 0.842], 'bruker', 'oxford', [128, 156]], [0.5019, 0.549662, 0.690872]],
  [[[0.5019, 0.3301, 0.842], 'bruker', 'tsl', [128, 156]], [0.5019, 0.6699, 0.842]],
  [[[0.5, 0.55, 0.71], 'oxford', 'bruker', [128, 156]], [0.5, 0.3296875, 0.8653125]],
  [[[0.5, 0.56, 0.71], 'oxford', 'bruker', [128, 156]], [0.5, 0.3175, 0.8653125]],
  [[[0.5, 0.3296875, 0.8653125], 'bruker', 'oxford', [128, 156]], [0.5, 0.55, 0.71]],
  [[[0.5, 0.3175, 0.8653125], 'bruker', 'oxford', [128, 156]], [0.5, 0.56, 0.71]],
  [[[0.5026, 0.3272, 0.8458], 'bruker', 'oxford', [128, 156]], [0.5026, 0.552041, 0.693994]],
  [[[0.5, 0.67, 0.84], 'tsl', 'bruker', [128, 156]], [0.5, 0.33, 0.84]],
  [[[0.5019, 0.5497, 0.6909], 'oxford', 'bruker', [128, 156]], [0.5019, 0.330053125, 0.842034375]],
  [[[0.5019, 0.5497, 0.6909], 'oxford', 'tsl', [128, 156]], [0.5019, 0.669946875, 0.842034375]],
  [[[0.5019, 0.6699, 0.842], 'tsl', 'bruker', [128, 156]], [0.5019, 0.3301, 0.842]],
  [[[0.5, 0.55, 0.71], 'oxford', 'tsl', [128, 156]], [0.5, 0.6703125, 0.8653125]],
  [[[0.5, 0.6703, 0.8653], 'tsl', 'bruker', [128, 156]], [0.5, 0.3297, 0.8653]],
  [[[0.5, 0.55, 0.71], 'oxford', 'bruker', [156, 128]], [0.5, 0.548718, 0.582564]],
  [[[0.5, 0.55, 0.71], 'oxford', 'bruker', [60, 60]], [0.5, 0.45, 0.71]],
  [[[0.5, 0.67, 0.84], 'tsl', 'bruker', [156, 128]], [0.5, 0.33, 0.689231]],
  // the PC in use shown back in the chosen kind of numbers (the inverse of the rows above, by hand)
  [[[0.5, 0.3296875, 0.8653125], 'bruker', 'tsl', [128, 156]], [0.5, 0.6703125, 0.8653125]],
  [[[0.6, 0.4, 0.7], 'bruker', 'oxford', [128, 156]], [0.6, 0.492308, 0.574359]],         // (1-0.4)/K, 0.7/K
  [[[0.5019, 0.3301, 0.842], 'bruker', 'oxford', [156, 128]], [0.5019, 0.816441, 1.026188]],   // 0.6699*1.21875, 0.842*1.21875
  [[[0.5, 0.548718, 0.582564], 'bruker', 'oxford', [156, 128]], [0.5, 0.55, 0.71]],
  [[[0.5019, 0.3301, 0.842], 'bruker', 'oxford', [60, 60]], [0.5019, 0.6699, 0.842]],
  [[[0.5, 0.45, 0.71], 'bruker', 'oxford', [60, 60]], [0.5, 0.55, 0.71]],
  [[[0.5019, 0.3301, 0.842], 'bruker', 'tsl', [156, 128]], [0.5019, 0.6699, 1.026188]],
  [[[0.5, 0.33, 0.689231], 'bruker', 'tsl', [156, 128]], [0.5, 0.67, 0.84]],
];

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    // a test can hold the status answer back (statusGate): the page then has not heard of the file's convention yet
    status: () => (backend.statusGate || Promise.resolve()).then(() => ({ data: {
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'], n_patterns: backend.patterns.length,
      pc: backend.statusPc || backend.pc, pc_basis: backend.perPixel ? 'mean' : 'single', pc_per_pixel: backend.perPixel,
      pc_map_shape: backend.perPixel ? [2, 2] : null,
      pc_in_use: {
        pc: backend.source === 'missing' ? null : backend.pc, per_pixel: backend.perPixel,
        kind: backend.kind || ({ missing: 'none', header: 'file', manual: 'typed' }[backend.source] || 'file'),
        calibration: null, convention: backend.convention, convention_known: backend.conventionKnown,
      },
      patterns: backend.patterns.map((p, i) => ({ index: i, ...p })), pc_source: backend.source,
    } })),
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
    renderPreview: (req) => { backend.previews.push(req.pc); return ok({ simulated_b64: null, ncc: 0.5, ncc_reason: null }); },
    convert: async (pc, source, target, shape) => {
      backend.calls.push(['convert', pc, source, target, shape]);
      if (backend.hold) await backend.hold;
      if (backend.convertError) throw backend.convertError;
      const hit = TABLE.find(([[p, s, t, sh]]) => JSON.stringify([p, s, t, sh]) === JSON.stringify([pc, source, target, shape]));
      if (!hit) { backend.unknown.push(`convert ${JSON.stringify([pc, source, target, shape])}`); throw new Error('no such conversion'); }
      return { data: { pc: hit[1], source, target } };
    },
    updatePC: async (x, y, z, idx) => {
      backend.calls.push(['updatePC', x, y, z, idx]);
      if (backend.updateError) throw backend.updateError;
      backend.before = backend.pc;
      backend.pc = [x, y, z];
      backend.source = 'manual';
      // the backend has the PC at once; a test can hold the ANSWER back (updateGates: one promise per press)
      const gate = backend.updateGates.shift();
      if (gate) await gate;
      return { data: { success: true, ci: 0.2, segments: [], pc_map_action: 'none', pc_stored: true,
                       previous: { pc: backend.before, had_pc: true }, ...backend.updateAnswer } };
    },
    setDetector: async (shape, pc) => {
      backend.calls.push(['setDetector', shape, pc]);
      return { data: { success: true, pc_map_action: 'none' } };
    },
    putBack: async (dataset) => {
      backend.calls.push(['putBack', dataset]);
      if (backend.hadNoPc) {                                  // the file had no PC before the press: it has none again
        backend.pc = backend.before;
        backend.source = 'missing';
        return { data: { dataset, pc: null, had_pc: false, restored: [dataset], not_restored: [], pc_per_pixel: false } };
      }
      backend.pc = backend.before;
      backend.source = 'header';
      return { data: { dataset, pc: backend.before, had_pc: true, restored: [dataset], not_restored: [], pc_per_pixel: false } };
    },
  }, {
    // an unknown method is not answered with {}: a call the test never meant to allow fails it
    get: (t, k) => {
      if (k in t || typeof k === 'symbol' || k === 'then') return t[k];
      return () => { backend.unknown.push(k); throw new Error(`pcApi.${String(k)} is not in this test's fake`); };
    },
  });
  const ebsdApi = new Proxy({
    info: () => ok({ active_dataset: backend.dataset }),
    getDetector: () => ok({ shape: backend.shape, pc: backend.pc, sample_tilt: 70, camera_tilt: 4.3, azimuthal: 0, binning: 1 }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi, indexApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';
import useDataStore from '../../stores/useDataStore';

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});
class FakeImage {
  set src(_v) { this.width = 100; this.height = 100; queueMicrotask(() => this.onload && this.onload()); }
}

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  globalThis.Image = FakeImage;
  Object.assign(backend, {
    pc: [0.5019, 0.3301, 0.842], before: null, source: 'header', kind: null, convention: 'oxford', conventionKnown: true,
    shape: [128, 156], dataset: 'scan_a', calls: [], previews: [], unknown: [], hold: null, convertError: null,
    updateError: null, updateAnswer: {}, perPixel: false, patterns: [], statusPc: null, statusGate: null, updateGates: [], hadNoPc: false,
  });
  globalThis.fetch = vi.fn(() => Promise.resolve({
    json: () => Promise.resolve({ files: [{ path: '/sht/Ni (Ni) [cF4] {20kV}.sht', filename: 'Ni (Ni) [cF4] {20kV}.sht' }] }),
  }));
});
afterEach(() => {
  cleanup();
  useDataStore.setState({ filePath: null });
  expect(backend.unknown).toEqual([]);
});

// The page is ready for typing when the head is there and the select has taken the open file's kind of numbers.
const ready = async () => {
  await screen.findByTestId('pc-file-head');
  const want = backend.conventionKnown ? backend.convention : 'bruker';
  await waitFor(() => expect(screen.getByLabelText('I type my numbers in:').value).toBe(want));
};
const field = (label) => screen.getByLabelText(label);
const type = (label, value) => fireEvent.change(field(label), { target: { value } });
const typeOxford = (x = '0.5', y = '0.55', z = '0.71') => { type('PCx:', x); type('PCy:', y); type('DD:', z); };
const choose = (value) => fireEvent.change(screen.getByLabelText('I type my numbers in:'), { target: { value } });
const useButton = () => screen.getByRole('button', { name: 'Use these numbers' });
const updates = () => backend.calls.filter((c) => c[0] === 'updatePC');
const converts = () => backend.calls.filter((c) => c[0] === 'convert');
const settle = (ms = 700) => new Promise((r) => setTimeout(r, ms));    // longer than the echo's debounce

describe('the head and the select', () => {
  it('the head names the kind of numbers it shows: Bruker', async () => {
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
  });

  it('is preselected from the convention of the open file, and the note says why', async () => {
    render(<PCRefinement />);
    await ready();
    await waitFor(() => expect(screen.getByLabelText('I type my numbers in:').value).toBe('oxford'));
    expect(screen.getByTestId('pc-conv-preselected').textContent).toBe('Preselected: the kind of numbers of a .h5oina file.');
    expect(field('PCx:')).toBeTruthy();                                         // Oxford's own symbols
    expect(screen.getByLabelText('DD:')).toBeTruthy();
  });

  it('is preselected EDAX/TSL for an EDAX file, with that file\'s note', async () => {
    backend.convention = 'tsl';
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByLabelText('I type my numbers in:').value).toBe('tsl'));
    expect(screen.getByTestId('pc-conv-preselected').textContent).toBe('Preselected: the kind of numbers of an EDAX/TSL file.');
    expect(screen.getByLabelText('y*:')).toBeTruthy();
  });

  it('stays on Bruker, with no such note, when the backend does not know the file\'s convention', async () => {
    backend.conventionKnown = false;                    // the convention field then only says "bruker" for lack of better
    backend.convention = 'bruker';
    render(<PCRefinement />);
    await ready();
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.getByLabelText('I type my numbers in:').value).toBe('bruker');
    expect(screen.queryByTestId('pc-conv-preselected')).toBeNull();
    expect(screen.getByLabelText('PC X:')).toBeTruthy();
  });

  it('does not take a kind of numbers away from a person who chose one before the backend had said the file\'s', async () => {
    let release;
    backend.statusGate = new Promise((resolve) => { release = resolve; });
    render(<PCRefinement />);
    await screen.findByText(/This is the size of scan_a\./);       // the page knows the file's name: no restart is due
    choose('tsl');
    await waitFor(() => expect(screen.getByLabelText('I type my numbers in:').value).toBe('tsl'));
    release();
    await screen.findByTestId('pc-file-head');
    await settle(100);
    expect(screen.getByLabelText('I type my numbers in:').value).toBe('tsl');           // not forced to Oxford
    expect(screen.queryByTestId('pc-conv-preselected')).toBeNull();
  });

  it('does not change the meaning of numbers already typed when the file\'s kind of numbers arrives late', async () => {
    let release;
    backend.statusGate = new Promise((resolve) => { release = resolve; });
    render(<PCRefinement />);
    await screen.findByText(/This is the size of scan_a\./);
    type('PC X:', '0.5'); type('PC Y:', '0.33'); type('PC Z:', '0.84');
    release();
    await screen.findByTestId('pc-file-head');
    await settle(100);
    expect(screen.getByLabelText('I type my numbers in:').value).toBe('bruker');         // still what they typed them as
    expect(field('PC Y:').value).toBe('0.33');
  });

  it('lists the three kinds of numbers by their program names', async () => {
    render(<PCRefinement />);
    await ready();
    const names = [...screen.getByLabelText('I type my numbers in:').querySelectorAll('option')].map((o) => o.textContent);
    expect(names).toEqual(['Bruker (Orienta)', 'EDAX/TSL (OIM)', 'Oxford (AZtec)']);
  });

  it('gives the pattern size as rows x columns of this file', async () => {
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-pattern-size').textContent).toBe('128 rows × 156 columns'));
    expect(screen.getByText('Rows are the pattern height, columns the pattern width. This is the size of scan_a.')).toBeTruthy();
  });
});

describe('the fields', () => {
  it('start empty; each says the PC in use in its own kind of numbers, with its hint underneath', async () => {
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-in-use-y').textContent).toBe('In use, as Oxford: 0.5497'));
    for (const label of ['PCx:', 'PCy:', 'DD:']) expect(field(label).value).toBe('');
    // by hand: (1 - 0.3301) / 1.21875 = 0.549662; 0.842 / 1.21875 = 0.690872
    expect(screen.getByTestId('pc-in-use-x').textContent).toBe('In use, as Oxford: 0.5019');
    expect(screen.getByTestId('pc-in-use-z').textContent).toBe('In use, as Oxford: 0.6909');
    expect(screen.getByTestId('pc-same-note').textContent).toBe('Same PC as above, in this convention.');
    expect(screen.getByTestId('pc-hint-y').textContent)
      .toBe('Vertical position, counted from the BOTTOM edge, in units of the pattern width.');
    expect(screen.getByTestId('pc-hint-z').textContent)
      .toBe('Distance between sample and detector (detector distance), in units of the pattern width.');
  });

  it('no hint claims what is typical: that depends on the maker', async () => {
    render(<PCRefinement />);
    await ready();
    for (const c of ['oxford', 'tsl', 'bruker']) {
      choose(c);
      await waitFor(() => expect(screen.getByLabelText('I type my numbers in:').value).toBe(c));
      for (const axis of ['x', 'y', 'z']) expect(screen.getByTestId(`pc-hint-${axis}`).textContent).not.toMatch(/typical/i);
    }
  });

  it('in Bruker\'s kind of numbers the PC in use stands there as it is, without asking the backend', async () => {
    backend.conventionKnown = false;
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-in-use-y').textContent).toBe('In use, as Bruker: 0.3301'));
    expect(converts()).toEqual([]);
  });

  it('say that no PC is in use for a file without one', async () => {
    backend.source = 'missing';
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-in-use-x').textContent).toBe('No PC in use'));
    expect(screen.queryByTestId('pc-same-note')).toBeNull();
  });

  it('show no badge until something is typed, then "not used yet"', async () => {
    render(<PCRefinement />);
    await ready();
    expect(screen.queryByTestId('pc-typed-badge')).toBeNull();
    type('PCx:', '0.5');
    expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet');
    type('PCx:', '');
    expect(screen.queryByTestId('pc-typed-badge')).toBeNull();
  });

  it('"Use these numbers" is active only when all three fields are numbers', async () => {
    render(<PCRefinement />);
    await ready();
    expect(useButton().disabled).toBe(true);
    expect(screen.getByTestId('pc-fill-hint').textContent).toBe('Fill in all three fields to use these numbers.');
    type('PCx:', '0.5'); type('PCy:', '0.55');
    expect(useButton().disabled).toBe(true);
    type('DD:', '0.71');
    expect(useButton().disabled).toBe(false);
    expect(screen.queryByTestId('pc-fill-hint')).toBeNull();
  });

  it('a field that is not a number is named, with what to type, and the button stays off', async () => {
    render(<PCRefinement />);
    await ready();
    type('PCx:', '0.5'); type('PCy:', 'abc'); type('DD:', '0.71');
    expect(screen.getByTestId('pc-field-error').textContent)
      .toBe('"abc" in PCy is not a number. Type digits with a dot, for example 0.5026.');
    expect(useButton().disabled).toBe(true);
    expect(screen.queryByTestId('pc-fill-hint')).toBeNull();                    // the error says it already
    type('PCy:', '0.5.5');                                                      // two dots are no number either
    expect(screen.getByTestId('pc-field-error').textContent).toMatch(/^"0\.5\.5" in PCy is not a number/);
    type('PCy:', '0,55');                                                       // a comma is a decimal mark
    expect(screen.queryByTestId('pc-field-error')).toBeNull();
    expect(useButton().disabled).toBe(false);
  });
});

describe('the live echo', () => {
  it('shows the Orienta numbers of what is typed, from the backend\'s conversion with the pattern size', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    const echo = await screen.findByTestId('pc-echo');
    // by hand: y = 1 - 0.55 * 1.21875 = 0.3296875; z = 0.71 * 1.21875 = 0.8653125
    expect(echo.textContent).toBe(
      'You typed Oxford numbers 0.5, 0.55, 0.71 for 128 rows × 156 columns. Orienta (Bruker) numbers: 0.5000, 0.3297, 0.8653.');
    expect(echo.querySelector('b').textContent).toBe('0.5000, 0.3297, 0.8653');
    expect(converts().at(-1)).toEqual(['convert', [0.5, 0.55, 0.71], 'oxford', 'bruker', [128, 156]]);
  });

  it('in Bruker\'s kind of numbers says these are the numbers Orienta uses, and asks nothing', async () => {
    backend.conventionKnown = false;
    render(<PCRefinement />);
    await ready();
    type('PC X:', '0.51'); type('PC Y:', '0.33'); type('PC Z:', '0.84');
    expect(screen.getByTestId('pc-echo').textContent)
      .toBe('You typed Bruker numbers 0.51, 0.33, 0.84. These are the numbers Orienta uses.');
    await settle();
    expect(converts()).toEqual([]);
  });

  it('is never stale: other digits take the old echo away at once, and a late answer for old digits is dropped', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    // the next digits: the echo of the old ones must not stand under them while the new answer is on its way
    let release;
    backend.hold = new Promise((resolve) => { release = resolve; });
    type('PCy:', '0.56');
    expect(screen.queryByTestId('pc-echo')).toBeNull();
    await waitFor(() => expect(converts().at(-1)[1]).toEqual([0.5, 0.56, 0.71]));   // asked, not answered
    // and the person types on before it answers: the answer for 0.56 arrives under 0.55 again
    backend.hold = null;
    type('PCy:', '0.55');
    await screen.findByTestId('pc-echo');
    release();
    await settle(50);
    expect(screen.getByTestId('pc-echo').textContent).toMatch(/Oxford numbers 0\.5, 0\.55, 0\.71 .*0\.5000, 0\.3297, 0\.8653\.$/);
  });

  it('waits for a pause in typing before it asks', async () => {
    render(<PCRefinement />);
    await ready();
    // The pause is 300 ms of the page's clock. With the machine's clock this test failed under load: a 100 ms wait
    // that came back after 300 ms found the question already asked. The timers are the test's from here on.
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      typeOxford('0.5', '0.5', '0.71');
      type('PCy:', '0.55');
      const asked = () => converts().filter((c) => c[2] === 'oxford' && c[3] === 'bruker');
      await act(async () => { vi.advanceTimersByTime(299); });     // inside the pause: nothing has been asked yet
      expect(asked()).toEqual([]);
      await act(async () => { vi.advanceTimersByTime(1); });       // 300 ms after the last key
      expect(asked()).toHaveLength(1);                             // not one per key
      expect(screen.getByTestId('pc-echo')).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('typing alone applies nothing', () => {
  it('no PC reaches the backend, the head and the preview stay on the PC in use, however long it is left', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    const before = backend.previews.length;
    typeOxford();
    fireEvent.keyDown(field('DD:'), { key: 'ArrowUp' });
    fireEvent.blur(field('DD:'));
    await settle(1200);                                             // longer than the 500 ms the page used to wait
    expect(updates()).toEqual([]);
    expect(backend.previews.slice(before)).toEqual([]);             // the preview is not redrawn at the typed numbers
    expect(backend.previews.at(-1)).toEqual([0.5019, 0.3301, 0.842]);
    expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) (from the file)');
    expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet');
  }, 15000);

  it('the preview follows the PC when "Use these numbers" is pressed, and not before', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(backend.previews.at(-1)).toEqual([0.5, 0.3296875, 0.8653125]), { timeout: 4000 });
  }, 15000);

  it('the preview does not move while the answer is on its way, but when the backend has taken the PC', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    let release;
    backend.updateGates = [new Promise((resolve) => { release = resolve; })];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    const before = backend.previews.length;
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(updates()).toHaveLength(1));
    await settle(900);                                              // longer than the preview's own pause
    expect(backend.previews.slice(before)).toEqual([]);             // the typed numbers are not in use yet
    release();
    await waitFor(() => expect(backend.previews.at(-1)).toEqual([0.5, 0.3296875, 0.8653125]), { timeout: 4000 });
  }, 15000);

  it('a PC the backend refused does not move the preview either', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    backend.updateError = Object.assign(new Error('409'), { response: { status: 409, data: {
      detail: { code: 'pc_job_running', message: 'A PC optimization is running.' } } } });
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    const before = backend.previews.length;
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-use-error');
    await settle(800);
    expect(backend.previews.slice(before)).toEqual([]);
  }, 15000);
});

describe('Use these numbers', () => {
  it('applies the numbers as the Bruker PC and says so in the words every way of giving a PC uses', async () => {
    const onNavigate = vi.fn();
    render(<PCRefinement onNavigate={onNavigate} />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    const grant = await screen.findByTestId('pc-grant');
    expect(updates()).toHaveLength(1);
    expect(updates()[0].slice(1, 4)).toEqual([0.5, 0.3296875, 0.8653125]);       // by hand, see the head of this file
    expect(screen.getByTestId('pc-grant-text').textContent).toBe(
      'PC in use for scan_a: 0.5000, 0.3297, 0.8653. Kept until you close Orienta. The file on disk is not changed.');
    expect(screen.getByTestId('pc-typed-grant').textContent).toMatch(
      /^You typed Oxford numbers 0\.5, 0\.55, 0\.71\. Taken as 0\.5000, 0\.3297, 0\.8653 \(Bruker\)\./);
    expect(grant.textContent).toMatch(/Previous PC: 0\.5019, 0\.3301, 0\.8420/);
    expect(screen.getByRole('button', { name: 'Show the bands over a pattern' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Go to indexing' }));
    expect(onNavigate).toHaveBeenCalledWith('indexing');
    await waitFor(() => expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: in use'));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5000, 0.3297, 0.8653 (Bruker) (typed)'));
    // the field beside each box now speaks of the new PC: (1 - 0.3296875) / 1.21875 = 0.55
    await waitFor(() => expect(screen.getByTestId('pc-in-use-y').textContent).toBe('In use, as Oxford: 0.5500'));
  }, 15000);

  it('a Bruker file\'s numbers need no conversion at all', async () => {
    backend.conventionKnown = false;
    render(<PCRefinement />);
    await ready();
    type('PC X:', '0.51'); type('PC Y:', '0.33'); type('PC Z:', '0.84');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    expect(updates()[0].slice(1, 4)).toEqual([0.51, 0.33, 0.84]);
    expect(screen.getByTestId('pc-typed-grant').textContent).toMatch(/^You typed Bruker numbers 0\.51, 0\.33, 0\.84\./);
    expect(converts()).toEqual([]);
  }, 15000);

  it('Enter in a field presses the button, when it is active', async () => {
    render(<PCRefinement />);
    await ready();
    type('PCx:', '0.5'); type('PCy:', '0.55');
    fireEvent.keyDown(field('PCy:'), { key: 'Enter' });
    await settle(100);
    expect(updates()).toEqual([]);                                  // two numbers are not enough
    type('DD:', '0.71');
    fireEvent.keyDown(field('DD:'), { key: 'Enter' });
    await screen.findByTestId('pc-grant');
    expect(updates()).toHaveLength(1);
  }, 15000);

  it('"Put it back" asks first, then gives the previous PC back, and the badge says "not used yet" again', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    expect(screen.getByText('Put back the previous PC (0.5019, 0.3301, 0.8420) for scan_a?')).toBeTruthy();
    expect(backend.calls.some((c) => c[0] === 'putBack')).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    await screen.findByTestId('pc-undo-done');
    expect(backend.calls.filter((c) => c[0] === 'putBack')).toEqual([['putBack', 'scan_a']]);
    expect(screen.getByTestId('pc-undo-done').textContent).toBe(
      'Previous PC put back for scan_a: 0.5019, 0.3301, 0.8420. Kept until you close Orienta. The file on disk is not changed.');
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    await waitFor(() => expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet'));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
  }, 15000);

  it('asks for the open dataset itself, not for a name made up for the line, when the page has no name for it', async () => {
    backend.dataset = null;
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    await screen.findByTestId('pc-undo-done');
    expect(backend.calls.filter((c) => c[0] === 'putBack')).toEqual([['putBack', null]]);
  }, 15000);

  it('editing a number takes the grant line away: it is about the numbers that were used', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    type('PCy:', '0.56');
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet');
    expect(updates()).toHaveLength(1);                              // and nothing was sent for 0.56
  }, 15000);

  it('a PC the backend refuses is said in words, keeps the typed numbers and shows no grant', async () => {
    backend.updateError = Object.assign(new Error('409'), { response: { status: 409, data: {
      detail: { code: 'pc_job_running', message: 'A PC optimization is running.' } } } });
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    const alert = await screen.findByTestId('pc-use-error');
    expect(alert.textContent).toMatch(/^The PC you typed was not applied: .*The PC in use is unchanged\.$/);
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(field('PCy:').value).toBe('0.55');
    expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet');
    expect(screen.getByTestId('pc-file-head').textContent).toMatch(/0\.5019, 0\.3301, 0\.8420 \(Bruker\) \(from the file\)$/);
  }, 15000);

  it('numbers edited while a press has no answer yet get no grant line from that answer', async () => {
    let release;
    backend.updateGates = [new Promise((resolve) => { release = resolve; })];
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(updates()).toHaveLength(1));
    type('PCy:', '0.56');
    release();
    await settle(300);
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet');
  }, 15000);

  it('of two presses only the newest puts its PC on the page, whichever answer comes last', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    let releaseA; let releaseB;
    backend.updateGates = [new Promise((r) => { releaseA = r; }), new Promise((r) => { releaseB = r; })];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await ready();
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    typeOxford();                                                    // press A: y 0.55
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(updates()).toHaveLength(1));
    type('PCy:', '0.56');                                            // press B: y 0.56, while A has no answer yet
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(updates()).toHaveLength(2));
    releaseB();
    await waitFor(() => expect(backend.previews.at(-1)).toEqual([0.5, 0.3175, 0.8653125]), { timeout: 4000 });
    releaseA();                                                      // the older answer arrives last
    await settle(800);
    expect(backend.previews.at(-1)).toEqual([0.5, 0.3175, 0.8653125]);
    expect(backend.previews).not.toContainEqual([0.5, 0.3296875, 0.8653125]);   // A never reached the preview
    // ...nor Apply Detector Settings, which sends the PC in use
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'setDetector')).toHaveLength(1));
    expect(backend.calls.find((c) => c[0] === 'setDetector')[2]).toEqual([0.5, 0.3175, 0.8653125]);
  }, 15000);

  it('the same numbers are not sent twice while the first press has no answer yet', async () => {
    let release;
    backend.updateGates = [new Promise((resolve) => { release = resolve; })];
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(updates()).toHaveLength(1));
    expect(useButton().disabled).toBe(true);
    fireEvent.click(useButton());
    release();
    await screen.findByTestId('pc-grant');
    expect(updates()).toHaveLength(1);
    expect(useButton().disabled).toBe(false);
  }, 15000);

  it('the grant line goes when the PC in use is no longer the one that was used (another way changed it)', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    backend.pc = [0.5026, 0.3272, 0.8458];                          // e.g. a Find PC from another place
    backend.source = 'refined';
    backend.kind = 'find';
    window.dispatchEvent(new CustomEvent('pc-patterns-updated'));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from Find PC)'));
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    await waitFor(() => expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet'));
  }, 15000);

  it('Apply Detector Settings sends the PC in use, not numbers that are only typed', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'setDetector')).toHaveLength(1));
    expect(backend.calls.find((c) => c[0] === 'setDetector')[2]).toEqual([0.5019, 0.3301, 0.842]);
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'setDetector')).toHaveLength(2));
    expect(backend.calls.filter((c) => c[0] === 'setDetector')[1][2]).toEqual([0.5, 0.3296875, 0.8653125]);
  }, 15000);

  it('another error is said too, where it used to stay quiet', async () => {
    backend.updateError = Object.assign(new Error('400'), { response: { status: 400, data: { detail: 'No detector available' } } });
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    expect((await screen.findByTestId('pc-use-error')).textContent).toBe('No detector available');
  }, 15000);
});

describe('the select converts what the fields hold', () => {
  it('Bruker to Oxford: the same PC in Oxford\'s numbers, with the labels of Oxford', async () => {
    backend.conventionKnown = false;
    render(<PCRefinement />);
    await ready();
    type('PC X:', '0.5019'); type('PC Y:', '0.3301'); type('PC Z:', '0.842');
    choose('oxford');
    await waitFor(() => expect(field('PCy:').value).toBe('0.5497'));
    expect(field('PCx:').value).toBe('0.5019');
    expect(field('DD:').value).toBe('0.6909');
    expect(converts().at(-1)).toEqual(['convert', [0.5019, 0.3301, 0.842], 'bruker', 'oxford', [128, 156]]);
    expect(screen.getByTestId('pc-switched').textContent).toBe('Same PC, shown in other numbers.');
    expect(screen.queryByTestId('pc-conv-preselected')).toBeNull();                // the person chose it
  });

  it('Oxford to EDAX/TSL goes from what the fields show, and typing again takes the message away', async () => {
    render(<PCRefinement />);
    await ready();
    type('PCx:', '0.5019'); type('PCy:', '0.5497'); type('DD:', '0.6909');
    choose('tsl');
    await waitFor(() => expect(field('y*:').value).toBe('0.6699'));
    expect(field('x*:').value).toBe('0.5019');
    expect(field('z*:').value).toBe('0.8420');
    expect(screen.getByTestId('pc-switched')).toBeTruthy();
    type('y*:', '0.67');
    expect(screen.queryByTestId('pc-switched')).toBeNull();
  });

  it('a field that is not a number: nothing is converted, and it is said', async () => {
    render(<PCRefinement />);
    await ready();
    type('PCx:', '0.5'); type('PCy:', 'abc');
    choose('tsl');
    expect(screen.getByTestId('pc-not-converted').textContent).toBe('A field is empty or not a number, so nothing was converted.');
    expect(field('y*:').value).toBe('abc');
    expect(field('x*:').value).toBe('0.5');
    // the PC in use is shown in the new numbers (that is asked), but what was typed is not converted
    expect(converts().filter((c) => c[2] === 'oxford')).toEqual([]);
  });

  it('empty fields: the labels change and nothing is said', async () => {
    render(<PCRefinement />);
    await ready();
    choose('tsl');
    expect(screen.getByLabelText('x*:').value).toBe('');
    expect(screen.queryByTestId('pc-not-converted')).toBeNull();
    expect(screen.queryByTestId('pc-switched')).toBeNull();
  });

  it('a conversion that fails says so and leaves the select and the numbers as they were', async () => {
    render(<PCRefinement />);
    await ready();
    type('PCx:', '0.5019'); type('PCy:', '0.5497'); type('DD:', '0.6909');
    backend.convertError = Object.assign(new Error('400'), { response: { status: 400, data: { detail: 'unknown convention' },
      headers: { 'x-orienta-code': 'unknownConvention' } } });
    choose('tsl');
    const alert = await screen.findByTestId('pc-conv-error');
    expect(alert.textContent).toMatch(/^The numbers could not be converted: /);
    expect(screen.getByLabelText('I type my numbers in:').value).toBe('oxford');
    expect(field('PCy:').value).toBe('0.5497');
  });
});

describe('another file starts the form again', () => {
  it('empties the fields and the lines under them', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    useDataStore.setState({ filePath: '/data/other.h5oina' });
    await waitFor(() => expect(field('PCy:').value).toBe(''));
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    await waitFor(() => expect(screen.queryByTestId('pc-typed-badge')).toBeNull());
    useDataStore.setState({ filePath: null });
  }, 15000);
});

describe('a per-pixel map', () => {
  it('says before anything is used what numbers used here do to it, and after "Use these numbers" that it was moved', async () => {
    backend.perPixel = true;
    backend.updateAnswer = { pc_map_action: 'shifted' };
    render(<PCRefinement />);
    const note = await screen.findByTestId('pc-map-note');
    expect(note.textContent).toMatch(/If you use numbers you typed, the whole map is moved so that its mean is that PC; the map keeps its shape\./);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    expect(updates()).toEqual([]);
    fireEvent.click(useButton());
    await waitFor(() => expect(screen.getByTestId('pc-map-note').textContent)
      .toBe('The PC map was moved so that its mean is the PC you entered. Its shape is unchanged.'));
  }, 15000);
});

describe('a press whose answer comes after the page moved on', () => {
  // The conversion of a press is held back by `backend.hold`; the echo has already been answered.
  const pressAndHold = async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    let release;
    backend.hold = new Promise((resolve) => { release = resolve; });
    const before = converts().length;
    fireEvent.click(useButton());
    await waitFor(() => expect(converts().length).toBe(before + 1));
    return () => { backend.hold = null; release(); };
  };

  it('another file opened while the numbers are being converted: nothing goes to the new file', async () => {
    const release = await pressAndHold();
    useDataStore.setState({ filePath: '/data/other.h5oina' });
    await waitFor(() => expect(field('PCy:').value).toBe(''));
    release();
    await settle(300);
    expect(updates()).toEqual([]);
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(useButton().disabled).toBe(true);                                 // nothing typed in the new file
  }, 15000);

  it('a number edited while the numbers are being converted: the old numbers are not sent', async () => {
    const release = await pressAndHold();
    type('PCy:', '0.56');
    release();
    await settle(300);
    expect(updates()).toEqual([]);
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(field('PCy:').value).toBe('0.56');
  }, 15000);

  it('the select changed while the numbers are being converted: the PC that was typed is applied once, as typed', async () => {
    const release = await pressAndHold();
    choose('tsl');
    await waitFor(() => expect(converts().at(-1)).toEqual(['convert', [0.5, 0.55, 0.71], 'oxford', 'tsl', [128, 156]]));
    release();
    await screen.findByTestId('pc-grant');
    expect(updates()).toHaveLength(1);
    expect(updates()[0].slice(1, 4)).toEqual([0.5, 0.3296875, 0.8653125]);
    expect(screen.getByLabelText('I type my numbers in:').value).toBe('tsl');
    await waitFor(() => expect(field('y*:').value).toBe('0.6703'));              // 0.55 * 1.21875, by hand
    expect(screen.getByTestId('pc-typed-grant').textContent).toMatch(/^You typed Oxford numbers 0\.5, 0\.55, 0\.71\./);
  }, 15000);

  it('another file opened while the backend takes the PC: the old file\'s PC reaches neither the preview nor Apply', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    let release;
    backend.updateGates = [new Promise((resolve) => { release = resolve; })];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await ready();
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await waitFor(() => expect(updates()).toHaveLength(1));
    // the person opens another file, whose PC is 0.6, 0.4, 0.7, before the answer comes
    backend.pc = [0.6, 0.4, 0.7];
    backend.dataset = 'scan_b';
    useDataStore.setState({ filePath: '/data/other.h5oina' });
    await waitFor(() => expect(field('PCy:').value).toBe(''));
    await settle(300);
    const before = backend.previews.length;
    release();
    await settle(900);                                                       // longer than the preview's own pause
    expect(backend.previews.slice(before)).not.toContainEqual([0.5, 0.3296875, 0.8653125]);
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'setDetector')).toHaveLength(1));
    expect(backend.calls.find((c) => c[0] === 'setDetector')[2]).toEqual([0.6, 0.4, 0.7]);
  }, 15000);
});

describe('patterns that are not 128 x 156', () => {
  it('a tall pattern (156 rows x 128 columns): Oxford numbers become Bruker numbers with that shape', async () => {
    backend.shape = [156, 128];
    render(<PCRefinement />);
    await ready();
    expect(screen.getByTestId('pc-pattern-size').textContent).toBe('156 rows × 128 columns');
    typeOxford();
    const echo = await screen.findByTestId('pc-echo');
    // by hand: y = 1 - 0.55 * 0.820513 = 0.548718; z = 0.71 * 0.820513 = 0.582564
    expect(echo.textContent).toBe(
      'You typed Oxford numbers 0.5, 0.55, 0.71 for 156 rows × 128 columns. Orienta (Bruker) numbers: 0.5000, 0.5487, 0.5826.');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    expect(updates()[0].slice(1, 4)).toEqual([0.5, 0.548718, 0.582564]);
    expect(converts().filter((c) => c[3] === 'bruker').at(-1)).toEqual(['convert', [0.5, 0.55, 0.71], 'oxford', 'bruker', [156, 128]]);
  }, 15000);

  it('a square pattern (60 x 60): the scale is 1, so y = 1 - 0.55 and z stays', async () => {
    backend.shape = [60, 60];
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    expect(updates()[0].slice(1, 4)).toEqual([0.5, 0.45, 0.71]);
    expect(converts().filter((c) => c[3] === 'bruker').at(-1)).toEqual(['convert', [0.5, 0.55, 0.71], 'oxford', 'bruker', [60, 60]]);
  }, 15000);

  it('EDAX/TSL numbers on a tall pattern: y* is 1 - y, z* is scaled to the shape', async () => {
    backend.shape = [156, 128];
    backend.convention = 'tsl';
    render(<PCRefinement />);
    await ready();
    type('x*:', '0.5'); type('y*:', '0.67'); type('z*:', '0.84');
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    expect(updates()[0].slice(1, 4)).toEqual([0.5, 0.33, 0.689231]);        // 0.84 * 0.820513, by hand
  }, 15000);
});

describe('Apply Detector Settings while typed numbers wait for "Use these numbers"', () => {
  const NOTE = 'Your typed numbers were not used. Press Use these numbers to use them.';
  const apply = () => fireEvent.click(screen.getByText('Apply Detector Settings'));

  it('says so under "Detector set", and sends the PC in use', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    apply();
    await screen.findByText('Detector set');
    expect(screen.getByTestId('pc-typed-not-used').textContent).toBe(NOTE);
    expect(backend.calls.find((c) => c[0] === 'setDetector')[2]).toEqual([0.5019, 0.3301, 0.842]);
  }, 15000);

  it('says nothing when no number was typed', async () => {
    render(<PCRefinement />);
    await ready();
    apply();
    await screen.findByText('Detector set');
    expect(screen.queryByTestId('pc-typed-not-used')).toBeNull();
  }, 15000);

  it('says nothing when the typed numbers were used first', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    apply();
    await screen.findByText('Detector set');
    expect(screen.queryByTestId('pc-typed-not-used')).toBeNull();
  }, 15000);

  it('goes when the numbers are used afterwards, and when they are cleared', async () => {
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    apply();
    await screen.findByTestId('pc-typed-not-used');
    fireEvent.click(useButton());
    await screen.findByTestId('pc-grant');
    await waitFor(() => expect(screen.queryByTestId('pc-typed-not-used')).toBeNull());
    expect(screen.getByText('Detector set')).toBeTruthy();
    // typed again and left unused: a new note needs a new press of Apply
    type('PCy:', '0.56');
    await waitFor(() => expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet'));
    expect(screen.queryByTestId('pc-typed-not-used')).toBeNull();
    apply();
    await screen.findByTestId('pc-typed-not-used');
    type('PCx:', ''); type('PCy:', ''); type('DD:', '');
    await waitFor(() => expect(screen.queryByTestId('pc-typed-not-used')).toBeNull());
  }, 15000);
});

describe('a typed PC on a file that had none', () => {
  it('"Remove this PC" takes it away again, after asking', async () => {
    backend.pc = [0.5, 0.5, 0.5];
    backend.source = 'missing';
    backend.hadNoPc = true;
    backend.updateAnswer = { previous: { pc: null, had_pc: false } };
    render(<PCRefinement />);
    await ready();
    typeOxford();
    await screen.findByTestId('pc-echo');
    fireEvent.click(useButton());
    const grant = await screen.findByTestId('pc-grant');
    expect(grant.textContent).toMatch(/This file had no PC before\./);
    expect(screen.queryByRole('button', { name: 'Put it back' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Remove this PC' }));
    expect(screen.getByText('Remove the PC of scan_a? It will have no PC again.')).toBeTruthy();
    expect(backend.calls.some((c) => c[0] === 'putBack')).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Yes, remove this PC' }));
    await screen.findByTestId('pc-undo-done');
    expect(backend.calls.filter((c) => c[0] === 'putBack')).toEqual([['putBack', 'scan_a']]);
    expect(screen.getByTestId('pc-undo-done').textContent).toBe(
      'PC removed. scan_a has no PC again. Indexing is refused until there is a PC. The file on disk is not changed.');
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    await waitFor(() => expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet'));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent).toBe('Open file: scan_a · PC in use: none'));
  }, 15000);
});
