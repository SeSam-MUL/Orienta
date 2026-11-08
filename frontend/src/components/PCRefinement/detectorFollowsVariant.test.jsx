// @vitest-environment jsdom
//
// The detector card re-read the backend when the file path changed. Two pattern sources of one file
// have the same path and a PC each: after a switch the fields showed the new PC, the "Detector:" text
// the old one, and "Detector set" stayed green. The name of the loaded entry says the variant.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({ repr: 'EBSDDetector processed', reads: 0 }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({ has_detector: true, patterns: [] }),
    setDetector: () => ok({ repr: backend.repr }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const ebsdApi = new Proxy({
    getDetector: () => {
      backend.reads += 1;
      return ok({ has_detector: true, shape: [60, 80], pc: [0.5, 0.3, 0.8], sample_tilt: 70,
                  camera_tilt: 0, azimuthal: 0, binning: 1, pixel_size: null, repr: backend.repr });
    },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi, indexApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';

const PATH = 'D:/data/scan.h5oina';
const entry = (name) => ({ files: [{ path: PATH, name, active: true }] });
const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  backend.repr = 'EBSDDetector processed';
  backend.reads = 0;
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
  useDataStore.setState({ ebsdLoaded: true, filePath: PATH });
  useLoadedFilesStore.setState(entry('scan'));
});
afterEach(() => cleanup());

const shown = () => { const box = screen.getByTitle(/EBSDDetector configuration output/); return box.value ?? box.textContent; };

describe('another pattern source of the open file comes up', () => {
  it('the card reads the detector of the new source and forgets the message of the old one', async () => {
    render(<PCRefinement />);
    await waitFor(() => expect(shown()).toContain('EBSDDetector processed'));
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await screen.findByText('Detector set');
    backend.repr = 'EBSDDetector unprocessed';
    await act(async () => { useLoadedFilesStore.setState(entry('scan · unprocessed')); });
    await waitFor(() => expect(shown()).toContain('EBSDDetector unprocessed'));
    expect(screen.queryByText('Detector set')).toBeNull();
  });

  it('a re-read of the same list asks nothing', async () => {
    render(<PCRefinement />);
    await waitFor(() => expect(backend.reads).toBeGreaterThan(0));
    const reads = backend.reads;
    await act(async () => { useLoadedFilesStore.setState(entry('scan')); });
    expect(backend.reads).toBe(reads);
  });
});
