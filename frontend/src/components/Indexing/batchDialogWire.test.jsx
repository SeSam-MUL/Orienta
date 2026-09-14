// @vitest-environment jsdom
/**
 * What the "Batch…" dialog actually puts on the wire for a spherical batch.
 *
 * The dialog offers "Spherical" in its method menu and had no field for a
 * master: it sent `sht_paths: []` on every dataset, and the backend turned
 * the empty path into its own working directory as the master file without
 * raising. It also sent no `backend`, so a batch ran the WSL CPU path
 * however the Indexing page was set — the GPU branch of that route could not
 * be reached at all.
 *
 * Written like `batchRequestWire.test.jsx`: the api module is mocked at the
 * boundary and the REAL request body is read, because a component test that
 * mocks `indexApi.batchStart` can only prove what the UI asked for — which is
 * exactly the blind spot here.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  indexApi: {
    discoverFiles: vi.fn(() => Promise.resolve({ data: { files: [] } })),
    batchStart: vi.fn(() => Promise.resolve({ data: {} })),
    batchStatus: vi.fn(() => Promise.resolve({ data: { running: false } })),
    batchStop: vi.fn(() => Promise.resolve({ data: {} })),
  },
}));

import { indexApi } from '../../services/api';
import BatchIndexingDialog from './BatchIndexingDialog';

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

async function queueOneSphericalFile({ sht, backend, extra = {} }) {
  render(<BatchIndexingDialog open onClose={() => {}}
    sphericalParams={{ backend, ...extra }} />);

  // Global method -> spherical, then the master, then queue one file.
  const methodSelect = document.querySelector('select');
  await act(async () => {
    fireEvent.change(methodSelect, { target: { value: 'spherical' } });
  });
  const shtInput = screen.getByTestId('batch-sht-input');
  await act(async () => {
    fireEvent.change(shtInput, { target: { value: sht } });
  });
  const textarea = document.querySelector('textarea');
  await act(async () => {
    fireEvent.change(textarea, { target: { value: 'D:/scans/one.h5oina' } });
  });
  await act(async () => {
    fireEvent.click(screen.getByText(/^\+ Add$/));
  });
  await act(async () => {
    fireEvent.click(screen.getByText(/Start Batch/));
  });

  expect(indexApi.batchStart).toHaveBeenCalledTimes(1);
  return indexApi.batchStart.mock.calls[0][0][0];
}

describe('the batch dialog sends what a spherical run needs', () => {
  it('carries the master the user picked', async () => {
    const ds = await queueOneSphericalFile({
      sht: 'D:/db/Al (Al) [cF4] {20kV}.sht', backend: 'spherical_gpu',
    });
    expect(ds.method).toBe('spherical');
    expect(ds.sht_paths).toEqual(['D:/db/Al (Al) [cF4] {20kV}.sht']);
  });

  it('carries every spherical setting of the page, not the model defaults', async () => {
    // gausbckg is the one that bites: the request model defaults it to false
    // and IndexingConfig to true, so a batch that sends nothing runs with a
    // background subtraction the page does not show.
    const ds = await queueOneSphericalFile({
      sht: 'D:/db/Al.sht',
      backend: 'emsphinx',
      extra: { bandwidth: 68, nregions: 4, refine: false, gausbckg: true, circmask: 0 },
    });
    expect(ds.bandwidth).toBe(68);
    expect(ds.nregions).toBe(4);
    expect(ds.refine).toBe(false);
    expect(ds.gausbckg).toBe(true);
    expect(ds.circmask).toBe(0);
  });

  it('carries the engine the Indexing page is set to', async () => {
    const gpu = await queueOneSphericalFile({
      sht: 'D:/db/Al.sht', backend: 'spherical_gpu',
    });
    expect(gpu.backend).toBe('spherical_gpu');

    cleanup();
    vi.clearAllMocks();

    const cpu = await queueOneSphericalFile({
      sht: 'D:/db/Al.sht', backend: 'emsphinx',
    });
    expect(cpu.backend).toBe('emsphinx');
  });

  it('still sends an empty list when no master was given', async () => {
    // Not a silent default: the backend must be able to tell "the user gave
    // no master" from "the user gave this one".
    const ds = await queueOneSphericalFile({ sht: '', backend: 'emsphinx' });
    expect(ds.sht_paths).toEqual([]);
  });
});
