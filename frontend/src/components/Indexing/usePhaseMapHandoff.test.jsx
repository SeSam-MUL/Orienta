// @vitest-environment jsdom
//
// The hand-off from the EDS page to the Indexing page, for every method.
//
// It used to force Hough ("only Hough has CIFs in its list"), and after that
// the per-phase routing only ever worked for Hough: pick Spherical or
// Dictionary afterwards and the list was empty with routing still on. These
// tests drive the real hook with the real file-matching and a mocked backend.
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { renderHook, cleanup, waitFor, act } from '@testing-library/react';

const mockPhaseMapIndexingConfig = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: { phaseMapIndexingConfig: (...a) => mockPhaseMapIndexingConfig(...a) },
  ebsdApi: {},
  isFileLoadInFlight: () => false,
}));

import useDataStore from '../../stores/useDataStore';
import usePhaseMapHandoff from './usePhaseMapHandoff';

const CFG = {
  loaded: true,
  cif_filenames: ['Al.cif', 'Si.cif'],
  n_classified: 600,
  n_total: 1000,
  percentage: 60,
  n_rows: 10,
  n_cols: 100,
  combined_mask: [],
};

const sht = (name, linked) => ({ path: `/db/${name}`, filename: name, linked_cif: linked });
const dict = (name, linked, extra = {}) => ({
  path: `/db/${name}`, filename: name, linked_cif: linked, file_type: 'dictionary',
  detector_shape: [118, 118], ...extra,
});
const master = (name, linked) => ({
  path: `/db/${name}`, filename: name, linked_cif: linked, file_type: 'master',
});
const cif = (name) => ({
  path: `/db/${name}`, filename: name, linked_cif: name, file_type: 'phase',
});

const SHTS = [sht('Si.sht', 'Si.cif'), sht('Al.sht', 'Al.cif'), sht('Fe.sht', 'Fe.cif')];
const DICTS = [dict('Al_dict.h5', 'Al.cif'), dict('Si_dict.h5', 'Si.cif')];
const CIFS = [cif('Al.cif'), cif('Si.cif'), cif('Fe.cif')];
const CTX = { currentPc: [0.5, 0.5, 0.5], detectorShape: [118, 118], geom: null };

let spies;

/** The log lines, with the translation keys as text. */
const logged = () => spies.log.mock.calls.map((c) => c[0]).join('\n');

function setup(initial) {
  spies = {
    onSelect: vi.fn(),
    onInstall: vi.fn(),
    log: vi.fn(),
    hasSelection: vi.fn(() => false),
    getDictContext: vi.fn(() => CTX),
  };
  const t = (key, params) => `${key} ${JSON.stringify(params || {})}`;
  const render = (props) => renderHook((p) => usePhaseMapHandoff({
    isActive: true, filePath: '/scan.h5oina', routingOn: false, ...spies, t, ...p,
  }), { initialProps: props });
  return render(initial);
}

beforeEach(() => {
  mockPhaseMapIndexingConfig.mockReset();
  mockPhaseMapIndexingConfig.mockResolvedValue({ data: CFG });
  useDataStore.setState({ pendingPhaseMapIndexing: true });
});
afterEach(() => {
  cleanup();
});

describe('the first hand-off', () => {
  it('Spherical: selects the SHT linked to each phase-map CIF, in the map order', async () => {
    setup({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: SHTS });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalledTimes(1));
    const selected = spies.onSelect.mock.calls[0][0];
    expect(selected.map((f) => f.filename)).toEqual(['Al.sht', 'Si.sht']);
    expect(spies.onInstall.mock.calls[0][0]).toBe(CFG);
    expect(spies.onInstall.mock.calls[0][1]).toEqual(selected);
    expect(useDataStore.getState().pendingPhaseMapIndexing).toBe(false);
    expect(logged()).toContain('messages.handoffApplied');
  });

  it('Dictionary: selects the fitting dictionary of each phase and ships exactly those', async () => {
    setup({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    const [selected, extra] = spies.onSelect.mock.calls[0];
    expect(selected.map((f) => f.filename)).toEqual(['Al_dict.h5', 'Si_dict.h5']);
    expect(extra.dictPaths).toEqual({
      '/db/Al_dict.h5': '/db/Al_dict.h5', '/db/Si_dict.h5': '/db/Si_dict.h5',
    });
  });

  it('Dictionary: a phase with only a master pattern is told to need a dictionary, and is not selected', async () => {
    setup({ method: 'dictionary', discoveredFor: 'dictionary',
      discoveredFiles: [dict('Al_dict.h5', 'Al.cif'), master('Si_master.h5', 'Si.cif')] });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename)).toEqual(['Al_dict.h5']);
    expect(logged()).toContain('messages.handoffNeedsDictionary {"count":1,"list":"Si.cif"}');
  });

  it('Dictionary: fits the dictionary to what the page says the detector is', async () => {
    const wrong = dict('Al_128x156.h5', 'Al.cif', { detector_shape: [128, 156] });
    setup({ method: 'dictionary', discoveredFor: 'dictionary',
      discoveredFiles: [wrong, dict('Si_dict.h5', 'Si.cif')] });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    expect(spies.getDictContext).toHaveBeenCalled();
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename)).toEqual(['Si_dict.h5']);
    expect(logged()).toContain('messages.handoffNeedsDictionary {"count":1,"list":"Al.cif"}');
  });

  describe('a Dictionary waits for the detector context of the loaded file', () => {
    // a dictionary for another detector: it passes a context that knows no shape
    const wrong = dict('Al_128x156.h5', 'Al.cif', { detector_shape: [128, 156] });
    const FILES = [wrong, dict('Si_dict.h5', 'Si.cif')];
    const LAX = { currentPc: null, detectorShape: null, geom: null };   // before it arrived

    it('does nothing until the context is there, then chooses with the REAL context', async () => {
      let context = LAX;                     // what the page holds until it has heard
      const { rerender } = setup({ method: 'dictionary', discoveredFor: 'dictionary',
        discoveredFiles: FILES, dictContextReady: false });
      spies.getDictContext.mockImplementation(() => context);
      await act(async () => { await Promise.resolve(); });
      expect(mockPhaseMapIndexingConfig).not.toHaveBeenCalled();
      expect(spies.onSelect).not.toHaveBeenCalled();
      expect(useDataStore.getState().pendingPhaseMapIndexing).toBe(true);   // still pending

      context = CTX;                          // the page heard: 118 x 118
      rerender({ method: 'dictionary', discoveredFor: 'dictionary',
        discoveredFiles: FILES, dictContextReady: true });
      await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
      // the 128x156 dictionary does not fit this detector; with the lax context it would have
      expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename)).toEqual(['Si_dict.h5']);
      expect(logged()).toContain('messages.handoffNeedsDictionary {"count":1,"list":"Al.cif"}');
    });

    it('never uses a context that is not ready, even when the list is there', async () => {
      setup({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
        dictContextReady: false });
      await act(async () => { await Promise.resolve(); });
      await act(async () => { await Promise.resolve(); });
      expect(spies.getDictContext).not.toHaveBeenCalled();
      expect(spies.onInstall).not.toHaveBeenCalled();
    });

    it('the redo after a method switch waits as well', async () => {
      const hook = setup({ method: 'spherical', discoveredFor: 'spherical',
        discoveredFiles: SHTS, dictContextReady: false });   // Spherical does not need it
      await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
      hook.rerender({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: SHTS,
        routingOn: true, dictContextReady: false });
      spies.onSelect.mockClear();
      hook.rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
        routingOn: true, dictContextReady: false });
      await act(async () => { await Promise.resolve(); });
      expect(spies.onSelect).not.toHaveBeenCalled();
      hook.rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
        routingOn: true, dictContextReady: true });
      await waitFor(() => expect(spies.onSelect).toHaveBeenCalledTimes(1));
    });

    it('Spherical and Hough do not wait for it', async () => {
      setup({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: SHTS,
        dictContextReady: false });
      await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    });
  });

  it('says which file it took when several fit', async () => {
    setup({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: [
      sht('Al 20kV.sht', 'Al.cif'), sht('Al 30kV.sht', 'Al.cif'), sht('Si.sht', 'Si.cif')] });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename))
      .toEqual(['Al 20kV.sht', 'Si.sht']);
    expect(logged()).toContain('messages.handoffSeveralFiles');
    expect(logged()).toContain('"taken":"Al 20kV.sht"');
    expect(logged()).toContain('"others":"Al 30kV.sht"');
  });

  it('Hough: unchanged -- the CIFs of the map, and a missing CIF is logged as before', async () => {
    mockPhaseMapIndexingConfig.mockResolvedValue({
      data: { ...CFG, cif_filenames: ['Al.cif', 'Ni.cif'] } });
    setup({ method: 'hough', discoveredFor: 'hough', discoveredFiles: CIFS });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename)).toEqual(['Al.cif']);
    expect(logged()).toContain('messages.handoffMissing {"count":1,"list":"Ni.cif"}');
    expect(logged()).not.toContain('handoffMissingForMethod');
  });

  it('names the phases that have no file for this method, in the user\'s language', async () => {
    setup({ method: 'spherical', discoveredFor: 'spherical',
      discoveredFiles: [sht('Al.sht', 'Al.cif')] });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    expect(logged()).toContain('messages.handoffMissingForMethod');
    expect(logged()).toContain('"list":"Si.cif"');
    // the method's display name and the file kind are locale keys, not raw ids
    expect(logged()).toContain('"kind":"messages.handoffKindSpherical {}"');
    // the short name ("Spherical"), not the long one with the backend in it
    expect(logged()).toContain('"method":"batchDialog.methodSpherical {}"');
  });

  it('an EMPTY list still hands over: every phase missing, mask and routing installed', async () => {
    setup({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: [] });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalledTimes(1));
    expect(spies.onSelect).not.toHaveBeenCalled();
    expect(spies.onInstall.mock.calls[0][1]).toEqual([]);
    expect(logged()).toContain('"list":"Al.cif, Si.cif"');
    expect(useDataStore.getState().pendingPhaseMapIndexing).toBe(false);
  });

  it('waits for the list of this method, whatever it holds', async () => {
    const { rerender } = setup({ method: 'spherical', discoveredFor: null, discoveredFiles: [] });
    await act(async () => { await Promise.resolve(); });
    expect(mockPhaseMapIndexingConfig).not.toHaveBeenCalled();
    expect(useDataStore.getState().pendingPhaseMapIndexing).toBe(true);
    rerender({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: [] });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
  });

  it('does not pick files out of the previous method\'s list', async () => {
    // method already switched to Spherical, the list on screen is still Hough's
    const { rerender } = setup({ method: 'spherical', discoveredFor: 'hough',
      discoveredFiles: CIFS });
    await act(async () => { await Promise.resolve(); });
    expect(mockPhaseMapIndexingConfig).not.toHaveBeenCalled();
    expect(spies.onSelect).not.toHaveBeenCalled();
    expect(useDataStore.getState().pendingPhaseMapIndexing).toBe(true);   // still pending

    rerender({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: SHTS });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename)).toEqual(['Al.sht', 'Si.sht']);
  });

  it('reports a map that is not loaded on the backend and drops the request', async () => {
    mockPhaseMapIndexingConfig.mockResolvedValue({ data: { loaded: false } });
    setup({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: SHTS });
    await waitFor(() => expect(useDataStore.getState().pendingPhaseMapIndexing).toBe(false));
    expect(spies.onSelect).not.toHaveBeenCalled();
    expect(spies.onInstall).not.toHaveBeenCalled();
    expect(spies.log.mock.calls[0][0]).toContain('messages.handoffSkipped');
  });
});

describe('changing the method with routing on', () => {
  async function handedOffAs(method, files) {
    const hook = setup({ method, discoveredFor: method, discoveredFiles: files });
    await waitFor(() => expect(spies.onInstall).toHaveBeenCalled());
    // the page now has routing on
    hook.rerender({ method, discoveredFor: method, discoveredFiles: files, routingOn: true });
    spies.onSelect.mockClear();
    spies.log.mockClear();
    return hook;
  }

  it('re-selects the phases for the new method once its list has arrived', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);

    // method switched; the old list is still on screen -> nothing yet
    rerender({ method: 'dictionary', discoveredFor: 'spherical', discoveredFiles: SHTS,
      routingOn: true });
    expect(spies.onSelect).not.toHaveBeenCalled();
    // ... and not a word in the log either: every phase would read as "missing"
    expect(spies.log).not.toHaveBeenCalled();

    rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
      routingOn: true });
    await waitFor(() => expect(spies.onSelect).toHaveBeenCalledTimes(1));
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename))
      .toEqual(['Al_dict.h5', 'Si_dict.h5']);
    expect(mockPhaseMapIndexingConfig).toHaveBeenCalledTimes(1);   // from the stored payload
  });

  it('switching back to Hough selects the CIFs again', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);
    rerender({ method: 'hough', discoveredFor: 'spherical', discoveredFiles: SHTS, routingOn: true });
    rerender({ method: 'hough', discoveredFor: 'hough', discoveredFiles: CIFS, routingOn: true });
    await waitFor(() => expect(spies.onSelect).toHaveBeenCalledTimes(1));
    expect(spies.onSelect.mock.calls[0][0].map((f) => f.filename)).toEqual(['Al.cif', 'Si.cif']);
  });

  it('a new method whose list is empty reports every phase missing', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);
    rerender({ method: 'dictionary', discoveredFor: 'spherical', discoveredFiles: SHTS,
      routingOn: true });
    rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: [],
      routingOn: true });
    await waitFor(() => expect(logged()).toContain('messages.handoffMissingForMethod'));
    expect(spies.onSelect).not.toHaveBeenCalled();
  });

  it('selects once per switch, not on every later change of the list', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);
    rerender({ method: 'dictionary', discoveredFor: 'spherical', discoveredFiles: SHTS,
      routingOn: true });
    rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
      routingOn: true });
    await waitFor(() => expect(spies.onSelect).toHaveBeenCalledTimes(1));
    // the list is refreshed (a file added by path, a retry): the user's selection stays
    rerender({ method: 'dictionary', discoveredFor: 'dictionary',
      discoveredFiles: [...DICTS, dict('Fe_dict.h5', 'Fe.cif')], routingOn: true });
    await act(async () => { await Promise.resolve(); });
    expect(spies.onSelect).toHaveBeenCalledTimes(1);
  });

  it('leaves a choice that is already made for the new method alone', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);
    spies.hasSelection.mockReturnValue(true);       // e.g. sent over from the Phase Test
    rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
      routingOn: true });
    await act(async () => { await Promise.resolve(); });
    expect(spies.onSelect).not.toHaveBeenCalled();
  });

  it('does nothing when routing was switched off', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);
    rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
      routingOn: false });
    await act(async () => { await Promise.resolve(); });
    expect(spies.onSelect).not.toHaveBeenCalled();
  });

  it('forgets the hand-off when another file is loaded', async () => {
    const { rerender } = await handedOffAs('spherical', SHTS);
    rerender({ method: 'spherical', discoveredFor: 'spherical', discoveredFiles: SHTS,
      routingOn: true, filePath: '/other.h5oina' });
    rerender({ method: 'dictionary', discoveredFor: 'dictionary', discoveredFiles: DICTS,
      routingOn: true, filePath: '/other.h5oina' });
    await act(async () => { await Promise.resolve(); });
    expect(spies.onSelect).not.toHaveBeenCalled();
  });
});
