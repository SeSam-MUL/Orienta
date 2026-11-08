// @vitest-environment jsdom
//
// The viewer, mounted, with the backend mocked: how a `choices` answer of /inspect reaches the generic
// load dialog and how the dialog's choice reaches /load. The answer is the backend's own
// (__fixtures__/inspectChoices.json). i18n is the real one (English).
import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

vi.mock('../../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  const fake = (base) => new Proxy(base, {
    get: (target, prop) => {
      if (prop === 'then' || typeof prop === 'symbol') return undefined;
      if (!(prop in target)) target[prop] = vi.fn(async () => ({ data: {} }));
      return target[prop];
    },
  });
  return { ...actual, ebsdApi: fake({}), edsApi: fake({}), pcApi: fake({}), scanGeometryApi: fake({}) };
});

vi.mock('../../common/ImageExportDialog', () => ({ default: () => <div /> }));

import EBSDViewer from '../EBSDViewer';
import { ebsdApi } from '../../../services/api';
import useToastStore from '../../../stores/useToastStore';
import useLoadedFilesStore from '../../../stores/useLoadedFilesStore';
import choices from '../__fixtures__/inspectChoices.json';

const PATH = '/d/two.h5oina';
const KEY = 'orienta.loadVariant./d/two.h5oina';
const LOADED = { grid_shape: [2, 3], pattern_shape: [4, 5], pattern_count: 6, format_type: 'Oxford', has_patterns: true };

const hadResizeObserver = 'ResizeObserver' in globalThis;
beforeAll(() => {
  if (!hadResizeObserver) globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
afterAll(() => { if (!hadResizeObserver) delete globalThis.ResizeObserver; });

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  URL.createObjectURL = vi.fn(() => 'blob:x');
  URL.revokeObjectURL = vi.fn();
  useToastStore.setState({ toasts: [] });
  useLoadedFilesStore.setState({ files: [] });
  ebsdApi.inspect.mockReset();
  ebsdApi.inspect.mockResolvedValue({ data: choices });
  ebsdApi.inspectPreview.mockReset();
  ebsdApi.inspectPreview.mockResolvedValue({ data: new Blob(['x']) });
  ebsdApi.loadWithProgress.mockReset();
  ebsdApi.loadWithProgress.mockResolvedValue({ data: LOADED });
  ebsdApi.switchFile.mockReset();
  ebsdApi.getPattern.mockReset();
  ebsdApi.getPattern.mockResolvedValue({ data: {} });
  ebsdApi.loadedFiles.mockReset();
  ebsdApi.loadedFiles.mockResolvedValue({ data: {} });
});
afterEach(() => cleanup());

const pathBox = () => screen.getByPlaceholderText(/.+/, { selector: 'input[type="text"]' });
async function typePathAndEnter(value) {
  const box = pathBox();
  fireEvent.change(box, { target: { value } });
  await act(async () => { fireEvent.keyDown(box, { key: 'Enter' }); });
}
async function openDialog(path = PATH) {
  render(<EBSDViewer isActive onNavigate={vi.fn()} />);
  await typePathAndEnter(path);
  return screen.findByRole('dialog', { name: /Open two\.h5oina/ });
}
const open = () => act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Open' })); });
const radio = (name) => screen.getByRole('radio', { name });

describe('a file that offers a choice', () => {
  it('opens the dialog and loads nothing until a choice is made; the choice then makes exactly one load', async () => {
    await openDialog();
    expect(ebsdApi.inspect).toHaveBeenCalledTimes(1);
    expect(ebsdApi.inspect).toHaveBeenCalledWith(PATH);
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();

    fireEvent.click(radio('1: Unprocessed patterns'));
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
    await open();
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    const [path, options] = ebsdApi.loadWithProgress.mock.calls[0];
    expect(path).toBe(PATH);
    expect(options.variant).toEqual({ measurement: '1', pattern_source: 'unprocessed' });
    expect(options.apex).toBeNull();
    expect(screen.queryByRole('dialog', { name: /Open two\.h5oina/ })).toBeNull();
    expect(ebsdApi.inspect).toHaveBeenCalledTimes(1);
  });

  it('asks the backend for the small pattern of each source of this file', async () => {
    await openDialog();
    await waitFor(() => expect(ebsdApi.inspectPreview).toHaveBeenCalledTimes(3));
    expect(ebsdApi.inspectPreview.mock.calls.map((c) => c.slice(0, 3).join('|')).sort())
      .toEqual([`${PATH}|1|processed`, `${PATH}|1|unprocessed`, `${PATH}|2|processed`]);
  });

  it('hands an abort signal to each preview request, and closing the dialog fires it', async () => {
    ebsdApi.inspectPreview.mockImplementation(() => new Promise(() => {}));   // never answers
    await openDialog();
    await waitFor(() => expect(ebsdApi.inspectPreview).toHaveBeenCalledTimes(2));   // two at a time, none answers
    const signals = ebsdApi.inspectPreview.mock.calls.map((c) => c[3]);
    expect(signals.every((s) => s instanceof AbortSignal && !s.aborted)).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(signals.every((s) => s.aborted)).toBe(true);
  });

  it('cancelling loads nothing and writes nothing', async () => {
    await openDialog();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog', { name: /Open two\.h5oina/ })).toBeNull();
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
    expect(localStorage.getItem(KEY)).toBeNull();
  });
});

describe('a file with one variant', () => {
  it('makes exactly one inspect and one load, without a dialog and without a variant', async () => {
    ebsdApi.inspect.mockResolvedValue({ data: { kind: 'plain' } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(PATH);
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    expect(ebsdApi.inspect).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog', { name: /Open two\.h5oina/ })).toBeNull();
    const [path, options] = ebsdApi.loadWithProgress.mock.calls[0];
    expect(path).toBe(PATH);
    expect(options.variant).toBeNull();
    expect(options.apex).toBeNull();
    expect(Object.keys(localStorage).filter((k) => k.startsWith('orienta.loadVariant.'))).toEqual([]);
  });
});

describe('the last choice, remembered per file', () => {
  it('is written after a load that worked, under the canonical path', async () => {
    await openDialog();
    fireEvent.click(radio('1: Unprocessed patterns'));
    await open();
    await waitFor(() => expect(JSON.parse(localStorage.getItem(KEY) || 'null'))
      .toEqual({ measurement: '1', pattern_source: 'unprocessed' }));
  });

  it('is not written when the load fails', async () => {
    ebsdApi.loadWithProgress.mockRejectedValue(Object.assign(new Error('boom'), { response: { data: { detail: 'boom' } } }));
    await openDialog();
    fireEvent.click(radio('1: Unprocessed patterns'));
    await open();
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    expect(localStorage.getItem(KEY)).toBeNull();
  });

  it('preselects it when the file is opened again, still shows the dialog, and Enter loads it', async () => {
    localStorage.setItem(KEY, JSON.stringify({ measurement: '1', pattern_source: 'unprocessed' }));
    await openDialog();
    expect(radio('1: Unprocessed patterns').checked).toBe(true);
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
    await act(async () => { fireEvent.keyDown(screen.getByRole('dialog', { name: /Open two\.h5oina/ }), { key: 'Enter' }); });
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    expect(ebsdApi.loadWithProgress.mock.calls[0][1].variant).toEqual({ measurement: '1', pattern_source: 'unprocessed' });
  });

  it('still works when the storage is blocked', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
    try {
      render(<EBSDViewer isActive onNavigate={vi.fn()} />);
      await typePathAndEnter(PATH);
      await screen.findByRole('dialog', { name: /Open two\.h5oina/ });
      expect(radio('1: Processed patterns').checked).toBe(true);
      await open();
      await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    } finally {
      vi.restoreAllMocks();
    }
  });
});

describe('an .h5oina that cannot be read', () => {
  it('says so in words that fit every HDF5 format, and loads the ordinary way only on request', async () => {
    ebsdApi.inspect.mockResolvedValue({ data: { kind: 'unreadable', problem: 'apexFileUnreadable', problem_detail: 'x' } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(PATH);
    expect(await screen.findByText(/Orienta will not offer a choice of measurement or pattern source for this file/)).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/measurement dialog/);
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Open anyway' })); });
    expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1);
    expect(ebsdApi.loadWithProgress.mock.calls[0][1].variant).toBeNull();
  });
});

// A second file asked for while a dialog is open (the path box and a drop stay reachable behind the
// overlay): there is one dialog at a time, and it is the dialog of the file asked for last.
describe('a file asked for while a dialog is already open', () => {
  const OTHER = '/d/other.h5oina';
  const PLAIN = '/d/plain.h5oina';
  const APEX_H5 = '/d/series/session.h5';
  const apexMeasurement = {
    group: '/Area 1/OIM Map 1', id: '3f2a8e10-5c44-4b8e-9a51-7d02bb1ec91e', area: 'Area 1', name: 'OIM Map 1',
    hex: false, grid_type: 'SqrGrid', n_rows: 4, n_cols: 3, n_points: 12, step_x: 1, step_y: 1, sample_tilt: 70,
    elevation: 5, azimuth: 0, beam_kv: 20, has_sem: false, has_iq: false, phases: [], pc: [0.5, 0.5, 0.5],
    loadable: true, problem: null, measurement_problem: null, problem_detail: null, measured_at: null,
    measured_at_kind: null, consistent: true,
    pattern_files: [{ path: '/d/series/a.up1', name: 'a.up1', n_patterns: 12, pattern_shape: [8, 8], pairing: 'guid',
      count: 'fits', matches_grid: true, problem: null, names_measurement: '3f2a8e10-5c44-4b8e-9a51-7d02bb1ec91e',
      names_group: '/Area 1/OIM Map 1' }],
  };
  const APEX = { kind: 'apex', h5_path: APEX_H5, folder: '/d/series', folder_problem: null,
    measurements: [apexMeasurement], unmatched_pattern_files: [] };
  const ANSWERS = { [PATH]: choices, [OTHER]: choices, [PLAIN]: { kind: 'plain' }, [APEX_H5]: APEX };
  const apexDialog = () => screen.queryByRole('dialog', { name: /Open measurements from session\.h5/ });
  const choicesDialog = (file) => screen.queryByRole('dialog', { name: new RegExp(`^Open ${file.replace('.', '\.')}$`) });

  beforeEach(() => {
    ebsdApi.inspect.mockImplementation(async (p) => ({ data: structuredClone(ANSWERS[p]) }));   // a new object per answer, as on the wire
  });

  it('an APEX file asked for replaces the open dialog: one dialog, not two stacked', async () => {
    await openDialog();
    await typePathAndEnter(APEX_H5);
    await screen.findByRole('dialog', { name: /Open measurements from session\.h5/ });
    expect(choicesDialog('two.h5oina')).toBeNull();
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
  });

  it('a file with a choice asked for while the APEX dialog is open replaces it', async () => {
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(APEX_H5);
    await screen.findByRole('dialog', { name: /Open measurements from session\.h5/ });
    await typePathAndEnter(PATH);
    await screen.findByRole('dialog', { name: /Open two\.h5oina/ });
    expect(apexDialog()).toBeNull();
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
  });

  it('a plain file asked for while the APEX dialog is open closes it as well', async () => {
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(APEX_H5);
    await screen.findByRole('dialog', { name: /Open measurements from session.h5/ });
    await typePathAndEnter(PLAIN);
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    expect(ebsdApi.loadWithProgress.mock.calls[0][0]).toBe(PLAIN);
    expect(apexDialog()).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('a plain file asked for loads and closes the dialog, so the older file cannot load over it', async () => {
    await openDialog();
    fireEvent.click(radio('1: Unprocessed patterns'));
    await typePathAndEnter(PLAIN);
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    expect(ebsdApi.loadWithProgress.mock.calls[0][0]).toBe(PLAIN);
    expect(screen.queryByRole('dialog')).toBeNull();
    // nothing is left to open the older file with: Enter anywhere loads nothing more
    fireEvent.keyDown(document.body, { key: 'Enter' });
    await act(async () => {});
    expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1);
  });

  it('another file with a choice takes over the dialog: its name, its own preselection, its own load', async () => {
    await openDialog();
    fireEvent.click(radio('1: Unprocessed patterns'));
    await typePathAndEnter(OTHER);
    await screen.findByRole('dialog', { name: /Open other\.h5oina/ });
    expect(choicesDialog('two.h5oina')).toBeNull();
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
    expect(radio('1: Processed patterns').checked).toBe(true);
    await open();
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    const [path, options] = ebsdApi.loadWithProgress.mock.calls[0];
    expect(path).toBe(OTHER);
    expect(options.variant).toEqual({ measurement: '1', pattern_source: 'processed' });
  });

  it('the same file asked for again keeps the dialog and the pick made in it', async () => {
    await openDialog();
    fireEvent.click(radio('1: Unprocessed patterns'));
    await typePathAndEnter(PATH);
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
    expect(radio('1: Unprocessed patterns').checked).toBe(true);
  });

  it('an unreadable file asked for closes the dialog and says why', async () => {
    ANSWERS['/d/broken.h5oina'] = { kind: 'unreadable', problem: 'apexFileUnreadable', problem_detail: 'x' };
    await openDialog();
    await typePathAndEnter('/d/broken.h5oina');
    await screen.findByText(/Orienta will not offer a choice of measurement or pattern source for this file/);
    expect(choicesDialog('two.h5oina')).toBeNull();
  });
});
