// @vitest-environment jsdom
//
// The viewer, mounted, with the backend mocked: the way a measurement dialog's loads run through
// handleApexLoad / doLoadFile / requestLoad (the pure pieces have their own tests in apexLoad.test.js;
// this one holds the VIEWER to them). The numbers and names are made up.
import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

vi.mock('../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  const fake = (base) => new Proxy(base, {
    get: (target, prop) => {
      if (prop === 'then' || typeof prop === 'symbol') return undefined;
      if (!(prop in target)) target[prop] = vi.fn(async () => ({ data: {} }));
      return target[prop];
    },
  });
  return {
    ...actual,
    ebsdApi: fake({}),
    edsApi: fake({}),
    pcApi: fake({}),
    scanGeometryApi: fake({}),
  };
});

// The export dialog is replaced by a stand-in that shows the name the viewer suggests.
vi.mock('../common/ImageExportDialog', () => ({
  default: (p) => <div data-testid="export-name" data-base={p.defaultBaseName} data-label={p.annotations?.label} />,
}));

import EBSDViewer from './EBSDViewer';
import { ebsdApi, scanGeometryApi } from '../../services/api';
import useToastStore from '../../stores/useToastStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import useDataStore from '../../stores/useDataStore';
import HEX_LIMITS from './__fixtures__/hexLimits.json';
import EN from '../../locales/en/ebsdviewer.json';

const H5 = '/d/series/session.h5';
const GA = '/Area 1/OIM Map 1';
const GB = '/Area 1/OIM Map 2';
const GC = '/Area 1/OIM Map 3';
const GUID_A = '3f2a8e10-5c44-4b8e-9a51-7d02bb1ec91e';
const GUID_B = '91d0c2f4-7e13-4f0a-8c6d-2b5e9a1f07ab';
const GUID_C = 'c7e41a09-3d8b-4e27-b6f5-90a4d17e5b12';

const measurement = (group, id, name, file) => ({
  group, id, area: 'Area 1', name, hex: false, grid_type: 'SqrGrid', n_rows: 4, n_cols: 3, n_points: 12,
  step_x: 1, step_y: 1, sample_tilt: 70, elevation: 5, azimuth: 0, beam_kv: 20, has_sem: false, has_iq: false,
  phases: [], pc: [0.5, 0.5, 0.5], loadable: true, problem: null, measurement_problem: null,
  problem_detail: null, measured_at: null, measured_at_kind: null, consistent: true,
  pattern_files: [{ path: `/d/series/${file}`, name: file, n_patterns: 12, pattern_shape: [8, 8],
    pairing: 'guid', count: 'fits', matches_grid: true, problem: null, names_measurement: id,
    names_group: group }],
});
const INSPECTION = {
  kind: 'apex', h5_path: H5, folder: '/d/series', folder_problem: null,
  measurements: [measurement(GA, GUID_A, 'OIM Map 1', 'a.up1'), measurement(GB, GUID_B, 'OIM Map 2', 'b.up1'),
    measurement(GC, GUID_C, 'OIM Map 3', 'c.up1')],
  unmatched_pattern_files: [],
};

const LOADED = { grid_shape: [4, 3], pattern_shape: [8, 8], pattern_count: 12, format_type: 'EDAX', has_patterns: true };

const hadResizeObserver = 'ResizeObserver' in globalThis;
beforeAll(() => {
  if (!hadResizeObserver) {
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  }
});
afterAll(() => { if (!hadResizeObserver) delete globalThis.ResizeObserver; });

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  useToastStore.setState({ toasts: [] });
  useLoadedFilesStore.setState({ files: [] });
  ebsdApi.inspect.mockReset();
  ebsdApi.inspect.mockResolvedValue({ data: INSPECTION });
  ebsdApi.loadWithProgress.mockReset();
  ebsdApi.loadWithProgress.mockResolvedValue({ data: LOADED });
  ebsdApi.switchFile.mockReset();
  ebsdApi.getPattern.mockReset();
  ebsdApi.getPattern.mockResolvedValue({ data: {} });
  ebsdApi.loadedFiles.mockReset();
  ebsdApi.loadedFiles.mockResolvedValue({ data: {} });
  scanGeometryApi.hexLimits.mockReset();
  scanGeometryApi.hexLimits.mockResolvedValue({ data: HEX_LIMITS });
});
afterEach(() => cleanup());

const TITLES = ['Area 1 / OIM Map 1', 'Area 1 / OIM Map 2', 'Area 1 / OIM Map 3'];
const loadedEntry = (title, file, id, active = false) => ({
  path: `/d/series/${file}`, name: file.replace(/\.up1$/, ''), active,
  apex: { h5_path: H5, measurement_id: id, measurement_name: title },
});
const tick = (title) => fireEvent.click(screen.getByRole('checkbox', { name: `Open ${title}` }));
const openTicked = () => act(async () => {
  fireEvent.click(screen.getByRole('button', { name: /^Open \d+ measurements?$/ }));
});
const pathBox = () => screen.getByPlaceholderText(/.+/, { selector: 'input[type="text"]' });
const recent = () => JSON.parse(localStorage.getItem('kikuchipy_recent_files') || '[]');
const toastTexts = (type) => useToastStore.getState().toasts.filter((x) => x.type === type).map((x) => x.message);
const inList = (re) => screen.getAllByText(re).find((el) => el.tagName !== 'OPTION');
const registry = (files) => ebsdApi.loadedFiles.mockResolvedValue({ data: { files } });

async function typePathAndEnter(value) {
  const box = pathBox();
  fireEvent.change(box, { target: { value } });
  await act(async () => { fireEvent.keyDown(box, { key: 'Enter' }); });
}
async function openDialog() {
  render(<EBSDViewer isActive onNavigate={vi.fn()} />);
  await typePathAndEnter(H5);
  return screen.findByRole('dialog', { name: /session\.h5/ });
}
const dialogGone = () => expect(screen.queryByRole('dialog', { name: /Open measurements/ })).toBeNull();

describe('a result activated on another page brings up its file or variant', () => {
  const settled = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

  it('the viewer re-reads what it shows, as after a switch in the header', async () => {
    useDataStore.setState({ filePath: H5 });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await settled();
    ebsdApi.datasets.mockClear();
    ebsdApi.getPattern.mockClear();
    await act(async () => { useDataStore.getState().noteFileChangedElsewhere(); });
    await waitFor(() => expect(ebsdApi.datasets).toHaveBeenCalledTimes(1));
    expect(ebsdApi.getPattern).toHaveBeenCalled();
  });

  it('a viewer that opens afterwards takes no count for an event', async () => {
    useDataStore.getState().noteFileChangedElsewhere();
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await settled();
    expect(ebsdApi.getPattern).not.toHaveBeenCalled();
  });
});

describe('an APEX metadata file opens the measurement dialog, and nothing loads before a choice', () => {
  it('asks the backend about the .h5, then shows the measurements it answered with', async () => {
    await openDialog();
    expect(ebsdApi.inspect).toHaveBeenCalledWith(H5);
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
    for (const title of TITLES) expect(screen.getByRole('checkbox', { name: `Open ${title}` })).toBeTruthy();
  });

  it('a second Enter while the first question is pending asks nothing more', async () => {
    let answer;
    ebsdApi.inspect.mockReturnValue(new Promise((resolve) => { answer = resolve; }));
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(H5);
    await act(async () => { fireEvent.keyDown(pathBox(), { key: 'Enter' }); });
    expect(ebsdApi.inspect).toHaveBeenCalledTimes(1);
    await act(async () => { answer({ data: INSPECTION }); });
    expect(await screen.findAllByRole('dialog', { name: /session\.h5/ })).toHaveLength(1);
  });

  it('says why an .h5 cannot be read, and opens it the ordinary way only on request', async () => {
    ebsdApi.inspect.mockResolvedValue({ data: { kind: 'unreadable', problem: 'apexFileUnreadable', problem_detail: 'x' } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(H5);
    expect(await screen.findByText(/cannot be opened as an HDF5 file/)).toBeTruthy();
    dialogGone();
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Open anyway' })); });
    expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1);
    expect(ebsdApi.loadWithProgress.mock.calls[0][0]).toBe(H5);
    expect(ebsdApi.loadWithProgress.mock.calls[0][1].apex).toBeNull();
  });
});

describe('the dialog\'s run of loads', () => {
  it('loads the ticked measurements in the order listed, each with its pairing', async () => {
    await openDialog();
    tick(TITLES[2]);
    tick(TITLES[0]);
    await openTicked();
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(2));
    const calls = ebsdApi.loadWithProgress.mock.calls;
    expect(calls[0][0]).toBe('/d/series/a.up1');
    expect(calls[0][1].apex).toEqual({ h5_path: H5, measurement_id: GUID_A });
    expect(calls[1][0]).toBe('/d/series/c.up1');
    expect(calls[1][1].apex).toEqual({ h5_path: H5, measurement_id: GUID_C });
    dialogGone();
  });

  it('stops at the first failure: the failed one and those after it stay closed, and the user is told which', async () => {
    ebsdApi.loadWithProgress
      .mockResolvedValueOnce({ data: LOADED })
      .mockRejectedValueOnce(Object.assign(new Error('boom'), { response: { data: { detail: 'boom' } } }));
    await openDialog();
    for (const title of TITLES) tick(title);
    await openTicked();
    await waitFor(() => expect(toastTexts('warning').length).toBeGreaterThan(0));
    expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(2);
    expect(toastTexts('warning')).toEqual(['Not opened: Area 1 / OIM Map 3. Pick the .h5 file again to open them.']);
    fireEvent.click(screen.getByTitle(EN.hoverTips.logToggleExpand));
    expect(document.body.textContent).toContain('Opening stopped at Area 1 / OIM Map 2.');
    expect(document.body.textContent).not.toContain('Opening stopped at Area 1 / OIM Map 1.');
  });

  it('a failure of the last one stops nothing and warns of nothing', async () => {
    ebsdApi.loadWithProgress.mockRejectedValueOnce(new Error('boom'));
    await openDialog();
    tick(TITLES[2]);
    await openTicked();
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1));
    expect(toastTexts('warning')).toEqual([]);
  });
});

describe('reopening goes through the measurement dialog again', () => {
  it('records the metadata file, not the pattern file, in the path box and in Recent', async () => {
    await openDialog();
    tick(TITLES[0]);
    await openTicked();
    await waitFor(() => expect(recent()).toEqual([H5]));
    expect(pathBox().value).toBe(H5);
  });

  it('Recent -> reopen asks the dialog again and loads nothing by itself', async () => {
    await openDialog();
    tick(TITLES[0]);
    await openTicked();
    await waitFor(() => expect(recent()).toEqual([H5]));
    ebsdApi.inspect.mockClear();
    await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
    fireEvent.click(screen.getByRole('button', { name: /^Recent/ }));
    // (the status bar names the same file in a span; the list of recent files uses divs)
    const entries = screen.getAllByTitle(H5).filter((el) => el.tagName === 'DIV');
    expect(entries.map((e) => e.textContent)).toEqual(['session.h5']);
    fireEvent.click(entries[0]);
    expect(await screen.findByRole('dialog', { name: /session\.h5/ })).toBeTruthy();
    expect(ebsdApi.inspect).toHaveBeenCalledWith(H5);
    expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(1);
  });

  it('a pattern file of a paired entry typed into the path box is asked about through its metadata file', async () => {
    registry([loadedEntry(TITLES[0], 'a.up1', GUID_A, true), loadedEntry(TITLES[1], 'b.up1', GUID_B)]);
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    await typePathAndEnter('/d/series/b.up1');
    expect(ebsdApi.inspect).toHaveBeenCalledWith(H5);
    expect(await screen.findByRole('dialog', { name: /session\.h5/ })).toBeTruthy();
    expect(ebsdApi.loadWithProgress).not.toHaveBeenCalled();
  });
});

describe('the loaded-files list of the viewer', () => {
  it('names a measurement by its measurement, and any other file by its own name', async () => {
    registry([loadedEntry(TITLES[0], 'a.up1', GUID_A, true), loadedEntry(TITLES[1], 'b.up1', GUID_B),
      { path: '/d/other/plain_scan.h5oina', name: 'plain_scan', active: false }]);
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    expect(inList(/Area 1 \/ OIM Map 1 \(a\)/)).toBeTruthy();
    expect(inList(/Area 1 \/ OIM Map 2 \(b\)/)).toBeTruthy();
    expect(inList(/plain_scan/)).toBeTruthy();
  });

  it('switching to a measurement whose pairing had to be dropped warns, naming the reason', async () => {
    registry([loadedEntry(TITLES[0], 'a.up1', GUID_A, true), loadedEntry(TITLES[1], 'b.up1', GUID_B)]);
    ebsdApi.switchFile.mockResolvedValue({ data: { ...LOADED, success: true, apex_pairing_dropped: true,
      apex_pairing_dropped_reason: 'apexUnreadable' } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    await act(async () => { fireEvent.click(inList(/Area 1 \/ OIM Map 2 \(b\)/)); });
    await waitFor(() => expect(toastTexts('warning')).toHaveLength(1));
    expect(toastTexts('warning')[0]).toContain('could not be used again (The metadata file could not be read.)');
  });

  it('and says nothing when the pairing was used', async () => {
    registry([loadedEntry(TITLES[0], 'a.up1', GUID_A, true), loadedEntry(TITLES[1], 'b.up1', GUID_B)]);
    ebsdApi.switchFile.mockResolvedValue({ data: { ...LOADED, success: true, apex_pairing_dropped: false } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    await act(async () => { fireEvent.click(inList(/Area 1 \/ OIM Map 2 \(b\)/)); });
    await waitFor(() => expect(toastTexts('success')).toHaveLength(1));
    expect(toastTexts('warning')).toEqual([]);
  });
});

describe('every way of switching shows the metadata file, not the pattern file, in the path box', () => {
  const twoMeasurements = () => registry([
    loadedEntry(TITLES[0], 'a.up1', GUID_A, true), loadedEntry(TITLES[1], 'b.up1', GUID_B)]);

  it('switching in the viewer\'s own list', async () => {
    twoMeasurements();
    ebsdApi.switchFile.mockResolvedValue({ data: { ...LOADED, success: true } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    await act(async () => { fireEvent.click(inList(/Area 1 \/ OIM Map 2 \(b\)/)); });
    await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalledWith('/d/series/b.up1'));
    await waitFor(() => expect(toastTexts('success')).toHaveLength(1));
    expect(pathBox().value).toBe(H5);
  });

  it('switching in the header switcher', async () => {
    twoMeasurements();
    ebsdApi.switchFile.mockResolvedValue({ data: { ...LOADED, success: true } });
    const { container } = render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    const select = container.querySelector('[data-file-switcher] select');
    expect(select).toBeTruthy();
    await act(async () => { fireEvent.change(select, { target: { value: '/d/series/b.up1' } }); });
    await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalledWith('/d/series/b.up1'));
    await waitFor(() => expect(pathBox().value).toBe(H5));
  });

  it('entering the viewer with a measurement already open in the backend', async () => {
    const files = [loadedEntry(TITLES[0], 'a.up1', GUID_A, true)];
    registry(files);
    useLoadedFilesStore.setState({ files });
    useDataStore.setState({ ebsdLoaded: false });
    ebsdApi.datasets.mockResolvedValue({ data: { active: 'a', file_path: '/d/series/a.up1', datasets: [{
      name: 'a', active: true, navigation_shape: [3, 4], signal_shape: [8, 8], n_patterns: 12 }] } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await waitFor(() => expect(pathBox().value).toBe(H5));
    await waitFor(() => expect(recent()).toEqual([H5]));
  });
});

describe('Escape while a prompt is open on top of the dialog', () => {
  it('closes only the prompt when nothing is ticked, and the dialog stays', async () => {
    const [a, b, c] = INSPECTION.measurements;
    ebsdApi.inspect.mockResolvedValue({ data: { ...INSPECTION,
      measurements: [a, { ...b, loadable: false, problem: 'apexNoPatternFile', pattern_files: [] }, c] } });
    await openDialog();
    fireEvent.click(screen.getByRole('button', { name: 'Choose file…' }));
    expect(await screen.findByRole('dialog', { name: 'Choose the pattern file' })).toBeTruthy();
    await act(async () => { fireEvent.keyDown(window, { key: 'Escape' }); });
    expect(screen.queryByRole('dialog', { name: 'Choose the pattern file' })).toBeNull();
    expect(screen.getByRole('dialog', { name: /session\.h5/ })).toBeTruthy();
    await act(async () => { fireEvent.keyDown(window, { key: 'Escape' }); });
    dialogGone();
  });

  it('closes the prompt and leaves the dialog and its ticks', async () => {
    const [a, b, c] = INSPECTION.measurements;
    ebsdApi.inspect.mockResolvedValue({ data: { ...INSPECTION,
      measurements: [a, { ...b, loadable: false, problem: 'apexNoPatternFile', pattern_files: [] }, c] } });
    await openDialog();
    tick(TITLES[0]);
    fireEvent.click(screen.getByRole('button', { name: 'Choose file…' }));
    expect(await screen.findByRole('dialog', { name: 'Choose the pattern file' })).toBeTruthy();
    await act(async () => { fireEvent.keyDown(window, { key: 'Escape' }); });
    expect(screen.queryByRole('dialog', { name: 'Choose the pattern file' })).toBeNull();
    expect(screen.getByRole('dialog', { name: /session\.h5/ })).toBeTruthy();
    expect(screen.getByRole('checkbox', { name: `Open ${TITLES[0]}` }).checked).toBe(true);
  });
});


describe('the status bar and the suggested export name follow the active measurement', () => {
  const entryA = (active) => loadedEntry(TITLES[0], 'a.up1', GUID_A, active);
  const entryB = (active) => loadedEntry(TITLES[1], 'b.up1', GUID_B, active);
  const exportButton = () => screen.getByRole('button', { name: EN.processing.exportPattern });
  const exportBase = () => screen.getByTestId('export-name').getAttribute('data-base');
  const statusBar = (path) => screen.getByTitle(path);

  beforeEach(() => { ebsdApi.getPattern.mockResolvedValue({ data: { image: 'AAAA' } }); });

  it('names the measurement that is open, not the .h5 both were opened from', async () => {
    registry([entryA(false), entryB(true)]);
    await openDialog();
    tick(TITLES[0]);
    tick(TITLES[1]);
    await openTicked();
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(pathBox().value).toBe(H5));
    const bar = await waitFor(() => statusBar('/d/series/b.up1'));
    expect(bar.textContent).toBe('Area 1 / OIM Map 2 (b)');
    await waitFor(() => expect(exportButton().disabled).toBe(false));
    await act(async () => { fireEvent.click(exportButton()); });
    expect(exportBase()).toBe('session_Area 1_OIM Map 2_r0_c0');
    expect(screen.getByTestId('export-name').getAttribute('data-label')).toContain('session_Area 1_OIM Map 2 ·');
    expect(document.body.textContent).not.toMatch(/session\.h5\b/);
  });

  it('follows a switch to the other measurement', async () => {
    registry([entryA(false), entryB(true)]);
    await openDialog();
    tick(TITLES[0]);
    tick(TITLES[1]);
    await openTicked();
    await waitFor(() => expect(statusBar('/d/series/b.up1').textContent).toBe('Area 1 / OIM Map 2 (b)'));
    await waitFor(() => expect(exportButton().disabled).toBe(false));
    await act(async () => { fireEvent.click(exportButton()); });
    const before = exportBase();
    registry([entryA(true), entryB(false)]);
    ebsdApi.switchFile.mockResolvedValue({ data: { ...LOADED, success: true } });
    await act(async () => { fireEvent.click(inList(/Area 1 \/ OIM Map 1 \(a\)/)); });
    await waitFor(() => expect(statusBar('/d/series/a.up1').textContent).toBe('Area 1 / OIM Map 1 (a)'));
    await waitFor(() => expect(exportBase()).toBe('session_Area 1_OIM Map 1_r0_c0'));
    expect(before).toBe('session_Area 1_OIM Map 2_r0_c0');
    expect(pathBox().value).toBe(H5);
  });

  it('leaves an ordinary file exactly as it was: its own name in the status bar and in the export name', async () => {
    const plain = '/data/scans/sample_01.h5oina';
    registry([{ path: plain, name: 'sample_01', active: true }]);
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(plain);
    expect((await waitFor(() => statusBar(plain))).textContent).toBe('sample_01.h5oina');
    await waitFor(() => expect(exportButton().disabled).toBe(false));
    await act(async () => { fireEvent.click(exportButton()); });
    expect(exportBase()).toBe('sample_01_r0_c0');
  });

  it('leaves an ordinary .h5 that is not APEX exactly as it was', async () => {
    const plain = '/data/scans/other_scan.h5';
    ebsdApi.inspect.mockResolvedValue({ data: { kind: 'plain' } });
    registry([{ path: plain, name: 'other_scan', active: true }]);
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await typePathAndEnter(plain);
    expect((await waitFor(() => statusBar(plain))).textContent).toBe('other_scan.h5');
    await waitFor(() => expect(exportButton().disabled).toBe(false));
    await act(async () => { fireEvent.click(exportButton()); });
    expect(exportBase()).toBe('other_scan_r0_c0');
  });

  it('shows what the user typed when it is not the metadata file of the open measurement, as before', async () => {
    registry([entryA(true)]);
    await openDialog();
    tick(TITLES[0]);
    await openTicked();
    await waitFor(() => expect(statusBar('/d/series/a.up1').textContent).toBe('Area 1 / OIM Map 1 (a)'));
    fireEvent.change(pathBox(), { target: { value: '/x/typed_but_not_loaded.h5' } });
    expect(screen.getByTitle('/x/typed_but_not_loaded.h5').textContent).toBe('typed_but_not_loaded.h5');
  });
});