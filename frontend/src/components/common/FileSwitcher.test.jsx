// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import FileSwitcher from './FileSwitcher';

vi.mock('../../services/api', () => ({
  ebsdApi: { loadedFiles: vi.fn(), switchFile: vi.fn() },
  isSupersededLoad: () => false,
}));
vi.mock('../../theme/components', () => ({
  colors: { bg: '#000', text: '#fff', textSecondary: '#aaa', border: '#444', cyan: '#0ff', red: '#f00' },
}));

// Mutable mock store state.
let storeState;
vi.mock('../../stores/useDataStore', () => {
  const useStore = (selector) => selector(storeState);
  useStore.getState = () => storeState;
  return { default: useStore };
});

// The loaded-files list now lives in a shared store. Use the REAL store (it
// reads the mocked ebsdApi.loadedFiles), reset before each test so the
// singleton doesn't leak state across tests.
import { ebsdApi } from '../../services/api';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import useToastStore from '../../stores/useToastStore';

beforeEach(() => {
  vi.clearAllMocks();
  cleanup();
  useLoadedFilesStore.setState({ files: [] });
  useToastStore.setState({ toasts: [] });
  storeState = {
    filePath: '/data/fileA.h5oina',
    isFileOpen: true,
    syncFromBackend: vi.fn().mockResolvedValue(true),
    setEBSDLoaded: vi.fn(),
    ebsdInfo: null,
  };
});

describe('FileSwitcher', () => {
  it('renders nothing when fewer than 2 files loaded', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [{ path: '/data/fileA.h5oina', name: 'fileA', active: true }] } });
    const { container } = render(<FileSwitcher />);
    await waitFor(() => {});
    expect(container.querySelector('[data-file-switcher]')).toBeNull();
  });

  it('renders a dropdown with all loaded files when 2+ present', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/data/fileA.h5oina', name: 'fileA', active: true },
      { path: '/data/fileB.h5oina', name: 'fileB', active: false },
    ] } });
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('fileA');
    expect(container.querySelector('[data-file-switcher]')).toBeTruthy();
    expect(container.querySelectorAll('option')).toHaveLength(2);
  });

  it('labels a file opened as a non-default variant with the variant, and keeps one entry per path', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/data/fileA.h5oina', name: 'fileA', active: false },
      { path: '/data/multi.h5oina', name: 'multi · 2 · unprocessed', active: true,
        variant: { measurement: '2', pattern_source: 'unprocessed' } },
    ] } });
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('multi · 2 · unprocessed');
    expect([...container.querySelectorAll('option')].map((o) => o.textContent))
      .toEqual(['fileA', 'multi · 2 · unprocessed']);
  });

  it('calls switchFile + syncFromBackend on selection change', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/data/fileA.h5oina', name: 'fileA', active: true },
      { path: '/data/fileB.h5oina', name: 'fileB', active: false },
    ] } });
    ebsdApi.switchFile.mockResolvedValue({ data: { success: true } });
    const onSwitched = vi.fn();
    const { container, findByText } = render(<FileSwitcher onSwitched={onSwitched} />);
    await findByText('fileA');
    const select = container.querySelector('select');
    fireEvent.change(select, { target: { value: '/data/fileB.h5oina' } });
    await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalledWith('/data/fileB.h5oina'));
    expect(storeState.syncFromBackend).toHaveBeenCalled();
    await waitFor(() => expect(onSwitched).toHaveBeenCalledWith('/data/fileB.h5oina'));
  });

  it('hands the switch answer to the store, on a page that passes no onSwitched (EDS, Phase Map)', async () => {
    // The answer is the viewer's `ebsdInfo`: its grid and scan geometry are the new file's.
    storeState.ebsdInfo = { grid_shape: [1, 1] };   // a viewer that has been entered
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/data/fileA.h5oina', name: 'fileA', active: true },
      { path: '/data/fileB.h5oina', name: 'fileB', active: false },
    ] } });
    const answer = { success: true, grid_shape: [3, 4], scan_geometry: { kind: 'hex', odd_row_shift: -0.5 } };
    ebsdApi.switchFile.mockResolvedValue({ data: answer });
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('fileA');
    fireEvent.change(container.querySelector('select'), { target: { value: '/data/fileB.h5oina' } });
    await waitFor(() => expect(storeState.setEBSDLoaded).toHaveBeenCalledWith(answer));
    expect(storeState.syncFromBackend).toHaveBeenCalled();
  });

  it('selects the backend-active file even when store filePath is stale', async () => {
    // Regression: after a page reload the store's filePath rehydrates late
    // (or with a different spelling), so the <select> used to default to the
    // FIRST option — showing the WRONG file as active. The dropdown must
    // follow the backend `active` flag instead.
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/data/fileA.h5oina', name: 'fileA', active: false },
      { path: '/data/fileB.h5oina', name: 'fileB', active: true },
    ] } });
    storeState.filePath = '';   // stale / not-yet-synced
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('fileA');
    const select = container.querySelector('select');
    expect(select.value).toBe('/data/fileB.h5oina');
    expect(select.selectedIndex).toBe(1);
  });

  it('shows an error when switchFile rejects', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/data/fileA.h5oina', name: 'fileA', active: true },
      { path: '/data/fileB.h5oina', name: 'fileB', active: false },
    ] } });
    ebsdApi.switchFile.mockRejectedValue(new Error('disk on fire'));
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('fileA');
    fireEvent.change(container.querySelector('select'), { target: { value: '/data/fileB.h5oina' } });
    await findByText(/disk on fire/);
  });
  it('names a measurement of an APEX file by its measurement, not only by its pattern file', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/d/a.up1', name: 'a', active: true,
        apex: { measurement_name: 'Area 1 / OIM Map 1', h5_path: '/d/s.h5', measurement_id: 'g1', pairing: 'guid' } },
      { path: '/d/b.up1', name: 'b', active: false,
        apex: { measurement_name: 'Area 1 / OIM Map 2', h5_path: '/d/s.h5', measurement_id: 'g2', pairing: 'guid' } },
      { path: '/d/c.h5oina', name: 'scan C', active: false },
    ] } });
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('Area 1 / OIM Map 1 (a)');
    const labels = [...container.querySelectorAll('option')].map((o) => o.textContent);
    expect(labels).toEqual(['Area 1 / OIM Map 1 (a)', 'Area 1 / OIM Map 2 (b)', 'scan C']);
  });

  it('says so when the remembered APEX pairing could not be used on a switch', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/d/a.up1', name: 'a', active: true },
      { path: '/d/b.up1', name: 'b', active: false },
    ] } });
    ebsdApi.switchFile.mockResolvedValue({ data: { success: true, apex_pairing_dropped: true,
      apex_pairing_dropped_reason: 'apexUnreadable' } });
    const { container, findByText } = render(<FileSwitcher />);
    await findByText('a');
    fireEvent.change(container.querySelector('select'), { target: { value: '/d/b.up1' } });
    await waitFor(() => expect(useToastStore.getState().toasts).toHaveLength(1));
    const toast = useToastStore.getState().toasts[0];
    expect(toast.type).toBe('warning');
    expect(toast.message).toBe('The APEX metadata file this measurement was opened with could not be used again '
      + '(The metadata file could not be read.). The pattern file was loaded without it, so its pattern '
      + 'centre and grid may now differ from the ones the measurement dialog showed.');
  });

  it('says nothing when the pairing was used', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: '/d/a.up1', name: 'a', active: true },
      { path: '/d/b.up1', name: 'b', active: false },
    ] } });
    ebsdApi.switchFile.mockResolvedValue({ data: { success: true, apex_pairing_dropped: false,
      apex_pairing_dropped_reason: null } });
    const onSwitched = vi.fn();
    const { container, findByText } = render(<FileSwitcher onSwitched={onSwitched} />);
    await findByText('a');
    fireEvent.change(container.querySelector('select'), { target: { value: '/d/b.up1' } });
    await waitFor(() => expect(onSwitched).toHaveBeenCalled());
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });
});
