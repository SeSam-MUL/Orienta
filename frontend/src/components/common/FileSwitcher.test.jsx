// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import FileSwitcher from './FileSwitcher';

vi.mock('../../services/api', () => ({
  ebsdApi: { loadedFiles: vi.fn(), switchFile: vi.fn() },
}));
vi.mock('../../theme/components', () => ({
  colors: { bg: '#000', text: '#fff', textSecondary: '#aaa', border: '#444', cyan: '#0ff', red: '#f00' },
}));

// Mutable mock store state.
let storeState;
vi.mock('../../stores/useDataStore', () => ({
  default: (selector) => selector(storeState),
}));

// The loaded-files list now lives in a shared store. Use the REAL store (it
// reads the mocked ebsdApi.loadedFiles), reset before each test so the
// singleton doesn't leak state across tests.
import { ebsdApi } from '../../services/api';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';

beforeEach(() => {
  vi.clearAllMocks();
  cleanup();
  useLoadedFilesStore.setState({ files: [] });
  storeState = {
    filePath: '/data/fileA.h5oina',
    isFileOpen: true,
    syncFromBackend: vi.fn().mockResolvedValue(true),
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
});
