import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../services/api', () => ({
  ebsdApi: {
    loadedFiles: vi.fn(),
    removeLoadedFile: vi.fn(),
    clearLoadedFiles: vi.fn(),
    clearAllLoadedFiles: vi.fn(),
  },
}));

import { ebsdApi } from '../services/api';
import useLoadedFilesStore from './useLoadedFilesStore';

beforeEach(() => {
  vi.clearAllMocks();
  useLoadedFilesStore.setState({ files: [] });
});

describe('useLoadedFilesStore', () => {
  it('refresh populates files from the backend', async () => {
    ebsdApi.loadedFiles.mockResolvedValue({
      data: { files: [{ path: 'a', name: 'a', active: true }] },
    });
    const ok = await useLoadedFilesStore.getState().refresh();
    expect(ok).toBe(true);
    expect(useLoadedFilesStore.getState().files).toEqual([
      { path: 'a', name: 'a', active: true },
    ]);
  });

  it('refresh empties the list (and returns false) when the backend errors', async () => {
    useLoadedFilesStore.setState({ files: [{ path: 'a', name: 'a', active: true }] });
    ebsdApi.loadedFiles.mockRejectedValue(new Error('backend down'));
    const ok = await useLoadedFilesStore.getState().refresh();
    expect(ok).toBe(false);
    expect(useLoadedFilesStore.getState().files).toEqual([]);
  });

  it('remove hits the API then refreshes', async () => {
    ebsdApi.removeLoadedFile.mockResolvedValue({});
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [] } });
    await useLoadedFilesStore.getState().remove('a');
    expect(ebsdApi.removeLoadedFile).toHaveBeenCalledWith('a');
    expect(ebsdApi.loadedFiles).toHaveBeenCalledTimes(1);
  });

  it('clearKeepActive hits /clear then refreshes', async () => {
    ebsdApi.clearLoadedFiles.mockResolvedValue({});
    ebsdApi.loadedFiles.mockResolvedValue({
      data: { files: [{ path: 'a', name: 'a', active: true }] },
    });
    await useLoadedFilesStore.getState().clearKeepActive();
    expect(ebsdApi.clearLoadedFiles).toHaveBeenCalledTimes(1);
    expect(useLoadedFilesStore.getState().files).toHaveLength(1);
  });

  it('clearAll hits /clear-all then refreshes to empty', async () => {
    ebsdApi.clearAllLoadedFiles.mockResolvedValue({});
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [] } });
    await useLoadedFilesStore.getState().clearAll();
    expect(ebsdApi.clearAllLoadedFiles).toHaveBeenCalledTimes(1);
    expect(useLoadedFilesStore.getState().files).toEqual([]);
  });

  it('a failing mutation propagates and leaves the list unchanged', async () => {
    useLoadedFilesStore.setState({ files: [{ path: 'a', name: 'a', active: true }] });
    ebsdApi.removeLoadedFile.mockRejectedValue(new Error('cannot remove active'));
    await expect(useLoadedFilesStore.getState().remove('a')).rejects.toThrow(
      'cannot remove active',
    );
    // Refresh was never reached → list untouched.
    expect(ebsdApi.loadedFiles).not.toHaveBeenCalled();
    expect(useLoadedFilesStore.getState().files).toHaveLength(1);
  });
});
