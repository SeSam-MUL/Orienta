// @vitest-environment jsdom
/**
 * syncFromBackend after a WebSocket gap must never wipe an open file because
 * a REQUEST failed. Only a real answer "nothing is loaded" may clear the store.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

const info = vi.fn();
const getMetadata = vi.fn();
let inFlight = false;
vi.mock('../services/api', () => ({
  ebsdApi: { info: (...a) => info(...a), getMetadata: (...a) => getMetadata(...a) },
  isFileLoadInFlight: () => inFlight,
}));

import useDataStore from './useDataStore';

const OPEN = { isFileOpen: true, filePath: '/data/a.h5oina', gridShape: [90, 120] };

beforeEach(() => {
  info.mockReset();
  getMetadata.mockReset();
  getMetadata.mockResolvedValue({ data: {} });
  inFlight = false;
  useDataStore.setState({ ...OPEN });
});

describe('syncFromBackend', () => {
  it('keeps the open file when the info request FAILS', async () => {
    info.mockRejectedValue(new Error('Network Error'));
    const result = await useDataStore.getState().syncFromBackend();
    expect(result).toBe(false);
    expect(useDataStore.getState().isFileOpen).toBe(true);
    expect(useDataStore.getState().filePath).toBe('/data/a.h5oina');
  });

  it('still clears the store when the backend really says nothing is loaded', async () => {
    info.mockResolvedValue({ data: { loaded: false } });
    await useDataStore.getState().syncFromBackend();
    expect(useDataStore.getState().isFileOpen).toBe(false);
  });

  it('keeps the store of an open file without a signal: an EDS-only file is open', async () => {
    info.mockResolvedValue({ data: { loaded: false, open_file: '/data/a.h5oina' } });
    await useDataStore.getState().syncFromBackend();
    expect(useDataStore.getState().isFileOpen).toBe(true);
    expect(useDataStore.getState().filePath).toBe('/data/a.h5oina');
  });

  it('does nothing while a file load or switch is in flight', async () => {
    inFlight = true;
    info.mockResolvedValue({ data: { loaded: false } });
    const result = await useDataStore.getState().syncFromBackend();
    expect(result).toBe(false);
    expect(info).not.toHaveBeenCalled();
    expect(useDataStore.getState().isFileOpen).toBe(true);
  });
});

describe('syncFromBackend and the raw patterns', () => {
  const answer = (extra) => ({
    data: { loaded: true, file_path: '/data/b.h5oina', data_shape: [2, 3, 4, 5], ...extra },
  });

  it('takes whether the measurement holds unprocessed patterns from the answer', async () => {
    info.mockResolvedValue(answer({ has_raw_patterns: false }));
    await useDataStore.getState().syncFromBackend();
    expect(useDataStore.getState().filePath).toBe('/data/b.h5oina');
    expect(useDataStore.getState().hasRawPatterns).toBe(false);

    useDataStore.setState({ ...OPEN });
    info.mockResolvedValue(answer({ has_raw_patterns: true }));
    await useDataStore.getState().syncFromBackend();
    expect(useDataStore.getState().hasRawPatterns).toBe(true);
  });

  it('does not claim raw patterns when the answer says nothing about them', async () => {
    info.mockResolvedValue(answer({}));
    await useDataStore.getState().syncFromBackend();
    expect(useDataStore.getState().hasRawPatterns).toBe(false);
  });
});
