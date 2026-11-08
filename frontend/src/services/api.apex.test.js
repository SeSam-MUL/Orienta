// @vitest-environment jsdom
/** The questions of the APEX dialog, and the choice that travels with a load. */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockPost = vi.fn();
const mockGet = vi.fn();
const mockAxiosInstance = {
  post: mockPost,
  get: mockGet,
  interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
};
vi.mock('axios', () => ({ default: { create: vi.fn(() => mockAxiosInstance) } }));

const { ebsdApi, scanGeometryApi } = await import('./api');

beforeEach(() => {
  mockPost.mockReset();
  mockGet.mockReset();
});

describe('ebsdApi: APEX metadata files', () => {
  it('inspect asks about one path', async () => {
    mockGet.mockResolvedValue({ data: { kind: 'plain' } });
    await ebsdApi.inspect('/d/a.h5');
    expect(mockGet).toHaveBeenCalledWith('/api/ebsd/inspect', { params: { path: '/d/a.h5' } });
  });

  it('the preview of a pattern source is fetched as a blob, with the measurement and the source', async () => {
    mockGet.mockResolvedValue({ data: new Blob(['x']) });
    await ebsdApi.inspectPreview('/d/a.h5oina', '1', 'unprocessed');
    expect(mockGet).toHaveBeenCalledWith('/api/ebsd/inspect/preview', {
      params: { path: '/d/a.h5oina', measurement: '1', source: 'unprocessed' }, responseType: 'blob' });
  });

  it('the preview of a pattern source can be cancelled with an abort signal', async () => {
    mockGet.mockResolvedValue({ data: new Blob(['x']) });
    const ctl = new AbortController();
    await ebsdApi.inspectPreview('/d/a.h5oina', '1', 'processed', ctl.signal);
    expect(mockGet.mock.calls[0][1].signal).toBe(ctl.signal);
  });

  it('a preview is fetched as a blob, with the measurement and the kind', async () => {
    mockGet.mockResolvedValue({ data: new Blob(['x']) });
    await ebsdApi.apexPreview('/d/a.h5', 'GUID', 'iq');
    expect(mockGet).toHaveBeenCalledWith('/api/ebsd/apex/preview', {
      params: { path: '/d/a.h5', measurement: 'GUID', kind: 'iq' }, responseType: 'blob' });
  });

  it('the overview image is the default preview', async () => {
    mockGet.mockResolvedValue({ data: new Blob(['x']) });
    await ebsdApi.apexPreview('/d/a.h5', 'GUID');
    expect(mockGet.mock.calls[0][1].params.kind).toBe('sem');
  });

  it('a pairing is checked with the three paths', async () => {
    mockPost.mockResolvedValue({ data: { ok: true } });
    await ebsdApi.apexCheckPairing({ h5_path: '/d/a.h5', measurement_id: 'G', pattern_path: '/d/p.up1' });
    expect(mockPost).toHaveBeenCalledWith('/api/ebsd/apex/check-pairing',
      { h5_path: '/d/a.h5', measurement_id: 'G', pattern_path: '/d/p.up1' });
  });

  it('a chosen file is described with the same three paths', async () => {
    mockPost.mockResolvedValue({ data: { matches_grid: true } });
    await ebsdApi.apexDescribePatternFile({ h5_path: '/d/a.h5', measurement_id: 'G', pattern_path: '/d/p.up1' });
    expect(mockPost).toHaveBeenCalledWith('/api/ebsd/apex/describe-pattern-file',
      { h5_path: '/d/a.h5', measurement_id: 'G', pattern_path: '/d/p.up1' });
  });

  it('another folder is searched for the whole metadata file at once', async () => {
    mockGet.mockResolvedValue({ data: { measurements: [] } });
    await ebsdApi.apexPatternFiles('/d/a.h5', '/e/more');
    expect(mockGet).toHaveBeenCalledWith('/api/ebsd/apex/pattern-files',
      { params: { h5_path: '/d/a.h5', folder: '/e/more' } });
  });
});

describe('scanGeometryApi', () => {
  it('asks the backend for what a hexagonal scan cannot do yet', async () => {
    mockGet.mockResolvedValue({ data: {} });
    await scanGeometryApi.hexLimits();
    expect(mockGet).toHaveBeenCalledWith('/api/scan-geometry/hex-limits');
  });
});

describe('ebsdApi.loadWithProgress carries the user\'s choice', () => {
  beforeEach(() => {
    mockGet.mockResolvedValue({ data: { stage: 'complete', stage_idx: 4, stage_total: 4, elapsed_seconds: 0.1 } });
    mockPost.mockResolvedValue({ data: { success: true } });
  });

  it('sends apex when given', async () => {
    const apex = { h5_path: '/d/a.h5', measurement_id: 'G' };
    await ebsdApi.loadWithProgress('/d/p.up1', { apex });
    expect(mockPost.mock.calls[0][1].apex).toEqual(apex);
  });

  it('sends no apex key at all otherwise, so an ordinary load is byte-for-byte what it was', async () => {
    await ebsdApi.loadWithProgress('/d/a.h5oina', {});
    expect(Object.keys(mockPost.mock.calls[0][1]).sort()).toEqual(['path', 'request_id', 'use_kikuchipy']);
    mockPost.mockClear();
    await ebsdApi.loadWithProgress('/d/a.h5oina', { apex: null });
    expect(Object.keys(mockPost.mock.calls[0][1]).sort()).toEqual(['path', 'request_id', 'use_kikuchipy']);
    mockPost.mockClear();
    await ebsdApi.loadWithProgress('/d/a.h5oina', { variant: null });
    expect(Object.keys(mockPost.mock.calls[0][1]).sort()).toEqual(['path', 'request_id', 'use_kikuchipy']);
  });

  it('sends the variant (measurement and pattern source) under `variant` when given', async () => {
    const variant = { measurement: '1', pattern_source: 'unprocessed' };
    await ebsdApi.loadWithProgress('/d/a.h5oina', { variant });
    expect(mockPost.mock.calls[0][1].variant).toEqual(variant);
    expect(mockPost.mock.calls[0][1].path).toBe('/d/a.h5oina');
    expect('apex' in mockPost.mock.calls[0][1]).toBe(false);
  });
});
