// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useDefaultLayers } from './useDefaultLayers';

vi.mock('../../../services/api', () => ({
  edsApi: { elements: vi.fn() },
  ebsdApi: { bandContrast: vi.fn() },
  h5Api:   { getElectronList: vi.fn() },
}));

import { edsApi, ebsdApi, h5Api } from '../../../services/api';

function mockAll({ elements = [], images = [], bcLabel = 'native' } = {}) {
  edsApi.elements.mockResolvedValue({ data: { elements } });
  h5Api.getElectronList.mockResolvedValue({ data: { images } });
  ebsdApi.bandContrast.mockResolvedValue({ data: { image: 'b64', label: bcLabel, shape: [128, 156] } });
}

describe('useDefaultLayers', () => {
  beforeEach(() => { vi.clearAllMocks(); });

  it('returns SE + BC + first three elements when all present', async () => {
    mockAll({ elements: ['Al Kα1', 'Fe Kα1', 'Cu Kα1', 'O Kα1'], images: ['SE1'], bcLabel: 'native' });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    const ids = result.current.layers.map(l => l.id);
    expect(ids).toContain('electron-SE1');
    expect(ids).toContain('bc');
    expect(ids).toContain('eds-Al Kα1');
    expect(result.current.layers.filter(l => l.kind === 'eds-element')).toHaveLength(3);
  });

  it('uses V-BSE fallback when electron-image list is empty', async () => {
    mockAll({ elements: ['Al Kα1'], images: [], bcLabel: 'native' });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    const kinds = result.current.layers.map(l => l.kind);
    expect(kinds).toContain('vbse');
    expect(kinds).not.toContain('electron');
  });

  it('drops BC layer when bandContrast returns label="fallback"', async () => {
    mockAll({ elements: ['Al Kα1'], images: [], bcLabel: 'fallback' });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.layers.find(l => l.kind === 'bc')).toBeUndefined();
  });

  it('returns ready=false and empty layers when file is not open', () => {
    const { result } = renderHook(() => useDefaultLayers(false));
    expect(result.current.ready).toBe(false);
    expect(result.current.layers).toEqual([]);
  });

  it('tolerates a failed elements call (treats as zero elements)', async () => {
    edsApi.elements.mockRejectedValue(new Error('boom'));
    h5Api.getElectronList.mockResolvedValue({ data: { images: [] } });
    ebsdApi.bandContrast.mockResolvedValue({ data: { image: 'b64', label: 'native', shape: [128, 156] } });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.layers.find(l => l.kind === 'bc')).toBeDefined();
    expect(result.current.layers.filter(l => l.kind === 'eds-element')).toHaveLength(0);
  });

  it('prefers SE over FSE as default electron-image', async () => {
    mockAll({
      elements: ['Al Kα1'],
      images: ['FSE/Oben links', 'SE/Elektronenbild 1', 'FSE/Unten Mitte'],
      bcLabel: 'native',
    });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    const electronLayer = result.current.layers.find(l => l.kind === 'electron');
    expect(electronLayer).toBeDefined();
    expect(electronLayer.electronName).toBe('SE/Elektronenbild 1');
  });

  it('exposes the full electronImages list', async () => {
    mockAll({
      elements: [],
      images: ['FSE/A', 'FSE/B', 'SE/C'],
      bcLabel: 'native',
    });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.electronImages).toEqual(['FSE/A', 'FSE/B', 'SE/C']);
  });

  it('re-probes when filePath changes', async () => {
    mockAll({ elements: ['Al Kα1'], images: [], bcLabel: 'native' });
    const { result, rerender } = renderHook(
      ({ fp }) => useDefaultLayers(true, fp),
      { initialProps: { fp: '/data/fileA.h5oina' } },
    );
    await waitFor(() => expect(result.current.ready).toBe(true));
    edsApi.elements.mockClear();
    rerender({ fp: '/data/fileB.h5oina' });
    await waitFor(() => expect(edsApi.elements).toHaveBeenCalled());
  });

  it('handles elements returned as plain strings or {name} objects', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al Kα1', { name: 'Fe Kα1' }, { element: 'Cu Kα1' }] } });
    h5Api.getElectronList.mockResolvedValue({ data: { images: [] } });
    ebsdApi.bandContrast.mockResolvedValue({ data: { image: 'b64', label: 'native', shape: [128, 156] } });
    const { result } = renderHook(() => useDefaultLayers(true));
    await waitFor(() => expect(result.current.ready).toBe(true));
    const elIds = result.current.layers.filter(l => l.kind === 'eds-element').map(l => l.id);
    expect(elIds).toEqual(['eds-Al Kα1', 'eds-Fe Kα1', 'eds-Cu Kα1']);
  });
});
