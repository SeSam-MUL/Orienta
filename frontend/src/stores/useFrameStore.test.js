// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../services/api', () => ({
  frameApi: {
    get: vi.fn(() => Promise.resolve({ data: { source_file: 'f.h5', spec: { rotation: { mode: 'preset', preset: 'identity' } }, frame_sig: 'aaa' } })),
    set: vi.fn((spec) => Promise.resolve({ data: { source_file: 'f.h5', spec, frame_sig: 'bbb', version: 2 } })),
  },
}));

import useFrameStore from './useFrameStore';
import { frameApi } from '../services/api';

beforeEach(() => { localStorage.clear(); useFrameStore.setState({ spec: null, frameSig: null, loaded: false, sourceFile: null }); });

describe('useFrameStore', () => {
  it('loadForFile pulls from backend', async () => {
    await useFrameStore.getState().loadForFile('f.h5');
    expect(frameApi.get).toHaveBeenCalled();
    expect(useFrameStore.getState().frameSig).toBe('aaa');
    expect(useFrameStore.getState().loaded).toBe(true);
  });

  it('setSpec PUTs and persists to localStorage', async () => {
    await useFrameStore.getState().loadForFile('f.h5');
    await useFrameStore.getState().setSpec({ rotation: { mode: 'axis_angle', axis: [0, 0, 1], angle_deg: 90 } });
    expect(frameApi.set).toHaveBeenCalled();
    expect(useFrameStore.getState().frameSig).toBe('bbb');
    expect(localStorage.getItem('frameSpec::f.h5')).toContain('axis_angle');
  });
});
