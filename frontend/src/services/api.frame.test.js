import { describe, it, expect, vi } from 'vitest';

vi.mock('axios', () => {
  const inst = { get: vi.fn(() => Promise.resolve({ data: {} })),
                 put: vi.fn(() => Promise.resolve({ data: {} })),
                 post: vi.fn(() => Promise.resolve({ data: {} })) };
  // `__inst` is exposed both as a sibling named export and on the default
  // object so `import axios from 'axios'` (as api.js does) can reach it.
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import { frameApi, poleFigureApi, stateApi } from './api';
import axios from 'axios';

describe('frame/pole-figure api', () => {
  it('GET /api/frame', async () => {
    await frameApi.get();
    expect(axios.__inst.get).toHaveBeenCalledWith('/api/frame');
  });
  it('PUT /api/frame', async () => {
    await frameApi.set({ rotation: { mode: 'preset', preset: 'identity' } });
    expect(axios.__inst.put).toHaveBeenCalledWith('/api/frame', { spec: expect.any(Object) });
  });
  it('GET /api/pole-figure with params', async () => {
    await poleFigureApi.image({ phaseId: 0, hkl: '100,110,111', subsample: 20000 });
    const call = axios.__inst.get.mock.calls.find(c => c[0] === '/api/pole-figure');
    expect(call[1].params).toMatchObject({ phase_id: 0, hkl: '100,110,111' });
  });
  it('GET /api/state-version', async () => {
    await stateApi.version();
    expect(axios.__inst.get).toHaveBeenCalledWith('/api/state-version');
  });
});
