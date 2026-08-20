/**
 * The electron endpoints serve two different consumers off one URL, so the
 * scope has to travel with the request — and the default has to stay "file",
 * because the H5 cockpit calls these with no scope at all.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const inst = { get: vi.fn(() => Promise.resolve({ data: {} })),
                 put: vi.fn(() => Promise.resolve({ data: {} })),
                 post: vi.fn(() => Promise.resolve({ data: {} })) };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import { h5Api } from './api';
import axios from 'axios';

describe('h5Api electron scope', () => {
  beforeEach(() => axios.__inst.get.mockClear());

  it('sends no scope when none is asked for (the cockpit\'s call)', async () => {
    await h5Api.getElectronImage('SE/Elektronenbild 1');
    expect(axios.__inst.get).toHaveBeenCalledWith(
      '/api/h5/electron/SE/Elektronenbild 1', { params: undefined });
  });

  it('sends scope=dataset when asked (the layer stacks\' call)', async () => {
    await h5Api.getElectronImage('SE/Elektronenbild 1', 'dataset');
    expect(axios.__inst.get).toHaveBeenCalledWith(
      '/api/h5/electron/SE/Elektronenbild 1', { params: { scope: 'dataset' } });
  });

  it('carries the scope on the list endpoint too', async () => {
    await h5Api.getElectronList();
    expect(axios.__inst.get).toHaveBeenLastCalledWith(
      '/api/h5/electron/list', { params: undefined });
    await h5Api.getElectronList('dataset');
    expect(axios.__inst.get).toHaveBeenLastCalledWith(
      '/api/h5/electron/list', { params: { scope: 'dataset' } });
  });
});
