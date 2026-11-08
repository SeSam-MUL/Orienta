/**
 * The reference declaration and the symmetrised-copy request have to reach the
 * backend under the names it reads.
 *
 * The page tests mock `dbApi`, and the backend tests call the routes directly:
 * either half passes while a renamed field in between does nothing. This pins
 * the WIRE: the real axios URL and body.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const post = vi.fn(() => Promise.resolve({ data: {} }));
  const inst = {
    post, get: vi.fn(), put: vi.fn(), delete: vi.fn(), patch: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
    defaults: { headers: { common: {} } },
  };
  return { default: { create: () => inst, ...inst } };
});

const load = async () => {
  const mod = await import('./api');
  const axios = (await import('axios')).default;
  return { dbApi: mod.dbApi, post: axios.post };
};

describe('CIF conversion wire', () => {
  beforeEach(() => vi.resetModules());

  it('sends the declaration as reference_declaration', async () => {
    const { dbApi, post } = await load();
    post.mockClear();
    await dbApi.convertCifToXtal('Zr.cif', '', 'unpublished');
    const [url, body] = post.mock.calls.at(-1);
    expect(url).toBe('/api/database/convert-cif-to-xtal');
    expect(body).toEqual({ cif_path: 'Zr.cif', output_dir: '', reference_declaration: 'unpublished' });
  });

  it('sends an empty declaration when the caller declares nothing', async () => {
    const { dbApi, post } = await load();
    post.mockClear();
    await dbApi.convertCifToXtal('Zr.cif');
    expect(post.mock.calls.at(-1)[1].reference_declaration).toBe('');
  });

  it('asks for the symmetrised copy of the named file', async () => {
    const { dbApi, post } = await load();
    post.mockClear();
    await dbApi.symmetriseCif('Zr P1.cif');
    expect(post.mock.calls.at(-1)[0]).toBe('/api/database/cif/Zr%20P1.cif/symmetrise');
  });

  it('sends the chosen tolerance as symprec, and nothing when none was chosen', async () => {
    const { dbApi, post } = await load();
    post.mockClear();
    await dbApi.symmetriseCif('Al.cif', 0.05);
    expect(post.mock.calls.at(-1)[1]).toEqual({ symprec: 0.05 });
    await dbApi.symmetriseCif('Al.cif');
    expect(post.mock.calls.at(-1)[1]).toBeUndefined();
  });
});
