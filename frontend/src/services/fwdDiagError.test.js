import { describe, it, expect, vi, beforeEach } from 'vitest';
import { forwardDiagApi } from './api';

describe('forwardDiagApi errors carry the code of a coded refusal', () => {
  beforeEach(() => vi.restoreAllMocks());
  it('reads a string detail and the X-Orienta-Code header', async () => {
    globalThis.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 400,
      headers: { get: (k) => (k.toLowerCase() === 'x-orienta-code' ? 'pc_missing' : null) },
      json: () => Promise.resolve({ detail: 'carries no pattern centre' }),
    }));
    await expect(forwardDiagApi.compute('r1', 128)).rejects.toMatchObject({
      message: 'carries no pattern centre', code: 'pc_missing' });
  });
  it('still reads the dict detail of the older refusals, with no code', async () => {
    globalThis.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 409, headers: { get: () => null },
      json: () => Promise.resolve({ detail: { error: 'sht metadata empty' } }),
    }));
    await expect(forwardDiagApi.compute('r1', 128)).rejects.toMatchObject({
      message: 'sht metadata empty', code: null });
  });
});
