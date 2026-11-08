// @vitest-environment node
/**
 * Copy-PC sends the FILES (paths) as well as their names: two files can share a stem, and the
 * backend writes into the entry registered for the file whose path it is given. The real
 * client is used, only axios is replaced, so the body that goes out is what is read.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const inst = {
    get: vi.fn(() => Promise.resolve({ data: {} })),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import { batchV2Api } from '../../services/api';

beforeEach(() => axios.__inst.post.mockClear());

describe('batchV2Api.copyPC', () => {
  it('puts the source and target file paths on the wire beside the names', async () => {
    await batchV2Api.copyPC('Scan5.up1', ['Scan5.up1'], 'C:/data/S/Scan5.up1', ['C:/data/B/Scan5.up1']);
    const [url, body] = axios.__inst.post.mock.calls[0];
    expect(url).toBe('/api/batch-v2/copy-pc');
    expect(body).toEqual({
      source_dataset: 'Scan5.up1', target_datasets: ['Scan5.up1'],
      source_file_path: 'C:/data/S/Scan5.up1', target_file_paths: ['C:/data/B/Scan5.up1'],
    });
  });

  it('still works with names alone (older callers)', async () => {
    await batchV2Api.copyPC('A', ['B']);
    const [, body] = axios.__inst.post.mock.calls[0];
    expect(body.source_file_path).toBeNull();
    expect(body.target_file_paths).toBeNull();
  });
});

describe('BatchFileTree', () => {
  it('hands the files\' paths to both copy-PC actions', async () => {
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const src = readFileSync(fileURLToPath(new URL('./BatchFileTree.jsx', import.meta.url)), 'utf8');
    const calls = src.split('batchV2Api.copyPC(').slice(1);
    expect(calls).toHaveLength(2);                                    // "Copy PC" and "Apply all"
    for (const c of calls) {
      const args = c.slice(0, c.indexOf(');'));
      expect(args).toContain('activeFile.file_path');
      expect(args).toContain('checkedFiles.map((f) => f.file_path)');
    }
  });
});
