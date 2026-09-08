import { describe, it, expect, vi } from 'vitest';
import {
  batchFileName, planBatchFiles, runImageBatch, makeFolderWriter, batchCaption,
} from './batchImageExport';

describe('batchFileName', () => {
  it('keeps the map label but not its path separators', () => {
    // A real layer label from the EDS page.
    expect(batchFileName('Scan1', 'SE/Elektronenbild 33 (Input1)', 'png'))
      .toBe('Scan1_SE_Elektronenbild 33 (Input1).png');
  });

  it('drops the characters Windows refuses', () => {
    expect(batchFileName('s', 'a:b*c?d"e<f>g|h', 'png')).toBe('s_a_b_c_d_e_f_g_h.png');
  });

  it('never produces a hidden file or a traversal', () => {
    expect(batchFileName('..', '..', 'png')).toBe('export.png');
    expect(batchFileName('s', '.hidden', 'png')).toBe('s_hidden.png');
  });

  it('falls back to a usable stem rather than an empty name', () => {
    expect(batchFileName('', '', 'png')).toBe('export.png');
  });
});

describe('planBatchFiles', () => {
  it('names one file per map, in order', () => {
    const items = [{ label: 'Al' }, { label: 'Si' }, { label: 'Fe' }];
    expect(planBatchFiles(items, { stem: 'Scan1', ext: 'png' }).map((p) => p.filename))
      .toEqual(['Scan1_Al.png', 'Scan1_Si.png', 'Scan1_Fe.png']);
  });

  it('breaks a collision instead of overwriting the first file', () => {
    const items = [{ label: 'Al' }, { label: 'Al' }, { label: 'Al' }];
    expect(planBatchFiles(items, { stem: 's', ext: 'png' }).map((p) => p.filename))
      .toEqual(['s_Al.png', 's_Al (2).png', 's_Al (3).png']);
  });

  it('treats names differing only in case as a collision', () => {
    // Windows would: writing both would leave one file, not two.
    const items = [{ label: 'al' }, { label: 'AL' }];
    const names = planBatchFiles(items, { stem: 's', ext: 'png' }).map((p) => p.filename);
    expect(new Set(names.map((n) => n.toLowerCase())).size).toBe(2);
  });
});

describe('runImageBatch', () => {
  const plan = [
    { item: { label: 'Al' }, filename: 'a.png' },
    { item: { label: 'Si' }, filename: 'b.png' },
    { item: { label: 'Fe' }, filename: 'c.png' },
  ];

  it('writes every picture and reports what it wrote', async () => {
    const write = vi.fn();
    const res = await runImageBatch({ plan, render: async () => 'blob', write });
    expect(res.written).toEqual(['a.png', 'b.png', 'c.png']);
    expect(res.failed).toEqual([]);
    expect(write.mock.calls.map((c) => c[1])).toEqual(['a.png', 'b.png', 'c.png']);
  });

  it('one map that cannot be built does not cost the others', async () => {
    const render = vi.fn(async (item) => {
      if (item.label === 'Si') throw new Error('bitmap not ready');
      return 'blob';
    });
    const res = await runImageBatch({ plan, render, write: vi.fn() });
    expect(res.written).toEqual(['a.png', 'c.png']);
    expect(res.failed).toEqual([{ filename: 'b.png', message: 'bitmap not ready' }]);
  });

  it('a render that hands back nothing is a failure, not a silent skip', async () => {
    const res = await runImageBatch({ plan, render: async () => null, write: vi.fn() });
    expect(res.written).toEqual([]);
    expect(res.failed).toHaveLength(3);
    expect(res.failed[0].message).toMatch(/nothing to write/);
  });

  it('stops when cancelled, and says it stopped', async () => {
    let n = 0;
    const res = await runImageBatch({
      plan,
      render: async () => { n += 1; return 'blob'; },
      write: vi.fn(),
      isCancelled: () => n >= 2,
    });
    expect(res.cancelled).toBe(true);
    expect(res.written).toEqual(['a.png', 'b.png']);
  });

  it('renders one at a time rather than all at once', async () => {
    let live = 0;
    let peak = 0;
    await runImageBatch({
      plan,
      render: async () => {
        live += 1; peak = Math.max(peak, live);
        await Promise.resolve();
        live -= 1;
        return 'blob';
      },
      write: vi.fn(),
    });
    expect(peak).toBe(1);
  });

  it('reports progress for each picture and a final done', async () => {
    const seen = [];
    await runImageBatch({
      plan, render: async () => 'blob', write: vi.fn(),
      onProgress: (p) => seen.push(p.done ? 'done' : p.filename),
    });
    expect(seen).toEqual(['a.png', 'b.png', 'c.png', 'done']);
  });
});

describe('makeFolderWriter', () => {
  const blob = { arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer };

  it('asks for the folder once and then writes into it', async () => {
    const api = {
      openFolder: vi.fn(async () => 'D:/figures'),
      writeImageInFolder: vi.fn(async (o) => `${o.dir}/${o.name}`),
    };
    const w = makeFolderWriter(api);
    expect(w.kind).toBe('folder');
    await w.pick();
    await w.write(blob, 'a.png');
    await w.write(blob, 'b.png');

    expect(api.openFolder).toHaveBeenCalledTimes(1);
    expect(api.writeImageInFolder.mock.calls.map((c) => c[0].name)).toEqual(['a.png', 'b.png']);
    expect(api.writeImageInFolder.mock.calls.every((c) => c[0].dir === 'D:/figures')).toBe(true);
  });

  it('will not write before a folder has been chosen', async () => {
    const w = makeFolderWriter({ openFolder: vi.fn(), writeImageInFolder: vi.fn() });
    await expect(w.write(blob, 'a.png')).rejects.toThrow(/no folder chosen/);
  });

  it('a cancelled folder pick leaves nothing to write into', async () => {
    const api = { openFolder: vi.fn(async () => null), writeImageInFolder: vi.fn() };
    const w = makeFolderWriter(api);
    expect(await w.pick()).toBe(null);
    await expect(w.write(blob, 'a.png')).rejects.toThrow(/no folder chosen/);
    expect(api.writeImageInFolder).not.toHaveBeenCalled();
  });

  it('falls back to downloads outside Electron', () => {
    expect(makeFolderWriter(null).kind).toBe('download');
    // A half-present API is not an API: writing would throw per file.
    expect(makeFolderWriter({ openFolder: () => {} }).kind).toBe('download');
  });
});

describe('batchCaption', () => {
  it('captions each file with its own map by default', () => {
    expect(batchCaption({ label: 'Fe Ka1' }, { perMap: true })).toBe('Fe Ka1');
    expect(batchCaption({ label: 'Cu La1,2' }, { perMap: true })).toBe('Cu La1,2');
  });

  it("leaves the dialog’s own text alone when the box is off", () => {
    // undefined, not null or '': buildSpec falls back to the typed text, and
    // an empty string would draw a plate with nothing on it.
    expect(batchCaption({ label: 'Fe Ka1' }, { perMap: false })).toBeUndefined();
  });

  it('does not invent a caption for a map that has no name', () => {
    expect(batchCaption({}, { perMap: true })).toBeUndefined();
  });
});
