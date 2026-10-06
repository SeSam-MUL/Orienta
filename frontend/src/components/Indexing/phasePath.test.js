import { describe, it, expect, vi } from 'vitest';
import {
  samePath, findByPath, mergeUserAdded, pathErrorFrom, phaseStem, addPhaseFromPath, cleanPastedPath,
} from './phasePath';

describe('samePath', () => {
  it('ignores slash direction and, for drive-letter paths, case', () => {
    expect(samePath('C:\\Data\\Al.cif', 'c:/data/al.cif')).toBe(true);
  });
  it('is case-sensitive for POSIX paths', () => {
    expect(samePath('/home/u/Al.cif', '/home/u/al.cif')).toBe(false);
    expect(samePath('/home/u/Al.cif', '/home/u/Al.cif')).toBe(true);
  });
  it('treats empty and missing as not the same', () => {
    expect(samePath('', '')).toBe(false);
    expect(samePath(null, 'x')).toBe(false);
  });
});

describe('findByPath', () => {
  const files = [{ path: 'C:\\lib\\Al.cif' }, { path: 'C:\\lib\\Si.cif' }];
  it('finds the library entry for the same file', () => {
    expect(findByPath(files, 'c:/lib/si.cif')).toBe(files[1]);
  });
  it('returns undefined when absent', () => {
    expect(findByPath(files, 'C:\\other\\Ni.cif')).toBeUndefined();
  });
});

describe('mergeUserAdded', () => {
  it('appends user-added files that the listing does not have', () => {
    const lib = [{ path: '/lib/Al.cif' }];
    const extra = [{ path: '/mine/X.cif', user_added: true }];
    expect(mergeUserAdded(lib, extra)).toEqual([...lib, ...extra]);
  });
  it('does not duplicate a file the library also lists', () => {
    const lib = [{ path: '/lib/Al.cif' }];
    expect(mergeUserAdded(lib, [{ path: '/lib/Al.cif', user_added: true }])).toEqual(lib);
  });
  it('returns the same array object when there is nothing to add', () => {
    const lib = [{ path: '/lib/Al.cif' }];
    expect(mergeUserAdded(lib, [])).toBe(lib);
  });
});

describe('pathErrorFrom', () => {
  it('passes a structured server detail through', () => {
    const detail = { code: 'not_found', message: 'No such file: x', params: { path: 'x' } };
    expect(pathErrorFrom({ response: { data: { detail } } })).toEqual(detail);
  });
  it('wraps a plain-string detail as a generic error', () => {
    expect(pathErrorFrom({ response: { data: { detail: 'boom' } } }))
      .toEqual({ code: 'generic', message: 'boom', params: {} });
  });
  it('falls back to the network error message', () => {
    expect(pathErrorFrom(new Error('Network Error')))
      .toEqual({ code: 'generic', message: 'Network Error', params: {} });
  });
});

describe('phaseStem', () => {
  it('is the file name without folder and .cif', () => {
    expect(phaseStem('C:\\a\\b\\duplex ferrite.CIF')).toBe('duplex ferrite');
    expect(phaseStem('/a/b/Al.cif')).toBe('Al');
    expect(phaseStem('')).toBe('');
  });
});


describe('addPhaseFromPath', () => {
  const rec = { path: '/mine/X.cif', filename: 'X.cif', formula: 'X', user_added: true };
  const mk = (over = {}) => {
    const select = vi.fn();
    const remember = vi.fn();
    const state = { method: 'hough', discoveredFiles: [], phaseFiles: [], select, ...over };
    return {
      select, remember,
      args: {
        method: 'hough',
        rawPath: '/mine/X.cif',
        check: vi.fn().mockResolvedValue({ data: { file: rec } }),
        latest: () => state,
        remember,
      },
    };
  };

  it('selects a good file and remembers it, since the library does not list it', async () => {
    const { args, select, remember } = mk();
    expect(await addPhaseFromPath(args)).toEqual({ ok: true, name: 'X' });
    expect(select).toHaveBeenCalledWith(rec);
    expect(remember).toHaveBeenCalledWith(rec);
    expect(args.check).toHaveBeenCalledWith('hough', '/mine/X.cif');
  });

  it('uses the library entry when the file is already listed, and remembers nothing', async () => {
    const lib = { path: '/MINE/X.cif'.replace('/MINE', '/mine'), filename: 'X.cif', formula: 'X (lib)' };
    const { args, select, remember } = mk({ discoveredFiles: [lib] });
    await addPhaseFromPath(args);
    expect(select).toHaveBeenCalledWith(lib);
    expect(remember).not.toHaveBeenCalled();
  });

  it('does not select a phase that is already selected', async () => {
    const { args, select } = mk({ phaseFiles: ['/mine/X.cif'] });
    expect(await addPhaseFromPath(args)).toEqual({ ok: true, already: true, name: 'X' });
    expect(select).not.toHaveBeenCalled();
  });

  it('returns the server refusal as an error and changes nothing', async () => {
    const { args, select, remember } = mk();
    const detail = { code: 'not_found', message: 'No such file', params: { path: '/mine/X.cif' } };
    args.check = vi.fn().mockRejectedValue({ response: { data: { detail } } });
    expect(await addPhaseFromPath(args)).toEqual({ ok: false, error: detail });
    expect(select).not.toHaveBeenCalled();
    expect(remember).not.toHaveBeenCalled();
  });

  it('refuses another FILE with the name of one already selected, and names it', async () => {
    // A phase (and a reflector selection) is identified by its file name.
    const { args, select, remember } = mk({ phaseFiles: ['/lib/X.cif'] });
    const res = await addPhaseFromPath(args);
    expect(res.ok).toBe(false);
    expect(res.error.code).toBe('phase_same_name');
    expect(res.error.params).toEqual({ name: 'X', loaded_path: '/lib/X.cif', path: '/mine/X.cif' });
    expect(select).not.toHaveBeenCalled();
    expect(remember).not.toHaveBeenCalled();
  });

  it('the name is compared without regard to case', async () => {
    const { args } = mk({ phaseFiles: ['/lib/x.cif'] });
    expect((await addPhaseFromPath(args)).error.code).toBe('phase_same_name');
  });

  it('drops the file when the method was switched while it was being checked', async () => {
    const { args, select, remember } = mk({ method: 'spherical' });
    const res = await addPhaseFromPath(args);       // started as 'hough'
    expect(res.ok).toBe(false);
    expect(select).not.toHaveBeenCalled();
    expect(remember).not.toHaveBeenCalled();
  });

  it('reads the page state AFTER the check, not before', async () => {
    const { args, select } = mk();
    let state = { method: 'hough', discoveredFiles: [], phaseFiles: [], select };
    args.latest = () => state;
    args.check = vi.fn().mockImplementation(async () => {
      state = { ...state, phaseFiles: ['/mine/X.cif'] };    // selected meanwhile
      return { data: { file: rec } };
    });
    expect((await addPhaseFromPath(args)).already).toBe(true);
  });
});

describe('cleanPastedPath', () => {
  const p = 'C:\\data\\Al.cif';
  it.each([
    ['straight double', `"${p}"`], ['straight single', `'${p}'`],
    ['typographic double', `“${p}”`], ['typographic single', `‘${p}’`],
    ['German', `„${p}“`], ['guillemets', `«${p}»`],
    ['quotes and spaces', `  “ ${p} ”  `], ['nested', `"“${p}”"`],
  ])('%s quotes are stripped', (_n, raw) => {
    expect(cleanPastedPath(raw)).toBe(p);
  });
  it('leaves a path alone, including an apostrophe inside it', () => {
    expect(cleanPastedPath(p)).toBe(p);
    expect(cleanPastedPath("/home/o'brien/Al.cif")).toBe("/home/o'brien/Al.cif");
  });
  it('does not strip a quote that has no partner', () => {
    expect(cleanPastedPath('"/x/Al.cif')).toBe('"/x/Al.cif');
  });
});
