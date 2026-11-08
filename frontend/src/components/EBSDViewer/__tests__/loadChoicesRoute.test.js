// Where a picked EBSD path leads. The backend (GET /api/ebsd/inspect) decides; routeLoad only
// reads its answer. The `choices` answer is the backend's own: __fixtures__/inspectChoices.json is
// held against the real route by tests/test_inspect_choices.py::test_the_fixture_matches_the_backend,
// so it cannot go stale.
import { describe, it, expect, vi } from 'vitest';
import choices from '../__fixtures__/inspectChoices.json';
import { routeLoad as fromApexLoad } from '../apexLoad';
import { routeLoad, isInspectable } from '../loadChoices';

const answering = (data) => vi.fn(async () => ({ data }));

describe('the fixture is the backend answer for a file with two sources', () => {
  it('is a choices answer with three variants (not an empty stand-in)', () => {
    expect(choices.kind).toBe('choices');
    expect(choices.n_variants).toBe(3);
    expect(choices.measurements.map((m) => m.sources.length)).toEqual([2, 1]);
    expect(choices.default_variant).toEqual({ measurement: '1', pattern_source: 'processed' });
  });
});

describe('isInspectable', () => {
  it('asks about the HDF5 names of EBSD files, in any case and with blanks around', () => {
    for (const p of ['a.h5', 'C:\\d\\a.HDF5', ' x.h5 ', '/d/a.h5oina', '/d/A.H5OINA', '/d/a.h5ebsd']) {
      expect(isInspectable(p), p).toBe(true);
    }
  });

  it('does not ask about any other name', () => {
    for (const p of ['a.up1', 'a.up2', 'Pattern.dat', 'Setting.txt', 'a.ebsp', 'a.ang', 'a.h5.txt', '', null, undefined]) {
      expect(isInspectable(p), String(p)).toBe(false);
    }
  });
});

describe('routeLoad: a file that offers a choice', () => {
  it('routes an .h5oina with two sources to the dialog, with the whole answer', async () => {
    const ask = answering(choices);
    const route = await routeLoad(' /d/two.h5oina ', ask);
    expect(route.action).toBe('dialog');
    expect(route.kind).toBe('choices');
    expect(route.inspection).toBe(choices);
    expect(route.dialog).toEqual({ path: '/d/two.h5oina', inspection: choices });
    expect(ask).toHaveBeenCalledTimes(1);
    expect(ask).toHaveBeenCalledWith('/d/two.h5oina');
  });

  it('routes a Bruker-style .h5 with several scans to the dialog as well', async () => {
    const route = await routeLoad('/d/b.h5', answering({ ...choices, format: 'bruker' }));
    expect(route.action).toBe('dialog');
    expect(route.kind).toBe('choices');
  });

  it('loads the ordinary way when a choices answer lists no measurement', async () => {
    expect(await routeLoad('/d/x.h5oina', answering({ kind: 'choices' }))).toEqual({ action: 'load' });
    expect(await routeLoad('/d/x.h5oina', answering({ kind: 'choices', measurements: [] }))).toEqual({ action: 'load' });
  });
});

describe('routeLoad: a file with one variant', () => {
  it('loads a single-variant file directly: one question, no dialog', async () => {
    const ask = answering({ kind: 'plain' });
    const route = await routeLoad('/d/one.h5oina', ask);
    expect(route).toEqual({ action: 'load' });
    expect(ask).toHaveBeenCalledTimes(1);
  });

  it('asks nothing about a name that cannot be a choice', async () => {
    const ask = answering(choices);
    for (const p of ['/d/a.up1', '/d/Pattern.dat', '/d/Setting.txt', '/d/a.ebsp']) {
      expect(await routeLoad(p, ask), p).toEqual({ action: 'load' });
    }
    expect(ask).not.toHaveBeenCalled();
  });

  it('loads the ordinary way when the question fails, so the load reports its own error', async () => {
    const fail = vi.fn(async () => { throw new Error('400 loadInspectFailed'); });
    expect(await routeLoad('/d/damaged.h5oina', fail)).toEqual({ action: 'load' });
    expect(fail).toHaveBeenCalledTimes(1);
  });
});

describe('routeLoad: the other kinds of answer keep their route', () => {
  it('opens the APEX dialog for an APEX metadata file, marked as such', async () => {
    const answer = { kind: 'apex', measurements: [{ group: 'g' }] };
    expect(await routeLoad(' /d/s.h5 ', answering(answer))).toEqual({
      action: 'dialog', kind: 'apex', inspection: answer, dialog: { h5Path: '/d/s.h5', inspection: answer } });
  });

  it('says why a file cannot be read', async () => {
    expect(await routeLoad('/d/a.h5oina', answering({ kind: 'unreadable', problem: 'apexFileUnreadable',
      problem_detail: 'a.h5oina is held open' }))).toEqual({
      action: 'unreadable', problem: 'apexFileUnreadable', detail: 'a.h5oina is held open' });
  });

  it('is the very function apexLoad.js has always exported', () => {
    expect(fromApexLoad).toBe(routeLoad);
  });
});
