// @vitest-environment jsdom
/**
 * Converting a CIF without DOI or reference, end to end through the page.
 *
 * The dialog has its own tests; what only this file can show is that the
 * page opens it, sends what the user chose to the right endpoint, and shows
 * the translated warning. The API module is mocked at the axios boundary of
 * `dbApi`, so a rename of a field on either side fails here.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

const api = vi.hoisted(() => ({
  browse: vi.fn(),
  batchCifInfo: vi.fn(),
  cifInfo: vi.fn(),
  xtalInfo: vi.fn(),
  syncStatus: vi.fn(),
  convertCifToXtal: vi.fn(),
  updateCifMeta: vi.fn(),
  symmetriseCif: vi.fn(),
  deleteFiles: vi.fn(),
}));
vi.mock('../../services/api', () => ({ dbApi: api }));

import i18n from '../../i18n';
import CrystalDatabasePage from './CrystalDatabasePage';

const FILES = {
  'NoRef.cif': { reference_missing: true },
  'NoRef2.cif': { reference_missing: true },
  'Cited.cif': { reference_missing: false, doi: '10.1000/cited' },
};

function info(name, extra = {}) {
  return {
    filename: name,
    phase_name: name.replace('.cif', ''),
    space_group: 'Fm-3m',
    space_group_number: 225,
    n_atoms: 4,
    fit_for_xtal: true,
    fit_warnings: [],
    fit_warning_codes: [],
    doi: '',
    reference: '',
    reference_missing: true,
    ...FILES[name],
    ...extra,
  };
}

function boot(names = Object.keys(FILES), detail = {}) {
  api.browse.mockImplementation(async (cat) => ({
    data: { entries: cat === 'cif' ? names.map(n => ({ name: n, path: n, has_xtal: false, location: 'local' })) : [] },
  }));
  // What the list's batch parse really returns: no fit verdict, no warning codes.
  const batchLike = (n) => {
    const { fit_for_xtal, fit_warnings, fit_warning_codes, ...rest } = info(n);   // eslint-disable-line no-unused-vars
    return rest;
  };
  api.batchCifInfo.mockImplementation(async (list) => ({ data: { results: list.map(batchLike) } }));
  api.cifInfo.mockImplementation(async (name) => ({ data: info(name, detail[name] || {}) }));
  api.syncStatus.mockResolvedValue({ data: { available: false } });
  api.xtalInfo.mockResolvedValue({ data: {} });
  api.convertCifToXtal.mockImplementation(async (path, _out, declaration) => ({
    data: {
      name: path.replace('.cif', '.xtal'), space_group: 225, n_atoms: 4,
      reference_status: declaration === 'unpublished' ? 'declared_unpublished' : 'found',
    },
  }));
  api.updateCifMeta.mockResolvedValue({ data: { updated: true } });
}

async function openPage() {
  await act(async () => { await i18n.changeLanguage('en'); });
  render(<CrystalDatabasePage isActive />);
  await screen.findByText('NoRef.cif');
  await waitFor(() => expect(api.batchCifInfo).toHaveBeenCalled());
}

async function select(name) {
  fireEvent.click(screen.getByText(name));
  await waitFor(() => expect(api.cifInfo).toHaveBeenCalledWith(name));
  await screen.findByText(/Fit for \.xtal/);
}

const convertButton = () => screen.getByRole('button', { name: 'Convert to .xtal' });

beforeEach(() => { boot(); });
afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('single file without a reference', () => {
  it('asks instead of converting or refusing', async () => {
    await openPage();
    await select('NoRef.cif');
    expect(convertButton()).toBeEnabled();                    // no longer a dead button
    fireEvent.click(convertButton());
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent('NoRef.cif has no DOI');
    expect(api.convertCifToXtal).not.toHaveBeenCalled();
  });

  it('saves an entered DOI into the CIF, then converts', async () => {
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(convertButton());
    fireEvent.change(await screen.findByLabelText('DOI or citation'), { target: { value: '10.5555/entered' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save and convert' }));
    await waitFor(() => expect(api.convertCifToXtal).toHaveBeenCalledTimes(1));
    expect(api.updateCifMeta).toHaveBeenCalledWith('NoRef.cif', { doi: '10.5555/entered' });
    expect(api.convertCifToXtal).toHaveBeenCalledWith('NoRef.cif', '', '');
    // The meta was written before the conversion was asked for.
    expect(api.updateCifMeta.mock.invocationCallOrder[0]).toBeLessThan(api.convertCifToXtal.mock.invocationCallOrder[0]);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('saves a citation as the reference text', async () => {
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(convertButton());
    fireEvent.change(await screen.findByLabelText('DOI or citation'), { target: { value: 'Smith 2020, Acta 1, 2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save and convert' }));
    await waitFor(() => expect(api.convertCifToXtal).toHaveBeenCalled());
    expect(api.updateCifMeta).toHaveBeenCalledWith('NoRef.cif', { reference: 'Smith 2020, Acta 1, 2' });
  });

  it('converts with the declaration "unpublished" and says so in the status line', async () => {
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(convertButton());
    fireEvent.click(await screen.findByRole('button', { name: 'Unpublished / own data' }));
    await waitFor(() => expect(api.convertCifToXtal).toHaveBeenCalledWith('NoRef.cif', '', 'unpublished'));
    expect(api.updateCifMeta).not.toHaveBeenCalled();
    await screen.findByText(/reference declared unpublished/);
  });

  it('cancel converts nothing and writes nothing', async () => {
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(convertButton());
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(api.convertCifToXtal).not.toHaveBeenCalled();
    expect(api.updateCifMeta).not.toHaveBeenCalled();
  });

  it('a file that has a reference converts at once', async () => {
    await openPage();
    await select('Cited.cif');
    fireEvent.click(convertButton());
    await waitFor(() => expect(api.convertCifToXtal).toHaveBeenCalledWith('Cited.cif', '', ''));
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('opens the dialog when only the backend knows the reference is missing', async () => {
    // Cached info claims a reference; the backend (422 with a code) disagrees.
    boot(undefined, { 'NoRef.cif': { reference_missing: false, doi: 'stale' } });
    api.convertCifToXtal.mockRejectedValueOnce({
      response: { status: 422, data: { detail: { code: 'reference_missing', message: 'no reference' } } },
    });
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(convertButton());
    expect(await screen.findByRole('dialog')).toHaveTextContent('NoRef.cif has no DOI');
  });

  it('shows a structured 422 as its message, not as [object Object]', async () => {
    api.convertCifToXtal.mockRejectedValueOnce({
      response: { status: 422, data: { detail: { code: 'other', message: 'the readable part' } } },
    });
    boot(undefined, { 'Cited.cif': {} });
    api.convertCifToXtal.mockRejectedValueOnce({
      response: { status: 422, data: { detail: { code: 'other', message: 'the readable part' } } },
    });
    await openPage();
    await select('Cited.cif');
    fireEvent.click(convertButton());
    await screen.findByText(/the readable part/);
  });
});

describe('batch conversion', () => {
  async function checkAll() {
    await openPage();
    const boxes = screen.getAllByRole('checkbox');
    // First checkbox may be a select-all in the list header; click every row box.
    for (const box of boxes) {
      if (box.getAttribute('aria-checked') !== 'true') fireEvent.click(box);
    }
    await screen.findByRole('button', { name: /Convert Selected \(3\)/ });
  }

  it('asks once for all files without a reference, listing them', async () => {
    await checkAll();
    fireEvent.click(screen.getByRole('button', { name: /Convert Selected \(3\)/ }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent('2 selected CIF files have no DOI');
    expect(within(dialog).getByText('NoRef.cif')).toBeInTheDocument();
    expect(within(dialog).queryByText('Cited.cif')).toBeNull();
    expect(api.convertCifToXtal).not.toHaveBeenCalled();
  });

  it('declare all: the two are declared, the cited one is converted as it is', async () => {
    await checkAll();
    fireEvent.click(screen.getByRole('button', { name: /Convert Selected \(3\)/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'Declare all 2 unpublished' }));
    await waitFor(() => expect(api.convertCifToXtal).toHaveBeenCalledTimes(3));
    const calls = Object.fromEntries(api.convertCifToXtal.mock.calls.map(([path, , decl]) => [path, decl]));
    expect(calls).toEqual({ 'NoRef.cif': 'unpublished', 'NoRef2.cif': 'unpublished', 'Cited.cif': '' });
  });

  it('skip: only the file with a reference is converted', async () => {
    await checkAll();
    fireEvent.click(screen.getByRole('button', { name: /Convert Selected \(3\)/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'Skip these 2' }));
    await waitFor(() => expect(api.convertCifToXtal).toHaveBeenCalledTimes(1));
    expect(api.convertCifToXtal).toHaveBeenCalledWith('Cited.cif', '', '');
    await screen.findByText(/2 skipped \(no reference\)/);
  });

  it('cancel converts nothing', async () => {
    await checkAll();
    fireEvent.click(screen.getByRole('button', { name: /Convert Selected \(3\)/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }));
    expect(api.convertCifToXtal).not.toHaveBeenCalled();
  });
});

describe('P 1 files with hidden symmetry', () => {
  const p1 = {
    'NoRef.cif': {
      space_group_number: 1,
      fit_warning_codes: ['p1_hidden_symmetry'],
      fit_warnings: ['Stored as space group P 1, but the atom positions have the symmetry Fm-3m (225) at 0.01 A. A symmetrised copy can be saved.'],
      detected_space_group: { number: 225, symbol: 'Fm-3m', symprec: 0.01 },
    },
  };

  it('names the detected group and saves a copy on request', async () => {
    boot(undefined, p1);
    api.symmetriseCif.mockResolvedValue({ data: { name: 'NoRef_sym.cif', space_group_number: 225, space_group_symbol: 'Fm-3m' } });
    await openPage();
    await select('NoRef.cif');
    expect(await screen.findByText(/symmetry Fm-3m \(#225\) at every tolerance from 0\.01 to 0\.1/)).toBeInTheDocument();
    const browseCalls = api.browse.mock.calls.length;
    fireEvent.click(screen.getByRole('button', { name: 'Save symmetrised copy' }));
    await waitFor(() => expect(api.symmetriseCif).toHaveBeenCalledWith('NoRef.cif', null));
    await screen.findByText(/Saved NoRef_sym.cif \(Fm-3m, #225\); cell and axes as in the original/);
    await waitFor(() => expect(api.browse.mock.calls.length).toBeGreaterThan(browseCalls));   // list reloaded
  });

  it('offers no single copy where the group depends on the tolerance, but one choice per group', async () => {
    boot(undefined, {
      'NoRef.cif': {
        space_group_number: 1,
        fit_warning_codes: ['p1_ambiguous_symmetry'],
        fit_warnings: ['x'],
        p1_ambiguous: true,
        symmetry_ladder: [
          { symprec: 0.01, number: 8, symbol: 'Cm' },
          { symprec: 0.02, number: 8, symbol: 'Cm' },
          { symprec: 0.05, number: 225, symbol: 'Fm-3m' },
          { symprec: 0.1, number: 225, symbol: 'Fm-3m' },
        ],
      },
    });
    api.symmetriseCif.mockResolvedValue({ data: { name: 'NoRef_sym.cif', space_group_number: 225, space_group_symbol: 'Fm-3m', symprec: 0.05, tolerance_chosen: true } });
    await openPage();
    await select('NoRef.cif');
    expect(await screen.findByText(/0\.01 Å: Cm \(8\); 0\.02 Å: Cm \(8\); 0\.05 Å: Fm-3m \(225\)/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Save symmetrised copy' })).toBeNull();
    expect(screen.getByRole('button', { name: /Cm \(#8, 0\.01/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Fm-3m \(#225, 0\.05/ }));
    await waitFor(() => expect(api.symmetriseCif).toHaveBeenCalledWith('NoRef.cif', 0.05));
    await screen.findByText(/tolerance 0\.05 Å chosen by you/);
  });

  it('shows the warning in German and not as the backend sentence', async () => {
    boot(undefined, p1);
    await openPage();
    await act(async () => { await i18n.changeLanguage('de'); });
    fireEvent.click(screen.getByText('NoRef.cif'));
    await screen.findByText(/Raumgruppe P 1 gespeichert/);
    expect(screen.queryByText(/Stored as space group P 1/)).toBeNull();
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('offers no copy for a file without a finding', async () => {
    await openPage();
    await select('Cited.cif');
    expect(screen.queryByRole('button', { name: 'Save symmetrised copy' })).toBeNull();
  });

  it('a refusal of the copy is reported, not swallowed', async () => {
    boot(undefined, p1);
    api.symmetriseCif.mockRejectedValue({
      response: { status: 422, data: { detail: { code: 'disordered', message: 'Partially occupied sites.' } } },
    });
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(await screen.findByRole('button', { name: 'Save symmetrised copy' }));
    await screen.findByText(/Could not save a symmetrised copy: .*partially occupied sites/);
  });
});

describe('editing the DOI of a CIF', () => {
  it('says so when what was typed is not a DOI, and shows nothing saved', async () => {
    api.updateCifMeta.mockRejectedValue({
      response: { status: 422, data: { detail: { code: 'invalid_doi', message: 'Not a DOI' } } },
    });
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(screen.getAllByTitle('Click to edit')[0]);       // the DOI row comes first
    fireEvent.change(screen.getByPlaceholderText('10.1234/example.doi'), { target: { value: 'n/a' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('not a DOI');
    expect(api.updateCifMeta).toHaveBeenCalledWith('NoRef.cif', { doi: 'n/a' });
  });
});

describe('why a symmetrised copy was refused', () => {
  const p1 = {
    'NoRef.cif': {
      space_group_number: 1,
      fit_warning_codes: ['p1_hidden_symmetry'],
      fit_warnings: ['Stored as space group P 1 ...'],
      detected_space_group: { number: 225, symbol: 'Fm-3m', symprec: 0.01 },
    },
  };

  it.each([
    ['en', 'disordered', /partially occupied sites/],
    ['de', 'disordered', /teilbesetzte Positionen/],
    ['de', 'not_writable', /nicht beschreibbar/],
    ['ja', 'changed_structure', /許容範囲/],
    ['zh', 'no_symmetry', /未发现高于 P 1 的对称性/],
  ])('%s: the code %s is shown in that language, not as the backend sentence', async (lang, code, pattern) => {
    boot(undefined, p1);
    api.symmetriseCif.mockRejectedValue({
      response: { status: 422, data: { detail: { code, message: 'ENGLISH BACKEND SENTENCE' } } },
    });
    await openPage();
    await act(async () => { await i18n.changeLanguage(lang); });
    fireEvent.click(screen.getByText('NoRef.cif'));
    await waitFor(() => expect(api.cifInfo).toHaveBeenCalled());
    const button = await screen.findByTitle(/225/);
    fireEvent.click(button);
    const line = await screen.findByText(pattern);
    expect(line.textContent).not.toContain('ENGLISH BACKEND SENTENCE');
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('falls back to the backend sentence for a code this build does not know', async () => {
    boot(undefined, p1);
    api.symmetriseCif.mockRejectedValue({
      response: { status: 422, data: { detail: { code: 'from_the_future', message: 'Newer sentence.' } } },
    });
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(await screen.findByRole('button', { name: 'Save symmetrised copy' }));
    await screen.findByText(/Newer sentence\./);
  });
});


describe('what the interface says about the cell and the copy', () => {
  const p1 = {
    'NoRef.cif': {
      space_group_number: 1,
      fit_warning_codes: ['p1_hidden_symmetry'],
      fit_warnings: ['Stored as space group P 1 ...'],
      detected_space_group: { number: 225, symbol: 'Fm-3m', symprec: 0.01 },
    },
  };

  async function save(cell, lang = 'en') {
    boot(undefined, p1);
    api.symmetriseCif.mockResolvedValue({
      data: { name: 'NoRef_sym.cif', space_group_number: 225, space_group_symbol: 'Fm-3m', cell },
    });
    await openPage();
    await act(async () => { await i18n.changeLanguage(lang); });
    fireEvent.click(screen.getByText('NoRef.cif'));
    await waitFor(() => expect(api.cifInfo).toHaveBeenCalled());
    fireEvent.click(await screen.findByTitle(/225/));
  }

  it('says the axes were kept', async () => {
    await save({ kept_source_axes: true, atoms_before: 16, atoms_after: 16, axes: '' });
    await screen.findByText(/cell and axes as in the original/);
  });

  it('says the cell was converted, with the atom counts and the relabelled axes', async () => {
    await save({ kept_source_axes: false, atoms_before: 10, atoms_after: 30, axes: 'a->b, b->c, c->a' });
    const line = await screen.findByText(/cell converted to the conventional standard setting/);
    expect(line.textContent).toContain('atoms per cell 10 → 30');
    expect(line.textContent).toContain('axes a->b, b->c, c->a (original → copy)');
  });

  it('names no axes when the cell shape changed (no relabelling describes it)', async () => {
    await save({ kept_source_axes: false, atoms_before: 1, atoms_after: 2, axes: '' });
    const line = await screen.findByText(/atoms per cell 1 → 2/);
    expect(line.textContent).not.toContain('axes');
  });

  it('says it in German too', async () => {
    await save({ kept_source_axes: false, atoms_before: 10, atoms_after: 30, axes: 'a->b, b->c, c->a' }, 'de');
    const line = await screen.findByText(/in die konventionelle Standardaufstellung umgerechnet/);
    expect(line.textContent).toContain('Atome pro Zelle 10 → 30');
    expect(line.textContent).toContain('Achsen a->b, b->c, c->a (Original → Kopie)');
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('replaces the button by a note that the copy exists, so a second click is impossible', async () => {
    await save({ kept_source_axes: true, atoms_before: 4, atoms_after: 4, axes: '' });
    expect(await screen.findByRole('status')).toHaveTextContent('A symmetrised copy exists: NoRef_sym.cif');
    expect(screen.queryByRole('button', { name: 'Save symmetrised copy' })).toBeNull();
    expect(api.symmetriseCif).toHaveBeenCalledTimes(1);
  });

  it('a refusal because the same crystal is already there names that file and also ends the button', async () => {
    boot(undefined, p1);
    api.symmetriseCif.mockRejectedValue({
      response: { status: 422, data: { detail: { code: 'same_content_exists', message: 'x', existing: 'Al7.cif' } } },
    });
    await openPage();
    fireEvent.click(screen.getByText('NoRef.cif'));
    await waitFor(() => expect(api.cifInfo).toHaveBeenCalled());
    fireEvent.click(await screen.findByTitle(/225/));
    await screen.findByText(/already in this folder as Al7\.cif/);
    expect(await screen.findByRole('status')).toHaveTextContent('A symmetrised copy exists: Al7.cif');
    expect(screen.queryByRole('button', { name: 'Save symmetrised copy' })).toBeNull();
  });
});

describe('length of a typed citation in the parameters panel', () => {
  it('caps both rows and says so when the backend refuses a text as too long', async () => {
    api.updateCifMeta.mockRejectedValue({
      response: { status: 422, data: { detail: { code: 'too_long', message: 'x' } } },
    });
    await openPage();
    await select('NoRef.cif');
    fireEvent.click(screen.getAllByTitle('Click to edit')[0]);
    expect(screen.getByPlaceholderText('10.1234/example.doi')).toHaveAttribute('maxlength', '200');
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    fireEvent.click(screen.getAllByTitle('Click to edit')[1]);
    const box = screen.getByPlaceholderText('Author et al., Journal, Year');
    expect(box).toHaveAttribute('maxlength', '2000');
    fireEvent.change(box, { target: { value: 'y'.repeat(50) } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('too long (at most 2000 characters)');
  });
});
