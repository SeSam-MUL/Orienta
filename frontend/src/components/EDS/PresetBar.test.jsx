// @vitest-environment jsdom
/**
 * The preset bar, pinned on the compatibility gate — which IS the feature.
 *
 * A preset that applies silently to a scan it does not fit produces a
 * plausible map from settings that mean something else, and nothing on screen
 * says so. So: check first, refuse on `ok: false`, keep the blockers on
 * screen as a panel (a toast is a warning nobody read), and make an override
 * a second deliberate click whose report travels into the export.
 *
 * `tolerance` is verified inert and is excluded from what a preset carries.
 * A stored tolerance would have the user tune a control, see no change, and
 * conclude the preset was broken.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

const listPresets = vi.fn();
const getPreset = vi.fn();
const checkPreset = vi.fn();
const savePreset = vi.fn();
const deletePreset = vi.fn();
const importPreset = vi.fn();
const exportPresetFile = vi.fn();
vi.mock('../../services/api', () => ({
  edsExportApi: {
    listPresets: (...a) => listPresets(...a),
    getPreset: (...a) => getPreset(...a),
    checkPreset: (...a) => checkPreset(...a),
    savePreset: (...a) => savePreset(...a),
    deletePreset: (...a) => deletePreset(...a),
    importPreset: (...a) => importPreset(...a),
    exportPresetFile: (...a) => exportPresetFile(...a),
  },
}));

import useDataStore from '../../stores/useDataStore';
import PresetBar, {
  applyPresetSettings, presetSettingsFrom, shortHash, saveTextFile,
  matchesPresetQuery, filterPresets, groupPresetsByMaterial, collectTags,
  collectMaterials, parseTags, NO_MATERIAL_CLASS,
} from './PresetBar';

/** The bar reads the loaded file from the same store the rest of the page does. */
const loadFile = (path) => act(() => { useDataStore.setState({ filePath: path }); });

const META = {
  name: 'Al matrix',
  author: 'S. Leroch',
  created: '2026-08-27',
  content_hash: 'deadbeefcafef00d',
  builtin: false,
  // Saved, stored and returned by the API since the feature shipped - and
  // rendered nowhere until the group-leader persona said what she actually
  // needed: not a hash, but "is this a steel recipe?".
  material_class: 'AA6061',
  tags: ['rolled', 'qa'],
  notes: 'For extruded AA6061 with coarse Fe-bearing intermetallics.',
};

const SETTINGS = {
  mode: 'cluster',
  scale: 5,
  n_clusters: 8,
  n_clusters_pinned: true,
  min_score: 0.35,
  element_weights: { Si: 3 },
  region_defs: [{ name: 'Si rich', elements: [{ element: 'Si', min_at_pct: 21 }] }],
  rules: { rules: [], matrix_elements: ['Al'] },
  phase_keys: ['al', 'si'],
  cluster_remainder: false,
  tolerance: 15,
};

function fakeHandle() {
  return {
    mode: 'pixel', scale: null, nClusters: null, minScore: 0.3,
    elementWeights: {}, regionDefs: [], rules: null,
    selectedPhaseKeys: new Set(['al']), clusterRemainder: true,
    setMode: vi.fn(), setScale: vi.fn(), setNClusters: vi.fn(),
    setMinScore: vi.fn(), setElementWeights: vi.fn(), setRegionDefs: vi.fn(),
    setRules: vi.fn(), setSelectedPhaseKeys: vi.fn(),
    setClusterRemainder: vi.fn(), setTolerance: vi.fn(),
    // Smoothing as a length, and what the scan it was authored on measured.
    scaleUnit: 'px', scaleUm: null,
    setScaleUnit: vi.fn(), setScaleUm: vi.fn(),
    authoredElements: ['Al', 'Si', 'Fe'], authoredStepUm: 0.5,
  };
}

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  useDataStore.setState({ filePath: null });
  listPresets.mockResolvedValue({ data: { presets: [META] } });
  getPreset.mockResolvedValue({ data: { preset: { ...META, settings: SETTINGS } } });
  checkPreset.mockResolvedValue({ data: { ok: true, blockers: [], warnings: [] } });
});
afterEach(() => { delete window.electronAPI; });

async function mount(over = {}) {
  const handle = over.handle || fakeHandle();
  const onApplied = vi.fn();
  render(<PresetBar handle={handle} onApplied={onApplied} {...over} />);
  await waitFor(() => expect(listPresets).toHaveBeenCalled());
  await waitFor(() => expect(screen.getByLabelText('presets.title')).toBeTruthy());
  return { handle, onApplied };
}

/** The picker is a list, not a <select>: open it, then click an entry. */
const openPicker = () => {
  if (!document.querySelector('[data-preset-list]')) {
    fireEvent.click(screen.getByLabelText('presets.title'));
  }
};

const select = (name) => {
  openPicker();
  fireEvent.click(document.querySelector(`[data-preset-option="${name}"]`));
};

const optionNames = () => [...document.querySelectorAll('[data-preset-option]')]
  .map((el) => el.getAttribute('data-preset-option'))
  .filter(Boolean);

describe('what a preset carries', () => {
  it('never carries tolerance — it is inert', () => {
    const s = presetSettingsFrom({ ...fakeHandle(), tolerance: 15 });
    expect('tolerance' in s).toBe(false);
    expect(Object.keys(s).some((k) => /toler/i.test(k))).toBe(false);
  });

  it('records whether the cluster count was PINNED, not just its value', () => {
    expect(presetSettingsFrom({ nClusters: null })).toMatchObject({
      n_clusters: null, n_clusters_pinned: false,
    });
    expect(presetSettingsFrom({ nClusters: 8 })).toMatchObject({
      n_clusters: 8, n_clusters_pinned: true,
    });
  });

  it('writes the phase list out explicitly rather than "all"', () => {
    const s = presetSettingsFrom({ selectedPhaseKeys: new Set(['a', 'b']) });
    expect(s.phase_keys.sort()).toEqual(['a', 'b']);
  });

  // A pixel count is not portable. "5" is a 2.5 um box at a 0.5 um step and
  // a 3.75 um box at a 0.75 um step — the same recipe, a 50 % different
  // analysis. Carrying the LENGTH is what makes one recipe mean the same
  // thing on two scans, which is the entire reason presets exist.
  it('carries the smoothing width as a length when the user set one', () => {
    const s = presetSettingsFrom({ ...fakeHandle(), scaleUnit: 'um', scaleUm: 2.5 });
    expect(s.scale_um).toBe(2.5);
  });

  it('carries no length while the user is working in pixels, even with a stale um value', () => {
    const s = presetSettingsFrom({ ...fakeHandle(), scaleUnit: 'px', scaleUm: 2.5 });
    // Sending both would let the backend's precedence rule silently pick the
    // one the user was not looking at.
    expect(s.scale_um).toBe(null);
  });
});

describe('applying a preset to the controls', () => {
  it('fills every control the phase map owns', () => {
    const h = fakeHandle();
    const done = applyPresetSettings(SETTINGS, h);
    expect(h.setMode).toHaveBeenCalledWith('cluster');
    expect(h.setScale).toHaveBeenCalledWith(5);
    expect(h.setNClusters).toHaveBeenCalledWith(8);
    expect(h.setMinScore).toHaveBeenCalledWith(0.35);
    expect(h.setElementWeights).toHaveBeenCalledWith({ Si: 3 });
    expect(h.setRegionDefs).toHaveBeenCalledWith(SETTINGS.region_defs);
    expect(h.setRules).toHaveBeenCalledWith(SETTINGS.rules);
    expect(h.setClusterRemainder).toHaveBeenCalledWith(false);
    expect([...h.setSelectedPhaseKeys.mock.calls[0][0]]).toEqual(['al', 'si']);
    expect(done).toContain('region_defs');
  });

  it('never touches tolerance, even when the file carries one', () => {
    const h = fakeHandle();
    applyPresetSettings(SETTINGS, h);
    expect(h.setTolerance).not.toHaveBeenCalled();
  });

  it('an unpinned cluster count comes back as auto, not as a number', () => {
    const h = fakeHandle();
    applyPresetSettings({ n_clusters: 8, n_clusters_pinned: false }, h);
    expect(h.setNClusters).toHaveBeenCalledWith(null);
  });

  it('a resolved pixel scale wins over the stored one — smoothing is a length', () => {
    const h = fakeHandle();
    applyPresetSettings({ scale: 5, scale_px: 9 }, h);
    expect(h.setScale).toHaveBeenCalledWith(9);
  });

  it('leaves absent fields alone rather than resetting them', () => {
    const h = fakeHandle();
    const done = applyPresetSettings({ mode: 'cluster' }, h);
    expect(done).toEqual(['mode']);
    expect(h.setRegionDefs).not.toHaveBeenCalled();
    expect(h.setRules).not.toHaveBeenCalled();
    // A preset written before the field existed must not silently re-declare
    // the unit the user is currently working in.
    expect(h.setScaleUnit).not.toHaveBeenCalled();
    expect(h.setScaleUm).not.toHaveBeenCalled();
  });

  it('round-trips a physical smoothing width — number AND unit', () => {
    const h = fakeHandle();
    const s = presetSettingsFrom({ ...fakeHandle(), scaleUnit: 'um', scaleUm: 2.5 });
    const done = applyPresetSettings(s, h);
    expect(h.setScaleUm).toHaveBeenCalledWith(2.5);
    // Restoring the number without the unit would send the pixel count on
    // the next run — the same defect, one step further down.
    expect(h.setScaleUnit).toHaveBeenCalledWith('um');
    expect(done).toContain('scale_um');
  });

  it('restores pixels when the preset was written in pixels', () => {
    const h = fakeHandle();
    applyPresetSettings({ scale: 5, scale_um: null }, h);
    expect(h.setScaleUnit).toHaveBeenCalledWith('px');
    expect(h.setScaleUm).not.toHaveBeenCalled();
  });
});

describe('the compatibility gate', () => {
  it('checks BEFORE it applies', async () => {
    const { handle } = await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(getPreset).toHaveBeenCalled());
    expect(checkPreset).toHaveBeenCalledWith('Al matrix');
    expect(checkPreset.mock.invocationCallOrder[0])
      .toBeLessThan(getPreset.mock.invocationCallOrder[0]);
    expect(handle.setMode).toHaveBeenCalledWith('cluster');
  });

  it('refuses on ok:false, names the blocker, and applies NOTHING', async () => {
    checkPreset.mockResolvedValue({
      data: {
        ok: false,
        // `missing_element` is the code the backend actually emits for this
        // (backend/api/services/eds_presets.py, check_compatibility). An
        // invented one passed here for months because the API is mocked —
        // see presetCompatibilityCodes.test.jsx, which now refuses it.
        blockers: [{ code: 'missing_element', message: 'This scan did not measure Mn' }],
        warnings: [],
      },
    });
    const { handle, onApplied } = await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));

    await waitFor(() => {
      expect(document.querySelector('[data-preset-blockers]')).toBeTruthy();
    });
    expect(document.querySelector('[data-preset-blockers]').textContent)
      .toContain('This scan did not measure Mn');
    expect(getPreset).not.toHaveBeenCalled();
    expect(handle.setMode).not.toHaveBeenCalled();
    expect(onApplied).not.toHaveBeenCalled();

    // Still there a tick later: a panel, not a toast.
    await new Promise((r) => setTimeout(r, 30));
    expect(document.querySelector('[data-preset-blockers]')).toBeTruthy();
  });

  it('an override is a second, explicit click and is recorded', async () => {
    const report = {
      ok: false,
      blockers: [{ code: 'matrix_mismatch', message: 'Pinned matrix Al is not dominant here' }],
      warnings: [],
    };
    checkPreset.mockResolvedValue({ data: report });
    const { handle, onApplied } = await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(document.querySelector('[data-preset-blockers]')).toBeTruthy());

    fireEvent.click(screen.getByRole('button', { name: 'presets.applyAnyway' }));
    await waitFor(() => expect(getPreset).toHaveBeenCalled());
    expect(handle.setMode).toHaveBeenCalledWith('cluster');

    const handed = onApplied.mock.calls[0][0];
    expect(handed.name).toBe('Al matrix');
    // The refusal itself is what has to reach provenance.json.
    expect(handed.compatibility.ok).toBe(false);
    expect(handed.compatibility.overridden).toBe(true);
    expect(handed.compatibility.blockers[0].code).toBe('matrix_mismatch');
  });

  it('warnings show but never block', async () => {
    checkPreset.mockResolvedValue({
      data: {
        ok: true, blockers: [],
        warnings: [{ code: 'step_size_differs', message: 'Step size differs from the preset' }],
      },
    });
    const { handle } = await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(handle.setMode).toHaveBeenCalled());
    expect(document.querySelector('[data-preset-blockers]')).toBeNull();
    await waitFor(() => {
      expect(document.querySelector('[data-preset-warnings]').textContent)
        .toContain('Step size differs from the preset');
    });
  });

  it('a refusal describes ONE preset — changing the selection clears it', async () => {
    checkPreset.mockResolvedValue({
      data: {
        ok: false,
        blockers: [{ code: 'missing_phase', message: 'nope' }],
        warnings: [],
      },
    });
    await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(document.querySelector('[data-preset-blockers]')).toBeTruthy());
    select('');
    expect(document.querySelector('[data-preset-blockers]')).toBeNull();
  });

  /**
   * ...and it describes ONE SCAN. The refusal used to be cleared on a new
   * selection, on delete and on dismiss - and on nothing else. Refuse on
   * scan A, load scan B, and B still showed A's blockers; pressing "apply
   * anyway" then stamped A's refusal with B's path.
   *
   * It could never fake an `ok: true` - `doOverride` hardcodes
   * `ok: false, overridden: true` - so the liability is closed. What it CAN
   * do is understate scan B's own incompatibilities, which is the same lie
   * pointed the other way: the reader of provenance.json sees the two things
   * A failed and concludes those are the two things B failed.
   *
   * Derived against the loaded file, not cleared by an effect: no render in
   * which the stale answer is still live, and no second copy to forget.
   */
  it('a refusal describes ONE SCAN — loading another file clears it', async () => {
    checkPreset.mockResolvedValue({
      data: {
        ok: false,
        blockers: [{ code: 'missing_element', message: 'This scan did not measure Mn' }],
        warnings: [],
      },
    });
    loadFile('D:/scans/A.h5oina');
    await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(document.querySelector('[data-preset-blockers]')).toBeTruthy());

    loadFile('D:/scans/B.h5oina');

    expect(document.querySelector('[data-preset-blockers]')).toBeNull();
    // And with the panel goes the only route by which A's blockers could be
    // stamped with B's path.
    expect(screen.queryByRole('button', { name: 'presets.applyAnyway' })).toBeNull();
  });

  it('the warnings from scan A do not describe scan B either', async () => {
    checkPreset.mockResolvedValue({
      data: {
        ok: true, blockers: [],
        warnings: [{ code: 'step_size_differs', message: 'Step size differs from the preset' }],
      },
    });
    loadFile('D:/scans/A.h5oina');
    const { handle } = await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(handle.setMode).toHaveBeenCalled());
    await waitFor(() => expect(document.querySelector('[data-preset-warnings]')).toBeTruthy());

    loadFile('D:/scans/B.h5oina');
    expect(document.querySelector('[data-preset-warnings]')).toBeNull();
  });

  it('coming back to the file it was checked against shows the same refusal', async () => {
    checkPreset.mockResolvedValue({
      data: {
        ok: false,
        blockers: [{ code: 'missing_phase', message: 'The CIF library has no beta-AlFeSi' }],
        warnings: [],
      },
    });
    loadFile('D:/scans/A.h5oina');
    await mount();
    select('Al matrix');
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(document.querySelector('[data-preset-blockers]')).toBeTruthy());

    loadFile('D:/scans/B.h5oina');
    expect(document.querySelector('[data-preset-blockers]')).toBeNull();

    // A derivation hides it against the wrong scan; it does not destroy a
    // true answer about the right one.
    loadFile('D:/scans/A.h5oina');
    expect(document.querySelector('[data-preset-blockers]').textContent)
      .toContain('The CIF library has no beta-AlFeSi');
  });
});

/**
 * Findability. A service lab expects 60-100 presets after a year, and until
 * now the picker was a flat list of NAMES: `material_class` was collected on
 * save under a tooltip promising it "is what makes a preset findable once
 * there are a hundred of them", then dropped, and `tags` had no UI at all.
 */
describe('finding one preset among a hundred', () => {
  const LIB = [
    { name: 'Al matrix', material_class: 'AA6061', tags: ['rolled'], author: 'S. L.' },
    { name: 'Cast Al-Si', material_class: 'cast Al-Si', tags: ['as-cast', 'qa'] },
    { name: 'Ferrite', material_class: 'low-alloy steel', tags: ['qa'] },
    { name: 'Scratch', tags: [] },
  ];

  it('matches on name, material class, tag, author or notes', () => {
    expect(matchesPresetQuery(LIB[0], 'al')).toBe(true);
    expect(matchesPresetQuery(LIB[0], 'AA6061')).toBe(true);
    expect(matchesPresetQuery(LIB[0], 'rolled')).toBe(true);
    expect(matchesPresetQuery(LIB[2], 'steel')).toBe(true);
    expect(matchesPresetQuery(LIB[2], 'aluminium')).toBe(false);
    // Word-wise, so the user need not guess the order of the fields.
    expect(matchesPresetQuery(LIB[1], 'qa cast')).toBe(true);
  });

  it('filters by material class, with a sentinel for the unlabelled ones', () => {
    expect(filterPresets(LIB, { material: 'AA6061' }).map((p) => p.name))
      .toEqual(['Al matrix']);
    expect(filterPresets(LIB, { material: NO_MATERIAL_CLASS }).map((p) => p.name))
      .toEqual(['Scratch']);
    // '' is "all material classes", not "the ones with none".
    expect(filterPresets(LIB, { material: '' })).toHaveLength(4);
  });

  it('ANDs tags — a second tag must narrow, never widen', () => {
    expect(filterPresets(LIB, { tags: ['qa'] }).map((p) => p.name))
      .toEqual(['Cast Al-Si', 'Ferrite']);
    expect(filterPresets(LIB, { tags: ['qa', 'as-cast'] }).map((p) => p.name))
      .toEqual(['Cast Al-Si']);
  });

  it('groups by material class, unlabelled last', () => {
    const g = groupPresetsByMaterial(LIB);
    expect(g.map((x) => x.material))
      .toEqual(['AA6061', 'cast Al-Si', 'low-alloy steel', '']);
  });

  it('collects the distinct classes and tags for the two filters', () => {
    expect(collectMaterials(LIB)).toEqual(['AA6061', 'cast Al-Si', 'low-alloy steel']);
    expect(collectTags(LIB)).toEqual(['as-cast', 'qa', 'rolled']);
  });

  it('shows the material class and the tags in the list, not just the name', async () => {
    listPresets.mockResolvedValue({ data: { presets: LIB } });
    await mount();
    openPicker();
    const list = document.querySelector('[data-preset-list]');
    expect(list.textContent).toContain('AA6061');
    expect(list.textContent).toContain('low-alloy steel');
    expect(list.textContent).toContain('#rolled');
    // The ones nobody labelled are grouped and named, not silently mixed in.
    expect(list.textContent).toContain('presets.noMaterial');
  });

  it('searching narrows the list — by material class as well as by name', async () => {
    listPresets.mockResolvedValue({ data: { presets: LIB } });
    await mount();
    openPicker();
    expect(optionNames()).toHaveLength(4);
    fireEvent.change(screen.getByLabelText('presets.search'),
                     { target: { value: 'steel' } });
    expect(optionNames()).toEqual(['Ferrite']);
  });

  it('the material filter and a tag chip both narrow the list', async () => {
    listPresets.mockResolvedValue({ data: { presets: LIB } });
    await mount();
    openPicker();
    fireEvent.change(screen.getByLabelText('presets.filterMaterial'),
                     { target: { value: 'cast Al-Si' } });
    expect(optionNames()).toEqual(['Cast Al-Si']);

    fireEvent.change(screen.getByLabelText('presets.filterMaterial'),
                     { target: { value: '' } });
    fireEvent.click(document.querySelector('[data-preset-tag="qa"]'));
    expect(optionNames()).toEqual(['Cast Al-Si', 'Ferrite']);
  });

  it('says so rather than showing an empty box when nothing matches', async () => {
    listPresets.mockResolvedValue({ data: { presets: LIB } });
    await mount();
    openPicker();
    fireEvent.change(screen.getByLabelText('presets.search'),
                     { target: { value: 'titanium' } });
    expect(optionNames()).toEqual([]);
    expect(document.querySelector('[data-preset-list]').textContent)
      .toContain('presets.noMatches');
  });
});

describe('identity', () => {
  it('shows author, date and a short hash', async () => {
    await mount();
    select('Al matrix');
    const id = document.querySelector('[data-preset-identity]');
    expect(id.textContent).toContain('S. Leroch');
    expect(id.textContent).toContain('2026-08-27');
    expect(id.textContent).toContain('deadbeef');
    expect(id.textContent).not.toContain('deadbeefcafef00d');
  });

  /**
   * "It tells me Priya's name, a date and hash 81aee0db. It does not tell me
   * it is a STEEL recipe." Both fields were already fetched; neither was
   * drawn. A non-author cannot judge a preset by its checksum.
   */
  it('says what the recipe is FOR — notes and material class', async () => {
    await mount();
    select('Al matrix');
    const id = document.querySelector('[data-preset-identity]');
    expect(id.textContent).toContain('AA6061');
    expect(id.textContent)
      .toContain('For extruded AA6061 with coarse Fe-bearing intermetallics.');
    expect(id.textContent).toContain('#rolled');
  });

  it('names the gap when there are no notes and no material class', async () => {
    listPresets.mockResolvedValue({
      data: { presets: [{ ...META, notes: '', material_class: '', tags: [] }] },
    });
    await mount();
    select('Al matrix');
    const id = document.querySelector('[data-preset-identity]');
    expect(id.textContent).toContain('presets.noNotes');
    expect(id.textContent).toContain('presets.noMaterial');
  });

  it('says so when there is no author — a preset with none is a rumour', async () => {
    listPresets.mockResolvedValue({ data: { presets: [{ ...META, author: '' }] } });
    await mount();
    select('Al matrix');
    expect(document.querySelector('[data-preset-identity]').textContent)
      .toContain('presets.noAuthor');
  });

  it('shortHash keeps eight characters, and copes with nothing', () => {
    expect(shortHash('0123456789abcdef')).toBe('01234567');
    expect(shortHash('')).toBe('');
    expect(shortHash(undefined)).toBe('');
  });

  it('a built-in preset cannot be deleted', async () => {
    listPresets.mockResolvedValue({ data: { presets: [{ ...META, builtin: true }] } });
    await mount();
    select('Al matrix');
    expect(screen.getByRole('button', { name: 'presets.delete' }).disabled).toBe(true);
  });
});

describe('saving', () => {
  it('sends the current controls, minus anything bound to this scan', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    const handle = { ...fakeHandle(), mode: 'cluster', nClusters: 6, minScore: 0.4 };
    await mount({ handle });
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'My recipe' } });
    fireEvent.change(screen.getByLabelText('presets.saveAuthor'), { target: { value: 'S. L.' } });
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));

    await waitFor(() => expect(savePreset).toHaveBeenCalled());
    const sent = savePreset.mock.calls[0][0];
    expect(sent.name).toBe('My recipe');
    expect(sent.author).toBe('S. L.');
    expect(sent.settings.n_clusters).toBe(6);
    expect(sent.settings.min_score).toBe(0.4);
    expect('tolerance' in sent.settings).toBe(false);
  });

  // `check_compatibility` gates `element_set_differs` on `authored_elements`
  // and `step_size_differs` on `authored_step_um`. A preset saved without
  // them carries two warnings that can never fire — which is worse than not
  // having them, because the gate then LOOKS like it passed.
  it('records what the recipe was authored against', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'R' } });
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));

    await waitFor(() => expect(savePreset).toHaveBeenCalled());
    const sent = savePreset.mock.calls[0][0];
    expect(sent.authoredElements).toEqual(['Al', 'Si', 'Fe']);
    expect(sent.authoredStepUm).toBe(0.5);
  });

  it('passes nothing rather than something wrong when the scan cannot supply it', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    const handle = { ...fakeHandle(), authoredElements: [], authoredStepUm: null };
    await mount({ handle });
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'R' } });
    // The form says so BEFORE the save, so the user can fix it by running a
    // classification rather than wondering why the gate never fires.
    expect(document.querySelector('[data-preset-save-form]').textContent)
      .toContain('presets.authoredAgainstNothing');
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));

    await waitFor(() => expect(savePreset).toHaveBeenCalled());
    const sent = savePreset.mock.calls[0][0];
    expect(sent.authoredElements).toEqual([]);
    expect(sent.authoredStepUm).toBe(null);
  });

  it('sends the material class and the pinned matrix element the form collects', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'R' } });
    fireEvent.change(screen.getByLabelText('presets.saveMaterial'),
                     { target: { value: 'AA6061' } });
    fireEvent.change(screen.getByLabelText('presets.saveMatrix'),
                     { target: { value: 'Al' } });
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));

    await waitFor(() => expect(savePreset).toHaveBeenCalled());
    const sent = savePreset.mock.calls[0][0];
    expect(sent.materialClass).toBe('AA6061');
    // The strongest of the compatibility refusals: without it, an aluminium
    // recipe applied to a steel is not refused at all.
    expect(sent.matrixElement).toBe('Al');
  });

  it('offers only elements this scan measured as the matrix', async () => {
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));
    const opts = [...screen.getByLabelText('presets.saveMatrix').options]
      .map((o) => o.value);
    expect(opts).toEqual(['', 'Al', 'Si', 'Fe']);
  });

  it('suggests the matrix element the rules already pin', async () => {
    const handle = { ...fakeHandle(), rules: { rules: [], matrix_elements: ['Si'] } };
    await mount({ handle });
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));
    expect(screen.getByLabelText('presets.saveMatrix').value).toBe('Si');
  });
});

/**
 * Tags could be CONSUMED and not AUTHORED.
 *
 * They are shown per row, ANDed as filter chips, searched over - and the save
 * dialog collected author, notes, material class and matrix element, and no
 * tags. `savePreset` never put the field on the wire either, though the
 * backend has accepted it since the feature shipped. In a real user directory
 * nine of ten presets carry `tags: []`; at the 60-100 a service lab expects,
 * the chip row above stays empty for ever and the filter she asked for has
 * nothing to filter on.
 */
describe('authoring tags, not just reading them', () => {
  const openSaveForm = () =>
    fireEvent.click(screen.getByRole('button', { name: 'presets.saveAs' }));

  it('parses a typed line: trimmed, empties dropped, order kept', () => {
    expect(parseTags(' rolled, qa ,, customer 4711 ,'))
      .toEqual(['rolled', 'qa', 'customer 4711']);
    // The typed order is the only ordering information the user gave us.
    expect(parseTags('qa, rolled')).toEqual(['qa', 'rolled']);
    expect(parseTags('')).toEqual([]);
    expect(parseTags(null)).toEqual([]);
    expect(parseTags(' , , ')).toEqual([]);
  });

  it('de-duplicates the way the filter compares — case-insensitively', () => {
    // `filterPresets` lower-cases both sides, so "QA" and "qa" select the
    // same presets; storing both would put two chips in the row that do the
    // same thing. First spelling wins.
    expect(parseTags('QA, rolled, qa, Rolled')).toEqual(['QA', 'rolled']);
  });

  it('is free text, not a vocabulary — group, project and instrument names', () => {
    expect(parseTags('6xxx extrusion, customer 4711, Symmetry S2'))
      .toEqual(['6xxx extrusion', 'customer 4711', 'Symmetry S2']);
  });

  it('sends the tags the form collects', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    await mount();
    openSaveForm();
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'R' } });
    fireEvent.change(screen.getByLabelText('presets.saveTags'),
                     { target: { value: ' rolled, qa , , rolled ' } });
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));

    await waitFor(() => expect(savePreset).toHaveBeenCalled());
    expect(savePreset.mock.calls[0][0].tags).toEqual(['rolled', 'qa']);
  });

  it('sends an empty list when the user typed none — never a phantom tag', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    await mount();
    openSaveForm();
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'R' } });
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));
    await waitFor(() => expect(savePreset).toHaveBeenCalled());
    expect(savePreset.mock.calls[0][0].tags).toEqual([]);
  });

  it('offers the tags the library already uses, without closing the list', async () => {
    listPresets.mockResolvedValue({
      data: {
        presets: [
          { ...META, tags: ['rolled', 'qa'] },
          { name: 'Cast Al-Si', tags: ['as-cast', 'qa'] },
        ],
      },
    });
    await mount();
    openSaveForm();
    const input = screen.getByLabelText('presets.saveTags');
    const list = document.getElementById(input.getAttribute('list'));
    expect([...list.querySelectorAll('option')].map((o) => o.value))
      .toEqual(['as-cast', 'qa', 'rolled']);
    // A completion, not a constraint: a name no enum could anticipate still
    // goes through.
    expect(input.tagName).toBe('INPUT');
  });

  /**
   * The whole point of typing them: they come back, and the filter above -
   * which has been built and empty all along - finally has something to
   * filter on.
   */
  it('round-trips — a saved tag comes back and drives the chip filter', async () => {
    savePreset.mockResolvedValue({ data: { ok: true } });
    await mount();
    openSaveForm();
    fireEvent.change(screen.getByLabelText('presets.saveName'), { target: { value: 'R' } });
    fireEvent.change(screen.getByLabelText('presets.saveTags'),
                     { target: { value: 'customer 4711' } });

    // What the backend then returns for the re-read the save triggers.
    const saved = { ...META, name: 'R', tags: ['customer 4711'] };
    listPresets.mockResolvedValue({ data: { presets: [META, saved] } });
    fireEvent.click(screen.getByRole('button', { name: 'presets.save' }));
    await waitFor(() => expect(listPresets).toHaveBeenCalledTimes(2));

    openPicker();
    await waitFor(() => {
      expect(document.querySelector('[data-preset-tag="customer 4711"]')).toBeTruthy();
    });
    fireEvent.click(document.querySelector('[data-preset-tag="customer 4711"]'));
    expect(optionNames()).toEqual(['R']);
  });
});

describe('the file channel', () => {
  it('imports a preset file and re-reads the list', async () => {
    importPreset.mockResolvedValue({ data: { preset: { name: 'Imported' } } });
    await mount();
    const input = document.querySelector('[data-preset-import-input]');
    const file = new File(['{"name":"Imported"}'], 'p.json', { type: 'application/json' });
    // jsdom's File has no .text() in every version this repo runs on.
    file.text = () => Promise.resolve('{"name":"Imported"}');
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    fireEvent.change(input);
    await waitFor(() => expect(importPreset).toHaveBeenCalledWith('{"name":"Imported"}'));
    await waitFor(() => expect(listPresets).toHaveBeenCalledTimes(2));
  });

  it('writes a preset out through Electron when it is there', async () => {
    const saveImage = vi.fn().mockResolvedValue('D:/p.json');
    window.electronAPI = { saveImage };
    const res = await saveTextFile('{"a":1}', 'p.json');
    expect(saveImage).toHaveBeenCalled();
    expect(saveImage.mock.calls[0][0].defaultPath).toBe('p.json');
    expect(res).toEqual({ via: 'electron', path: 'D:/p.json' });
  });

  it('survives a backend with no preset store at all', async () => {
    listPresets.mockRejectedValue(new Error('404'));
    render(<PresetBar handle={fakeHandle()} />);
    await waitFor(() => expect(listPresets).toHaveBeenCalled());
    // The bar is still there; presets are an accelerator, not a dependency.
    expect(document.querySelector('[data-preset-bar]')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'presets.apply' }).disabled).toBe(true);
  });
});
