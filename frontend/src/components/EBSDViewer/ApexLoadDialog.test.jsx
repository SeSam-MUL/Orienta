// @vitest-environment jsdom
//
// The measurement dialog: one table,
// a row per measurement, a Details row under it. The numbers are made up. The texts are the English
// ones of the locale (the test setup initialises i18n in English).
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { Profiler } from 'react';
import { render, screen, cleanup, fireEvent, waitFor, within, act } from '@testing-library/react';
import ApexLoadDialog from './ApexLoadDialog';
import i18n from '../../i18n';
import HEX_LIMITS from './__fixtures__/hexLimits.json';
import EN from '../../locales/en/ebsdviewer.json';

const GA = '/Area 1/OIM Map 1';
const GB = '/Area 2/OIM Map 1';
const GC = '/Area 2/OIM Map 2';
const GUID_A = '3f2a8e10-5c44-4b8e-9a51-7d02bb1ec91e';
const GUID_B = '91d0c2f4-7e13-4f0a-8c6d-2b5e9a1f07ab';
const GUID_C = 'c7e41a09-3d8b-4e27-b6f5-90a4d17e5b12';

const file = (name, over = {}) => ({
  path: `/d/series/${name}`, name, n_patterns: 2880, pattern_shape: [128, 156], pairing: 'guid', count: 'fits',
  matches_grid: true, problem: null, names_measurement: GUID_A, names_group: GA, ...over,
});

const measurement = (over) => ({
  step_x: 0.3, step_y: 0.2598, sample_tilt: 70, elevation: 5, azimuth: 0, beam_kv: 20, has_sem: true,
  has_iq: true, phases: [{ index: 1, name: 'Phase A', formula: 'Al' }], measured_at: null,
  measured_at_kind: null, measured_at_timezone: null, measurement_problem: null, problem_detail: null,
  consistent: true, ...over,
});

// A: hexagonal, 60 x 48 = 2880, its file found by ID.  B: square, 40 x 30 = 1200, no file.
// C: hexagonal, 50 x 40 = 2000, its file holds 1856 patterns.
const A = measurement({
  group: GA, id: GUID_A, area: 'Area 1', name: 'OIM Map 1', hex: true, grid_type: 'HexGrid',
  n_rows: 60, n_cols: 48, n_points: 2880, pc: [0.51, 0.45, 0.62], loadable: true, problem: null,
  pattern_files: [file('map_a.up1')], measured_at: '2026-03-14T10:42:11', measured_at_kind: 'scan_end',
  measured_at_timezone: 'unknown',
});
const B = measurement({
  group: GB, id: GUID_B, area: 'Area 2', name: 'OIM Map 1', hex: false, grid_type: 'SqrGrid',
  n_rows: 40, n_cols: 30, n_points: 1200, step_x: 1, step_y: 1, pc: [0.53, 0.44, 0.66], loadable: false,
  problem: 'apexNoPatternFile', has_sem: false, has_iq: false, beam_kv: null, phases: [], pattern_files: [],
});
const C = measurement({
  group: GC, id: GUID_C, area: 'Area 2', name: 'OIM Map 2', hex: true, grid_type: 'HexGrid',
  n_rows: 50, n_cols: 40, n_points: 2000, pc: [0.52, 0.46, 0.63], loadable: false,
  problem: 'apexPatternCountMismatch',
  pattern_files: [file('map_c.up1', { n_patterns: 1856, count: 'too_few', matches_grid: false,
    problem: 'apexPatternCountMismatch', names_measurement: GUID_C, names_group: GC })],
});
const LOOSE = { path: '/d/series/loose.up1', name: 'loose.up1', n_patterns: 1200, pattern_shape: [128, 156],
  problem: null, names_measurement: null, names_group: null, fits: [GUID_B] };

const INSPECTION = {
  kind: 'apex', h5_path: '/d/series/session.h5', folder: '/d/series', folder_problem: null,
  measurements: [A, B, C], unmatched_pattern_files: [LOOSE],
};

let urlCounter = 0;
beforeEach(() => {
  urlCounter = 0;
  URL.createObjectURL = vi.fn(() => `blob:preview-${urlCounter++}`);
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => cleanup());

function setup(over = {}) {
  const props = {
    isOpen: true, h5Path: '/d/series/session.h5', inspection: INSPECTION,
    onClose: vi.fn(), onLoad: vi.fn(),
    choosePatternFile: vi.fn(async () => '/e/final_v3.up1'),
    chooseFolder: vi.fn(async () => '/e/more'),
    describePatternFile: vi.fn(async (m, p) => file(p.split('/').pop(), {
      path: p, n_patterns: m.n_rows * m.n_cols, pairing: 'count', names_measurement: null, names_group: null })),
    searchFolder: vi.fn(async () => ({ measurements: [A, { ...B, loadable: true, problem: null,
      pattern_files: [file('b_found.up1', { n_patterns: 1200, names_measurement: GUID_B, names_group: GB })] }, C],
    unmatched_pattern_files: [], folder: '/e/more' })),
    loadPreview: vi.fn(async () => new Blob(['x'])),
    loadHexLimits: vi.fn(async () => HEX_LIMITS),
    ...over,
  };
  const view = render(<ApexLoadDialog {...props} />);
  return { ...props, view };
}

/** Let a promise settle and React apply what it caused: a macrotask, inside act. */
const later = (fn) => act(async () => { fn(); await new Promise((r) => setTimeout(r, 10)); });
const row = (group) => screen.getByTestId(`measurement-${group}`);
const openButton = () => screen.getByRole('button', { name: /^Open \d+ measurements?$/ });
const box = (title) => screen.getByRole('checkbox', { name: `Open ${title}` });
const click = (el) => fireEvent.click(el);

describe('the table (layout B)', () => {
  it('has a title, one instruction, the folder searched and a summary line', () => {
    setup();
    const dialog = screen.getByRole('dialog');
    expect(dialog.getAttribute('aria-label')).toBe('Open measurements from session.h5');
    expect(dialog.textContent).toContain('Tick the measurements you want to open.');
    const folder = screen.getByTestId('apex-folder').textContent;
    expect(folder).toContain('Pattern files searched in:');
    expect(folder).toContain('/d/series');
    expect(folder).toContain('(the folder of session.h5)');
    expect(screen.getByTestId('apex-summary').textContent)
      .toBe('3 measurements: 1 can be opened, 1 needs a pattern file, 1 pattern file does not fit. 0 ticked.');
  });

  it('shows the facts a person decides on, in one row per measurement', () => {
    setup();
    const a = row(GA).textContent;
    expect(a).toContain('Area 1 / OIM Map 1');
    expect(a).toContain('Hexagonal grid');
    expect(a).toContain('scan ended 2026-03-14');
    expect(a).toContain('60 rows × 48 points');
    expect(a).toContain('step 0.3 µm, row spacing 0.26 µm');
    expect(a).toContain('patterns 156 × 128 px');
    expect(a).toContain('✓ map_a.up1');
    expect(a).toContain('matched by its measurement ID; 2,880 patterns, one per point');
    const b = row(GB).textContent;
    expect(b).toContain('Area 2 / OIM Map 1');
    expect(b).toContain('Square grid');
    expect(b).toContain('40 rows × 30 points');
    expect(b).toContain('step 1 µm');
    expect(b).not.toContain('row spacing');
    expect(b).not.toContain('scan ended');
    expect(b).toContain('No pattern file for this measurement in this folder.');
    expect(b).toContain('This measurement needs a pattern file with 1,200 patterns (40 rows × 30 points).');
    expect(b).toContain('If the pattern file no longer exists, export the patterns of this measurement again from the EDAX software (.up1 or .up2).');
    expect(b).not.toMatch(/\(\d+ × \d+\)/);
    expect(b).toContain('A renamed file in the same folder is found by the ID in its header. If the file was moved to another folder, choose it or search that folder.');
  });

  it('puts the reason next to a greyed box: the count with both numbers, for a file with too few patterns', () => {
    setup();
    const c = row(GC).textContent;
    expect(c).toContain('✗ This measurement needs a pattern file with 2,000 patterns (50 rows × 40 points). map_c.up1 holds 1,856 patterns.');
    expect(c).toContain('If the pattern file no longer exists, export the patterns of this measurement again from the EDAX software (.up1 or .up2).');
    expect(c).not.toMatch(/\(\d+ × \d+\)/);
    expect(c).toContain('may be incomplete (for example from an interrupted export) or belong to a smaller scan');
    expect(c).toContain('cannot tell which points the missing patterns belong to');
    expect(c).toContain('Choose file…');
  });

  it('shows the overview thumbnail as an overview image, and says when there is none', async () => {
    const { loadPreview } = setup();
    await waitFor(() => expect(screen.getByAltText('Overview image of Area 1 / OIM Map 1')).toBeTruthy());
    expect(screen.getByAltText('Overview image of Area 1 / OIM Map 1').getAttribute('title'))
      .toMatch(/reduced and contrast-stretched, not the file's original picture/);
    expect(row(GB).textContent).toContain('No overview image');
    expect(loadPreview.mock.calls.map((c) => [c[0], c[1]]).sort()).toEqual([[GUID_A, 'sem'], [GUID_C, 'sem']]);
  });
});

describe('what is ticked, and what can be ticked', () => {
  it('ticks nothing when the file holds several measurements, even when one can be opened', () => {
    setup();
    expect(box('Area 1 / OIM Map 1').checked).toBe(false);
    expect(openButton().textContent).toBe('Open 0 measurements');
    expect(openButton().disabled).toBe(true);
    expect(screen.getByTestId('apex-load-hint').textContent).toBe('Tick a measurement to enable this button.');
  });

  it('allows a tick exactly where the backend says the measurement can be opened', () => {
    setup();
    expect(box('Area 1 / OIM Map 1').disabled).toBe(false);
    expect(box('Area 2 / OIM Map 1').disabled).toBe(true);
    expect(box('Area 2 / OIM Map 2').disabled).toBe(true);
  });

  it('carries the reason on the greyed box as a sentence', () => {
    setup();
    expect(box('Area 2 / OIM Map 1').getAttribute('title')).toContain('No pattern file for this measurement in this folder.');
    expect(box('Area 2 / OIM Map 2').getAttribute('title')).toContain('holds 1,856 patterns');
    expect(box('Area 1 / OIM Map 1').getAttribute('title')).toBeNull();
  });

  it('ticks the one measurement of a file that holds exactly one, when it can be opened', () => {
    setup({ inspection: { ...INSPECTION, measurements: [A], unmatched_pattern_files: [] } });
    expect(box('Area 1 / OIM Map 1').checked).toBe(true);
    expect(openButton().textContent).toBe('Open 1 measurement');
  });

  it('ticks nothing for a single measurement that cannot be opened', () => {
    setup({ inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [] } });
    expect(box('Area 2 / OIM Map 1').checked).toBe(false);
  });

  it('opens what is ticked with the pairing it showed', () => {
    const { onLoad } = setup();
    click(box('Area 1 / OIM Map 1'));
    expect(openButton().textContent).toBe('Open 1 measurement');
    expect(screen.queryByTestId('apex-load-hint')).toBeNull();
    click(openButton());
    expect(onLoad).toHaveBeenCalledWith([{
      path: '/d/series/map_a.up1', label: 'Area 1 / OIM Map 1',
      apex: { h5_path: '/d/series/session.h5', measurement_id: GUID_A } }]);
  });
});

describe('the hexagonal line (built from the backend\'s list of limits)', () => {
  const lineOf = (g) => screen.getByTestId(`hexline-${g}`).textContent;
  const ASSUMES = 'Orienta assumes every second row is shifted by half a step; your file has no field for it.';
  const WORKS = 'Indexing works. Details below.';

  it('says in one short line that indexing works, then the assumed row shift, and nothing about saving', async () => {
    setup();
    await waitFor(() => expect(lineOf(GA)).toContain(WORKS));
    expect(lineOf(GA)).toBe(`${WORKS} ${ASSUMES}`);
    expect(lineOf(GA)).not.toMatch(/saved|saving|denoising|grain tools|batch|does not record|Zeilenversatz|sheared/);
    expect(screen.queryByTestId(`hexline-${GB}`)).toBeNull();           // a square scan has none
    expect(screen.getByTestId(`hexline-${GC}`)).toBeTruthy();
  });

  it('says only that indexing works when the backend lists no limit at all', async () => {
    setup({ loadHexLimits: vi.fn(async () => ({ ...HEX_LIMITS, categories: [], groups: {} })) });
    await waitFor(() => expect(lineOf(GA)).toContain('Indexing works. Details below.'));
    expect(lineOf(GA)).not.toContain('cannot be saved');
  });

  it('says only what is true without a list when the list cannot be loaded', async () => {
    setup({ loadHexLimits: vi.fn(async () => { throw new Error('offline'); }) });
    await waitFor(() => expect(lineOf(GA)).toContain('some other steps are not available yet'));
    expect(lineOf(GA)).not.toContain('cannot be saved');
    expect(lineOf(GA)).toContain(ASSUMES);
  });

  it('asks for the list only when a measurement is hexagonal', async () => {
    const { loadHexLimits } = setup({ inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [] } });
    await Promise.resolve();
    expect(loadHexLimits).not.toHaveBeenCalled();
  });

  it('does not say "indexing works" on a hexagonal row that cannot be opened at all', async () => {
    const stuck = { ...C, measurement_problem: 'apexUnexpectedLattice', problem: 'apexUnexpectedLattice' };
    setup({ inspection: { ...INSPECTION, measurements: [A, stuck], unmatched_pattern_files: [] } });
    await waitFor(() => expect(lineOf(GA)).toContain(WORKS));
    expect(screen.queryByTestId(`hexline-${GC}`)).toBeNull();
    expect(row(GC).textContent).not.toMatch(/Indexing works|indexing works/);
    expect(row(GC).textContent).toContain('This measurement cannot be opened: The measurement says hexagonal');
  });
});

describe('the summary has no note about saving', () => {
  it('neither with a hexagonal measurement that can be opened, nor before the list is in, nor without one', async () => {
    setup();
    expect(screen.queryByTestId('apex-summary-hex')).toBeNull();
    await later(() => {});
    expect(screen.queryByTestId('apex-summary-hex')).toBeNull();
    expect(document.body.textContent).not.toMatch(/cannot be saved|saved as a file/);
  });

  it('asks the backend for the list although only a hexagonal measurement that cannot be opened is there', async () => {
    const { loadHexLimits } = setup({ inspection: { ...INSPECTION, measurements: [B, C], unmatched_pattern_files: [] },
      describePatternFile: vi.fn() });
    await later(() => {});
    expect(screen.queryByTestId('apex-summary-hex')).toBeNull();
    expect(loadHexLimits).toHaveBeenCalled();          // C is hexagonal, but cannot be opened
  });
});

describe('Details', () => {
  it('opens a row under the measurement and closes it again', () => {
    setup();
    expect(screen.queryByTestId(`details-${GA}`)).toBeNull();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    expect(screen.getByTestId(`details-${GA}`)).toBeTruthy();
    click(within(row(GA)).getByRole('button', { name: 'Less' }));
    expect(screen.queryByTestId(`details-${GA}`)).toBeNull();
  });

  it('names the conventions, the geometry, the phases, the ID in full and the date as a date only', async () => {
    const { loadPreview } = setup();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    const d = screen.getByTestId(`details-${GA}`).textContent;
    expect(d).toContain('x* 0.5100 · y* 0.4500 · z* 0.6200');
    expect(d).toContain('EDAX (TSL) convention, as stored in the file.');
    expect(d).toContain('converts it to the Bruker convention that kikuchipy uses when opening');
    expect(d).toContain('Sample tilt 70° · camera elevation 5° · beam 20 kV');
    expect(d).toContain('Phase A');
    expect(d).toContain('you choose the phases for indexing in Orienta');
    expect(d).toContain(GUID_A);
    expect(d).toContain('Scan ended2026-03-14');
    expect(d).not.toMatch(/time zone|the date only/);
    expect(d).not.toMatch(/10:42/);                                      // no time of day
    await waitFor(() => expect(loadPreview).toHaveBeenCalledWith(GUID_A, 'iq'));
  });

  it('asks for the image-quality map only once Details is open', async () => {
    const { loadPreview } = setup();
    await Promise.resolve();
    expect(loadPreview.mock.calls.some((c) => c[1] === 'iq')).toBe(false);
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    return waitFor(() => expect(loadPreview.mock.calls.some((c) => c[1] === 'iq')).toBe(true));
  });

  const openHexDetails = async () => {
    setup();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    await waitFor(() => expect(screen.getByTestId('hex-limit-list')).toBeTruthy());
    return screen.getByTestId(`details-${GA}`);
  };

  it('starts the hexagonal paragraph with the assumption, then says what the maps look like', async () => {
    const dd = (await openHexDetails()).querySelector('dd');
    expect(dd.querySelector('p').textContent).toBe('Orienta assumes that the points of the 2nd, 4th, 6th … row sit half a step further to the right — the usual layout of EDAX hexagonal scans. Your file has no field for it.');
    const d = dd.textContent;
    expect(d).toContain('Where Orienta draws a map of this scan, it draws the hexagons as they lie on the sample: true proportions, with every second row shifted by half a step.');
    expect(d).not.toMatch(/plain grid|15 %/);
    expect(d).not.toMatch(/sheared|draws the true grid|which it measured|does not record/);
  });

  const SAVING = 'Saving: .ang and Orienta .h5 files keep the hexagonal grid as measured. A .ctf file and the square .ang option exist only on a square grid: each point takes the nearest measured point, as OIM Analysis does it, and the file says so. While the direction of the row offset is assumed (your file has no field for it), every saved file says that too.';

  it('has exactly one Saving paragraph in Details, after the one on the maps and the one saying indexing is not affected', async () => {
    const dd = (await openHexDetails()).querySelector('dd');
    const paragraphs = [...dd.querySelectorAll(':scope > p')].map((p) => p.textContent);
    expect(paragraphs.filter((p) => p === SAVING)).toHaveLength(1);
    const drawn = paragraphs.findIndex((p) => p.startsWith('Where Orienta draws a map'));
    expect(paragraphs[drawn + 1]).toBe('Indexing, orientations and patterns are not affected.');
    expect(paragraphs[drawn + 2]).toBe(SAVING);
    const d = dd.textContent;
    expect(d).not.toMatch(/cannot be saved|saved as a file/);
  });

  it('says the same, and nothing about a limit on saving, when the list cannot be loaded', async () => {
    setup({ loadHexLimits: vi.fn(async () => { throw new Error('offline'); }) });
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    const d = screen.getByTestId(`details-${GA}`).textContent;
    expect(d).toContain(SAVING);
    expect(d).not.toMatch(/cannot be saved|saved as a file/);
  });

  it('says what still works, from the backend\'s list, and that saving is not a limit', async () => {
    const d = (await openHexDetails()).textContent;
    expect(d).toContain('Indexing, orientations and patterns are not affected.');
    expect(d).toContain('Still works: image export (PNG; phase maps, viewer maps, patterns), saving results as .ang or .h5 (.ctf and square .ang are resampled), and refining one global pattern centre.');
    expect(d).not.toMatch(/saving results as a file|saving an indexing result as a file/);
  });

  it('lists each limit once, as its own list item, under a heading that does not repeat its members', async () => {
    const list = (await openHexDetails()).querySelector('[data-testid="hex-limit-list"]');
    const cats = EN.apexDialog.hexLimits.categories;
    const names = EN.apexDialog.hexLimits.keys;
    const groups = [...list.children];
    const headings = groups.map((li) => li.querySelector('b').textContent);
    // every category that HAS a limit once, in the backend's order (a category with none, as `cleanup`
    // since the phase tools work on a hexagonal scan, is not shown)
    expect(headings).toEqual(HEX_LIMITS.categories
      .filter((c) => Object.values(HEX_LIMITS.groups).flat().some((i) => i.category === c))
      .map((c) => cats[c]));
    for (const h of headings) {
      for (const n of Object.values(names)) expect(h, `${h} repeats ${n}`).not.toContain(n);
    }
    const leaves = [...list.querySelectorAll('li > ul > li')];
    const texts = leaves.map((li) => li.textContent);
    expect(new Set(texts).size).toBe(texts.length);                          // no entry twice
    const keys = [...new Set(Object.values(HEX_LIMITS.groups).flat().map((i) => i.key))];
    expect(leaves).toHaveLength(keys.length);                                // one element per limit
    for (const key of keys) expect(texts.filter((t) => t.startsWith(names[key])), key).toHaveLength(1);
    // a leaf names no other limit: nothing is run together
    for (const t of texts) expect(keys.filter((k) => t.includes(names[k])).length).toBe(1);
  });

  it('says in user terms what is limited, says "Orienta refuses this with a message" once, and keeps option names in a collapsed technical line', async () => {
    const details = await openHexDetails();
    const d = details.textContent;
    expect(d.split('Orienta refuses this with a message').length).toBe(2);
    expect(d).not.toMatch(/Orienta refuses it/);
    const list = details.querySelector('[data-testid="hex-limit-list"]');
    expect(list.textContent).toContain('all maps of the Analysis page');
    expect(list.textContent).toContain('averaging neighbouring patterns in Crystal Hint: only with some settings');
    for (const option of ['averaging radius']) {
      expect(list.textContent, option).not.toContain(option);
    }
    const tech = details.querySelector('details[data-testid="hex-technical"]');
    expect(tech).toBeTruthy();
    expect(tech.hasAttribute('open')).toBe(false);
    expect(tech.querySelector('summary').textContent).toBe('Technical detail');
    expect(tech.textContent).toContain('averaging neighbouring patterns in Crystal Hint: refused when any of these options is set: averaging radius');
    // both batches index and write a hexagonal scan: the list names no batch limit
    expect(d).not.toContain('multi-phase batch page');
    expect(d).not.toContain('auto-export of the indexing batch');
    expect(d).not.toMatch(/\([^)]*\([^)]*\)/);
  });

  it('says when the beam energy is not recorded and when the file has no image-quality map', () => {
    setup();
    click(within(row(GB)).getByRole('button', { name: 'Details' }));
    const d = screen.getByTestId(`details-${GB}`).textContent;
    expect(d).toContain('beam energy not recorded in the file');
    expect(d).toContain('The file has no image-quality map for this measurement.');
    expect(d).toContain('No phases are named in the file; you choose the phases for indexing in Orienta.');
    expect(d).not.toContain('Phases named in the file');
    expect(d).not.toContain('Hexagonal grid');
  });
});

describe('the pattern file of a row', () => {
  it('says "matched by its measurement ID" only for a pairing by ID', async () => {
    setup();
    expect(row(GA).textContent).toContain('matched by its measurement ID');
    // a file the user chose that names no measurement is paired by count, and says what was checked
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('final_v3.up1, chosen by you'));
    const b = row(GB).textContent;
    expect(b).toContain('matched by pattern count only (1,200 patterns, one per point); the file carries no measurement ID');
    expect(b).toContain('This file carries no measurement ID; only its pattern count fits. Use it only if you know it was exported from this measurement.');
    expect(b).not.toContain('matched by its measurement ID');
    expect(b).not.toContain('carries this measurement\'s ID');
  });

  it('says a chosen file that names this measurement was checked by its ID', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('named.up1', { path: p, n_patterns: 1200,
      pairing: 'guid', names_measurement: GUID_B, names_group: GB })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('checked: it carries this measurement\'s ID; 1,200 patterns, one per point'));
  });

  it('proves a chosen file before it counts: checking, then ticked and openable in the order listed', async () => {
    let finish;
    const pending = new Promise((resolve) => { finish = resolve; });
    const { describePatternFile, onLoad } = setup({ describePatternFile: vi.fn(() => pending) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('Checking final_v3.up1…'));
    expect(box('Area 2 / OIM Map 1').disabled).toBe(true);
    expect(describePatternFile.mock.calls[0][0].id).toBe(GUID_B);
    expect(describePatternFile.mock.calls[0][1]).toBe('/e/final_v3.up1');
    finish(file('final_v3.up1', { path: '/e/final_v3.up1', n_patterns: 1200, pairing: 'count',
      names_measurement: null, names_group: null }));
    await waitFor(() => expect(box('Area 2 / OIM Map 1').checked).toBe(true));
    click(box('Area 1 / OIM Map 1'));
    expect(openButton().textContent).toBe('Open 2 measurements');
    click(openButton());
    expect(onLoad.mock.calls[0][0].map((j) => [j.apex.measurement_id, j.path])).toEqual([
      [GUID_A, '/d/series/map_a.up1'], [GUID_B, '/e/final_v3.up1']]);
  });

  it('refuses a file with too few patterns with both numbers, as possibly incomplete', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('short.up1', { path: p, n_patterns: 1000, count: 'too_few',
      matches_grid: false, problem: 'apexPatternCountMismatch', names_measurement: null, names_group: null })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).toContain('This measurement needs a pattern file with 1,200 patterns (40 rows × 30 points). final_v3.up1 holds 1,000 patterns.');
    expect(alert.textContent).toContain('may be incomplete');
    expect(alert.textContent).toContain('export the patterns of this measurement again from the EDAX software (.up1 or .up2)');
    expect(alert.textContent).toContain('belong to a smaller scan');
    expect(box('Area 2 / OIM Map 1').disabled).toBe(true);
  });

  it('refuses a file with too many patterns without calling it incomplete', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('long.up1', { path: p, n_patterns: 1500, count: 'too_many',
      matches_grid: false, problem: 'apexPatternCountMismatch', names_measurement: null, names_group: null })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).toContain('This measurement needs a pattern file with 1,200 patterns (40 rows × 30 points). final_v3.up1 holds 1,500 patterns, more than this measurement has points.');
    expect(alert.textContent).not.toMatch(/incomplete/);
    expect(alert.textContent).toContain('does not cut patterns off');
  });

  it('says which measurement of the file a refused file belongs to', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('a_again.up1', { path: p, n_patterns: 2880,
      matches_grid: false, problem: 'apexGuidMismatch', count: 'fits', names_measurement: GUID_A, names_group: GA })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).toContain('final_v3.up1 belongs to Area 1 / OIM Map 1, not to this measurement.');
  });

  it('words any other code of the backend in the language of the page', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('odd.up1', { path: p, matches_grid: false,
      problem: 'apexGridMismatch', names_measurement: null, names_group: null })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).toContain('final_v3.up1: The pattern file and the metadata disagree about the grid');
  });

  it('words a coded refusal of the question itself', async () => {
    const err = { response: { data: { detail: 'English prose' }, headers: { 'x-orienta-code': 'apexMeasurementNotFound' } } };
    setup({ describePatternFile: vi.fn(async () => { throw err; }) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).toBe('That measurement is not in the metadata file.');
  });

  it('does nothing when the file dialog is cancelled', async () => {
    const { describePatternFile } = setup({ choosePatternFile: vi.fn(async () => null) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await Promise.resolve(); await Promise.resolve();
    expect(describePatternFile).not.toHaveBeenCalled();
  });

  it('ignores a slow answer about an earlier choice once the user has taken another file', async () => {
    let slowAnswer;
    const slow = new Promise((resolve) => { slowAnswer = resolve; });
    const { describePatternFile } = setup({ describePatternFile: vi.fn(() => slow) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(describePatternFile).toHaveBeenCalledTimes(1));
    click(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }));
    expect(row(GB).textContent).toContain('loose.up1, chosen by you');
    await later(() => slowAnswer(file('final_v3.up1', { path: '/e/final_v3.up1', n_patterns: 1000, count: 'too_few',
      matches_grid: false, problem: 'apexPatternCountMismatch', names_measurement: null, names_group: null })));
    expect(row(GB).textContent).toContain('loose.up1, chosen by you');   // the late refusal did not take it away
    expect(within(row(GB)).queryByRole('alert')).toBeNull();
    expect(box('Area 2 / OIM Map 1').disabled).toBe(false);
  });

  it('does not let a choice of the old file finish into the new one', async () => {
    let slowAnswer;
    const slow = new Promise((resolve) => { slowAnswer = resolve; });
    const props = {
      isOpen: true, h5Path: '/d/series/session.h5', inspection: INSPECTION, onClose: () => {}, onLoad: () => {},
      choosePatternFile: async () => '/e/final_v3.up1', chooseFolder: async () => null,
      describePatternFile: () => slow, searchFolder: async () => ({}), loadPreview: async () => new Blob(['x']),
      loadHexLimits: async () => HEX_LIMITS,
    };
    const { rerender } = render(<ApexLoadDialog {...props} />);
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('Checking final_v3.up1…'));
    rerender(<ApexLoadDialog {...props} h5Path="/d/other/session.h5" inspection={{ ...INSPECTION, measurements: [B] }} />);
    await later(() => slowAnswer(file('final_v3.up1', { path: '/e/final_v3.up1', n_patterns: 1200, pairing: 'count',
      names_measurement: null, names_group: null })));
    expect(box('Area 2 / OIM Map 1').disabled).toBe(true);
    expect(row(GB).textContent).toContain('No pattern file for this measurement in this folder.');
  });

  it('keeps a file that was found when a chosen file is refused', async () => {
    const two = { ...A, pattern_files: [file('map_a.up1'), file('map_a_copy.up1')] };
    setup({ inspection: { ...INSPECTION, measurements: [two], unmatched_pattern_files: [] },
      describePatternFile: vi.fn(async (m, p) => file('x.up1', { path: p, n_patterns: 5, count: 'too_few',
        matches_grid: false, problem: 'apexPatternCountMismatch', names_measurement: null, names_group: null })) });
    click(within(row(GA)).getByRole('button', { name: 'Choose another file…' }));
    await within(row(GA)).findByRole('alert');
    expect(row(GA).textContent).toContain('✓ map_a.up1');
    expect(box('Area 1 / OIM Map 1').disabled).toBe(false);
  });

  it('offers the files of the folder that name no measurement but fit by their pattern count', () => {
    const { onLoad } = setup();
    expect(row(GB).textContent).toContain('Pattern files in this folder that carry no measurement ID, but whose pattern count fits this measurement:');
    // the warning and what can be checked are in the row, not in a tooltip
    expect(row(GB).textContent).toContain('This file carries no measurement ID; only its pattern count fits. Use it only if you know it was exported from this measurement.');
    expect(row(GB).textContent).toContain('1,200 patterns, 156 × 128 px');
    // the button's own description says what it does; the warning is in the row, not in the tooltip
    expect(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }).getAttribute('title')).not.toMatch(/measurement ID|only its pattern count/);
    expect(within(row(GC)).queryByRole('button', { name: 'Use loose.up1' })).toBeNull();
    click(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }));
    expect(box('Area 2 / OIM Map 1').checked).toBe(true);
    expect(row(GB).textContent).toContain('loose.up1, chosen by you');
    expect(row(GB).textContent).toContain('matched by pattern count only (1,200 patterns, one per point); the file carries no measurement ID');
    expect(row(GB).textContent).toContain('Use it only if you know it was exported from this measurement.');
    expect(row(GB).textContent).toContain('patterns 156 × 128 px');
    click(openButton());
    expect(onLoad.mock.calls[0][0]).toEqual([{ path: '/d/series/loose.up1', label: 'Area 2 / OIM Map 1',
      apex: { h5_path: '/d/series/session.h5', measurement_id: GUID_B } }]);
  });

  it('shows every file that names a measurement and never picks silently among them', () => {
    const two = { ...A, pattern_files: [file('map_a.up1'), file('map_a_copy.up1')] };
    setup({ inspection: { ...INSPECTION, measurements: [two], unmatched_pattern_files: [] } });
    expect(row(GA).textContent).toContain('1 more pattern file carries this measurement\'s ID: map_a_copy.up1. The first by name is used; choose a file to use another.');
    expect(within(row(GA)).getByRole('button', { name: 'Choose another file…' })).toBeTruthy();
  });

  it('offers no choice for a row that is fine with one file', () => {
    setup();
    expect(within(row(GA)).queryByRole('button', { name: /Choose/ })).toBeNull();
  });

  it('says a folder that could not be listed is not "no pattern file"', () => {
    const noFolder = { ...B, problem: 'apexFolderUnreadable' };
    setup({ inspection: { ...INSPECTION, folder_problem: 'apexFolderUnreadable', measurements: [noFolder],
      unmatched_pattern_files: [] } });
    const b = row(GB).textContent;
    expect(b).toContain('The folder could not be listed, so no pattern file was looked for.');
    expect(b).not.toContain('No pattern file for this measurement in this folder.');
  });

  it('gives a measurement that cannot be opened whatever file it gets its reason and no choice', () => {
    const grid = { ...B, measurement_problem: 'apexGridMismatch', problem: 'apexGridMismatch' };
    setup({ inspection: { ...INSPECTION, measurements: [grid], unmatched_pattern_files: [LOOSE] } });
    const b = row(GB).textContent;
    expect(b).toContain('This measurement cannot be opened: The pattern file and the metadata disagree about the grid');
    expect(within(row(GB)).queryByRole('button', { name: /Choose/ })).toBeNull();
    expect(within(row(GB)).queryByRole('button', { name: /Use loose/ })).toBeNull();
    expect(box('Area 2 / OIM Map 1').disabled).toBe(true);
    expect(screen.getByTestId('apex-summary').textContent).toBe('1 measurement: 1 cannot be opened. 0 ticked.');
  });

  it('lists a group that could not be read without breaking the table', () => {
    const broken = { group: '/Area 3/OIM Map 1', id: null, area: 'Area 3', name: 'OIM Map 1', hex: false,
      n_rows: null, n_cols: null, step_x: null, step_y: null, pc: null, phases: [], has_sem: false, has_iq: false,
      loadable: false, problem: 'apexUnreadable', measurement_problem: 'apexUnreadable',
      problem_detail: 'OIM Map 1 of s.h5 cannot be read', pattern_files: [], consistent: null };
    setup({ inspection: { ...INSPECTION, measurements: [A, broken], unmatched_pattern_files: [] } });
    const r = screen.getByTestId('measurement-/Area 3/OIM Map 1').textContent;
    expect(r).toContain('Area 3 / OIM Map 1');
    expect(r).toContain('This measurement\'s entry in session.h5 is damaged, so it cannot be opened.');
    expect(r).not.toContain('cannot be opened: ');
    expect(r).toContain('Open session.h5 in the EDAX software and save it again, or export this measurement again. The other measurements can still be opened.');
    expect(r).not.toMatch(/Square|Hexagonal|rows/);                   // no grid chip, no size
    expect(within(screen.getByTestId('measurement-/Area 3/OIM Map 1')).getAllByText('—').length).toBeGreaterThan(0);
    // nothing to show beyond the problem: no Details toggle; the technical detail rides on the reason
    expect(within(screen.getByTestId('measurement-/Area 3/OIM Map 1')).queryByRole('button', { name: 'Details' })).toBeNull();
    expect(screen.getByTitle('OIM Map 1 of s.h5 cannot be read')).toBeTruthy();
  });
});

describe('searching another folder for all rows at once', () => {
  it('lists the new folder, fills the rows without a file, and keeps what was found and ticked', async () => {
    const { searchFolder, chooseFolder } = setup();
    click(box('Area 1 / OIM Map 1'));
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(screen.getByTestId('apex-folder').textContent).toContain('/e/more'));
    expect(chooseFolder).toHaveBeenCalledTimes(1);
    expect(searchFolder).toHaveBeenCalledWith('/e/more');
    expect(screen.getByTestId('apex-folder').textContent).toContain('(a folder you chose; rows that already have a pattern file keep it)');
    expect(row(GB).textContent).toContain('b_found.up1');
    expect(box('Area 2 / OIM Map 1').disabled).toBe(false);
    expect(box('Area 1 / OIM Map 1').checked).toBe(true);
    expect(row(GA).textContent).toContain('✓ map_a.up1');
    expect(screen.getByTestId('apex-summary').textContent)
      .toBe('3 measurements: 2 can be opened, 1 pattern file does not fit. 1 ticked.');
  });

  it('never takes away a file a row already has: the new folder\'s file is an alternative, used only on request', async () => {
    const { onLoad } = setup({ searchFolder: vi.fn(async () => ({ folder: '/e/more', unmatched_pattern_files: [],
      measurements: [{ ...A, pattern_files: [file('a_other.up1', { path: '/e/more/a_other.up1' })] }, B, C] })) });
    click(box('Area 1 / OIM Map 1'));
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(row(GA).textContent).toContain('Other files found in /e/more for this measurement:'));
    expect(row(GA).textContent).toContain('✓ map_a.up1');
    expect(box('Area 1 / OIM Map 1').checked).toBe(true);
    // the rows that had no file still have none
    expect(row(GB).textContent).toContain('No pattern file for this measurement in this folder.');
    expect(row(GC).textContent).toContain('map_c.up1 holds 1,856 patterns');
    click(openButton());
    expect(onLoad.mock.calls[0][0][0].path).toBe('/d/series/map_a.up1');
    // ... and on request the alternative replaces it, still proven by the backend's own answer
    click(within(row(GA)).getByRole('button', { name: 'Use a_other.up1' }));
    expect(row(GA).textContent).toContain('a_other.up1, chosen by you');
    expect(row(GA).textContent).toContain('checked: it carries this measurement\'s ID');
    click(openButton());
    expect(onLoad.mock.calls[1][0][0].path).toBe('/e/more/a_other.up1');
  });

  it('keeps a file the user chose, and its tick, when the folder is searched', async () => {
    setup();
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(box('Area 2 / OIM Map 1').checked).toBe(true));
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(screen.getByTestId('apex-folder').textContent).toContain('/e/more'));
    expect(row(GB).textContent).toContain('final_v3.up1, chosen by you');
    expect(box('Area 2 / OIM Map 1').checked).toBe(true);
  });

  it('keeps the old list and words the refusal when the folder cannot be listed', async () => {
    const err = { response: { data: { detail: 'prose' }, headers: { 'x-orienta-code': 'apexFolderUnreadable' } } };
    setup({ searchFolder: vi.fn(async () => { throw err; }) });
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('That folder could not be listed');
    expect(screen.getByTestId('apex-folder').textContent).toContain('(the folder of session.h5)');
    expect(row(GA).textContent).toContain('map_a.up1');
  });

  it('does nothing when the folder dialog is cancelled', async () => {
    const { searchFolder } = setup({ chooseFolder: vi.fn(async () => null) });
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await Promise.resolve(); await Promise.resolve();
    expect(searchFolder).not.toHaveBeenCalled();
  });
});

describe('the footer and the way out', () => {
  it('says in two short lines where the measurements open and how to open the others later', () => {
    setup();
    const lines = [...screen.getByRole('dialog').querySelectorAll('div')]
      .filter((d) => /^(Each ticked|Measurements you do not)/.test(d.textContent) && d.children.length === 0)
      .map((d) => d.textContent);
    expect(lines).toEqual([
      'Each ticked measurement opens here, in the EBSD Viewer, as its own dataset.',
      'Measurements you do not tick stay closed; to open them later, open session.h5 again.',
    ]);
    expect(screen.getByRole('dialog').textContent).not.toMatch(/batch/i);
  });

  it('says under the title that the EDAX results are not taken over, before the instruction', () => {
    setup();
    const note = screen.getByTestId('apex-top-note');
    expect(note.textContent).toBe('Orienta indexes the patterns itself; orientations and quality values from the EDAX software are not taken over. Your file is not changed.');
    const dialog = screen.getByRole('dialog');
    const order = [dialog.children[0], note, ...[...dialog.querySelectorAll('p')].filter((p) => /^Tick the measurements/.test(p.textContent))];
    expect(order[0].compareDocumentPosition(order[1]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(order[1].compareDocumentPosition(order[2]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(dialog.textContent.split('quality values from the EDAX software').length).toBe(2);
  });

  it('closes on Escape and on Cancel, and renders nothing when closed', () => {
    const { onClose } = setup();
    fireEvent.keyDown(window, { key: 'Escape' });
    click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledTimes(2);                     // nothing was ticked: Esc closes at once
    cleanup();
    render(<ApexLoadDialog isOpen={false} h5Path="/d/s.h5" inspection={INSPECTION} onClose={() => {}} onLoad={() => {}} />);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('starts over for another file: a new answer forgets the choices of the old one', async () => {
    const props = {
      isOpen: true, h5Path: '/d/series/session.h5', inspection: INSPECTION, onClose: () => {}, onLoad: () => {},
      choosePatternFile: async () => '/e/final_v3.up1', chooseFolder: async () => null,
      describePatternFile: async (m, p) => file('final_v3.up1', { path: p, n_patterns: 1200, pairing: 'count',
        names_measurement: null, names_group: null }),
      searchFolder: async () => ({}), loadPreview: async () => new Blob(['x']), loadHexLimits: async () => HEX_LIMITS,
    };
    const { rerender } = render(<ApexLoadDialog {...props} />);
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(box('Area 2 / OIM Map 1').checked).toBe(true));
    rerender(<ApexLoadDialog {...props} h5Path="/d/other/session.h5" inspection={{ ...INSPECTION, measurements: [B] }} />);
    expect(box('Area 2 / OIM Map 1').checked).toBe(false);
    expect(row(GB).textContent).toContain('No pattern file for this measurement in this folder.');
  });
});

// ================================================================== the table body

describe('the table body holds rows and nothing else', () => {
  const onlyElements = () => [...screen.getByRole('dialog').querySelector('tbody').childNodes]
    .every((n) => n.nodeType === 1);

  it('has no stray text between its rows, closed or with Details open', () => {
    setup();
    expect(onlyElements()).toBe(true);
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    click(within(row(GB)).getByRole('button', { name: 'Details' }));
    expect(onlyElements()).toBe(true);
    expect(screen.getByRole('dialog').querySelector('tbody').textContent).not.toMatch(/^,|,$/);
  });
});

describe('numbers are written in the language of the page', () => {
  afterEach(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });

  it('writes 2.880 patterns and 0,26 µm in German, never 2,880 or 0.26', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    setup();
    const a = row(GA).textContent;
    expect(a).toContain('60 Zeilen × 48 Punkte');
    expect(a).toContain('Schritt 0,3 µm, Zeilenabstand 0,26 µm');
    expect(a).toContain('Muster 156 × 128 px');
    expect(a).toContain('2.880 Muster');
    expect(a).not.toMatch(/2,880|0\.26/);
    expect(row(GC).textContent).toContain('map_c.up1 enthält 1.856 Muster');
    expect(row(GC).textContent).toContain('braucht eine Musterdatei mit 2.000 Mustern (50 Zeilen × 40 Punkte)');
    expect(row(GB).textContent).toContain('braucht eine Musterdatei mit 1.200 Mustern (40 Zeilen × 30 Punkte)');
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    const d = screen.getByTestId(`details-${GA}`).textContent;
    expect(d).toContain('x* 0,5100 · y* 0,4500 · z* 0,6200');
    expect(d).toContain('Beschleunigungsspannung 20 kV');
    expect(d).toContain('Musterzentrum');
    expect(screen.getByTestId('apex-summary').textContent)
      .toBe('3 Messungen: 1 lässt sich öffnen, 1 braucht eine Musterdatei, 1 Musterdatei passt nicht. 0 angekreuzt.');
  });

  it('writes 2,880 and 0.26 in English', () => {
    setup();
    expect(row(GA).textContent).toContain('2,880 patterns');
    expect(row(GA).textContent).toContain('row spacing 0.26 µm');
  });

  it('writes the same numbers in Chinese and Japanese with a point and a comma for thousands', async () => {
    await act(async () => { await i18n.changeLanguage('zh'); });
    setup();
    expect(row(GA).textContent).toContain('2,880');
    expect(row(GA).textContent).toContain('0.26');
    cleanup();
    await act(async () => { await i18n.changeLanguage('ja'); });
    setup();
    expect(row(GA).textContent).toContain('2,880');
  });
});

describe('the accessible description of a tick', () => {
  it('points at an element that exists, with an id that has no space or slash in it', () => {
    setup();
    for (const checkbox of screen.getAllByRole('checkbox')) {
      const id = checkbox.getAttribute('aria-describedby');
      expect(id).toMatch(/^[A-Za-z0-9_-]+$/);
      const target = document.getElementById(id);
      expect(target).toBeTruthy();
      expect(target.textContent.length).toBeGreaterThan(5);
      // the description is the pattern-file cell of the SAME row
      expect(checkbox.closest('tr').contains(target)).toBe(true);
    }
  });

  it('gives every row its own description', () => {
    setup();
    const ids = screen.getAllByRole('checkbox').map((c) => c.getAttribute('aria-describedby'));
    expect(new Set(ids).size).toBe(3);
  });
});

describe('Escape', () => {
  const esc = (init = {}) => fireEvent.keyDown(window, { key: 'Escape', ...init });

  it('closes at once when nothing is ticked or chosen', () => {
    const { onClose } = setup();
    esc();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('is left to the prompt that is open on top of the dialog', () => {
    const { onClose } = setup({ escapeBlocked: true });
    esc();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('is left to whoever already handled it', () => {
    const { onClose } = setup();
    const event = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    event.preventDefault();
    window.dispatchEvent(event);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('asks before throwing away a tick: the first Esc says so, the second closes', () => {
    const { onClose } = setup();
    click(box('Area 1 / OIM Map 1'));
    esc();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('status').textContent).toBe(
      'Your ticks are kept. Press Esc again to close without opening anything, or use Cancel.');
    esc();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('forgets the question when the choices change', () => {
    const { onClose } = setup();
    click(box('Area 1 / OIM Map 1'));
    esc();
    click(box('Area 1 / OIM Map 1'));                          // untick: nothing left to lose
    expect(screen.queryByRole('status')).toBeNull();
    esc();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('asks before throwing away a file the user chose', async () => {
    const { onClose } = setup();
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('chosen by you'));
    click(box('Area 2 / OIM Map 1'));                           // untick, the chosen file is still there
    esc();
    expect(onClose).not.toHaveBeenCalled();
    esc();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Cancel closes without asking', () => {
    const { onClose } = setup();
    click(box('Area 1 / OIM Map 1'));
    click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('a new file is shown from its first render', () => {
  it('never draws the previous file\'s rows with the new file\'s inspection', () => {
    const props = {
      isOpen: true, h5Path: '/d/series/session.h5', inspection: INSPECTION, onClose: () => {}, onLoad: () => {},
      choosePatternFile: async () => null, chooseFolder: async () => null, describePatternFile: async () => ({}),
      searchFolder: async () => ({}), loadPreview: async () => new Blob(['x']), loadHexLimits: async () => HEX_LIMITS,
    };
    let host;
    const seen = [];
    const spy = () => { if (host) seen.push(host.textContent); };
    const view = (p) => (
      <Profiler id="d" onRender={spy}>
        <ApexLoadDialog {...p} />
      </Profiler>
    );
    const { container, rerender } = render(view(props));
    host = container;
    expect(container.textContent).toContain('Area 1 / OIM Map 1');
    seen.length = 0;
    rerender(view({ ...props, h5Path: '/d/other/session.h5', inspection: {
      ...INSPECTION, measurements: [B], unmatched_pattern_files: [] } }));
    expect(seen.length).toBeGreaterThan(0);
    for (const text of seen) {
      expect(text).not.toContain('Area 1 / OIM Map 1');          // no flash of the previous file
      expect(text).not.toMatch(/0 measurements[:,]/);
    }
    expect(container.textContent).toContain('Open measurements from session.h5');
  });

  it('shows no summary at all for a listing without measurements', () => {
    setup({ inspection: { ...INSPECTION, measurements: [], unmatched_pattern_files: [] } });
    expect(screen.queryByTestId('apex-summary')).toBeNull();
  });
});

describe('a loose file can be taken for one row at a time', () => {
  const B2 = { ...B, group: '/Area 2/OIM Map 3', id: 'guid-b2', name: 'OIM Map 3' };
  const both = { ...LOOSE, fits: [GUID_B, 'guid-b2'] };
  const inspection = { ...INSPECTION, measurements: [B, B2], unmatched_pattern_files: [both] };

  it('tells the second row that the file is already used, with the name of the first row', () => {
    const { onLoad } = setup({ inspection });
    click(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }));
    const second = row('/Area 2/OIM Map 3');
    expect(within(second).queryByRole('button', { name: 'Use loose.up1' })).toBeNull();
    expect(second.textContent).toContain('loose.up1: already used for Area 2 / OIM Map 1.');
    expect(second.textContent).toContain('choose it with Choose file…; Orienta checks it again');
    // only one job carries the file
    click(openButton());
    expect(onLoad.mock.calls[0][0].map((j) => j.path)).toEqual(['/d/series/loose.up1']);
  });

  it('can still be chosen by hand for the second row, with the check', async () => {
    const { describePatternFile } = setup({
      inspection, choosePatternFile: vi.fn(async () => '/d/series/loose.up1') });
    click(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }));
    click(within(row('/Area 2/OIM Map 3')).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(describePatternFile).toHaveBeenCalledTimes(1));
    expect(describePatternFile.mock.calls[0][0].id).toBe('guid-b2');
  });

  it('offers it again to the first row once the second took another', () => {
    setup({ inspection });
    expect(within(row(GB)).getByRole('button', { name: 'Use loose.up1' })).toBeTruthy();
    expect(within(row('/Area 2/OIM Map 3')).getByRole('button', { name: 'Use loose.up1' })).toBeTruthy();
  });
});

describe('answers that arrive after the dialog moved on are ignored', () => {
  const props = (over = {}) => ({
    isOpen: true, h5Path: '/d/series/session.h5', inspection: INSPECTION, onClose: () => {}, onLoad: () => {},
    choosePatternFile: async () => '/e/final_v3.up1', chooseFolder: async () => '/e/more',
    describePatternFile: async () => ({}), searchFolder: async () => ({ measurements: [], folder: '/e/more' }),
    loadPreview: async () => new Blob(['x']), loadHexLimits: async () => HEX_LIMITS, ...over,
  });
  const NEXT = { h5Path: '/d/other/session.h5', inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [] } };
  const deferred = () => { let resolve; let reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

  it('a folder search that finishes after another file was opened changes nothing', async () => {
    const d = deferred();
    const p = props({ searchFolder: vi.fn(() => d.promise) });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(p.searchFolder).toHaveBeenCalled());
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    await later(() => d.resolve({ folder: '/e/more', unmatched_pattern_files: [], measurements: [A, B, C] }));
    expect(screen.getByTestId('apex-folder').textContent).toContain('(the folder of session.h5)');
    expect(screen.getByTestId('apex-folder').textContent).toContain('/d/series');
    expect(screen.queryByTestId(`measurement-${GA}`)).toBeNull();
    expect(screen.getByRole('button', { name: 'Search another folder…' }).disabled).toBe(false);
  });

  it('a folder search that fails after another file was opened shows no error on the new file', async () => {
    const d = deferred();
    const p = props({ searchFolder: vi.fn(() => d.promise) });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(p.searchFolder).toHaveBeenCalled());
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    await later(() => d.reject({ response: { headers: { 'x-orienta-code': 'apexFolderUnreadable' }, data: {} } }));
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('a file picked in the file dialog after another file was opened is not checked for the new file', async () => {
    const d = deferred();
    const p = props({ choosePatternFile: vi.fn(() => d.promise), describePatternFile: vi.fn() });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(p.choosePatternFile).toHaveBeenCalled());
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    await later(() => d.resolve('/e/final_v3.up1'));
    expect(p.describePatternFile).not.toHaveBeenCalled();
    expect(row(GB).textContent).not.toContain('Checking');
  });

  it('a picture that arrives after its measurement left is never turned into an address', async () => {
    const d = deferred();
    const p = props({ loadPreview: vi.fn((id) => (id === GUID_A ? d.promise : Promise.resolve(new Blob(['x'])))) });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    await waitFor(() => expect(p.loadPreview).toHaveBeenCalledWith(GUID_A, 'sem'));
    URL.createObjectURL.mockClear();
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    await later(() => d.resolve(new Blob(['late'])));
    expect(URL.createObjectURL).not.toHaveBeenCalledWith(expect.objectContaining({ size: 4 }));
  });

  it('a list of limits that arrives after the dialog closed is not shown when it opens again', async () => {
    const first = deferred();
    const second = deferred();
    const loadHexLimits = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const p = props({ loadHexLimits });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    await waitFor(() => expect(loadHexLimits).toHaveBeenCalledTimes(1));
    rerender(<ApexLoadDialog {...p} isOpen={false} />);
    await later(() => first.resolve(HEX_LIMITS));                     // the stale answer
    rerender(<ApexLoadDialog {...p} isOpen />);
    await waitFor(() => expect(loadHexLimits).toHaveBeenCalledTimes(2));
    expect(screen.getByTestId(`hexline-${GA}`).textContent).toContain('some other steps are not available');
    await later(() => second.resolve(HEX_LIMITS));
    expect(screen.getByTestId(`hexline-${GA}`).textContent).toContain('Indexing works. Details below.');
  });
});

describe('answers that arrive late: the guards that stand alone', () => {
  const props = (over = {}) => ({
    isOpen: true, h5Path: '/d/series/session.h5', inspection: INSPECTION, onClose: () => {}, onLoad: () => {},
    choosePatternFile: async () => '/e/final_v3.up1', chooseFolder: async () => '/e/more',
    describePatternFile: async () => ({}), searchFolder: async () => ({ measurements: [], folder: '/e/more' }),
    loadPreview: async () => new Blob(['x']), loadHexLimits: async () => HEX_LIMITS, ...over,
  });
  const NEXT = { h5Path: '/d/other/session.h5', inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [] } };
  const deferred = () => { let resolve; let reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

  it('a folder picked in the folder dialog after another file was opened is not searched', async () => {
    const d = deferred();
    const p = props({ chooseFolder: vi.fn(() => d.promise), searchFolder: vi.fn() });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(p.chooseFolder).toHaveBeenCalled());
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    await later(() => d.resolve('/e/more'));
    expect(p.searchFolder).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Search another folder…' }).disabled).toBe(false);
  });

  it('a stale search that ends does not stop the "Searching…" of the search the new file started', async () => {
    const stale = deferred();
    const current = deferred();
    const searchFolder = vi.fn().mockReturnValueOnce(stale.promise).mockReturnValueOnce(current.promise);
    const p = props({ searchFolder });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(searchFolder).toHaveBeenCalledTimes(1));
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    click(screen.getByRole('button', { name: 'Search another folder…' }));
    await waitFor(() => expect(searchFolder).toHaveBeenCalledTimes(2));
    expect(screen.getByRole('button', { name: 'Searching…' }).disabled).toBe(true);
    await later(() => stale.resolve({ measurements: [], unmatched_pattern_files: [], folder: '/e/old' }));
    expect(screen.getByRole('button', { name: 'Searching…' }).disabled).toBe(true);
    await later(() => current.resolve({ measurements: [], unmatched_pattern_files: [], folder: '/e/new' }));
    expect(screen.getByRole('button', { name: 'Search another folder…' }).disabled).toBe(false);
  });

  it('a chosen file that is refused after another file was opened shows no error on the new file', async () => {
    const d = deferred();
    const p = props({ describePatternFile: vi.fn(() => d.promise) });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(p.describePatternFile).toHaveBeenCalled());
    rerender(<ApexLoadDialog {...p} {...NEXT} />);
    await later(() => d.reject({ response: { headers: { 'x-orienta-code': 'apexBadPath' }, data: {} } }));
    expect(row(GB).textContent).not.toContain('cannot be part of a path');
    expect(within(row(GB)).queryByRole('alert')).toBeNull();
  });

  it('a file being checked blocks a second choice for the same row until the answer is in', async () => {
    const d = deferred();
    const p = props({ describePatternFile: vi.fn(() => d.promise) });
    render(<ApexLoadDialog {...p} />);
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('Checking final_v3.up1'));
    expect(within(row(GB)).getByRole('button', { name: 'Choose file…' }).disabled).toBe(true);
    await later(() => d.resolve({ ...file('final_v3.up1'), path: '/e/final_v3.up1', n_patterns: 1200,
      pairing: 'count', names_measurement: null, names_group: null }));
    expect(within(row(GB)).getByRole('button', { name: 'Choose another file…' }).disabled).toBe(false);
  });

  it('the picture of the previous file\'s measurement is not shown for a measurement that reuses its group', async () => {
    const wait = deferred();
    const A2 = { ...A, id: 'guid-of-another-session' };
    const p = props({ loadPreview: vi.fn((id) => (id === 'guid-of-another-session' ? wait.promise : Promise.resolve(new Blob(['x'])))) });
    const { container, rerender } = render(<ApexLoadDialog {...p} />);
    await waitFor(() => expect(container.querySelector('img')).toBeTruthy());
    rerender(<ApexLoadDialog {...p} h5Path="/d/other/session.h5"
      inspection={{ ...INSPECTION, measurements: [A2], unmatched_pattern_files: [] }} />);
    await waitFor(() => expect(p.loadPreview).toHaveBeenCalledWith('guid-of-another-session', 'sem'));
    expect(container.querySelector('img')).toBeNull();
    expect(within(row(GA)).getByRole('img', { name: 'Overview image of Area 1 / OIM Map 1' }).textContent).toBe('…');
  });

  it('a picture that fails for an earlier request does not take away the one a later request delivered', async () => {
    const first = deferred();
    const p = props({ loadPreview: vi.fn(() => first.promise) });
    const { container, rerender } = render(<ApexLoadDialog {...p} />);
    await waitFor(() => expect(p.loadPreview).toHaveBeenCalled());
    rerender(<ApexLoadDialog {...p} loadPreview={vi.fn(async () => new Blob(['x']))} />);
    await waitFor(() => expect(container.querySelectorAll('img').length).toBe(2));
    await later(() => first.reject(new Error('late')));
    expect(container.querySelectorAll('img').length).toBe(2);
    expect(container.textContent).not.toContain('No preview');
  });

  it('a list of limits that fails for a dialog that was closed does not take away the one of the reopened dialog', async () => {
    const first = deferred();
    const second = deferred();
    const loadHexLimits = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const p = props({ loadHexLimits });
    const { rerender } = render(<ApexLoadDialog {...p} />);
    await waitFor(() => expect(loadHexLimits).toHaveBeenCalledTimes(1));
    rerender(<ApexLoadDialog {...p} isOpen={false} />);
    rerender(<ApexLoadDialog {...p} isOpen />);
    await waitFor(() => expect(loadHexLimits).toHaveBeenCalledTimes(2));
    await later(() => second.resolve(HEX_LIMITS));
    expect(screen.getByTestId(`hexline-${GA}`).textContent).toContain('Indexing works. Details below.');
    await later(() => first.reject(new Error('late')));
    expect(screen.getByTestId(`hexline-${GA}`).textContent).toContain('Indexing works. Details below.');
  });
});

describe('a step is written in the language of the page, also without a row spacing', () => {
  afterEach(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });

  it('writes "Schritt 1,5 µm" in German and "step 1.5 µm" in English for a square scan', async () => {
    const wide = { ...B, step_x: 1.5, step_y: 1.5 };
    setup({ inspection: { ...INSPECTION, measurements: [wide], unmatched_pattern_files: [] } });
    expect(row(GB).textContent).toContain('step 1.5 µm');
    cleanup();
    await act(async () => { await i18n.changeLanguage('de'); });
    setup({ inspection: { ...INSPECTION, measurements: [wide], unmatched_pattern_files: [] } });
    expect(row(GB).textContent).toContain('Schritt 1,5 µm');
    expect(row(GB).textContent).not.toContain('1.5');
  });
});

// ================================================================== what to do next for a damaged entry or a file that does not fit

describe('the damaged entry, the missing file and the file that does not fit each say what to do next', () => {
  const broken = { group: '/Area 3/OIM Map 1', id: null, area: 'Area 3', name: 'OIM Map 1', hex: false,
    n_rows: null, n_cols: null, step_x: null, step_y: null, pc: null, phases: [], has_sem: false, has_iq: false,
    loadable: false, problem: 'apexBadId', measurement_problem: 'apexBadId', problem_detail: 'Id is not a GUID',
    pattern_files: [], consistent: null };
  const NEXT = 'Open session.h5 in the EDAX software and save it again, or export this measurement again. The other measurements can still be opened.';
  const REEXPORT = 'If the pattern file no longer exists, export the patterns of this measurement again from the EDAX software (.up1 or .up2).';

  it('names the actual file of a damaged entry, in the German of the app too', async () => {
    setup({ inspection: { ...INSPECTION, measurements: [broken], unmatched_pattern_files: [] },
      h5Path: '/q/other_name.h5' });
    expect(screen.getByTestId('measurement-/Area 3/OIM Map 1').textContent)
      .toContain('Open other_name.h5 in the EDAX software and save it again');
    expect(screen.getByTestId('measurement-/Area 3/OIM Map 1').textContent).not.toContain(NEXT);
    cleanup();
    await act(async () => { await i18n.changeLanguage('de'); });
    setup({ inspection: { ...INSPECTION, measurements: [broken], unmatched_pattern_files: [] } });
    expect(screen.getByTestId('measurement-/Area 3/OIM Map 1').textContent).toContain(
      'Öffnen Sie session.h5 in der EDAX-Software und speichern Sie sie erneut, oder exportieren Sie diese Messung neu. Die anderen Messungen lassen sich trotzdem öffnen.');
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('gives no such advice to a measurement that is merely missing its file, or one that cannot be opened for another reason', () => {
    const grid = { ...B, measurement_problem: 'apexGridMismatch', problem: 'apexGridMismatch' };
    setup({ inspection: { ...INSPECTION, measurements: [grid], unmatched_pattern_files: [] } });
    expect(row(GB).textContent).not.toContain('save it again');
    expect(row(GB).textContent).not.toContain(REEXPORT);
  });

  it('tells a row without a file where a file comes from, after the other options', () => {
    setup();
    const text = row(GB).textContent;
    expect(text.indexOf('choose it or search that folder.')).toBeGreaterThan(-1);
    expect(text.indexOf(REEXPORT)).toBeGreaterThan(text.indexOf('choose it or search that folder.'));
  });

  it('tells a file that is too short the same, and keeps "interrupted export or a smaller scan"', () => {
    setup();
    const text = row(GC).textContent;
    expect(text).toContain('(for example from an interrupted export) or belong to a smaller scan');
    expect(text).toContain(REEXPORT);
  });

  it('does not tell a file that is too long to export again', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('long.up1', { path: p, n_patterns: 1500, count: 'too_many',
      matches_grid: false, problem: 'apexPatternCountMismatch', names_measurement: null, names_group: null })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).not.toContain('export the patterns of this measurement again');
  });
});

describe('a file that carries no measurement ID cannot be judged by its name, so the row says what can be checked', () => {
  it('shows the warning and the candidate\'s size and count in the row, for every candidate', () => {
    const two = { ...LOOSE, path: '/d/series/other.up1', name: 'other.up1', pattern_shape: [64, 64] };
    setup({ inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [LOOSE, two] } });
    const text = row(GB).textContent;
    expect(text).toContain('1,200 patterns, 156 × 128 px');
    expect(text).toContain('1,200 patterns, 64 × 64 px');
    expect(text.split('This file carries no measurement ID; only its pattern count fits.').length).toBe(2);
    expect(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }).getAttribute('title')).not.toMatch(/measurement ID|only its pattern count/);
  });

  it('says only the count when the file has no known pattern size, and "1 pattern" in the singular', () => {
    const noSize = { ...LOOSE, pattern_shape: null, n_patterns: 1, fits: [GUID_B] };
    setup({ inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [noSize] } });
    expect(row(GB).textContent).toContain('1 pattern');
    expect(row(GB).textContent).not.toContain('1 patterns');
    expect(row(GB).textContent).not.toMatch(/undefined|NaN|× px/);
  });

  it('keeps the count-only note after choosing it, in a warning colour and without the look of a match by ID', () => {
    setup();
    click(within(row(GB)).getByRole('button', { name: 'Use loose.up1' }));
    const cell = within(row(GB)).getByText(/loose\.up1, chosen by you/);
    expect(cell.style.color).toBe('var(--accent-orange)');
    expect(cell.textContent.startsWith('! ')).toBe(true);
    expect(row(GB).textContent).toContain('matched by pattern count only');
    expect(row(GB).textContent).toContain('Use it only if you know it was exported from this measurement.');
    expect(row(GB).textContent).not.toContain('✓ loose.up1');
  });

  it('a file chosen by its measurement ID keeps the green tick and no warning', async () => {
    setup({ describePatternFile: vi.fn(async (m, p) => file('named.up1', { path: p, n_patterns: 1200,
      pairing: 'guid', names_measurement: GUID_B, names_group: GB })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(row(GB).textContent).toContain('checked: it carries this measurement\'s ID'));
    expect(row(GB).textContent).toContain('✓ named.up1, chosen by you');
    expect(row(GB).textContent).not.toContain('only its pattern count fits');
    expect(row(GB).textContent).not.toContain('matched by pattern count only');
  });
});

describe('the summary and the rows agree on a pattern file that does not fit', () => {
  it('counts it as its own kind, never as "needs a pattern file", and not as ready', () => {
    setup();
    const text = screen.getByTestId('apex-summary').textContent;
    expect(text).toContain('1 pattern file does not fit');
    expect(text).toContain('1 needs a pattern file');
    expect(row(GC).textContent).toContain('so it cannot open this measurement with it');
    expect(row(GC).textContent).toContain('✗');
  });

  it('says it in the singular and the plural', () => {
    const C2 = { ...C, group: '/Area 3/OIM Map 3', id: 'guid-c2', name: 'OIM Map 3' };
    setup({ inspection: { ...INSPECTION, measurements: [C, C2], unmatched_pattern_files: [] } });
    expect(screen.getByTestId('apex-summary').textContent).toBe('2 measurements: 2 pattern files do not fit. 0 ticked.');
  });
});

describe('the grid chip says grid, and a damaged entry shows none', () => {
  it('labels the chip with "grid" so it is not read as the crystal system', () => {
    setup();
    const chip = (r) => [...r.querySelectorAll('span')].map((x) => x.textContent).filter((x) => /grid$/.test(x));
    expect(chip(row(GA))).toEqual(['Hexagonal grid']);
    expect(chip(row(GB))).toEqual(['Square grid']);
  });

  it('shows no chip for a grid type this version does not know, which would otherwise read "Square grid"', () => {
    const unknown = { ...B, grid_type: 'FancyGrid', measurement_problem: 'apexUnknownGrid', problem: 'apexUnknownGrid' };
    setup({ inspection: { ...INSPECTION, measurements: [unknown], unmatched_pattern_files: [] } });
    expect(row(GB).textContent).not.toMatch(/Square grid|Hexagonal grid/);
    expect(row(GB).textContent).toContain('This measurement cannot be opened: The measurement has a grid type this version does not know.');
  });

  it('shows no hexagonal sentence either for a hexagonal measurement whose table does not match its grid', async () => {
    const odd = { ...A, consistent: false };
    setup({ inspection: { ...INSPECTION, measurements: [odd], unmatched_pattern_files: [] } });
    await later(() => {});
    expect(screen.queryByTestId(`hexline-${GA}`)).toBeNull();
    expect(row(GA).textContent).not.toMatch(/Hexagonal grid|indexing works/);
  });

  it('shows no chip and a dash for the size when the measurement is not consistent', () => {
    const odd = { ...B, consistent: false, measurement_problem: 'apexGridMismatch', problem: 'apexGridMismatch' };
    setup({ inspection: { ...INSPECTION, measurements: [odd], unmatched_pattern_files: [] } });
    expect(row(GB).textContent).not.toMatch(/Square grid|Hexagonal grid|rows ×/);
    expect(row(GB).textContent).toContain('—');
  });
});

describe('Details names the pattern-centre refinement that really exists for the kind of grid', () => {
  it('for a hexagonal row only the global pattern centre can be refined later', () => {
    setup();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    const d = screen.getByTestId(`details-${GA}`).textContent;
    expect(d).toContain('the global pattern centre can be refined later in PC Refinement');
    expect(d).not.toContain('you can refine it later');
  });

  it('for a square row the pattern centre can be refined later, as before', () => {
    setup();
    click(within(row(GB)).getByRole('button', { name: 'Details' }));
    expect(screen.getByTestId(`details-${GB}`).textContent).toContain('you can refine it later in PC Refinement');
  });

  it('lists what still works from the backend\'s own list', async () => {
    setup();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    await waitFor(() => expect(screen.getByTestId(`details-${GA}`).textContent)
      .toContain('Still works: image export (PNG; phase maps, viewer maps, patterns), saving results as .ang or .h5 (.ctf and square .ang are resampled), and refining one global pattern centre.'));
  });

  it('names the Analysis page under its own heading, not as a tool that compares neighbouring points', async () => {
    setup();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    await waitFor(() => expect(screen.getByTestId('hex-limit-list')).toBeTruthy());
    const list = screen.getByTestId('hex-limit-list');
    const entry = [...list.children].find((li) => li.textContent.includes('all maps of the Analysis page'));
    expect(entry.querySelector('b').textContent).toBe('the Analysis page');
    expect(list.textContent).not.toMatch(/compare neighbouring points/);
  });

  it('says what an item that names its kinds refuses, from a hand-built item (the real list has none now)', async () => {
    const handBuilt = {
      groups: { before_indexing: [], after_indexing: [
        { key: 'analysis_map', category: 'maps', code: 'hex_analysis_page_pending', partial: true,
          affects: ['kind_a', 'kind_b'], parameters: [] }], batch: [] },
      categories: ['maps'], not_limited: [],
    };
    setup({ loadHexLimits: vi.fn(async () => handBuilt) });
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    await waitFor(() => expect(screen.getByTestId('hex-limit-list')).toBeTruthy());
    expect(screen.getByTestId('hex-limit-list').textContent)
      .toContain('all maps of the Analysis page: only for kind_a and kind_b');
  });
});


// ================================================================== pattern size of a count-only file

const SIZE = [128, 156];                                  // rows, cols of a pattern: 156 x 128 px
const sized = (m, shape = SIZE) => ({ ...m, pattern_shape: shape });

describe('a count-only file is compared with the measurement\'s own pattern size', () => {
  const only = (m, loose) => ({ inspection: { ...INSPECTION, measurements: [m], unmatched_pattern_files: loose } });
  const BIG = { ...LOOSE, path: '/d/series/other.up1', name: 'other.up1', pattern_shape: [64, 64] };

  it('shows both sizes side by side when both are known, and offers the file when they agree', () => {
    setup(only(sized(B), [LOOSE]));
    expect(row(GB).textContent).toContain('1,200 patterns; measurement: 156 × 128 px \u00b7 this file: 156 × 128 px');
    expect(within(row(GB)).getByRole('button', { name: 'Use loose.up1' })).toBeTruthy();
  });

  it('does not offer a file whose pattern size differs, and says why', () => {
    setup(only(sized(B), [LOOSE, BIG]));
    expect(within(row(GB)).queryByRole('button', { name: 'Use other.up1' })).toBeNull();
    expect(within(row(GB)).getByRole('button', { name: 'Use loose.up1' })).toBeTruthy();
    expect(row(GB).textContent).toContain('other.up1 (64 × 64 px) is not offered: this measurement\'s patterns are 156 × 128 px.');
  });

  it('offers the file as before when the measurement\'s size is not recorded', () => {
    setup(only(B, [LOOSE, BIG]));
    expect(within(row(GB)).getByRole('button', { name: 'Use other.up1' })).toBeTruthy();
    expect(row(GB).textContent).toContain('1,200 patterns, 64 × 64 px');
    expect(row(GB).textContent).not.toContain('is not offered');
  });

  it('refuses a file chosen by hand that has another pattern size, like a wrong count', async () => {
    const { onLoad } = setup({ ...only(sized(B), []), describePatternFile: vi.fn(async (m, p) => file('mine.up1', {
      path: p, n_patterns: 1200, pairing: 'count', pattern_shape: [64, 64], names_measurement: null, names_group: null })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    const alert = await within(row(GB)).findByRole('alert');
    expect(alert.textContent).toContain('final_v3.up1 has patterns of 64 × 64 px, but this measurement\'s are 156 × 128 px');
    expect(box('Area 2 / OIM Map 1').disabled).toBe(true);
    expect(box('Area 2 / OIM Map 1').checked).toBe(false);
    expect(onLoad).not.toHaveBeenCalled();
  });

  it('takes a file chosen by hand whose size agrees, and keeps both sizes visible', async () => {
    setup({ ...only(sized(B), []), describePatternFile: vi.fn(async (m, p) => file('mine.up1', {
      path: p, n_patterns: 1200, pairing: 'count', names_measurement: null, names_group: null })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(box('Area 2 / OIM Map 1').checked).toBe(true));
    expect(row(GB).textContent).toContain('measurement: 156 × 128 px \u00b7 this file: 156 × 128 px');
  });

  it('does not judge a file that names this measurement by its ID on the size: the ID is the stronger proof', async () => {
    setup({ ...only(sized(B), []), describePatternFile: vi.fn(async (m, p) => file('named.up1', {
      path: p, n_patterns: 1200, pairing: 'guid', pattern_shape: [64, 64], names_measurement: GUID_B, names_group: GB })) });
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(box('Area 2 / OIM Map 1').checked).toBe(true));
  });
});

describe('what "Use <file>" does is said and done', () => {
  it('says on the button that the file is used for the measurement and then ticked, and ticks it', () => {
    setup();
    const button = within(row(GB)).getByRole('button', { name: 'Use loose.up1' });
    expect(button.getAttribute('title')).toBe('Use this file for the measurement (it is then ticked)');
    expect(box('Area 2 / OIM Map 1').checked).toBe(false);
    click(button);
    expect(box('Area 2 / OIM Map 1').checked).toBe(true);
    expect(openButton().textContent).toBe('Open 1 measurement');
  });
});

describe('the hint at the main button when nothing can be ticked', () => {
  const NONE = 'None of these measurements can be opened as they are; see each row.';

  it('says so when every row is disabled', () => {
    setup({ inspection: { ...INSPECTION, measurements: [B, C], unmatched_pattern_files: [] } });
    expect(screen.getByTestId('apex-load-hint').textContent).toBe(NONE);
    expect(openButton().disabled).toBe(true);
  });

  it('asks for a tick when something can be ticked', () => {
    setup();
    expect(screen.getByTestId('apex-load-hint').textContent).toBe('Tick a measurement to enable this button.');
  });

  it('changes to the tick hint once a file the user chose makes a row openable', async () => {
    setup({ inspection: { ...INSPECTION, measurements: [B], unmatched_pattern_files: [] },
      describePatternFile: vi.fn(async (m, p) => file('mine.up1', { path: p, n_patterns: 1200, pairing: 'count',
        names_measurement: null, names_group: null })) });
    expect(screen.getByTestId('apex-load-hint').textContent).toBe(NONE);
    click(within(row(GB)).getByRole('button', { name: 'Choose file…' }));
    await waitFor(() => expect(box('Area 2 / OIM Map 1').checked).toBe(true));
    expect(screen.queryByTestId('apex-load-hint')).toBeNull();
  });
});

describe('the summary line does not promise more than it knows', () => {
  it('says that measurements can be opened and that the user ticks the ones wanted', () => {
    setup({ inspection: { ...INSPECTION, measurements: [A, sized(A, SIZE)].map((m, i) => ({ ...m, group: `/Area ${i + 1}/OIM Map 1`, id: `g${i}` })), unmatched_pattern_files: [] } });
    expect(screen.getByTestId('apex-summary').textContent).toBe('2 measurements can be opened; tick the ones you want. 0 ticked.');
  });

  it('says it in the singular', () => {
    setup({ inspection: { ...INSPECTION, measurements: [A], unmatched_pattern_files: [] } });
    expect(screen.getByTestId('apex-summary').textContent).toBe('1 measurement can be opened. 1 ticked.');
  });

  it('says nothing about hexagonal results when only square rows can be opened', async () => {
    const square = { ...B, loadable: true, problem: null, pattern_files: [file('b.up1', { n_patterns: 1200, names_measurement: GUID_B, names_group: GB })] };
    const { loadHexLimits } = setup({ inspection: { ...INSPECTION, measurements: [square, C], unmatched_pattern_files: [] } });
    await waitFor(() => expect(loadHexLimits).toHaveBeenCalled());
    await later(() => {});
    expect(screen.queryByTestId('apex-summary-hex')).toBeNull();
  });
});

describe('the phases line and the row spacing', () => {
  it('names the phases when the file names them, and says plainly when it names none', () => {
    setup();
    click(within(row(GA)).getByRole('button', { name: 'Details' }));
    expect(screen.getByTestId(`details-${GA}`).textContent).toContain('Phases named in the filePhase A');
    click(within(row(GB)).getByRole('button', { name: 'Details' }));
    const none = screen.getByTestId(`details-${GB}`).textContent;
    expect(none).toContain('No phases are named in the file; you choose the phases for indexing in Orienta.');
    expect(none).not.toContain('Phases named in the file');
  });

  it('explains the row spacing of a hexagonal scan once: step times 0.866', () => {
    setup();
    const text = row(GA).textContent;
    expect(text).toContain('row spacing 0.26 \u00b5m (hexagonal: step × 0.866)');
    expect(text.split('0.866').length).toBe(2);
  });

  it('gives no such explanation to a square scan whose two steps differ', () => {
    setup({ inspection: { ...INSPECTION, measurements: [{ ...B, step_x: 1, step_y: 1.5 }], unmatched_pattern_files: [] } });
    expect(row(GB).textContent).toContain('row spacing 1.5 \u00b5m');
    expect(row(GB).textContent).not.toContain('0.866');
  });

  it('writes the factor with the decimal sign of the page', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    setup();
    expect(row(GA).textContent).toContain('Zeilenabstand 0,26 \u00b5m (hexagonal: Schritt × 0,866)');
    await act(async () => { await i18n.changeLanguage('en'); });
  });
});

describe('a damaged entry has no Details toggle, any other row has one', () => {
  it('shows the toggle for a measurement that cannot be opened for another reason, and for one that is inconsistent', () => {
    const grid = { ...B, measurement_problem: 'apexGridMismatch', problem: 'apexGridMismatch', consistent: false };
    setup({ inspection: { ...INSPECTION, measurements: [A, grid], unmatched_pattern_files: [] } });
    expect(within(row(GA)).getByRole('button', { name: 'Details' })).toBeTruthy();
    expect(within(row(GB)).getByRole('button', { name: 'Details' })).toBeTruthy();
  });

  it('names the file in the sentence about the damaged entry', () => {
    const broken = { group: '/Area 3/OIM Map 1', id: null, area: 'Area 3', name: 'OIM Map 1', hex: false,
      n_rows: null, n_cols: null, step_x: null, step_y: null, pc: null, phases: [], has_sem: false, has_iq: false,
      loadable: false, problem: 'apexBadId', measurement_problem: 'apexBadId', problem_detail: 'x', pattern_files: [],
      consistent: null };
    setup({ inspection: { ...INSPECTION, measurements: [broken], unmatched_pattern_files: [] }, h5Path: '/q/other_name.h5' });
    const text = screen.getByTestId('measurement-/Area 3/OIM Map 1').textContent;
    expect(text).toContain('This measurement\'s entry in other_name.h5 is damaged, so it cannot be opened.');
    expect(text).not.toContain('metadata file');
  });
});

describe('a pattern file that is too short names the requirement first', () => {
  it('puts what the measurement needs before what the file holds and before the advice, in the row and in the tooltip', () => {
    setup();
    for (const text of [row(GC).textContent, box('Area 2 / OIM Map 2').getAttribute('title')]) {
      const need = text.indexOf('needs a pattern file with 2,000 patterns (50 rows × 40 points)');
      const holds = text.indexOf('map_c.up1 holds 1,856 patterns');
      const advice = text.indexOf('If the pattern file no longer exists');
      expect(need).toBeGreaterThan(-1);
      expect(holds).toBeGreaterThan(need);
      expect(advice).toBeGreaterThan(holds);
    }
    expect(row(GC).textContent.split('needs a pattern file with 2,000').length).toBe(2);   // said once
  });
});
