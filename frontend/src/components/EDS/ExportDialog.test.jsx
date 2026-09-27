// @vitest-environment jsdom
/**
 * The export dialog, pinned on the three things that make it honest rather
 * than merely present.
 *
 *  1. The counts are on screen BEFORE anything is written. A user must not
 *     have to export to find out what they would get.
 *  2. A scan with no step size says so, prominently. The µm columns are then
 *     omitted on purpose, and an omission discovered in the CSV is a bug
 *     report.
 *  3. A comma decimal with a comma delimiter is refused in the UI, before the
 *     request. Two users independently raised the German-Excel case; with
 *     both set to a comma every value additionally splits across two columns.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

const exportPreview = vi.fn();
const runExport = vi.fn();
vi.mock('../../services/api', () => ({
  edsExportApi: {
    exportPreview: (...a) => exportPreview(...a),
    runExport: (...a) => runExport(...a),
  },
}));

import ExportDialog, {
  ARTEFACTS, DEFAULT_ARTEFACTS, separatorClash, canRunExport,
  summaryClauses, buildSummaryLine, scanName, appLabel, fmtNum,
} from './ExportDialog';

const PREVIEW = {
  ok: true,
  n_phases: 4,
  n_regions: 7,
  n_particles: 1842,
  n_classified_px: 9800,
  n_total_px: 10800,
  step_x_um: 0.658,
  step_y_um: 0.658,
  area_available: true,
  connectivity: 8,
  warnings: [],
};

/**
 * A `provenance` block with the field names `_provenance()` really writes.
 * The summary line is only as honest as its source, so the test reads from
 * the same shape rather than from a convenient flat object.
 */
const PROV = {
  app: { app: 'Orienta', version: '2026-08-27', commit: 'f1080b69' },
  source: { path: 'D:/data/Sample B.h5oina' },
  grid: { n_rows: 120, n_cols: 90, n_px: 10800 },
  step: { x_um: 0.5, y_um: 0.5, area_available: true },
  counts: {
    n_regions: 7, n_phases_with_pixels: 4, n_particles: 135,
  },
  classification: { smoothing: { scale_px: 3, box_um: 1.5 } },
  particles: { connectivity: 8, min_particle_px: 4 },
  determinism: { deterministic: true, random_state: 0, n_init: 10 },
};

const DONE = {
  ok: true,
  folder: 'D:/out/Scan1_EDS_2026-08-27_1130',
  files: [{ name: 'particles.csv', rows: 135, bytes: 1 }],
  warnings: [],
  provenance: PROV,
};

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  exportPreview.mockResolvedValue({ data: PREVIEW });
  runExport.mockResolvedValue({
    data: {
      ok: true,
      folder: 'D:/out/Scan1_EDS_2026-08-27_1130',
      files: [{ name: 'particles.csv', rows: 1842, bytes: 1 }],
      warnings: [],
    },
  });
});
afterEach(() => { delete window.electronAPI; });

const open = (over = {}) => render(<ExportDialog open onClose={vi.fn()} {...over} />);

describe('the pure parts', () => {
  it('only comma+comma is a clash', () => {
    expect(separatorClash(',', ',')).toBe(true);
    expect(separatorClash(',', ';')).toBe(false);
    expect(separatorClash(',', '\t')).toBe(false);
    expect(separatorClash('.', ',')).toBe(false);
  });

  it('refuses to run without a destination, or with a clash', () => {
    expect(canRunExport({ destDir: 'D:/x', decimal: '.', delimiter: ',' })).toBe(true);
    expect(canRunExport({ destDir: '   ', decimal: '.', delimiter: ',' })).toBe(false);
    expect(canRunExport({ destDir: 'D:/x', decimal: ',', delimiter: ',' })).toBe(false);
    expect(canRunExport({ destDir: 'D:/x', decimal: '.', delimiter: ',', busy: true }))
      .toBe(false);
  });

  it('the two large artefacts are the two that default off', () => {
    const off = ARTEFACTS.filter((a) => !a.on).map((a) => a.id);
    expect(off).toEqual(['labels', 'pixels']);
    expect(DEFAULT_ARTEFACTS.phases).toBe(true);
    expect(DEFAULT_ARTEFACTS.particles).toBe(true);
    expect(DEFAULT_ARTEFACTS.pixels).toBe(false);
  });
});

describe('what you would get, before writing', () => {
  it('asks the dry run on open and shows the real counts', async () => {
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    await waitFor(() => {
      const el = document.querySelector('[data-export-preview]');
      // Grouped the way this runtime groups it, rather than a hard-coded
      // "1,842" that would fail on a machine with a different default locale.
      expect(el.textContent).toContain((1842).toLocaleString());
      expect(el.textContent).toContain('"regions":"7"');
    });
    // Nothing was written to get that number.
    expect(runExport).not.toHaveBeenCalled();
  });

  it('recounts when the neighbourhood changes — it changes the count', async () => {
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalledTimes(1));
    fireEvent.click(document.querySelector('[data-choice="connectivity:4"]'));
    await waitFor(() => expect(exportPreview).toHaveBeenCalledTimes(2));
    expect(exportPreview.mock.calls[1][0]).toMatchObject({ connectivity: 4 });
  });
});

describe('a scan with no step size', () => {
  it('says the µm columns are left out, prominently', async () => {
    exportPreview.mockResolvedValue({ data: { ...PREVIEW, area_available: false } });
    open();
    await waitFor(() => {
      expect(document.querySelector('[data-export-area-warning]')).toBeTruthy();
    });
    expect(document.querySelector('[data-export-area-warning]').textContent)
      .toContain('export.areaMissing');
  });

  it('stays quiet when the step size is there', async () => {
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    await waitFor(() => {
      expect(document.querySelector('[data-export-preview]').textContent)
        .toContain((1842).toLocaleString());
    });
    expect(document.querySelector('[data-export-area-warning]')).toBeNull();
  });
});

describe('the comma/comma refusal', () => {
  it('blocks the request in the UI rather than sending it', async () => {
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText('export.dest'), {
      target: { value: 'D:/out' },
    });
    // A dot decimal with a comma delimiter is fine.
    const run = screen.getByRole('button', { name: 'export.run' });
    expect(run.disabled).toBe(false);

    fireEvent.click(document.querySelector('[data-choice="decimal:,"]'));
    expect(document.querySelector('[data-export-separator-clash]')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'export.run' }).disabled).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'export.run' }));
    expect(runExport).not.toHaveBeenCalled();

    // A semicolon resolves it.
    fireEvent.click(document.querySelector('[data-choice="delimiter:;"]'));
    expect(document.querySelector('[data-export-separator-clash]')).toBeNull();
    expect(screen.getByRole('button', { name: 'export.run' }).disabled).toBe(false);
  });
});

describe('running it', () => {
  it('sends the chosen artefacts and locale, and reports the rows written', async () => {
    open({ presetName: 'Al matrix', compatibility: { ok: true } });
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText('export.dest'), { target: { value: 'D:/out' } });
    fireEvent.click(document.querySelector('[data-artefact="pixels"]'));
    fireEvent.click(screen.getByRole('button', { name: 'export.run' }));

    await waitFor(() => expect(runExport).toHaveBeenCalled());
    const sent = runExport.mock.calls[0][0];
    expect(sent.destDir).toBe('D:/out');
    expect(sent.artefacts.pixels).toBe(true);
    expect(sent.artefacts.labels).toBe(false);
    expect(sent.decimal).toBe('.');
    expect(sent.connectivity).toBe(8);
    expect(sent.minParticlePx).toBe(0);
    expect(sent.presetName).toBe('Al matrix');
    expect(sent.compatibility).toEqual({ ok: true });

    await waitFor(() => {
      const res = document.querySelector('[data-export-result]');
      expect(res.textContent).toContain('D:/out/Scan1_EDS_2026-08-27_1130');
      expect(res.textContent).toContain('particles.csv');
      expect(res.textContent).toContain((1842).toLocaleString());
    });
  });

  it('shows the backend detail rather than a generic failure', async () => {
    runExport.mockRejectedValue({ response: { data: { detail: 'destination not writable' } } });
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText('export.dest'), { target: { value: 'D:/out' } });
    fireEvent.click(screen.getByRole('button', { name: 'export.run' }));
    await waitFor(() => {
      expect(document.querySelector('[data-export-error]').textContent)
        .toContain('destination not writable');
    });
  });

  it('says so when the dry run itself fails, instead of showing nothing', async () => {
    exportPreview.mockRejectedValue({ response: { data: { detail: 'no map' } } });
    open();
    await waitFor(() => {
      expect(document.querySelector('[data-export-preview]').textContent).toContain('no map');
    });
  });
});

describe('the size limit is a flag, not a filter', () => {
  it('states that nothing is deleted', async () => {
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    expect(document.body.textContent).toContain('export.minPxNote');
  });
});

describe('the destination', () => {
  it('offers Browse under Electron and a typed path without it', async () => {
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    expect(screen.queryByRole('button', { name: 'export.browse' })).toBeNull();
    cleanup();

    window.electronAPI = { openFolder: vi.fn().mockResolvedValue('D:/picked') };
    open();
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'export.browse' })).toBeTruthy();
    });
    fireEvent.click(screen.getByRole('button', { name: 'export.browse' }));
    await waitFor(() => {
      expect(screen.getByLabelText('export.dest').value).toBe('D:/picked');
    });
  });
});


/**
 * The pasteable line.
 *
 * Two testers asked for it independently and neither got it: a failure
 * analyst wanted it for figure captions, a group leader for a grant report,
 * and both were left assembling it by hand out of provenance.json. Every
 * value in it is one the export response returned - the rule that matters is
 * the negative one: a clause whose value this run did not supply is OMITTED,
 * never printed with a dash or a guess, because this text ends up under a
 * figure in a paper.
 */
describe('the summary line', () => {
  const t = (k, o) => (o && Object.keys(o).length ? `${k}:${JSON.stringify(o)}` : k);

  it('reads the scan name off the source path, without the extension', () => {
    expect(scanName('D:/data/Sample B.h5oina')).toBe('Sample B');
    expect(scanName('C:\\scans\\a.b.h5')).toBe('a.b');
    expect(scanName('')).toBe(null);
    expect(scanName(undefined)).toBe(null);
  });

  it('names the build, and does not print the commit twice', () => {
    expect(appLabel({ app: 'Orienta', version: '2026-08-27', commit: 'f1080b69' }))
      .toBe('Orienta 2026-08-27 (f1080b69)');
    // Some builds already carry the hash inside `version`.
    expect(appLabel({ app: 'Orienta', version: '2026-08-27 (f1080b69)', commit: 'f1080b69' }))
      .toBe('Orienta 2026-08-27 (f1080b69)');
    expect(appLabel(null)).toBe(null);
  });

  it('formats numbers for prose, not for a spreadsheet', () => {
    expect(fmtNum(1.5)).toBe('1.5');
    expect(fmtNum(45.0)).toBe('45');
    expect(fmtNum(0.6579, 3)).toBe('0.658');
    expect(fmtNum(null)).toBe(null);
    expect(fmtNum('n/a')).toBe(null);
  });

  it('states the extent, the structure, the count, the box, the build and the caveat', () => {
    const keys = summaryClauses({ provenance: PROV }).map((c) => c.key);
    expect(keys).toEqual([
      'export.sum.scan',
      'export.sum.regions',
      'export.sum.particles',
      'export.sum.smoothing',
      'export.sum.app',
      'export.sum.deterministic',
      'export.sum.semiQuant',
    ]);

    const byKey = Object.fromEntries(
      summaryClauses({ provenance: PROV }).map((c) => [c.key, c.vars]));
    // 90 columns x 0.5 um and 120 rows x 0.5 um - the map's own extent, not
    // a number typed into this file.
    expect(byKey['export.sum.scan'])
      .toMatchObject({ name: 'Sample B', width: '45', height: '60', step: '0.5' });
    expect(byKey['export.sum.regions']).toEqual({ regions: '7', phases: '4' });
    expect(byKey['export.sum.particles']).toEqual({ particles: '135' });
    expect(byKey['export.sum.smoothing']).toEqual({ px: '3', um: '1.5' });
    expect(byKey['export.sum.app']).toEqual({ app: 'Orienta 2026-08-27 (f1080b69)' });
    expect(byKey['export.sum.deterministic']).toEqual({ seed: '0' });
  });

  it('omits a clause it has no value for, rather than printing a placeholder', () => {
    const thin = {
      provenance: {
        source: { path: 'D:/data/Sample B.h5oina' },
        grid: { n_rows: 120, n_cols: 90 },
        step: { x_um: null, y_um: null, area_available: false },
        counts: {},
        classification: {},
        particles: {},
        determinism: {},
      },
    };
    const keys = summaryClauses(thin).map((c) => c.key);
    // No step size -> pixels, and the line SAYS the step size is missing.
    expect(keys).toContain('export.sum.scanPx');
    expect(keys).not.toContain('export.sum.scan');
    // No counts, no smoothing, no app, no determinism claim.
    expect(keys).not.toContain('export.sum.regions');
    expect(keys).not.toContain('export.sum.particles');
    expect(keys).not.toContain('export.sum.smoothing');
    expect(keys).not.toContain('export.sum.app');
    expect(keys).not.toContain('export.sum.deterministic');
    // The one clause that is always true of this app's numbers stays.
    expect(keys).toContain('export.sum.semiQuant');
    // And nothing in the rendered line is an em dash standing in for a value.
    expect(buildSummaryLine(thin, t)).not.toContain('\u2014');
  });

  it('an export with no provenance at all still produces only true clauses', () => {
    expect(summaryClauses({}).map((c) => c.key)).toEqual(['export.sum.semiQuant']);
    expect(summaryClauses(null).map((c) => c.key)).toEqual(['export.sum.semiQuant']);
  });

  it('qualifies the particle count only when the backend counted the qualifiers', () => {
    const withFlags = {
      provenance: {
        ...PROV,
        counts: {
          ...PROV.counts,
          n_particles_below_size_limit: 81,
          n_particles_touching_edge: 38,
        },
      },
    };
    const c = summaryClauses(withFlags).find((x) => x.key.startsWith('export.sum.particles'));
    expect(c.key).toBe('export.sum.particlesFlagged');
    expect(c.vars).toEqual({ particles: '135', below: '81', limit: '4', edge: '38' });

    // Without them the count is stated plainly. Half a qualifier would be a
    // sentence that reads like a measurement and is not one.
    const half = {
      provenance: {
        ...PROV,
        counts: { ...PROV.counts, n_particles_below_size_limit: 81 },
      },
    };
    expect(summaryClauses(half).find((x) => x.key.startsWith('export.sum.particles')).key)
      .toBe('export.sum.particles');
  });

  it('joins the clauses into one line', () => {
    const line = buildSummaryLine({ provenance: PROV }, (k) => k.split('.').pop());
    expect(line).toBe('scan regions particles smoothing app deterministic semiQuant');
  });
});

describe('the summary on screen', () => {
  it('appears after a successful export, selectable and read-only', async () => {
    runExport.mockResolvedValue({ data: DONE });
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    // Nothing to paste before anything was written.
    expect(document.querySelector('[data-export-summary]')).toBeNull();

    fireEvent.change(screen.getByLabelText('export.dest'), { target: { value: 'D:/out' } });
    fireEvent.click(screen.getByRole('button', { name: 'export.run' }));

    await waitFor(() => expect(document.querySelector('[data-export-summary]')).toBeTruthy());
    const box = document.querySelector('[data-export-summary]');
    expect(box.readOnly).toBe(true);
    // Built from the response: the scan, its extent, the structure, the box.
    expect(box.value).toContain('Sample B');
    expect(box.value).toContain('export.sum.regions');
    expect(box.value).toContain('Orienta 2026-08-27 (f1080b69)');
    expect(box.value).toContain('export.sum.semiQuant');
  });

  it('copies it in one click', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText }, configurable: true,
    });
    runExport.mockResolvedValue({ data: DONE });
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText('export.dest'), { target: { value: 'D:/out' } });
    fireEvent.click(screen.getByRole('button', { name: 'export.run' }));
    await waitFor(() => expect(document.querySelector('[data-export-summary]')).toBeTruthy());

    fireEvent.click(screen.getByRole('button', { name: 'export.copy' }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText.mock.calls[0][0])
      .toBe(document.querySelector('[data-export-summary]').value);
    await waitFor(() => {
      expect(document.querySelector('[data-export-copy-status]').textContent)
        .toContain('export.copied');
    });
    delete navigator.clipboard;
  });

  it('says so when the clipboard is not reachable, instead of doing nothing', async () => {
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    document.execCommand = vi.fn(() => false);
    runExport.mockResolvedValue({ data: DONE });
    open();
    await waitFor(() => expect(exportPreview).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText('export.dest'), { target: { value: 'D:/out' } });
    fireEvent.click(screen.getByRole('button', { name: 'export.run' }));
    await waitFor(() => expect(document.querySelector('[data-export-summary]')).toBeTruthy());

    fireEvent.click(screen.getByRole('button', { name: 'export.copy' }));
    await waitFor(() => {
      expect(document.querySelector('[data-export-copy-status]').textContent)
        .toContain('export.copyFailed');
    });
  });
});
