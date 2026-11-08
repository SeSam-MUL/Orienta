// @vitest-environment jsdom
/**
 * A finished job's warnings reach the user.
 *
 * The backend's finishing step attaches a coded warning
 * (`masterUnreadableByKikuchipy`) to a job whose master kikuchipy cannot load,
 * but no frontend code read `job.warnings`, so the job simply ended "completed".
 * This file holds the reader (`JobWarnings`) and that the Simulation page shows it
 * in the queue row and the history row.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import i18n from '../../i18n';
import JobWarnings from './JobWarnings';

const FILE = 'Mg_COD-9008506_master_E20kV_npx500.h5';
const WARNING = { code: 'masterUnreadableByKikuchipy', file: FILE, program: 'EMEBSDmasterOpenCL.f90' };

vi.mock('../../services/api', () => ({
  simApi: {
    systemStatus: () => Promise.resolve({ data: { emsoft_available: true, has_gpu: false, cpu_count: 8 } }),
    getConfig: () => Promise.resolve({ data: { config: {} } }),
    history: () => Promise.resolve({ data: { jobs: [{
      id: 'h1', crystal: 'C:/db/XTAL_Library/Mg_COD-9008506.xtal', method: 'master_only',
      status: 'completed', started_at: '2026-10-12T09:00:00+00:00',
      warnings: [{ code: 'masterUnreadableByKikuchipy', file: 'Mg_COD-9008506_master_E20kV_npx500.h5',
                   program: 'EMEBSDmasterOpenCL.f90' }],
    }, {
      id: 'h2', crystal: 'C:/db/XTAL_Library/Al.xtal', method: 'master_only',
      status: 'completed', started_at: '2026-10-12T10:00:00+00:00', warnings: [],
    }] } }),
    getStatus: () => Promise.resolve({ data: {} }),
    batchStatus: () => Promise.resolve({ data: { jobs: [] } }),
    scanMissing: () => Promise.resolve({ data: { missing: [] } }),
    crystalPicker: () => Promise.resolve({ data: { phases: [] } }),
    serverConfig: () => Promise.resolve({ data: {} }),
    getNmlTemplate: () => Promise.resolve({ data: '' }),
    recommendEngine: () => Promise.resolve({ data: { engine: 'emsoft', reason: '' } }),
    clearHistory: () => Promise.resolve({ data: {} }),
    stop: () => Promise.resolve({ data: {} }),
    batchCancel: () => Promise.resolve({ data: {} }),
  },
}));
vi.mock('./CrystalPickerModal', () => ({ default: () => null }));
vi.mock('./SimulateMissingDialog', () => ({ default: () => null }));

afterEach(cleanup);
beforeAll(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });

describe('JobWarnings', () => {
  it('names the file, the program and what it means', async () => {
    render(<JobWarnings warnings={[WARNING]} />);
    const row = screen.getByTestId('job-warnings').querySelector('[data-code]');
    expect(row.dataset.code).toBe('masterUnreadableByKikuchipy');
    expect(row.textContent).toContain(FILE);
    expect(row.textContent).toContain('EMEBSDmasterOpenCL.f90');
    expect(row.textContent).toMatch(/cannot be used for Dictionary indexing/);
  });

  it('says "an unknown program" when the file did not name one', () => {
    render(<JobWarnings warnings={[{ ...WARNING, program: null }]} />);
    expect(screen.getByTestId('job-warnings').textContent).toContain('an unknown program');
  });

  it('shows a code it does not know instead of dropping it', () => {
    render(<JobWarnings warnings={[{ code: 'somethingNew', file: 'x.h5' }]} />);
    expect(screen.getByTestId('job-warnings').textContent).toContain('somethingNew');
  });

  it.each([[undefined], [null], [[]]])('renders nothing for %j', (w) => {
    render(<JobWarnings warnings={w} />);
    expect(screen.queryByTestId('job-warnings')).toBeNull();
  });

  it.each([
    ['de', /Löschen Sie die Datei und simulieren Sie die Phase neu/],
    ['ja', /辞書インデックス法に使用できません/],
    ['zh', /无法用于字典索引/],
  ])('speaks %s', async (lang, pattern) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    render(<JobWarnings warnings={[WARNING]} />);
    const text = screen.getByTestId('job-warnings').textContent;
    expect(text).toMatch(pattern);
    expect(text).toContain(FILE);
    await act(async () => { await i18n.changeLanguage('en'); });
  });
});

describe('the Simulation page shows it', () => {
  it('on the history row of a finished job, and only on that row', async () => {
    const { default: SimulationPage } = await import('./SimulationPage');
    render(<SimulationPage isActive />);
    fireEvent.click(await screen.findByRole('tab', { name: /upload history/i }));
    await waitFor(() => expect(screen.getAllByTestId('job-warnings')).toHaveLength(1));
    expect(screen.getByTestId('job-warnings').textContent).toContain(FILE);
    // the second history entry has no warnings and shows none
    expect(screen.getByText('Al.xtal')).toBeInTheDocument();
  });

  it('wires the queue row and the batch poll copy (held on the source)', async () => {
    const { default: src } = await import('./SimulationPage.jsx?raw');
    expect(src).toContain('<JobWarnings warnings={job.warnings} />');
    expect(src).toContain('warnings: match.warnings || j.warnings');
    expect(src).toContain('<JobWarnings warnings={entry.warnings} />');
  });
});
