// @vitest-environment jsdom
/**
 * "Simulate All Missing" must not answer "All materials already have simulation
 * results." for a phase whose only master the library refuses (kikuchipy cannot
 * load it): the scan said nothing, the job warning said
 * "simulate again".
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import i18n from '../../i18n';

const scan = vi.hoisted(() => ({ answer: { missing: [], unreadable: [] } }));

vi.mock('../../services/api', () => ({
  simApi: {
    systemStatus: () => Promise.resolve({ data: { emsoft_available: true, has_gpu: false, cpu_count: 8 } }),
    getConfig: () => Promise.resolve({ data: { config: {} } }),
    history: () => Promise.resolve({ data: { jobs: [] } }),
    getStatus: () => Promise.resolve({ data: {} }),
    batchStatus: () => Promise.resolve({ data: { jobs: [] } }),
    scanMissing: () => Promise.resolve({ data: scan.answer }),
    crystalPicker: () => Promise.resolve({ data: { phases: [] } }),
    serverConfig: () => Promise.resolve({ data: {} }),
    getNmlTemplate: () => Promise.resolve({ data: '' }),
    recommendEngine: () => Promise.resolve({ data: { engine: 'emsoft', reason: '' } }),
  },
}));
vi.mock('./CrystalPickerModal', () => ({ default: () => null }));
vi.mock('./SimulateMissingDialog', () => ({ default: () => null }));

import SimulationPage from './SimulationPage';

afterEach(cleanup);
beforeAll(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });

const ENTRY = { stem: 'Mg', xtal_path: 'X/Mg.xtal', code: 'masterUnreadableByKikuchipy',
                file: 'Mg_master_E20kV_npx500.h5', program: 'EMEBSDmasterOpenCL.f90' };

async function runScan() {
  render(<SimulationPage isActive />);
  const button = await screen.findByRole('button', { name: /simulate all missing/i });
  await act(async () => { fireEvent.click(button); });
}

describe('the scan answer on the Simulation page', () => {
  it('does not say "all done" when a master is refused', async () => {
    scan.answer = { missing: [], unreadable: [ENTRY] };
    await runScan();
    expect(await screen.findByText(/cannot be used for Dictionary indexing/)).toBeInTheDocument();
    expect(screen.getByText(/Mg_master_E20kV_npx500.h5/)).toBeInTheDocument();
    expect(screen.queryByText('All materials already have simulation results.')).toBeNull();
  });

  it('still says "all done" when nothing is missing and nothing is refused', async () => {
    scan.answer = { missing: [], unreadable: [] };
    await runScan();
    expect(await screen.findByText('All materials already have simulation results.')).toBeInTheDocument();
  });

  it('speaks German in the polite form', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    scan.answer = { missing: [], unreadable: [ENTRY] };
    render(<SimulationPage isActive />);
    const button = await screen.findByRole('button', { name: /fehlende/i });
    await act(async () => { fireEvent.click(button); });
    expect(await screen.findByText(/Löschen Sie die Datei und simulieren Sie die Phase neu/)).toBeInTheDocument();
    await act(async () => { await i18n.changeLanguage('en'); });
  });
});
