// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup, render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import CrystalStructureViewer from './CrystalStructureViewer';
import { dbApi } from '../../services/api';

const { fakeHandle, createCrystalScene } = vi.hoisted(() => {
  const fakeHandle = {
    update: vi.fn(), setOptions: vi.fn(), resetView: vi.fn(),
    screenshot: vi.fn(), dispose: vi.fn(),
  };
  return { fakeHandle, createCrystalScene: vi.fn(() => fakeHandle) };
});

vi.mock('../../services/api', () => ({ dbApi: { structure: vi.fn() } }));
vi.mock('./crystalScene3d', () => ({ createCrystalScene }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o && o.error ? `err ${o.error}` : k) }),
}));

afterEach(() => { cleanup(); vi.clearAllMocks(); });

const payload = {
  source: 'cif',
  cell_vectors: [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
  space_group: { number: 225, symbol: 'Fm-3m', crystal_system: 'cubic' },
  atoms: [
    { label: 'Al1', element: 'Al', Z: 13, cart: [0, 0, 0], frac: [0, 0, 0], occ: 1, species: [{ el: 'Al', occ: 1 }], color: '#bfa6a6', radius: 1.21 },
    { label: 'Fe1', element: 'Fe', Z: 26, cart: [2, 2, 2], frac: [0.5, 0.5, 0.5], occ: 1, species: [{ el: 'Fe', occ: 1 }], color: '#e06633', radius: 1.32 },
  ],
  bonds: [[[0, 0, 0], [2, 2, 2]]],
  polyhedra: [{ element: 'Fe', color: '#e06633', center: [2, 2, 2], vertices: [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]] }],
  meta: { formula: 'AlFe', n_atoms: 2, disordered: false },
};

describe('CrystalStructureViewer (three.js)', () => {
  it('does not fetch for a non-local entry', () => {
    render(<CrystalStructureViewer filename="Al.cif" isLocal={false} fileType="CIF" />);
    expect(dbApi.structure).not.toHaveBeenCalled();
  });

  it('creates the scene and loads geometry on success', async () => {
    dbApi.structure.mockResolvedValue({ data: payload });
    render(<CrystalStructureViewer filename="Al.cif" isLocal fileType="CIF" />);
    await waitFor(() => expect(createCrystalScene).toHaveBeenCalled());
    await waitFor(() => expect(fakeHandle.update).toHaveBeenCalled());
    const [passedPayload] = fakeHandle.update.mock.calls.at(-1);
    expect(passedPayload.atoms).toHaveLength(2);
    // legend renders one entry per element
    expect(screen.getByText('Al')).toBeInTheDocument();
    expect(screen.getByText('Fe')).toBeInTheDocument();
  });

  it('shows an error message on a failed fetch', async () => {
    dbApi.structure.mockRejectedValue({ message: 'boom' });
    render(<CrystalStructureViewer filename="Al.cif" isLocal fileType="CIF" />);
    await waitFor(() => expect(screen.getByText(/boom/i)).toBeInTheDocument());
  });

  it('reset + screenshot buttons drive the scene handle', async () => {
    dbApi.structure.mockResolvedValue({ data: payload });
    render(<CrystalStructureViewer filename="Al.cif" isLocal fileType="CIF" />);
    await waitFor(() => expect(fakeHandle.update).toHaveBeenCalled());
    fireEvent.click(screen.getByTitle('crystalStructure.resetTooltip'));
    fireEvent.click(screen.getByTitle('crystalStructure.screenshotTooltip'));
    expect(fakeHandle.resetView).toHaveBeenCalled();
    expect(fakeHandle.screenshot).toHaveBeenCalled();
  });

  it('clicking a legend entry toggles element visibility via setOptions', async () => {
    dbApi.structure.mockResolvedValue({ data: payload });
    render(<CrystalStructureViewer filename="Al.cif" isLocal fileType="CIF" />);
    await waitFor(() => expect(fakeHandle.update).toHaveBeenCalled());
    fireEvent.click(screen.getByText('Fe'));
    await waitFor(() => {
      const lastCall = fakeHandle.setOptions.mock.calls.at(-1);
      expect(lastCall?.[0].hidden.has('Fe')).toBe(true);
    });
  });
});
