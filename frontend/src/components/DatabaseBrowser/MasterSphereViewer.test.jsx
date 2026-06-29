// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const sphere = vi.fn();
vi.mock('../../services/api', () => ({
  dbApi: { sphere: (...a) => sphere(...a) },
}));

// Plotly is heavy + jsdom-hostile and lazy-loaded; stub it with a marker.
vi.mock('react-plotly.js', () => ({
  default: (props) => (
    <div data-testid="sphere-plot" data-traces={(props.data || []).length} />
  ),
}));

import MasterSphereViewer from './MasterSphereViewer';

const PAYLOAD = {
  filename: 'Al3FeSi2 (..) [AlFe] {20kV}.sht',
  grid: [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
  phi: [0, 2.094, 4.188],
  theta: [0.4, 1.2, 2.6],
  bandwidth: 88,
  file_bandwidth: 384,
  meta: { formula: 'AlFe', space_group: 140, point_group: '4/mmm', voltage_kV: 20.0 },
};

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('MasterSphereViewer', () => {
  it('renders the sphere plot + metadata on success', async () => {
    sphere.mockResolvedValue({ data: PAYLOAD });
    render(<MasterSphereViewer filename={PAYLOAD.filename} isLocal />);
    // lazy Plot resolves after the fetch
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(sphere).toHaveBeenCalledWith(PAYLOAD.filename);
    // metadata caption
    expect(screen.getByText(/4\/mmm/)).toBeInTheDocument();
    expect(screen.getByText(/SG 140/)).toBeInTheDocument();
    expect(screen.getByText(/bw 88/)).toBeInTheDocument();
  });

  it('shows an error message when the fetch fails', async () => {
    sphere.mockRejectedValue({ response: { data: { detail: 'could not read this .sht' } } });
    render(<MasterSphereViewer filename={PAYLOAD.filename} isLocal />);
    expect(await screen.findByText(/could not read this .sht/i)).toBeInTheDocument();
    expect(screen.queryByTestId('sphere-plot')).not.toBeInTheDocument();
  });

  it('does NOT fetch for a non-local entry', async () => {
    render(<MasterSphereViewer filename={PAYLOAD.filename} isLocal={false} />);
    expect(screen.getByText(/download this file first/i)).toBeInTheDocument();
    await waitFor(() => expect(sphere).not.toHaveBeenCalled());
  });

  it('re-fetches with higher bandwidth when preset changes to Hoch', async () => {
    sphere.mockResolvedValue({ data: {
      grid: [[0.5]], phi: [0], theta: [1.57], bandwidth: 128, file_bandwidth: 384,
      meta: { formula: 'Ni', space_group: 225, point_group: 'm-3m', voltage_kV: 20 },
    }});
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    const select = await screen.findByLabelText(/resolution/i);
    fireEvent.change(select, { target: { value: 'high' } });
    await waitFor(() => {
      expect(sphere).toHaveBeenLastCalledWith(
        'Ni.sht', { max_bandwidth: 256, target_size: 192 });
    });
  });
});
