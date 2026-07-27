// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

afterEach(() => cleanup());

vi.mock('../../../services/api', () => ({
  indexApi: {
    phaseTestStart: vi.fn(), phaseTestProgress: vi.fn(),
    phaseTestCancel: vi.fn(), phaseTestPhases: vi.fn(),
    phaseTestRemask: vi.fn(), singlePixelPhaseTest: vi.fn(),
    phaseTestPixelChemistry: vi.fn(),
  },
  ebsdApi: { overview: vi.fn(), getPattern: vi.fn() },
  edsApi: { elements: vi.fn(), getMap: vi.fn() },
}));
vi.mock('../../../stores/useDataStore', () => ({
  default: (sel) => sel({ currentIndex: 11, gridShape: [3, 4],
    setPosition: () => {}, ebsdLoaded: true, isFileOpen: true }),
}));

import { indexApi, ebsdApi, edsApi } from '../../../services/api';
import SinglePixelPhaseTestDialog from '../SinglePixelPhaseTestDialog';

// The run flow polls indexApi.phaseTestProgress on a 500 ms interval, BUT it
// also fires one immediate poll right after start. By resolving
// phaseTestProgress with status:'done' on the very first call, the test is
// deterministic with NO fake timers — the immediate poll completes the run.
describe('SinglePixelPhaseTestDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64' } });
    ebsdApi.getPattern.mockResolvedValue({ data: { image: 'PATB64' } });
    indexApi.phaseTestPhases.mockResolvedValue({ data: [
      { key: 'Al', label: 'Al', space_group: 'Fm-3m' },
      { key: 'Al13Fe4', label: 'Al13Fe4', space_group: 'C2/m' },
    ] });
    indexApi.phaseTestPixelChemistry.mockResolvedValue({
      data: { pixel_index: 11, pixel_at_pct: null } });
    // Default: no EDS elements → the scan-picker overlay control stays hidden,
    // so existing tests still see only the mask-radius range input.
    edsApi.elements.mockResolvedValue({ data: { elements: [] } });
    edsApi.getMap.mockResolvedValue({ data: { image: 'EDSB64' } });
  });

  it('shows "preview failed" (not "no pattern") when the measured-pattern fetch errors', async () => {
    ebsdApi.getPattern.mockRejectedValue(new Error('boom'));
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    expect(await screen.findByText(/preview failed/i)).toBeInTheDocument();
  });

  it('renders EDS-overlay element chips when the file has EDS elements', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al', 'Fe'] } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    expect(await screen.findByRole('button', { name: 'Al' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Fe' })).toBeInTheDocument();
  });

  it('fetches the colour-tinted EDS map when an overlay element chip is toggled on', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al', 'Fe'] } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Fe' }));
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledWith(
      'Fe', 'counts', 'hot', expect.any(String)));
  });

  it('supports overlaying MULTIPLE EDS elements at once', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al', 'Fe', 'Mg'] } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Al' }));
    fireEvent.click(screen.getByRole('button', { name: 'Mg' }));
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledWith('Al', 'counts', 'hot', expect.any(String)));
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledWith('Mg', 'counts', 'hot', expect.any(String)));
  });

  it('renders nothing when closed', () => {
    const { container } = render(
      <SinglePixelPhaseTestDialog open={false} onClose={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows the active pixel index when open', () => {
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    expect(screen.getByDisplayValue('11')).toBeInTheDocument();
  });

  it('shows a measured-pattern preview for the active pixel', async () => {
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    const preview = await waitFor(() =>
      screen.getByAltText(/Measured pattern/i));
    expect(preview).toBeInTheDocument();
    expect(preview).toHaveAttribute('src', 'data:image/png;base64,PATB64');
  });

  it('fetches the preview with the Dynamic BG display filter by default (bgRemove on)', async () => {
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() => expect(ebsdApi.getPattern).toHaveBeenCalled());
    // BG-remove defaults ON → preview fetches the background-removed pattern for
    // the active pixel (row 2, col 3 for pixelIndex 11 in a 3×4 grid).
    expect(ebsdApi.getPattern).toHaveBeenLastCalledWith(
      2, 3, { params: { display_filter: 'Dynamic BG' } });
  });

  it('renders a "BG remove" checkbox checked by default', async () => {
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    const label = await waitFor(() =>
      screen.getByText('BG remove').closest('label'));
    const checkbox = label.querySelector('input[type="checkbox"]');
    expect(checkbox.checked).toBe(true);
  });

  it('Auto-Run start params include bg_remove: true by default', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JBG', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'running', done: 0, total: 2, current: 'Building…', error: null, result: null,
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() =>
      expect(indexApi.phaseTestStart).toHaveBeenCalledWith(
        expect.objectContaining({ bg_remove: true })));
  });

  it('unchecking "BG remove" re-fetches the raw preview and sends bg_remove: false', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JBG2', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'running', done: 0, total: 2, current: 'Building…', error: null, result: null,
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    // Default preview goes through the Dynamic BG filter.
    await waitFor(() => expect(ebsdApi.getPattern).toHaveBeenLastCalledWith(
      2, 3, { params: { display_filter: 'Dynamic BG' } }));
    // Uncheck BG remove → preview re-fetches the raw pattern (no config).
    const label = screen.getByText('BG remove').closest('label');
    fireEvent.click(label.querySelector('input[type="checkbox"]'));
    await waitFor(() =>
      expect(ebsdApi.getPattern).toHaveBeenLastCalledWith(2, 3, undefined));
    // Auto-Run now sends bg_remove: false.
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() =>
      expect(indexApi.phaseTestStart).toHaveBeenCalledWith(
        expect.objectContaining({ bg_remove: false })));
  });

  it('Quality dropdown defaults to Standard (128) and drives max_bandwidth on Auto-Run', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JQ', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'running', done: 0, total: 2, current: 'Building…', error: null, result: null,
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    const select = screen.getByTitle(/sharper simulated bands/i);
    expect(select.value).toBe('128');  // Standard default
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() =>
      expect(indexApi.phaseTestStart).toHaveBeenCalledWith(
        expect.objectContaining({ max_bandwidth: 128 })));
  });

  it('changing Quality to Fast (88) sends max_bandwidth: 88', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JQ2', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'running', done: 0, total: 2, current: 'Building…', error: null, result: null,
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.change(screen.getByTitle(/sharper simulated bands/i),
      { target: { value: '88' } });
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() =>
      expect(indexApi.phaseTestStart).toHaveBeenCalledWith(
        expect.objectContaining({ max_bandwidth: 88 })));
  });

  it('does NOT offer a "Sharp (256)" quality option', async () => {
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    const select = screen.getByTitle(/sharper simulated bands/i);
    const values = Array.from(select.querySelectorAll('option')).map((o) => o.value);
    expect(values).toEqual(['88', '128']);
    expect(screen.queryByText(/Sharp \(256\)/)).not.toBeInTheDocument();
  });

  it('shows the mask-radius value label and re-masks (no re-render) when the slider is released', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JR', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'done', done: 2, total: 2, current: 'Al', error: null,
      result: {
        experimental_png: 'E', eds_weighting_effective: 'off',
        mask_applied: true, bandwidth: 128,
        candidates: [
          { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
            r_score: 0.5, rank: 1, simulated_png: 'S', ncc_diff_png: 'D',
            euler_deg: [0, 0, 0] },
        ],
        excluded: [],
      },
    } });
    indexApi.phaseTestRemask.mockResolvedValue({ data: {
      experimental_png: 'E', mask_applied: true, bandwidth: 128,
      candidates: [
        { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
          r_score: 0.5, rank: 1, simulated_png: 'S', ncc_diff_png: 'D' },
      ],
      excluded: [],
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    // Default mask radius 1.0 → "100%" label renders.
    expect(screen.getByText('100%')).toBeInTheDocument();
    // Complete one Auto-Run so a result exists.
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(screen.getByText(/R = 0.500/)).toBeInTheDocument());
    expect(indexApi.phaseTestStart).toHaveBeenCalledTimes(1);
    // Releasing the slider (mouseUp) re-masks via /remask — NOT a 2nd full run.
    const slider = document.querySelector('input[type="range"]');
    fireEvent.mouseUp(slider);
    await waitFor(() => expect(indexApi.phaseTestRemask).toHaveBeenCalledTimes(1));
    expect(indexApi.phaseTestStart).toHaveBeenCalledTimes(1);  // no re-render
  });

  it('unchecking the Mask checkbox after a completed run calls /remask with aperture: full', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JRM', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'done', done: 2, total: 2, current: 'Al', error: null,
      result: {
        experimental_png: 'E', eds_weighting_effective: 'off',
        mask_applied: true, bandwidth: 128,
        candidates: [
          { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
            r_score: 0.5, rank: 1, simulated_png: 'S', ncc_diff_png: 'D',
            euler_deg: [0, 0, 0] },
        ],
        excluded: [],
      },
    } });
    indexApi.phaseTestRemask.mockResolvedValue({ data: {
      experimental_png: 'E', mask_applied: false, bandwidth: 128,
      candidates: [
        { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
          r_score: 0.42, rank: 1, simulated_png: 'S', ncc_diff_png: 'D' },
      ],
      excluded: [],
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    // Complete one Auto-Run so a result + lastJobIdRef exist.
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(screen.getByText(/R = 0.500/)).toBeInTheDocument());
    // Uncheck the Mask checkbox → /remask with the unmasked aperture.
    const maskLabel = screen.getByText('Mask').closest('label');
    fireEvent.click(maskLabel.querySelector('input[type="checkbox"]'));
    await waitFor(() =>
      expect(indexApi.phaseTestRemask).toHaveBeenCalledWith(
        expect.objectContaining({ job_id: 'JRM', aperture: 'full' })));
    // The remasked (unmasked) result is shown; no 2nd full run.
    await waitFor(() => expect(screen.getByText(/R = 0.420/)).toBeInTheDocument());
    expect(indexApi.phaseTestStart).toHaveBeenCalledTimes(1);
  });

  it('auto-run start→poll→done populates the ranked list and loads the top phase', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'J1', total: 2, experimental_png: 'E',
      pixel_at_pct: { Al: 71 }, eds_weighting_effective: 'filter', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'done', done: 2, total: 2, current: 'Al13Fe4', error: null,
      result: {
        experimental_png: 'E', eds_weighting_effective: 'filter',
        mask_applied: true, bandwidth: 128,
        candidates: [
          { phase_key: 'Al13Fe4', formula: 'Al13Fe4', display_formula: 'Al13Fe4',
            r_score: 0.61, chemistry_fit: 0.94, rank: 1, simulated_png: 'S1',
            ncc_diff_png: 'D1', euler_deg: [1, 2, 3] },
          { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
            r_score: 0.14, chemistry_fit: 0.22, rank: 2, simulated_png: 'S2',
            ncc_diff_png: 'D2', euler_deg: [4, 5, 6] },
        ],
        excluded: [{ phase_key: 'Ni', formula: 'Ni', display_formula: 'Ni',
          excluded_by_eds: true }],
      },
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(screen.getByText('Al13Fe4')).toBeInTheDocument());
    expect(screen.getByText(/R = 0.61/)).toBeInTheDocument();   // top loaded
    expect(screen.getByText(/1 excluded/i)).toBeInTheDocument();
    // Default Quality (Standard) sends max_bandwidth: 128.
    expect(indexApi.phaseTestStart).toHaveBeenCalledWith(
      expect.objectContaining({ max_bandwidth: 128 }));
  });

  it('clicking a row swaps the comparison to that phase', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'J2', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'done', done: 2, total: 2, current: 'Al', error: null,
      result: {
        experimental_png: 'E', eds_weighting_effective: 'off',
        mask_applied: false, bandwidth: 128,
        candidates: [
          { phase_key: 'Al13Fe4', formula: 'Al13Fe4', display_formula: 'Al13Fe4',
            r_score: 0.61, rank: 1, simulated_png: 'S1', ncc_diff_png: 'D1',
            euler_deg: [1, 2, 3] },
          { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
            r_score: 0.14, rank: 2, simulated_png: 'S2', ncc_diff_png: 'D2',
            euler_deg: [4, 5, 6] },
        ],
        excluded: [],
      },
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(screen.getByText('Al')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Al'));
    expect(screen.getByText(/R = 0.14/)).toBeInTheDocument();
  });

  it('defaults the mask to the inscribed circle (aperture: circular)', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JM', total: 2, experimental_png: 'E', row: 2, col: 3,
    } });
    // Keep the job running (done===0) so the start params are the only thing
    // under test here; no need to drive it to done.
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'running', done: 0, total: 2, current: 'Building phase models — first run is slower…',
      error: null, result: null,
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() =>
      expect(indexApi.phaseTestStart).toHaveBeenCalledWith(
        expect.objectContaining({ aperture: 'circular' })));
  });

  it('shows the cold-build "Building phase models" message while done===0', async () => {
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JB', total: 25, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'running', done: 0, total: 25,
      current: 'Building phase models — first run is slower…',
      error: null, result: null,
    } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    // The building text + the first-run sub-note render; Cancel stays visible.
    await waitFor(() =>
      expect(screen.getByText(/Building phase models/i)).toBeInTheDocument());
    expect(screen.getByText(/first run builds the models/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument();
  });

  it('renders the phase picker and toggling selection updates the run-button label', async () => {
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    // Phases fetched → picker header shows the count (2/2) and run button
    // defaults to "all phases".
    const header = await waitFor(() => screen.getByText(/Phases \(2\/2\)/));
    expect(screen.getByRole('button', { name: /Auto-Run all phases/i })).toBeInTheDocument();
    // Expand the picker, then deselect one phase via its row's checkbox.
    fireEvent.click(header);
    // The "Al13Fe4" phase row renders its label; its checkbox is the input
    // inside the same <label> element.
    const phaseLabel = await waitFor(() =>
      screen.getByText('Al13Fe4').closest('label'));
    expect(phaseLabel).toBeTruthy();
    const checkbox = phaseLabel.querySelector('input[type="checkbox"]');
    expect(checkbox.checked).toBe(true);
    fireEvent.click(checkbox);
    // Strict subset selected → header count + run-button label both update.
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Run 1 selected/i })).toBeInTheDocument());
    expect(screen.getByText(/Phases \(1\/2\)/)).toBeInTheDocument();
  });

  it('fetches live EDS chemistry for the active pixel on open', async () => {
    indexApi.phaseTestPixelChemistry.mockResolvedValue({
      data: { pixel_index: 11, pixel_at_pct: { Al: 76.7, Fe: 7.0, Si: 10.7 } } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() =>
      expect(indexApi.phaseTestPixelChemistry).toHaveBeenCalledWith(11));
    // The readout shows the live chemistry (elements >=0.5 at%) for pixel 11
    // (row 2, col 3 in a 3×4 grid).
    await waitFor(() =>
      expect(screen.getByText(/EDS @ \(2, 3\): Al 77% · Fe 7% · Si 11%/)).toBeInTheDocument());
  });

  it('re-fetches live EDS chemistry when the pixel index changes', async () => {
    indexApi.phaseTestPixelChemistry.mockResolvedValue({
      data: { pixel_index: 11, pixel_at_pct: null } });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    await waitFor(() =>
      expect(indexApi.phaseTestPixelChemistry).toHaveBeenCalledWith(11));
    // Move the crosshair via the pixel-index input → chemistry re-fetches for
    // the new pixel (this is the fix for the "frozen on first pixel" bug).
    const input = screen.getByDisplayValue('11');
    fireEvent.change(input, { target: { value: '5000' } });
    await waitFor(() =>
      expect(indexApi.phaseTestPixelChemistry).toHaveBeenCalledWith(5000));
  });

  it('"Use for indexing" adds the phase, keeps the dialog OPEN, and shows a confirmation', async () => {
    const onClose = vi.fn();
    const onUsePhase = vi.fn(() => 'added');
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JU', total: 1, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'done', done: 1, total: 1, current: 'Al', error: null,
      result: {
        experimental_png: 'E', eds_weighting_effective: 'off',
        mask_applied: false, bandwidth: 128,
        candidates: [
          { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
            r_score: 0.5, rank: 1, simulated_png: 'S', ncc_diff_png: 'D',
            euler_deg: [0, 0, 0], cif_path: 'C:/lib/Al.cif' },
        ],
        excluded: [],
      },
    } });
    render(<SinglePixelPhaseTestDialog open onClose={onClose}
      currentMethod="hough" onUsePhase={onUsePhase} />);
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(screen.getByText(/R = 0.500/)).toBeInTheDocument());
    // Click the (enabled, has cif_path) Hough "Use for indexing" button.
    fireEvent.click(screen.getByRole('button', { name: /✓ Hough/ }));
    expect(onUsePhase).toHaveBeenCalledWith(
      expect.objectContaining({ phase_key: 'Al' }), 'hough');
    // Dialog must NOT close — the user keeps collecting phases across pixels.
    expect(onClose).not.toHaveBeenCalled();
    // And a confirmation line appears.
    expect(await screen.findByText(/added to the Hough indexing list/i)).toBeInTheDocument();
  });

  it('"Use for indexing" on a phase already in the list shows the "already in list" note', async () => {
    const onClose = vi.fn();
    const onUsePhase = vi.fn(() => 'exists');
    indexApi.phaseTestStart.mockResolvedValue({ data: {
      job_id: 'JE', total: 1, experimental_png: 'E', row: 2, col: 3,
    } });
    indexApi.phaseTestProgress.mockResolvedValue({ data: {
      status: 'done', done: 1, total: 1, current: 'Al', error: null,
      result: {
        experimental_png: 'E', eds_weighting_effective: 'off',
        mask_applied: false, bandwidth: 128,
        candidates: [
          { phase_key: 'Al', formula: 'Al', display_formula: 'Al',
            r_score: 0.5, rank: 1, simulated_png: 'S', ncc_diff_png: 'D',
            euler_deg: [0, 0, 0], cif_path: 'C:/lib/Al.cif' },
        ],
        excluded: [],
      },
    } });
    render(<SinglePixelPhaseTestDialog open onClose={onClose}
      currentMethod="hough" onUsePhase={onUsePhase} />);
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(screen.getByText(/R = 0.500/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /✓ Hough/ }));
    expect(onClose).not.toHaveBeenCalled();
    expect(await screen.findByText(/already in the Hough list/i)).toBeInTheDocument();
  });
});
