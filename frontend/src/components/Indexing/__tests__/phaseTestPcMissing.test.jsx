// @vitest-environment jsdom
// The phase test refuses a file with no pattern centre with the coded 400 the
// indexing starts use; the dialog shows its own translation, not the English prose.
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

describe('phase test without a pattern centre', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64' } });
    ebsdApi.getPattern.mockResolvedValue({ data: { image: 'PATB64' } });
    indexApi.phaseTestPhases.mockResolvedValue({ data: [{ key: 'Al', label: 'Al', space_group: 'Fm-3m' }] });
    indexApi.phaseTestPixelChemistry.mockResolvedValue({ data: { pixel_index: 11, pixel_at_pct: null } });
    edsApi.elements.mockResolvedValue({ data: { elements: [] } });
    edsApi.getMap.mockResolvedValue({ data: { image: 'EDSB64' } });
  });

  it('shows the translated sentence of the code, not the English prose', async () => {
    indexApi.phaseTestStart.mockRejectedValue({
      response: {
        status: 400,
        headers: { 'x-orienta-code': 'pc_missing' },
        data: { detail: 'ENGLISH PROSE FROM THE BACKEND' },
      },
      message: 'Request failed with status code 400',
    });
    render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: /Auto-Run/i }));
    await waitFor(() => expect(indexApi.phaseTestStart).toHaveBeenCalled());
    expect(await screen.findByText(/carries no pattern centre/)).toBeInTheDocument();
    expect(screen.queryByText(/ENGLISH PROSE/)).toBeNull();
  });
});
