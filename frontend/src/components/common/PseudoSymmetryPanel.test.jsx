// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import PseudoSymmetryPanel from './PseudoSymmetryPanel';

vi.mock('../../services/api', () => ({
  indexApi: {
    patternMatchVariants: vi.fn(),
    applyVariantToGrain: vi.fn(),
    undoGrainFlip: vi.fn(),
  },
}));
vi.mock('../../theme/tokens', () => ({
  colors: { bg: '#000', text: '#fff', textSecondary: '#aaa', border: '#444', accent: '#0ff' },
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}|${JSON.stringify(o)}` : k) }),
}));

import { indexApi } from '../../services/api';

const PIXEL = { row: 3, col: 5 };
const SPHERICAL = { indexing_method: 'spherical', r_quality: 'good' };
const VARIANTS = {
  point_group: 'm-3',
  candidates: [
    { label: 'current', r_score: 0.17, thumbnail: 'aGk=', quat: [1, 0, 0, 0], euler: [0, 0, 0] },
    { label: 'variant 1', r_score: 0.41, thumbnail: 'aGk=', quat: [0.7, 0, 0, 0.7], euler: [90, 0, 0] },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  cleanup();
});

describe('PseudoSymmetryPanel', () => {
  it('renders nothing for non-spherical results', () => {
    const { container } = render(
      <PseudoSymmetryPanel selectedPixel={PIXEL} matchData={{ indexing_method: 'hough' }} />);
    expect(container.innerHTML).toBe('');
  });

  it('shows the entry button, with suspicion hint when orientation came from Hough', () => {
    const { getByText, rerender, queryByText } = render(
      <PseudoSymmetryPanel selectedPixel={PIXEL}
        matchData={{ ...SPHERICAL, orientation_source: 'hough' }} />);
    getByText(/matchesDialog\.tryVariants/);
    getByText('matchesDialog.suspicionHint');
    rerender(<PseudoSymmetryPanel selectedPixel={PIXEL} matchData={SPHERICAL} />);
    expect(queryByText('matchesDialog.suspicionHint')).toBeNull();
  });

  it('loads and renders the variant gallery', async () => {
    indexApi.patternMatchVariants.mockResolvedValue({ data: VARIANTS });
    const { getByText } = render(
      <PseudoSymmetryPanel selectedPixel={PIXEL} matchData={SPHERICAL} />);
    fireEvent.click(getByText(/matchesDialog\.tryVariants/));
    await waitFor(() => getByText('variant 1'));
    // third arg = reference-pick options ({} when no reference is set)
    expect(indexApi.patternMatchVariants).toHaveBeenCalledWith(3, 5, {});
  });

  it('applies to grain with the refine flag, reports refine outcome, offers undo', async () => {
    indexApi.patternMatchVariants.mockResolvedValue({ data: VARIANTS });
    indexApi.applyVariantToGrain.mockResolvedValue({ data: {
      n_changed: 36, undo_available: true,
      refine: { status: 'applied', n_refined: 30, median_render_ncc_delta: 0.021 },
    } });
    indexApi.undoGrainFlip.mockResolvedValue({ data: { n_restored: 36 } });
    const onApplied = vi.fn();
    const { getByText } = render(
      <PseudoSymmetryPanel selectedPixel={PIXEL} matchData={SPHERICAL} onApplied={onApplied} />);
    fireEvent.click(getByText(/matchesDialog\.tryVariants/));
    await waitFor(() => getByText('variant 1'));
    fireEvent.click(getByText('variant 1'));
    fireEvent.click(getByText(/matchesDialog\.applyGrain\|/));
    await waitFor(() => expect(indexApi.applyVariantToGrain).toHaveBeenCalled());
    const arg = indexApi.applyVariantToGrain.mock.calls[0][0];
    expect(arg).toMatchObject({ row: 3, col: 5, refine: true, thresholdDeg: 5 });
    expect(arg.quat).toEqual([0.7, 0, 0, 0.7]);
    await waitFor(() => getByText(/refineApplied/));
    expect(onApplied).toHaveBeenCalledTimes(1);
    // undo
    fireEvent.click(getByText(/matchesDialog\.undoBtn/));
    await waitFor(() => expect(indexApi.undoGrainFlip).toHaveBeenCalled());
    await waitFor(() => getByText(/undoDone/));
    expect(onApplied).toHaveBeenCalledTimes(2);
  });

  it('refine checkbox off sends refine: false', async () => {
    indexApi.patternMatchVariants.mockResolvedValue({ data: VARIANTS });
    indexApi.applyVariantToGrain.mockResolvedValue({ data: { n_changed: 4, undo_available: true, refine: null } });
    const { getByText, getByRole } = render(
      <PseudoSymmetryPanel selectedPixel={PIXEL} matchData={SPHERICAL} />);
    fireEvent.click(getByText(/matchesDialog\.tryVariants/));
    await waitFor(() => getByText('variant 1'));
    fireEvent.click(getByText('variant 1'));
    fireEvent.click(getByRole('checkbox'));
    fireEvent.click(getByText(/matchesDialog\.applyGrain\|/));
    await waitFor(() => expect(indexApi.applyVariantToGrain).toHaveBeenCalled());
    expect(indexApi.applyVariantToGrain.mock.calls[0][0].refine).toBe(false);
  });

  it('surfaces a rejected refine honestly', async () => {
    indexApi.patternMatchVariants.mockResolvedValue({ data: VARIANTS });
    indexApi.applyVariantToGrain.mockResolvedValue({ data: {
      n_changed: 12, undo_available: true,
      refine: { status: 'rejected', reason: 'render_ncc_gate' },
    } });
    const { getByText } = render(
      <PseudoSymmetryPanel selectedPixel={PIXEL} matchData={SPHERICAL} />);
    fireEvent.click(getByText(/matchesDialog\.tryVariants/));
    await waitFor(() => getByText('variant 1'));
    fireEvent.click(getByText('variant 1'));
    fireEvent.click(getByText(/matchesDialog\.applyGrain\|/));
    await waitFor(() => getByText(/refineRejected/));
  });
});
