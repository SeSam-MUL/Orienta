// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import PatternExportDialog from '../PatternExportDialog';

vi.mock('../composePatternFigure', async (importOriginal) => {
  const mod = await importOriginal();
  return {
    ...mod,
    loadSources: vi.fn(async (m) => Object.fromEntries(
      Object.entries(m).filter(([, v]) => v).map(([k]) => [k, { _img: k }]))),
    composePatternFigure: vi.fn(async () => new Blob(['x'], { type: 'image/png' })),
    downloadFigure: vi.fn(),
    copyFigureToClipboard: vi.fn(async () => {}),
  };
});
import { composePatternFigure, downloadFigure } from '../composePatternFigure';

const sources = { experimental: 'AAA', simulated: 'BBB', ncc: null, heatmap: 'CCC' };

beforeEach(() => { vi.clearAllMocks(); cleanup(); });

describe('PatternExportDialog', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <PatternExportDialog open={false} onClose={() => {}} sources={sources} rNcc={{ r: 0.4, ncc: 0.6 }} stepUm={0.1} markers={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('builds a default model from non-null sources and exports', async () => {
    const { getByText } = render(
      <PatternExportDialog open onClose={() => {}} sources={sources} rNcc={{ r: 0.4, ncc: 0.6 }} stepUm={0.1} markers={[]} />);
    await waitFor(() => {}); // let loadSources resolve
    fireEvent.click(getByText('Export'));
    await waitFor(() => expect(composePatternFigure).toHaveBeenCalled());
    expect(downloadFigure).toHaveBeenCalled();
  });

  it('passes the chosen scale to the exporter', async () => {
    const { getByText } = render(
      <PatternExportDialog open onClose={() => {}} sources={sources} rNcc={{ r: 0.4, ncc: 0.6 }} stepUm={0.1} markers={[]} />);
    await waitFor(() => {});
    fireEvent.click(getByText('4×'));
    fireEvent.click(getByText('Export'));
    await waitFor(() => expect(composePatternFigure).toHaveBeenCalled());
    const opts = composePatternFigure.mock.calls[0][2];
    expect(opts.scale).toBe(4);
  });
});
