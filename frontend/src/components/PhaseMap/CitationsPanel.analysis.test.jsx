// @vitest-environment jsdom
// The citation panel on the Analysis page: `source="analysis"` asks GET /api/citations/analysis (the grains have
// no result id) and refetches whenever `refreshKey` changes; `null` hides it.
import { render, screen, waitFor, cleanup, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import CitationsPanel from './CitationsPanel';

afterEach(() => cleanup());

vi.mock('../../services/api', () => ({
  citationsApi: { forResult: vi.fn(), forAnalysis: vi.fn() },
}));
import { citationsApi } from '../../services/api';

// the keys of the real route's answer (backend/api/services/citations/bundle.py)
const payload = {
  bibtex: '@article{bachmann2011,\n  title = {Grain detection}\n}\n',
  methods: 'Grains were reconstructed at 5 deg.',
  plain: 'Bachmann, F. (2011). Grain detection.',
  steps: [{ key: 'analysis.grain_reconstruction', params: {} }],
  undeclared: [],
  warnings: [],
  app_version: { version: '0.4.8' },
};

beforeEach(() => {
  citationsApi.forResult.mockReset();
  citationsApi.forAnalysis.mockReset();
  citationsApi.forAnalysis.mockResolvedValue({ data: payload });
});

describe('CitationsPanel for the Analysis page', () => {
  it('renders nothing and asks nothing without grains', () => {
    const { container } = render(<CitationsPanel source="analysis" refreshKey={null} />);
    expect(container).toBeEmptyDOMElement();
    expect(citationsApi.forAnalysis).not.toHaveBeenCalled();
  });

  it('shows the grains\' methods text under its own title, never asking for a result', async () => {
    render(<CitationsPanel source="analysis" refreshKey="1|matlab_script" />);
    await waitFor(() => expect(screen.getByText(/reconstructed at 5 deg/)).toBeInTheDocument());
    expect(screen.getByText('Citations for these grains')).toBeInTheDocument();
    expect(citationsApi.forAnalysis).toHaveBeenCalledTimes(1);
    expect(citationsApi.forResult).not.toHaveBeenCalled();
  });

  it('asks again when the key changes, and drops an answer that comes too late', async () => {
    let resolveFirst;
    const first = new Promise((resolve) => { resolveFirst = resolve; });
    citationsApi.forAnalysis
      .mockImplementationOnce(() => first)
      .mockImplementationOnce(() => Promise.resolve({ data: { ...payload, methods: 'Now at 7 deg.' } }));
    const { rerender } = render(<CitationsPanel source="analysis" refreshKey="1" />);
    rerender(<CitationsPanel source="analysis" refreshKey="2" />);
    await waitFor(() => expect(screen.getByText(/Now at 7 deg/)).toBeInTheDocument());
    await act(async () => {
      resolveFirst({ data: { ...payload, methods: 'STALE settings.' } });
      await first;
    });
    expect(screen.queryByText(/STALE/)).not.toBeInTheDocument();
    expect(citationsApi.forAnalysis).toHaveBeenCalledTimes(2);
  });

  it('shows the backend\'s reason when there are no grains to cite', async () => {
    citationsApi.forAnalysis.mockRejectedValue({ response: { data: { detail: 'Reconstruct the grains first.' } } });
    render(<CitationsPanel source="analysis" refreshKey="1" />);
    await waitFor(() => expect(screen.getByText('Reconstruct the grains first.')).toBeInTheDocument());
  });

  // The Analysis side panel is narrow: in German the three tabs and "Kopieren" are wider than it, and without
  // wrapping the copy button was cut off behind a horizontal scrollbar (desktop check 2026-11-10).
  it('lets the tab row wrap, so the copy button stays inside a narrow panel', async () => {
    render(<CitationsPanel source="analysis" refreshKey="1" />);
    await waitFor(() => expect(screen.getByText(/reconstructed at 5 deg/)).toBeInTheDocument());
    const row = screen.getByRole('button', { name: 'BibTeX' }).parentElement;
    expect(row).toHaveStyle({ display: 'flex', flexWrap: 'wrap' });
    expect(row).toContainElement(screen.getByRole('button', { name: 'Copy' }));
  });

  it('a result panel keeps its own title', async () => {
    citationsApi.forResult.mockResolvedValue({ data: payload });
    render(<CitationsPanel resultId="r1" />);
    await waitFor(() => expect(screen.getByText('Citations for this result')).toBeInTheDocument());
    expect(citationsApi.forAnalysis).not.toHaveBeenCalled();
  });
});
