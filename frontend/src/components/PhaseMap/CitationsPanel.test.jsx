// @vitest-environment jsdom
// NOTE: the task brief's test used `@testing-library/user-event`, which is
// not a dependency of this project (only @testing-library/react is). Adding
// it would violate the "no new third-party dependency" constraint, so this
// uses `fireEvent.click` from the already-present @testing-library/react —
// same observable behaviour (a click on the Copy button), no new package.
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import CitationsPanel from './CitationsPanel';

// This project's vitest config does not set `test.globals: true`, so
// @testing-library/react's automatic per-test cleanup (which hooks a global
// `afterEach`) never registers — every other jsdom test in this repo that
// renders more than once explicitly calls `cleanup()` (see e.g.
// AnnotationToolbar.drag.test.jsx). Without it, DOM from earlier `it()`
// blocks in this file lingers and later `getByRole`/`queryByText` calls see
// more than one match.
afterEach(() => cleanup());

vi.mock('../../services/api', () => ({
  citationsApi: { forResult: vi.fn() },
}));
import { citationsApi } from '../../services/api';

const payload = {
  bibtex: '@misc{orienta,\n  title = {Orienta}\n}\n',
  methods: 'Orientations were determined by Hough/Radon indexing.',
  plain: 'Samberger, S. (2026). Orienta. https://doi.org/10.5281/zenodo.22664080',
  steps: [{ key: 'indexing.hough', params: {} }],
  undeclared: [],
  app_version: { version: '0.3.0' },
};

beforeEach(() => {
  citationsApi.forResult.mockReset();
  citationsApi.forResult.mockResolvedValue({ data: payload });
});

describe('CitationsPanel', () => {
  it('renders nothing without a result id', () => {
    const { container } = render(<CitationsPanel resultId={null} />);
    expect(container).toBeEmptyDOMElement();
    expect(citationsApi.forResult).not.toHaveBeenCalled();
  });

  it('shows the methods paragraph for the given result', async () => {
    render(<CitationsPanel resultId="hough_1" />);
    await waitFor(() =>
      expect(screen.getByText(/Hough\/Radon indexing/)).toBeInTheDocument());
    expect(citationsApi.forResult).toHaveBeenCalledWith('hough_1');
  });

  it('refetches when the result id changes, and never merges the two', async () => {
    const { rerender } = render(<CitationsPanel resultId="a" />);
    await waitFor(() => expect(citationsApi.forResult).toHaveBeenCalledWith('a'));
    citationsApi.forResult.mockResolvedValue({
      data: { ...payload, methods: 'Spherical-harmonic indexing at bandwidth 88.' },
    });
    rerender(<CitationsPanel resultId="b" />);
    await waitFor(() =>
      expect(screen.getByText(/bandwidth 88/)).toBeInTheDocument());
    expect(screen.queryByText(/Hough\/Radon/)).not.toBeInTheDocument();
  });

  it('copies the active format to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue();
    Object.assign(navigator, { clipboard: { writeText } });
    render(<CitationsPanel resultId="hough_1" />);
    await waitFor(() => screen.getByText(/Hough\/Radon indexing/));
    fireEvent.click(screen.getByRole('button', { name: /copy/i }));
    expect(writeText).toHaveBeenCalledWith(payload.methods);
  });

  it('shows undeclared steps instead of hiding them', async () => {
    citationsApi.forResult.mockResolvedValue({
      data: { ...payload, undeclared: ['some.addon.step'] },
    });
    render(<CitationsPanel resultId="hough_1" />);
    await waitFor(() =>
      expect(screen.getByText(/some\.addon\.step/)).toBeInTheDocument());
  });
});
