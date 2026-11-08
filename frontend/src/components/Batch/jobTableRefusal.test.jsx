// @vitest-environment jsdom
// A batch job whose Hough library was refused for memory is worded in the language of the page
// with the one-click answer for ITS phase (the notice of the Indexing and PC Refinement pages),
// not as the English text of the backend. Real JobTable and HoughRefusal; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';

const calls = vi.hoisted(() => ({ fit: [] }));

vi.mock('../../services/api', () => ({
  indexApi: {
    fitHoughReflectors: vi.fn((path) => {
      calls.fit.push(path);
      return Promise.resolve({ data: { fit: { rows: 50, predicted_bytes: null } } });
    }),
  },
}));

import JobTable from './JobTable';

const GiB = 1024 ** 3;
const refused = (extra = {}) => ({
  file_name: 'one.h5oina', phase_name: 'Mg', phase_path: 'D:/lib/Mg.cif', status: 'failed',
  error_msg: 'ENGLISH PROSE FROM THE BACKEND', error_code: 'hough_library_too_big',
  error_params: {
    phases: [{ name: 'Mg', rows: 70, predicted_bytes: 2 * GiB, fit_rows: 50, fit_families: 6,
      fit_reason: null, own_selection: false }],
    budget_bytes: GiB, free_bytes: 2 * GiB, min_rows: 40, min_families: 4,
  },
  ...extra,
});

beforeEach(() => { cleanup(); calls.fit = []; });

describe('a batch job refused for the size of its Hough library', () => {
  it('shows the translated notice with the phase, not the English prose', () => {
    render(<JobTable jobs={[refused()]} />);
    const notice = screen.getByTestId('hough-refusal');
    expect(notice.textContent).toMatch(/Hough indexing did not start/);
    expect(notice.textContent).toMatch(/Mg: 70 reflector rows would need a band-triplet library of 2.00 GiB/);
    expect(document.body.textContent).not.toMatch(/ENGLISH PROSE/);
  });

  it('the one click fits the list of the phase of THIS job', async () => {
    const second = refused({ phase_name: 'Al', phase_path: 'D:/lib/Al.cif' });
    second.error_params.phases[0].name = 'Al';
    render(<JobTable jobs={[refused(), second]} />);
    const [first, other] = screen.getAllByTestId('hough-refusal');
    fireEvent.click(within(other).getByTestId('hough-refusal-fit-Al'));
    await waitFor(() => expect(calls.fit).toEqual(['D:/lib/Al.cif']));
    fireEvent.click(within(first).getByTestId('hough-refusal-fit-Mg'));
    await waitFor(() => expect(calls.fit).toEqual(['D:/lib/Al.cif', 'D:/lib/Mg.cif']));
  });

  it('a row stored before the params existed still gets the translated notice', () => {
    render(<JobTable jobs={[refused({ error_params: null })]} />);
    const notice = screen.getByTestId('hough-refusal');
    expect(notice.textContent).toMatch(/Hough indexing did not start/);
    expect(within(notice).queryByRole('button')).toBeNull();
    expect(document.body.textContent).not.toMatch(/ENGLISH PROSE/);
  });

  it('any other failure keeps its own text', () => {
    render(<JobTable jobs={[refused({ error_code: 'pc_missing', error_params: null })]} />);
    expect(screen.queryByTestId('hough-refusal')).toBeNull();
    expect(screen.getByTestId('job-failure').textContent).toMatch(/carries no pattern centre/);
  });
});
