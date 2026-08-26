// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, fireEvent, waitFor } from '@testing-library/react';

const mockExport = vi.fn(() => Promise.resolve(new Blob(['zip'])));
vi.mock('../../services/api', () => ({
  exportDiagnostics: (...a) => mockExport(...a),
}));

const mockDownload = vi.fn();
vi.mock('./imageExport', () => ({
  downloadBlob: (...a) => mockDownload(...a),
}));

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', border: '#444', text: '#fff',
    textSecondary: '#aaa', accent: '#0ff', textOnAccent: '#000',
    green: '#0f0', red: '#f00',
  },
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

import ProblemReportDialog from './ProblemReportDialog';

describe('ProblemReportDialog', () => {
  it('sends the typed description with the report', async () => {
    const { getByRole, getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.change(getByRole('textbox'), {
      target: { value: 'Indexing stopped at 40% and the map stayed empty' },
    });
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(mockExport).toHaveBeenCalledTimes(1));
    expect(mockExport.mock.calls[0][0]).toEqual({
      description: 'Indexing stopped at 40% and the map stayed empty',
    });
  });

  it('downloads the zip under a problem-report name', async () => {
    const { getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(mockDownload).toHaveBeenCalledTimes(1));
    const [, filename] = mockDownload.mock.calls[0];
    expect(filename).toMatch(/^orienta-problem-report-\d{4}-\d{2}-\d{2}\.zip$/);
  });

  it('works without a description — an empty report still carries the logs', async () => {
    const { getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(mockExport).toHaveBeenCalled());
    expect(mockExport.mock.calls[0][0]).toEqual({ description: '' });
    expect(mockDownload).toHaveBeenCalled();
  });

  it('confirms success and offers Close afterwards', async () => {
    const { getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(getByText(/Report saved/)).toBeTruthy());
    expect(getByText('Close')).toBeTruthy();
  });

  it('shows an error and does not download when the export fails', async () => {
    mockExport.mockRejectedValueOnce(new Error('backend down'));
    const { getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(getByText(/Export failed/)).toBeTruthy());
    expect(mockDownload).not.toHaveBeenCalled();
  });

  it('states that no measurement data is included and nothing is auto-sent', () => {
    const { getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    expect(getByText(/NO measurement data/)).toBeTruthy();
    expect(getByText(/Nothing is sent automatically/)).toBeTruthy();
  });

  it('closes on Escape and on Cancel', () => {
    const onClose = vi.fn();
    const { getByText } = render(<ProblemReportDialog onClose={onClose} />);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
