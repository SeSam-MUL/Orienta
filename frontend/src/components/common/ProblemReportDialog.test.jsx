// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, fireEvent, waitFor } from '@testing-library/react';

const mockExport = vi.fn(() => Promise.resolve(new Blob(['zip'])));
const mockVersion = vi.fn(() => Promise.resolve({
  version: 'v0.2.1', branch: 'main',
  repo_url: 'https://github.com/SeSam-MUL/Orienta',
}));
vi.mock('../../services/api', () => ({
  exportDiagnostics: (...a) => mockExport(...a),
  getAppVersion: (...a) => mockVersion(...a),
}));

vi.mock('../../services/errorReporter', () => ({
  getLastFingerprint: () => 'a1b2c3d4',
}));
vi.mock('../../services/breadcrumbs', () => ({
  formatBreadcrumbs: () => '  -1.0s [nav] page → indexing',
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
    expect(mockExport.mock.calls[0][0]).toMatchObject({
      description: 'Indexing stopped at 40% and the map stayed empty',
      fingerprint: 'a1b2c3d4',
    });
  });

  it('shows the error id so duplicates are recognisable', () => {
    const { getByText } = render(<ProblemReportDialog onClose={() => {}} />);
    expect(getByText('a1b2c3d4')).toBeTruthy();
  });

  it('includes the screenshot when one was captured', async () => {
    const { getByText } = render(
      <ProblemReportDialog onClose={() => {}} screenshot="BASE64PNG" />,
    );
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(mockExport).toHaveBeenCalled());
    expect(mockExport.mock.calls[0][0].screenshot).toBe('BASE64PNG');
  });

  it('lets the user leave the screenshot out', async () => {
    const { getByText, getByRole } = render(
      <ProblemReportDialog onClose={() => {}} screenshot="BASE64PNG" />,
    );
    fireEvent.click(getByRole('checkbox'));
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(mockExport).toHaveBeenCalled());
    expect(mockExport.mock.calls[0][0].screenshot).toBeNull();
  });

  it('offers no screenshot choice when none could be captured', () => {
    const { queryByRole } = render(<ProblemReportDialog onClose={() => {}} />);
    expect(queryByRole('checkbox')).toBeNull();
  });

  it('offers the GitHub issue only after the report exists', async () => {
    const { getByText, queryByText } = render(<ProblemReportDialog onClose={() => {}} />);
    await waitFor(() => expect(mockVersion).toHaveBeenCalled());
    expect(queryByText('Open a GitHub issue')).toBeNull();
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(getByText('Open a GitHub issue')).toBeTruthy());
  });

  it('opens a pre-filled issue carrying the error id and version', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    const { getByText, getByRole } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.change(getByRole('textbox'), { target: { value: 'it broke' } });
    fireEvent.click(getByText('Create report'));
    await waitFor(() => getByText('Open a GitHub issue'));
    fireEvent.click(getByText('Open a GitHub issue'));
    const url = decodeURIComponent(open.mock.calls[0][0]);
    expect(url).toContain('github.com/SeSam-MUL/Orienta/issues/new');
    expect(url).toContain('a1b2c3d4');
    expect(url).toContain('it broke');
    expect(url).toContain('v0.2.1');
    open.mockRestore();
  });

  it('hides the issue button when the repository is unknown', async () => {
    mockVersion.mockResolvedValueOnce({ version: 'v0.2.1', repo_url: null });
    const { getByText, queryByText } = render(<ProblemReportDialog onClose={() => {}} />);
    fireEvent.click(getByText('Create report'));
    await waitFor(() => expect(getByText(/Report saved/)).toBeTruthy());
    expect(queryByText('Open a GitHub issue')).toBeNull();
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
    expect(mockExport.mock.calls[0][0]).toMatchObject({ description: '' });
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
