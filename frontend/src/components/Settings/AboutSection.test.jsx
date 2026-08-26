// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, waitFor, fireEvent } from '@testing-library/react';

const mockGetAppVersion = vi.fn(() => Promise.resolve({
  app: 'Orienta',
  version: '2026-08-26 (a1b2c3d)',
  commit: 'a1b2c3d',
  branch: 'main',
  source: 'git',
}));
const mockExportDiagnostics = vi.fn(() => Promise.resolve(new Blob(['zip'])));

vi.mock('../../services/api', () => ({
  getAppVersion: (...a) => mockGetAppVersion(...a),
  exportDiagnostics: (...a) => mockExportDiagnostics(...a),
}));

const mockDownloadBlob = vi.fn();
vi.mock('../common/imageExport', () => ({
  downloadBlob: (...a) => mockDownloadBlob(...a),
}));

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', yellow: '#fd0', cyan: '#0ff',
  },
  alpha: () => '#000',
  spacing: { sm: 4, md: 8, lg: 16, innerSpacing: 8 },
  GroupBox: ({ title, children }) => <fieldset><legend>{title}</legend>{children}</fieldset>,
  Label: ({ children, ...p }) => <label {...p}>{children}</label>,
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

import AboutSection from './AboutSection';

describe('AboutSection', () => {
  it('shows the GPL appropriate legal notices: copyright, no-warranty, license', () => {
    const { getByText } = render(<AboutSection />);
    expect(getByText(/Copyright © 2026/)).toBeTruthy();
    expect(getByText(/ABSOLUTELY NO WARRANTY/)).toBeTruthy();
    expect(getByText(/GNU General Public License, version 3/)).toBeTruthy();
  });

  it('links to the full license text and the source code', () => {
    const { getByText } = render(<AboutSection />);
    const license = getByText(/Full license text/).closest('a');
    expect(license.getAttribute('href')).toBe('https://www.gnu.org/licenses/gpl-3.0.html');
    const source = getByText(/Source code/).closest('a');
    expect(source.getAttribute('href')).toBe('https://github.com/SeSam-MUL/Orienta');
  });

  it('states local-only processing and the backup advice', () => {
    const { getByText } = render(<AboutSection />);
    expect(getByText(/locally on this machine/)).toBeTruthy();
    expect(getByText(/keep backups of your original data/)).toBeTruthy();
  });

  it('shows the app version fetched from the backend', async () => {
    const { getByText } = render(<AboutSection />);
    await waitFor(() => {
      expect(getByText(/2026-08-26 \(a1b2c3d\) · main/)).toBeTruthy();
    });
  });

  it('falls back to "unknown" when the version fetch fails', async () => {
    mockGetAppVersion.mockRejectedValueOnce(new Error('backend down'));
    const { getByText } = render(<AboutSection />);
    await waitFor(() => {
      expect(getByText(/unknown \(no git information found\)/)).toBeTruthy();
    });
  });

  it('offers the problem report and opens its dialog', async () => {
    const { getByText, getByRole } = render(<AboutSection />);
    fireEvent.click(getByText('Report a problem…'));
    // The dialog asks the one question no log can answer.
    expect(getByRole('dialog')).toBeTruthy();
    expect(
      getByText(/What were you doing, and what did you expect/),
    ).toBeTruthy();
  });

  it('creates the report through the dialog', async () => {
    const { getByText, getByRole } = render(<AboutSection />);
    fireEvent.click(getByText('Report a problem…'));
    fireEvent.change(getByRole('textbox'), { target: { value: 'it broke' } });
    fireEvent.click(getByText('Create report'));
    await waitFor(() => {
      expect(mockExportDiagnostics).toHaveBeenCalledWith({ description: 'it broke' });
      expect(mockDownloadBlob).toHaveBeenCalledTimes(1);
    });
    const [, filename] = mockDownloadBlob.mock.calls[0];
    expect(filename).toMatch(/^orienta-problem-report-\d{4}-\d{2}-\d{2}\.zip$/);
  });
});
