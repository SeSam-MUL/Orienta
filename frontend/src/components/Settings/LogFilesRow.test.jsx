// @vitest-environment jsdom
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, cleanup, waitFor, fireEvent } from '@testing-library/react';

const mockGetLogInfo = vi.fn();
vi.mock('../../services/api', () => ({
  getLogInfo: (...a) => mockGetLogInfo(...a),
}));

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', yellow: '#fd0', cyan: '#0ff',
  },
  Label: ({ children, ...p }) => <label {...p}>{children}</label>,
}));

import LogFilesRow from './LogFilesRow';

const LOG_DIR = '/home/x/.orienta/runtime/logs';

beforeEach(() => {
  mockGetLogInfo.mockResolvedValue({ log_dir: LOG_DIR, exists: true, files: [] });
});

afterEach(() => {
  cleanup();
  delete window.electronAPI;
  vi.clearAllMocks();
});

describe('LogFilesRow in the desktop app', () => {
  it('opens the log folder through the bridge and shows the real path', async () => {
    const openLogFolder = vi.fn().mockResolvedValue({ ok: true, path: LOG_DIR });
    window.electronAPI = { openLogFolder };
    const { getByText, getByDisplayValue } = render(<LogFilesRow />);
    fireEvent.click(getByText('Show log files'));
    await waitFor(() => expect(openLogFolder).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(getByDisplayValue(LOG_DIR)).toBeTruthy());
  });

  it('says so when the folder could not be opened, and still shows the path', async () => {
    window.electronAPI = { openLogFolder: vi.fn().mockResolvedValue({ ok: false, path: LOG_DIR, error: 'no file manager' }) };
    const { getByText, getByDisplayValue } = render(<LogFilesRow />);
    fireEvent.click(getByText('Show log files'));
    await waitFor(() => expect(getByText(/could not be opened \(no file manager\)/)).toBeTruthy());
    expect(getByDisplayValue(LOG_DIR)).toBeTruthy();
  });

  it('falls back to the path the bridge reports when the backend does not answer', async () => {
    mockGetLogInfo.mockRejectedValue(new Error('down'));
    window.electronAPI = { openLogFolder: vi.fn().mockResolvedValue({ ok: true, path: '/from/main/logs' }) };
    const { getByText, getByDisplayValue } = render(<LogFilesRow />);
    fireEvent.click(getByText('Show log files'));
    await waitFor(() => expect(getByDisplayValue('/from/main/logs')).toBeTruthy());
  });
});

describe('LogFilesRow in a browser', () => {
  it('has no folder to open: it reveals the path as copyable text instead', async () => {
    const { getByText, queryByDisplayValue, getByDisplayValue } = render(<LogFilesRow />);
    await waitFor(() => expect(mockGetLogInfo).toHaveBeenCalled());
    expect(queryByDisplayValue(LOG_DIR)).toBeNull();            // hidden until asked for
    fireEvent.click(getByText('Show log files'));
    await waitFor(() => expect(getByDisplayValue(LOG_DIR)).toBeTruthy());
    expect(getByText(/Open this folder in your file manager/)).toBeTruthy();
  });

  it('copies the path to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue();
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const { getByText } = render(<LogFilesRow />);
    fireEvent.click(getByText('Show log files'));
    await waitFor(() => getByText('Copy path'));
    fireEvent.click(getByText('Copy path'));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(LOG_DIR));
    await waitFor(() => expect(getByText('Copied')).toBeTruthy());
  });

  it('tells the user when the backend is not answering', async () => {
    mockGetLogInfo.mockRejectedValue(new Error('down'));
    const { getByText } = render(<LogFilesRow />);
    fireEvent.click(getByText('Show log files'));
    await waitFor(() => expect(getByText(/log folder is not known yet/)).toBeTruthy());
  });
});
