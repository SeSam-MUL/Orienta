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
const mockCheckForUpdate = vi.fn(() => Promise.resolve({
  available: false, reason: 'up_to_date', install_kind: 'git',
}));

vi.mock('../../services/api', () => ({
  getAppVersion: (...a) => mockGetAppVersion(...a),
  exportDiagnostics: (...a) => mockExportDiagnostics(...a),
  checkForUpdate: (...a) => mockCheckForUpdate(...a),
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

  it('credits PyEBSDIndex for the Hough indexing and the rest of the stack', () => {
    const { getByText } = render(<AboutSection />);
    const note = getByText(/PyEBSDIndex/);
    expect(note.textContent).toMatch(/Rowenhorst, Callahan & Ånes, J\. Appl\. Cryst\. 57, 3–19, 2024/);
    expect(note.textContent).toMatch(/kikuchipy, orix, diffsims and EMsoft/);
  });

  it('shows the app version fetched from the backend', async () => {
    const { getByText } = render(<AboutSection />);
    await waitFor(() => {
      expect(getByText(/2026-08-26 \(a1b2c3d\) · main/)).toBeTruthy();
    });
  });

  // This test used to make the FETCH fail and expect "unknown". That encoded
  // the defect Sebastian hit after installing build 3: the first start after a
  // runtime update is exactly when the backend is not up yet, the one fetch on
  // mount failed, and the line read "unknown (no git information found)"
  // forever while the endpoint was answering v0.4.6 the whole time. "Unknown"
  // is now reserved for what it says: the backend answered, and it does not
  // know its own version.
  it('says "unknown" when the backend answers and really does not know', async () => {
    mockGetAppVersion.mockResolvedValueOnce({ app: 'Orienta', version: 'unknown' });
    const { getByText } = render(<AboutSection />);
    await waitFor(() => {
      expect(getByText(/unknown \(no git information found\)/)).toBeTruthy();
    });
  });

  it('does not say "unknown" while the backend has not answered yet', async () => {
    let release;
    mockGetAppVersion.mockImplementationOnce(
      () => new Promise((_, reject) => { release = reject; }),
    );
    const { getByText, queryByText } = render(<AboutSection />);
    expect(getByText(/checking…/)).toBeTruthy();
    expect(queryByText(/unknown \(no git information found\)/)).toBeNull();
    release(new Error('backend down'));
  });

  // "Could not ask" and "asked, no git information" are different facts and
  // must read differently. Sebastian saw the second sentence while the first
  // was true, so the line accused git of something the backend had already
  // answered. This also keeps the string itself honest: without a test the
  // branch renders a bare i18n key and nobody notices.
  it('says the backend is not answering, not that git is missing', async () => {
    mockGetAppVersion.mockRejectedValueOnce(new Error('ECONNREFUSED'));
    const { getByText, queryByText } = render(<AboutSection />);
    await waitFor(() => {
      expect(getByText(/the backend is not answering yet/)).toBeTruthy();
    });
    expect(queryByText(/no git information found/)).toBeNull();
  });

  // The defect itself, at the wire: a failed fetch must be retried, and the
  // line must catch up on its own once the backend comes up. Without the
  // retry the second call never happens and the text stays on "checking…".
  it('retries a failed fetch and shows the version once the backend is up', async () => {
    vi.useFakeTimers();
    try {
      mockGetAppVersion
        .mockRejectedValueOnce(new Error('backend down'))
        .mockResolvedValueOnce({ app: 'Orienta', version: 'v0.4.6', source: 'version-file' });
      const { getByText } = render(<AboutSection />);
      expect(mockGetAppVersion).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1000);
      expect(mockGetAppVersion).toHaveBeenCalledTimes(2);
      // Drain the resolved promise on the fake clock. `vi.waitFor` here would
      // spend its own 1000 ms budget racing the hook's 1000 ms retry — a
      // review measured 9-14 ms of margin, which is a flake on a slower box.
      await vi.advanceTimersByTimeAsync(0);
      expect(getByText(/v0\.4\.6/)).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });

  it('backs off instead of hammering a backend that stays down', async () => {
    vi.useFakeTimers();
    try {
      mockGetAppVersion.mockRejectedValue(new Error('backend down'));
      render(<AboutSection />);
      expect(mockGetAppVersion).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1000);   // 1 s
      expect(mockGetAppVersion).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(1000);   // next wait is 2 s, not 1 s
      expect(mockGetAppVersion).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(1000);
      expect(mockGetAppVersion).toHaveBeenCalledTimes(3);
    } finally {
      mockGetAppVersion.mockReset();
      vi.useRealTimers();
    }
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
      expect(mockExportDiagnostics.mock.calls[0][0])
        .toMatchObject({ description: 'it broke' });
      expect(mockDownloadBlob).toHaveBeenCalledTimes(1);
    });
    const [, filename] = mockDownloadBlob.mock.calls[0];
    expect(filename).toMatch(/^orienta-problem-report-\d{4}-\d{2}-\d{2}\.zip$/);
  });
});

// A manual check must SAY something. The reason-to-message map used to render
// nothing at all for a reason nobody had listed — the user pressed the button
// and the line stayed empty, which reads as "everything is fine". An installed
// copy now asks GitHub directly, so it can hear reasons a git checkout never
// hears (rate limits above all: GitHub counts unauthenticated calls per IP, so
// one busy university network is enough).
describe('AboutSection — what a manual update check says', () => {
  const clickCheck = async (result) => {
    mockCheckForUpdate.mockResolvedValueOnce(result);
    const view = render(<AboutSection />);
    fireEvent.click(view.getByText('Check for updates'));
    return view;
  };

  it('names a rate limit instead of claiming the app is current', async () => {
    const { getByText, queryByText } = await clickCheck({
      available: false, reason: 'rate_limited', install_kind: 'bundle',
    });
    await waitFor(() => {
      expect(getByText(/too many requests from this network/)).toBeTruthy();
    });
    expect(queryByText('You have the latest version.')).toBeNull();
  });

  it('says the raw reason rather than nothing when it does not know one', async () => {
    const { getByText } = await clickCheck({
      available: false, reason: 'something_nobody_mapped', install_kind: 'bundle',
    });
    await waitFor(() => {
      expect(getByText(/something_nobody_mapped/)).toBeTruthy();
    });
  });

  it('still says "up to date" when that is the answer', async () => {
    const { getByText } = await clickCheck({
      available: false, reason: 'up_to_date', install_kind: 'bundle',
    });
    await waitFor(() => {
      expect(getByText('You have the latest version.')).toBeTruthy();
    });
  });
});
