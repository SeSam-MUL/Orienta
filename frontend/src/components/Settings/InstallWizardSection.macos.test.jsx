// @vitest-environment jsdom
/**
 * What the EMsoft install section may offer on each platform.
 *
 * A Mac tester on 2026-09-25 was shown, in one box: a line reading "macOS:
 * EMsoft + EMSphInx build natively (no WSL required)", a step numbered 1, then
 * a step numbered 3 containing a field labelled "WSL sudo password" and a
 * button "Run in WSL". The endpoint behind that field validates the password
 * against WSL, which does not exist on a Mac, so it could only ever fail.
 *
 * A password box that cannot work is worse than no box: it invites someone to
 * type a real password into a dead end.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, waitFor } from '@testing-library/react';

const mockWslStatus = vi.fn();
vi.mock('../../services/api', () => ({
  installApi: {
    wslStatus: (...a) => mockWslStatus(...a),
    installWsl: vi.fn(),
    createUser: vi.fn(),
    resetPassword: vi.fn(),
    validatePassword: vi.fn(),
  },
}));

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff', textOnAccent: '#000',
    green: '#0f0', red: '#f00', yellow: '#fd0', cyan: '#0ff',
  },
  alpha: () => '#000',
  spacing: { sm: 4, md: 8, lg: 16, innerSpacing: 8 },
  GroupBox: ({ title, children }) => <fieldset><legend>{title}</legend>{children}</fieldset>,
  Label: ({ children, ...p }) => <label {...p}>{children}</label>,
  Button: ({ children, ...p }) => <button type="button" {...p}>{children}</button>,
  Input: (p) => <input {...p} />,
  Select: ({ children, ...p }) => <select {...p}>{children}</select>,
}));

afterEach(() => { cleanup(); vi.clearAllMocks(); });

import InstallWizardSection from './InstallWizardSection';

const statusFor = (os) => ({
  data: {
    installed: true,
    distro: os === 'windows' ? 'Ubuntu' : 'native-linux',
    username: 'someone',
    corrupted: false,
    emsoft_installed: false,
    platform: { os, arch: os === 'macos' ? 'arm64' : 'x86_64', needs_wsl: os === 'windows' },
  },
});

const renderFor = async (os) => {
  mockWslStatus.mockResolvedValue(statusFor(os));
  const view = render(<InstallWizardSection />);
  await waitFor(() => expect(mockWslStatus).toHaveBeenCalled());
  return view;
};

describe('the EMsoft install section on macOS', () => {
  it('asks for no password at all', async () => {
    const { container } = await renderFor('macos');
    await waitFor(() => {
      expect(container.querySelectorAll('input[type="password"]').length).toBe(0);
    });
  });

  it('offers no button that would run something in WSL', async () => {
    const { container } = await renderFor('macos');
    await waitFor(() => {
      const labels = [...container.querySelectorAll('button')].map((b) => b.textContent);
      expect(labels.some((l) => /WSL/i.test(l || ''))).toBe(false);
    });
  });

  it('says EMsoft is optional and why it is not on offer', async () => {
    const { findByText } = await renderFor('macos');
    // The load-bearing half: the app works without it.
    expect(await findByText(/needs no EMsoft/i)).toBeTruthy();
    // And the honest half: it exists but nobody has run it.
    expect(await findByText(/nobody has ever run it/i)).toBeTruthy();
  });

  it('does not head the box with a green "no WSL needed" tick', async () => {
    // That header promised an installation the box no longer contains.
    const { queryByText } = await renderFor('macos');
    await waitFor(() => {
      expect(queryByText(/No WSL needed/i)).toBeNull();
    });
  });
});

describe('the EMsoft install section elsewhere', () => {
  it('still asks for the sudo password on Windows', async () => {
    const { container } = await renderFor('windows');
    await waitFor(() => {
      expect(container.querySelectorAll('input[type="password"]').length).toBeGreaterThan(0);
    });
  });

  it('asks Linux for no password either', async () => {
    // Linux joined macOS on 2026-09-25: the endpoint runs the script natively
    // there, but nobody has ever run it, so it is not offered. The field also
    // said "WSL sudo password" and the button "Run in WSL" -- on a machine
    // with no WSL.
    const { container } = await renderFor('linux');
    await waitFor(() => {
      expect(container.querySelectorAll('input[type="password"]').length).toBe(0);
    });
  });

  it('tells Linux the same thing, in its own words', async () => {
    const { findByText } = await renderFor('linux');
    expect(await findByText(/needs no EMsoft/i)).toBeTruthy();
    // apt, not Homebrew -- the sentence names the branch that exists there.
    expect(await findByText(/through apt/i)).toBeTruthy();
  });
});

describe('when the status request fails', () => {
  it('shows no WSL password field — on any platform', async () => {
    // wslStatus starts null and STAYS null on error: nothing retries. The
    // first gate here read "not macOS", and null is not macOS, so a Mac whose
    // backend call errored got the full WSL password step permanently. The
    // same hole made it flash on every normal load.
    mockWslStatus.mockRejectedValue(new Error('backend down'));
    const { container } = render(<InstallWizardSection />);
    await waitFor(() => expect(mockWslStatus).toHaveBeenCalled());
    await waitFor(() => {
      expect(container.querySelectorAll('input[type="password"]').length).toBe(0);
    });
  });

  it('says why, rather than showing nothing at all', async () => {
    mockWslStatus.mockRejectedValue(new Error('backend down'));
    const { findByText } = render(<InstallWizardSection />);
    expect(await findByText(/backend down/)).toBeTruthy();
  });
});
