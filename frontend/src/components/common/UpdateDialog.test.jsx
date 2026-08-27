// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, fireEvent, waitFor } from '@testing-library/react';

const mockStart = vi.fn(() => Promise.resolve({ started: true }));
const mockProgress = vi.fn(() => Promise.resolve({ state: 'running', step: 'build', log: [] }));
vi.mock('../../services/api', () => ({
  startUpdate: (...a) => mockStart(...a),
  getUpdateProgress: (...a) => mockProgress(...a),
}));

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', border: '#444', text: '#fff',
    textSecondary: '#aaa', accent: '#0ff', textOnAccent: '#000',
    green: '#0f0', red: '#f00', yellow: '#fd0', cyan: '#0ff',
  },
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

import UpdateDialog from './UpdateDialog';

const gitInfo = {
  install_kind: 'git', current: 'v0.1.0', latest: 'v0.2.0',
  notes: '## v0.2.0\n- something new',
};
const zipInfo = { install_kind: 'zip', current: 'v0.1.0', latest: 'v0.2.0', notes: '' };

describe('UpdateDialog — a git install', () => {
  it('names both versions and shows what will happen', () => {
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    expect(getByText(/You have v0\.1\.0/)).toBeTruthy();
    expect(getByText(/available: v0\.2\.0/)).toBeTruthy();
    expect(getByText(/rebuild the interface and then restart/)).toBeTruthy();
  });

  it('shows the release notes', () => {
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    expect(getByText(/- something new/)).toBeTruthy();
  });

  it('installs the offered version', async () => {
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    fireEvent.click(getByText('Install update'));
    await waitFor(() => expect(mockStart).toHaveBeenCalledWith('v0.2.0'));
  });

  it('reports the running step and warns against closing', async () => {
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    fireEvent.click(getByText('Install update'));
    await waitFor(() => expect(getByText(/Building the interface/)).toBeTruthy());
    expect(getByText(/leave this window open/)).toBeTruthy();
  });

  it('offers a restart when the update finished', async () => {
    mockProgress.mockResolvedValue({ state: 'done', step: 'done', installed: 'v0.2.0', log: [] });
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    fireEvent.click(getByText('Install update'));
    await waitFor(() => expect(getByText('Restart now')).toBeTruthy());
    expect(getByText(/Update installed/)).toBeTruthy();
  });

  it('says the old version still works when the update failed', async () => {
    mockProgress.mockResolvedValue({
      state: 'failed', step: 'build', error: 'Building the interface failed.', log: ['x'],
    });
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    fireEvent.click(getByText('Install update'));
    await waitFor(() => expect(getByText(/Building the interface failed/)).toBeTruthy());
    expect(getByText(/previous version is still installed/)).toBeTruthy();
  });

  it('surfaces a refusal from the backend', async () => {
    mockStart.mockRejectedValueOnce({
      response: { data: { detail: '3 file(s) have local changes.' } },
    });
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={() => {}} onSkip={() => {}} />);
    fireEvent.click(getByText('Install update'));
    await waitFor(() => expect(getByText(/local changes/)).toBeTruthy());
    expect(mockProgress).not.toHaveBeenCalled();
  });

  it('later and skip are distinct actions', () => {
    const onClose = vi.fn();
    const onSkip = vi.fn();
    const { getByText } = render(<UpdateDialog info={gitInfo} onClose={onClose} onSkip={onSkip} />);
    fireEvent.click(getByText('Later'));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(getByText('Skip this version'));
    expect(onSkip).toHaveBeenCalledTimes(1);
  });
});

describe('UpdateDialog — a zip install', () => {
  it('explains why it cannot update and offers no install button', () => {
    const { getByText, queryByText } = render(
      <UpdateDialog info={zipInfo} onClose={() => {}} onSkip={() => {}} />,
    );
    expect(getByText(/cannot update itself/)).toBeTruthy();
    expect(getByText(/downloaded archive rather than a git clone/)).toBeTruthy();
    expect(queryByText('Install update')).toBeNull();
  });
});
