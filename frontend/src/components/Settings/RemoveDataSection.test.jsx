// @vitest-environment jsdom
/**
 * The removal section: who is offered it, and in what order it asks.
 *
 * The rules that decide what gets deleted live in electron/remove_data.js and
 * are tested against real folders. What is tested here is the part a user
 * sees: that Windows is not offered a second way to delete the same
 * gigabytes, that the library question comes only after the first yes, and
 * that closing the first question deletes nothing.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (k, o) => (o?.count != null ? `${k}:${o.count}` : k) }) }));
vi.mock('../../theme/components', () => ({
  colors: { text: '#fff', textSecondary: '#aaa', border: '#444', bgTertiary: '#111', green: '#0f0', red: '#f00', yellow: '#fd0' },
  spacing: { sm: 8 },
  GroupBox: ({ title, children }) => <section><h2>{title}</h2>{children}</section>,
  Label: ({ children }) => <span>{children}</span>,
}));

import RemoveDataSection, { shouldOffer } from './RemoveDataSection';

/** A click, awaited, so the state it sets has settled before the next line. */
const click = async (element) => { await act(async () => { fireEvent.click(element); }); };

const plan = {
  ok: true, home: '/Users/t/Library/Application Support/Orienta',
  entries: ['runtime', 'python'], foreign: [], library: '/…/runtime/Database',
};
let removeSpy;

beforeEach(() => {
  removeSpy = vi.fn().mockResolvedValue({ ok: true, removed: ['a', 'b'], kept: [] });
  window.electronAPI = {
    planDataRemoval: vi.fn().mockResolvedValue(plan),
    removeData: removeSpy,
  };
});
afterEach(() => { cleanup(); delete window.electronAPI; });

describe('who gets offered this', () => {
  it('is offered on macOS and Linux', () => {
    expect(shouldOffer('darwin', true)).toBe(true);
    expect(shouldOffer('linux', true)).toBe(true);
  });

  it('is NOT offered on Windows, which has an uninstaller for it', () => {
    expect(shouldOffer('win32', true)).toBe(false);
  });

  it('is not offered in a browser, which has no folder to remove', () => {
    expect(shouldOffer('darwin', false)).toBe(false);
  });

  it('renders nothing at all on Windows', () => {
    const { container } = render(<RemoveDataSection platform="win32" />);
    expect(container.innerHTML).toBe('');
  });
});

describe('the order it asks in', () => {
  it('asks about the library only after yes to the files', async () => {
    render(<RemoveDataSection platform="darwin" />);
    await screen.findByText('settings:removeData.button');
    await click(screen.getByText('settings:removeData.button'));
    expect(screen.getByText('settings:removeData.askFiles')).toBeTruthy();
    expect(screen.queryByText('settings:removeData.askLibrary')).toBeNull();
    await click(screen.getByText('settings:removeData.yes'));
    expect(screen.getByText('settings:removeData.askLibrary')).toBeTruthy();
    expect(removeSpy).not.toHaveBeenCalled();    // nothing has happened yet
  });

  it('keeps the library when that is the answer', async () => {
    render(<RemoveDataSection platform="darwin" />);
    await screen.findByText('settings:removeData.button');
    await click(screen.getByText('settings:removeData.button'));
    await click(screen.getByText('settings:removeData.yes'));
    await click(screen.getByText('settings:removeData.keepLibrary'));
    await waitFor(() => expect(removeSpy).toHaveBeenCalledWith({ removeLibrary: false }));
  });

  it('removes the library only on the explicit second yes', async () => {
    render(<RemoveDataSection platform="darwin" />);
    await screen.findByText('settings:removeData.button');
    await click(screen.getByText('settings:removeData.button'));
    await click(screen.getByText('settings:removeData.yes'));
    await click(screen.getByText('settings:removeData.yesLibrary'));
    await waitFor(() => expect(removeSpy).toHaveBeenCalledWith({ removeLibrary: true }));
  });

  it('cancelling the first question deletes nothing', async () => {
    render(<RemoveDataSection platform="darwin" />);
    await screen.findByText('settings:removeData.button');
    await click(screen.getByText('settings:removeData.button'));
    await click(screen.getByText('settings:removeData.no'));
    expect(removeSpy).not.toHaveBeenCalled();
    expect(screen.getByText('settings:removeData.button')).toBeTruthy();
  });

  it('shows the refusal instead of a button when the folder is not ours', async () => {
    window.electronAPI.planDataRemoval = vi.fn().mockResolvedValue({ ok: false, reason: 'isRepo' });
    render(<RemoveDataSection platform="linux" />);
    await screen.findByText('settings:removeData.refused.isRepo');
    expect(screen.queryByText('settings:removeData.button')).toBeNull();
  });
});

describe('the way SettingsPage actually renders it', () => {
  it('appears with no platform prop, because the preload says which OS this is', async () => {
    // The first version fell back to `process.platform`, and the renderer has
    // no `process` (contextIsolation, no node integration). It therefore
    // rendered NOTHING on the two platforms it exists for, and every test
    // passed because jsdom has a `process` and each test passed a prop.
    const saved = globalThis.process;
    try {
      delete globalThis.process;
      window.electronAPI.platform = 'darwin';
      render(<RemoveDataSection />);
      await screen.findByText('settings:removeData.title');
    } finally {
      globalThis.process = saved;
    }
  });

  it('still renders nothing on Windows when asked the same way', async () => {
    window.electronAPI.platform = 'win32';
    const { container } = render(<RemoveDataSection />);
    expect(container.innerHTML).toBe('');
  });
});
