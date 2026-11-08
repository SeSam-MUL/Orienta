// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import i18n from '../../i18n';
import BackendEndedNotice from './BackendEndedNotice';

// The shape electron/backend_exit.js backendEndNotice() sends.
const NOTICE = {
  at: '2026-11-08T17:30:01.000Z',
  exit: 'exit code 0xC0000005 (access violation: a native crash in Python or a library)',
  lastRequest: 'POST /api/phase-map/apply-to-grain',
  running: ['POST /api/phase-map/apply-to-grain'],
  commitMiB: 41250,
  restarting: true,
};

function fakeApi(last = null) {
  const listeners = new Set();
  return {
    onBackendEnded: (l) => { listeners.add(l); return () => listeners.delete(l); },
    lastBackendEnd: () => Promise.resolve(last),
    emit: (n) => listeners.forEach((l) => l(n)),
    listeners,
  };
}

beforeEach(async () => { sessionStorage.clear(); await i18n.changeLanguage('en'); });
afterEach(() => cleanup());

describe('BackendEndedNotice', () => {
  it('says nothing in a browser (no shell) and nothing before an end', async () => {
    const { container } = render(<BackendEndedNotice api={null} />);
    expect(container).toBeEmptyDOMElement();
    const api = fakeApi();
    render(<BackendEndedNotice api={api} />);
    await act(async () => {});
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('shows how it ended, what it was doing, and that it is restarted', async () => {
    const api = fakeApi();
    render(<BackendEndedNotice api={api} />);
    await act(async () => { api.emit(NOTICE); });
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('The backend stopped unexpectedly');
    expect(alert).toHaveTextContent('0xC0000005');
    expect(alert).toHaveTextContent('POST /api/phase-map/apply-to-grain');
    expect(alert).toHaveTextContent('Orienta is starting it again');
    expect(alert).toHaveTextContent('logs/backend-console.log');
  });

  it('says when it will not be restarted any more', async () => {
    const api = fakeApi();
    render(<BackendEndedNotice api={api} />);
    await act(async () => { api.emit({ ...NOTICE, restarting: false, lastRequest: null }); });
    expect(screen.getByRole('alert')).toHaveTextContent('not started again automatically');
  });

  it('a page loaded after the end still shows it, until it is dismissed', async () => {
    const api = fakeApi(NOTICE);
    const first = render(<BackendEndedNotice api={api} />);
    await act(async () => {});
    expect(screen.getByRole('alert')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByRole('alert')).toBeNull();
    first.unmount();
    // a reload: the same end is not brought back
    render(<BackendEndedNotice api={api} />);
    await act(async () => {});
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('removes its listener when it goes away', async () => {
    const api = fakeApi();
    const { unmount } = render(<BackendEndedNotice api={api} />);
    await act(async () => {});
    expect(api.listeners.size).toBe(1);
    unmount();
    expect(api.listeners.size).toBe(0);
  });

  it('is in every language', async () => {
    for (const lang of ['de', 'ja', 'zh']) {
      cleanup();
      await i18n.changeLanguage(lang);
      const api = fakeApi();
      render(<BackendEndedNotice api={api} />);
      await act(async () => { api.emit(NOTICE); });
      const text = screen.getByRole('alert').textContent;
      expect(text, lang).not.toContain('backendEnded.');
      expect(text, lang).toContain('0xC0000005');
    }
  });
});
