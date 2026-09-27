// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import i18n from '../../i18n';
import Sidebar from './Sidebar';

// Sidebar pulls the theme from context — stub it so the test focuses on i18n.
vi.mock('../../theme/ThemeProvider', () => ({
  useTheme: () => ({ theme: 'dracula', setTheme: () => {} }),
}));

const pages = [
  { id: '_section', sectionKey: 'data' },
  { id: 'dashboard', shortcut: '1', icon: 'LayoutDashboard' },
  { id: 'eds', shortcut: '3', icon: 'Atom' },
];

function renderSidebar() {
  return render(
    <Sidebar
      pages={pages}
      currentPage="dashboard"
      onNavigate={() => {}}
      backendStatus="connected"
      onToggleH5Viewer={() => {}}
      h5ViewerOpen={false}
      collapsed={false}
      onToggleCollapse={() => {}}
      dataReady={{}}
    />,
  );
}

afterEach(() => cleanup());

describe('Sidebar i18n', () => {
  it('renders nav labels and live-switches en → de → ja', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    renderSidebar();
    expect(screen.getByText('Dashboard')).toBeTruthy();
    expect(screen.getByText('EDS Analysis')).toBeTruthy();
    expect(screen.getByText('Data')).toBeTruthy(); // section header

    await act(async () => { await i18n.changeLanguage('de'); });
    expect(screen.getByText('EDS-Analyse')).toBeTruthy();
    expect(screen.getByText('Daten')).toBeTruthy();

    await act(async () => { await i18n.changeLanguage('ja'); });
    expect(screen.getByText('EDS 解析')).toBeTruthy();
    expect(screen.getByText('データ')).toBeTruthy();

    // reset the shared singleton so unrelated tests see the default language
    await act(async () => { await i18n.changeLanguage('en'); });
  });
});

/**
 * The backend state in the status row.
 *
 * Sidebar calls useTranslation(['nav', 'shell']), and react-i18next binds `t`
 * to the FIRST namespace. A bare key like 'sidebar.connected' therefore
 * resolves against `nav`, misses, and renders as the key itself — which this
 * branch shipped for one commit, in every language. The namespace has to be
 * written out.
 */
describe('Sidebar backend state', () => {
  it('names the state, never a translation key', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    const { container } = renderSidebar();
    expect(container.textContent).toContain('Verbunden');
    expect(container.textContent).not.toContain('sidebar.');
    cleanup();
  });

  it('says it in English too, and in Japanese', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    let r = renderSidebar();
    expect(r.container.textContent).toContain('Connected');
    expect(r.container.textContent).not.toContain('sidebar.');
    cleanup();

    await act(async () => { await i18n.changeLanguage('ja'); });
    r = renderSidebar();
    expect(r.container.textContent).toContain('接続済み');
    expect(r.container.textContent).not.toContain('sidebar.');
    cleanup();
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('shows a disconnected backend by name, not by raw status', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    const { container } = render(
      <Sidebar
        pages={pages} currentPage="dashboard" onNavigate={() => {}}
        backendStatus="disconnected" onToggleH5Viewer={() => {}}
        h5ViewerOpen={false} collapsed={false} onToggleCollapse={() => {}}
        dataReady={{}}
      />,
    );
    expect(container.textContent).toContain('Getrennt');
    expect(container.textContent).not.toContain('disconnected');
    cleanup();
    await act(async () => { await i18n.changeLanguage('en'); });
  });
});
