// @vitest-environment jsdom
//
// The ML Hub tile (Dashboard) and the sidebar entry exist only while
// GET /api/ml/status answers 200 with enabled === true. Hidden on
// {enabled: false}, hidden while the answer is still unknown (no response),
// shown on {enabled: true}.
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import fs from 'node:fs';
import path from 'node:path';

const mockStatus = vi.fn();
vi.mock('../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, mlApi: { ...actual.mlApi, status: (...a) => mockStatus(...a) } };
});
vi.mock('../../theme/ThemeProvider', () => ({
  useTheme: () => ({ theme: 'dracula', setTheme: () => {} }),
}));
// WebGL/canvas backgrounds are irrelevant here and do not run in jsdom.
vi.mock('../Dashboard/CrystalBackground', () => ({ default: () => null }));
vi.mock('../Dashboard/VantaBackground', () => ({ default: () => null }));
vi.mock('../Dashboard/WandererOverlay', () => ({ default: () => null }));

import i18n from '../../i18n';
import Sidebar from './Sidebar';
import Dashboard from '../Dashboard/Dashboard';
import { SIDEBAR_PAGES, visibleSidebarPages } from './sidebarPages';
import { __resetMlHubEnabledForTests } from '../../hooks/useMlHubEnabled';

const LABEL = 'ML Predictor'; // en nav:pages.mlhub.label and dashboard card title

const ON = { status: 200, data: { enabled: true } };
const OFF = { status: 200, data: { enabled: false } };

function renderSidebar() {
  return render(
    <Sidebar
      pages={SIDEBAR_PAGES}
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

beforeEach(async () => {
  __resetMlHubEnabledForTests();
  await act(async () => { await i18n.changeLanguage('en'); });
});
afterEach(() => {
  cleanup();
  __resetMlHubEnabledForTests();
  mockStatus.mockReset();
  vi.useRealTimers();
});

describe('sidebar entry', () => {
  it('is hidden on enabled:false, together with its now-empty Tools header', async () => {
    mockStatus.mockResolvedValue(OFF);
    renderSidebar();
    await act(async () => {});
    expect(screen.queryByText(LABEL)).toBeNull();
    expect(screen.queryByText('Tools')).toBeNull();
    expect(screen.getByText('EDS Analysis')).toBeTruthy(); // the rest is still there
  });

  it('is hidden while the answer is unknown (no response)', async () => {
    vi.useFakeTimers();
    mockStatus.mockRejectedValue(new Error('Network Error'));
    renderSidebar();
    await vi.advanceTimersByTimeAsync(0);
    expect(screen.queryByText(LABEL)).toBeNull();
  });

  it('shows on enabled:true, and appears after a retry that finally gets one', async () => {
    vi.useFakeTimers();
    mockStatus.mockRejectedValueOnce(new Error('Network Error'));
    mockStatus.mockResolvedValue(ON);
    renderSidebar();
    await vi.advanceTimersByTimeAsync(0);
    expect(screen.queryByText(LABEL)).toBeNull();
    await vi.advanceTimersByTimeAsync(1000);
    expect(screen.getByText(LABEL)).toBeTruthy();
    expect(screen.getByText('Tools')).toBeTruthy();
  });
});

describe('dashboard tile', () => {
  const renderDashboard = () => render(<Dashboard onNavigate={() => {}} onFileOpen={() => {}} />);

  it('is hidden on enabled:false', async () => {
    mockStatus.mockResolvedValue(OFF);
    renderDashboard();
    await act(async () => {});
    expect(screen.queryByText(LABEL)).toBeNull();
    expect(screen.getByText('PC Refinement')).toBeTruthy(); // other cards are untouched
  });

  it('shows on enabled:true', async () => {
    mockStatus.mockResolvedValue(ON);
    renderDashboard();
    await act(async () => {});
    expect(screen.getByText(LABEL)).toBeTruthy();
  });
});

describe('visibleSidebarPages', () => {
  it('drops mlhub by default and keeps it when asked, keeping every other page', () => {
    const off = visibleSidebarPages(SIDEBAR_PAGES);
    const on = visibleSidebarPages(SIDEBAR_PAGES, { mlHub: true });
    expect(off.some((p) => p.id === 'mlhub')).toBe(false);
    expect(on.some((p) => p.id === 'mlhub')).toBe(true);
    const ids = (l) => l.filter((p) => !p.sectionKey).map((p) => p.id);
    expect(ids(on).filter((i) => i !== 'mlhub')).toEqual(ids(off));
    // No section header is left without a page under it.
    off.forEach((p, i) => {
      if (p.sectionKey) expect(off[i + 1] && !off[i + 1].sectionKey).toBeTruthy();
    });
  });
});

// App.jsx is not rendered by any test (see pageMounts.test.js), so the two
// things it owns are read from the source: the page is mounted only on a
// 200, and a disabled answer sends the user off the page.
describe('App.jsx', () => {
  const APP = fs.readFileSync(path.resolve(import.meta.dirname, '..', '..', 'App.jsx'), 'utf8');

  it("mounts MLHubPage only when the state is 'enabled'", () => {
    expect(APP).toContain("{mlHubState === 'enabled' && <Suspense fallback={<PageLoader />}><MLHubPage");
  });

  it("leaves the page for the dashboard when the state is 'disabled'", () => {
    expect(APP).toMatch(/mlHubState === 'disabled' && currentPage === 'mlhub'\) setCurrentPage\('dashboard'\)/);
  });
});
