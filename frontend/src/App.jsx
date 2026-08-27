/**
 * App.jsx — Main application shell matching PyQt5 MainHub exactly.
 *
 * PyQt5 structure:
 *   MainHub(QMainWindow)
 *   ├── Sidebar (270px, #1a1b26)
 *   │   ├── Title "Kikuchipy GUI"
 *   │   ├── StatusIndicator
 *   │   ├── 9 Module buttons (checkable, exclusive)
 *   │   └── HDF5 Viewer tool button (bottom)
 *   └── Right panel
 *       ├── Toolbar (36px, #21222c, EDS Colors button)
 *       ├── QStackedWidget (module pages)
 *       └── Status bar
 */

import { useState, useEffect, useCallback, useMemo, useRef, lazy, Suspense, Component } from 'react';
import { useTranslation } from 'react-i18next';
import i18n from './i18n';
import { setLanguage, LANGUAGES } from './i18n';
import { healthCheck, createWebSocket } from './services/api';
import { reportError } from './services/errorReporter';
import { addBreadcrumb } from './services/breadcrumbs';
import DiagnosticsExportButton from './components/common/DiagnosticsExportButton';
import UpdateDialog from './components/common/UpdateDialog';
import ProblemReportDialog from './components/common/ProblemReportDialog';
import { captureScreenshot } from './services/screenshot';
import useUpdateCheck from './hooks/useUpdateCheck';
import useDataStore from './stores/useDataStore';
import useResultStore from './stores/useResultStore';
import { colors, layout } from './theme/tokens';
import Sidebar from './components/shared/Sidebar';
import StatusBar from './components/shared/StatusBar';
import ToastContainer from './components/shared/ToastContainer';
import DevPanel from './components/shared/DevPanel';
// Dashboard is the initial route — keep it synchronous so the first paint is fast.
import Dashboard from './components/Dashboard/Dashboard';

// All other 14 pages + the HDF5 Viewer overlay + Periodic Table dialog
// are lazy-loaded so the initial bundle drops from 839 KB to ~200 KB.
// Each page becomes its own chunk loaded on demand when the user navigates.
const HDF5Viewer = lazy(() => import('./components/HDF5Viewer/HDF5Viewer'));
const EBSDViewer = lazy(() => import('./components/EBSDViewer/EBSDViewer'));
const PCRefinement = lazy(() => import('./components/PCRefinement/PCRefinement'));
const SimulationPage = lazy(() => import('./components/Simulation/SimulationPage'));
const IndexingPage = lazy(() => import('./components/Indexing/IndexingPage'));
const PhaseMapPage = lazy(() => import('./components/PhaseMap/PhaseMapPage'));
const AnalysisPage = lazy(() => import('./components/Analysis/AnalysisPage'));
const MLHubPage = lazy(() => import('./components/MLHub/MLHubPage'));
const DatabasePage = lazy(() => import('./components/DatabaseBrowser/DatabasePage'));
const CrystalDatabasePage = lazy(() => import('./components/CrystalDatabase/CrystalDatabasePage'));
const EDSPage = lazy(() => import('./components/EDS/EDSPage'));
const PeriodicTableDialog = lazy(() => import('./components/EDS/PeriodicTableDialog'));
const SettingsPage = lazy(() => import('./components/Settings/SettingsPage'));
const BatchPage = lazy(() => import('./components/Batch/BatchPage'));
const RefinementPage = lazy(() => import('./components/Refinement/RefinementPage'));
const CrystalHintPage = lazy(() => import('./components/CrystalHint/CrystalHintPage'));

// Fallback UI shown while a route chunk is being fetched. Kept simple so
// it flashes only briefly on first navigation to a given page.
const PageLoader = () => {
  const { t } = useTranslation('shell');
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%', color: '#888', fontSize: '10pt',
    }}>
      {t('loadingModule')}
    </div>
  );
};
import useDevLogs from './hooks/useDevLogs';

// Error Boundary prevents white-screen crashes
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(err, info) {
    console.error('React crash:', err, info);
    // Persist in the backend log — the "This module crashed" screenshot users
    // send never contains the stack; this line puts it into logs/orienta.log.
    reportError('react-boundary', err, { componentStack: info?.componentStack || '' });
  }
  render() {
    if (this.state.error) {
      return (
        <div role="alert" style={{
          padding: 40, background: colors.bg, height: '100%',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.4 }}>{'\u26A0'}</div>
          <h2 style={{ color: colors.red, marginBottom: 8, fontSize: 16 }}>This module crashed</h2>
          <pre style={{
            color: colors.textSecondary, fontSize: 11, whiteSpace: 'pre-wrap',
            maxWidth: 500, textAlign: 'center', marginBottom: 16, lineHeight: 1.5,
          }}>
            {this.state.error.message || String(this.state.error)}
          </pre>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
            <button
              onClick={() => this.setState({ error: null })}
              title={i18n.t('shell:hoverTips.errorTryAgain')}
              style={{
                padding: '8px 20px',
                background: colors.accent, border: 'none', borderRadius: 4,
                color: colors.textOnAccent, fontWeight: 600, cursor: 'pointer',
                transition: 'filter 0.12s, transform 0.1s',
              }}
              onMouseEnter={e => { e.currentTarget.style.filter = 'brightness(1.15)'; e.currentTarget.style.transform = 'scale(1.03)'; }}
              onMouseLeave={e => { e.currentTarget.style.filter = 'none'; e.currentTarget.style.transform = 'scale(1)'; }}
            >
              Retry
            </button>
            <button
              onClick={() => window.location.reload()}
              title={i18n.t('shell:hoverTips.errorReload')}
              style={{
                padding: '8px 20px',
                background: colors.border, border: 'none', borderRadius: 4,
                color: colors.text, fontWeight: 500, cursor: 'pointer',
                transition: 'filter 0.12s, transform 0.1s',
              }}
              onMouseEnter={e => { e.currentTarget.style.filter = 'brightness(1.2)'; e.currentTarget.style.transform = 'scale(1.03)'; }}
              onMouseLeave={e => { e.currentTarget.style.filter = 'none'; e.currentTarget.style.transform = 'scale(1)'; }}
            >
              Reload App
            </button>
            {/* The crash screen is where a bug report is most likely to
                happen — offer the zip instead of a screenshot. */}
            <DiagnosticsExportButton variant="crash" />
          </div>
          <div style={{
            color: colors.textSecondary, fontSize: 10, marginTop: 12,
            maxWidth: 460, textAlign: 'center', opacity: 0.8,
          }}>
            {i18n.t('settings:about.exportDiagnosticsHint')}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/*
 * Sidebar navigation — grouped by workflow stage.
 * Sections: Data → Calibration & Simulation → Index & Analyze → Tools
 * Keyboard shortcuts: Ctrl+1..9,0 for pages, Ctrl+H for HDF5 Viewer
 */
// Labels + tooltips are resolved at render time from the `nav` i18n namespace
// by page id (nav:pages.<id>.label / .tooltip) and section key
// (nav:sections.<sectionKey>). Keep this list to ids/shortcuts/icons only.
const SIDEBAR_PAGES = [
  // --- Data ---
  { id: '_section', sectionKey: 'data' },
  { id: 'dashboard',    shortcut: '1', icon: 'LayoutDashboard' },
  { id: 'ebsdviewer',   shortcut: '2', icon: 'ScanLine' },
  { id: 'eds',          shortcut: '3', icon: 'Atom' },
  // --- Calibration & Simulation ---
  { id: '_section', sectionKey: 'calibration' },
  { id: 'pcrefinement', shortcut: '4', icon: 'Crosshair' },
  { id: 'crystal',      shortcut: '5', icon: 'Gem' },
  { id: 'simulation',   shortcut: '6', icon: 'Cpu' },
  { id: 'database',     shortcut: '7', icon: 'HardDrive' },
  // --- Index & Analyze ---
  { id: '_section', sectionKey: 'indexAnalyze' },
  { id: 'indexing',     shortcut: '8', icon: 'Grid3x3' },
  { id: 'phasemap',     shortcut: '9', icon: 'Map' },
  { id: 'analysis',     shortcut: '0', icon: 'BarChart3' },
  { id: 'batch',        icon: 'Package' },
  { id: 'refinement',   icon: 'SlidersHorizontal' },
  { id: 'crystalhint',  icon: 'Atom' },
  // --- Tools ---
  { id: '_section', sectionKey: 'tools' },
  { id: 'mlhub',        icon: 'Brain' },
  // --- Settings ---
  { id: '_section', sectionKey: 'settings' },
  { id: 'settings',     icon: 'Settings' },
];

function App() {
  const { t, i18n } = useTranslation(['nav', 'shell']);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [h5ViewerOpen, setH5ViewerOpen] = useState(false);
  const [edsColorsOpen, setEdsColorsOpen] = useState(false);
  // Reachable from every page: a problem is reported where it happened, and
  // the trail it collects is most complete right then.
  const [reportOpen, setReportOpen] = useState(false);
  const [reportShot, setReportShot] = useState(null);

  // Capture BEFORE the dialog renders, or the picture shows the dialog
  // instead of the screen the user is complaining about.
  const openReport = useCallback(async () => {
    setReportShot(await captureScreenshot());
    setReportOpen(true);
  }, []);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.innerWidth < 1100);
  const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
  const [devPanelOpen, setDevPanelOpen] = useState(false);
  // Looks for a newer release a few seconds after start; silent on failure.
  const { updateInfo, close: closeUpdate, skip: skipUpdate } = useUpdateCheck();
  // Only buffer dev logs when the panel is actually visible. Long-running
  // batches/simulations hammer the WebSocket with status events; without
  // gating, App.jsx (root component) re-renders ~10×/s for hours which has
  // contributed to renderer freezes that took the backend down with them.
  const { logs: devLogs, clearLogs: clearDevLogs, isConnected: devWsConnected } =
    useDevLogs({ enabled: devPanelOpen });

  // Auto-collapse sidebar on narrow windows
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth < 1100) setSidebarCollapsed(true);
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Data availability for sidebar badges
  const isFileOpen = useDataStore(s => s.isFileOpen);
  const filePath = useDataStore(s => s.filePath);
  const hasEDS = useDataStore(s => s.hasEDS);
  const indexingResult = useResultStore(s => s.indexingResult);
  const analysisLoaded = useResultStore(s => s.analysisLoaded);
  const dataReady = useMemo(() => ({
    ebsdviewer: isFileOpen,
    analysis: analysisLoaded,
    indexing: isFileOpen,
    pcrefinement: isFileOpen,
    phasemap: !!indexingResult,
    eds: hasEDS,
  }), [isFileOpen, hasEDS, indexingResult, analysisLoaded]);

  // Sync store from backend on first connect + on each reconnect.
  // Covers the case where a file was loaded outside the React "Open File"
  // flow (direct API, test-env script, Electron dialog, etc.) so the
  // StatusBar and sidebar badges don't stay stale (BUG-E).
  const syncFromBackend = useDataStore(s => s.syncFromBackend);
  const prevBackendStatus = useRef('checking');
  useEffect(() => {
    if (backendStatus === 'connected' && prevBackendStatus.current !== 'connected') {
      syncFromBackend();
    }
    prevBackendStatus.current = backendStatus;
  }, [backendStatus, syncFromBackend]);

  // Health check on mount with fast retry, then every 30s once connected
  useEffect(() => {
    let intervalId = null;
    let retryCount = 0;
    const maxFastRetries = 10; // Fast-poll up to 10 times (20s) on startup

    const check = () => {
      healthCheck()
        .then(() => {
          setBackendStatus(prev => prev === 'connected' ? prev : 'connected');
          // Once connected, switch to slow polling
          if (intervalId && retryCount <= maxFastRetries) {
            clearInterval(intervalId);
            intervalId = setInterval(check, 30000);
          }
          retryCount = maxFastRetries + 1; // stop fast retries
        })
        .catch(() => {
          setBackendStatus(prev => prev === 'disconnected' ? prev : 'disconnected');
          retryCount++;
        });
    };
    check();
    // Fast retry every 2s until connected, then 30s
    intervalId = setInterval(check, 2000);
    return () => { if (intervalId) clearInterval(intervalId); };
  }, []);

  // WebSocket keepalive — prevents backend watchdog from killing the server
  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    const connect = () => {
      ws = createWebSocket(() => {
        // Handle backend push messages if needed
      });
      ws.onclose = () => {
        reconnectTimer = setTimeout(connect, 5000);
      };
    };
    connect();
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, []);

  // Shut down backend when browser tab/window closes + warn about data loss
  useEffect(() => {
    const handleUnload = () => {
      navigator.sendBeacon('/api/shutdown', '');
    };
    const handleBeforeUnload = (e) => {
      // Only warn in browser mode, not in Electron (Electron should always close cleanly)
      if (isFileOpen && !window.electronAPI) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('unload', handleUnload);
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('unload', handleUnload);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isFileOpen]);

  const handleToggleH5Viewer = useCallback(() => setH5ViewerOpen(prev => !prev), []);

  // Keyboard shortcuts: Ctrl+1..9 for pages, Ctrl+H for HDF5 Viewer
  const handleKeyDown = useCallback((e) => {
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    // Escape closes floating panels, topmost first
    if (e.key === 'Escape') {
      if (reportOpen) { setReportOpen(false); return; }
      if (shortcutHelpOpen) { setShortcutHelpOpen(false); return; }
      if (edsColorsOpen) { setEdsColorsOpen(false); return; }
      if (h5ViewerOpen) { setH5ViewerOpen(false); return; }
    }

    // ? key shows keyboard shortcut help
    if (e.key === '?' && !e.ctrlKey && !e.altKey) {
      setShortcutHelpOpen(prev => !prev);
      return;
    }

    if (e.ctrlKey && !e.shiftKey && !e.altKey) {
      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        setH5ViewerOpen(prev => !prev);
        return;
      }
      if (e.key.toLowerCase() === 'b') {
        e.preventDefault();
        setSidebarCollapsed(prev => !prev);
        return;
      }
      // EDS is now a regular sidebar page (Ctrl+3), no special toggle needed
      const page = SIDEBAR_PAGES.find(p => p.shortcut === e.key);
      if (page) {
        e.preventDefault();
        setCurrentPage(page.id);
      }
    }
  }, [edsColorsOpen, h5ViewerOpen, shortcutHelpOpen]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Cross-module navigation
  const handleNavigate = useCallback((pageId) => {
    if (pageId === 'h5viewer') {
      setH5ViewerOpen(true);
      return;
    }
    // An id no page answers to used to be accepted anyway: the window went
    // blank, and the page that never mounted also never loaded its data — the
    // "→ Phase Map" button sent 'phase-map' while the page is called
    // 'phasemap', and the result gallery stayed empty. Refuse the move and say
    // so instead of showing nothing.
    if (!SIDEBAR_PAGES.some((p) => p.id === pageId)) {
      console.warn(`[App] ignoring navigation to unknown page "${pageId}"`);
      return;
    }
    setCurrentPage(pageId);
  }, []);

  // Recent-file click on Dashboard → navigate to EBSD Viewer and auto-load
  const handleFileOpen = useCallback((path) => {
    try { sessionStorage.setItem('ebsd_preload_path', path); } catch { /* unavailable */ }
    setCurrentPage('ebsdviewer');
  }, []);

  // Listen for cross-module navigation events from child components
  useEffect(() => {
    const onNav = (e) => handleNavigate(e.detail?.page, e.detail?.context);
    window.addEventListener('navigate-to', onNav);
    return () => window.removeEventListener('navigate-to', onNav);
  }, [handleNavigate]);

  // Dispatch custom event when page changes so mounted components can react
  // Also scroll the newly visible page to top
  const mainRef = useRef(null);
  useEffect(() => {
    // "Which screen was the user on?" — the first question of every bug report.
    addBreadcrumb('nav', `page → ${currentPage}`);
    window.dispatchEvent(new CustomEvent('page-changed', { detail: { page: currentPage } }));
    // Scroll the active page container to top
    if (mainRef.current) {
      const active = mainRef.current.querySelector(`[data-page="${currentPage}"]`);
      if (active) active.scrollTop = 0;
    }
    // Update window/tab title
    const label = t(`nav:pages.${currentPage}.label`, { defaultValue: 'Orienta' });
    document.title = `${label} — Orienta`;
  }, [currentPage, t, i18n.language]);

  // Page style helper — memoized to avoid creating fresh objects on every render
  const pageStyles = useMemo(() => {
    const base = { flexDirection: 'column', height: '100%', overflow: 'auto' };
    const ids = SIDEBAR_PAGES.filter(p => !p.sectionKey).map(p => p.id);
    return Object.fromEntries(ids.map(id => [
      id,
      {
        ...base,
        display: currentPage === id ? 'flex' : 'none',
        animation: currentPage === id ? 'pageFadeIn 0.15s ease-out' : 'none',
      },
    ]));
  }, [currentPage]);
  const pageStyle = (id) => pageStyles[id] || { display: 'none', flexDirection: 'column', height: '100%', overflow: 'auto' };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', minWidth: 800, background: colors.bg }}>
      {/* Sidebar — 270px, matches PyQt5 */}
      <Sidebar
        pages={SIDEBAR_PAGES}
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        backendStatus={backendStatus}
        onToggleH5Viewer={handleToggleH5Viewer}
        h5ViewerOpen={h5ViewerOpen}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(prev => !prev)}
        dataReady={dataReady}
      />

      {/* Right panel: Toolbar + Content + StatusBar */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* Toolbar — 36px, matches PyQt5 hub_gui.py */}
        <div style={{
          height: layout.toolbarHeight,
          minHeight: layout.toolbarHeight,
          background: colors.bgTertiary,
          borderBottom: `1px solid ${colors.border}`,
          display: 'flex',
          alignItems: 'center',
          padding: '0 8px',
          gap: 8,
          flexShrink: 0,
        }}>
          <button
            onClick={() => setEdsColorsOpen(prev => !prev)}
            style={{
              background: edsColorsOpen ? colors.purple : 'transparent',
              color: edsColorsOpen ? colors.bg : colors.purple,
              fontWeight: 'bold',
              padding: '4px 12px',
              border: `1px solid ${colors.border}`,
              borderRadius: 3,
              fontSize: '9pt',
              cursor: 'pointer',
              transition: 'background 0.12s, color 0.12s, border-color 0.12s',
            }}
            title={t('shell:toolbar.edsColorsTooltip')}
            onMouseEnter={e => { if (!edsColorsOpen) { e.currentTarget.style.background = `${colors.purple}22`; e.currentTarget.style.borderColor = colors.purple; } }}
            onMouseLeave={e => { if (!edsColorsOpen) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = colors.border; } }}
          >
            {t('shell:toolbar.edsColors')}
          </button>
          {/* Report a problem — in the toolbar so it is reachable from every
              page, not only from Settings and the crash screen. */}
          <button
            onClick={openReport}
            style={{
              background: 'transparent',
              color: colors.yellow,
              padding: '4px 12px',
              border: `1px solid ${colors.border}`,
              borderRadius: 3,
              fontSize: '9pt',
              cursor: 'pointer',
              transition: 'background 0.12s, color 0.12s, border-color 0.12s',
            }}
            title={t('shell:toolbar.reportProblemTooltip')}
            onMouseEnter={e => { e.currentTarget.style.background = `${colors.yellow}22`; e.currentTarget.style.borderColor = colors.yellow; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = colors.border; }}
          >
            {t('shell:toolbar.reportProblem')}
          </button>
          {/* Language switcher — top toolbar, always visible */}
          <select
            value={i18n.resolvedLanguage}
            onChange={(e) => setLanguage(e.target.value)}
            title={t('shell:sidebar.language')}
            aria-label={t('shell:sidebar.language')}
            style={{
              background: colors.bgTertiary,
              color: colors.text,
              border: `1px solid ${colors.border}`,
              borderRadius: 3,
              fontSize: '9pt',
              padding: '3px 6px',
              cursor: 'pointer',
            }}
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{`\u{1F310} ${l.label}`}</option>
            ))}
          </select>
          <div style={{ flex: 1 }} />
          {/* Keyboard shortcuts help */}
          <button
            onClick={() => setShortcutHelpOpen(true)}
            style={{
              background: 'transparent', border: `1px solid ${colors.border}`,
              borderRadius: 3, color: colors.textSecondary, cursor: 'pointer',
              width: 24, height: 24, fontSize: 12, fontWeight: 700,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.12s, color 0.12s, border-color 0.12s',
            }}
            title={t('shell:toolbar.shortcutsTooltip')}
            onMouseEnter={e => { e.currentTarget.style.color = colors.accent; e.currentTarget.style.borderColor = colors.accent; e.currentTarget.style.background = `${colors.accent}11`; }}
            onMouseLeave={e => { e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.borderColor = colors.border; e.currentTarget.style.background = 'transparent'; }}
          >?</button>
          {/* Current page indicator */}
          <span style={{ fontSize: '9pt', color: colors.textSecondary, opacity: 0.7 }}>
            {t(`nav:pages.${currentPage}.label`, { defaultValue: currentPage })}
          </span>
        </div>

        {/* Backend disconnected banner */}
        {backendStatus === 'disconnected' && (
          <div role="alert" style={{
            padding: '4px 12px',
            background: `${colors.red}18`,
            borderBottom: `1px solid ${colors.red}44`,
            fontSize: '9pt',
            color: colors.red,
            textAlign: 'center',
            flexShrink: 0,
            animation: 'fadeSlideIn 0.25s ease-out',
          }}>
            {t('shell:disconnected.message')}{' '}
            <code style={{ background: `${colors.red}22`, padding: '1px 4px', borderRadius: 2 }}>
              python -m uvicorn backend.api.main:app --port 8000
            </code>
          </div>
        )}

        {/* Content area — true QStackedWidget: all pages stay mounted, only active is visible */}
        {/* Each page gets its own ErrorBoundary so a crash in one page doesn't lock all others */}
        <main ref={mainRef} style={{ flex: 1, overflow: 'hidden', padding: 0 }}>
          <div data-page="dashboard" style={pageStyle('dashboard')}><ErrorBoundary><Dashboard onNavigate={handleNavigate} onFileOpen={handleFileOpen} /></ErrorBoundary></div>
          {/* Lazy-loaded pages — one <Suspense> per page so the PageLoader only
              shows in the cell whose chunk is being fetched, not the whole app. */}
          <div data-page="ebsdviewer" style={pageStyle('ebsdviewer')}><ErrorBoundary><Suspense fallback={<PageLoader />}><EBSDViewer onNavigate={handleNavigate} isActive={currentPage === 'ebsdviewer'} /></Suspense></ErrorBoundary></div>
          <div data-page="analysis" style={pageStyle('analysis')}><ErrorBoundary><Suspense fallback={<PageLoader />}><AnalysisPage onNavigate={handleNavigate} isActive={currentPage === 'analysis'} /></Suspense></ErrorBoundary></div>
          <div data-page="crystal" style={pageStyle('crystal')}><ErrorBoundary><Suspense fallback={<PageLoader />}><CrystalDatabasePage onNavigate={handleNavigate} isActive={currentPage === 'crystal'} /></Suspense></ErrorBoundary></div>
          <div data-page="simulation" style={pageStyle('simulation')}><ErrorBoundary><Suspense fallback={<PageLoader />}><SimulationPage onNavigate={handleNavigate} isActive={currentPage === 'simulation'} /></Suspense></ErrorBoundary></div>
          <div data-page="database" style={pageStyle('database')}><ErrorBoundary><Suspense fallback={<PageLoader />}><DatabasePage variant="browser" onNavigate={handleNavigate} isActive={currentPage === 'database'} /></Suspense></ErrorBoundary></div>
          <div data-page="indexing" style={pageStyle('indexing')}><ErrorBoundary><Suspense fallback={<PageLoader />}><IndexingPage onNavigate={handleNavigate} isActive={currentPage === 'indexing'} /></Suspense></ErrorBoundary></div>
          <div data-page="mlhub" style={pageStyle('mlhub')}><ErrorBoundary><Suspense fallback={<PageLoader />}><MLHubPage onNavigate={handleNavigate} isActive={currentPage === 'mlhub'} /></Suspense></ErrorBoundary></div>
          <div data-page="phasemap" style={pageStyle('phasemap')}><ErrorBoundary><Suspense fallback={<PageLoader />}><PhaseMapPage onNavigate={handleNavigate} isActive={currentPage === 'phasemap'} /></Suspense></ErrorBoundary></div>
          <div data-page="pcrefinement" style={pageStyle('pcrefinement')}><ErrorBoundary><Suspense fallback={<PageLoader />}><PCRefinement key={filePath || 'no-file'} onNavigate={handleNavigate} /></Suspense></ErrorBoundary></div>
          <div data-page="eds" style={pageStyle('eds')}><ErrorBoundary><Suspense fallback={<PageLoader />}><EDSPage onNavigate={handleNavigate} isActive={currentPage === 'eds'} /></Suspense></ErrorBoundary></div>
          <div data-page="settings" style={pageStyle('settings')}><ErrorBoundary><Suspense fallback={<PageLoader />}><SettingsPage onNavigate={handleNavigate} isActive={currentPage === 'settings'} /></Suspense></ErrorBoundary></div>
          <div data-page="batch" style={pageStyle('batch')}><ErrorBoundary><Suspense fallback={<PageLoader />}><BatchPage onNavigate={handleNavigate} isActive={currentPage === 'batch'} /></Suspense></ErrorBoundary></div>
          <div data-page="refinement" style={pageStyle('refinement')}><ErrorBoundary><Suspense fallback={<PageLoader />}><RefinementPage onNavigate={handleNavigate} isActive={currentPage === 'refinement'} /></Suspense></ErrorBoundary></div>
          <div data-page="crystalhint" style={pageStyle('crystalhint')}><ErrorBoundary><Suspense fallback={<PageLoader />}><CrystalHintPage onNavigate={handleNavigate} isActive={currentPage === 'crystalhint'} /></Suspense></ErrorBoundary></div>
        </main>

        {/* Status bar */}
        <StatusBar backendStatus={backendStatus} onNavigate={handleNavigate} />

        {/* EDS Colors — Periodic Table (lazy: only loaded when user opens it) */}
        {edsColorsOpen && (
          <Suspense fallback={<PageLoader />}>
            <PeriodicTableDialog onClose={() => setEdsColorsOpen(false)} />
          </Suspense>
        )}

        {/* HDF5 Viewer floating panel (like PyQt5 separate window) */}
        {h5ViewerOpen && (
          <div style={{
            position: 'absolute',
            top: layout.toolbarHeight,
            left: 0,
            right: 0,
            bottom: layout.statusBarHeight,
            background: colors.bg,
            zIndex: 100,
            display: 'flex',
            flexDirection: 'column',
            borderLeft: `2px solid ${colors.accent}`,
            animation: 'pageFadeIn 0.15s ease-out',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '6px 16px',
              background: colors.bgSecondary,
              borderBottom: `1px solid ${colors.border}`,
              flexShrink: 0,
            }}>
              <span style={{ color: colors.accent, fontWeight: 700, fontSize: '11pt' }}>
                {t('nav:hdf5Viewer.label')}
              </span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ fontSize: '8pt', color: colors.textSecondary }}>Ctrl+H</span>
                <button
                  onClick={() => setH5ViewerOpen(false)}
                  title={t('shell:hoverTips.h5ViewerClose')}
                  aria-label={t('shell:hoverTips.h5ViewerClose')}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${colors.border}`,
                    color: colors.textSecondary,
                    borderRadius: 4,
                    padding: '2px 8px',
                    cursor: 'pointer',
                    fontSize: 14,
                    transition: 'color 0.12s, background 0.12s, border-color 0.12s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.color = colors.red; e.currentTarget.style.borderColor = colors.red; e.currentTarget.style.background = `${colors.red}11`; }}
                  onMouseLeave={e => { e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.borderColor = colors.border; e.currentTarget.style.background = 'transparent'; }}
                >
                  &times;
                </button>
              </div>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <Suspense fallback={<PageLoader />}>
                <HDF5Viewer onNavigate={handleNavigate} />
              </Suspense>
            </div>
          </div>
        )}
      </div>

      {/* Dev Panel — right edge, collapsible */}
      <DevPanel
        logs={devLogs}
        onClear={clearDevLogs}
        isConnected={devWsConnected}
        open={devPanelOpen}
        onOpenChange={setDevPanelOpen}
      />

      {/* Keyboard Shortcut Help Panel — press ? to toggle */}
      {shortcutHelpOpen && (
        <div
          onClick={() => setShortcutHelpOpen(false)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 999,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: colors.bgSecondary, border: `1px solid ${colors.border}`,
              borderRadius: 10, padding: '20px 28px', minWidth: 340, maxWidth: 420,
              boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
              animation: 'pageFadeIn 0.15s ease-out',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <span style={{ color: colors.accent, fontWeight: 700, fontSize: 14 }}>{t('shell:shortcuts.title')}</span>
              <button
                onClick={() => setShortcutHelpOpen(false)}
                title={t('shell:hoverTips.shortcutHelpClose')}
                aria-label={t('shell:hoverTips.shortcutHelpClose')}
                style={{
                  background: 'transparent', border: 'none', color: colors.textSecondary, cursor: 'pointer', fontSize: 16,
                  borderRadius: 3, padding: '0 3px', transition: 'color 0.12s, background 0.12s',
                }}
                onMouseEnter={e => { e.currentTarget.style.color = colors.text; e.currentTarget.style.background = `${colors.border}55`; }}
                onMouseLeave={e => { e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.background = 'transparent'; }}
              >&times;</button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 16px', fontSize: 12 }}>
              {[
                ['Ctrl+1', 'Dashboard'], ['Ctrl+2', 'EBSD Viewer'], ['Ctrl+3', 'EDS Analysis'],
                ['Ctrl+4', 'PC Refinement'], ['Ctrl+5', 'Crystal DB'], ['Ctrl+6', 'Simulation'],
                ['Ctrl+7', 'Database'], ['Ctrl+8', 'Indexing'], ['Ctrl+9', 'Phase Maps'],
                ['Ctrl+0', 'Analysis'],
                ['---'],
                ['Ctrl+H', 'Toggle HDF5 Viewer'],
                ['Ctrl+B', 'Toggle Sidebar'], ['Ctrl+K', 'Compare datasets'],
                ['Esc', 'Close panel / dialog'], ['?', 'This help panel'],
                ['---'],
                ['Alt+D', 'BG Dynamic'], ['Alt+S', 'BG Static'],
                ['Alt+A', 'Frame Average'], ['Alt+C', 'Auto-contrast'],
                ['Alt+R', 'Reset processing'], ['R', 'Reset pattern zoom'],
                ['Ctrl+L', 'Focus file input'], ['Ctrl+Shift+D', 'Toggle Dev Panel'],
                ['\u2190\u2191\u2192\u2193', 'Navigate patterns'], ['Shift+Arrow', 'Jump 10 pixels'],
              ].map((item, i) => item.length === 1 ? (
                <div key={i} style={{ gridColumn: '1 / -1', height: 1, background: colors.border, margin: '4px 0' }} />
              ) : (
                <span key={i} style={{ display: 'contents' }}>
                  <kbd style={{
                    background: colors.bg, border: `1px solid ${colors.border}`,
                    borderRadius: 3, padding: '1px 6px', fontSize: 11,
                    color: colors.accent, fontFamily: 'monospace', textAlign: 'center',
                  }}>{item[0]}</kbd>
                  <span style={{ color: colors.text }}>{item[1]}</span>
                </span>
              ))}
            </div>
            <div style={{ marginTop: 12, fontSize: 10, color: colors.textSecondary, textAlign: 'center' }}>
              Press <kbd style={{
                background: colors.bg, border: `1px solid ${colors.border}`,
                borderRadius: 2, padding: '0 4px', fontSize: 10,
              }}>?</kbd> or <kbd style={{
                background: colors.bg, border: `1px solid ${colors.border}`,
                borderRadius: 2, padding: '0 4px', fontSize: 10,
              }}>Esc</kbd> to close
            </div>
          </div>
        </div>
      )}
      <ToastContainer />
      {reportOpen && (
        <ProblemReportDialog
          screenshot={reportShot}
          onClose={() => { setReportOpen(false); setReportShot(null); }}
        />
      )}
      {updateInfo && (
        <UpdateDialog info={updateInfo} onClose={closeUpdate} onSkip={skipUpdate} />
      )}
    </div>
  );
}

export default App;
