/**
 * The sidebar's pages, and the only place a page's shortcut digit is written.
 *
 * It lived inside App.jsx, which meant the Dashboard could not read it and
 * kept its own copy. On 2026-09-25 the two had drifted so far that SEVEN of
 * the ten numbers opened a different page than the dashboard announced --
 * Ctrl+3 said "Analysis" and opened EDS, Ctrl+9 said "ML predictor" and
 * opened Phase Maps -- and the dashboard advertised a Ctrl+E that no handler
 * has. Nobody noticed because each list was correct on its own terms.
 *
 * Labels and tooltips are resolved at render time from the `nav` i18n
 * namespace by page id (nav:pages.<id>.label / .tooltip) and section key
 * (nav:sections.<sectionKey>). Keep this list to ids/shortcuts/icons only.
 */
export const SIDEBAR_PAGES = [
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
  { id: 'addons',       icon: 'Puzzle' },
  // --- Tools ---
  { id: '_section', sectionKey: 'tools' },
  { id: 'mlhub',        icon: 'Brain' },
  // --- Settings ---
  { id: '_section', sectionKey: 'settings' },
  { id: 'settings',     icon: 'Settings' },
];

/** The pages that HAVE a digit, in the order the sidebar shows them. */
export function pageShortcuts() {
  return SIDEBAR_PAGES.filter((p) => p.shortcut && !p.sectionKey)
    .map((p) => ({ id: p.id, shortcut: p.shortcut }));
}
