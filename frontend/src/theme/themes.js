/**
 * Theme definitions for Orienta.
 * Each theme maps to the CSS variable names declared in index.css.
 * ThemeProvider applies these as :root overrides at runtime.
 */

export const THEME_NAMES = ['dracula', 'light', 'scientific', 'nord', 'solarized'];

export const themes = {
  dracula: {
    name: 'Dracula',
    // Backgrounds
    '--bg-primary':    '#282a36',
    '--bg-secondary':  '#1e1f29',
    '--bg-tertiary':   '#44475a',
    '--bg-hover':      '#3d3f4f',
    // Text
    '--text-primary':   '#f8f8f2',
    '--text-secondary': '#8b93a8',  // was #6272a4, now WCAG AA compliant (~5.2:1 on #282a36)
    '--text-muted':     '#6272a4',  // was #555770, moved old secondary here for disabled/muted
    // Accents
    '--accent-purple':  '#bd93f9',
    '--accent-pink':    '#ff79c6',
    '--accent-green':   '#50fa7b',
    '--accent-cyan':    '#8be9fd',
    '--accent-orange':  '#ffb86c',
    '--accent-red':     '#ff5555',
    '--accent-yellow':  '#f1fa8c',
    // UI chrome
    '--border-color':   '#44475a',
    // Sidebar
    '--sidebar-bg':     '#1a1b26',
    '--sidebar-active': '#2a2b3d',
    '--sidebar-border': '#3b3d54',
    '--sidebar-text':   '#c0c0c0',
    // Cards
    '--card-bg':        '#2d2f3d',
    '--card-bg-hover':  '#353749',
    '--card-border':    '#3b3d54',
    // Text on accent
    '--text-on-accent': '#282a36',
    // Mystic Fog (dashboard background) — purple haze on dark
    fog: {
      highlightColor: 0x5e497c,
      midtoneColor:   0x383a4d,
      lowlightColor:  0x181922,
      baseColor:      0x282a36,
    },
  },

  light: {
    name: 'Light',
    // Backgrounds
    '--bg-primary':    '#f5f5f5',
    '--bg-secondary':  '#ffffff',
    '--bg-tertiary':   '#e0e0e0',
    '--bg-hover':      '#e8e8f0',
    // Text
    '--text-primary':   '#1a1a2e',
    '--text-secondary': '#666677',
    '--text-muted':     '#999999',
    // Accents
    '--accent-purple':  '#7c3aed',
    '--accent-pink':    '#ec4899',
    '--accent-green':   '#10b981',
    '--accent-cyan':    '#0891b2',
    '--accent-orange':  '#f59e0b',
    '--accent-red':     '#ef4444',
    '--accent-yellow':  '#eab308',
    // UI chrome
    '--border-color':   '#d0d0d0',
    // Sidebar
    '--sidebar-bg':     '#e8e8ee',
    '--sidebar-active': '#d0d0e0',
    '--sidebar-border': '#c0c0cc',
    '--sidebar-text':   '#444455',
    // Cards
    '--card-bg':        '#ffffff',
    '--card-bg-hover':  '#f0f0f8',
    '--card-border':    '#d0d0d0',
    // Text on accent
    '--text-on-accent': '#ffffff',
    // Mystic Fog (dashboard background) — soft lavender mist on white
    fog: {
      highlightColor: 0xc9b8ef,
      midtoneColor:   0xe2def2,
      lowlightColor:  0xcfcfdc,
      baseColor:      0xf5f5f5,
    },
  },

  scientific: {
    name: 'Scientific',
    // Backgrounds
    '--bg-primary':    '#0f172a',
    '--bg-secondary':  '#1e293b',
    '--bg-tertiary':   '#334155',
    '--bg-hover':      '#2d3a50',
    // Text
    '--text-primary':   '#e2e8f0',
    '--text-secondary': '#94a3b8',
    '--text-muted':     '#64748b',
    // Accents
    '--accent-purple':  '#818cf8',
    '--accent-pink':    '#f472b6',
    '--accent-green':   '#34d399',
    '--accent-cyan':    '#22d3ee',
    '--accent-orange':  '#fb923c',
    '--accent-red':     '#f87171',
    '--accent-yellow':  '#fbbf24',
    // UI chrome
    '--border-color':   '#475569',
    // Sidebar
    '--sidebar-bg':     '#0c1222',
    '--sidebar-active': '#1e293b',
    '--sidebar-border': '#334155',
    '--sidebar-text':   '#94a3b8',
    // Cards
    '--card-bg':        '#1e293b',
    '--card-bg-hover':  '#263548',
    '--card-border':    '#334155',
    // Text on accent
    '--text-on-accent': '#0f172a',
    // Mystic Fog (dashboard background) — deep teal/blue
    fog: {
      highlightColor: 0x1a4a5a,
      midtoneColor:   0x18324a,
      lowlightColor:  0x0a1020,
      baseColor:      0x0f172a,
    },
  },

  nord: {
    name: 'Nord',
    // Backgrounds — Polar Night
    '--bg-primary':    '#2e3440',
    '--bg-secondary':  '#3b4252',
    '--bg-tertiary':   '#434c5e',
    '--bg-hover':      '#4c566a',
    // Text — Snow Storm
    '--text-primary':   '#eceff4',
    '--text-secondary': '#d8dee9',
    '--text-muted':     '#a0aab8',
    // Accents — Frost + Aurora
    '--accent-purple':  '#b48ead',
    '--accent-pink':    '#d08770',
    '--accent-green':   '#a3be8c',
    '--accent-cyan':    '#88c0d0',
    '--accent-orange':  '#d08770',
    '--accent-red':     '#bf616a',
    '--accent-yellow':  '#ebcb8b',
    // UI chrome
    '--border-color':   '#4c566a',
    // Sidebar
    '--sidebar-bg':     '#242933',
    '--sidebar-active': '#3b4252',
    '--sidebar-border': '#434c5e',
    '--sidebar-text':   '#d8dee9',
    // Cards
    '--card-bg':        '#3b4252',
    '--card-bg-hover':  '#434c5e',
    '--card-border':    '#4c566a',
    // Text on accent
    '--text-on-accent': '#2e3440',
    // Mystic Fog (dashboard background) — frost mist
    fog: {
      highlightColor: 0x5e7d8c,
      midtoneColor:   0x3b4252,
      lowlightColor:  0x222730,
      baseColor:      0x2e3440,
    },
  },

  solarized: {
    name: 'Solarized',
    // Backgrounds — Solarized Dark base tones
    '--bg-primary':    '#002b36',
    '--bg-secondary':  '#073642',
    '--bg-tertiary':   '#094252',
    '--bg-hover':      '#0a4a5c',
    // Text
    '--text-primary':   '#fdf6e3',
    '--text-secondary': '#839496',
    '--text-muted':     '#586e75',
    // Accents — Solarized palette
    '--accent-purple':  '#6c71c4',
    '--accent-pink':    '#d33682',
    '--accent-green':   '#859900',
    '--accent-cyan':    '#2aa198',
    '--accent-orange':  '#cb4b16',
    '--accent-red':     '#dc322f',
    '--accent-yellow':  '#b58900',
    // UI chrome
    '--border-color':   '#094252',
    // Sidebar
    '--sidebar-bg':     '#001e27',
    '--sidebar-active': '#073642',
    '--sidebar-border': '#094252',
    '--sidebar-text':   '#93a1a1',
    // Cards
    '--card-bg':        '#073642',
    '--card-bg-hover':  '#094252',
    '--card-border':    '#0a4a5c',
    // Text on accent
    '--text-on-accent': '#002b36',
    // Mystic Fog (dashboard background) — teal on deep cyan-blue
    fog: {
      highlightColor: 0x1a6b66,
      midtoneColor:   0x0a4250,
      lowlightColor:  0x001a22,
      baseColor:      0x002b36,
    },
  },
};

export default themes;
