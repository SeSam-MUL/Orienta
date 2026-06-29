/**
 * Design tokens for Orienta — Modern Scientific Dark.
 *
 * RULE: Import from here. Never hardcode colors/spacing in components.
 *
 * Multi-theme note: The `colors` object always reflects CSS variable
 * references. At runtime the ThemeProvider overrides CSS variables on :root,
 * so any component that uses `var(--accent-purple)` etc. will automatically
 * reflect the active theme.
 *
 * Use `getThemeColors(themeName)` when you need the raw hex values for a
 * specific theme outside of CSS (e.g. Chart.js series colours).
 */

import { themes } from './themes';

/**
 * Return the color set for a given theme by name.
 * Falls back to 'dracula' if the name is unknown.
 * Returns a plain object: { '--bg-primary': '#...', ... }
 */
export function getThemeColors(themeName) {
  return themes[themeName] ?? themes.dracula;
}

// ---------------------------------------------------------------------------
// Colors — CSS variable references for multi-theme support.
// Components use these; ThemeProvider swaps the underlying CSS variables.
// ---------------------------------------------------------------------------
export const colors = {
  // Primary backgrounds
  bg:              'var(--bg-primary)',
  bgSecondary:     'var(--bg-secondary)',
  bgTertiary:      'var(--bg-tertiary)',

  // Sidebar
  sidebarBg:       'var(--sidebar-bg)',
  sidebarActive:   'var(--sidebar-active)',
  sidebarBorder:   'var(--sidebar-border)',

  // Borders & dividers
  border:          'var(--border-color)',

  // Accent colors
  accent:          'var(--accent-cyan)',
  accentHover:     'var(--accent-cyan)',
  purple:          'var(--accent-purple)',
  green:           'var(--accent-green)',
  yellow:          'var(--accent-yellow)',
  red:             'var(--accent-red)',
  orange:          'var(--accent-orange)',
  pink:            'var(--accent-pink)',
  cyan:            'var(--accent-cyan)',

  // Dashboard card backgrounds
  cardBg:          'var(--card-bg)',
  cardBgHover:     'var(--card-bg-hover)',
  cardBorder:      'var(--card-border)',

  // Card accent colors (these stay the same per theme via CSS vars)
  cardGreen:       'var(--accent-green)',
  cardPink:        'var(--accent-pink)',
  cardYellow:      'var(--accent-yellow)',
  cardPurple:      'var(--accent-purple)',
  cardRed:         'var(--accent-red)',
  cardOrange:      'var(--accent-orange)',
  cardBlue:        'var(--text-secondary)',

  // Text
  text:            'var(--text-primary)',
  textSecondary:   'var(--text-secondary)',
  textOnAccent:    'var(--text-on-accent)',
};

/**
 * Create a semi-transparent version of a CSS variable color.
 * Uses color-mix() which is supported in all modern browsers.
 * @param {string} cssVar — e.g. colors.green ('var(--accent-green)')
 * @param {number} percent — opacity percentage 0-100
 */
export function alpha(cssVar, percent) {
  return `color-mix(in srgb, ${cssVar} ${percent}%, transparent)`;
}

// ---------------------------------------------------------------------------
// Spacing — increased from PyQt5 values for web context
// ---------------------------------------------------------------------------
export const spacing = {
  // Outer container (MainWindow, large panels)
  outerMargin:     16,   // was 8 — more breathing room
  outerSpacing:    12,   // was 8

  // GroupBox / sections
  groupMargin:     12,   // was 6
  groupSpacing:    10,   // was 6

  // Inner widgets (labels, inputs)
  innerMargin:     8,    // was 4
  innerSpacing:    6,    // was 4

  // Compact mode (sidebar with many fields)
  compactMargin:   4,    // was 2
  compactSpacing:  4,    // was 2

  // Buttons
  buttonSpacing:   10,   // was 8
  buttonHeight:    34,   // was 28 — closer to WCAG 44px target

  // Separator
  separatorHeight: 1,
  separatorMargin: 6,    // was 4
};

// ---------------------------------------------------------------------------
// Typography — wider scale for better visual hierarchy
// ---------------------------------------------------------------------------
export const fonts = {
  heading:    { fontSize: '18px', fontWeight: 600 },             // Panel titles (was 13pt ~17px)
  subheading: { fontSize: '14px', fontWeight: 600 },             // GroupBox titles (was 11pt ~15px)
  body:       { fontSize: '13px', fontWeight: 'normal' },        // Standard labels (was 10pt ~13px)
  small:      { fontSize: '11px', fontWeight: 'normal' },        // Status bar, tooltips (was 9pt ~12px)
  monospace:  { fontSize: '12px', fontFamily: "'Courier New', monospace" }, // was 10pt
};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
export const layout = {
  sidebarWidth:         260,     // slightly narrower for more content space
  sidebarCollapsedWidth: 52,     // icon-only collapsed sidebar
  statusBarHeight: 28,           // Status bar at bottom
  toolbarHeight:   40,           // was 36 — slightly taller for comfortable buttons
  scrollbarWidth:  8,
};

// ---------------------------------------------------------------------------
// Sidebar button style
// ---------------------------------------------------------------------------
export const sidebarStyles = {
  button: {
    backgroundColor: 'transparent',
    color: 'var(--sidebar-text)',
    fontWeight: 500,
    fontSize: '13px',
    textAlign: 'left',
    padding: '10px 16px',
    border: 'none',
    borderRadius: 0,
    borderLeft: '3px solid transparent',
    cursor: 'pointer',
    display: 'block',
    width: '100%',
    transition: 'all 0.15s',
  },
  buttonHover: {
    backgroundColor: 'var(--sidebar-active)',
    color: 'var(--accent-cyan)',
  },
  buttonActive: {
    backgroundColor: 'var(--sidebar-active)',
    color: 'var(--accent-cyan)',
    borderLeft: '3px solid var(--accent-cyan)',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: 'var(--accent-cyan)',
    padding: '18px 20px',
    backgroundColor: 'transparent',
  },
  separator: {
    backgroundColor: 'var(--sidebar-border)',
    height: 1,
    margin: '4px 16px',
  },
};
