/**
 * Sidebar — workflow-grouped navigation with Lucide icons and section headers.
 * Sections: Data → Calibration → Index & Analyze → Tools
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, layout } from '../../theme/tokens';
import { useTheme } from '../../theme/ThemeProvider';
import { setLanguage, LANGUAGES } from '../../i18n';
import {
  LayoutDashboard, ScanLine, Atom, Crosshair, Gem, Cpu, HardDrive,
  Grid3x3, Map, BarChart3, Brain, Database, Settings,
  Package, SlidersHorizontal, Zap,
} from 'lucide-react';

/** Map icon name strings from SIDEBAR_PAGES to Lucide components */
const ICON_MAP = {
  LayoutDashboard, ScanLine, Atom, Crosshair, Gem, Cpu, HardDrive,
  Grid3x3, Map, BarChart3, Brain, Database, Settings,
  Package, SlidersHorizontal, Zap,
};

const S = {
  sidebar: {
    width: layout.sidebarWidth,
    minWidth: layout.sidebarWidth,
    maxWidth: layout.sidebarWidth,
    background: colors.sidebarBg,
    borderRight: `1px solid ${colors.sidebarBorder}`,
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
    transition: 'background 0.3s ease, border-color 0.3s ease',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: colors.accent,
    padding: '18px 20px 2px',
    textAlign: 'center',
    background: 'transparent',
  },
  subtitle: {
    fontSize: '11px',
    color: colors.textSecondary,
    textAlign: 'center',
    padding: '0 20px 12px',
  },
  separator: {
    height: 1,
    background: colors.sidebarBorder,
    margin: '0 0',
    flexShrink: 0,
  },
  nav: {
    flex: 1,
    overflowY: 'auto',
    padding: '8px 0',
  },
  navBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    width: '100%',
    background: 'transparent',
    color: 'var(--sidebar-text)',
    fontWeight: 500,
    fontSize: '13px',
    textAlign: 'left',
    padding: '10px 16px',
    border: 'none',
    borderRadius: 0,
    borderLeft: '3px solid transparent',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  navBtnHover: {
    background: colors.sidebarActive,
    color: colors.accent,
    borderLeft: `3px solid ${colors.accent}44`,
  },
  navBtnActive: {
    background: colors.sidebarActive,
    color: colors.accent,
    borderLeft: `3px solid ${colors.accent}`,
    boxShadow: `inset 4px 0 8px -4px ${colors.accent}44`,
  },
  shortcut: {
    marginLeft: 'auto',
    fontSize: '8pt',
    opacity: 0.5,
    fontFamily: 'monospace',
    color: 'inherit',
  },
  statusArea: {
    padding: '6px 16px',
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    display: 'inline-block',
    flexShrink: 0,
    transition: 'background 0.5s ease',
  },
  statusLabel: {
    fontSize: '8pt',
    color: colors.textSecondary,
  },
  footer: {
    padding: '8px 16px',
    borderTop: `1px solid ${colors.sidebarBorder}`,
    fontSize: '9pt',
    color: colors.textSecondary,
  },
};

const THEME_ICONS = { dracula: '🦇', light: '☀', scientific: '🔬', nord: '❄', solarized: '🌅' };
const THEME_LABELS = { dracula: 'Dracula', light: 'Light', scientific: 'Scientific', nord: 'Nord', solarized: 'Solarized' };
const THEME_ORDER = ['dracula', 'light', 'scientific', 'nord', 'solarized'];

export default function Sidebar({
  pages, currentPage, onNavigate, backendStatus,
  onToggleH5Viewer, h5ViewerOpen,
  collapsed, onToggleCollapse,
  dataReady = {},
}) {
  const [hovered, setHovered] = useState(null);
  const { theme, setTheme } = useTheme();
  const { t, i18n } = useTranslation(['nav', 'shell']);

  const cycleTheme = () => {
    const idx = THEME_ORDER.indexOf(theme);
    const next = THEME_ORDER[(idx + 1) % THEME_ORDER.length];
    setTheme(next);
  };

  const currentLang = i18n.resolvedLanguage || i18n.language;
  const cycleLanguage = () => {
    const codes = LANGUAGES.map((l) => l.code);
    const idx = codes.indexOf(currentLang);
    setLanguage(codes[(idx + 1) % codes.length]);
  };

  const statusColor = {
    connected: colors.green,
    disconnected: colors.red,
    checking: colors.orange,
  }[backendStatus] || colors.textSecondary;

  const sidebarW = collapsed ? layout.sidebarCollapsedWidth : layout.sidebarWidth;

  return (
    <div style={{ ...S.sidebar, width: sidebarW, minWidth: sidebarW, maxWidth: sidebarW, transition: 'width 0.2s ease' }}>
      {/* Collapse toggle */}
      <button
        onClick={onToggleCollapse}
        title={collapsed ? `${t('shell:sidebar.expand')} (Ctrl+B)` : `${t('shell:sidebar.collapse')} (Ctrl+B)`}
        aria-label={collapsed ? t('shell:sidebar.expand') : t('shell:sidebar.collapse')}
        style={{
          background: 'transparent', border: 'none', color: colors.textSecondary,
          cursor: 'pointer', padding: '6px 0', fontSize: '10pt', textAlign: 'center',
          flexShrink: 0, transition: 'color 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.color = colors.accent; }}
        onMouseLeave={e => { e.currentTarget.style.color = colors.textSecondary; }}
      >
        {collapsed ? '\u25B6' : '\u25C0'}
      </button>
      {/* Title */}
      {!collapsed && <div style={S.title}>{t('shell:appTitle')}</div>}
      {!collapsed && <div style={S.subtitle}>{t('shell:appSubtitle')}</div>}
      <div style={S.separator} />

      {/* Status indicator area */}
      <div style={{ ...S.statusArea, justifyContent: collapsed ? 'center' : 'flex-start', padding: collapsed ? '6px 0' : '6px 16px' }}>
        <span style={{ ...S.statusDot, background: statusColor }} title={`Backend: ${backendStatus}`} />
        {!collapsed && (
          <span style={S.statusLabel}>
            Backend: {backendStatus}
          </span>
        )}
      </div>
      <div style={S.separator} />

      {/* Navigation buttons — grouped by workflow section */}
      <nav className="thin-scrollbar" style={S.nav} aria-label={t('shell:hoverTips.moduleNav')}>
        {pages.map((page, idx) => {
          // Section headers
          if (page.sectionKey) {
            if (collapsed) return null; // hide section labels when collapsed
            return (
              <div key={`sec-${idx}`} style={{
                fontSize: '9px', fontWeight: 600, textTransform: 'uppercase',
                letterSpacing: '1.2px', color: colors.textSecondary,
                padding: idx === 0 ? '2px 16px 4px' : '12px 16px 4px',
                opacity: 0.7, userSelect: 'none',
              }}>
                {t(`nav:sections.${page.sectionKey}`)}
              </div>
            );
          }

          const isActive = currentPage === page.id;
          const isHovered = hovered === page.id;
          const IconComponent = page.icon ? ICON_MAP[page.icon] : null;
          const label = t(`nav:pages.${page.id}.label`);
          const tooltip = t(`nav:pages.${page.id}.tooltip`, { defaultValue: label });

          return (
            <button
              key={page.id}
              style={{
                ...S.navBtn,
                ...(isActive ? S.navBtnActive : {}),
                ...(isHovered && !isActive ? S.navBtnHover : {}),
                ...(collapsed ? { justifyContent: 'center', padding: '12px 0', position: 'relative' } : {}),
              }}
              onClick={() => onNavigate(page.id)}
              onMouseEnter={() => setHovered(page.id)}
              onMouseLeave={() => setHovered(null)}
              title={collapsed ? `${label}${page.shortcut ? ` (Ctrl+${page.shortcut})` : ''}` : tooltip}
            >
              {IconComponent ? (
                <IconComponent size={16} strokeWidth={1.8} style={{ flexShrink: 0, opacity: isActive ? 1 : 0.6 }} />
              ) : page.icon && (
                <span style={{ fontSize: '12pt', width: 20, textAlign: 'center', flexShrink: 0, opacity: isActive ? 1 : 0.7 }}>
                  {page.icon}
                </span>
              )}
              {!collapsed && <span>{label}</span>}
              {dataReady[page.id] && (
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', background: colors.green,
                  flexShrink: 0, marginLeft: collapsed ? 0 : 'auto',
                  position: collapsed ? 'absolute' : 'static',
                  top: collapsed ? 4 : undefined, right: collapsed ? 4 : undefined,
                  boxShadow: `0 0 4px ${colors.green}88`,
                }} title={t('shell:sidebar.dataLoaded')} />
              )}
              {!collapsed && !dataReady[page.id] && page.shortcut && (
                <span style={S.shortcut}>Ctrl+{page.shortcut}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Spacer already handled by flex — tools pushed to bottom */}
      <div style={S.separator} />

      {/* HDF5 Viewer tool button (bottom, non-checkable) */}
      <button
        style={{
          ...S.navBtn,
          ...(h5ViewerOpen ? S.navBtnActive : {}),
          ...(hovered === '_h5viewer' && !h5ViewerOpen ? S.navBtnHover : {}),
          margin: '4px 0',
          ...(collapsed ? { justifyContent: 'center', padding: '12px 0' } : {}),
        }}
        onClick={onToggleH5Viewer}
        onMouseEnter={() => setHovered('_h5viewer')}
        onMouseLeave={() => setHovered(null)}
        title={collapsed ? `${t('nav:hdf5Viewer.label')} (Ctrl+H)` : t('nav:hdf5Viewer.tooltip')}
        aria-label={t('nav:hdf5Viewer.label')}
      >
        <Database size={16} strokeWidth={1.8} style={{ flexShrink: 0, opacity: h5ViewerOpen ? 1 : 0.6 }} />
        {!collapsed && <span>{t('nav:hdf5Viewer.label')}</span>}
        {!collapsed && <span style={S.shortcut}>Ctrl+H</span>}
      </button>

      {/* Footer: theme selector + status */}
      <div style={{ ...S.footer, padding: collapsed ? '8px 4px' : '8px 16px' }}>
        {collapsed ? (
          /* Collapsed: theme cycle + language cycle buttons */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <button
              onClick={cycleTheme}
              title={t('shell:sidebar.themeTooltip', { name: THEME_LABELS[theme] })}
              aria-label={t('shell:sidebar.themeTooltip', { name: THEME_LABELS[theme] })}
              style={{
                background: 'transparent', border: `1px solid ${colors.sidebarBorder}`,
                borderRadius: 4, color: colors.textSecondary, cursor: 'pointer',
                fontSize: '10pt', padding: '4px 0', width: '100%', textAlign: 'center',
              }}
            >
              {THEME_ICONS[theme]}
            </button>
            <button
              onClick={cycleLanguage}
              title={t('shell:sidebar.language')}
              aria-label={t('shell:sidebar.language')}
              style={{
                background: 'transparent', border: `1px solid ${colors.sidebarBorder}`,
                borderRadius: 4, color: colors.textSecondary, cursor: 'pointer',
                fontSize: '8pt', fontWeight: 600, padding: '4px 0', width: '100%', textAlign: 'center',
              }}
            >
              {(LANGUAGES.find((l) => l.code === currentLang) || LANGUAGES[0]).short}
            </button>
          </div>
        ) : (
          <>
            {/* Theme selector row */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
              {THEME_ORDER.map((t) => {
                const isCurrentTheme = theme === t;
                return (
                  <button
                    key={t}
                    onClick={() => setTheme(t)}
                    title={THEME_LABELS[t]}
                    style={{
                      background: isCurrentTheme ? colors.sidebarActive : 'transparent',
                      border: `1px solid ${isCurrentTheme ? colors.accent : colors.sidebarBorder}`,
                      borderRadius: 4,
                      color: isCurrentTheme ? colors.accent : colors.textSecondary,
                      cursor: 'pointer',
                      fontSize: '7.5pt',
                      padding: '2px 6px',
                      transition: 'all 0.15s',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 3,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    <span style={{ fontSize: '8pt' }}>{THEME_ICONS[t]}</span>
                    <span>{THEME_LABELS[t]}</span>
                  </button>
                );
              })}
            </div>
            {/* Language selector row */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
              {LANGUAGES.map((lng) => {
                const isCurrent = currentLang === lng.code;
                return (
                  <button
                    key={lng.code}
                    onClick={() => setLanguage(lng.code)}
                    title={lng.label}
                    style={{
                      background: isCurrent ? colors.sidebarActive : 'transparent',
                      border: `1px solid ${isCurrent ? colors.accent : colors.sidebarBorder}`,
                      borderRadius: 4,
                      color: isCurrent ? colors.accent : colors.textSecondary,
                      cursor: 'pointer', fontSize: '7.5pt', padding: '2px 6px',
                      transition: 'all 0.15s',
                    }}
                  >
                    {lng.short}
                  </button>
                );
              })}
            </div>
            {/* Status row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '8pt' }}>
              <span style={{ ...S.statusDot, background: statusColor }} />
              <span>{backendStatus === 'connected' ? 'Connected' : backendStatus}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
