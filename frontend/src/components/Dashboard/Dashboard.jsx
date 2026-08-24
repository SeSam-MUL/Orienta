/**
 * Dashboard — module overview with workflow-ordered cards.
 *
 * Layout:
 *   Title "Orienta" — accent, centered
 *   Row 1: Data & Calibration modules
 *   Row 2: Indexing & Analysis modules
 *   Recent Files section
 *   Footer hint
 */

import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha } from '../../theme/tokens';
import { Button, useConfirm, ConfirmDialog } from '../../theme/components';
import { Wordmark } from '../common/Brand';
import useDataStore from '../../stores/useDataStore';
import useResultStore from '../../stores/useResultStore';
import {
  ScanLine, Atom, Crosshair, Gem, Cpu, HardDrive,
  Grid3x3, Map, BarChart3, Brain, Package, SlidersHorizontal,
} from 'lucide-react';
import CrystalBackground from './CrystalBackground';
import VantaBackground from './VantaBackground';
import WandererOverlay from './WandererOverlay';

// ---------------------------------------------------------------------------
// Module card definitions — workflow order, unified accent color
// ---------------------------------------------------------------------------

// Card text (title/description/tooltip) is looked up at render time via
// t('dashboard:cards.<id>.*'); only id, accent color and icon live here.
const ROW1 = [
  { id: 'ebsdviewer', accentVar: '--accent-cyan', LucideIcon: ScanLine },
  { id: 'eds', accentVar: '--accent-cyan', LucideIcon: Atom },
  { id: 'pcrefinement', accentVar: '--accent-cyan', LucideIcon: Crosshair },
  { id: 'crystal', accentVar: '--accent-cyan', LucideIcon: Gem },
];

const ROW2 = [
  { id: 'indexing', accentVar: '--accent-cyan', LucideIcon: Grid3x3 },
  { id: 'phasemap', accentVar: '--accent-cyan', LucideIcon: Map },
  { id: 'analysis', accentVar: '--accent-cyan', LucideIcon: BarChart3 },
  { id: 'mlhub', accentVar: '--accent-cyan', LucideIcon: Brain },
  { id: 'batch', accentVar: '--accent-purple', LucideIcon: Package },
  { id: 'refinement', accentVar: '--accent-green', LucideIcon: SlidersHorizontal },
];

// ---------------------------------------------------------------------------
// localStorage helpers — mirrors QSettings("Orienta","MainHub")
// ---------------------------------------------------------------------------

const LS_KEY = 'kikuchipy_recent_files';
const MAX_RECENT = 8;

function loadRecentFiles() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveRecentFiles(list) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(list));
  } catch {
    // ignore quota errors
  }
}

export function addRecentFile(filePath) {
  let recent = loadRecentFiles();
  recent = recent.filter((p) => p !== filePath);
  recent.unshift(filePath);
  recent = recent.slice(0, MAX_RECENT);
  saveRecentFiles(recent);
}

// ---------------------------------------------------------------------------
// Dashboard background preference
// ---------------------------------------------------------------------------

const BG_KEY = 'kikuchipy_dashboard_bg';

function getDashboardBg() {
  try {
    return localStorage.getItem(BG_KEY) || 'clean';
  } catch {
    return 'clean';
  }
}

// ---------------------------------------------------------------------------
// ModuleCard — styled to match _create_card() from dashboard_gui.py
// background-color: #2d2f3d, border: 1px solid #3b3d54, border-radius: 10px
// hover: border 2px solid #8be9fd, background-color: #353749
// title: 14pt bold, accent color — desc: 9pt #aaa
// ---------------------------------------------------------------------------

function ModuleCard({ id, accentVar, icon, LucideIcon, onNavigate, ready }) {
  const { t } = useTranslation('dashboard');
  const [hovered, setHovered] = useState(false);

  const accent = `var(${accentVar})`;
  const title = t(`dashboard:cards.${id}.title`);
  const description = t(`dashboard:cards.${id}.description`);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onNavigate(id)}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onNavigate(id)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={t(`dashboard:cards.${id}.tooltip`)}
      style={{
        background: hovered
          ? 'rgba(255, 255, 255, 0.07)'
          : 'rgba(255, 255, 255, 0.04)',
        backdropFilter: 'blur(20px) saturate(1.2)',
        WebkitBackdropFilter: 'blur(20px) saturate(1.2)',
        border: `1px solid ${hovered ? `${accent}55` : 'rgba(255,255,255,0.10)'}`,
        borderRadius: 16,
        padding: 0,
        cursor: 'pointer',
        minHeight: 120,
        display: 'flex',
        flexDirection: 'column',
        userSelect: 'none',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        transform: hovered ? 'translateY(-4px)' : 'translateY(0)',
        boxShadow: hovered
          ? `0 16px 48px rgba(0,0,0,0.35), 0 0 20px ${accent}15, inset 0 1px 0 rgba(255,255,255,0.06)`
          : '0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.04)',
        flex: 1,
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      {/* Top accent line — subtle gradient, visible on hover */}
      <div style={{
        height: 1,
        background: `linear-gradient(90deg, transparent, ${accent}, transparent)`,
        opacity: hovered ? 0.6 : 0,
        transition: 'opacity 0.3s ease',
      }} />

      <div style={{ padding: '20px 20px 22px', display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>
        {/* Title row with icon */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          {LucideIcon ? (
            <LucideIcon size={28} strokeWidth={1.6} style={{
              color: accent,
              opacity: hovered ? 1 : 0.6,
              transition: 'opacity 0.2s ease',
              flexShrink: 0,
            }} />
          ) : icon && (
            <span style={{
              fontSize: 18,
              color: accent,
              opacity: hovered ? 1 : 0.7,
              transition: 'opacity 0.2s ease',
            }}>
              {icon}
            </span>
          )}
          <span style={{
            fontSize: 13,
            fontWeight: 600,
            color: 'rgba(255,255,255,0.85)',
            lineHeight: 1.2,
            flex: 1,
          }}>
            {title}
          </span>
          {ready && (
            <span title={t('dashboard:ready.tooltip')} style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '1px 6px', borderRadius: 8,
              background: 'rgba(80,250,123,0.12)', flexShrink: 0,
            }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: colors.green }} />
              <span style={{ fontSize: '7pt', color: colors.green, fontWeight: 600 }}>{t('dashboard:ready.label')}</span>
            </span>
          )}
        </div>

        {/* Description — flowing text, equal card heights via flex stretch */}
        <div style={{
          fontSize: '9pt',
          color: colors.textSecondary,
          lineHeight: 1.5,
          flex: 1,
        }}>
          {description}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CardsRow — QHBoxLayout equivalent
// ---------------------------------------------------------------------------

function CardsRow({ cards, onNavigate, readyMap }) {
  return (
    <div style={{
      display: 'flex',
      gap: 14,
      width: '100%',
      alignItems: 'stretch',
      animation: 'pageFadeIn 0.3s ease-out',
    }}>
      {cards.map((card) => (
        <ModuleCard
          key={card.id}
          id={card.id}
          accentVar={card.accentVar}
          LucideIcon={card.LucideIcon}
          onNavigate={onNavigate}
          ready={readyMap?.[card.id]}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// RecentFilesSection — mirrors QGroupBox "Recent Files" + QListWidget
// ---------------------------------------------------------------------------

function RecentFilesSection({ onFileOpen }) {
  const { t } = useTranslation('dashboard');
  const [recentFiles, setRecentFiles] = useState([]);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [askConfirm, confirmProps] = useConfirm();

  useEffect(() => {
    setRecentFiles(loadRecentFiles());
    // Re-read when user navigates back to Dashboard
    const onPageChanged = (e) => {
      if (e.detail?.page === 'dashboard') setRecentFiles(loadRecentFiles());
    };
    window.addEventListener('page-changed', onPageChanged);
    return () => window.removeEventListener('page-changed', onPageChanged);
  }, []);

  const handleQuickLoad = useCallback(() => {
    // Load the first file that exists — we cannot verify existence in browser
    // so we just emit the first entry, matching _quick_load_last() behavior
    if (recentFiles.length > 0) {
      onFileOpen(recentFiles[0]);
    }
  }, [recentFiles, onFileOpen]);

  const handleClear = useCallback(() => {
    askConfirm({
      title: t('dashboard:recent.confirm.title'),
      message: t('dashboard:recent.confirm.message'),
      confirmLabel: t('dashboard:recent.confirm.confirmLabel'),
      onConfirm: () => { saveRecentFiles([]); setRecentFiles([]); },
    });
  }, [askConfirm, t]);

  const handleItemClick = useCallback((path) => {
    onFileOpen(path);
  }, [onFileOpen]);

  const basename = (path) => {
    // Works for both / and \ separators
    const parts = path.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1] || path;
  };

  const dirname = (path) => {
    const normalized = path.replace(/\\/g, '/');
    const idx = normalized.lastIndexOf('/');
    return idx > 0 ? path.substring(0, idx) : '';
  };

  return (
    <div style={{
      background: 'rgba(12, 16, 28, 0.35)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.05)',
      borderRadius: 12,
      padding: '16px 20px',
    }}>
      {/* GroupBox title — matches QGroupBox { font-size: 12pt; font-weight: bold; } */}
      <div style={{
        fontSize: '12pt',
        fontWeight: 'bold',
        color: colors.text,
        marginBottom: 8,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}>
        {t('dashboard:recent.title')}
        {recentFiles.length > 0 && (
          <span style={{
            fontSize: '8pt', fontWeight: 600, color: colors.textSecondary,
            background: colors.border, borderRadius: 8,
            padding: '1px 7px', lineHeight: 1.6,
          }}>
            {recentFiles.length}
          </span>
        )}
      </div>

      {/* List — matches QListWidget with max height 180px */}
      <div className="thin-scrollbar" style={{
        maxHeight: 180,
        overflowY: 'auto',
        background: 'rgba(0,0,0,0.15)',
        border: '1px solid rgba(255,255,255,0.04)',
        borderRadius: 6,
        marginBottom: 8,
      }}>
        {recentFiles.length === 0 ? (
          <div style={{
            padding: '20px 16px',
            color: colors.textSecondary,
            fontSize: '10pt',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: '18pt', marginBottom: 6, opacity: 0.4 }}>{'\u2B21'}</div>
            <div style={{ fontStyle: 'italic' }}>{t('dashboard:recent.empty.title')}</div>
            <div style={{ fontSize: '8pt', marginTop: 4, opacity: 0.6 }}>
              {t('dashboard:recent.empty.hint')}
            </div>
          </div>
        ) : (
          recentFiles.map((path, i) => (
            <div
              key={path}
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseLeave={() => setHoveredIndex(null)}
              onClick={() => handleItemClick(path)}
              title={t('dashboard:recent.itemTooltip', { path })}
              style={{
                padding: '6px 10px',
                background: hoveredIndex === i ? colors.border : 'transparent',
                cursor: 'pointer',
                fontSize: '10pt',
                color: colors.text,
                display: 'flex',
                gap: 8,
                alignItems: 'baseline',
                transition: 'background 0.15s',
                borderLeft: hoveredIndex === i ? `2px solid ${colors.accent}` : '2px solid transparent',
              }}
            >
              <span style={{ fontSize: '10pt', opacity: 0.5, flexShrink: 0 }}>
                {basename(path).endsWith('.h5oina') ? '\u2B22' : '\u25A3'}
              </span>
              <span style={{ fontWeight: 500 }}>{basename(path)}</span>
              <span style={{
                color: colors.textSecondary,
                fontSize: '9pt',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {dirname(path)}
              </span>
            </div>
          ))
        )}
      </div>

      {/* Button row — matching btn_quick_load (green) + btn_clear_recent */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Button
          variant="success"
          onClick={handleQuickLoad}
          disabled={recentFiles.length === 0}
          title={t('dashboard:recent.quickLoadTooltip')}
        >
          {t('dashboard:recent.quickLoad')}
        </Button>
        <div style={{ flex: 1 }} />
        <Button
          variant="default"
          onClick={handleClear}
          disabled={recentFiles.length === 0}
          title={t('dashboard:recent.clearTooltip')}
        >
          {t('dashboard:recent.clear')}
        </Button>
      </div>
      <ConfirmDialog {...confirmProps} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// ShortcutHint — collapsible keyboard shortcut reference
// ---------------------------------------------------------------------------

function ShortcutHint() {
  const { t } = useTranslation('dashboard');
  const [open, setOpen] = useState(false);

  return (
    <div style={{ textAlign: 'center' }}>
      <button
        onClick={() => setOpen((v) => !v)}
        title={t('dashboard:shortcuts.toggleTooltip')}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          fontSize: '9pt', color: colors.textSecondary, opacity: 0.6,
          padding: '4px 8px', transition: 'opacity 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.opacity = '1'; }}
        onMouseLeave={e => { e.currentTarget.style.opacity = '0.6'; }}
      >
        {t('dashboard:shortcuts.hint')}
        {' '}<span style={{ fontSize: '8pt' }}>{open ? '\u25B2' : '\u25BC'}</span>
      </button>
      {open && (
        <div style={{
          display: 'inline-grid', gridTemplateColumns: 'auto auto',
          gap: '4px 12px', fontSize: '8.5pt', color: colors.textSecondary,
          textAlign: 'left', marginTop: 6, animation: 'pageFadeIn 0.2s ease-out',
        }}>
          <span className="kbd">Ctrl+1</span><span>{t('dashboard:shortcuts.dashboard')}</span>
          <span className="kbd">Ctrl+2</span><span>{t('dashboard:shortcuts.ebsdViewer')}</span>
          <span className="kbd">Ctrl+3</span><span>{t('dashboard:shortcuts.analysis')}</span>
          <span className="kbd">Ctrl+4</span><span>{t('dashboard:shortcuts.crystalDb')}</span>
          <span className="kbd">Ctrl+5</span><span>{t('dashboard:shortcuts.simulation')}</span>
          <span className="kbd">Ctrl+6</span><span>{t('dashboard:shortcuts.database')}</span>
          <span className="kbd">Ctrl+7</span><span>{t('dashboard:shortcuts.phaseMaps')}</span>
          <span className="kbd">Ctrl+8</span><span>{t('dashboard:shortcuts.indexing')}</span>
          <span className="kbd">Ctrl+9</span><span>{t('dashboard:shortcuts.mlPredictor')}</span>
          <span className="kbd">Ctrl+0</span><span>{t('dashboard:shortcuts.pcRefinement')}</span>
          <span className="kbd">Ctrl+H</span><span>{t('dashboard:shortcuts.hdf5Viewer')}</span>
          <span className="kbd">Ctrl+E</span><span>{t('dashboard:shortcuts.edsAnalysis')}</span>
          <span className="kbd">Ctrl+B</span><span>{t('dashboard:shortcuts.toggleSidebar')}</span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard — main export, matches DashboardPage._build_ui() exactly
// ---------------------------------------------------------------------------

export default function Dashboard({ onNavigate, onFileOpen }) {
  const { t } = useTranslation('dashboard');
  const isFileOpen = useDataStore(s => s.isFileOpen);
  const analysisLoaded = useResultStore(s => s.analysisLoaded);
  const hasIndexingResult = useResultStore(s => !!s.indexingResult);

  const readyMap = {
    ebsdviewer: isFileOpen,
    analysis: analysisLoaded,
    indexing: isFileOpen,
    phasemap: hasIndexingResult,
    pcrefinement: isFileOpen,
    eds: isFileOpen,
  };

  const handleFileOpen = useCallback((path) => {
    if (onFileOpen) onFileOpen(path);
  }, [onFileOpen]);

  // Re-read background preference on every navigation back to Dashboard
  const [bgMode, setBgMode] = useState(getDashboardBg);

  useEffect(() => {
    // Re-read when user navigates to Dashboard via page-changed event
    const onPageChanged = (e) => {
      if (e.detail?.page === 'dashboard') setBgMode(getDashboardBg());
    };
    // Also listen for storage changes (from Settings page in same tab)
    const onStorage = () => setBgMode(getDashboardBg());
    window.addEventListener('page-changed', onPageChanged);
    window.addEventListener('dashboard-bg-changed', onStorage);
    return () => {
      window.removeEventListener('page-changed', onPageChanged);
      window.removeEventListener('dashboard-bg-changed', onStorage);
    };
  }, []);

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '100%',
      overflow: 'hidden',
    }}>
      {/* Background layer */}
      {bgMode === 'crystal' && <CrystalBackground />}
      {bgMode === 'fog' && <VantaBackground />}
      {bgMode === 'fog' && <WandererOverlay />}

      {/* Clean background via CSS */}
      {bgMode === 'clean' && (
        <div style={{
          position: 'absolute',
          inset: 0,
          zIndex: 0,
          background: colors.bg,
        }}>
          {/* Dot grid pattern */}
          <div style={{
            position: 'absolute',
            inset: 0,
            backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.02) 1px, transparent 1px)',
            backgroundSize: '20px 20px',
          }} />
          {/* Colored gradient blobs */}
          <div style={{
            position: 'absolute',
            inset: 0,
            background: `
              radial-gradient(ellipse 600px 400px at 30% 20%, rgba(139,233,253,0.06) 0%, transparent 70%),
              radial-gradient(ellipse 500px 500px at 70% 60%, rgba(189,147,249,0.05) 0%, transparent 70%),
              radial-gradient(ellipse 400px 300px at 50% 90%, rgba(80,250,123,0.03) 0%, transparent 70%)
            `,
          }} />
        </div>
      )}

      {/* Dashboard content */}
      <div style={{
        position: 'relative',
        zIndex: 2,
        padding: '36px 40px',
        display: 'flex',
        flexDirection: 'column',
        gap: 24,
        height: '100%',
        boxSizing: 'border-box',
        overflowY: 'auto',
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <Wordmark
            height={78}
            title={t('dashboard:title')}
            style={{ filter: `drop-shadow(0 0 60px ${alpha(colors.accent, 20)})` }}
          />
        </div>

        <div style={{
          fontSize: 14,
          color: colors.textSecondary,
          textAlign: 'center',
          marginTop: -16,
          letterSpacing: '0.03em',
        }}>
          {t('dashboard:subtitle')}
        </div>

        {/* Row label */}
        <div style={{
          fontSize: 11,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '1.5px',
          color: colors.textSecondary,
          opacity: 0.5,
          marginBottom: -12,
        }}>
          {t('dashboard:sections.dataCalibration')}
        </div>

        <CardsRow cards={ROW1} onNavigate={onNavigate} readyMap={readyMap} />

        <div style={{
          fontSize: 11,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '1.5px',
          color: colors.textSecondary,
          opacity: 0.5,
          marginBottom: -12,
        }}>
          {t('dashboard:sections.indexAnalysis')}
        </div>

        <CardsRow cards={ROW2} onNavigate={onNavigate} readyMap={readyMap} />

        <RecentFilesSection onFileOpen={handleFileOpen} />

        <div style={{ flex: 1 }} />

        <ShortcutHint />
      </div>
    </div>
  );
}
