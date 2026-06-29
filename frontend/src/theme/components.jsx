/**
 * Shared UI components for Orienta.
 *
 * RULE: All pages import from here. No per-component Button/Input/etc.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { colors, spacing, fonts, alpha } from './tokens';

// Re-export tokens for convenience
export { colors, spacing, fonts, alpha } from './tokens';

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------
export function Button({
  children, onClick, disabled = false, variant = 'default',
  style: extra = {}, title, small = false, type = 'button',
  'aria-label': ariaLabel,
}) {
  const [hovered, setHovered] = useState(false);

  const base = {
    padding: small ? '4px 10px' : '6px 16px',
    borderRadius: 4,
    fontSize: small ? '11px' : '13px',
    fontWeight: 500,
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'background 0.12s, opacity 0.12s, transform 0.08s, filter 0.12s',
    opacity: disabled ? 0.45 : 1,
    whiteSpace: 'nowrap',
    border: 'none',
    height: small ? 26 : spacing.buttonHeight,
    lineHeight: 1,
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    ...extra,
  };

  const variants = {
    default: {
      background: hovered && !disabled ? 'var(--bg-hover)' : colors.border,
      color: colors.text,
      border: `1px solid ${colors.border}`,
    },
    primary: {
      background: colors.accent,
      color: colors.textOnAccent,
      filter: hovered && !disabled ? 'brightness(1.15)' : 'none',
    },
    purple: {
      background: colors.purple,
      color: colors.textOnAccent,
      filter: hovered && !disabled ? 'brightness(1.15)' : 'none',
    },
    danger: {
      background: colors.red,
      color: colors.textOnAccent,
      filter: hovered && !disabled ? 'brightness(1.15)' : 'none',
    },
    success: {
      background: colors.green,
      color: colors.textOnAccent,
      filter: hovered && !disabled ? 'brightness(1.15)' : 'none',
    },
    warning: {
      background: colors.orange,
      color: colors.textOnAccent,
      filter: hovered && !disabled ? 'brightness(1.15)' : 'none',
    },
    ghost: {
      background: hovered && !disabled ? colors.sidebarActive : 'transparent',
      color: hovered && !disabled ? colors.accent : colors.textSecondary,
      border: `1px solid ${hovered && !disabled ? colors.border : 'transparent'}`,
    },
  };

  return (
    <button
      type={type}
      style={{ ...base, ...variants[variant] }}
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Input — matches PyQt5 QLineEdit / QSpinBox
// ---------------------------------------------------------------------------
export function Input({
  value, onChange, placeholder, type = 'text', disabled = false,
  min, max, step, style: extra = {}, ...rest
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      min={min}
      max={max}
      step={step}
      style={{
        width: '100%',
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        color: colors.text,
        fontSize: '10pt',
        padding: '4px 8px',
        height: spacing.buttonHeight,
        outline: 'none',
        boxSizing: 'border-box',
        ...extra,
      }}
      {...rest}
    />
  );
}

// ---------------------------------------------------------------------------
// NumberInput — matches PyQt5 QSpinBox / QDoubleSpinBox
// ---------------------------------------------------------------------------
export function NumberInput({
  value, onChange, min, max, step = 1, disabled = false,
  style: extra = {}, label, ...rest
}) {
  return (
    <input
      type="number"
      value={value}
      onChange={onChange}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      style={{
        width: 80,
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        color: colors.text,
        fontSize: '10pt',
        padding: '4px 8px',
        height: spacing.buttonHeight,
        outline: 'none',
        boxSizing: 'border-box',
        ...extra,
      }}
      {...rest}
    />
  );
}

// ---------------------------------------------------------------------------
// Select — matches PyQt5 QComboBox
// ---------------------------------------------------------------------------
export function Select({
  value, onChange, options = [], disabled = false, style: extra = {},
  title, 'aria-label': ariaLabel,
}) {
  return (
    <select
      value={value}
      onChange={onChange}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      style={{
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        color: colors.text,
        fontSize: '10pt',
        padding: '4px 8px',
        height: spacing.buttonHeight,
        outline: 'none',
        cursor: 'pointer',
        ...extra,
      }}
    >
      {options.map((opt) => {
        const val = typeof opt === 'string' ? opt : opt.value;
        const label = typeof opt === 'string' ? opt : opt.label;
        const tip = typeof opt === 'object' ? opt.tip : undefined;
        return <option key={val} value={val} title={tip}>{label}</option>;
      })}
    </select>
  );
}

// ---------------------------------------------------------------------------
// Label — matches PyQt5 QLabel
// ---------------------------------------------------------------------------
export function Label({ children, secondary = false, small = false, style: extra = {} }) {
  return (
    <span style={{
      color: secondary ? colors.textSecondary : colors.text,
      fontSize: small ? '9pt' : '10pt',
      ...extra,
    }}>
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// SectionTitle — matches PyQt5 GroupBox title styling
// ---------------------------------------------------------------------------
export function SectionTitle({ children, style: extra = {} }) {
  return (
    <div style={{
      fontSize: '11pt',
      fontWeight: 'bold',
      color: colors.accent,
      marginBottom: spacing.groupSpacing,
      ...extra,
    }}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// GroupBox — matches PyQt5 QGroupBox styling
// ---------------------------------------------------------------------------
export function GroupBox({ title, children, style: extra = {} }) {
  return (
    <div className="group-focus" style={{
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
      padding: `${spacing.groupMargin}px`,
      marginBottom: spacing.outerSpacing,
      transition: 'border-color 0.15s, box-shadow 0.15s',
      ...extra,
    }}>
      {title && (
        <div style={{
          fontSize: '11pt',
          fontWeight: 700,
          color: colors.accent,
          marginBottom: spacing.groupSpacing,
        }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Card — matches PyQt5 dashboard card styling
// ---------------------------------------------------------------------------
export function Card({ children, onClick, accentColor, style: extra = {}, 'aria-label': ariaLabel }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      style={{
        background: colors.bgSecondary,
        border: `1px solid ${hovered ? (accentColor || colors.accent) : colors.border}`,
        borderRadius: 8,
        padding: '16px 14px',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'transform 0.15s, border-color 0.15s, box-shadow 0.15s',
        transform: hovered && onClick ? 'translateY(-2px)' : 'none',
        boxShadow: hovered && onClick
          ? `0 4px 16px rgba(0,0,0,0.3)`
          : '0 1px 4px rgba(0,0,0,0.2)',
        ...extra,
      }}
      onClick={onClick}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(e); } } : undefined}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={ariaLabel}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tabs — matches PyQt5 QTabWidget
// ---------------------------------------------------------------------------
export function Tabs({ tabs, activeTab, onTabChange, style: extra = {} }) {
  const [hovered, setHovered] = useState(null);
  return (
    <div role="tablist" style={{
      display: 'flex',
      borderBottom: `1px solid ${colors.border}`,
      ...extra,
    }}>
      {tabs.map((tab) => {
        const active = activeTab === tab.id;
        const isHov = hovered === tab.id && !active;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={active}
            onClick={() => onTabChange(tab.id)}
            onMouseEnter={() => setHovered(tab.id)}
            onMouseLeave={() => setHovered(null)}
            title={tab.tip}
            style={{
              padding: '8px 16px',
              background: isHov ? `${colors.border}44` : 'transparent',
              border: 'none',
              borderBottom: active ? `2px solid ${colors.accent}` : isHov ? `2px solid ${colors.border}` : '2px solid transparent',
              color: active ? colors.accent : isHov ? colors.text : colors.textSecondary,
              fontWeight: active ? 600 : 400,
              fontSize: '10pt',
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TabPanel — content area for a tab
// ---------------------------------------------------------------------------
export function TabPanel({ visible, children, style: extra = {} }) {
  if (!visible) return null;
  return (
    <div role="tabpanel" style={{
      padding: spacing.outerMargin,
      flex: 1,
      overflow: 'auto',
      animation: 'pageFadeIn 0.12s ease-out',
      ...extra,
    }}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CollapsibleGroup — matches PyQt5 CollapsibleGroupBox from base_widgets.py
// ---------------------------------------------------------------------------
export function CollapsibleGroup({ title, defaultCollapsed = false, children }) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [hovered, setHovered] = useState(false);

  // React-controlled background instead of direct style.background mutation.
  // The previous onMouseLeave wrote an empty string, which reverted the
  // button to the browser's default chrome (near-white on some themes) —
  // title text vanished into the button's default colour. Driving the
  // colour from state keeps the header readable in every hover state.
  return (
    <div style={{ marginBottom: spacing.compactSpacing }}>
      <button
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          textAlign: 'left',
          border: `1px solid ${colors.border}`,
          padding: '5px 10px',
          fontWeight: 600,
          fontSize: '10pt',
          backgroundColor: hovered ? colors.bgTertiary : colors.bgSecondary,
          color: colors.text,
          borderRadius: 4,
          cursor: 'pointer',
          width: '100%',
          display: 'block',
          transition: 'background 0.15s, border-color 0.15s',
        }}
      >
        <span style={{
          display: 'inline-block',
          transition: 'transform 0.15s',
          transform: collapsed ? 'rotate(0deg)' : 'rotate(90deg)',
          color: colors.accent,
          marginRight: 6,
        }}>{'\u25b6'}</span>  {title}
      </button>
      {!collapsed && (
        <div style={{
          padding: spacing.innerMargin,
          display: 'flex',
          flexDirection: 'column',
          gap: spacing.innerSpacing,
          animation: 'pageFadeIn 0.12s ease-out',
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ResizableSplitter — matches PyQt5 QSplitter (horizontal)
// ---------------------------------------------------------------------------
export function ResizableSplitter({
  left, right, defaultLeftWidth = 300,
  minLeftWidth = 150, maxLeftWidth = 600,
  style: extra = {},
}) {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  const onMouseDown = useCallback((e) => {
    dragging.current = true;
    startX.current = e.clientX;
    startWidth.current = leftWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMouseMove = (e) => {
      if (!dragging.current) return;
      const delta = e.clientX - startX.current;
      const newWidth = Math.min(maxLeftWidth, Math.max(minLeftWidth, startWidth.current + delta));
      setLeftWidth(newWidth);
    };

    const onMouseUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [leftWidth, minLeftWidth, maxLeftWidth]);

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden', ...extra }}>
      <div style={{ width: leftWidth, minWidth: minLeftWidth, flexShrink: 0, overflow: 'auto' }}>
        {left}
      </div>
      {/* Drag handle with grip dots */}
      <div
        onMouseDown={onMouseDown}
        style={{
          width: 6,
          cursor: 'col-resize',
          background: colors.border,
          flexShrink: 0,
          transition: 'background 0.15s, width 0.15s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = colors.accent; e.currentTarget.style.width = '7px'; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = colors.border; e.currentTarget.style.width = '6px'; }}
      >
        <div style={{ width: 2, height: 20, display: 'flex', flexDirection: 'column', gap: 3, alignItems: 'center' }}>
          {[0,1,2].map(i => <div key={i} style={{ width: 2, height: 2, borderRadius: '50%', background: 'currentColor', opacity: 0.4 }} />)}
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {right}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// VerticalSplitter — matches PyQt5 QSplitter(Qt.Vertical)
// ---------------------------------------------------------------------------
export function VerticalSplitter({
  top, bottom, defaultTopHeight = 300,
  minTopHeight = 100, maxTopHeight = 800,
  style: extra = {},
}) {
  const [topHeight, setTopHeight] = useState(defaultTopHeight);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);

  const onMouseDown = useCallback((e) => {
    dragging.current = true;
    startY.current = e.clientY;
    startHeight.current = topHeight;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';

    const onMouseMove = (e) => {
      if (!dragging.current) return;
      const delta = e.clientY - startY.current;
      const newH = Math.min(maxTopHeight, Math.max(minTopHeight, startHeight.current + delta));
      setTopHeight(newH);
    };

    const onMouseUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [topHeight, minTopHeight, maxTopHeight]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden', ...extra }}>
      <div style={{ height: topHeight, minHeight: minTopHeight, flexShrink: 0, overflow: 'auto' }}>
        {top}
      </div>
      <div
        onMouseDown={onMouseDown}
        style={{
          height: 6,
          cursor: 'row-resize',
          background: colors.border,
          flexShrink: 0,
          transition: 'background 0.15s, height 0.15s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = colors.accent; e.currentTarget.style.height = '7px'; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = colors.border; e.currentTarget.style.height = '6px'; }}
      >
        <div style={{ height: 2, display: 'flex', gap: 3, alignItems: 'center' }}>
          {[0,1,2].map(i => <div key={i} style={{ width: 2, height: 2, borderRadius: '50%', background: 'currentColor', opacity: 0.4 }} />)}
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {bottom}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ScrollPanel — matches PyQt5 QScrollArea pattern from BaseSidebar
// ---------------------------------------------------------------------------
export function ScrollPanel({ children, style: extra = {} }) {
  return (
    <div style={{
      overflow: 'auto',
      flex: 1,
      padding: spacing.compactMargin,
      ...extra,
    }}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// FormRow — horizontal label + input pair
// ---------------------------------------------------------------------------
export function FormRow({ label, children, style: extra = {} }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: spacing.innerSpacing,
      marginBottom: spacing.innerSpacing,
      ...extra,
    }}>
      {label && (
        <span style={{
          color: colors.textSecondary,
          fontSize: '9pt',
          minWidth: 80,
          flexShrink: 0,
        }}>
          {label}
        </span>
      )}
      <div style={{ flex: 1 }}>
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProgressBar — matches PyQt5 QProgressBar
// ---------------------------------------------------------------------------
export function ProgressBar({ value = 0, max = 100, label, color, visible = true, style: extra = {} }) {
  if (visible === false) return null;
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const isActive = pct > 0 && pct < 100;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 2,
      ...extra,
    }}>
      {label && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span style={{ fontSize: '9pt', color: colors.textSecondary }}>{label}</span>
          {pct > 0 && <span style={{ fontSize: '8pt', color: colors.textSecondary, fontFamily: 'monospace' }}>{Math.round(pct)}%</span>}
        </div>
      )}
      <div
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label || 'Progress'}
        style={{
          height: 8,
          background: colors.bg,
          borderRadius: 4,
          border: `1px solid ${colors.border}`,
          overflow: 'hidden',
        }}
      >
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: isActive
            ? `linear-gradient(90deg, ${color || colors.accent}, ${colors.purple}, ${color || colors.accent})`
            : (color || colors.accent),
          backgroundSize: isActive ? '200% 100%' : undefined,
          animation: isActive ? 'skeleton-shimmer 2s ease-in-out infinite' : undefined,
          borderRadius: 4,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EmptyState — centered placeholder for pages/panels with no data
// ---------------------------------------------------------------------------
export function EmptyState({ icon, title, children, action, style: extra = {} }) {
  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 32,
      textAlign: 'center',
      gap: 8,
      animation: 'fadeSlideIn 0.3s ease-out',
      ...extra,
    }}>
      {icon && <div style={{ fontSize: 28, opacity: 0.3, color: colors.textSecondary }}>{icon}</div>}
      {title && <div style={{ fontSize: '11pt', fontWeight: 600, color: colors.textSecondary }}>{title}</div>}
      {children && <div style={{ fontSize: '9pt', color: colors.textSecondary, opacity: 0.7, maxWidth: 360, lineHeight: 1.5 }}>{children}</div>}
      {action && <div style={{ marginTop: 8 }}>{action}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Separator — matches PyQt5 QFrame HLine
// ---------------------------------------------------------------------------
export function Separator({ style: extra = {} }) {
  return (
    <div style={{
      height: spacing.separatorHeight,
      background: colors.border,
      margin: `${spacing.separatorMargin}px 0`,
      ...extra,
    }} />
  );
}

// ---------------------------------------------------------------------------
// StatusDot — small colored indicator
// ---------------------------------------------------------------------------
export function StatusDot({ color, label, style: extra = {} }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, ...extra }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: color, display: 'inline-block', flexShrink: 0,
      }} />
      {label && <span style={{ fontSize: '9pt', color: colors.textSecondary }}>{label}</span>}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Skeleton — animated loading placeholder
// ---------------------------------------------------------------------------
const skeletonKeyframes = `
@keyframes skeleton-shimmer {
  0% { background-position: -200px 0; }
  100% { background-position: calc(200px + 100%) 0; }
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}`;

// Inject the keyframes once
if (typeof document !== 'undefined' && !document.getElementById('skeleton-style')) {
  const style = document.createElement('style');
  style.id = 'skeleton-style';
  style.textContent = skeletonKeyframes;
  document.head.appendChild(style);
}

export function Skeleton({ width = '100%', height = 16, radius = 4, style: extra = {} }) {
  return (
    <div style={{
      width,
      height,
      borderRadius: radius,
      background: `linear-gradient(90deg, ${colors.border} 25%, ${colors.bgTertiary} 50%, ${colors.border} 75%)`,
      backgroundSize: '200px 100%',
      animation: 'skeleton-shimmer 1.5s ease-in-out infinite',
      ...extra,
    }} />
  );
}

// ---------------------------------------------------------------------------
// LoadingOverlay — centered spinner with optional message
// ---------------------------------------------------------------------------
export function LoadingOverlay({ message = 'Loading...', visible = true }) {
  if (!visible) return null;
  return (
    <div role="status" aria-label={message} style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 12,
      padding: 40,
      color: colors.textSecondary,
      flex: 1,
      animation: 'fadeSlideIn 0.3s ease-out',
    }}>
      <div style={{
        width: 32,
        height: 32,
        border: `3px solid ${colors.border}`,
        borderTop: `3px solid ${colors.accent}`,
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
      }} />
      <span style={{ fontSize: '10pt' }}>{message}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TreeView — matches PyQt5 QTreeWidget for HDF5 structure
// ---------------------------------------------------------------------------
export function TreeView({ nodes = [], onSelect, style: extra = {} }) {
  return (
    <div role="tree" style={{
      background: colors.bg,
      border: `1px solid ${colors.border}`,
      borderRadius: 4,
      overflow: 'auto',
      fontSize: '9pt',
      fontFamily: "'Courier New', monospace",
      ...extra,
    }}>
      {nodes.map((node, i) => (
        <TreeNode key={i} node={node} depth={0} onSelect={onSelect} />
      ))}
    </div>
  );
}

function TreeNode({ node, depth, onSelect }) {
  const [expanded, setExpanded] = useState(depth < 1);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div role="treeitem" aria-expanded={hasChildren ? expanded : undefined}>
      <div
        tabIndex={0}
        onClick={() => {
          if (hasChildren) setExpanded(!expanded);
          if (onSelect) onSelect(node);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (hasChildren) setExpanded(!expanded); if (onSelect) onSelect(node); }
          if (e.key === 'ArrowRight' && hasChildren && !expanded) { e.preventDefault(); setExpanded(true); }
          if (e.key === 'ArrowLeft' && hasChildren && expanded) { e.preventDefault(); setExpanded(false); }
        }}
        style={{
          padding: '2px 4px',
          paddingLeft: depth * 16 + 4,
          cursor: 'pointer',
          color: hasChildren ? colors.accent : colors.text,
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}
        onMouseEnter={(e) => e.currentTarget.style.background = colors.sidebarActive}
        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
      >
        {hasChildren && <span>{expanded ? '\u25bc' : '\u25b6'}</span>}
        {!hasChildren && <span style={{ width: 10 }} />}
        <span>{node.name || node.label}</span>
        {node.type && (
          <span style={{ color: colors.textSecondary, marginLeft: 'auto' }}>
            {node.type}
          </span>
        )}
      </div>
      {expanded && hasChildren && node.children.map((child, i) => (
        <TreeNode key={i} node={child} depth={depth + 1} onSelect={onSelect} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// useConfirm — hook that returns [ask, ConfirmDialogProps]
// Usage: const [ask, confirmProps] = useConfirm();
//        ask({ title, message, onConfirm, variant });
//        <ConfirmDialog {...confirmProps} />
// ---------------------------------------------------------------------------
export function useConfirm() {
  const [state, setState] = useState(null);
  const ask = useCallback((opts) => setState(opts), []);
  const close = useCallback(() => setState(null), []);
  const props = {
    open: !!state,
    title: state?.title,
    message: state?.message,
    confirmLabel: state?.confirmLabel,
    variant: state?.variant,
    onConfirm: () => { state?.onConfirm?.(); close(); },
    onCancel: close,
  };
  return [ask, props];
}

// ---------------------------------------------------------------------------
// ConfirmDialog — themed replacement for window.confirm()
// ---------------------------------------------------------------------------
export function ConfirmDialog({
  open, title = 'Confirm', message, confirmLabel = 'Confirm',
  cancelLabel = 'Cancel', variant = 'danger', onConfirm, onCancel,
}) {
  const dialogRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const h = (e) => {
      if (e.key === 'Escape') onCancel();
      // Focus trap: cycle Tab between the two buttons
      if (e.key === 'Tab') {
        const focusable = dialogRef.current?.querySelectorAll('button');
        if (!focusable || focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) { e.preventDefault(); last.focus(); }
        } else {
          if (document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      }
    };
    window.addEventListener('keydown', h);
    // Auto-focus the cancel button (safe default)
    const timer = setTimeout(() => {
      dialogRef.current?.querySelector('button')?.focus();
    }, 50);
    return () => { window.removeEventListener('keydown', h); clearTimeout(timer); };
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      onClick={onCancel}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 10000, animation: 'pageFadeIn 0.15s ease-out',
      }}
    >
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-label={title}
        aria-describedby="confirm-dialog-msg"
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bg, border: `1px solid ${colors.border}`,
          borderRadius: 8, padding: '20px 24px', minWidth: 320, maxWidth: 440,
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          animation: 'pageFadeIn 0.2s ease-out',
        }}
      >
        <div style={{ fontSize: '11pt', fontWeight: 600, color: colors.text, marginBottom: 8 }}>
          {title}
        </div>
        <div id="confirm-dialog-msg" style={{ fontSize: '10pt', color: colors.textSecondary, lineHeight: 1.5, marginBottom: 18 }}>
          {message}
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <Button onClick={onCancel}>{cancelLabel}</Button>
          <Button variant={variant} onClick={onConfirm}>{confirmLabel}</Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// usePrompt — hook for themed replacement of window.prompt()
// Usage: const [prompt, promptProps] = usePrompt();
//        prompt({ title, message, defaultValue, onSubmit, placeholder });
//        <PromptDialog {...promptProps} />
// ---------------------------------------------------------------------------
export function usePrompt() {
  const [state, setState] = useState(null);
  const prompt = useCallback((opts) => setState({ ...opts, value: opts.defaultValue || '' }), []);
  const close = useCallback(() => setState(null), []);
  const props = {
    open: !!state,
    title: state?.title,
    message: state?.message,
    value: state?.value || '',
    placeholder: state?.placeholder,
    submitLabel: state?.submitLabel,
    onChange: (v) => setState((s) => s ? { ...s, value: v } : null),
    onSubmit: (v) => { state?.onSubmit?.(v); close(); },
    onCancel: close,
  };
  return [prompt, props];
}

// ---------------------------------------------------------------------------
// PromptDialog — themed replacement for window.prompt()
// ---------------------------------------------------------------------------
export function PromptDialog({
  open, title = 'Input', message, value = '', placeholder = '',
  submitLabel = 'OK', onChange, onSubmit, onCancel,
}) {
  const dialogRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const h = (e) => {
      if (e.key === 'Escape') onCancel();
      if (e.key === 'Tab') {
        const focusable = dialogRef.current?.querySelectorAll(
          'input, button, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable || focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) { e.preventDefault(); last.focus(); }
        } else {
          if (document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      }
    };
    window.addEventListener('keydown', h);
    const timer = setTimeout(() => {
      if (inputRef.current) { inputRef.current.focus(); inputRef.current.select(); }
    }, 50);
    return () => { window.removeEventListener('keydown', h); clearTimeout(timer); };
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      onClick={onCancel}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 10000, animation: 'pageFadeIn 0.15s ease-out',
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bg, border: `1px solid ${colors.border}`,
          borderRadius: 8, padding: '20px 24px', minWidth: 360, maxWidth: 500,
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          animation: 'pageFadeIn 0.2s ease-out',
        }}
      >
        <div style={{ fontSize: '11pt', fontWeight: 600, color: colors.text, marginBottom: 8 }}>
          {title}
        </div>
        {message && (
          <div style={{ fontSize: '10pt', color: colors.textSecondary, lineHeight: 1.5, marginBottom: 10 }}>
            {message}
          </div>
        )}
        <input
          ref={inputRef}
          type="text"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange?.(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && value.trim()) onSubmit?.(value); }}
          style={{
            width: '100%', boxSizing: 'border-box',
            padding: '6px 10px', fontSize: '10pt',
            background: colors.bgSecondary, color: colors.text,
            border: `1px solid ${colors.border}`, borderRadius: 4,
            outline: 'none', marginBottom: 16,
            transition: 'border-color 0.15s',
          }}
          onFocus={(e) => { e.target.style.borderColor = colors.purple; }}
          onBlur={(e) => { e.target.style.borderColor = colors.border; }}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <Button onClick={onCancel}>Cancel</Button>
          <Button variant="primary" onClick={() => onSubmit?.(value)} disabled={!value.trim()}>
            {submitLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
