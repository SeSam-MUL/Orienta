import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import SwipeCompareController from '../EDS/SwipeCompareController';

/**
 * Consolidated header toolbar for the PhaseMap inspection tools. Contains:
 *   - View toggle (Stack | Grid)
 *   - Tile-min slider (visible only in Grid view)
 *   - Linescan ON/OFF toggle
 *   - Magnifier (Lens) ON/OFF toggle
 *   - Export PNG button
 *   - SwipeCompareController (A/B layer dropdowns)
 *
 * All state is owned by the parent — this is a pure presentational shell.
 */
export default function ToolToolbar({
  view, setView,
  tileMinWidth, setTileMinWidth,
  linescanMode, setLinescanMode,
  magnifierEnabled, setMagnifierEnabled,
  onExport,
  swipe, setSwipe,
  layers,
}) {
  const { t } = useTranslation('phasemap');
  return (
    <div
      data-tool-toolbar
      style={{
        display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
        padding: '6px 8px',
        background: colors.bgSecondary,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        fontSize: '9pt',
      }}
    >
      <SegmentedToggle
        value={view}
        options={[
          { value: 'stack', label: t('phasemap:tools.stack'), title: t('phasemap:hoverTips.viewStack') },
          { value: 'grid',  label: t('phasemap:tools.grid'), title: t('phasemap:hoverTips.viewGrid') },
        ]}
        onChange={setView}
      />

      {view === 'grid' && (
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: colors.textSecondary }}>
          <span>{t('phasemap:tools.tile')}</span>
          <input
            type="range"
            min={140} max={520} step={10}
            value={tileMinWidth}
            onChange={(e) => setTileMinWidth?.(Number(e.target.value))}
            style={{ width: 120, accentColor: colors.purple }}
            aria-label={t('phasemap:tools.tileMinWidthAria')}
            title={t('phasemap:tools.tileMinWidthTooltip', { n: tileMinWidth })}
          />
          <span style={{ minWidth: 40, textAlign: 'right', color: colors.text }}>
            {t('phasemap:tools.tilePx', { n: tileMinWidth })}
          </span>
        </span>
      )}

      <ToggleButton
        on={linescanMode}
        onClick={() => setLinescanMode?.(!linescanMode)}
        title={t('phasemap:tools.linescanTooltip')}
      >
        {linescanMode ? t('phasemap:tools.linescanOn') : t('phasemap:tools.linescanOff')}
      </ToggleButton>

      <ToggleButton
        on={magnifierEnabled}
        onClick={() => setMagnifierEnabled?.(!magnifierEnabled)}
        title={t('phasemap:tools.lensTooltip')}
      >
        {magnifierEnabled ? t('phasemap:tools.lensOn') : t('phasemap:tools.lensOff')}
      </ToggleButton>

      <button
        onClick={onExport}
        title={t('phasemap:tools.exportPngTooltip')}
        style={{
          background: 'transparent',
          border: `1px solid ${colors.purple}`,
          color: colors.purple,
          borderRadius: 3,
          padding: '2px 8px',
          fontSize: '8.5pt',
          cursor: 'pointer',
          fontWeight: 600,
        }}
      >
        {t('phasemap:tools.exportPng')}
      </button>

      <div style={{ marginLeft: 'auto' }}>
        <SwipeCompareController
          layers={layers}
          value={swipe}
          onChange={setSwipe}
        />
      </div>
    </div>
  );
}

function SegmentedToggle({ value, options, onChange }) {
  return (
    <div
      data-segmented-toggle
      style={{
        display: 'inline-flex',
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        overflow: 'hidden',
        fontSize: '8.5pt',
      }}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange?.(opt.value)}
            title={opt.title}
            style={{
              background: active ? colors.purple : 'transparent',
              color: active ? colors.bg : colors.text,
              border: 'none',
              padding: '4px 12px',
              cursor: 'pointer',
              fontWeight: active ? 700 : 500,
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function ToggleButton({ on, onClick, title, children }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        background: on ? colors.cyan : 'transparent',
        color: on ? colors.bg : colors.cyan,
        border: `1px solid ${colors.cyan}`,
        borderRadius: 3,
        padding: '2px 8px',
        fontSize: '8.5pt',
        cursor: 'pointer',
        fontWeight: 600,
      }}
    >
      {children}
    </button>
  );
}
