import { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button, Input, GroupBox } from '../../theme/components';

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function XtalBadge({ exists, hasWarning, fitForXtal, fitWarnings, onDeleteXtal }) {
  const { t } = useTranslation('crystaldatabase');
  const [hovered, setHovered] = useState(false);

  if (exists) {
    return (
      <span
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 3,
          padding: '1px 6px', borderRadius: 10,
          fontSize: 10, fontWeight: 600, letterSpacing: '0.04em',
          whiteSpace: 'nowrap', flexShrink: 0,
          background: hovered ? alpha(colors.red, 15) : alpha(colors.green, 12),
          color: hovered ? colors.red : colors.green,
          border: `1px solid ${hovered ? alpha(colors.red, 40) : alpha(colors.green, 35)}`,
          transition: 'background 0.2s, color 0.2s, border-color 0.2s',
          cursor: hovered ? 'pointer' : 'default',
        }}
        title={hovered ? t('fileList.xtalBadge.clickDelete') : t('fileList.xtalBadge.exists')}
        onClick={(e) => {
          e.stopPropagation();
          if (onDeleteXtal) onDeleteXtal();
        }}
      >
        .xtal
        {hovered && (
          <span style={{ fontSize: 11, lineHeight: 1, fontWeight: 700 }}>&times;</span>
        )}
      </span>
    );
  }

  // No XTAL yet — determine color by fit status and parse warning
  const warnTips = (fitWarnings || []).join(' ');
  let badgeColor, badgeText, badgeTitle;
  if (hasWarning) {
    badgeColor = colors.yellow;
    badgeText = '!';
    badgeTitle = t('fileList.xtalBadge.parseWarning');
  } else if (fitForXtal === false) {
    badgeColor = colors.red;
    badgeText = '!';
    badgeTitle = warnTips || t('fileList.xtalBadge.notFit');
  } else if (fitForXtal === true) {
    badgeColor = colors.cyan;
    badgeText = '\u2014';
    badgeTitle = t('fileList.xtalBadge.readyToConvert');
  } else {
    // fitForXtal unknown (not yet parsed)
    badgeColor = colors.border;
    badgeText = '\u2014';
    badgeTitle = t('fileList.xtalBadge.noXtal');
  }

  return (
    <span
      title={badgeTitle}
      style={{
        display: 'inline-flex', alignItems: 'center',
        padding: '1px 6px', borderRadius: 10,
        fontSize: 10, fontWeight: 600, letterSpacing: '0.04em',
        whiteSpace: 'nowrap', flexShrink: 0,
        background: fitForXtal === false ? alpha(badgeColor, 8) : 'transparent',
        color: badgeColor,
        border: `1px solid ${badgeColor}`,
        transition: 'background 0.4s, color 0.4s, border-color 0.4s',
      }}>
      {badgeText}
    </span>
  );
}

function LocationDot({ location }) {
  const { t } = useTranslation('crystaldatabase');
  const dotColor =
    location === 'both'   ? colors.green  :
    location === 'local'  ? colors.yellow :
    location === 'server' ? colors.cyan   :
    colors.border;

  const label =
    location === 'both'   ? t('fileList.location.both')   :
    location === 'local'  ? t('fileList.location.local')  :
    location === 'server' ? t('fileList.location.server') :
    t('fileList.location.unknown');

  return (
    <span
      title={label}
      style={{
        display: 'inline-block',
        width: 8, height: 8, borderRadius: '50%',
        background: dotColor,
        flexShrink: 0,
      }}
    />
  );
}

function CustomCheckbox({ checked, onChange, onClick, title }) {
  return (
    <span
      role="checkbox"
      aria-checked={checked}
      tabIndex={0}
      title={title}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === ' ' || e.key === 'Enter') onClick(e); }}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 14, height: 14,
        border: `1.5px solid ${checked ? colors.cyan : colors.border}`,
        borderRadius: 3,
        background: checked ? colors.cyan : 'transparent',
        cursor: 'pointer',
        flexShrink: 0,
        transition: 'background 0.12s, border-color 0.12s',
        color: checked ? colors.bg : 'transparent',
        fontSize: 9, fontWeight: 700, lineHeight: 1,
        userSelect: 'none',
      }}
    >
      {checked ? '\u2713' : ''}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState({ onLoadFiles, onLoadFolder }) {
  const { t } = useTranslation('crystaldatabase');
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: 10, padding: '24px 16px',
      color: colors.textSecondary,
    }}>
      <span style={{ fontSize: 32, opacity: 0.15, userSelect: 'none' }}>
        &#x1F48E;
      </span>
      <div style={{ fontWeight: 600, fontSize: 13, color: colors.text, opacity: 0.6 }}>
        {t('fileList.empty.title')}
      </div>
      <div style={{ fontSize: 11, textAlign: 'center', maxWidth: 200, lineHeight: 1.5, opacity: 0.7 }}>
        {t('fileList.empty.subtitle')}
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
        <Button small onClick={onLoadFiles} style={{ fontSize: 11 }} title={t('fileList.loadFilesTooltip')}>{t('fileList.loadFiles')}</Button>
        <Button small onClick={onLoadFolder} style={{ fontSize: 11 }} title={t('fileList.loadFolderTooltip')}>{t('fileList.loadFolder')}</Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CIF row
// ---------------------------------------------------------------------------

function CifRow({ file, index, isSelected, onSelect, onToggleCheck, onDeleteXtal }) {
  const { t } = useTranslation('crystaldatabase');
  const [hovered, setHovered] = useState(false);

  const handleCheckClick = useCallback((e) => {
    e.stopPropagation();
    onToggleCheck(index);
  }, [index, onToggleCheck]);

  const rowBg =
    isSelected ? alpha(colors.accent, 10) :
    hovered    ? alpha(colors.border, 20) :
    'transparent';

  return (
    <div
      onClick={() => onSelect(index)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={t('fileList.rowTooltip')}
      style={{
        display: 'flex', alignItems: 'center', gap: 7,
        minHeight: 36, padding: '7px 10px',
        cursor: 'pointer',
        background: rowBg,
        borderLeft: isSelected ? `3px solid ${colors.accent}` : '3px solid transparent',
        transition: 'background 0.1s, border-color 0.1s',
        boxSizing: 'border-box',
      }}
    >
      <CustomCheckbox
        checked={!!file.checked}
        onClick={handleCheckClick}
        title={t('fileList.checkTooltip')}
      />
      <span style={{
        flex: 1,
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
        transition: 'color 0.1s',
        gap: 1,
      }}
        title={file.path || file.name}
      >
        <span style={{
          fontSize: 12,
          color: isSelected ? colors.accent : colors.text,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {file.name}
        </span>
        {file.parsedInfo?.phase_name && (
          <span style={{
            fontSize: 10, color: colors.textSecondary,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {file.parsedInfo.phase_name}{file.parsedInfo.space_group ? ` · ${file.parsedInfo.space_group}` : ''}
          </span>
        )}
      </span>
      <XtalBadge
        exists={!!file.hasXtal}
        hasWarning={!!file.parsedInfo?.parse_warning}
        fitForXtal={file.parsedInfo?.fit_for_xtal}
        fitWarnings={file.parsedInfo?.fit_warnings}
        onDeleteXtal={() => onDeleteXtal(index)}
      />
      <LocationDot location={file.location} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CifFileList({
  cifFiles = [],
  selectedIdx,
  onSelect,
  onToggleCheck,
  onCheckAll,
  onDeleteXtal,
  onLoadFiles,
  onLoadFolder,
  filter = '',
  onFilterChange,
}) {
  const { t } = useTranslation('crystaldatabase');
  const checkedCount = cifFiles.filter((f) => f.checked).length;
  const allChecked = cifFiles.length > 0 && checkedCount === cifFiles.length;
  const someChecked = checkedCount > 0 && !allChecked;

  const filteredFiles = filter.trim()
    ? cifFiles
        .map((f, idx) => ({ f, idx }))
        .filter(({ f }) => f.name.toLowerCase().includes(filter.trim().toLowerCase()))
    : cifFiles.map((f, idx) => ({ f, idx }));

  const handleSelectAll = useCallback((e) => {
    e.stopPropagation();
    onCheckAll(!allChecked);
  }, [allChecked, onCheckAll]);

  const titleText = t('fileList.title', { count: cifFiles.length });

  return (
    <GroupBox
      title={titleText}
      style={{ display: 'flex', flexDirection: 'column', height: '100%', marginBottom: 0, padding: 10 }}
    >
      {/* Search row */}
      <div style={{ position: 'relative', marginBottom: 6, flexShrink: 0 }}>
        <Input
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          placeholder={t('fileList.filterPlaceholder')}
          title={t('fileList.filterTooltip')}
          style={{ height: 26, fontSize: 11, paddingRight: filter ? 26 : 8, paddingLeft: 8 }}
        />
        {filter && (
          <button
            onClick={() => onFilterChange('')}
            title={t('fileList.clearFilter')}
            style={{
              position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
              background: 'none', border: 'none', cursor: 'pointer',
              color: colors.textSecondary, fontSize: 13, lineHeight: 1, padding: 0,
            }}
          >
            &times;
          </button>
        )}
      </div>

      {/* Select all header */}
      {cifFiles.length > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          padding: '4px 10px',
          borderBottom: `1px solid ${colors.border}`,
          flexShrink: 0,
        }}>
          <span
            role="checkbox"
            aria-checked={allChecked ? 'true' : someChecked ? 'mixed' : 'false'}
            tabIndex={0}
            title={t('fileList.selectAllTooltip')}
            onClick={handleSelectAll}
            onKeyDown={(e) => { if (e.key === ' ' || e.key === 'Enter') handleSelectAll(e); }}
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: 14, height: 14,
              border: `1.5px solid ${(allChecked || someChecked) ? colors.cyan : colors.border}`,
              borderRadius: 3,
              background: allChecked ? colors.cyan : someChecked ? alpha(colors.cyan, 40) : 'transparent',
              cursor: 'pointer', flexShrink: 0,
              transition: 'background 0.12s, border-color 0.12s',
              color: allChecked ? colors.bg : colors.cyan,
              fontSize: 9, fontWeight: 700, lineHeight: 1, userSelect: 'none',
            }}
          >
            {allChecked ? '\u2713' : someChecked ? '\u2212' : ''}
          </span>
          <span style={{ fontSize: 11, color: colors.textSecondary, flex: 1 }}>
            {t('fileList.selectAll')}
          </span>
          {checkedCount > 0 && (
            <span style={{ fontSize: 10, color: colors.cyan }}>
              {t('fileList.selectedCount', { count: checkedCount })}
            </span>
          )}
        </div>
      )}

      {/* File list */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        minHeight: 0,
        scrollbarWidth: 'thin',
        scrollbarColor: `${colors.border} transparent`,
      }}>
        {cifFiles.length === 0 ? (
          <EmptyState onLoadFiles={onLoadFiles} onLoadFolder={onLoadFolder} />
        ) : filteredFiles.length === 0 ? (
          <div style={{
            padding: 16, textAlign: 'center', fontSize: 11,
            color: colors.textSecondary,
          }}>
            {t('fileList.noMatch', { filter })}
          </div>
        ) : (
          filteredFiles.map(({ f, idx }) => (
            <CifRow
              key={idx}
              file={f}
              index={idx}
              isSelected={selectedIdx === idx}
              onSelect={onSelect}
              onToggleCheck={onToggleCheck}
              onDeleteXtal={onDeleteXtal}
            />
          ))
        )}
      </div>

      {/* Load buttons at bottom — only shown when files exist */}
      {cifFiles.length > 0 && (
        <div style={{
          display: 'flex', gap: 6,
          paddingTop: 8, marginTop: 4,
          borderTop: `1px solid ${colors.border}`,
          flexShrink: 0,
        }}>
          <Button onClick={onLoadFiles} style={{ flex: 1, fontSize: 11 }} small title={t('fileList.loadFilesTooltip')}>
            {t('fileList.loadFiles')}
          </Button>
          <Button onClick={onLoadFolder} style={{ flex: 1, fontSize: 11 }} small title={t('fileList.loadFolderTooltip')}>
            {t('fileList.loadFolder')}
          </Button>
        </div>
      )}
    </GroupBox>
  );
}
