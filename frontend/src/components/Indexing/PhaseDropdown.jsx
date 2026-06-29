/**
 * PhaseDropdown — Grouped phase selector with checkboxes.
 *
 * Props:
 *   discoveredFiles: array of file objects from backend
 *   groups: array of group names (sorted by backend)
 *   selectedPaths: array of currently selected file paths
 *   onTogglePath: (path) => void — toggle a single path
 *   onSetAll: (paths) => void — set all selected paths at once
 *   method: 'hough' | 'dictionary' | 'spherical'
 *   open: boolean
 *   onClose: () => void
 */

import { useRef, useEffect, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';

function fileTypeTag(fileType, t) {
  switch (fileType) {
    case 'cif':        return t('phaseDropdown.tagCif');
    case 'master':     return t('phaseDropdown.tagMaster');
    case 'dictionary': return t('phaseDropdown.tagDict');
    default:           return fileType ?? '';
  }
}

export default function PhaseDropdown({
  discoveredFiles = [],
  groups = [],
  selectedPaths = [],
  onTogglePath,
  onSetAll,
  method = 'hough',
  open,
  onClose,
}) {
  const { t } = useTranslation('indexing');
  const containerRef = useRef(null);
  const searchRef = useRef(null);
  const [search, setSearch] = useState('');

  // Close on outside click (but not when inside a floating panel)
  useEffect(() => {
    if (!open) return;
    function handleClick(e) {
      // Don't close if click is inside the floating panel (drag events)
      if (e.target.closest?.('[data-floating-panel]')) return;
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        onClose?.();
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open, onClose]);

  // Focus search on open
  useEffect(() => {
    if (open) {
      setSearch('');
      setTimeout(() => searchRef.current?.focus(), 50);
    }
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    function handleKey(e) {
      if (e.key === 'Escape') onClose?.();
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  // For dictionary: hide pure dict files
  const visibleFiles = useMemo(() => {
    let files = method === 'dictionary'
      ? discoveredFiles.filter(f => f.file_type !== 'dictionary')
      : discoveredFiles;
    // Apply search filter
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      files = files.filter(f =>
        (f.formula || '').toLowerCase().includes(q) ||
        (f.phase_name || '').toLowerCase().includes(q) ||
        (f.filename || '').toLowerCase().includes(q) ||
        (f.element_group || '').toLowerCase().includes(q)
      );
    }
    return files;
  }, [discoveredFiles, method, search]);

  // Group files
  const grouped = useMemo(() => {
    const map = {};
    for (const group of groups) {
      const members = visibleFiles.filter(f => f.element_group === group);
      if (members.length > 0) map[group] = members;
    }
    return map;
  }, [visibleFiles, groups]);

  const ungrouped = visibleFiles.filter(f => !groups.includes(f.element_group));
  const selectedCount = selectedPaths.length;

  if (!open) return null;

  function handleSelectAll() {
    const allPaths = visibleFiles.map(f => f.path);
    // Merge with existing selections (don't lose non-visible selections)
    const merged = [...new Set([...selectedPaths, ...allPaths])];
    onSetAll?.(merged);
  }

  function handleSelectNone() {
    // Only deselect visible items (preserve hidden selections from search filter)
    const visibleSet = new Set(visibleFiles.map(f => f.path));
    const remaining = selectedPaths.filter(p => !visibleSet.has(p));
    onSetAll?.(remaining);
  }

  // When rendered inside a floating panel, use static positioning
  const isEmbedded = !containerRef.current?.closest?.('[data-floating-panel]') === false;

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative', left: 0, right: 0, zIndex: 100,
        backgroundColor: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 4, maxHeight: 450, display: 'flex', flexDirection: 'column',
      }}
    >
      {/* Search field */}
      <div style={{ padding: '6px 10px', borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}>
        <input
          ref={searchRef}
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder={t('phaseDropdown.search')}
          title={t('hoverTips.phaseSearch')}
          style={{
            width: '100%', padding: '5px 8px', fontSize: '9pt',
            background: C.bg, color: C.text, border: `1px solid ${C.border}`,
            borderRadius: 3, outline: 'none', boxSizing: 'border-box',
          }}
          onFocus={e => e.target.style.borderColor = C.cyan}
          onBlur={e => e.target.style.borderColor = C.border}
        />
      </div>

      {/* Scrollable list */}
      <div style={{ overflowY: 'auto', flex: 1 }}>
        {Object.entries(grouped).map(([groupName, files]) => (
          <div key={groupName}>
            <div style={{
              padding: '5px 10px 2px', color: C.cyan, fontSize: '8pt',
              fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
              userSelect: 'none',
            }}>
              {groupName}
            </div>
            {files.map(file => (
              <CheckboxEntry
                key={file.path}
                file={file}
                checked={selectedPaths.includes(file.path)}
                onToggle={() => onTogglePath?.(file)}
                t={t}
              />
            ))}
          </div>
        ))}

        {ungrouped.length > 0 && (
          <div>
            <div style={{
              padding: '5px 10px 2px', color: C.cyan, fontSize: '8pt',
              fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
            }}>
              {t('phaseDropdown.more')}
            </div>
            {ungrouped.map(file => (
              <CheckboxEntry
                key={file.path}
                file={file}
                checked={selectedPaths.includes(file.path)}
                onToggle={() => onTogglePath?.(file)}
                t={t}
              />
            ))}
          </div>
        )}

        {visibleFiles.length === 0 && (
          <div style={{ padding: '12px 10px', color: C.textSecondary, fontSize: '10pt', textAlign: 'center' }}>
            {search ? t('phaseDropdown.noMatch') : t('phaseDropdown.noPhases')}
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        padding: '6px 10px', borderTop: `1px solid ${C.border}`, flexShrink: 0,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        backgroundColor: C.bgSecondary,
      }}>
        <span style={{ color: C.green, fontSize: '9pt', fontWeight: 600 }}>
          {t('phaseDropdown.selectedCountFooter', { count: selectedCount })}
        </span>
        <div style={{ display: 'flex', gap: 6 }}>
          <FooterBtn label={t('phaseDropdown.all')} onClick={handleSelectAll} title={t('hoverTips.phaseSelectAll')} />
          <FooterBtn label={t('phaseDropdown.none')} onClick={handleSelectNone} title={t('hoverTips.phaseSelectNone')} />
          <button
            onClick={() => onClose?.()}
            title={t('hoverTips.phaseDropdownDone')}
            style={{
              padding: '3px 12px', fontSize: '9pt', fontWeight: 600,
              background: C.green, color: C.bg, border: 'none',
              borderRadius: 3, cursor: 'pointer',
            }}
          >
            {t('phaseDropdown.done')}
          </button>
        </div>
      </div>

      {/* Manual file picker link */}
      <div style={{
        padding: '5px 10px', borderTop: `1px solid ${C.border}`,
        backgroundColor: C.bgSecondary, flexShrink: 0,
      }}>
        <button
          onClick={() => { onTogglePath?.(null); /* signals manual picker */ }}
          title={t('hoverTips.phaseAddManually')}
          style={{
            background: 'none', border: 'none', color: C.cyan,
            fontSize: '9pt', cursor: 'pointer', padding: 0,
            textDecoration: 'underline',
          }}
        >
          {t('phaseDropdown.addManually')}
        </button>
      </div>
    </div>
  );
}

function CheckboxEntry({ file, checked, onToggle, t }) {
  return (
    <div
      onClick={onToggle}
      title={t('hoverTips.phaseCheckboxRow')}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '4px 10px', cursor: 'pointer',
        transition: 'background-color 0.1s',
      }}
      onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(139,233,253,0.07)'}
      onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onToggle}
        onClick={e => e.stopPropagation()}
        style={{ accentColor: C.green, cursor: 'pointer', flexShrink: 0 }}
      />
      {/* Canonical phase-identity label (formula → structure → α/β tag) — the
          SAME format the Phase-Tester shows. Falls back to raw formula. */}
      <span style={{ color: C.green, fontWeight: 700, fontSize: '10pt', flexShrink: 1,
                     overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={file.display_label || file.formula}>
        {file.display_label || file.formula || '—'}
      </span>
      {file.crystal_system && (
        <span style={{ color: C.textSecondary, fontSize: '9pt', flexShrink: 0 }}>{file.crystal_system}</span>
      )}
      <span style={{ flex: 1 }} />
      <span style={{
        color: C.textSecondary, fontSize: '8pt',
        backgroundColor: 'rgba(255,255,255,0.05)',
        border: `1px solid ${C.border}`, borderRadius: 3,
        padding: '1px 5px', maxWidth: 140, overflow: 'hidden',
        textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0,
      }} title={file.filename}>
        {fileTypeTag(file.file_type, t)} · {file.filename}
      </span>
    </div>
  );
}

function FooterBtn({ label, onClick, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        padding: '3px 8px', fontSize: '9pt',
        background: 'none', color: C.textSecondary,
        border: `1px solid ${C.border}`, borderRadius: 3,
        cursor: 'pointer',
      }}
      onMouseEnter={e => e.currentTarget.style.color = C.text}
      onMouseLeave={e => e.currentTarget.style.color = C.textSecondary}
    >
      {label}
    </button>
  );
}
