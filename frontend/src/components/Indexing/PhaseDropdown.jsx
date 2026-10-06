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
 *   collectionKeys: Set|null — keys of the active collection (its stems, for
 *     Hough). `null` = no collection active, offer everything.
 *   allowedPaths: Set|null — the paths the server resolved for THIS method
 *     (Spherical/Dictionary; see the filter comment on `visibleFiles`).
 *   onShowAll: () => void — escapes the filter for this picker only.
 *   overriddenCollectionName: string|null — set while `onShowAll` has been
 *     used (`collectionKeys`/`allowedPaths` are both null as a RESULT of
 *     that, not because no collection is active) and names which one is
 *     being ignored here. The escape row above disappears once overridden —
 *     it is a filter escape, not a status line — so without this the
 *     toolbar could still read "Collection: Matrix" while this picker
 *     silently offered the whole library with nothing saying so.
 *   onReapplyCollection: () => void — undoes onShowAll for this picker.
 *   onAddPath: (path: string) => Promise<{ok, name?, already?, error?}> — when
 *     given, the footer offers a path field (and, in Electron, a Browse button)
 *     for a phase file outside the library, instead of the "+ Add file
 *     manually…" link. `error` is `{code, message, params}`. Files the caller
 *     adds this way carry `user_added: true` and are never hidden by a group
 *     filter: the user just put them there.
 */

import { useRef, useEffect, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import { keyForPath } from '../PhaseCollections/collectionFilter';
import { cleanPastedPath } from './phasePath';
import { groupLabel } from './phaseGroups';

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
  // 'idle' | 'loading' | 'loaded' | 'error'. The default keeps every other
  // caller and test on the old behaviour: say what the list says.
  loadState = 'loaded',
  onRetry = null,
  groups = [],
  selectedPaths = [],
  onTogglePath,
  onSetAll,
  method = 'hough',
  open,
  onClose,
  collectionKeys = null,
  allowedPaths = null,
  onShowAll,
  overriddenCollectionName = null,
  onReapplyCollection,
  onAddPath = null,
}) {
  const { t } = useTranslation(['indexing', 'collections']);
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

  // Collection filter runs FIRST, before the dictionary-type filter and the
  // search, so `grouped`/`ungrouped`/`handleSelectAll`/`handleSelectNone`
  // (all derived from `visibleFiles`) get it for free.
  //
  // Two different filters, because the filenames differ. Hough files ARE
  // named by the stem (the CIF itself), so the stem is the key. Spherical
  // and Dictionary files are not: an .sht is
  // `Formula (CIF_stem) [Pearson] {kV}.sht`, and a master is matched by a
  // fuzzy filename search that lives on the backend — so those two match on
  // the PATHS the server resolved (`allowedPaths`), never on the stem.
  // Filtering them by stem would offer nothing at all (see
  // indexingCollection.test.jsx's REGRESSION case).
  const visibleFiles = useMemo(() => {
    let files = discoveredFiles;
    if (collectionKeys) {
      files = method === 'hough'
        ? files.filter(f => f.user_added || collectionKeys.has(keyForPath(f.path)))
        : files.filter(f => f.user_added || allowedPaths?.has(f.path));
    }
    // For dictionary: hide pure dict files
    if (method === 'dictionary') files = files.filter(f => f.file_type !== 'dictionary');
    // Apply search filter
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      files = files.filter(f =>
        (f.formula || '').toLowerCase().includes(q) ||
        (f.phase_name || '').toLowerCase().includes(q) ||
        (f.filename || '').toLowerCase().includes(q) ||
        (f.element_group || '').toLowerCase().includes(q) ||
        groupLabel(f.element_group, t).toLowerCase().includes(q)
      );
    }
    return files;
  }, [discoveredFiles, collectionKeys, allowedPaths, method, search, t]);

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
              {groupLabel(groupName, t)}
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

        {/* A failed refresh keeps the old list on screen -- by design. But
            then the empty-state block below never renders, so the failure was
            invisible and Start went grey with no reason given. This banner is
            the one place that says it while there is still something to see.
            It must sit OUTSIDE the empty-state block: an earlier version was
            nested inside it and therefore only ever rendered when there was
            nothing to warn about. */}
        {loadState === 'error' && visibleFiles.length > 0 && (
          <div style={{
            padding: '6px 10px', fontSize: '9pt', color: C.yellow,
            borderBottom: `1px solid ${C.border}`,
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span>{t('phaseDropdown.staleWarning')}</span>
            {onRetry && (
              <button onClick={onRetry} style={{
                marginLeft: 'auto', padding: '2px 8px', fontSize: '8.5pt',
                background: C.border, border: 'none', borderRadius: 3,
                color: C.text, cursor: 'pointer',
              }}>{t('phaseDropdown.retry')}</button>
            )}
          </div>
        )}

        {/* "No phases found" is a claim about the library, so it is only
            made when the library actually answered. While the request is out
            or has failed, say that instead -- the user's next move differs
            completely (wait / retry vs. put files in the library). */}
        {visibleFiles.length === 0 && (
          <div style={{ padding: '12px 10px', color: C.textSecondary, fontSize: '10pt', textAlign: 'center' }}>
        {/* State first, search second: "No matches" is also a claim
                about data we do not have yet. */}
            {loadState === 'loading' || loadState === 'idle'
              ? t('phaseDropdown.loading')
              : loadState === 'error'
                ? (
                  <>
                    <div style={{ color: C.yellow }}>{t('phaseDropdown.unreachable')}</div>
                    {onRetry && (
                      <button
                        onClick={onRetry}
                        style={{
                          marginTop: 8, padding: '4px 12px', fontSize: '9pt',
                          background: C.border, border: 'none', borderRadius: 4,
                          color: C.text, cursor: 'pointer',
                        }}
                      >
                        {t('phaseDropdown.retry')}
                      </button>
                    )}
                  </>
                )
                : search ? t('phaseDropdown.noMatch') : t('phaseDropdown.noPhases')}
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

      {/* Collection filter escape — only shown while a collection actually
          narrows this list. `collectionKeys` is `null` once the parent
          drops the filter (no active collection, or the user already used
          this escape), so the row disappears on its own. */}
      {collectionKeys && onShowAll && (
        <div style={{
          padding: '4px 10px', borderTop: `1px solid ${C.border}`,
          backgroundColor: C.bgSecondary, flexShrink: 0, textAlign: 'right',
        }}>
          <button
            onClick={onShowAll}
            style={{
              background: 'none', border: 'none', color: C.cyan,
              fontSize: '9pt', cursor: 'pointer', padding: 0,
              textDecoration: 'underline',
            }}
          >
            {t('collections:counts.showAll')}
          </button>
        </div>
      )}

      {/* Persistent notice for the OPPOSITE state: the escape above was
          used, so `collectionKeys`/`allowedPaths` are both null and the row
          above is gone — without this, nothing on this picker would say the
          toolbar's named collection is being ignored here. Stays up until
          `onReapplyCollection` is used or the active collection itself
          changes (IndexingPage.jsx resets `overrideAll` on that). */}
      {overriddenCollectionName && (
        <div style={{
          padding: '4px 10px', borderTop: `1px solid ${C.border}`,
          backgroundColor: C.bgSecondary, flexShrink: 0,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8,
        }}>
          <span style={{ fontSize: '9pt', color: C.orange }}>
            {t('collections:counts.overriddenNotice', { name: overriddenCollectionName })}
          </span>
          {onReapplyCollection && (
            <button
              onClick={onReapplyCollection}
              style={{
                background: 'none', border: 'none', color: C.cyan,
                fontSize: '9pt', cursor: 'pointer', padding: 0,
                textDecoration: 'underline', whiteSpace: 'nowrap',
              }}
            >
              {t('collections:counts.reapply')}
            </button>
          )}
        </div>
      )}

      {/* Phase file outside the library */}
      {onAddPath ? (
        <PhasePathRow method={method} onAddPath={onAddPath} t={t} />
      ) : (
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
      )}
    </div>
  );
}

// What a native file dialog should offer, per indexing method.
const BROWSE_FILTERS = {
  hough: [{ name: 'CIF Files', extensions: ['cif'] }],
  dictionary: [{ name: 'Master Pattern', extensions: ['h5', 'hdf5'] }],
  spherical: [{ name: 'SHT Files', extensions: ['sht'] }],
};

const METHOD_KEY = { hough: 'Hough', dictionary: 'Dictionary', spherical: 'Spherical' };

/** The server's `{code, message, params}` in the user's language. */
function describePathError(error, t) {
  const code = error?.code || 'generic';
  // "generic" is a failure the server described in its own words (PC
  // Refinement's loader answers with the reason as plain text): show that, not
  // a fixed sentence that says less.
  if (code === 'generic' && error?.message) return error.message;
  return t(`phaseDropdown.pathError.${code}`, {
    ...(error?.params || {}),
    // A code this build has no wording for still gets the server's sentence.
    defaultValue: error?.message || t('phaseDropdown.pathError.generic'),
  });
}

/**
 * Path field for a phase file that is not in the library.
 *
 * Typing or pasting a path is the only way in a plain browser (start_app.py),
 * where there is no native dialog; Electron adds a Browse button on top.
 * What was typed stays in the field after a refusal so it can be corrected.
 */
function PhasePathRow({ method, onAddPath, t }) {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);       // { error: bool, text }
  const canBrowse = typeof window !== 'undefined' && !!window.electronAPI?.openFile;

  async function submit(value) {
    const path = cleanPastedPath(value ?? text);
    if (!path || busy) return;
    setBusy(true);
    setNote(null);
    try {
      const res = await onAddPath(path);
      if (res?.ok) {
        setText('');
        setNote({
          error: false,
          text: t(res.inLibrary ? 'phaseDropdown.pathInLibrary'
                  : res.already ? 'phaseDropdown.pathAlready' : 'phaseDropdown.pathAdded',
                  { name: res.name }),
        });
      } else {
        setNote({ error: true, text: describePathError(res?.error, t) });
      }
    } finally {
      setBusy(false);
    }
  }

  async function browse() {
    const picked = await window.electronAPI.openFile({ filters: BROWSE_FILTERS[method] || [] });
    if (picked) {
      setText(picked);
      submit(picked);
    }
  }

  return (
    <div style={{
      padding: '6px 10px', borderTop: `1px solid ${C.border}`,
      backgroundColor: C.bgSecondary, flexShrink: 0,
    }}>
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <input
          type="text"
          value={text}
          onChange={e => { setText(e.target.value); setNote(null); }}
          onKeyDown={e => { if (e.key === 'Enter') submit(); }}
          placeholder={t(`phaseDropdown.pathPlaceholder${METHOD_KEY[method] || 'Hough'}`)}
          title={t('hoverTips.phasePathInput')}
          aria-label={t('phaseDropdown.pathLabel')}
          disabled={busy}
          style={{
            flex: 1, minWidth: 0, padding: '4px 6px', fontSize: '9pt',
            background: C.bg, color: C.text, border: `1px solid ${C.border}`,
            borderRadius: 3, outline: 'none', boxSizing: 'border-box',
          }}
          onFocus={e => e.target.style.borderColor = C.cyan}
          onBlur={e => e.target.style.borderColor = C.border}
        />
        <button
          onClick={() => submit()}
          disabled={busy || !text.trim()}
          title={t('hoverTips.phasePathAdd')}
          style={{
            padding: '3px 10px', fontSize: '9pt', fontWeight: 600,
            background: C.cyan, color: C.bg, border: 'none', borderRadius: 3,
            cursor: busy || !text.trim() ? 'default' : 'pointer',
            opacity: busy || !text.trim() ? 0.5 : 1,
          }}
        >
          {busy ? t('phaseDropdown.pathChecking') : t('phaseDropdown.pathAdd')}
        </button>
        {canBrowse && (
          <button
            onClick={browse}
            disabled={busy}
            title={t('hoverTips.phasePathBrowse')}
            style={{
              padding: '3px 8px', fontSize: '9pt', background: 'none',
              color: C.textSecondary, border: `1px solid ${C.border}`,
              borderRadius: 3, cursor: 'pointer',
            }}
          >
            {t('phaseDropdown.pathBrowse')}
          </button>
        )}
      </div>
      {/* Always in the page, so a screen reader announces what appears in it. */}
      <div
        role={note?.error ? 'alert' : 'status'}
        aria-live={note?.error ? 'assertive' : 'polite'}
        data-testid="phase-path-note"
        style={{
          marginTop: note ? 4 : 0, fontSize: '9pt', wordBreak: 'break-word',
          color: note?.error ? C.red : C.green,
        }}
      >
        {note ? note.text : null}
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
