/**
 * CollectionPicker — the toolbar control for the active phase collection.
 *
 * A dropdown, not a dialog: it sits next to "EDS Colors" in the global
 * toolbar (App.jsx) and lets the user narrow every phase picker in the app
 * (Phase Test, Indexing, the database browser) to one named collection
 * instead of the whole library, without leaving the page they are on.
 *
 * Self-contained: it reads `useCollectionStore` directly and owns no server
 * state of its own. Selecting a collection, or clearing the hidden list,
 * both go through the store's `setActive` / `setHidden` — which write to the
 * server FIRST and only then reload — so nothing here ever stands in for
 * that round trip with a local flag.
 *
 * The dropdown's open/closed state lives here, not in App.jsx: App's own
 * Escape handling has one dependency array that every branch must be listed
 * in, and a second branch reading its own local state has no need to touch
 * that array at all. See `frontend/src/App.jsx`'s `handleKeyDown`.
 */
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import useCollectionStore from '../../stores/useCollectionStore';
import { colors } from '../../theme/tokens';

const STAR = String.fromCharCode(0x2605); // filled star, U+2605

function memberCount(c) {
  return c.effective_member_count ?? c.member_count ?? 0;
}

export default function CollectionPicker({ onManage }) {
  const { t } = useTranslation('collections');
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  const data = useCollectionStore((s) => s.data);
  const load = useCollectionStore((s) => s.load);
  const setActive = useCollectionStore((s) => s.setActive);
  const setHidden = useCollectionStore((s) => s.setHidden);

  // Fetch once on mount. A failed fetch is handled inside the store (it
  // keeps whatever was already on screen); this component never needs to
  // know it happened — the button renders either way.
  useEffect(() => {
    load();
  }, [load]);

  // Close on outside click and on Escape. Local state, local listeners.
  useEffect(() => {
    if (!open) return undefined;
    const onDocClick = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const collections = data?.collections || [];
  const unassigned = data?.unassigned || [];
  const state = data?.state || {};
  const activeName = state.active || null;
  const hiddenList = state.hidden || [];

  const byName = {};
  for (const c of collections) byName[c.name] = c;

  // I3: `state.active` can name a collection that `load_all()` no longer
  // returns — its file went missing or unreadable (broken JSON, unknown
  // `schema`) between saves — while `_local.json` still calls it active.
  // Every picker's `activeKeySet(collections, activeName)` treats an
  // unresolved name as "no filter" (see collectionFilter.js's own
  // docstring: "renamed or deleted: filter nothing"), so every phase picker
  // in the app silently widens to the WHOLE library at exactly the moment
  // the toolbar still reads "Collection: <name>". This is the dangerous
  // half of the fix: without it, nothing on screen says the widening
  // happened.
  const activeMissing = !!(activeName && !byName[activeName]);

  // A collection is hidden if the server marked it so, or if its parent
  // (transitively) is hidden — the brief is explicit that a hidden
  // collection's children must also vanish from the list, not just the
  // collection whose name is literally in state.hidden.
  const isHidden = (c) => {
    let cur = c;
    const seen = new Set();
    while (cur && !seen.has(cur.name)) {
      if (cur.hidden) return true;
      seen.add(cur.name);
      cur = cur.parent ? byName[cur.parent] : null;
    }
    return false;
  };

  const visibleTopLevel = collections.filter((c) => !c.parent && !isHidden(c));
  const childrenOf = (name) => collections.filter((c) => c.parent === name && !isHidden(c));

  // exclusive: false marks the one working-set collection (members may also
  // belong elsewhere); show it apart from the partition, with a star.
  const workingSet = visibleTopLevel.filter((c) => c.exclusive === false);
  const partitioned = visibleTopLevel.filter((c) => c.exclusive !== false);

  const allKeys = new Set();
  for (const c of collections) for (const m of c.members || []) allKeys.add(m.key);
  for (const m of unassigned) allKeys.add(m.key);
  const totalCount = allKeys.size;

  const handleSelect = (name) => {
    setOpen(false);
    setActive(name);
  };

  const handleShowHidden = () => {
    setHidden([]);
  };

  const rowStyle = (selected, indent = 0) => ({
    padding: `6px 12px 6px ${12 + indent}px`,
    fontSize: '9pt',
    color: selected ? colors.bg : colors.text,
    background: selected ? colors.purple : 'transparent',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    borderRadius: 2,
  });

  const hoverOn = (selected) => (e) => {
    if (!selected) e.currentTarget.style.background = `${colors.purple}22`;
  };
  const hoverOff = (selected) => (e) => {
    if (!selected) e.currentTarget.style.background = 'transparent';
  };

  const renderRow = (c, indent = 0, prefix = '') => {
    const selected = activeName === c.name;
    return (
      <div key={c.name}>
        <div
          style={rowStyle(selected, indent)}
          onClick={() => handleSelect(c.name)}
          onMouseEnter={hoverOn(selected)}
          onMouseLeave={hoverOff(selected)}
        >
          {prefix}{c.name} ({memberCount(c)})
        </div>
        {childrenOf(c.name).map((child) => renderRow(child, indent + 16))}
      </div>
    );
  };

  const allSelected = !activeName;

  return (
    <div ref={rootRef} style={{ position: 'relative' }}>
      <button
        data-testid="collection-picker-button"
        onClick={() => setOpen((prev) => !prev)}
        style={{
          background: activeMissing ? colors.orange : (open || activeName ? colors.purple : 'transparent'),
          color: activeMissing ? colors.bg : (open || activeName ? colors.bg : colors.purple),
          fontWeight: 'bold',
          padding: '4px 12px',
          border: `1px solid ${activeMissing ? colors.orange : colors.border}`,
          borderRadius: 3,
          fontSize: '9pt',
          cursor: 'pointer',
          transition: 'background 0.12s, color 0.12s, border-color 0.12s',
        }}
        title={activeMissing ? t('picker.activeMissing', { name: activeName }) : t('picker.label')}
        onMouseEnter={(e) => {
          if (!open && !activeName) {
            e.currentTarget.style.background = `${colors.purple}22`;
            e.currentTarget.style.borderColor = colors.purple;
          }
        }}
        onMouseLeave={(e) => {
          if (!open && !activeName) {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.borderColor = colors.border;
          }
        }}
      >
        {activeMissing && '⚠ '}{t('picker.label')}{activeName ? `: ${activeName}` : ''}
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: 4,
            minWidth: 240,
            maxHeight: 360,
            overflowY: 'auto',
            background: colors.bgTertiary,
            border: `1px solid ${colors.border}`,
            borderRadius: 4,
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            zIndex: 200,
            padding: 4,
          }}
        >
          {/* I3: the active collection's name resolved to nothing. Clicking
              it switches to "All phases" — the same state every filter is
              ALREADY silently in, just now written down and shown. */}
          {activeMissing && (
            <div
              data-testid="collection-active-missing-notice"
              onClick={() => handleSelect(null)}
              style={{
                padding: '6px 12px',
                fontSize: '8.5pt',
                color: colors.orange,
                cursor: 'pointer',
                borderBottom: `1px solid ${colors.border}`,
                marginBottom: 4,
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = `${colors.orange}22`; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
            >
              {t('picker.activeMissing', { name: activeName })}
            </div>
          )}

          {/* Standing notice — stays as long as anything is hidden, this is
              not a toast that fades. Clicking it clears the hidden list. */}
          {hiddenList.length > 0 && (
            <div
              data-testid="collection-hidden-notice"
              onClick={handleShowHidden}
              style={{
                padding: '6px 12px',
                fontSize: '8.5pt',
                color: colors.yellow,
                cursor: 'pointer',
                borderBottom: `1px solid ${colors.border}`,
                marginBottom: 4,
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = `${colors.yellow}22`; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
            >
              {t('picker.hiddenNotice', { count: hiddenList.length })}
            </div>
          )}

          <div
            style={rowStyle(allSelected)}
            onClick={() => handleSelect(null)}
            onMouseEnter={hoverOn(allSelected)}
            onMouseLeave={hoverOff(allSelected)}
          >
            {t('picker.allPhases', { count: totalCount })}
          </div>

          {workingSet.length > 0 && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: `1px solid ${colors.border}` }}>
              <div style={{ padding: '2px 12px', fontSize: '7.5pt', color: colors.textSecondary, textTransform: 'uppercase' }}>
                {t('picker.workingSet')}
              </div>
              {workingSet.map((c) => renderRow(c, 0, `${STAR} `))}
            </div>
          )}

          {partitioned.length > 0 && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: `1px solid ${colors.border}` }}>
              {partitioned.map((c) => renderRow(c))}
            </div>
          )}

          {onManage && (
            <div
              style={{
                marginTop: 4,
                paddingTop: 4,
                borderTop: `1px solid ${colors.border}`,
                padding: '6px 12px',
                fontSize: '8.5pt',
                color: colors.textSecondary,
                cursor: 'pointer',
              }}
              onClick={() => { setOpen(false); onManage(); }}
              onMouseEnter={(e) => { e.currentTarget.style.color = colors.text; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = colors.textSecondary; }}
            >
              {t('picker.manage')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
