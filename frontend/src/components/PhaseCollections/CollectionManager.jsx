/**
 * CollectionManager — the dialog where collections are actually MADE, not
 * just used. Opened from the toolbar picker's "manage" entry (App.jsx) and
 * from the database browser's own "Manage collections…" button
 * (DatabaseBrowser/DatabasePage.jsx) — the same component both times, reading
 * and writing the same `useCollectionStore`, so an edit made from either
 * place is visible in both the moment this dialog's own `refresh()` lands.
 *
 * Three things live ONLY here (Tasks 1-9 built everything that consumes a
 * collection; nothing before this one could create, name, or grow one):
 *
 *  - The working set: one `exclusive: false` collection, created once via
 *    the star button. `exclusive` is a plain model field
 *    (`phase_collections.py#PhaseCollection`) with no special status of its
 *    own — "not deletable" is a UI decision made here, not a server rule, so
 *    the delete button is simply withheld for any collection with
 *    `exclusive === false`, which is exactly how the toolbar picker already
 *    tells the working set apart from a partition (`CollectionPicker.jsx`,
 *    `workingSet = visibleTopLevel.filter(c => c.exclusive === false)`).
 *  - The suggestion run: `POST /suggest` proposes, nothing is written until
 *    the accepted names go to `POST /suggest/apply` — a name that already
 *    exists comes back under `skipped` and is shown, never dropped.
 *  - Missing-member rendering: a member the library has lost
 *    (`present: false`, `phase_collections.py#annotate`) prints struck
 *    through HERE — this is the only place a missing MEMBER itself, and
 *    which key it is, is ever shown. The `collections:counts.missingFromLibrary`
 *    header count built for this same purpose is also used by the database
 *    browser's own group headers (`DatabaseBrowser/DatabasePage.jsx`), which
 *    needed the number without the member list; every other consumer (the
 *    picker, the Phase Tester, the EDS panel) asks a different question
 *    entirely — "is this key usable HERE" — with its own, already-shipped
 *    answer (`counts.usableHere` / `counts.notUsableHere`).
 */
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import useCollectionStore from '../../stores/useCollectionStore';
import { collectionsApi } from '../../services/api';
import {
  colors, alpha, spacing, Button, Input, Select,
  useConfirm, ConfirmDialog, usePrompt, PromptDialog,
} from '../../theme/components';

// The working set's persisted NAME — deliberately not `t('picker.workingSet')`.
// A collection's `name` is data written to `Database/Collections/*.json`, not
// UI text; naming it from the current UI language would mean a language
// switch stops the just-created collection from matching its own picker
// heading, and a second user on a different locale would get a second,
// differently-named working set instead of finding the first one. Stable,
// English, and renamable afterward like any other collection.
const WORKING_SET_NAME = 'Working set';

function errorText(e) {
  return e?.response?.data?.detail || e?.message || String(e);
}

/**
 * I3: `GET /api/phase-collections/` has always returned `data.problems`
 * (`phase_collections.py#load_all` for `bad_schema`/`orphan_parent`/
 * `duplicate_key`, `propose_repairs` for `missing_phase`) — nothing in the
 * frontend ever read the field. A collection file with unreadable JSON or an
 * unknown `schema` silently drops out of the list (spec §10.5/§10.6: "abgelehnt,
 * benannt" — rejected AND NAMED, not rejected and forgotten); a key filed in
 * two exclusive collections at once (§10.3) or a `parent` pointing at a
 * deleted collection (§10.4) likewise had nowhere on screen to say so. This
 * turns each `problems` entry into the sentence the spec requires, instead
 * of the raw dict a naive fallback would otherwise render as
 * "[object Object]".
 */
function describeProblem(p, t) {
  switch (p.kind) {
    case 'bad_schema':
      return t('manager.problemBadSchema', { detail: p.detail });
    case 'orphan_parent':
      return t('manager.problemOrphanParent', { name: p.detail, parent: p.parent });
    case 'duplicate_key':
      return t('manager.problemDuplicateKey', {
        key: p.key, collections: (p.collections || []).join(', '),
      });
    case 'missing_phase':
      return t('manager.problemMissingPhase', {
        key: p.missing_key, collection: p.collection,
        candidates: (p.candidates || []).join(', '),
      });
    default:
      return p.detail || p.kind || JSON.stringify(p);
  }
}

/** One collection's member list: reorder, remove, missing-struck-through. */
function MemberList({ t, collection, otherCollections, busy, onReorder, onRemove, onMove }) {
  const members = collection.members || [];
  if (members.length === 0) {
    return (
      <div style={{ padding: '4px 8px', fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('manager.emptyCollection')}
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {members.map((m, i) => (
        <div
          key={m.key}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '3px 8px', fontSize: '8.5pt',
            borderBottom: `1px solid ${alpha(colors.border, 15)}`,
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            <button
              type="button" disabled={busy || i === 0}
              onClick={() => onReorder(collection, i, -1)}
              title={t('manager.moveUp')}
              style={{ ...arrowStyle, opacity: i === 0 ? 0.3 : 1 }}
            >{'▴'}</button>
            <button
              type="button" disabled={busy || i === members.length - 1}
              onClick={() => onReorder(collection, i, 1)}
              title={t('manager.moveDown')}
              style={{ ...arrowStyle, opacity: i === members.length - 1 ? 0.3 : 1 }}
            >{'▾'}</button>
          </div>
          <span
            title={m.formula || m.key}
            style={{
              flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              color: m.present ? colors.text : colors.textSecondary,
              textDecoration: m.present ? 'none' : 'line-through',
            }}
          >
            {m.label || m.key}
          </span>
          {!m.present && (
            <span style={{ fontSize: '7.5pt', color: colors.orange, flexShrink: 0 }}>
              {t('manager.notInLibrary')}
            </span>
          )}
          {otherCollections.length > 0 && (
            <select
              disabled={busy}
              value=""
              onChange={(e) => { if (e.target.value) onMove(collection, m.key, e.target.value); }}
              title={t('manager.moveToTooltip')}
              style={selectStyle}
            >
              <option value="">{t('manager.moveTo')}</option>
              {otherCollections.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          )}
          <button
            type="button" disabled={busy}
            onClick={() => onRemove(collection, m.key)}
            title={t('manager.removeMemberTooltip')}
            style={removeStyle}
          >{'×'}</button>
        </div>
      ))}
    </div>
  );
}

const arrowStyle = {
  background: 'transparent', border: 'none', color: colors.textSecondary,
  cursor: 'pointer', fontSize: '8pt', lineHeight: 1, padding: '0 2px',
};
const removeStyle = {
  background: 'transparent', border: 'none', color: colors.textSecondary,
  cursor: 'pointer', fontSize: '11pt', lineHeight: 1, padding: '0 4px', flexShrink: 0,
};
const selectStyle = {
  background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 3,
  color: colors.textSecondary, fontSize: '8pt', padding: '1px 4px', flexShrink: 0,
};

/** One collection card: header (name, missing count, expand, rename, delete)
 *  plus its member list when expanded. Renders itself, then recurses one
 *  level for its own children — collections nest one level only
 *  (`phase_collections.py#save`), so there is nothing deeper to recurse into. */
function CollectionCard({
  t, collection, depth, children, allCollections, busy,
  expanded, onToggleExpand, onRename, onDelete, onReorder, onRemoveMember, onMoveMember,
}) {
  const isOpen = expanded.has(collection.name);
  const missing = (collection.members || []).filter((m) => !m.present).length;
  const isWorkingSet = collection.exclusive === false;
  const otherCollections = allCollections.filter((c) => c.name !== collection.name);

  return (
    <div style={{ marginLeft: depth * 18, marginBottom: 2 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '5px 8px', borderRadius: 4,
        background: alpha(colors.purple, isWorkingSet ? 10 : 4),
        border: `1px solid ${alpha(colors.purple, isWorkingSet ? 25 : 14)}`,
      }}>
        <button
          type="button" onClick={() => onToggleExpand(collection.name)}
          title={isOpen ? t('manager.collapse') : t('manager.expand')}
          style={{ background: 'transparent', border: 'none', color: colors.textSecondary,
                   cursor: 'pointer', fontSize: '8pt', padding: '0 2px', flexShrink: 0 }}
        >{isOpen ? '▾' : '▸'}</button>

        {isWorkingSet && (
          <span title={t('picker.workingSet')} style={{ color: colors.yellow, flexShrink: 0 }}>
            {'★'}
          </span>
        )}

        <span style={{ fontWeight: 600, fontSize: '9pt', color: colors.text, flex: 1, minWidth: 0,
                       overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {collection.name}
        </span>

        <span style={{ fontSize: '8pt', color: colors.textSecondary, flexShrink: 0 }}>
          ({(collection.members || []).length})
        </span>

        {missing > 0 && (
          <span style={{ fontSize: '8pt', color: colors.orange, flexShrink: 0 }}>
            {t('collections:counts.missingFromLibrary', { count: missing })}
          </span>
        )}

        <Button small variant="ghost" disabled={busy} onClick={() => onRename(collection)}
                title={t('manager.renameTooltip')}>
          {t('manager.rename')}
        </Button>
        {!isWorkingSet && (
          <Button small variant="ghost" disabled={busy} onClick={() => onDelete(collection)}
                  title={t('manager.deleteTooltip')}>
            {t('manager.delete')}
          </Button>
        )}
      </div>

      {isOpen && (
        <div style={{ marginLeft: 18, marginTop: 2, border: `1px solid ${alpha(colors.border, 20)}`,
                       borderRadius: 4 }}>
          <MemberList
            t={t} collection={collection} otherCollections={otherCollections} busy={busy}
            onReorder={onReorder} onRemove={onRemoveMember} onMove={onMoveMember}
          />
        </div>
      )}

      {children}
    </div>
  );
}

export default function CollectionManager({ onClose }) {
  const { t } = useTranslation('collections');
  const data = useCollectionStore((s) => s.data);
  const load = useCollectionStore((s) => s.load);

  const collections = data?.collections || [];
  const unassigned = data?.unassigned || [];
  const problems = data?.problems || [];

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [newName, setNewName] = useState('');
  const [newParent, setNewParent] = useState('');
  const [expanded, setExpanded] = useState(() => new Set());
  const [addTarget, setAddTarget] = useState({}); // unassigned key -> chosen collection name

  const [suggestions, setSuggestions] = useState(null);   // null = not run
  const [suggestChecked, setSuggestChecked] = useState({});
  const [suggestSkipped, setSuggestSkipped] = useState(null);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestCreatedCount, setSuggestCreatedCount] = useState(null);

  const [prompt, promptProps] = usePrompt();
  const [askConfirm, confirmProps] = useConfirm();

  const topLevel = collections.filter((c) => !c.parent);
  // One level only (`phase_collections.py#save`: a `parent` whose OWN
  // `parent` is set is rejected, and a collection that already HAS children
  // cannot itself become a child) — but a parent MAY have any number of
  // SIBLING children; the spec's own worked example (§4.2) files both
  // "Fe-haltig" and "Mg-haltig" under "Intermetallics in Al". `topLevel`
  // above already excludes anything that is itself a child (its `parent` is
  // set), which is the only exclusion the one-level rule requires here.
  //
  // I4: this used to ALSO drop any top-level collection that already had a
  // child (`&& !collections.some((x) => x.parent === c.name)`), which
  // allowed at most ONE sub-collection ever, per parent — with one child
  // present, no parent was offered at all and the second sibling could not
  // be created.
  const parentOptions = topLevel.filter((c) => c.exclusive !== false);
  const childrenOf = (name) => collections.filter((c) => c.parent === name);
  const hasWorkingSet = collections.some((c) => c.exclusive === false);

  const refresh = useCallback(() => load(), [load]);

  // Self-sufficient on open: whichever caller opens this dialog (the
  // toolbar picker's "manage" entry, or the database browser's own button)
  // may not itself have loaded the store yet — CollectionPicker only loads
  // on ITS OWN mount, and a fresh page visit could reach the "Manage
  // collections…" button before that effect has run a second time.
  useEffect(() => { load(); }, [load]);

  const run = useCallback(async (fn) => {
    setBusy(true); setErr(null);
    try { await fn(); await refresh(); }
    catch (e) { setErr(errorText(e)); }
    finally { setBusy(false); }
  }, [refresh]);

  const toggleExpand = useCallback((name) => {
    setExpanded((s) => {
      const next = new Set(s);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  }, []);

  const handleCreate = (e) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    run(async () => {
      await collectionsApi.create({ name, parent: newParent || null });
      setNewName(''); setNewParent('');
    });
  };

  const handleCreateWorkingSet = () => {
    run(() => collectionsApi.create({ name: WORKING_SET_NAME, exclusive: false }));
  };

  const handleRename = (c) => {
    prompt({
      title: t('manager.renameTitle', { name: c.name }),
      defaultValue: c.name,
      submitLabel: t('manager.rename'),
      onSubmit: (value) => {
        const next = String(value).trim();
        if (!next || next === c.name) return;
        run(() => collectionsApi.rename(c.name, next));
      },
    });
  };

  const handleDelete = (c) => {
    askConfirm({
      title: t('manager.deleteTitle'),
      message: t('manager.deleteMessage', { name: c.name }),
      confirmLabel: t('manager.delete'),
      variant: 'danger',
      onConfirm: () => run(() => collectionsApi.remove(c.name)),
    });
  };

  const handleReorder = (c, index, dir) => {
    const keys = (c.members || []).map((m) => m.key);
    const j = index + dir;
    if (j < 0 || j >= keys.length) return;
    [keys[index], keys[j]] = [keys[j], keys[index]];
    run(() => collectionsApi.update({ name: c.name, member_keys: keys }));
  };

  const handleRemoveMember = (c, key) => {
    run(() => collectionsApi.removeMembers(c.name, [key]));
  };

  const handleMoveMember = (fromCollection, key, toName) => {
    if (!toName || toName === fromCollection.name) return;
    // `addMembers` alone is correct here — it POSTs to `assign()`, which
    // already strips the key from every OTHER exclusive collection when (and
    // only when) the TARGET is exclusive, and leaves everything else
    // untouched when the target is the non-exclusive working set. An
    // explicit `removeMembers(fromCollection.name, ...)` here used to run
    // unconditionally, un-filing the phase from its exclusive home even when
    // the move was only a star into the working set — overriding `assign()`
    // and contradicting the user guide's promise that starring never does
    // that. The database browser's own "move to collection" control
    // (`DatabasePage.jsx`) already relies on `addMembers` alone; this
    // matches it.
    run(() => collectionsApi.addMembers(toName, [key]));
  };

  const handleAddUnassigned = (key) => {
    const target = addTarget[key];
    if (!target) return;
    run(async () => {
      await collectionsApi.addMembers(target, [key]);
      setAddTarget((s) => { const next = { ...s }; delete next[key]; return next; });
    });
  };

  const handleSuggest = async () => {
    setSuggesting(true); setErr(null); setSuggestSkipped(null); setSuggestCreatedCount(null);
    try {
      const res = await collectionsApi.suggest();
      const list = res.data?.suggestions || [];
      setSuggestions(list);
      const checked = {};
      for (const s of list) checked[s.name] = true;
      setSuggestChecked(checked);
    } catch (e) { setErr(errorText(e)); }
    finally { setSuggesting(false); }
  };

  const handleApplySuggestions = () => {
    const accepted = (suggestions || []).filter((s) => suggestChecked[s.name]).map((s) => s.name);
    if (accepted.length === 0) { setSuggestions(null); return; }
    run(async () => {
      const res = await collectionsApi.applySuggest(accepted);
      setSuggestSkipped(res.data?.skipped || []);
      setSuggestCreatedCount((res.data?.created || []).length);
      setSuggestions(null);
    });
  };

  const overlay = {
    position: 'fixed', inset: 0, background: alpha(colors.bg, 80),
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1300,
  };
  const modal = {
    background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8,
    boxShadow: `0 8px 32px ${alpha(colors.border, 40)}`,
    width: 560, maxWidth: '94vw', maxHeight: '86vh',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
  };

  return (
    <>
      <div style={overlay} onClick={onClose}>
        <div style={modal} role="dialog" aria-label={t('manager.title')} onClick={(e) => e.stopPropagation()}>
          <div style={{
            padding: spacing.innerMargin, borderBottom: `1px solid ${colors.border}`,
            display: 'flex', alignItems: 'center', flexShrink: 0,
          }}>
            <div style={{ fontSize: '11pt', fontWeight: 700, color: colors.accent, flex: 1 }}>
              {t('manager.title')}
            </div>
            <button
              type="button" onClick={onClose} title={t('manager.close')}
              style={{ background: 'transparent', border: 'none', color: colors.textSecondary,
                       cursor: 'pointer', fontSize: '14pt', lineHeight: 1, padding: '0 4px' }}
            >{'×'}</button>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: spacing.innerMargin }}>
            {err && (
              <div role="alert" style={{
                padding: '6px 10px', marginBottom: spacing.outerSpacing,
                background: alpha(colors.red, 8), border: `1px solid ${colors.red}`,
                borderRadius: 4, fontSize: '9pt', color: colors.red,
              }}>
                {err}
              </div>
            )}

            {/* I3: structural problems the server found while loading the
                filing (bad JSON, unknown schema, a key filed twice, an
                orphaned parent, a renamed file with a repair candidate) —
                distinct from `err` above, which is this session's own last
                failed ACTION. These can be true on a clean load with nothing
                clicked yet. */}
            {problems.length > 0 && (
              <div role="alert" style={{
                padding: '6px 10px', marginBottom: spacing.outerSpacing,
                background: alpha(colors.orange, 8), border: `1px solid ${colors.orange}`,
                borderRadius: 4, fontSize: '9pt', color: colors.orange,
                display: 'flex', flexDirection: 'column', gap: 4,
              }}>
                {problems.map((p, i) => (
                  <div key={`${p.kind}-${i}`}>{describeProblem(p, t)}</div>
                ))}
              </div>
            )}

            {/* --- Create, and the working set --- */}
            <form onSubmit={handleCreate} style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
              <Input
                value={newName} onChange={(e) => setNewName(e.target.value)}
                placeholder={t('manager.namePlaceholder')}
                style={{ flex: 1, minWidth: 160 }}
              />
              <Select
                value={newParent} onChange={(e) => setNewParent(e.target.value)}
                options={[{ value: '', label: t('manager.parentNone') },
                          ...parentOptions.map((c) => ({ value: c.name, label: c.name }))]}
                title={t('manager.parentLabel')}
                style={{ width: 160 }}
              />
              <Button type="submit" variant="primary" small disabled={busy || !newName.trim()}>
                {t('manager.create')}
              </Button>
              {!hasWorkingSet && (
                <Button type="button" small disabled={busy} onClick={handleCreateWorkingSet}
                        title={t('manager.createWorkingSetTooltip')}>
                  {t('manager.createWorkingSet')}
                </Button>
              )}
            </form>

            {/* --- Suggestions --- */}
            <div style={{ marginBottom: 10 }}>
              <Button small disabled={suggesting || busy} onClick={handleSuggest}
                      title={t('manager.suggestTooltip')}>
                {suggesting ? t('manager.suggesting') : t('manager.suggest')}
              </Button>
              {suggestCreatedCount !== null && (
                <span style={{ marginLeft: 8, fontSize: '8.5pt', color: colors.green }}>
                  {t('manager.suggestCreated', { count: suggestCreatedCount })}
                </span>
              )}
              {suggestSkipped && suggestSkipped.length > 0 && (
                <div role="status" style={{ marginTop: 4, fontSize: '8.5pt', color: colors.orange }}>
                  {t('manager.suggestSkipped', { count: suggestSkipped.length, names: suggestSkipped.join(', ') })}
                </div>
              )}
              {suggestions !== null && (
                <div style={{ marginTop: 6, border: `1px solid ${colors.border}`, borderRadius: 4, padding: 6 }}>
                  {suggestions.length === 0 ? (
                    <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
                      {t('manager.suggestEmpty')}
                    </div>
                  ) : (
                    <>
                      {suggestions.map((s) => (
                        <label key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 6,
                                                       fontSize: '8.5pt', padding: '2px 0', cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={!!suggestChecked[s.name]}
                            onChange={() => setSuggestChecked((prev) => ({ ...prev, [s.name]: !prev[s.name] }))}
                          />
                          <span style={{ flex: 1 }}>{s.name}</span>
                          <span style={{ color: colors.textSecondary }}>
                            {t('manager.suggestCount', { count: s.keys.length })}
                          </span>
                        </label>
                      ))}
                      <div style={{ display: 'flex', gap: 6, marginTop: 6, justifyContent: 'flex-end' }}>
                        <Button small onClick={() => setSuggestions(null)}>{t('manager.suggestDiscard')}</Button>
                        <Button small variant="primary" onClick={handleApplySuggestions} disabled={busy}>
                          {t('manager.suggestApply')}
                        </Button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* --- The collections themselves --- */}
            {topLevel.length === 0 ? (
              <div style={{ fontSize: '9pt', color: colors.textSecondary, padding: '8px 0' }}>
                {t('manager.noCollectionsYet')}
              </div>
            ) : (
              topLevel.map((c) => (
                <CollectionCard
                  key={c.name} t={t} collection={c} depth={0}
                  allCollections={collections} busy={busy}
                  expanded={expanded} onToggleExpand={toggleExpand}
                  onRename={handleRename} onDelete={handleDelete}
                  onReorder={handleReorder} onRemoveMember={handleRemoveMember}
                  onMoveMember={handleMoveMember}
                >
                  {childrenOf(c.name).map((child) => (
                    <CollectionCard
                      key={child.name} t={t} collection={child} depth={1}
                      allCollections={collections} busy={busy}
                      expanded={expanded} onToggleExpand={toggleExpand}
                      onRename={handleRename} onDelete={handleDelete}
                      onReorder={handleReorder} onRemoveMember={handleRemoveMember}
                      onMoveMember={handleMoveMember}
                    />
                  ))}
                </CollectionCard>
              ))
            )}

            {/* --- Unassigned --- */}
            {unassigned.length > 0 && (
              <div style={{ marginTop: 14 }}>
                <div style={{ fontSize: '8pt', color: colors.textSecondary, textTransform: 'uppercase',
                              marginBottom: 4 }}>
                  {t('manager.unassignedTitle', { count: unassigned.length })}
                </div>
                {unassigned.map((m) => (
                  <div key={m.key} style={{ display: 'flex', alignItems: 'center', gap: 6,
                                             padding: '2px 4px', fontSize: '8.5pt' }}>
                    <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
                                   whiteSpace: 'nowrap' }} title={m.formula || m.key}>
                      {m.label || m.key}
                    </span>
                    {topLevel.length > 0 && (
                      <>
                        <select
                          value={addTarget[m.key] || ''}
                          onChange={(e) => setAddTarget((s) => ({ ...s, [m.key]: e.target.value }))}
                          disabled={busy}
                          style={selectStyle}
                        >
                          <option value="">{t('manager.addTo')}</option>
                          {topLevel.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                        </select>
                        <Button small disabled={busy || !addTarget[m.key]}
                                onClick={() => handleAddUnassigned(m.key)}>
                          {t('manager.add')}
                        </Button>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      <PromptDialog {...promptProps} />
      <ConfirmDialog {...confirmProps} />
    </>
  );
}
