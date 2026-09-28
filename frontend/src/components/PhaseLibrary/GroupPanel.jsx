/**
 * The groups, down the left-hand side.
 *
 * Sebastian, on the collections this replaces: "ich will auch was haben mit
 * dem ich per drag and drop phasen herumschieben kann". So a phase is filed
 * by dragging it onto a group -- and by a button, and by a menu item, because
 * a drag is the one gesture a keyboard cannot make and 93 rows is a long way
 * to reach with a mouse (§2.9).
 *
 * THREE WAYS TO FILE THE SAME PHASE, on purpose:
 *   drag a row onto a group          the gesture he asked for
 *   Alt-drag                         file it HERE and nowhere else
 *   "+ N" on the group row           the selection, no drag, keyboard-reachable
 *   the ⋯ menu                       the same, named, with move alongside
 *
 * COUNTS ARE MEASURED AGAINST WHAT IS SHOWN. `groupCounts` does the work;
 * the point is that today's collections leave `Mg-Systeme 3` sitting there
 * while the list below shows nothing, and a counter that contradicts the
 * list is worse than no counter. When a filter is on, the row says "1 of 3".
 *
 * AND IT SAYS WHERE THE GROUPS LIVE. While they are in this browser, that
 * sentence is on screen. Somebody who files 36 phases into six groups and
 * then finds them gone on the measurement PC has lost an afternoon to a line
 * we did not write.
 */
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { draggable, dropTargetForElements }
  from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { colors, spacing } from '../../theme/components';
import { groupCounts, canNest } from './groups';
import {
  dropAction, dropHint, dropWouldChange, isPhaseDrag,
  groupDragData, isGroupDrag, nestAction,
} from './dragPayload';
import RowMenu from './RowMenu';
import SuggestGroups from './SuggestGroups';

const S = {
  panel: { display: 'flex', flexDirection: 'column', gap: 2,
           paddingBottom: spacing.groupMargin },
  head: { display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing },
  title: { fontSize: '10pt', fontWeight: 600, margin: 0 },
  newButton: {
    background: 'transparent', border: 'none', color: colors.purple,
    cursor: 'pointer', font: 'inherit', fontSize: '9pt', padding: 0,
  },
  where: { fontSize: '8pt', color: colors.textSecondary, lineHeight: 1.35 },
  row: {
    display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
    padding: '3px 6px', fontSize: '9pt', borderRadius: 3,
    border: '1px solid transparent',
  },
  rowOver: { border: `1px dashed ${colors.purple}`, background: `${colors.purple}22` },
  rowInert: { border: `1px dashed ${colors.border}`, opacity: 0.7 },
  rowNest: { border: `1px dashed ${colors.green || '#7fd17f'}`,
             background: `${colors.green || '#7fd17f'}18` },
  lifted: { opacity: 0.4 },
  // One level, so one indent. A child is offset and its name is quieter --
  // the eye should read "under" without a line drawing.
  child: { paddingLeft: 18 },
  parentPick: { display: 'flex', flexWrap: 'wrap', gap: 4, padding: '2px 6px' },
  // The group, opened. Indented under its row, one line per phase, each
  // with the one control the whole arrangement was missing.
  members: { display: 'flex', flexDirection: 'column', gap: 1,
             padding: '2px 0 4px 22px' },
  member: { display: 'flex', alignItems: 'baseline', gap: 6, fontSize: '8.5pt',
            color: colors.textSecondary },
  memberName: { flex: 1, minWidth: 0, overflow: 'hidden',
                textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  x: {
    background: 'transparent', border: 'none', color: colors.textSecondary,
    cursor: 'pointer', font: 'inherit', fontSize: '8.5pt', padding: '0 2px',
    lineHeight: 1,
  },
  emptyGroup: { fontSize: '8pt', color: colors.textSecondary,
                padding: '2px 0 4px 22px', lineHeight: 1.35 },
  caret: {
    background: 'transparent', border: 'none', color: colors.text,
    cursor: 'pointer', font: 'inherit', padding: 0, textAlign: 'left',
    flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  active: { fontWeight: 600, color: colors.purple },
  pick: {
    background: 'transparent', border: `1px solid ${colors.border}`,
    color: colors.purple, cursor: 'pointer', font: 'inherit', fontSize: '8pt',
    padding: '1px 6px', borderRadius: 3,
  },
  name: { flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
          whiteSpace: 'nowrap' },
  count: { color: colors.textSecondary, fontVariantNumeric: 'tabular-nums' },
  add: {
    background: 'transparent', border: `1px solid ${colors.border}`,
    color: colors.purple, cursor: 'pointer', font: 'inherit', fontSize: '8pt',
    padding: '0 4px', borderRadius: 3,
  },
  empty: { fontSize: '8pt', color: colors.textSecondary, lineHeight: 1.4 },
  problem: { fontSize: '8pt', color: colors.red || '#ff6b6b', lineHeight: 1.35 },
  input: {
    width: '100%', padding: '3px 6px', fontSize: '9pt',
    background: colors.bgSecondary, color: colors.text,
    border: `1px solid ${colors.purple}`, borderRadius: 3,
  },
};

/**
 * One group row: a drop target, a count, a shortcut and a menu.
 *
 * The drop target is bound in an effect and torn down by the cleanup
 * pragmatic-drag-and-drop returns; `groupsRef` is a ref rather than a
 * dependency so that re-binding does not happen on every membership change
 * -- rebinding mid-drag drops the drag.
 */
function GroupRow({
  group, count, selection, groupsRef, onAdd, onMove, onRename, onDelete, onUse,
  onNest, onRemove, parents, isChild, isActive, labelFor,
}) {
  const { t } = useTranslation('phaselibrary');
  const ref = useRef(null);
  /**
   * OPENING A GROUP TO SEE WHAT IS IN IT.
   *
   * There was no way to. Both first-time readers stopped at the same place:
   * "zz-loop-AlFe 1 phase is the entire feedback. There is no visible way to
   * open the group, list its members, or take one back out" -- and one of
   * them said plainly that this is where a real person gives up, not in
   * frustration but because they stop trusting the arrangement they just
   * built and go back to typing phase names by hand.
   *
   * Closed by default: the panel is a column beside a 93-row list, and
   * eight open groups would push the list off the screen.
   */
  const [open, setOpen] = useState(false);
  const [over, setOver] = useState(null);   // null | 'change' | 'inert' | 'nest'
  const [hint, setHint] = useState(null);   // {key, count} or null
  const [editing, setEditing] = useState(false);
  const [picking, setPicking] = useState(false);
  const [lifted, setLifted] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    // Recomputed on every `onDrag` and not only on enter, because the answer
    // depends on the Alt key and somebody can press it while hovering. A
    // hint that was right when the pointer arrived and wrong when it leaves
    // is worse than none.
    const look = ({ source, location }) => {
      if (isGroupDrag(source.data)) {
        // A group over a group is a nest, and it lights up differently:
        // "put these phases here" and "put this group under that one" are
        // not the same promise.
        setOver(nestAction(source.data, groupsRef.current, group.id, canNest)
          ? 'nest' : 'inert');
        setHint(null);
        return;
      }
      const input = location.current.input;
      setOver(dropWouldChange(source.data, input, groupsRef.current, group.id)
        ? 'change' : 'inert');
      setHint(dropHint(source.data, input));
    };
    const clear = () => { setOver(null); setHint(null); };
    const stopDrop = dropTargetForElements({
      element: el,
      canDrop: ({ source }) => isPhaseDrag(source.data) || isGroupDrag(source.data),
      getData: () => ({ groupId: group.id }),
      onDragEnter: look,
      onDrag: look,
      onDragLeave: clear,
      onDrop: ({ source, location }) => {
        clear();
        const nest = nestAction(source.data, groupsRef.current, group.id, canNest);
        if (nest) { onNest(nest.groupId, nest.parentId); return; }
        // The modifier is read HERE, at release, not at pick-up: somebody
        // may change their mind halfway across the screen, which is what
        // every file manager does and therefore what the hand expects.
        const action = dropAction(source.data, location.current.input);
        if (!action) return;
        // BOTH verbs hand over the whole list. Move used to go one phase
        // at a time, which made Undo take back one twelfth of a drag and
        // sent twelve unsynchronised writes at one file.
        if (action.verb === 'move') onMove(group.id, action.keys);
        else onAdd(group.id, action.keys);
      },
    });
    const stopDrag = draggable({
      element: el,
      getInitialData: () => groupDragData(group.id),
      onDragStart: () => setLifted(true),
      onDrop: () => setLifted(false),
    });
    return () => { stopDrop(); stopDrag(); };
  }, [group.id, groupsRef, onAdd, onMove, onNest]);

  const n = selection.length;
  // Only the selected phases this group actually HOLDS. Offering "remove 5"
  // when the group has two of them names a number that cannot happen, and
  // the user then cannot tell whether the other three failed or were never
  // there.
  const inHere = selection.filter((k) => group.members.includes(k));
  const label = count.total === count.shown
    ? t('groups.count', { count: count.total })
    : t('groups.countFiltered', { shown: count.shown, total: count.total });

  if (editing) {
    return (
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const v = new FormData(e.currentTarget).get('name');
          setEditing(false);
          if (String(v).trim()) onRename(group.id, String(v));
        }}
      >
        <input
          name="name" defaultValue={group.name} style={S.input} autoFocus
          aria-label={t('groups.renameLabel', { name: group.name })}
          onKeyDown={(e) => { if (e.key === 'Escape') setEditing(false); }}
          // Clicking away KEEPS what was typed. It used to discard it in
          // silence: Enter and Escape both behaved, and the mouse -- which
          // is what most people reach for -- threw the new name away with
          // no sign that anything had happened. Escape is still the way to
          // abandon it, and it is the one that says so.
          onBlur={(e) => {
            setEditing(false);
            const v = e.target.value;
            if (v.trim() && v !== group.name) onRename(group.id, v);
          }}
        />
      </form>
    );
  }

  if (picking) {
    // The way to nest without dragging. A list of buttons rather than a
    // `<select>`, for the reason the ⋯ menu is not one either: these are
    // actions, and a select visits each option with the arrow keys.
    return (
      <div style={S.parentPick} data-testid={`parent-pick-${group.id}`}>
        <span style={S.count}>{t('groups.nestUnder', { name: group.name })}</span>
        {parents.map((p) => (
          <button key={p.id} type="button" style={S.pick}
                  onClick={() => { setPicking(false); onNest(group.id, p.id); }}>
            {p.name}
          </button>
        ))}
        {group.parent && (
          <button type="button" style={S.pick}
                  onClick={() => { setPicking(false); onNest(group.id, null); }}>
            {t('groups.toTopLevel')}
          </button>
        )}
        <button type="button" style={S.pick} onClick={() => setPicking(false)}>
          {t('name.cancel')}
        </button>
      </div>
    );
  }

  const row = (
    <div
      ref={ref}
      style={{
        ...S.row,
        ...(isChild ? S.child : null),
        ...(over === 'change' ? S.rowOver : null),
        ...(over === 'inert' ? S.rowInert : null),
        ...(over === 'nest' ? S.rowNest : null),
        ...(lifted ? S.lifted : null),
      }}
      data-group-id={group.id}
      data-group-child={isChild ? 'yes' : undefined}
      data-drop-state={over || 'idle'}
    >
      <button
        type="button"
        style={isActive ? { ...S.caret, ...S.active } : S.caret}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={group.name}
        data-testid={`group-open-${group.id}`}
      >
        <span aria-hidden="true">{open ? '▾ ' : '▸ '}</span>
        {group.name}
        {/* Which group a run will use is a fact about THIS list, and it was
            readable only from a chip in a different part of the screen. */}
        {isActive && <span style={S.count}> · {t('groups.activeHere')}</span>}
      </button>
      {/* While a drag is over the row the count gives way to what the drop
          will do. Two numbers in the same place -- "12 phases" and "+3" --
          read as one confusing number. */}
      <span style={S.count} data-testid={`group-count-${group.id}`}>
        {hint ? t(hint.key, { count: hint.count }) : label}
      </span>
      {n > 0 && (
        // The keyboard path, and the mouse path for anybody who does not
        // want to drag across a 93-row list. Its label says what it will do
        // and to how many, because "+" alone is a guess.
        <button
          type="button" style={S.add}
          onClick={() => onAdd(group.id, selection)}
          aria-label={t('groups.addSelected', { count: n, name: group.name })}
        >
          {t('groups.addShort', { count: n })}
        </button>
      )}
      <RowMenu
        label={t('groups.menuLabel', { name: group.name })}
        items={[
          { key: 'add',
            label: n ? t('groups.menuAdd', { count: n }) : t('groups.menuAddNone'),
            disabled: !n,
            onSelect: () => onAdd(group.id, selection) },
          { key: 'move',
            label: n ? t('groups.menuMoveAlt', { count: n }) : t('groups.menuMoveNone'),
            disabled: !n,
            onSelect: () => onMove(group.id, selection) },
          // TAKING A PHASE OUT HAD NO BUTTON AT ALL. `useGroups.drop` existed
          // and was tested, and nothing called it -- because under the old
          // folder model, filing a phase elsewhere WAS the removal, and that
          // is exactly the gesture the tag model redefined. A returning user
          // put it plainly: "my working loop, 'this phase does not belong
          // here, out', has no button", and the only removal on the screen
          // deleted the whole group.
          { key: 'remove',
            label: inHere.length
              ? t('groups.menuRemove', { count: inHere.length })
              : t('groups.menuRemoveNone'),
            disabled: !inHere.length,
            // A LIST, always. This looped `onRemove` per phase, so one
            // gesture became N changes and N requests; `onRemove` now
            // takes keys so no caller can go back to looping.
            onSelect: () => onRemove(group.id, inHere) },
          { key: 'use',
            label: group.members.length
              ? t('groups.menuUse') : t('groups.menuUseEmpty'),
            disabled: !group.members.length,
            onSelect: () => onUse(group) },
          { key: 'nest',
            label: parents.length
              ? t('groups.menuNest') : t('groups.menuNestNone'),
            // Disabled and saying why, rather than gone: one level only, so
            // a group that already has children can never become a child.
            disabled: !parents.length && !group.parent,
            onSelect: () => setPicking(true) },
          { key: 'rename', label: t('groups.menuRename'),
            onSelect: () => setEditing(true) },
          { key: 'delete', label: t('groups.menuDelete'), danger: true,
            onSelect: () => onDelete(group.id) },
        ]}
      />
    </div>
  );

  return (
    <>
      {row}
      {open && (group.members.length ? (
        <div style={S.members} data-testid={`group-members-${group.id}`}>
          {group.members.map((key) => (
            <div key={key} style={S.member} data-member-key={key}>
              <span style={S.memberName} title={key}>{labelFor(key)}</span>
              <button
                type="button" style={S.x}
                onClick={() => onRemove(group.id, [key])}
                aria-label={t('groups.removeOne',
                  { phase: labelFor(key), name: group.name })}
                title={t('groups.removeOne',
                  { phase: labelFor(key), name: group.name })}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p style={S.emptyGroup}>{t('groups.emptyOpened')}</p>
      ))}
    </>
  );
}

/**
 * Roots in order, each followed by its own children.
 *
 * The list arrives flat and nesting is one level, so this is one pass, not
 * a tree walk. A child whose parent is gone is shown at the top level
 * rather than dropped -- an invisible group is worse than a misplaced one,
 * and `deleteGroup` takes children with it precisely so this stays rare.
 */
export function inReadingOrder(groups) {
  const byId = new Map(groups.map((g) => [g.id, g]));
  const out = [];
  for (const g of groups) {
    if (g.parent && byId.has(g.parent)) continue;
    out.push({ group: g, isChild: false });
    for (const c of groups) {
      if (c.parent === g.id) out.push({ group: c, isChild: true });
    }
  }
  return out;
}

export default function GroupPanel({
  // `transitional` defaults to FALSE: the shipped app keeps groups in
  // the library folder, and the sentence this guards says they live in
  // the browser and do not travel. Defaulting to true meant a caller
  // that forgot the prop printed a claim that is false wherever it
  // ships. A default should be the state the product is actually in.
  groups, visibleKeys, selection, transitional = false,
  unavailable = false, canUndo = false,
  onCreate, onAdd, onMove, onRename, onDelete, onUndo, onUse, onNest,
  onSuggested, onRemove, onStopUsing, activeGroupName = null,
  labelFor = (k) => k, unsaved = [], onReread, libraryTotal = 0,
}) {
  const { t } = useTranslation('phaselibrary');
  const [naming, setNaming] = useState(false);
  const counts = groupCounts(groups, visibleKeys);
  const byId = Object.fromEntries(counts.map((c) => [c.id, c]));

  // A ref, so a drop target bound once does not need re-binding every time a
  // membership changes -- re-binding mid-drag drops the drag.
  const groupsRef = useRef(groups);
  groupsRef.current = groups;

  return (
    <div style={S.panel} data-testid="group-panel">
      <div style={S.head}>
        <h3 style={S.title}>{t('groups.title')}</h3>
        <button type="button" style={S.newButton} onClick={() => setNaming(true)}>
          {t('groups.new')}
        </button>
        {canUndo && (
          <button type="button" style={S.newButton} onClick={onUndo}>
            {t('groups.undo')}
          </button>
        )}
        {/* THE WAY BACK TO THE WHOLE LIBRARY. There was none: the only
            caller of `setActive` always passed a group, the dropdown that
            used to offer "all phases" is gone, and the blocked-Start
            message told people to "clear it in the Phase Library" -- an
            instruction that could not be followed. With one group in the
            library and a stale reference, every picker was empty and every
            run blocked, with no action available at all. */}
        {activeGroupName && (
          <button type="button" style={S.newButton} onClick={onStopUsing}
                  data-testid="stop-using-group"
                  title={t('groups.stopUsingTip')}>
            {t('groups.stopUsing')}
          </button>
        )}
      </div>

      {naming && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const v = String(new FormData(e.currentTarget).get('name') || '');
            setNaming(false);
            // An empty name is refused rather than defaulted: a group called
            // "Untitled" is a group nobody finds again.
            if (v.trim()) onCreate(v);
          }}
        >
          <input
            name="name" style={S.input} autoFocus
            placeholder={t('groups.newPlaceholder')}
            aria-label={t('groups.newLabel')}
            onKeyDown={(e) => { if (e.key === 'Escape') setNaming(false); }}
          />
        </form>
      )}

      {groups.length === 0 && !naming && (
        <p style={S.empty}>{t('groups.none')}</p>
      )}

      {inReadingOrder(groups).map(({ group: g, isChild }) => (
        <GroupRow
          key={g.id}
          group={g}
          count={byId[g.id] || { shown: 0, total: 0 }}
          selection={selection}
          groupsRef={groupsRef}
          isChild={isChild}
          // Who this group could go under, worked out with the SAME rule the
          // drop target uses, so the menu and the drag can never disagree
          // about what is allowed.
          parents={groups.filter((p) => canNest(groups, g.id, p.id))}
          onAdd={onAdd}
          onMove={onMove}
          onRename={onRename}
          onDelete={onDelete}
          onUse={onUse}
          onNest={onNest}
          onRemove={onRemove}
          labelFor={labelFor}
          isActive={activeGroupName != null && g.name === activeGroupName}
        />
      ))}

      {/* An arrangement proposed from the library itself -- the only thing
          here that does not need somebody to have an opinion first. */}
      <SuggestGroups onApplied={onSuggested} libraryTotal={libraryTotal}
                     labelFor={labelFor} />

      {/* Where they live. Not a footnote: it is the difference between
          "my groups are gone" and "my groups are on the other machine". */}
      {transitional && <p style={S.where}>{t('groups.thisMachine')}</p>}
      {unavailable && <p style={S.problem}>{t('groups.storageUnavailable')}</p>}
      {/* EVERY change that did not get written, not just the last one. One
          slot was cleared by the next successful change, so a failure
          followed by a success left the unsaved change on screen with
          nothing saying so -- and there was no way to re-read short of
          restarting. */}
      {unsaved.length > 0 && (
        <div style={S.problem} data-testid="unsaved-changes">
          <p style={{ margin: 0 }}>
            {t('groups.notSavedCount', { count: unsaved.length })}
          </p>
          {/* The last three, AND a line saying how many are not shown.
              The count above says "7 changes are on screen but not in the
              library:" and then listed three, so four of them were
              unnameable -- the very state the list was added to end. */}
          {unsaved.slice(-3).map((u, i) => (
            <p key={`${u.verb}-${u.groupName}-${i}`} style={{ margin: 0 }}>
              {t('groups.notSavedOne', { name: u.groupName, reason: u.reason })}
            </p>
          ))}
          {unsaved.length > 3 && (
            <p style={{ margin: 0 }} data-testid="unsaved-overflow">
              {t('groups.notSavedMore', { count: unsaved.length - 3 })}
            </p>
          )}
          <button type="button" style={S.newButton} onClick={onReread}>
            {t('groups.reread')}
          </button>
        </div>
      )}
    </div>
  );
}
