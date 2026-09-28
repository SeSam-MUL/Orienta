/**
 * What a drag carries, and what a drop means.
 *
 * WHY THIS IS A SEPARATE FILE OF PURE FUNCTIONS. The spec's own measurement
 * (§6): "synthetische Drag-Events erreichen den Handler nicht; keiner der
 * vier hat je wirklich gezogen." A jsdom test cannot drive a real drag, so
 * anything decided INSIDE a drag handler is decided where no test can reach
 * it. Everything that is a judgement lives here and is tested directly; the
 * handler is left with nothing to do but call these and pass the answer on.
 * The wiring itself is checked by hand in the running app, with a real
 * pointer, because that is the only place it can be checked.
 */

/** The `type` every phase drag carries, so other drags are ignored. */
export const PHASE_DRAG = 'phase-library/phase';

/**
 * The payload for a row being dragged.
 *
 * DRAGGING AN UNSELECTED ROW DRAGS THAT ROW, not the selection. Anything
 * else surprises: with nine phases ticked, taking hold of a tenth and
 * dropping it must not file the other nine. Dragging a row that IS in the
 * selection takes the whole selection, which is the reason to have made one.
 */
export function dragData(key, selection) {
  const sel = Array.isArray(selection) ? selection : [];
  const keys = sel.includes(key) ? sel : [key];
  return { type: PHASE_DRAG, key, keys };
}

/** Whether a drop target should light up for this drag. */
export function isPhaseDrag(data) {
  return Boolean(data) && data.type === PHASE_DRAG
    && Array.isArray(data.keys) && data.keys.length > 0;
}

/**
 * What the drop means, given the data and the keys held down at the moment
 * of RELEASE.
 *
 * The modifier is read on drop and not on pick-up, so somebody can change
 * their mind halfway across the screen -- which is how every file manager
 * behaves and therefore what the hand expects.
 *
 * ALT, not Ctrl and not Shift: under Windows those two already mean copy and
 * move while dragging, and giving them a third meaning here would make this
 * one window disagree with the desktop around it.
 */
export function dropAction(data, input) {
  if (!isPhaseDrag(data)) return null;
  const alt = Boolean(input && input.altKey);
  return { verb: alt ? 'move' : 'add', keys: data.keys };
}

/**
 * The sentence under the cursor while a drag is in the air.
 *
 * Returned as a key plus a count so the caller translates it; the panel is
 * in four languages and a hard-coded "Add 3 phases" would be in one.
 */
export function dropHint(data, input) {
  const action = dropAction(data, input);
  if (!action) return null;
  return { key: action.verb === 'move' ? 'drag.willMove' : 'drag.willAdd',
    count: action.keys.length };
}

/**
 * Whether this drop changes anything, so the target can say "already here"
 * instead of lighting up as though it will do something.
 *
 * `add` onto a group that holds every dragged phase does nothing. `move` in
 * the same situation still does something IF the phases are in other groups
 * too -- it takes them out of those -- and nothing if they are not, which is
 * why the group list has to be consulted and not just the target.
 */
export function dropWouldChange(data, input, groups, targetId) {
  const action = dropAction(data, input);
  if (!action) return false;
  const target = groups.find((g) => g.id === targetId);
  if (!target) return false;
  const missing = action.keys.some((k) => !target.members.includes(k));
  if (action.verb === 'add') return missing;
  const elsewhere = groups.some(
    (g) => g.id !== targetId && action.keys.some((k) => g.members.includes(k)));
  return missing || elsewhere;
}

// ==========================================================================
// Dragging a GROUP onto a GROUP: nesting.
// ==========================================================================

/** The `type` a dragged group carries. Distinct, so a drop can tell them apart. */
export const GROUP_DRAG = 'phase-library/group';

/**
 * A group being dragged.
 *
 * Its own type rather than a flag on the phase payload: a drop target has to
 * decide between "file these phases here" and "make this group a child of
 * that one", and those are different questions with different answers. One
 * payload with a discriminator inside it would put that decision in every
 * handler instead of in the type.
 */
export function groupDragData(id) {
  return { type: GROUP_DRAG, groupId: id };
}

export function isGroupDrag(data) {
  return Boolean(data) && data.type === GROUP_DRAG
    && typeof data.groupId === 'string' && data.groupId.length > 0;
}

/**
 * What a group dropped on a group means, if anything.
 *
 * Returns null when it means nothing -- onto itself, onto its own child,
 * onto a group that is already a child (one level only), or when the
 * dragged group is already a parent. `canNest` owns those rules; this only
 * adds the one it cannot know: dropping a child onto the parent it already
 * has changes nothing, and a target that lights up for a no-op teaches
 * people to distrust the highlight.
 */
export function nestAction(data, groups, targetId, canNest) {
  if (!isGroupDrag(data)) return null;
  const child = groups.find((g) => g.id === data.groupId);
  if (!child) return null;
  if (child.parent === targetId) return null;
  if (!canNest(groups, data.groupId, targetId)) return null;
  return { verb: 'nest', groupId: data.groupId, parentId: targetId };
}
