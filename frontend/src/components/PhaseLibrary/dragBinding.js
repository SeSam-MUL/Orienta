/**
 * Making a phase row draggable.
 *
 * One line of library, on its own, for two reasons. It is the only place in
 * the feature that turns a row into something you can pick up, so when a
 * drag misbehaves there is one file to read; and `dragPayload.js` next door
 * is pure by charter -- it is tested directly precisely BECAUSE the handlers
 * cannot be, and importing a DOM library into it would spoil that.
 *
 * `getInitialData` is called at pick-up, so the payload reflects the
 * selection as it was when the drag started. That is the right moment for
 * WHAT is being dragged. The modifier key that decides what the drop MEANS
 * is read later, at release, in the drop target -- see `dropAction`.
 */
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { PHASE_DRAG } from './dragPayload';

/**
 * @param element the row
 * @param getPayload () => ({key, keys}) -- normally `dragData(key, selection)`
 * @param onDragChange (dragging: boolean) => void, for the row's own styling
 * @returns the cleanup function the caller must call on unmount
 */
export function bindPhaseDraggable(element, getPayload, onDragChange) {
  return draggable({
    element,
    getInitialData: () => ({ ...getPayload(), type: PHASE_DRAG }),
    onDragStart: () => onDragChange && onDragChange(true),
    onDrop: () => onDragChange && onDragChange(false),
  });
}
