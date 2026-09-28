import { describe, it, expect } from 'vitest';
import {
  PHASE_DRAG, dragData, isPhaseDrag, dropAction, dropHint, dropWouldChange,
  GROUP_DRAG, groupDragData, isGroupDrag, nestAction,
} from './dragPayload';
import { canNest } from './groups';

const groups = () => [
  { id: 'a', name: 'A', members: ['Al', 'Si'] },
  { id: 'b', name: 'B', members: ['Al'] },
  { id: 'c', name: 'C', members: [] },
];

describe('what a drag carries', () => {
  it('an unselected row drags itself, not the selection', () => {
    // With nine ticked, taking hold of a tenth and dropping it must not
    // file the other nine.
    expect(dragData('Zn', ['Al', 'Si']).keys).toEqual(['Zn']);
  });

  it('a row inside the selection drags the whole selection', () => {
    expect(dragData('Al', ['Al', 'Si']).keys).toEqual(['Al', 'Si']);
  });

  it('keeps the row that was grabbed, as well as the set', () => {
    expect(dragData('Si', ['Al', 'Si']).key).toBe('Si');
  });

  it('copes with no selection at all', () => {
    expect(dragData('Al', null).keys).toEqual(['Al']);
    expect(dragData('Al', []).keys).toEqual(['Al']);
  });

  it('is typed, so other drags on the page are ignored', () => {
    expect(dragData('Al', []).type).toBe(PHASE_DRAG);
    expect(isPhaseDrag(dragData('Al', []))).toBe(true);
    expect(isPhaseDrag({ type: 'something-else', keys: ['Al'] })).toBe(false);
    expect(isPhaseDrag({ type: PHASE_DRAG, keys: [] })).toBe(false);
    expect(isPhaseDrag(null)).toBe(false);
    expect(isPhaseDrag({ type: PHASE_DRAG })).toBe(false);
  });
});

describe('what a drop means', () => {
  const data = dragData('Al', []);

  it('a plain drop adds', () => {
    expect(dropAction(data, { altKey: false })).toEqual({ verb: 'add', keys: ['Al'] });
  });

  it('Alt moves', () => {
    expect(dropAction(data, { altKey: true })).toEqual({ verb: 'move', keys: ['Al'] });
  });

  it('Ctrl and Shift are left alone -- Windows already uses them', () => {
    expect(dropAction(data, { ctrlKey: true }).verb).toBe('add');
    expect(dropAction(data, { shiftKey: true }).verb).toBe('add');
    expect(dropAction(data, { metaKey: true }).verb).toBe('add');
  });

  it('means nothing for a drag that is not ours', () => {
    expect(dropAction({ type: 'file' }, { altKey: true })).toBe(null);
  });

  it('survives a drop with no input at all', () => {
    expect(dropAction(data, null).verb).toBe('add');
  });
});

describe('the sentence under the cursor', () => {
  it('names the verb and the number of phases', () => {
    expect(dropHint(dragData('Al', ['Al', 'Si']), { altKey: false }))
      .toEqual({ key: 'drag.willAdd', count: 2 });
    expect(dropHint(dragData('Al', ['Al', 'Si']), { altKey: true }))
      .toEqual({ key: 'drag.willMove', count: 2 });
  });

  it('is a key and a count, not a sentence -- the panel is in four languages', () => {
    const hint = dropHint(dragData('Al', []), {});
    expect(typeof hint.key).toBe('string');
    expect(hint.key).not.toMatch(/\s/);
  });

  it('is nothing for a drag that is not ours', () => {
    expect(dropHint({ type: 'file' }, {})).toBe(null);
  });
});

describe('whether the drop would change anything', () => {
  it('adding a phase the group already holds changes nothing', () => {
    expect(dropWouldChange(dragData('Al', []), {}, groups(), 'a')).toBe(false);
  });

  it('adding a phase the group does not hold changes something', () => {
    expect(dropWouldChange(dragData('Zn', []), {}, groups(), 'a')).toBe(true);
  });

  it('a selection that is only partly there still changes something', () => {
    expect(dropWouldChange(dragData('Al', ['Al', 'Zn']), {}, groups(), 'a')).toBe(true);
  });

  it('MOVING a phase it already holds still changes something when it is filed elsewhere', () => {
    // `Al` is in a and in b. Alt-dropping it on a takes it out of b, which
    // is a real change even though a already has it.
    expect(dropWouldChange(dragData('Al', []), { altKey: true }, groups(), 'a'))
      .toBe(true);
  });

  it('moving a phase into the only group it is in changes nothing', () => {
    expect(dropWouldChange(dragData('Si', []), { altKey: true }, groups(), 'a'))
      .toBe(false);
  });

  it('a target that is gone changes nothing', () => {
    expect(dropWouldChange(dragData('Al', []), {}, groups(), 'nope')).toBe(false);
  });

  it('a drag that is not ours changes nothing', () => {
    expect(dropWouldChange({ type: 'file' }, {}, groups(), 'a')).toBe(false);
  });
});

describe('dragging a group onto a group', () => {
  const nested = () => [
    { id: 'a', name: 'A', parent: null, members: [] },
    { id: 'b', name: 'B', parent: 'a', members: [] },
    { id: 'c', name: 'C', parent: null, members: [] },
    { id: 'd', name: 'D', parent: null, members: [] },
  ];

  it('carries its own type, so a phase drop and a nest never mix', () => {
    expect(groupDragData('a').type).toBe(GROUP_DRAG);
    expect(isGroupDrag(groupDragData('a'))).toBe(true);
    expect(isGroupDrag(dragData('Al', []))).toBe(false);
    expect(isPhaseDrag(groupDragData('a'))).toBe(false);
  });

  it('is not a group drag without a group', () => {
    expect(isGroupDrag({ type: GROUP_DRAG })).toBe(false);
    expect(isGroupDrag({ type: GROUP_DRAG, groupId: '' })).toBe(false);
    expect(isGroupDrag(null)).toBe(false);
  });

  it('a root onto another root nests it', () => {
    expect(nestAction(groupDragData('c'), nested(), 'd', canNest))
      .toEqual({ verb: 'nest', groupId: 'c', parentId: 'd' });
  });

  it('onto the parent it already has means nothing', () => {
    // A target that lights up for a no-op teaches people to distrust it.
    expect(nestAction(groupDragData('b'), nested(), 'a', canNest)).toBe(null);
  });

  it('onto itself, onto a child, or when it is itself a parent: nothing', () => {
    expect(nestAction(groupDragData('c'), nested(), 'c', canNest)).toBe(null);
    expect(nestAction(groupDragData('c'), nested(), 'b', canNest)).toBe(null);
    expect(nestAction(groupDragData('a'), nested(), 'c', canNest)).toBe(null);
  });

  it('a group that is not there means nothing', () => {
    expect(nestAction(groupDragData('zz'), nested(), 'a', canNest)).toBe(null);
  });

  it('a phase drag is not a nest', () => {
    expect(nestAction(dragData('Al', []), nested(), 'a', canNest)).toBe(null);
  });
});
