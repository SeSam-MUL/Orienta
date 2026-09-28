// @vitest-environment jsdom
/**
 * The group panel, everything except the drag.
 *
 * WHAT IS NOT HERE, AND WHY IT IS NOT FAKED. The spec measured it (§6):
 * "synthetische Drag-Events erreichen den Handler nicht; keiner der vier hat
 * je wirklich gezogen." A jsdom `dragstart` does not reach
 * pragmatic-drag-and-drop, so a test that dispatched one and asserted a
 * membership would be asserting its own mock. The judgement a drop makes
 * lives in `dragPayload.js` and is tested there, directly; the gesture is
 * checked by hand in the running app with a real pointer. Anything in
 * between would be a green test over an untested feature, which is worse
 * than a gap somebody can see.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  render, screen, cleanup, fireEvent, within,
} from '@testing-library/react';
import GroupPanel from './GroupPanel';

afterEach(cleanup);

const groups = () => [
  { id: 'g1', name: 'Al-Fe', parent: null, members: ['Al', 'Al13Fe4'] },
  { id: 'g2', name: 'Mg systems', parent: null, members: [] },
];

function show(over = {}) {
  const props = {
    groups: groups(),
    visibleKeys: ['Al', 'Al13Fe4', 'Si'],
    selection: [],
    onCreate: vi.fn(), onAdd: vi.fn(), onMove: vi.fn(), onRename: vi.fn(),
    onDelete: vi.fn(), onUndo: vi.fn(), onUse: vi.fn(),
    onNest: vi.fn(), onSuggested: vi.fn(), onRemove: vi.fn(),
    onStopUsing: vi.fn(), onReread: vi.fn(),
    ...over,
  };
  render(<GroupPanel {...props} />);
  return props;
}

describe('the counts say what the list says', () => {
  it('a plain count when nothing is filtered out', () => {
    show();
    expect(screen.getByTestId('group-count-g1').textContent).toMatch(/2 phases/);
  });

  it('"1 of 2" when a filter has hidden one of the members', () => {
    // Today `Mg-Systeme 3` sits there while the list shows nothing. A
    // counter that contradicts the list is worse than no counter.
    show({ visibleKeys: ['Al'] });
    expect(screen.getByTestId('group-count-g1').textContent).toMatch(/1 of 2/);
  });

  it('an empty group and a filtered-to-nothing group do not read alike', () => {
    show({ visibleKeys: ['Si'] });
    expect(screen.getByTestId('group-count-g1').textContent).toMatch(/0 of 2/);
    expect(screen.getByTestId('group-count-g2').textContent).toMatch(/0 phases/);
  });
});

describe('making a group', () => {
  it('asks for a name and passes it on', () => {
    const p = show();
    fireEvent.click(screen.getByText('+ New group'));
    const box = screen.getByLabelText('Name for the new group');
    fireEvent.change(box, { target: { value: 'Al-Si' } });
    fireEvent.submit(box.closest('form'));
    expect(p.onCreate).toHaveBeenCalledWith('Al-Si');
  });

  it('refuses a name that is only spaces, quietly, without making one', () => {
    const p = show();
    fireEvent.click(screen.getByText('+ New group'));
    const box = screen.getByLabelText('Name for the new group');
    fireEvent.change(box, { target: { value: '   ' } });
    fireEvent.submit(box.closest('form'));
    expect(p.onCreate).not.toHaveBeenCalled();
  });

  it('Escape gives up without making one', () => {
    const p = show();
    fireEvent.click(screen.getByText('+ New group'));
    fireEvent.keyDown(screen.getByLabelText('Name for the new group'),
      { key: 'Escape' });
    expect(screen.queryByLabelText('Name for the new group')).toBe(null);
    expect(p.onCreate).not.toHaveBeenCalled();
  });
});

describe('filing without dragging', () => {
  // A drag is the one gesture a keyboard cannot make, and 93 rows is a long
  // way to drag with a mouse.
  it('no shortcut button when nothing is selected', () => {
    show();
    expect(screen.queryByLabelText(/Add the .* selected phases/)).toBe(null);
  });

  it('a shortcut per group once phases are ticked, saying what it will do', () => {
    show({ selection: ['Si', 'Zn'] });
    const b = screen.getByLabelText('Add the 2 selected phases to Al-Fe');
    expect(b.textContent).toBe('+2');
  });

  it('the shortcut hands over the whole selection', () => {
    const p = show({ selection: ['Si', 'Zn'] });
    fireEvent.click(screen.getByLabelText('Add the 2 selected phases to Al-Fe'));
    expect(p.onAdd).toHaveBeenCalledWith('g1', ['Si', 'Zn']);
  });
});

describe('the menu', () => {
  const openMenu = (name) => {
    fireEvent.click(screen.getByLabelText(`Actions for ${name}`));
    return screen.getByRole('menu', { name: `Actions for ${name}` });
  };

  it('is a menu, not a select', () => {
    // A select changes its value as the arrow keys move through it, so a
    // keyboard user visits "Delete" on the way past.
    show();
    expect(screen.queryByRole('combobox')).toBe(null);
    expect(openMenu('Al-Fe').getAttribute('role')).toBe('menu');
  });

  it('adds, moves, renames, deletes and hands the group to a run', () => {
    const p = show({ selection: ['Si'] });
    const menu = openMenu('Al-Fe');
    const items = within(menu).getAllByRole('menuitem').map((b) => b.textContent);
    expect(items).toEqual([
      'Add 1 selected phase',
      // Alt is NAMED here. A modifier mentioned in no label, no tooltip and
      // no hint is a modifier that does not exist -- a returning user went
      // looking for the old move behaviour, did not find it on any screen,
      // and had to open the manual.
      'Move 1 selected phase here (or Alt-drag)',
      'Remove selected phases (none of them is in this group)',
      'Use this group for the next run',
      'Subgroup of…',
      'Rename…',
      'Delete group',
    ]);
    fireEvent.click(within(menu).getByText('Delete group'));
    expect(p.onDelete).toHaveBeenCalledWith('g1');
  });

  it('move hands over the whole selection at once, not one phase at a time', () => {
    // It used to go one at a time. Two consequences, both found by an
    // adversarial read: Undo took back one twelfth of a twelve-phase drag,
    // and twelve parallel writes hit one file that nothing locks, so a
    // phase could end up on screen and not in the folder with every
    // request answering 200.
    const p = show({ selection: ['Si', 'Zn'] });
    fireEvent.click(within(openMenu('Al-Fe'))
      .getByText('Move 2 selected phases here (or Alt-drag)'));
    expect(p.onMove.mock.calls).toEqual([['g1', ['Si', 'Zn']]]);
  });

  it('an unusable item is disabled AND says why, rather than vanishing', () => {
    // A menu whose contents change between visits is a menu people stop
    // trusting.
    show({ selection: [] });
    const menu = openMenu('Al-Fe');
    const add = within(menu).getByText('Add selected phases (nothing selected)');
    expect(add.disabled).toBe(true);
  });

  it('"use for the next run" is off for a group with nothing in it, and says why', () => {
    // It was disabled with no reason at all, in a menu whose first two
    // entries name theirs -- which is worse than never establishing the
    // pattern: a reader then assumes the third one is broken, not blocked.
    show();
    const item = within(openMenu('Mg systems'))
      .getByText('Use this group for the next run (it is empty)');
    expect(item.disabled).toBe(true);
  });

  it('renaming happens in place and passes the new name on', () => {
    const p = show();
    fireEvent.click(within(openMenu('Al-Fe')).getByText('Rename…'));
    const box = screen.getByLabelText('New name for Al-Fe');
    fireEvent.change(box, { target: { value: 'Al-Fe-Si' } });
    fireEvent.submit(box.closest('form'));
    expect(p.onRename).toHaveBeenCalledWith('g1', 'Al-Fe-Si');
  });

  it('Escape out of a rename leaves the name alone', () => {
    const p = show();
    fireEvent.click(within(openMenu('Al-Fe')).getByText('Rename…'));
    fireEvent.keyDown(screen.getByLabelText('New name for Al-Fe'), { key: 'Escape' });
    expect(p.onRename).not.toHaveBeenCalled();
    expect(screen.getByText('Al-Fe')).toBeTruthy();
  });

  it('arrow keys move between items and skip the disabled ones', () => {
    show({ selection: [] });      // add, move and remove are all off
    const menu = openMenu('Al-Fe');
    const items = within(menu).getAllByRole('menuitem');
    const usable = items.filter((b) => !b.disabled);
    expect(usable.length).toBeGreaterThan(1);
    // opens on the first item that can be used
    expect(document.activeElement).toBe(usable[0]);
    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(usable[1]);
    fireEvent.keyDown(menu, { key: 'ArrowUp' });
    expect(document.activeElement).toBe(usable[0]);
  });

  it('Escape closes it and gives focus back to the button that opened it', () => {
    show();
    const button = screen.getByLabelText('Actions for Al-Fe');
    fireEvent.click(button);
    fireEvent.keyDown(screen.getByRole('menu', { name: 'Actions for Al-Fe' }),
      { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBe(null);
    expect(document.activeElement).toBe(button);
  });

  it('clicking elsewhere closes it', () => {
    show();
    fireEvent.click(screen.getByLabelText('Actions for Al-Fe'));
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole('menu')).toBe(null);
  });
});

describe('saying where the groups live', () => {
  it('says "this machine" while they really are in the browser', () => {
    show({ transitional: true });
    expect(screen.getByText(/this browser, on this machine/)).toBeTruthy();
  });

  it('and says nothing of the sort by default, because they are not', () => {
    // The prop used to DEFAULT to true, so a caller that forgot it
    // announced "groups are kept in this browser... they do not travel
    // with the library" -- false in every shipped build, and the opposite
    // of what the manual tells the reader. A default should be the state
    // the product is actually in.
    show();
    expect(screen.queryByText(/this browser, on this machine/)).toBe(null);
  });

  it('a change that was not saved is on screen, not swallowed', () => {
    show({ unsaved: [{ verb: 'added', groupName: 'Al-Fe', reason: 'read-only' }] });
    const box = screen.getByTestId('unsaved-changes');
    expect(box.textContent).toMatch(/1 change is on screen but not in the library/);
    expect(box.textContent).toMatch(/Al-Fe — read-only/);
  });

  it('several are all counted, and the recent ones named', () => {
    // There was one slot, cleared by the next successful change: a failure
    // followed by a success erased the warning while the unsaved change
    // stayed on screen, and after two failures the first was unnameable.
    show({ unsaved: [
      { verb: 'added', groupName: 'A', reason: 'read-only' },
      { verb: 'removed', groupName: 'B', reason: 'no such group' },
    ] });
    const box = screen.getByTestId('unsaved-changes');
    expect(box.textContent).toMatch(/2 changes are on screen/);
    expect(box.textContent).toMatch(/no such group/);
  });

  it('offers a read, because that is the only thing that makes it true again', () => {
    const p = show({ unsaved: [{ verb: 'added', groupName: 'A', reason: 'x' }] });
    fireEvent.click(screen.getByText('Read the library again'));
    expect(p.onReread).toHaveBeenCalled();
  });

  it('says nothing when everything landed', () => {
    show();
    expect(screen.queryByTestId('unsaved-changes')).toBe(null);
  });

  it('a library that cannot be read at all is its own sentence', () => {
    // It used to blame the BROWSER -- "this browser will not let the page
    // store anything" -- a leftover from when groups lived in
    // localStorage. They live in the library folder, so the sentence a
    // user meets when a share is not mounted named the wrong culprit and
    // sent them to their browser settings.
    show({ unavailable: true });
    expect(screen.getByText(/the library did not answer/)).toBeTruthy();
    // ... and does not blame the browser, which is where this sentence
    // used to point.
    expect(screen.getByText(/the library did not answer/).textContent)
      .not.toMatch(/browser/i);
  });

  it('and says how many unsaved changes it is NOT showing', () => {
    // The count said seven and the list showed three, so four were
    // unnameable -- the state this list exists to end.
    const unsaved = ['a', 'b', 'c', 'd', 'e'].map((n) => (
      { verb: 'created', groupName: n, reason: 'read-only' }));
    show({ unsaved });
    expect(screen.getByTestId('unsaved-overflow').textContent)
      .toMatch(/2 more/);
  });

  it('offers nothing to take back when nothing has happened', () => {
    show();
    expect(screen.queryByText('Undo')).toBe(null);
  });

  it('takes a change back when there is one', () => {
    const p = show({ canUndo: true });
    fireEvent.click(screen.getByText('Undo'));
    expect(p.onUndo).toHaveBeenCalled();
  });
});

describe('with no groups at all', () => {
  it('says how to make one, and how to fill it both ways', () => {
    show({ groups: [] });
    const text = screen.getByTestId('group-panel').textContent;
    expect(text).toMatch(/drag phases onto it/);
    expect(text).toMatch(/\+ button/);
  });
});

describe('subgroups', () => {
  // Sebastian, after the collections merge: "habe ich jetzt eigentlich noch
  // die Al-Fe-Si usw untergategorien?" -- he did not, and the only place
  // that could make one was the dialog this panel replaces.
  const nested = () => [
    { id: 'g1', name: 'Al-Fe', parent: null, members: ['Al'] },
    { id: 'g2', name: 'Mg systems', parent: null, members: [] },
    { id: 'g3', name: 'Al-Fe-Si', parent: 'g1', members: ['Si'] },
  ];

  it('a child is drawn under its parent, and marked as one', () => {
    show({ groups: nested() });
    const rows = [...document.querySelectorAll('[data-group-id]')]
      .map((n) => [n.getAttribute('data-group-id'),
        n.getAttribute('data-group-child') || 'no']);
    expect(rows).toEqual([['g1', 'no'], ['g3', 'yes'], ['g2', 'no']]);
  });

  it('a child whose parent has vanished is shown, not lost', () => {
    // An invisible group is worse than a misplaced one.
    show({ groups: [{ id: 'g3', name: 'Orphan', parent: 'gone', members: [] }] });
    expect(screen.getByText('Orphan')).toBeTruthy();
  });

  it('offers exactly the groups it could go under, and no others', () => {
    show({ groups: nested() });
    fireEvent.click(screen.getByLabelText('Actions for Mg systems'));
    fireEvent.click(screen.getByText('Subgroup of…'));
    const offers = [...screen.getByTestId('parent-pick-g2')
      .querySelectorAll('button')].map((b) => b.textContent);
    // `Al-Fe` yes -- one level means a CHILD has no children, not that a
    // parent may have only one (my first version of this test had that
    // backwards). Itself no. `Al-Fe-Si` no, it is already a child.
    expect(offers).toEqual(['Al-Fe', 'Cancel']);
  });

  it('and hands the chosen parent on', () => {
    const p = show({ groups: nested() });
    fireEvent.click(screen.getByLabelText('Actions for Mg systems'));
    fireEvent.click(screen.getByText('Subgroup of…'));
    fireEvent.click(within(screen.getByTestId('parent-pick-g2')).getByText('Al-Fe'));
    expect(p.onNest).toHaveBeenCalledWith('g2', 'g1');
  });

  it('a group with children cannot itself become a child', () => {
    show({ groups: nested() });
    fireEvent.click(screen.getByLabelText('Actions for Al-Fe'));
    const item = screen.getByText('Subgroup of… (nowhere to put it)');
    expect(item.disabled).toBe(true);
  });

  it('a child can be promoted back to the top', () => {
    const p = show({ groups: nested() });
    fireEvent.click(screen.getByLabelText('Actions for Al-Fe-Si'));
    fireEvent.click(screen.getByText('Subgroup of…'));
    fireEvent.click(within(screen.getByTestId('parent-pick-g3'))
      .getByText('Top level'));
    expect(p.onNest).toHaveBeenCalledWith('g3', null);
  });

  it('and a root has no "top level" offer, because it is already there', () => {
    show({ groups: nested() });
    fireEvent.click(screen.getByLabelText('Actions for Mg systems'));
    fireEvent.click(screen.getByText('Subgroup of…'));
    expect(within(screen.getByTestId('parent-pick-g2'))
      .queryByText('Top level')).toBe(null);
  });
});
