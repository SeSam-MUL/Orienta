import { describe, it, expect } from 'vitest';
import {
  makeGroup, groupsOfPhase, addMember, removeMember, moveTo, renameGroup,
  deleteGroup, groupCounts, ungrouped, journalEntry, canNest,
} from './groups';

const T = '2026-09-27T10:00:00Z';
const T2 = '2026-09-27T11:00:00Z';

function fixture() {
  return [
    { id: 'g1', name: 'Al-Fe', parent: null, members: ['Al', 'Al13Fe4'],
      created: T, updated: T, author: 'seb' },
    { id: 'g2', name: 'Cubic', parent: null, members: ['Al', 'Si'],
      created: T, updated: T, author: 'seb' },
  ];
}

describe('making a group', () => {
  it('trims the name', () => {
    expect(makeGroup({ id: 'x', name: '  Al-Fe  ' }).name).toBe('Al-Fe');
  });

  it('refuses a name that is only whitespace', () => {
    // A group called "Untitled" is a group nobody finds again.
    expect(() => makeGroup({ id: 'x', name: '   ' })).toThrow(/needs a name/);
    expect(() => makeGroup({ id: 'x', name: '' })).toThrow(/needs a name/);
    expect(() => makeGroup({ id: 'x' })).toThrow(/needs a name/);
  });

  it('starts empty, at the root, and records who and when', () => {
    const g = makeGroup({ id: 'x', name: 'N', author: 'seb', now: T });
    expect(g.members).toEqual([]);
    expect(g.parent).toBe(null);
    expect(g).toMatchObject({ author: 'seb', created: T, updated: T });
  });
});

describe('membership is a tag, not a folder', () => {
  it('a phase can be in several groups at once', () => {
    expect(groupsOfPhase(fixture(), 'Al')).toEqual(['g1', 'g2']);
  });

  it('adding to one group leaves the other alone', () => {
    const next = addMember(fixture(), 'g1', 'Si', { now: T2 });
    expect(next[0].members).toEqual(['Al', 'Al13Fe4', 'Si']);
    expect(next[1].members).toEqual(['Al', 'Si']);
  });

  it('removing from one group leaves the phase in the other', () => {
    const next = removeMember(fixture(), 'g1', 'Al', { now: T2 });
    expect(groupsOfPhase(next, 'Al')).toEqual(['g2']);
  });

  it('adds in the order they were added, not sorted', () => {
    // The order is a record of what somebody did; sorting throws it away.
    let gs = [makeGroup({ id: 'g', name: 'N' })];
    gs = addMember(gs, 'g', 'Zn');
    gs = addMember(gs, 'g', 'Al');
    expect(gs[0].members).toEqual(['Zn', 'Al']);
  });
});

describe('a change that changes nothing is not a change', () => {
  // The journal is shared. An entry saying "added Al" when Al was already
  // there is a lie in a record eight people read.
  it('adding a member twice returns the very same array', () => {
    const gs = fixture();
    expect(addMember(gs, 'g1', 'Al')).toBe(gs);
  });

  it('removing a member that is not there returns the very same array', () => {
    const gs = fixture();
    expect(removeMember(gs, 'g1', 'Si')).toBe(gs);
  });

  it('naming a group that does not exist returns the very same array', () => {
    const gs = fixture();
    expect(addMember(gs, 'nope', 'Al')).toBe(gs);
    expect(removeMember(gs, 'nope', 'Al')).toBe(gs);
  });

  it('a real add stamps `updated` only on the group that changed', () => {
    const next = addMember(fixture(), 'g1', 'Si', { now: T2 });
    expect(next[0].updated).toBe(T2);
    expect(next[1].updated).toBe(T);
  });
});

describe('moveTo is the exception, and it is the only one', () => {
  it('puts the phase here and takes it out of everywhere else', () => {
    const next = moveTo(fixture(), 'g2', 'Al13Fe4', { now: T2 });
    expect(next[1].members).toContain('Al13Fe4');
    expect(next[0].members).not.toContain('Al13Fe4');
  });

  it('moving to a group it is already in does not remove it from there', () => {
    const next = moveTo(fixture(), 'g1', 'Al', { now: T2 });
    expect(next[0].members).toEqual(['Al', 'Al13Fe4']);
    expect(next[1].members).toEqual(['Si']);
  });

  it('leaves groups it was never in untouched, timestamp and all', () => {
    const next = moveTo(fixture(), 'g1', 'Zn', { now: T2 });
    expect(next[1].updated).toBe(T);
    expect(next[0].members).toContain('Zn');
  });
});

describe('renaming', () => {
  it('changes the name and nothing else', () => {
    const next = renameGroup(fixture(), 'g1', ' Al-Fe-Si ', { now: T2 });
    expect(next[0].name).toBe('Al-Fe-Si');
    expect(next[0].id).toBe('g1');
    expect(next[0].members).toEqual(['Al', 'Al13Fe4']);
    expect(next[0].updated).toBe(T2);
  });

  it('refuses an empty name', () => {
    expect(() => renameGroup(fixture(), 'g1', '  ')).toThrow(/needs a name/);
  });

  it('leaves the other groups alone', () => {
    expect(renameGroup(fixture(), 'g1', 'X')[1]).toEqual(fixture()[1]);
  });
});

describe('deleting', () => {
  it('promotes its children instead of deleting them', () => {
    // This test asserted the opposite until an adversarial read compared it
    // with the folder: `phase_collections.delete` sets each child's parent
    // to None and keeps it. The screen removed a child the folder kept, so
    // twelve memberships "vanished" and came back at the top level on the
    // next start. One truth, and it is the folder's.
    const gs = [...fixture(),
      { id: 'g3', name: 'child', parent: 'g1', members: ['Si'] }];
    const after = deleteGroup(gs, 'g1');
    expect(after.map((g) => g.id)).toEqual(['g2', 'g3']);
    expect(after.find((g) => g.id === 'g3').parent).toBe(null);
    expect(after.find((g) => g.id === 'g3').members).toEqual(['Si']);
  });

  it('deleting a childless group leaves the rest', () => {
    expect(deleteGroup(fixture(), 'g2').map((g) => g.id)).toEqual(['g1']);
  });

  it('does not touch memberships in other groups', () => {
    expect(deleteGroup(fixture(), 'g1')[0].members).toEqual(['Al', 'Si']);
  });
});

describe('counts are measured on what is shown', () => {
  // Today `Mg-Systeme 3` stays put while the list shows nothing. A counter
  // that contradicts the list is worse than no counter.
  it('counts only the members that survived the filter', () => {
    expect(groupCounts(fixture(), ['Al'])).toEqual([
      { id: 'g1', shown: 1, total: 2 },
      { id: 'g2', shown: 1, total: 2 },
    ]);
  });

  it('a group filtered down to nothing reads differently from an empty one', () => {
    const gs = [...fixture(), makeGroup({ id: 'g3', name: 'empty' })];
    const c = groupCounts(gs, ['Zn']);
    expect(c[0]).toEqual({ id: 'g1', shown: 0, total: 2 });
    expect(c[2]).toEqual({ id: 'g3', shown: 0, total: 0 });
  });

  it('counts a phase once even when the filter repeats it', () => {
    expect(groupCounts(fixture(), ['Al', 'Al'])[0].shown).toBe(1);
  });
});

describe('what is in no group at all', () => {
  it('lists the shown phases nobody filed', () => {
    expect(ungrouped(fixture(), ['Al', 'Si', 'Zn', 'Mg'])).toEqual(['Zn', 'Mg']);
  });

  it('keeps the order the list had', () => {
    expect(ungrouped(fixture(), ['Mg', 'Zn'])).toEqual(['Mg', 'Zn']);
  });

  it('is everything when there are no groups', () => {
    expect(ungrouped([], ['Al', 'Si'])).toEqual(['Al', 'Si']);
  });
});

describe('the journal', () => {
  it('records the verb, the group, the phase, who and when', () => {
    expect(journalEntry('added', {
      groupId: 'g1', groupName: 'Al-Fe', key: 'Si', author: 'seb', now: T,
    })).toEqual({
      verb: 'added', groupId: 'g1', groupName: 'Al-Fe', key: 'Si',
      author: 'seb', at: T,
    });
  });

  it('carries no phase for a change that is about the group itself', () => {
    expect(journalEntry('renamed', {
      groupId: 'g1', groupName: 'New', author: 'seb', now: T,
    }).key).toBe(null);
  });

  it('carries the NAME as well as the id, so a deleted group still reads', () => {
    // An entry that says only `g1` is unreadable once `g1` is gone.
    expect(journalEntry('deleted', {
      groupId: 'g1', groupName: 'Al-Fe', author: 'seb', now: T,
    }).groupName).toBe('Al-Fe');
  });
});

describe('one level of nesting, checked not assumed', () => {
  const nested = () => [
    { id: 'a', name: 'A', parent: null, members: [] },
    { id: 'b', name: 'B', parent: 'a', members: [] },
    { id: 'c', name: 'C', parent: null, members: [] },
  ];

  it('allows a root under another root', () => {
    expect(canNest(nested(), 'c', 'a')).toBe(true);
  });

  it('refuses nesting under a group that is already a child', () => {
    expect(canNest(nested(), 'c', 'b')).toBe(false);
  });

  it('refuses nesting a group that already has children', () => {
    expect(canNest(nested(), 'a', 'c')).toBe(false);
  });

  it('refuses nesting a group inside itself', () => {
    // `c`, deliberately: it is a root AND childless, so neither of the other
    // two rules can refuse it. Asking this of `a` passes even with the
    // self-check deleted -- `a` has a child, and the third rule catches it.
    // A mutation run is what said so.
    expect(canNest(nested(), 'c', 'c')).toBe(false);
  });

  it('refuses a parent that does not exist', () => {
    expect(canNest(nested(), 'c', 'nope')).toBe(false);
  });
});
