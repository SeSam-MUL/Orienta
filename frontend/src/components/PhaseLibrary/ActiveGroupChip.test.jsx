// @vitest-environment jsdom
/**
 * The one line on every page that says the phase choice is narrowed.
 *
 * It is what is left of a dropdown that both chose and displayed. Choosing
 * moved to the library; this kept the harder half of the job, and the
 * regression it lived through is why: the filter stopped resolving, every
 * run silently widened to the whole library, and this line went on naming
 * a group that was not being applied. So half of what is below is about
 * the chip telling the truth when something is wrong.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';
import i18n from '../../i18n';
import ActiveGroupChip from './ActiveGroupChip';
import useCollectionStore from '../../stores/useCollectionStore';

const COLLECTIONS = [
  { id: 'Al_systems', name: 'Al systems', parent: null,
    member_count: 2, effective_member_count: 3, members: [] },
];

async function show(state, collections = COLLECTIONS) {
  await act(async () => { await i18n.changeLanguage('en'); });
  useCollectionStore.setState({ data: { collections, state } });
  const onOpenLibrary = vi.fn();
  render(<ActiveGroupChip onOpenLibrary={onOpenLibrary} />);
  return onOpenLibrary;
}

afterEach(() => {
  cleanup();
  useCollectionStore.setState({ data: { collections: [], unassigned: [], problems: [], state: {} } });
});

describe('when a group is active', () => {
  it('names it, and says how many phases it offers', async () => {
    await show({ active: 'Al systems', hidden: [] });
    const chip = screen.getByTestId('active-group-chip');
    expect(chip.textContent).toMatch(/Group: Al systems/);
    // The EFFECTIVE count -- a parent offers its children's phases too, and
    // the number here has to be the number the run will see.
    expect(chip.textContent).toMatch(/3 phases/);
  });

  it('says "Group", not "Collection"', async () => {
    // Sebastian did not like the word, and with the manager gone it has no
    // business surviving in the interface. The routes and the files keep
    // their names; this is about what a person reads.
    await show({ active: 'Al systems', hidden: [] });
    expect(screen.getByTestId('active-group-chip').textContent)
      .not.toMatch(/collection/i);
  });

  it('takes you to the library, which is where choosing happens now', async () => {
    const go = await show({ active: 'Al systems', hidden: [] });
    fireEvent.click(screen.getByTestId('active-group-chip'));
    expect(go).toHaveBeenCalled();
  });
});

describe('when nothing is active', () => {
  it('shows nothing at all', async () => {
    // Every phase is on offer, which is the ordinary case and needs no
    // announcement. A chip reading "Group: none" is noise on every page.
    await show({ active: null, hidden: [] });
    expect(screen.queryByTestId('active-group-chip')).toBe(null);
  });

  it('and the same before anything has loaded', async () => {
    await show({});
    expect(screen.queryByTestId('active-group-chip')).toBe(null);
  });
});

describe('when the active group is not there any more', () => {
  it('says so, instead of naming a group that is not being applied', async () => {
    await show({ active: 'Gone yesterday', hidden: [] });
    const chip = screen.getByTestId('active-group-chip');
    expect(chip.textContent).toMatch(/not found/);
    expect(chip.getAttribute('data-missing')).toBe('yes');
  });

  it('does not claim a phase count it cannot have', async () => {
    await show({ active: 'Gone yesterday', hidden: [] });
    expect(screen.getByTestId('active-group-chip').textContent)
      .not.toMatch(/phases?$/);
  });

  it('explains what happened and what to do', async () => {
    await show({ active: 'Gone yesterday', hidden: [] });
    const tip = screen.getByTestId('active-group-chip').getAttribute('title');
    expect(tip).toMatch(/renamed or deleted/);
    expect(tip).toMatch(/Nothing is on offer/);
  });

  it('an id where a name belongs reads as missing, which is what it is', async () => {
    // The exact regression: `GET /` sent `Al_systems` where the filter
    // compares `Al systems`. It resolved to nothing, and the old chip said
    // "Collection: Al_systems" as though all was well.
    await show({ active: 'Al_systems', hidden: [] });
    expect(screen.getByTestId('active-group-chip').getAttribute('data-missing'))
      .toBe('yes');
  });

  it('a group that is there but empty is NOT missing', async () => {
    // Two different screens: "this group holds nothing" and "this group is
    // gone". Only one of them is a problem.
    await show({ active: 'Empty one', hidden: [] },
      [{ id: 'Empty_one', name: 'Empty one', parent: null,
        member_count: 0, effective_member_count: 0, members: [] }]);
    const chip = screen.getByTestId('active-group-chip');
    expect(chip.getAttribute('data-missing')).toBe('no');
    expect(chip.textContent).toMatch(/0 phases/);
  });
});

describe('somebody has to ask the server', () => {
  it('reads the groups once when it mounts', async () => {
    // The dropdown this replaces did it. Removing that left nothing loading
    // the store at start-up -- measured in the running app: no chip in the
    // toolbar at all, while `_local.json` had an active group the whole
    // time. An indicator that appears late is worse than none.
    const load = vi.fn(() => Promise.resolve());
    useCollectionStore.setState({ data: { collections: [], state: {} }, load });
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<ActiveGroupChip onOpenLibrary={() => {}} />);
    expect(load).toHaveBeenCalledTimes(1);
  });

  it('does not ask again on every render', async () => {
    const load = vi.fn(() => Promise.resolve());
    useCollectionStore.setState({ data: { collections: [], state: {} }, load });
    await act(async () => { await i18n.changeLanguage('en'); });
    const { rerender } = render(<ActiveGroupChip onOpenLibrary={() => {}} />);
    rerender(<ActiveGroupChip onOpenLibrary={() => {}} />);
    rerender(<ActiveGroupChip onOpenLibrary={() => {}} />);
    expect(load).toHaveBeenCalledTimes(1);
  });

  it('asks even though it will render nothing -- that is the point', async () => {
    // With no active group the chip draws nothing, and the hook still has
    // to run: it is what finds out whether there IS one. A hook placed
    // after the early return would also be the "Rendered more hooks than
    // during the previous render" crash of 2026-08-14.
    const load = vi.fn(() => Promise.resolve());
    useCollectionStore.setState({ data: { collections: [], state: {} }, load });
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<ActiveGroupChip onOpenLibrary={() => {}} />);
    expect(screen.queryByTestId('active-group-chip')).toBe(null);
    expect(load).toHaveBeenCalled();
  });
});
