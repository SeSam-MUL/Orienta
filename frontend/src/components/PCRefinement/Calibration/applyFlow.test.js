import { describe, it, expect, beforeEach } from 'vitest';
import useApplyFlow, { isFarFile } from './applyFlow';

const CAL = { id: 'sem', name: 'SEM', version: 2, pc: [0.5, 0.4, 0.6] };
const FAR = '/d/scan_far.h5oina';

describe('applyFlow: what survives the page being mounted anew for another file', () => {
  beforeEach(() => useApplyFlow.setState({ flow: null }));

  it('holds the calibration and the file it is from until it is closed', () => {
    useApplyFlow.getState().open(CAL, 'scan_a');
    expect(useApplyFlow.getState().flow).toEqual({ calibration: CAL, origin: 'scan_a', far: null });
    useApplyFlow.getState().close();
    expect(useApplyFlow.getState().flow).toBeNull();
  });

  it('hands out the start of Find PC once, and only to the file it was asked for (told by its path)', () => {
    const f = useApplyFlow.getState();
    f.open(CAL, 'scan_a');
    f.setFar({ dataset: 'scan_far', path: FAR, step: 'starting' });
    expect(useApplyFlow.getState().claimStart('/d/scan_a.h5oina')).toBe(false);   // the page of another file
    expect(useApplyFlow.getState().flow.far.step).toBe('starting');
    expect(useApplyFlow.getState().claimStart(FAR)).toBe(true);
    expect(useApplyFlow.getState().flow.far.step).toBe('searching');
    expect(useApplyFlow.getState().claimStart(FAR)).toBe(false);                   // a second page instance gets nothing
  });

  it('does not tell the file by a dataset name: a copy of the dataset on the far file claims it, a path of another file does not', () => {
    const f = useApplyFlow.getState();
    f.open(CAL, 'scan_a');
    f.setFar({ dataset: 'scan_far', path: FAR, step: 'starting' });
    expect(useApplyFlow.getState().claimStart('scan_far')).toBe(false);            // a name is not a path
    expect(useApplyFlow.getState().claimStart(FAR)).toBe(true);
  });

  it('knows the far file by the path its switch answered with, too', () => {
    const f = useApplyFlow.getState();
    f.open(CAL, 'scan_a');
    f.setFar({ dataset: 'scan_far', path: FAR, step: 'starting' });
    f.patchFar({ openedPath: 'D:\\d\\scan_far.h5oina' });
    expect(isFarFile(useApplyFlow.getState().flow.far, 'D:\\d\\scan_far.h5oina')).toBe(true);
    expect(isFarFile(useApplyFlow.getState().flow.far, FAR)).toBe(true);
    expect(isFarFile(useApplyFlow.getState().flow.far, '/d/other.h5oina')).toBe(false);
    expect(isFarFile(useApplyFlow.getState().flow.far, null)).toBe(false);
    expect(isFarFile(null, FAR)).toBe(false);
    expect(useApplyFlow.getState().claimStart('D:\\d\\scan_far.h5oina')).toBe(true);
  });

  it('hands out nothing when there is no flow or the file could not be opened', () => {
    expect(useApplyFlow.getState().claimStart(FAR)).toBe(false);
    useApplyFlow.getState().open(CAL, 'scan_a');
    useApplyFlow.getState().setFar({ dataset: 'scan_far', path: FAR, step: 'error', error: 'x' });
    expect(useApplyFlow.getState().claimStart(FAR)).toBe(false);
  });

  it('hands out nothing after the window was closed', () => {
    const f = useApplyFlow.getState();
    f.open(CAL, 'scan_a');
    f.setFar({ dataset: 'scan_far', path: FAR, step: 'starting' });
    f.close();
    expect(useApplyFlow.getState().claimStart(FAR)).toBe(false);
  });
});
