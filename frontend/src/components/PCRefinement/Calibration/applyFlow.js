// The window "Apply to other files" and the file it opened for Find PC, kept OUTSIDE the PC page.
//
// The app mounts the PC page with the open file's path as its key (App.jsx), so opening another file unmounts the
// page and mounts a new one. Whatever the first page held in its state is gone then. This store is what survives:
//   flow.calibration  the calibration version being applied (the answer of Save, or the offer that was used)
//   flow.origin       the file whose PC it is (not a row of the window)
//   flow.far          the file the user chose "Find PC for it" for: {dataset, path, openedPath, step, error, saved}
//                     step: 'starting' (open it, then start Find PC once) | 'searching' | 'error'
//                     `openedPath`: the path the switch answer named for that file, once it answered
// A page that mounts while `far.step` is 'starting' and the open file is the far file claims the start (`claimStart`
// answers true exactly once), so Find PC is started once, by whichever page instance is mounted for that file.
// The far file is told by its PATH, not by a dataset name: switching back to a file restores the dataset that was
// active there last, which may be a derived copy of the one the row names.
import { create } from 'zustand';

/** True when `openPath` is the far file: the path of the row or the path its switch answered with. */
export const isFarFile = (far, openPath) => !!far && !!openPath && (far.path === openPath || far.openedPath === openPath);

const useApplyFlow = create((set, get) => ({
  flow: null,
  open: (calibration, origin) => set({ flow: { calibration, origin, far: null } }),
  close: () => set({ flow: null }),
  setFar: (far) => set((s) => (s.flow ? { flow: { ...s.flow, far } } : s)),
  patchFar: (patch) => set((s) => (s.flow && s.flow.far ? { flow: { ...s.flow, far: { ...s.flow.far, ...patch } } } : s)),
  /** True once for the file whose Find PC is waiting to be started (`openPath` is the open file); the step then reads 'searching'. */
  claimStart: (openPath) => {
    const far = get().flow?.far;
    if (!far || far.step !== 'starting' || !isFarFile(far, openPath)) return false;
    get().patchFar({ step: 'searching' });
    return true;
  },
}));

export default useApplyFlow;
