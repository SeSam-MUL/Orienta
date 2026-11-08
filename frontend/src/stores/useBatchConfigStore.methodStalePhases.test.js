import { describe, it, expect, beforeEach } from 'vitest';
import useBatchConfigStore from './useBatchConfigStore';

describe('batch config: phases stay bound to the method they were picked under', () => {
  beforeEach(() => {
    useBatchConfigStore.getState().clearFiles();
  });

  it('switching method clears phases picked under a different method, instead of keeping their old path/method', () => {
    const store = useBatchConfigStore.getState();
    store.addFiles(['a.up1']);
    // A phase discovered and applied while the method was 'spherical': its path is an
    // SHT master, picked from the method-dependent discovery list for 'spherical'.
    store.applyGlobalPhases([{ name: 'Ti', path: 'Ti_beta.sht', method: 'spherical' }]);
    expect(useBatchConfigStore.getState().globalPhases).toHaveLength(1);

    // The user switches the batch to Hough. The discovery list for 'hough' lists CIFs,
    // not SHT masters, so the previously-picked phase's path can never be valid for the
    // new method — it must not survive the switch under either its old path or method.
    store.setMethod('hough');

    const state = useBatchConfigStore.getState();
    expect(state.globalPhases).toHaveLength(0);
    expect(state.files[0].config.phases).toHaveLength(0);
    expect(state.files[0].phases_configured).toBe(false);
  });

  it('switching method back to the one phases were picked under keeps them', () => {
    const store = useBatchConfigStore.getState();
    store.addFiles(['a.up1']);
    store.setMethod('hough');
    store.applyGlobalPhases([{ name: 'Ti', path: 'alpha-Ti.cif', method: 'hough' }]);

    // No-op switch (already 'hough'): must not drop the phases.
    store.setMethod('hough');

    expect(useBatchConfigStore.getState().globalPhases).toHaveLength(1);
    expect(useBatchConfigStore.getState().files[0].config.phases).toHaveLength(1);
  });
});
