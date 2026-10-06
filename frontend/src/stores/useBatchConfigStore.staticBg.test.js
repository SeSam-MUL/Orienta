import { describe, it, expect, beforeEach } from 'vitest';
import useBatchConfigStore from './useBatchConfigStore';

describe('batch config: static background reference', () => {
  beforeEach(() => {
    useBatchConfigStore.getState().clearFiles();
  });

  it('a new file does not default the reference to pattern (0, 0)', () => {
    useBatchConfigStore.getState().addFiles(['a.h5oina']);
    const cfg = useBatchConfigStore.getState().files[0].config;
    // null = no reference: the backend subtracts the mean of all patterns of
    // the file. 0/0 would mean "the pattern at the scan origin".
    expect(cfg.static_bg_row).toBeNull();
    expect(cfg.static_bg_col).toBeNull();
  });
});
