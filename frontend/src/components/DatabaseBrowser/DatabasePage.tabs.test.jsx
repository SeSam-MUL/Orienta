// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { TAB_DEFS, entryMatchesTab } from './DatabasePage';

describe('Database Browser tabs (3-way SHT / MC h5 / Master H5 split)', () => {
  it('exposes SHT / MC h5 / Master H5 / CIF / XTAL / Dictionary tabs in order', () => {
    expect(TAB_DEFS.map((t) => t.id)).toEqual(['sht', 'mc', 'master', 'cif', 'xtal', 'dict']);
    expect(TAB_DEFS.map((t) => t.label)).toEqual(['SHT', 'MC h5', 'Master H5', 'CIF', 'XTAL', 'Dictionary']);
  });

  it('each new tab matches exactly one category and hides the type sub-filter', () => {
    const byId = Object.fromEntries(TAB_DEFS.map((t) => [t.id, t]));
    expect([...byId.sht.fileTypes]).toEqual(['sht']);
    expect([...byId.mc.fileTypes]).toEqual(['h5']);
    expect([...byId.master.fileTypes]).toEqual(['master']);
    for (const id of ['sht', 'mc', 'master']) {
      expect(byId[id].typeFilterItems).toBeNull();
    }
  });

  it('routes each backend category to exactly one of the three tabs', () => {
    const byId = Object.fromEntries(TAB_DEFS.map((t) => [t.id, t]));
    const cases = [
      ['sht', 'sht'],       // .sht
      ['h5', 'mc'],         // Monte-Carlo .h5 (residual category)
      ['master', 'master'], // master-pattern .h5
    ];
    for (const [category, expectedTab] of cases) {
      const entry = { category };
      for (const id of ['sht', 'mc', 'master']) {
        expect(entryMatchesTab(entry, byId[id])).toBe(id === expectedTab);
      }
    }
  });

  it('entryMatchesTab falls back through file_type / type / category', () => {
    const shtTab = TAB_DEFS.find((t) => t.id === 'sht');
    expect(entryMatchesTab({ file_type: 'SHT' }, shtTab)).toBe(true);
    expect(entryMatchesTab({ type: 'sht' }, shtTab)).toBe(true);
    expect(entryMatchesTab({ category: 'h5' }, shtTab)).toBe(false);
  });
});
