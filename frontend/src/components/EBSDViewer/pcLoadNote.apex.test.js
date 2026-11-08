import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { pcLoadNote } from './pcLoadNote';

const locale = (lang) => JSON.parse(readFileSync(
  fileURLToPath(new URL(`../../locales/${lang}/ebsdviewer.json`, import.meta.url)), 'utf8'));

describe('pcLoadNote for an APEX measurement', () => {
  it('says where the pattern centre came from', () => {
    expect(pcLoadNote({ pc_defaulted: false, pc_source: 'apex', pc_origin: 'apex', pc: [0.5, 0.6, 0.6] }))
      .toEqual({ level: 'log', key: 'pcFromApex', params: {} });
  });

  it('still warns first when the store has no PC', () => {
    expect(pcLoadNote({ pc_defaulted: true, pc_source: 'default', pc: null }).key).toBe('pcDefaultWarning');
  });

  for (const lang of ['en', 'de', 'ja', 'zh']) {
    it(`${lang} has the APEX line and a warning that is not about .osc files only`, () => {
      const m = locale(lang).logMessages;
      expect(typeof m.pcFromApex).toBe('string');
      expect(m.pcFromApex).toMatch(/APEX/);
      expect(m.pcDefaultWarning).not.toMatch(/\.osc/);          // NORDIF and Oxford .ebsp have none either
    });
  }
});
