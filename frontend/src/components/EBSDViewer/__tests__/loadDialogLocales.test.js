// @vitest-environment node
//
// The texts of the generic load dialog. Every key the component asks for exists in en/de/ja/zh with the
// same placeholders (the source of the component is read, so a key added there without a text turns
// this red), German addresses the reader formally, and the "cannot be read" question no longer
// speaks of "the measurement dialog": it now meets every HDF5 format, not only EDAX APEX files.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const LANGS = ['en', 'de', 'ja', 'zh'];
const locale = (lang) => JSON.parse(read(`../../../locales/${lang}/ebsdviewer.json`));
const SOURCE = read('../LoadDialog.jsx');

/** `t('a.b')` and `t(`a.${x}`)` calls of the component: the dotted keys, with `${...}` taken as every key below. */
const asked = [...SOURCE.matchAll(/\bt\('([A-Za-z_.]+)'/g)].map((m) => m[1]);
const dynamic = [...SOURCE.matchAll(/\bt\(`([A-Za-z_.]+)\.\$\{/g)].map((m) => m[1]);

const leaf = (node, dotted) => dotted.split('.').reduce((a, k) => (a == null ? a : a[k]), node);
const plural = (node, dotted) => leaf(node, dotted) ?? leaf(node, `${dotted}_other`);
const holes = (text) => [...String(text).matchAll(/\{\{(\w+)[^}]*\}\}/g)].map((m) => m[1]).sort();

describe('the dialog asks for texts (a guard over nothing proves nothing)', () => {
  it('reads a believable number of keys from the component', () => {
    expect(asked.length).toBeGreaterThanOrEqual(12);
    expect(dynamic).toContain('source');
    expect(dynamic).toContain('bits');
  });
});

describe.each(LANGS)('%s', (lang) => {
  const loc = locale(lang).loadDialog;

  it('has a text for every key the component asks for', () => {
    const missing = asked.filter((k) => typeof plural(loc, k) !== 'string' || plural(loc, k).trim() === '');
    expect(missing).toEqual([]);
  });

  it('has the three pattern sources and the bit-depth lines', () => {
    for (const key of ['source.processed', 'source.unprocessed', 'source.single',
      'bits.processed', 'bits.unprocessed', 'bits.other', 'unprocessedTag']) {
      expect(typeof leaf(loc, key), key).toBe('string');
    }
  });

  it('uses the placeholders of the English text', () => {
    const en = locale('en').loadDialog;
    const flat = (node, prefix = '') => Object.entries(node).flatMap(([k, v]) => (typeof v === 'object'
      ? flat(v, `${prefix}${k}.`) : [[`${prefix}${k}`, v]]));
    const wrong = flat(en).filter(([k, v]) => holes(v).join() !== holes(leaf(loc, k)).join()).map(([k]) => k);
    expect(wrong).toEqual([]);
  });
});

describe('the fixed wording of the dialog', () => {
  it('says the two bit depths in English in the agreed words', () => {
    const en = locale('en').loadDialog;
    expect(en.bits.unprocessed).toBe('16-bit counts');
    expect(en.bits.processed).toBe('8-bit, background corrected by the acquisition software');
  });

  it('tells that a typed pattern centre belongs to the variant', () => {
    expect(locale('en').loadDialog.pcNote).toMatch(/pattern centre/);
    expect(locale('en').loadDialog.pcNote).toMatch(/other kind of patterns/);
  });

  it('addresses the German reader formally', () => {
    const texts = JSON.stringify(locale('de').loadDialog) + locale('de').apexDialog.unreadableMessage;
    expect(texts).not.toMatch(/\b(du|dein\w*|dich|dir)\b/i);
  });
});

describe('the "cannot be read" question fits every HDF5 format', () => {
  it.each(LANGS)('%s no longer names the measurement dialog', (lang) => {
    const text = locale(lang).apexDialog.unreadableMessage;
    expect(text).toContain('{{reason}}');
    expect(text).not.toMatch(/measurement dialog|Messungsdialog|測定ダイアログ|测量对话框/);
  });

  it('English says what is not offered: a choice of measurement or pattern source', () => {
    expect(locale('en').apexDialog.unreadableMessage).toMatch(/choice of measurement or pattern source/);
  });
});
