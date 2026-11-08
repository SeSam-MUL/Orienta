// @vitest-environment node
//
// The APEX dialog says every refusal code and every limit of a hexagonal scan in four languages.
// The two lists come from the backend (fixtures generated from edax_apex.APEX_CODES and
// hex_guard.limits(); tests/test_apex_dialog_frontend_fixtures.py holds the fixtures against the
// backend, so they cannot go stale). A code, key, category, affected kind or parameter the backend
// has and the locales lack turns this red, and so does a text the backend no longer has.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const json = (rel) => JSON.parse(read(rel));
const LANGS = ['en', 'de', 'ja', 'zh'];
const locale = (lang, ns = 'ebsdviewer') => json(`../../locales/${lang}/${ns}.json`);

const CODES = json('./__fixtures__/apexCodes.json').codes;
const LIMITS = json('./__fixtures__/hexLimits.json');
const ITEMS = Object.values(LIMITS.groups).flat();
const sorted = (xs) => [...new Set(xs)].sort();

describe('the fixtures are not empty (a test over an empty list proves nothing)', () => {
  it('has codes, limits, categories and the things that work', () => {
    expect(CODES.length).toBeGreaterThanOrEqual(18);
    expect(ITEMS.length).toBeGreaterThanOrEqual(12);
    expect(LIMITS.categories).toHaveLength(5);
    expect(LIMITS.not_limited.map((n) => n.key)).toEqual(['image_export', 'file_export', 'global_pc_refinement']);
  });
});

describe.each(LANGS)('%s', (lang) => {
  const d = locale(lang);

  it('words every APEX refusal code under errors.*', () => {
    const missing = CODES.filter((c) => typeof d.errors?.[c] !== 'string' || d.errors[c].length < 6);
    expect(missing).toEqual([]);
  });

  it('words the refusal of a hexagonal UP1/UP2 file without its metadata, and says what to do', () => {
    expect(d.errors.edaxHexUpUnsupported.length).toBeGreaterThan(40);
    expect(d.errors.edaxHexUpUnsupported).toMatch(/APEX/);
  });

  it('has a text for every limit key, category, affected kind, parameter and thing that works', () => {
    const h = d.apexDialog.hexLimits;
    const keys = sorted(ITEMS.map((i) => i.key));
    const affects = sorted(ITEMS.flatMap((i) => i.affects || []));
    const parameters = sorted(ITEMS.flatMap((i) => i.parameters || []));
    const works = sorted(LIMITS.not_limited.map((n) => n.key));
    for (const [part, wanted] of [
      ['keys', keys], ['categories', LIMITS.categories], ['affects', affects],
      ['parameters', parameters], ['works', works],
    ]) {
      const have = Object.keys(h[part] || {});
      expect({ part, missing: wanted.filter((k) => typeof h[part]?.[k] !== 'string' || !h[part][k].trim()) })
        .toEqual({ part, missing: [] });
      // ... and no text the backend no longer has
      expect({ part, extra: have.filter((k) => !wanted.includes(k)) }).toEqual({ part, extra: [] });
    }
  });

  it('has the hexagonal sentence shapes in common.json (the dialog can show them through problemText)', () => {
    const common = locale(lang, 'common');
    const codes = sorted(ITEMS.map((i) => i.code));
    expect(codes.length).toBeGreaterThan(0);
    expect(codes.filter((c) => typeof common.errors?.[c] !== 'string')).toEqual([]);
  });

  it('says in Details which formats a hexagonal scan is saved in, and no text says it cannot be saved', () => {
    const detail = d.apexDialog.detail;
    // the formats a hand-written list of the export routes names: .ang and Orienta .h5 as measured,
    // .ctf and the square .ang resampled
    for (const format of ['.ang', '.h5', '.ctf']) expect(detail.hexSaving, format).toContain(format);
    // everything that said "cannot be saved" is gone: its keys, and the category of the limits list
    expect(detail.hexSavingNo).toBeUndefined();
    expect(d.apexDialog.summary.hexNoSave).toBeUndefined();
    expect(d.apexDialog.hexRowSave).toBeUndefined();
    expect(d.apexDialog.hexLimits.categories.export).toBeUndefined();
    expect(d.apexDialog.hexLimits.keys.result_export).toBeUndefined();
    expect(detail.hexEffect_refused).toBeUndefined();
    expect(d.apexDialog.hexLimits.works.file_export).toMatch(/\.ang/);
    expect(d.apexDialog.hexLimits.works.file_export).toMatch(/\.h5/);
  });

  it('refuses an .ang in OIM\'s short-row form with a sentence that names what the user can do', () => {
    const common = locale(lang, 'common');
    const text = common.errors.hex_ang_unreadable;
    expect(typeof text).toBe('string');
    expect(text).toContain('.ang');
    expect(text).toContain('.h5');                                   // load the scan's .h5 instead
    expect(text).toMatch(/OIM Analysis/);                            // or convert it there
    expect(common.errors.hex_export_pending).toBeUndefined();        // the export refusal is gone
  });
});

describe('every text key the dialog and the viewer ask for exists', () => {
  // Literal keys only: `t('pair.tooFew', ...)` in the dialog (all under apexDialog), and
  // `t('apexDialog.x')` in the viewer. A plural key exists when its _one/_other pair does.
  const exists = (tree, dotted) => {
    const walk = (node, parts) => {
      if (parts.length === 0) return true;
      if (node == null || typeof node !== 'object') return false;
      if (parts.length === 1) {
        const k = parts[0];
        return typeof node[k] === 'string' || typeof node[`${k}_other`] === 'string';
      }
      return walk(node[parts[0]], parts.slice(1));
    };
    return walk(tree, dotted.split('.'));
  };
  const literalKeys = (src, pattern) => [...src.matchAll(pattern)].map((m) => m[1]);

  const dialog = read('./ApexLoadDialog.jsx');
  const viewer = read('./EBSDViewer.jsx');
  const dialogKeys = literalKeys(dialog, /\bt\(\s*'([A-Za-z0-9_.]+)'/g);
  const viewerKeys = literalKeys(viewer, /\bt\(\s*'(apexDialog\.[A-Za-z0-9_.]+)'/g);

  it('finds the keys it is supposed to check', () => {
    expect(dialogKeys.length).toBeGreaterThan(40);
    expect(viewerKeys.length).toBeGreaterThan(3);
  });

  it.each(LANGS)('%s has every literal key of the dialog and of the viewer', (lang) => {
    const d = locale(lang);
    const missing = [
      ...dialogKeys.filter((k) => !(k.startsWith('errors.') ? exists(d, k) : exists(d.apexDialog, k)))
        .map((k) => `dialog: ${k}`),
      ...viewerKeys.filter((k) => !exists(d, k)).map((k) => `viewer: ${k}`),
    ];
    expect(missing).toEqual([]);
  });
});

describe('words the first round of readers misread, or that the app spells another way', () => {
  const texts = (lang) => {
    const out = [];
    const walk = (node, prefix) => {
      for (const [k, v] of Object.entries(node)) {
        if (v && typeof v === 'object') walk(v, `${prefix}${k}.`);
        else out.push([`${prefix}${k}`, String(v)]);
      }
    };
    const d = locale(lang);
    walk(d.apexDialog, 'apexDialog.');
    for (const code of CODES) out.push([`errors.${code}`, d.errors[code]]);
    return out;
  };
  const offenders = (lang, re) => texts(lang).filter(([, v]) => re.test(v)).map(([k]) => k);

  it.each(LANGS)('%s never says a file "names" a measurement (read as the file NAME)', (lang) => {
    const re = { en: /\bnames? (this|the|a|no) measurement|\bname this measurement/i, de: /nennt (diese|die|eine|keine) Messung/i,
      ja: /測定を指/, zh: /指明所属测量|指向本次测量|指向另一次测量/ }[lang];
    expect(offenders(lang, re)).toEqual([]);
  });

  it.each(LANGS)('%s does not call the hexagonal map "sheared" or speak of an assumed "Zeilenversatz"', (lang) => {
    expect(offenders(lang, /sheared|geschert|Zeilenversatz|せん断|剪切/)).toEqual([]);
  });

  it.each(LANGS)('%s does not say that maps of a hexagonal scan are a plain grid (they are drawn as measured)', (lang) => {
    expect(locale(lang).apexDialog.detail.hexMapDrawn).not.toMatch(/plain grid|schlichtes Raster|単純な格子|普通网格|15 ?%/);
  });

  it('German uses the app\'s own words', () => {
    expect(offenders('de', /Pattern Center|Frame-Mittelung|kontrastgedehnt|µm auseinander|Karten und Browser/)).toEqual([]);
  });

  it('Chinese says 标定 for indexing, never 索引', () => {
    expect(offenders('zh', /索引/)).toEqual([]);
  });

  it('Japanese says 指数付け and パターン中心, ビューア, and never インデックス付け or パターンセンター', () => {
    expect(offenders('ja', /インデックス付け|パターンセンター|ビューアー|ブラウザー/)).toEqual([]);
  });

  it.each(['en', 'de'])('%s: the summary of one measurement does not say "all"', (lang) => {
    const d = locale(lang).apexDialog.summary;
    expect(d.allReady_one).not.toMatch(/\ball\b|\balle\b/i);
  });
  it.each(['ja', 'zh'])('%s: the summary of one measurement does not say "all"', (lang) => {
    const d = locale(lang).apexDialog.summary;
    expect(d.allReady_one).not.toMatch(/すべて|全部|均/);
  });

  it.each(LANGS)('%s names the grid in words, never "(rows × cols)", and says "grid" on the chip', (lang) => {
    expect(offenders(lang, /\{\{rows\}\}\s*×\s*\{\{cols\}\}/)).toEqual([]);
    const d = locale(lang).apexDialog;
    const word = { en: /grid$/, de: /Raster$/, ja: /格子$/, zh: /网格$/ }[lang];
    expect(d.hex).toMatch(word);
    expect(d.square).toMatch(word);
  });

  it.each(LANGS)('%s says the file "has no field" for the row shift, never that it is incomplete or "does not record" it', (lang) => {
    const d = locale(lang).apexDialog;
    const field = { en: /no field for it/, de: /kein Feld/, zh: /\u6ca1\u6709\u5bf9\u5e94\u7684\u5b57\u6bb5/, ja: /\u8a18\u9332\u3059\u308b\u9805\u76ee\u304c\u3042\u308a\u307e\u305b\u3093/ }[lang];
    expect(d.hexRowAssumes).toMatch(field);
    expect(d.detail.hexLayout).toMatch(field);
    expect(d.detail.hexLayout).toMatch(/Orienta/);
    expect([d.hexRowAssumes, d.detail.hexLayout].join(' ')).not.toMatch(/does not record|nicht vermerkt|\u6ca1\u6709\u8bb0\u5f55|\u8a18\u9332\u3055\u308c\u3066\u3044\u307e\u305b\u3093/);
  });

  it.each(LANGS)('%s: no heading of the limits list repeats the name of a limit (the list showed them twice)', (lang) => {
    const h = locale(lang).apexDialog.hexLimits;
    const names = Object.values(h.keys);
    const offenders_ = Object.entries(h.categories)
      .flatMap(([cat, heading]) => names.filter((n) => heading.includes(n)).map((n) => `${cat}: ${n}`));
    expect(offenders_).toEqual([]);
    expect(Object.values(h.categories).filter((x) => /[(\uff08]/.test(x))).toEqual([]);   // no member list in brackets
  });

  it.each(LANGS)('%s says who refuses: no "refused" without Orienta as its subject in Details', (lang) => {
    const d = locale(lang).apexDialog.detail;
    // said once, in the sentence above the list; the entries below do not repeat it
    expect(d.hexOff).toMatch(/Orienta/);
    expect(d.hexWhen).toBeUndefined();                 // the option names are a collapsed technical line now
    expect(typeof d.hexTechnicalLine).toBe('string');
    for (const k of ['hexOnlyFor', 'hexSomeSettings']) expect(d[k], k).not.toMatch(/Orienta/);
  });

  it.each(LANGS)('%s: the main button always says what it opens, the instruction is its own hint', (lang) => {
    const d = locale(lang).apexDialog;
    expect(d.loadNone).toBeUndefined();
    expect(typeof d.loadHint).toBe('string');
    expect(d.load_other).toMatch(/\{\{count\}\}/);
  });

  it.each(LANGS)('%s: the footer says what is not taken over from EDAX and that the file is not changed', (lang) => {
    const d = locale(lang).apexDialog;
    expect(d.footer.edaxResults).toBeUndefined();      // it is under the title now
    expect(d.topNote).toMatch({ en: /quality values/, de: /Qualitätswerte/, ja: /品質値/, zh: /质量值/ }[lang]);
    expect(d.topNote).toMatch({ en: /Your file is not changed/, de: /Ihre Datei wird nicht verändert/,
      ja: /変更されません/, zh: /不会被修改/ }[lang]);
    expect(d.topNote).not.toMatch(/not imported|nicht importiert|取り込まれません|不会被导入/);
  });

  it('English and German name the dropped parenthesis of the scan date no more', () => {
    for (const lang of ['en', 'de', 'ja', 'zh']) expect(locale(lang).apexDialog.detail.scanEndedValue).toBe('{{date}}');
  });

  it('English: the count-only warning says plainly what to check', () => {
    const p = locale('en').apexDialog.pair;
    expect(p.candidateWarning).toBe('This file carries no measurement ID; only its pattern count fits. Use it only if you know it was exported from this measurement.');
    expect(p.chosenByCount_other).toMatch(/^matched by pattern count only/);
    expect(p.matchedByCount_other).toMatch(/^matched by pattern count only/);
  });

  it('no singular form of the English dialog puts a plural noun after its number', () => {
    const singular = texts('en').filter(([k]) => /_one$/.test(k));
    expect(singular.length).toBeGreaterThan(10);
    const plural = /\{\{(?:grouped|count)\}\}\s+(?:more\s+)?(?:patterns|points|rows|measurements|pattern files|files)\b/;
    expect(singular.filter(([, v]) => plural.test(v)).map(([k]) => k)).toEqual([]);
  });

  it('English does not say "holds 1 patterns" and has singular forms', () => {
    const p = locale('en').apexDialog.pair;
    expect(p.tooFew_one).toMatch(/holds \{\{grouped\}\} pattern\./);
    expect(p.tooMany_one).toMatch(/holds \{\{grouped\}\} pattern,/);
    expect(p.matchedById_one).toMatch(/pattern, one per point/);
  });
});
