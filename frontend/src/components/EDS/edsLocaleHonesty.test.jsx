/**
 * Guards on the EDS strings themselves — no rendering, no components.
 *
 * Three of these were shipped defects, each found by a user rather than by a
 * test, and each of a kind a rendering test cannot catch: the string was
 * displayed perfectly, it just was not true.
 *
 *  1. `wtPctTip` advertised "ZAF/Cliff-Lorimer corrected" while the code runs a
 *     standardless Cliff-Lorimer normalisation with no ZAF at all
 *     (`eds_utils.counts_to_weight_pct`), and `EDS.md` says so in two places.
 *     A tooltip is what users actually read, so the UI was the loudest place
 *     the wrong physics was claimed.
 *  2. `tabs.noMapYet` told the user to press "Re-classify" in the one state
 *     where the button reads "Auto-Classify" — `PhaseMapPanel.jsx` picks
 *     `hasMap ? reclassify : autoClassify`, and `noMapYet` is shown precisely
 *     when there is no map.
 *  3. Re-classifying discards hand-given region NAMES while keeping
 *     hand-painted PIXELS (`assign_region_phase` does not lock,
 *     `assign_mask` does). True and deliberate, but it lived only in a
 *     tooltip. The confirmation strings must exist in every language and must
 *     spell the asymmetry out, because the asymmetry is the surprise.
 *
 * The parity check is the load-bearing part: a fix applied to English only is
 * the same defect for three quarters of the users.
 */
import { describe, it, expect } from 'vitest';

import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';

const LOCALES = { en, de, ja, zh };

function flatten(obj, prefix = '', acc = {}) {
  for (const k of Object.keys(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    const v = obj[k];
    if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, key, acc);
    else acc[key] = v;
  }
  return acc;
}

describe('EDS locale key parity', () => {
  it('every language carries exactly the English key set', () => {
    const enKeys = Object.keys(flatten(en)).sort();
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(Object.keys(flatten(bundle)).sort(), `${lang} key set`).toEqual(enKeys);
    }
  });

  it('no string is left as the untranslated English original', () => {
    // Not a translation-quality check — it only catches a locale file that was
    // filled by copying English in, which has happened here before.
    const enFlat = flatten(en);
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      if (lang === 'en') continue;
      const flat = flatten(bundle);
      for (const key of ['mode.wtPctTip', 'mode.atPctTip', 'tabs.noMapYet',
        'phaseMap.confirmReclassify.kept',
        // Prose, not symbols. `scaleUnitPx`/`scaleUnitUm` are deliberately
        // identical everywhere — "px" and "µm" are not words.
        'phaseMap.scaleUnitTooltip', 'phaseMap.scaleNoStep',
        'presets.saveMatrixTooltip', 'presets.authoredAgainstNothing',
        'export.art.definitions']) {
        expect(flat[key], `${lang}.${key}`).not.toBe(enFlat[key]);
      }
    }
  });
});

describe('the quantification tooltips do not overclaim', () => {
  it('no language still advertises a ZAF correction', () => {
    // The Latin "ZAF" is what the four strings all used; each language also
    // gets its own word for "corrected" checked below.
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const flat = flatten(bundle);
      for (const [key, value] of Object.entries(flat)) {
        if (typeof value !== 'string') continue;
        if (!/ZAF/.test(value)) continue;
        // ZAF may only be mentioned to say it is NOT applied.
        const denies = {
          en: /without .*ZAF|no .*ZAF/i,
          de: /ohne .*ZAF/i,
          ja: /ZAF[^。]*(用いない|なし|しない)/,
          zh: /(未|没有|不)[^。]*ZAF/,
        }[lang];
        expect(denies.test(value), `${lang}.${key} mentions ZAF: ${value}`).toBe(true);
      }
    }
  });

  it('the Wt.% tooltip names Cliff-Lorimer and calls the result semi-quantitative', () => {
    const marks = {
      en: [/Cliff-Lorimer/, /[Ss]emi-quantitative/],
      de: [/Cliff-Lorimer/, /[Hh]albquantitativ/],
      ja: [/Cliff-Lorimer/, /半定量/],
      zh: [/Cliff-Lorimer/, /半定量/],
    };
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const re of marks[lang]) {
        expect(re.test(bundle.mode.wtPctTip), `${lang}.mode.wtPctTip`).toBe(true);
      }
    }
  });

  it('the At.% tooltip inherits the caveat rather than dropping it', () => {
    const marks = {
      en: /[Ss]emi-quantitative/,
      de: /[Hh]albquantitativ/,
      ja: /半定量/,
      zh: /半定量/,
    };
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(marks[lang].test(bundle.mode.atPctTip), `${lang}.mode.atPctTip`).toBe(true);
    }
  });
});

describe('the empty state names the button that is on screen', () => {
  it('noMapYet points at the Auto-Classify label, not the Re-classify one', () => {
    // With no map, PhaseMapPanel renders `phaseMap.autoClassify`. The hint and
    // `noMapHintAction` must agree with it, in each language's own wording.
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const auto = bundle.phaseMap.autoClassify;
      expect(bundle.tabs.noMapYet, `${lang}.tabs.noMapYet`).toContain(auto);
      expect(bundle.phaseMap.noMapHintAction, `${lang}.noMapHintAction`).toBe(auto);
      expect(bundle.tabs.noMapYet, `${lang} names the wrong button`)
        .not.toContain(bundle.phaseMap.reclassify);
    }
  });
});

describe('the re-classify confirmation spells out the painted/named asymmetry', () => {
  const FIELDS = ['title', 'bodyOne', 'bodyOther', 'kept', 'confirm', 'cancel'];

  it('every language has all six fields, non-empty', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const c = bundle.phaseMap.confirmReclassify;
      expect(c, `${lang} confirmReclassify`).toBeTruthy();
      for (const f of FIELDS) {
        expect(typeof c[f], `${lang}.${f}`).toBe('string');
        expect(c[f].trim().length, `${lang}.${f} is empty`).toBeGreaterThan(0);
      }
    }
  });

  it('both body forms interpolate the count of names about to be lost', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const c = bundle.phaseMap.confirmReclassify;
      expect(c.bodyOne, `${lang}.bodyOne`).toContain('{{count}}');
      expect(c.bodyOther, `${lang}.bodyOther`).toContain('{{count}}');
    }
  });

  it('the "kept" line says painted pixels survive — the half users do not expect', () => {
    // If this line ever loses the "pixels are kept" half it stops being a
    // warning and becomes a scare: the user would think everything is lost.
    const painted = {
      en: /painted/i,
      de: /gemalte|Malen/,
      ja: /塗/,
      zh: /涂/,
    };
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(painted[lang].test(bundle.phaseMap.confirmReclassify.kept),
        `${lang}.kept must mention painted pixels`).toBe(true);
    }
  });
});

describe('the export list describes the file that is actually written', () => {
  /**
   * `export.art.definitions` promised "every region definition AND PHASE
   * RULE". `_definitions_table` iterates `state.region_defs` and nothing
   * else; the rules are written to `provenance.json` under
   * `classification.rules`. A user who trusted the label would look for the
   * rules in definitions.csv, not find them, and conclude the export lost
   * them — the fourth defect of this kind on this page, and the same shape
   * as the first three: displayed perfectly, simply not true.
   */
  it('names the file, and says where the phase rules really are', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const s = bundle.export.art.definitions;
      expect(s, `${lang} definitions label`).toContain('definitions.csv');
      // Not "rules are in here": where to go and get them.
      expect(s, `${lang} must point at provenance.json`).toContain('provenance.json');
      expect(s, `${lang} must name the key`).toContain('classification.rules');
    }
  });
});

describe('the smoothing width is offered as a length, in every language', () => {
  /**
   * The whole point of `scale_um` is that a pixel count is not portable: the
   * same "5" is a 2.5 um box at a 0.5 um step and a 3.75 um box at a 0.75 um
   * step. If a language drops that explanation the control looks like a
   * second, redundant way of typing the same number.
   */
  const KEYS = [
    'scaleUnit', 'scaleUnitTooltip', 'scaleUnitPx', 'scaleUnitUm',
    'scaleUmLabel', 'scaleUmPlaceholder', 'scalePxValue',
    'scaleResolved', 'scaleResolvedXY', 'scaleResolvedPxOnly',
    'scaleNoStep', 'scaleNotApplicable',
  ];

  it('has every string, non-empty', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const k of KEYS) {
        expect(typeof bundle.phaseMap[k], `${lang}.phaseMap.${k}`).toBe('string');
        expect(bundle.phaseMap[k].trim().length, `${lang}.${k} is empty`)
          .toBeGreaterThan(0);
      }
    }
  });

  it('the tooltip carries the two boxes the same setting produces', () => {
    // The numbers are the argument. A tooltip that says "you can also use
    // micrometres" without them explains nothing.
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const s = bundle.phaseMap.scaleUnitTooltip;
      expect(s, `${lang} scaleUnitTooltip`).toMatch(/2[.,]5/);
      expect(s, `${lang} scaleUnitTooltip`).toMatch(/3[.,]75/);
    }
  });

  it('the resolved readouts interpolate both the pixel count and the length', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(bundle.phaseMap.scaleResolved, `${lang}.scaleResolved`)
        .toContain('{{px}}');
      expect(bundle.phaseMap.scaleResolved, `${lang}.scaleResolved`)
        .toContain('{{um}}');
      // Both edges, because the box is square in pixels and need not be
      // square in microns.
      expect(bundle.phaseMap.scaleResolvedXY, `${lang}.scaleResolvedXY`)
        .toContain('{{x}}');
      expect(bundle.phaseMap.scaleResolvedXY, `${lang}.scaleResolvedXY`)
        .toContain('{{y}}');
    }
  });

  it('the un-honoured request names what was asked for AND what was used', () => {
    // Naming only one of the two is how a fallback reads as a success.
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(bundle.phaseMap.scaleNoStep, `${lang}.scaleNoStep`).toContain('{{um}}');
      expect(bundle.phaseMap.scaleNoStep, `${lang}.scaleNoStep`).toContain('{{px}}');
    }
  });
});

describe('the preset save form records what the compatibility gate needs', () => {
  const KEYS = [
    'saveMaterial', 'saveMaterialPlaceholder', 'saveMaterialTooltip',
    'saveMatrix', 'saveMatrixTooltip', 'saveMatrixNone',
    'authoredAgainst', 'authoredAgainstStep', 'authoredAgainstNothing',
  ];

  it('has every string, non-empty', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const k of KEYS) {
        expect(typeof bundle.presets[k], `${lang}.presets.${k}`).toBe('string');
        expect(bundle.presets[k].trim().length, `${lang}.${k} is empty`)
          .toBeGreaterThan(0);
      }
    }
  });

  it('the authored-against lines interpolate what is being recorded', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(bundle.presets.authoredAgainst, `${lang}.authoredAgainst`)
        .toContain('{{elements}}');
      expect(bundle.presets.authoredAgainstStep, `${lang}.authoredAgainstStep`)
        .toContain('{{elements}}');
      expect(bundle.presets.authoredAgainstStep, `${lang}.authoredAgainstStep`)
        .toContain('{{step}}');
    }
  });
});


/**
 * The re-classify confirmation is wired in `usePhaseMap.handleAutoClassify`,
 * which every one of the three Apply routes calls - the rail, the region
 * definitions and the rules. The two editors' tooltips, though, still carried
 * the pre-confirmation wording: "the regions are rebuilt, so any names you
 * gave them by hand are lost." True of the mechanism, false about the
 * experience, and it is the sentence a user reads before deciding whether to
 * click. Both must now describe what actually happens: you are asked first,
 * and painted pixels survive.
 */
describe('the Apply tooltips describe the guarded behaviour', () => {
  const asks = {
    en: /asked to confirm/i,
    de: /nachgefragt/,
    ja: /\u78ba\u8a8d/,
    zh: /\u786e\u8ba4/,
  };
  const painted = {
    en: /painted/i,
    de: /gemalte/,
    ja: /\u5857/,
    zh: /\u6d82/,
  };

  it('both editors say the confirmation comes first', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const s of [bundle.defs.applyTooltip, bundle.rules.applyTooltip]) {
        expect(asks[lang].test(s), `${lang} must promise the confirmation: ${s}`)
          .toBe(true);
        expect(painted[lang].test(s), `${lang} must keep the painted half: ${s}`)
          .toBe(true);
      }
    }
  });
});

/**
 * "Gates layers below via multiply blend" is the implementation talking. The
 * non-expert tester could not act on it. The label must say what the user
 * gets, not which Canvas composite operation produces it.
 */
describe('Convert to Mask is written for a user, not for a compositor', () => {
  it('no language explains it by naming the blend mode', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const s = bundle.threshold.convertToMaskTooltip;
      expect(/multiply/i.test(s), `${lang} still names the blend mode: ${s}`).toBe(false);
      expect(s.trim().length, `${lang} is empty`).toBeGreaterThan(20);
    }
  });
});

/**
 * The enrichment legend explained 0.77x by deriving it ("the mirror of
 * 1.3x"). The derivation is correct and nobody needs it at the moment they
 * are reading a colour; the two plain sentences before it carry the meaning.
 */
describe('the enrichment legend stops before the arithmetic', () => {
  const reciprocal = {
    en: /mirror|reciprocal/i,
    de: /Kehrwert/,
    ja: /\u9006\u6570/,
    zh: /\u5012\u6570/,
  };

  it('states the two thresholds without deriving the second from the first', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      const s = bundle.inspector.scaleLegend;
      expect(reciprocal[lang].test(s), `${lang} still derives 0.77: ${s}`).toBe(false);
      // The numbers themselves stay - they are what the colours mean.
      // German writes them with a comma, and that is correct German.
      expect(s, `${lang} lost the neutral level`).toMatch(/1[.,]0/);
      expect(s, `${lang} lost the depletion level`).toMatch(/0[.,]77/);
    }
  });
});

/**
 * The pasteable summary line. It ends up in a figure caption, so the caveat
 * clause must survive translation intact in all four languages, and every
 * clause must keep the interpolations it is built from - a clause that lost
 * `{{particles}}` would print a sentence with the number missing.
 */
describe('the export summary line, in every language', () => {
  const KEYS = {
    scan: ['{{name}}', '{{width}}', '{{height}}', '{{step}}'],
    sizeUm: ['{{width}}', '{{height}}', '{{step}}'],
    scanPx: ['{{name}}', '{{cols}}', '{{rows}}'],
    sizePx: ['{{cols}}', '{{rows}}'],
    scanOnly: ['{{name}}'],
    regions: ['{{regions}}', '{{phases}}'],
    particles: ['{{particles}}'],
    particlesFlagged: ['{{particles}}', '{{below}}', '{{limit}}', '{{edge}}'],
    smoothing: ['{{px}}', '{{um}}'],
    smoothingPx: ['{{px}}'],
    app: ['{{app}}'],
    deterministic: ['{{seed}}'],
    semiQuant: [],
  };

  it('every clause exists and keeps its values', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const [key, vars] of Object.entries(KEYS)) {
        const s = bundle.export.sum[key];
        expect(typeof s, `${lang}.export.sum.${key}`).toBe('string');
        for (const v of vars) {
          expect(s, `${lang}.export.sum.${key} lost ${v}`).toContain(v);
        }
      }
    }
  });

  it('the semi-quantitative caveat is translated, not copied', () => {
    const en = LOCALES.en.export.sum.semiQuant;
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      if (lang === 'en') continue;
      expect(bundle.export.sum.semiQuant, `${lang} copied the English`).not.toBe(en);
    }
  });

  it('the pixel-only clauses say the step size is missing rather than implying zero', () => {
    const missing = {
      en: /no step size/i,
      de: /keine Schrittweite/,
      ja: /\u30b9\u30c6\u30c3\u30d7\u30b5\u30a4\u30ba\u306e\u8a18\u9332\u306a\u3057/,
      zh: /\u672a\u8bb0\u5f55\u6b65\u957f/,
    };
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const key of ['scanPx', 'sizePx']) {
        expect(missing[lang].test(bundle.export.sum[key]),
          `${lang}.export.sum.${key}: ${bundle.export.sum[key]}`).toBe(true);
      }
    }
  });
});

/**
 * The two panel strings that replaced the inert slider. Both are prose a
 * non-expert has to be able to act on, so both must actually be translated.
 */
describe('the min-score explanation and the tolerance note', () => {
  it('exist, are non-empty and are not the English text', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const k of ['minScoreNote', 'toleranceRemoved']) {
        expect(typeof bundle.phaseMap[k], `${lang}.phaseMap.${k}`).toBe('string');
        expect(bundle.phaseMap[k].trim().length, `${lang}.${k} is empty`)
          .toBeGreaterThan(20);
        if (lang !== 'en') {
          expect(bundle.phaseMap[k], `${lang}.${k} copied the English`)
            .not.toBe(LOCALES.en.phaseMap[k]);
        }
      }
    }
  });

  it('the tolerance keys are KEPT — the backend still records the value', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(typeof bundle.phaseMap.tolerance, `${lang}.phaseMap.tolerance`).toBe('string');
      expect(typeof bundle.phaseMap.toleranceInert, `${lang}.toleranceInert`).toBe('string');
    }
  });
});

/**
 * The preset picker's own strings. A filter whose label did not translate is
 * a filter three quarters of the users cannot use.
 */
describe('the preset picker strings', () => {
  const KEYS = [
    'pickTooltip', 'search', 'searchPlaceholder', 'noMatches', 'filterMaterial',
    'allMaterials', 'noMaterial', 'tags', 'tagFilterTooltip', 'countShown',
    'forMaterial', 'noNotes',
  ];

  it('exist in every language, non-empty', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      for (const k of KEYS) {
        expect(typeof bundle.presets[k], `${lang}.presets.${k}`).toBe('string');
        expect(bundle.presets[k].trim().length, `${lang}.${k} is empty`).toBeGreaterThan(0);
      }
    }
  });

  it('the two interpolated ones keep their values', () => {
    for (const [lang, bundle] of Object.entries(LOCALES)) {
      expect(bundle.presets.countShown, `${lang}.countShown`).toContain('{{shown}}');
      expect(bundle.presets.countShown, `${lang}.countShown`).toContain('{{total}}');
      expect(bundle.presets.forMaterial, `${lang}.forMaterial`).toContain('{{material}}');
    }
  });
});
