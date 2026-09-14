import { describe, it, expect } from 'vitest';
import {
  LAYER_SOURCES, findLayerDef, layerLabel, buildAddLayerOptions,
} from './layerSources';
import en from '../../locales/en/phasemap.json';
import de from '../../locales/de/phasemap.json';
import ja from '../../locales/ja/phasemap.json';
import zh from '../../locales/zh/phasemap.json';

// The legend sentence, written out here on purpose. It is a guarantee about
// what the colours MEAN, and two ways of getting it wrong have already been
// caught on this layer:
//
//   * CYAN is not a map of what CHANGED. The route marks a whole grain
//     `grain` whenever the decision is "reassign", and a reassignment may
//     re-assert the phase the grain already had (the rim-repair case) --
//     measured on a real run, 64 cyan pixels against 28 changed ones, so
//     "assigned it a NEW phase" overstated by 2.3x.
//   * ORANGE has THREE causes, not two: interior too small to sample,
//     chemistry spread over the limit, and a chemistry tie among phases the
//     pattern did not pick (grain_phase_assignment.py ~476, ~493, ~508).
const LEGEND_EN =
  'How each pixel got its phase: cyan = the grain step decided this grain’s '
  + 'phase and wrote it here — it may be the phase the pixel already had, so '
  + 'cyan is not a count of what changed; orange = left alone, because the '
  + 'chemistry could not decide for the grain (interior too small to measure, '
  + 'composition uneven, or several phases fitting equally well); '
  + 'uncoloured = as indexing left it.';

describe('assignment-source layer registration', () => {
  it('is declared in a LAYER_SOURCES group, so the Add Layer dropdown offers it', () => {
    // PhaseMapPage builds `availableToAdd` by walking every LAYER_SOURCES
    // group, so being in one of them is what makes the layer selectable.
    const groups = Object.entries(LAYER_SOURCES)
      .filter(([, g]) => g.layers.some((l) => l.id === 'assignment-source'))
      .map(([key]) => key);
    expect(groups).toEqual(['diagnostics']);
  });

  it('resolves to the backend kind the route serves', () => {
    const def = findLayerDef('assignment-source');
    expect(def).toMatchObject({
      id: 'assignment-source',
      kind: 'assignment_source',
      source: 'diagnostics',
    });
  });

  it('is not gated behind Compute Diagnostics or Refine R+PC', () => {
    // Provenance comes from the grain-assignment run, not from either of
    // those computes; flagging it would grey the row out forever.
    const def = findLayerDef('assignment-source');
    expect(def.requiresCompute).toBeUndefined();
    expect(def.requiresRefinement).toBeUndefined();
  });

  it('composites normally and near-opaque, like its Phase Check sibling', () => {
    // `multiply` would darken the map underneath instead of marking pixels,
    // and a low opacity would let the phase colours read through the
    // provenance colours -- both turn a categorical layer into a tint.
    expect(findLayerDef('assignment-source')).toMatchObject({
      defaultBlend: 'normal', defaultOpacity: 0.85,
    });
  });
});

describe('buildAddLayerOptions', () => {
  // The "+ Add Layer" dropdown contents. Extracted from PhaseMapPage so the
  // caption can be pinned: a full page mount is not done in this repo, and
  // an unpinned caption reverts to the untranslated literal silently.
  const t = (key) => `translated(${key})`;
  const opts = (over = {}) => buildAddLayerOptions({
    layers: [], dynamicLayers: [], sourceLinked: false, sourceReason: null,
    t, ...over,
  });
  const find = (o, value) => o.find((x) => x.value === value);

  it('captions the provenance layer with its translated name', () => {
    expect(find(opts(), 'assignment-source')).toEqual({
      value: 'assignment-source',
      label: 'Forward Diagnostics: translated(layers.assignmentSource)',
      // group + name are what the picker groups and searches on; the joined
      // label stays for the type dropdown, which shows one flat line.
      group: 'Forward Diagnostics',
      name: 'translated(layers.assignmentSource)',
      disabled: false,
      tip: null,
    });
  });

  it('leaves every other caption on its English literal', () => {
    expect(find(opts(), 'phase').label).toBe('Indexing Result: Phase Map');
    expect(find(opts(), 'phase-margin').label).toBe('Forward Diagnostics: Phase Check');
  });

  it('omits layers already in the stack', () => {
    const o = opts({ layers: [{ id: 'assignment-source' }, { id: 'phase' }] });
    expect(find(o, 'assignment-source')).toBeUndefined();
    expect(find(o, 'phase')).toBeUndefined();
    expect(find(o, 'bc')).toBeDefined();
  });

  it('disables the H5OINA group with its reason until a source is linked', () => {
    LAYER_SOURCES.h5.layers.push({ id: 'se:probe', label: 'SE: probe' });
    try {
      const off = find(opts({ sourceReason: 'no source file linked' }), 'se:probe');
      expect(off).toMatchObject({ disabled: true, tip: 'no source file linked' });
      const on = find(opts({ sourceLinked: true, sourceReason: 'no source file linked' }), 'se:probe');
      expect(on).toMatchObject({ disabled: false, tip: null });
    } finally {
      LAYER_SOURCES.h5.layers.pop();
    }
  });

  it('appends the dynamically discovered layers under their own group', () => {
    const o = opts({ dynamicLayers: [{ id: 'eds:Al', label: 'EDS: Al', groupLabel: 'H5OINA Source' }] });
    expect(find(o, 'eds:Al')).toEqual({
      value: 'eds:Al', label: 'H5OINA Source: EDS: Al',
      group: 'H5OINA Source', name: 'EDS: Al',
      disabled: false, tip: null,
    });
  });

  // Availability: a layer whose data nobody has loaded can only produce an
  // error chip in the stack, so the picker needs to know — and needs to be
  // able to say what is missing.
  it('marks a whole group unavailable with its reason', () => {
    const o = opts({ availability: { result: { ok: false, reason: 'no result' } } });
    expect(find(o, 'phase')).toMatchObject({ disabled: true, tip: 'no result' });
    expect(find(o, 'ci')).toMatchObject({ disabled: true, tip: 'no result' });
    // Groups are independent — EBSD is untouched by a missing result.
    expect(find(o, 'vbse')).toMatchObject({ disabled: false });
  });

  it('gates the derived layers on their own computation, not just the group', () => {
    const o = opts({
      availability: {
        diagnosticsComputed: { ok: false, reason: 'compute first' },
        refinementComputed: { ok: false, reason: 'refine first' },
      },
    });
    expect(find(o, 'forward-ncc')).toMatchObject({ disabled: true, tip: 'compute first' });
    expect(find(o, 'refined-ncc')).toMatchObject({ disabled: true, tip: 'refine first' });
    // Phase Check sits in the diagnostics group but is produced by its own
    // button, so "Compute Diagnostics" must not gate it.
    expect(find(o, 'phase-margin')).toMatchObject({ disabled: false });
  });

  it('names the missing result before the missing computation', () => {
    // Both are unmet; the group one is the thing to fix first.
    const o = opts({
      availability: {
        diagnostics: { ok: false, reason: 'no result' },
        diagnosticsComputed: { ok: false, reason: 'compute first' },
      },
    });
    expect(find(o, 'forward-ncc').tip).toBe('no result');
  });

  it('gates the provenance layer on the run that produces it', () => {
    const o = opts({ availability: { assignmentSource: { ok: false, reason: 'assign first' } } });
    expect(find(o, 'assignment-source')).toMatchObject({ disabled: true, tip: 'assign first' });
    expect(find(o, 'phase-margin')).toMatchObject({ disabled: false });
  });

  it('gates a discovered layer like the group it was discovered under', () => {
    const o = opts({
      dynamicLayers: [{ id: 'eds:Al', label: 'EDS: Al', source: 'h5', groupLabel: 'H5OINA Source' }],
      availability: { h5: { ok: false, reason: 'link a source' } },
    });
    expect(find(o, 'eds:Al')).toMatchObject({ disabled: true, tip: 'link a source' });
  });

  it('claims nothing where no availability was given', () => {
    // An unstated gate must not read as "unavailable" — the old callers pass
    // no availability at all and their options stay exactly as they were.
    const o = opts({ availability: {} });
    expect(o.every((x) => x.disabled === false)).toBe(true);
  });
});

describe('layerLabel', () => {
  const t = (key) => `translated(${key})`;

  it('translates a definition that declares a labelKey', () => {
    expect(layerLabel(findLayerDef('assignment-source'), t))
      .toBe('translated(layers.assignmentSource)');
  });

  it('leaves every other layer on its English literal', () => {
    // Control that must survive: the catalog names have never been
    // translated, and opting one layer in must not change the rest.
    expect(layerLabel(findLayerDef('phase'), t)).toBe('Phase Map');
    expect(layerLabel(findLayerDef('bc'), t)).toBe('Band Contrast');
    expect(layerLabel(findLayerDef('phase-margin'), t)).toBe('Phase Check');
  });

  it('falls back for dynamic layers, which have no catalog entry', () => {
    expect(layerLabel(null, t, 'EDS: Al')).toBe('EDS: Al');
    expect(layerLabel(undefined, t)).toBe('');
  });

  it('returns the literal when no translator is supplied', () => {
    expect(layerLabel(findLayerDef('assignment-source'), undefined))
      .toBe('Assignment Source');
  });
});

const LOCALES = [['en', en], ['de', de], ['ja', ja], ['zh', zh]];

describe('assignment-source legend text', () => {
  it('English is the pinned sentence, naming all three ambiguous causes', () => {
    expect(en.layers.assignmentSourceTip).toBe(LEGEND_EN);
    expect(en.layers.assignmentSourceTip).toMatch(/too small to measure/);
    expect(en.layers.assignmentSourceTip).toMatch(/uneven/);
    expect(en.layers.assignmentSourceTip).toMatch(/equally well/);
  });

  it('English does not sell cyan as a count of changed pixels', () => {
    // The regression that was measured: 64 cyan pixels, 28 changed. A legend
    // saying "assigned it a new phase" overstates the change by 2.3x.
    expect(en.layers.assignmentSourceTip).toMatch(/not a count of what changed/);
    expect(en.layers.assignmentSourceTip).not.toMatch(/new phase/i);
  });

  it('no locale claims cyan means the phase changed', () => {
    // Per-language wording of the same regression -- these are the exact
    // phrases the first draft used in each locale.
    expect(de.layers.assignmentSourceTip).not.toMatch(/neu zugewiesen/);
    expect(ja.layers.assignmentSourceTip).not.toMatch(/新しい相/);
    expect(zh.layers.assignmentSourceTip).not.toMatch(/重新指派/);
  });

  it('every locale STATES the caveat, not just avoids the old phrase', () => {
    // The negative pins above only forbid the first draft's exact words. A
    // re-translation that shortens the string -- the ordinary way a tooltip
    // gets tidied -- would drop the caveat entirely and pass them, silently
    // reopening the 2.3x overstatement in de/ja/zh while English stayed
    // pinned. So assert the caveat is PRESENT in each language, not merely
    // that the old wording is absent.
    expect(de.layers.assignmentSourceTip).toMatch(/zählt also nicht, was sich geändert hat/);
    expect(ja.layers.assignmentSourceTip).toMatch(/変更された数ではない/);
    expect(zh.layers.assignmentSourceTip).toMatch(/不等于已更改的数量/);
  });

  it('all four locales carry both keys, none left on English', () => {
    for (const [name, loc] of [['de', de], ['ja', ja], ['zh', zh]]) {
      expect(loc.layers.assignmentSource, name).toBeTruthy();
      expect(loc.layers.assignmentSourceTip, name).toBeTruthy();
      expect(loc.layers.assignmentSource, name).not.toBe(en.layers.assignmentSource);
      expect(loc.layers.assignmentSourceTip, name).not.toBe(en.layers.assignmentSourceTip);
    }
  });

  it('every locale names three ambiguous causes, not two', () => {
    for (const [name, loc] of LOCALES) {
      const tip = loc.layers.assignmentSourceTip;
      // Every parenthesis in the sentence, because which one carries the
      // causes differs by language: en/de put the cyan caveat behind a dash
      // and only bracket the causes, ja/zh bracket both.
      const groups = [...tip.matchAll(/[(（]([^)）]*)[)）]/g)].map((m) => m[1]);
      expect(groups.length, name).toBeGreaterThan(0);
      // Three items need two separators, in whichever comma the language
      // uses. Checking the shape rather than the wording keeps this honest
      // across languages the author cannot proof-read.
      const widest = Math.max(
        ...groups.map((g) => (g.match(/[,，、]/g) || []).length));
      expect(widest, `${name}: ${JSON.stringify(groups)}`).toBeGreaterThanOrEqual(2);
      expect(tip.length, name).toBeGreaterThan(40);
    }
  });
});
