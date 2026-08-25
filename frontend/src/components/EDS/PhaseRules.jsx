/**
 * Rules that decide which phases may compete for a region.
 *
 * From a user report: EDS ambiguity — features smaller than the interaction
 * volume, so every spot mixes in its surroundings — makes the automatic pick
 * land on the wrong candidate. Measured on a real scan that is true but local:
 * of seven structures, two decide between their top two candidates on under
 * 1 at% of margin while the other five are clear by 1.3 to 3.7. So rules are
 * per phase and opt-in; they settle the close calls and leave the rest alone.
 *
 * A rule means exactly one thing: this phase MAY compete here, or it may not.
 * Not a weight. That is what makes the diagnostic honest — "no candidate
 * passed; nearest was Si.cif, blocked by Si >= 40, measured 25.3" — and a soft
 * factor could not produce it.
 *
 * The fastest path to a rule is not typing but **Rule from this structure**:
 * the inspector already shows a structure's measured composition, so the rows
 * arrive filled in and the user corrects numbers instead of inventing them.
 */
import React, { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button, Label, colors as C, alpha } from '../../theme/components';

/** Elements that take no part in the grouping, so none in a rule either. */
const IGNORED = new Set(['C', 'O']);

/** Composition renormalised over the scored elements, in at%. */
export function renormalise(meanAtPct) {
  const entries = Object.entries(meanAtPct || {})
    .filter(([el]) => !IGNORED.has(el));
  const total = entries.reduce((s, [, v]) => s + Math.max(0, v), 0);
  if (!(total > 0)) return {};
  return Object.fromEntries(entries.map(([el, v]) => [el, Math.max(0, v) / total * 100]));
}

/**
 * Seed a rule from a structure the user already selected.
 *
 * Only the elements that are actually concentrated here get a row: a band on
 * an element that merely sits at the background level constrains nothing and
 * would only make the rule brittle. `enrichment` comes from the inspector, so
 * the same 1.3x the classifier gates on decides what is worth a rule.
 *
 * The band is mean +/- 2*spread, rounded — wide enough to survive the dilution
 * gradient that made the user ask for rules in the first place, and clamped to
 * the measured value so a seeded rule can never fail on the very structure it
 * came from.
 */
export function seedRuleFromStructure(detail, { widen = 2 } = {}) {
  const els = (detail?.elements || []).filter((e) => !IGNORED.has(e.element));
  const interesting = els.filter((e) => (e.enrichment ?? 0) >= 1.3);
  const chosen = interesting.length ? interesting : els.slice(0, 2);
  return {
    phase_key: detail?.cif_filename || '',
    phase_formula: detail?.formula || '',
    elements: chosen.map((e) => {
      const pad = Math.max(1, (e.spread || 0) * widen);
      return {
        element: e.element,
        min_at_pct: Math.max(0, Math.round((e.at_pct - pad) * 10) / 10),
        max_at_pct: Math.min(100, Math.round((e.at_pct + pad) * 10) / 10),
      };
    }),
    ratios: [],
    enrichment: [],
  };
}

/** How many structures a rule set would admit for its phase. Display only. */
export function countMatching(rule, structures) {
  if (!rule) return 0;
  return (structures || []).filter((s) => {
    const comp = renormalise(s.mean_at_pct);
    return (rule.elements || []).every((er) => {
      const v = comp[er.element];
      if (v == null) return false;
      if (er.min_at_pct != null && v < er.min_at_pct) return false;
      if (er.max_at_pct != null && v > er.max_at_pct) return false;
      return true;
    });
  }).length;
}

function NumField({ value, onChange, placeholder, ariaLabel }) {
  return (
    <input
      type="number"
      step="0.1"
      value={value ?? ''}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      style={{
        width: 62, fontSize: '8.5pt', padding: '2px 4px',
        background: 'transparent', color: C.text,
        border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
      }}
    />
  );
}

export default function PhaseRules({
  rules, setRules, structures, allPhases, inspectorDetail, elements,
  onReclassify, busy,
}) {
  const { t } = useTranslation('eds');
  const [selectedKey, setSelectedKey] = useState(null);
  // Half-entered ratio: the numerator, waiting for a denominator.
  const [ratioNum, setRatioNum] = useState('');

  const elementNames = useMemo(
    () => (elements || []).filter((e) => !IGNORED.has(e)),
    [elements],
  );
  const list = rules?.rules || [];
  const active = useMemo(
    () => list.find((r) => r.phase_key === selectedKey) || null,
    [list, selectedKey],
  );

  const upsert = useCallback((rule) => {
    setRules((prev) => {
      const rest = (prev?.rules || []).filter((r) => r.phase_key !== rule.phase_key);
      return { ...(prev || {}), rules: [...rest, rule] };
    });
  }, [setRules]);

  const remove = useCallback((key) => {
    setRules((prev) => ({
      ...(prev || {}),
      rules: (prev?.rules || []).filter((r) => r.phase_key !== key),
    }));
    setSelectedKey((k) => (k === key ? null : k));
  }, [setRules]);

  const seedFromStructure = useCallback(() => {
    if (!inspectorDetail?.cif_filename) return;
    const seeded = seedRuleFromStructure(inspectorDetail);
    upsert(seeded);
    setSelectedKey(seeded.phase_key);
  }, [inspectorDetail, upsert]);

  const patchActive = useCallback((fn) => {
    if (!active) return;
    upsert(fn({ ...active }));
  }, [active, upsert]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ fontSize: '8pt', color: C.textSecondary, lineHeight: 1.45 }}>
        {t('rules.explain')}
      </div>

      {/* The matrix is a property of the SAMPLE, so it is declared once for
          the map rather than per phase. It is inferred when left blank; the
          declaration only matters when the inference would be wrong, and it
          is wrong loudly - the scorer's own docstring warns the choice
          "inverts the whole metric". */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6,
                    flexWrap: 'wrap' }}
           title={t('rules.matrixTooltip')}>
        <Label secondary small>{t('rules.matrix')}</Label>
        <select
          value={(rules?.matrix_elements || [])[0] || ''}
          aria-label={t('rules.matrix')}
          onChange={(e) => {
            // Read the value NOW, not inside the updater. React runs the
            // updater later, and by then this controlled select has been
            // reset to its `value` prop - so a deferred read gets the OLD
            // value and the declaration silently does nothing.
            const picked = e.target.value;
            setRules((prev) => ({
              ...(prev || {}),
              matrix_elements: picked ? [picked] : [],
            }));
          }}
          style={{ fontSize: '8.5pt', padding: '2px 4px',
                   background: 'transparent', color: C.text,
                   border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3 }}
        >
          <option value="">{t('rules.matrixAuto')}</option>
          {elementNames.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <span style={{ fontSize: '7.5pt', color: C.textSecondary }}>
          {t('rules.matrixHint')}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap',
                    alignItems: 'flex-start' }}>
        {/* --- which phases have rules ---------------------------------- */}
        <div style={{ flex: '1 1 220px', minWidth: 0 }}>
          <Label secondary small style={{ display: 'block' }}>
            {t('rules.withRules', { count: list.length })}
          </Label>
          <div style={{ fontSize: '7.5pt', color: C.textSecondary,
                        marginBottom: 3, lineHeight: 1.3 }}>
            {t('rules.withRulesHint')}
          </div>
          <div className="thin-scrollbar"
               style={{ display: 'flex', flexDirection: 'column', gap: 1,
                        maxHeight: 150, overflowY: 'auto' }}>
            {list.length === 0 && (
              <Label secondary small>{t('rules.noneYet')}</Label>
            )}
            {list.map((r) => {
              const n = countMatching(r, structures);
              return (
                <div
                  key={r.phase_key}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedKey(r.phase_key)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault(); setSelectedKey(r.phase_key);
                    }
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '2px 4px', borderRadius: 3, cursor: 'pointer',
                    fontSize: '8.5pt', color: C.text,
                    background: r.phase_key === selectedKey
                      ? alpha(C.cyan, 18) : 'transparent',
                  }}
                >
                  <span style={{ flex: 1, minWidth: 0, overflow: 'hidden',
                                 textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.phase_key}
                  </span>
                  {/* A phase whose rules match nothing is the failure a user
                      cannot see otherwise, so it is called out rather than
                      silently absent from the map. */}
                  <span style={{ fontSize: '8pt', flexShrink: 0,
                                 color: n === 0 ? (C.orange || '#f0b429')
                                                : C.textSecondary }}>
                    {n === 0 ? t('rules.matchesNone')
                             : t('rules.matchesN', { count: n })}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); remove(r.phase_key); }}
                    title={t('rules.removeTooltip', { name: r.phase_key })}
                    style={{ background: 'transparent', border: 'none',
                             color: C.red, cursor: 'pointer', fontSize: '9pt',
                             padding: '0 2px' }}
                  >
                    ×
                  </button>
                </div>
              );
            })}
          </div>

          <Button
            variant="secondary"
            disabled={!inspectorDetail?.cif_filename || busy}
            onClick={seedFromStructure}
            title={inspectorDetail?.cif_filename
              ? t('rules.seedTooltip', { name: inspectorDetail.cif_filename })
              : t('rules.seedNeedStructure')}
            style={{ width: '100%', marginTop: 4 }}
          >
            {t('rules.seed')}
          </Button>
        </div>

        {/* --- the selected rule ---------------------------------------- */}
        <div style={{ flex: '2 1 320px', minWidth: 0 }}>
          <Label secondary small style={{ display: 'block' }}>
            {active ? t('rules.editing', { name: active.phase_key })
                    : t('rules.pickOne')}
          </Label>
          {active && (
            <>
              <div style={{ fontSize: '7.5pt', color: C.textSecondary,
                            marginBottom: 3, lineHeight: 1.3 }}>
                {t('rules.allMustHold')}
              </div>

              {(active.elements || []).map((er, i) => (
                <div key={`${er.element}-${i}`}
                     style={{ display: 'flex', alignItems: 'center', gap: 6,
                              marginBottom: 2, fontSize: '8.5pt' }}>
                  <span style={{ width: 34, color: C.text }}>{er.element}</span>
                  <NumField
                    value={er.min_at_pct}
                    ariaLabel={t('rules.minFor', { element: er.element })}
                    placeholder={t('rules.min')}
                    onChange={(v) => patchActive((r) => ({
                      ...r,
                      elements: r.elements.map((x, j) =>
                        (j === i ? { ...x, min_at_pct: v } : x)),
                    }))}
                  />
                  <span style={{ color: C.textSecondary }}>–</span>
                  <NumField
                    value={er.max_at_pct}
                    ariaLabel={t('rules.maxFor', { element: er.element })}
                    placeholder={t('rules.max')}
                    onChange={(v) => patchActive((r) => ({
                      ...r,
                      elements: r.elements.map((x, j) =>
                        (j === i ? { ...x, max_at_pct: v } : x)),
                    }))}
                  />
                  <span style={{ color: C.textSecondary }}>at%</span>
                  <button
                    type="button"
                    onClick={() => patchActive((r) => ({
                      ...r, elements: r.elements.filter((_, j) => j !== i),
                    }))}
                    aria-label={t('rules.removeClause')}
                    style={{ background: 'transparent', border: 'none',
                             color: C.red, cursor: 'pointer' }}
                  >
                    ×
                  </button>
                </div>
              ))}

              {(active.ratios || []).map((rr, i) => (
                <div key={`ratio-${i}`}
                     style={{ display: 'flex', alignItems: 'center', gap: 6,
                              marginBottom: 2, fontSize: '8.5pt' }}>
                  <span style={{ width: 34, color: C.text }}>
                    {rr.numerator}:{rr.denominator}
                  </span>
                  <NumField
                    value={rr.min_ratio}
                    ariaLabel={t('rules.minRatio')}
                    placeholder={t('rules.min')}
                    onChange={(v) => patchActive((r) => ({
                      ...r,
                      ratios: r.ratios.map((x, j) => (j === i ? { ...x, min_ratio: v } : x)),
                    }))}
                  />
                  <span style={{ color: C.textSecondary }}>–</span>
                  <NumField
                    value={rr.max_ratio}
                    ariaLabel={t('rules.maxRatio')}
                    placeholder={t('rules.max')}
                    onChange={(v) => patchActive((r) => ({
                      ...r,
                      ratios: r.ratios.map((x, j) => (j === i ? { ...x, max_ratio: v } : x)),
                    }))}
                  />
                  <button
                    type="button"
                    onClick={() => patchActive((r) => ({
                      ...r, ratios: r.ratios.filter((_, j) => j !== i),
                    }))}
                    aria-label={t('rules.removeClause')}
                    style={{ background: 'transparent', border: 'none',
                             color: C.red, cursor: 'pointer' }}
                  >
                    ×
                  </button>
                </div>
              ))}

              {(active.enrichment || []).map((en, i) => (
                <div key={`enrich-${i}`}
                     style={{ display: 'flex', alignItems: 'center', gap: 6,
                              marginBottom: 2, fontSize: '8.5pt' }}>
                  <span style={{ width: 34, color: C.text }}>{en.element}</span>
                  <NumField
                    value={en.min_factor}
                    ariaLabel={t('rules.minEnrichment')}
                    placeholder={t('rules.min')}
                    onChange={(v) => patchActive((r) => ({
                      ...r,
                      enrichment: r.enrichment.map((x, j) =>
                        (j === i ? { ...x, min_factor: v } : x)),
                    }))}
                  />
                  <span style={{ color: C.textSecondary }}>
                    {t('rules.timesBackground')}
                  </span>
                  <button
                    type="button"
                    onClick={() => patchActive((r) => ({
                      ...r, enrichment: r.enrichment.filter((_, j) => j !== i),
                    }))}
                    aria-label={t('rules.removeClause')}
                    style={{ background: 'transparent', border: 'none',
                             color: C.red, cursor: 'pointer' }}
                  >
                    ×
                  </button>
                </div>
              ))}

              <div style={{ display: 'flex', gap: 4, marginTop: 4,
                            flexWrap: 'wrap' }}>
                <select
                  value=""
                  aria-label={t('rules.addElement')}
                  onChange={(e) => {
                    if (!e.target.value) return;
                    const el = e.target.value;
                    patchActive((r) => ({
                      ...r,
                      elements: [...(r.elements || []),
                                 { element: el, min_at_pct: null, max_at_pct: null }],
                    }));
                    e.target.value = '';
                  }}
                  style={{ fontSize: '8.5pt', padding: '2px 4px',
                           background: 'transparent', color: C.text,
                           border: `1px solid ${alpha(C.purple, 25)}`,
                           borderRadius: 3 }}
                >
                  <option value="">{t('rules.addElement')}</option>
                  {elementNames.map((e) => <option key={e} value={e}>{e}</option>)}
                </select>
                {/* A ratio needs two elements, so it cannot be one dropdown
                    like the others. Picking the numerator arms it; picking the
                    denominator creates the row. */}
                <select
                  value={ratioNum}
                  aria-label={t('rules.addRatioNumerator')}
                  onChange={(e) => setRatioNum(e.target.value)}
                  style={{ fontSize: '8.5pt', padding: '2px 4px',
                           background: 'transparent', color: C.text,
                           border: `1px solid ${alpha(C.purple, 25)}`,
                           borderRadius: 3 }}
                >
                  <option value="">{t('rules.addRatio')}</option>
                  {elementNames.map((e) => <option key={e} value={e}>{e}</option>)}
                </select>
                {ratioNum && (
                  <select
                    value=""
                    aria-label={t('rules.addRatioDenominator')}
                    onChange={(e) => {
                      if (!e.target.value) return;
                      const den = e.target.value;
                      patchActive((r) => ({
                        ...r,
                        ratios: [...(r.ratios || []),
                                 { numerator: ratioNum, denominator: den,
                                   min_ratio: null, max_ratio: null }],
                      }));
                      setRatioNum('');
                    }}
                    style={{ fontSize: '8.5pt', padding: '2px 4px',
                             background: 'transparent', color: C.text,
                             border: `1px solid ${alpha(C.cyan, 45)}`,
                             borderRadius: 3 }}
                  >
                    <option value="">{t('rules.ratioPerElement', { element: ratioNum })}</option>
                    {elementNames.filter((e) => e !== ratioNum)
                      .map((e) => <option key={e} value={e}>{e}</option>)}
                  </select>
                )}
                <select
                  value=""
                  aria-label={t('rules.addEnrichment')}
                  onChange={(e) => {
                    if (!e.target.value) return;
                    const el = e.target.value;
                    patchActive((r) => ({
                      ...r,
                      enrichment: [...(r.enrichment || []),
                                   { element: el, min_factor: 2, max_factor: null }],
                    }));
                    e.target.value = '';
                  }}
                  style={{ fontSize: '8.5pt', padding: '2px 4px',
                           background: 'transparent', color: C.text,
                           border: `1px solid ${alpha(C.purple, 25)}`,
                           borderRadius: 3 }}
                >
                  <option value="">{t('rules.addEnrichment')}</option>
                  {elementNames.map((e) => <option key={e} value={e}>{e}</option>)}
                </select>
              </div>
              <div style={{ fontSize: '7.5pt', color: C.textSecondary,
                            marginTop: 3, lineHeight: 1.35 }}>
                {t('rules.enrichmentHint')}
              </div>
            </>
          )}
        </div>
      </div>

      <Button
        variant="primary"
        disabled={busy}
        onClick={onReclassify}
        title={t('rules.applyTooltip')}
      >
        {t('rules.apply')}
      </Button>
    </div>
  );
}
