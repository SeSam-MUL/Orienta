/**
 * Declare the regions yourself, instead of letting the fit decide.
 *
 * A user reported the case this exists for: "pure Silicon and AlFeMnSi phases
 * are not possible to separate from each other because the automatic
 * definition of regions based on composition lumps them into the same
 * category. The high amount of Al background signal is likely the issue here."
 *
 * That is a real property of the feature space, not a bad fit. The grouping
 * measures distance over the whole composition, so an ~85 at% aluminium
 * matrix dominates it and the few at% of iron that actually tells the two
 * apart is a rounding error next to it. No cluster count fixes that, because
 * a count cannot say WHICH element carries the distinction.
 *
 * Two answers live here, and they are different tools:
 *
 *   - Element weights keep the grouping automatic and tell it what to care
 *     about. Cheap to try, and it does not always help: measured on SampleB,
 *     weighting Fe/Mn/Si by 3 moved the map from 7 regions to 8 and left
 *     coherence at 0.933 — a small change, not a rescue.
 *   - A definition removes the question from the fit. You state the
 *     composition window and the pixels inside it are a region. Measured on
 *     the same map, "Si >= 20 at% and Fe <= 1" collected 202 px into ONE
 *     region where the automatic grouping had cut the same particle into
 *     three concentric rings at Si 25 / 38 / 56 at%.
 *
 * The clause vocabulary is deliberately identical to PhaseRules: content
 * ranges, ratio ranges, enrichment. One editor, one semantics, one set of
 * surprises. The difference is what the answer is used for — a rule gates a
 * phase's eligibility, a definition claims pixels.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { edsApi } from '../../services/api';
import { Label, colors as C, alpha, spacing } from '../../theme/components';

const NUM = { width: 52, fontSize: '8.5pt', padding: '1px 3px' };

/**
 * Why a window caught nothing, in the reader's language.
 *
 * The backend answers in English prose AND as a code. Codes it knows are
 * translated; anything else falls back to the prose, because a reason the
 * frontend has not learned yet must still say something. The `rule` code
 * deliberately has no translation: that text names an element and comes from
 * the clause evaluator, so the sentence IS the answer.
 */
export function reasonText(t, pv) {
  const code = pv?.reason_code;
  if (code && code !== 'rule') {
    const key = `defs.reason.${code}`;
    const s = t(key);
    if (s && s !== key) return s;
  }
  return pv?.reason || t('defs.previewNone');
}

/** Does this clause actually constrain anything? */
function binds(c) {
  return ['min_at_pct', 'max_at_pct', 'min_ratio', 'max_ratio',
    'min_factor', 'max_factor'].some((k) => c?.[k] != null);
}

/**
 * A definition claims nothing unless at least one clause BITES.
 *
 * Counting clauses was not enough: "Add" creates a clause with both bounds
 * unset, which made the definition look non-empty while constraining
 * nothing — so it matched every pixel and one definition ate the whole map,
 * with the warning chip suppressed because the definition was "not empty".
 * Mirrors `phase_rules._binds` on the backend.
 */
export function isEmptyDef(d) {
  return ![...(d?.elements || []), ...(d?.ratios || []),
    ...(d?.enrichment || [])].some(binds);
}

/**
 * Seed a definition from a region the user picked on the map.
 *
 * The window is the region's own spread, widened to +-2 sigma, and only over
 * the elements that are actually enriched there. Seeding from every element
 * would write eight clauses of which six say "and the matrix is still the
 * matrix" — true, useless, and they make the window fragile the moment the
 * neighbouring chemistry shifts.
 */
export function seedDefFromRegion(detail, minEnrichment = 1.3) {
  if (!detail) return null;
  // `seed` is the region measured the way a window will be READ - on the
  // smoothed composition the classifier evaluates against. The displayed
  // composition is the raw measurement, and on a small feature the two
  // differ enough that a window seeded from the raw numbers does not
  // reproduce its own region (measured: 40.97 at% raw vs 36.88 smoothed).
  const rows = (detail.seed?.length ? detail.seed : (detail.composition || []))
    .filter(
    (r) => (r.enrichment == null ? true : r.enrichment >= minEnrichment),
  );
  const elements = rows.map((r) => {
    const spread = Number(r.spread_at_pct) || 0;
    const mean = Number(r.at_pct) || 0;
    return {
      element: r.element,
      min_at_pct: Math.max(0, +(mean - 2 * spread).toFixed(2)),
      max_at_pct: +(mean + 2 * spread).toFixed(2),
    };
  });
  return {
    name: detail.cif_filename || `Region ${detail.region_id}`,
    phase_key: '',
    elements,
    ratios: [],
    enrichment: [],
  };
}

function Clause({ children, onRemove, title }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4,
                  fontSize: '8.5pt', color: C.text }} title={title}>
      {children}
      <button type="button" onClick={onRemove}
              style={{ marginLeft: 'auto', cursor: 'pointer', border: 'none',
                       background: 'transparent', color: C.textSecondary,
                       fontSize: '10pt', lineHeight: 1, padding: '0 2px' }}>
        ×
      </button>
    </div>
  );
}

export default function RegionDefs({
  defs, setDefs, elements = [], allPhases = [], inspectorDetail,
  clusterRemainder, setClusterRemainder, scale,
  elementWeights, setElementWeights,
  onReclassify, busy,
  // Arming the map for a pixel pick, and handing the preview overlay up so
  // the map can draw it. Both live on the page, because the map does.
  pixelPick, onArmPixelPick, onOverlay, pixelArmed,
}) {
  const { t } = useTranslation('eds');
  const [preview, setPreview] = useState(null);
  const [previewErr, setPreviewErr] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [newEl, setNewEl] = useState('');
  const [ratioNum, setRatioNum] = useState('');
  const [openIdx, setOpenIdx] = useState(0);

  const list = defs || [];

  // A definition can also arrive from outside — the region inspector's
  // "define this region" button adds one and switches to this tab. Landing
  // on a collapsed new row while a different one sits open reads as though
  // nothing happened, so whatever was just appended opens.
  const prevLen = useRef(list.length);
  useEffect(() => {
    if (list.length > prevLen.current) setOpenIdx(list.length - 1);
    prevLen.current = list.length;
  }, [list.length]);

  const patch = useCallback((i, next) => {
    setDefs(list.map((d, j) => (j === i ? { ...d, ...next } : d)));
    // The preview describes the windows as they were when it ran. Leaving it
    // on screen after an edit makes it describe the PREVIOUS window while
    // looking current — the one readout whose entire job is "what does this
    // window claim" quietly answering about a different one.
    setPreview(null);
  }, [list, setDefs]);

  // A pixel picked on the map turns into a window. Guarded on the arming
  // flag so an ordinary click keeps doing what it always did.
  const seedFromPixel = useCallback(async (row, col) => {
    try {
      const res = await edsApi.seedRegionDefFromPixel({ row, col, scale });
      const d = res.data?.definition;
      if (d?.elements?.length) {
        setDefs([...(defs || []), d]);
      } else {
        // Nothing is concentrated there. Saying so beats adding an empty
        // window the user then has to work out is empty.
        setPreviewErr(t('defs.seedPixelNothing', { row, col }));
      }
    } catch (err) {
      setPreviewErr(err.response?.data?.detail || err.message);
    }
  }, [defs, setDefs, scale, t]);

  useEffect(() => {
    if (pixelPick) seedFromPixel(pixelPick.row, pixelPick.col);
    // seedFromPixel is intentionally out of the deps: it changes on every
    // edit, and re-seeding on an unrelated edit would add duplicates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pixelPick]);

  const addDef = useCallback((seed) => {
    const d = seed || {
      name: t('defs.newName', { n: list.length + 1 }),
      phase_key: '', elements: [], ratios: [], enrichment: [],
    };
    setDefs([...list, d]);
    setOpenIdx(list.length);
    setPreview(null);
  }, [list, setDefs, t]);

  const removeDef = useCallback((i) => {
    setDefs(list.filter((_, j) => j !== i));
    setPreview(null);
  }, [list, setDefs]);

  const move = useCallback((i, delta) => {
    const j = i + delta;
    if (j < 0 || j >= list.length) return;
    const next = [...list];
    [next[i], next[j]] = [next[j], next[i]];
    setDefs(next);
    setOpenIdx(j);
    setPreview(null);
  }, [list, setDefs]);

  // Whatever the last preview claimed, drawn over the map. Cleared
  // whenever the windows change, so the picture cannot outlive the
  // thresholds it describes - the same rule as the per-row counts.
  useEffect(() => {
    onOverlay?.(preview?.overlay || null);
  }, [preview, onOverlay]);

  useEffect(() => () => onOverlay?.(null), [onOverlay]);

  const runPreview = useCallback(async () => {
    setPreviewing(true); setPreviewErr(null);
    try {
      // The smoothing MUST travel with the request. Without it the
      // preview silently measures on the module default while the run uses
      // whatever the Scale slider says, and the two disagree by exactly the
      // amount the user just changed — while the endpoint's own docstring
      // promises they cannot.
      const res = await edsApi.previewRegionDefs({ regionDefs: list, scale });
      setPreview(res.data);
    } catch (err) {
      setPreviewErr(err.response?.data?.detail || err.message);
    } finally {
      setPreviewing(false);
    }
  }, [list, scale]);

  const weights = elementWeights || {};
  const weighted = useMemo(
    () => Object.entries(weights).filter(([, v]) => Number(v) !== 1),
    [weights],
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column',
                  gap: spacing.innerSpacing }}>
      <Label secondary small style={{ display: 'block' }}>
        {t('defs.explain')}
      </Label>

      {/* --- element weights: the cheap lever, offered first --------------- */}
      <details>
        <summary style={{ cursor: 'pointer', fontSize: '9pt', color: C.text }}>
          {t('defs.weightsTitle')}
          {weighted.length > 0 && (
            <span style={{ color: C.cyan, marginLeft: 6, fontSize: '8pt' }}>
              {t('defs.weightsActive', { count: weighted.length })}
            </span>
          )}
        </summary>
        <Label secondary small style={{ display: 'block', margin: '3px 0' }}>
          {t('defs.weightsExplain')}
        </Label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {elements.map((el) => {
            const v = weights[el] == null ? 1 : Number(weights[el]);
            return (
              <div key={el} style={{ display: 'flex', alignItems: 'center',
                                     gap: 6, fontSize: '8.5pt' }}>
                <span style={{ width: 28, color: C.text }}>{el}</span>
                <input
                  type="range" min={0} max={10} step={0.5} value={v}
                  onChange={(e) => setElementWeights({
                    ...weights, [el]: Number(e.target.value),
                  })}
                  style={{ flex: 1 }}
                />
                <span style={{ width: 30, textAlign: 'right',
                               color: v === 1 ? C.textSecondary : C.cyan,
                               fontVariantNumeric: 'tabular-nums' }}>
                  ×{v}
                </span>
              </div>
            );
          })}
        </div>
        <Label secondary small style={{ display: 'block', marginTop: 2 }}>
          {t('defs.weightZero')}
        </Label>
        {weighted.length > 0 && (
          <button type="button" onClick={() => setElementWeights({})}
                  style={{ marginTop: 4, fontSize: '8.5pt', cursor: 'pointer',
                           padding: '2px 6px', borderRadius: 3,
                           background: 'transparent', color: C.textSecondary,
                           border: `1px solid ${C.border}` }}>
            {t('defs.weightsReset')}
          </button>
        )}
      </details>

      {/* --- the definitions ---------------------------------------------- */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
        <span style={{ fontSize: '9pt', color: C.text }}>
          {t('defs.listTitle', { count: list.length })}
        </span>
        <button type="button" onClick={() => addDef()} disabled={busy}
                title={t('defs.addTooltip')}
                style={{ marginLeft: 'auto', fontSize: '8.5pt',
                         cursor: busy ? 'default' : 'pointer',
                         padding: '2px 8px', borderRadius: 3,
                         background: alpha(C.cyan, 15), color: C.text,
                         border: `1px solid ${alpha(C.cyan, 45)}` }}>
          {t('defs.add')}
        </button>
        <button
          type="button"
          onClick={() => onArmPixelPick?.(!pixelArmed)}
          disabled={busy || !onArmPixelPick}
          title={t('defs.fromPixelTooltip')}
          style={{ fontSize: '8.5pt',
                   cursor: onArmPixelPick ? 'pointer' : 'default',
                   padding: '2px 8px', borderRadius: 3,
                   background: pixelArmed ? alpha(C.cyan, 30) : 'transparent',
                   color: C.text,
                   border: `1px solid ${pixelArmed ? alpha(C.cyan, 60)
                                                   : C.border}` }}>
          {pixelArmed ? t('defs.fromPixelArmed') : t('defs.fromPixel')}
        </button>
        <button
          type="button"
          onClick={() => addDef(seedDefFromRegion(inspectorDetail))}
          disabled={busy || !inspectorDetail}
          title={inspectorDetail ? t('defs.seedTooltip')
            : t('defs.seedDisabledTooltip')}
          style={{ fontSize: '8.5pt',
                   cursor: (busy || !inspectorDetail) ? 'default' : 'pointer',
                   padding: '2px 8px', borderRadius: 3,
                   opacity: inspectorDetail ? 1 : 0.45,
                   background: 'transparent', color: C.text,
                   border: `1px solid ${C.border}` }}>
          {t('defs.seed')}
        </button>
      </div>

      {list.length === 0 && (
        <div style={{ padding: '5px 6px', borderRadius: 4,
                      border: `1px solid ${alpha(C.cyan, 30)}` }}>
          <Label secondary small style={{ display: 'block' }}>
            {t('defs.none')}
          </Label>
          <Label secondary small style={{ display: 'block', marginTop: 3 }}>
            {t('defs.firstMove')}
          </Label>
        </div>
      )}

      {list.map((d, i) => {
        const open = openIdx === i;
        const pv = preview?.definitions?.[i];
        return (
          <div key={i} style={{ border: `1px solid ${C.border}`,
                                borderRadius: 4, padding: '4px 6px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <button type="button" onClick={() => setOpenIdx(open ? -1 : i)}
                      style={{ border: 'none', background: 'transparent',
                               color: C.textSecondary, cursor: 'pointer',
                               fontSize: '9pt', padding: 0, width: 12 }}>
                {open ? '▾' : '▸'}
              </button>
              <input
                value={d.name || ''}
                onChange={(e) => patch(i, { name: e.target.value })}
                style={{ flex: 1, minWidth: 0, fontSize: '9pt',
                         background: 'transparent', color: C.text,
                         border: 'none', borderBottom: `1px solid ${C.border}` }}
              />
              {/* Order is priority and the only conflict rule there is, so it
                  has to be movable and visible. */}
              <button type="button" onClick={() => move(i, -1)} disabled={i === 0}
                      title={t('defs.moveUpTooltip')}
                      style={{ border: 'none', background: 'transparent',
                               color: C.textSecondary, fontSize: '9pt',
                               cursor: i === 0 ? 'default' : 'pointer',
                               opacity: i === 0 ? 0.3 : 1, padding: '0 2px' }}>
                ↑
              </button>
              <button type="button" onClick={() => move(i, 1)}
                      disabled={i === list.length - 1}
                      title={t('defs.moveDownTooltip')}
                      style={{ border: 'none', background: 'transparent',
                               color: C.textSecondary, fontSize: '9pt',
                               cursor: i === list.length - 1 ? 'default' : 'pointer',
                               opacity: i === list.length - 1 ? 0.3 : 1,
                               padding: '0 2px' }}>
                ↓
              </button>
              <button type="button" onClick={() => removeDef(i)}
                      style={{ border: 'none', background: 'transparent',
                               color: C.textSecondary, cursor: 'pointer',
                               fontSize: '10pt', padding: '0 2px' }}>
                ×
              </button>
            </div>

            {/* What this window actually caught, from the real map. */}
            {pv && (
              <div style={{ fontSize: '8pt', marginTop: 2,
                            color: pv.n_pixels ? C.cyan : (C.orange || '#f0b429') }}
                   title={t('defs.previewRowTooltip')}>
                {pv.n_pixels
                  ? t('defs.previewHit', { px: pv.n_pixels.toLocaleString(),
                                           pct: pv.percentage })
                  : reasonText(t, pv)}
                {pv.overlap_pixels > 0 && (
                  <span style={{ color: C.textSecondary, marginLeft: 6 }}>
                    {t('defs.previewOverlap', { px: pv.overlap_pixels })}
                  </span>
                )}
              </div>
            )}
            {/* The count alone cannot tell you whether you caught the RIGHT
                pixels — 202 px of matrix and 202 px of silicon particle read
                identically. The backend already measures and sends this;
                dropping it left the one number that answers the question
                on the floor. */}
            {pv?.mean_at_pct && Object.keys(pv.mean_at_pct).length > 0 && (
              <div style={{ fontSize: '8pt', color: C.textSecondary,
                            marginTop: 1 }}
                   title={t('defs.previewMeanTooltip')}>
                {Object.entries(pv.mean_at_pct).slice(0, 5)
                  .map(([el, v]) => `${el} ${v}`).join(' · ')} at%
              </div>
            )}
            {isEmptyDef(d) && (
              <div style={{ fontSize: '8pt', marginTop: 2,
                            color: C.orange || '#f0b429' }}>
                {t('defs.emptyWarning')}
              </div>
            )}

            {open && (
              <div style={{ marginTop: 4, display: 'flex',
                            flexDirection: 'column', gap: 2 }}>
                {(d.elements || []).map((er, j) => (
                  <Clause key={`e${j}`} title={t('defs.elementTooltip')}
                          onRemove={() => patch(i, {
                            elements: d.elements.filter((_, x) => x !== j),
                          })}>
                    <span style={{ width: 28 }}>{er.element}</span>
                    <input type="number" style={NUM} value={er.min_at_pct ?? ''}
                           placeholder={t('defs.min')}
                           onChange={(e) => patch(i, {
                             elements: d.elements.map((x, y) => (y === j
                               ? { ...x, min_at_pct: e.target.value === '' ? null
                                 : Number(e.target.value) } : x)),
                           })} />
                    <span style={{ color: C.textSecondary }}>–</span>
                    <input type="number" style={NUM} value={er.max_at_pct ?? ''}
                           placeholder={t('defs.max')}
                           onChange={(e) => patch(i, {
                             elements: d.elements.map((x, y) => (y === j
                               ? { ...x, max_at_pct: e.target.value === '' ? null
                                 : Number(e.target.value) } : x)),
                           })} />
                    <span style={{ color: C.textSecondary }}>at%</span>
                  </Clause>
                ))}

                {/* A ratio says "this much Fe for this much Si", which
                    survives a change in matrix level that an absolute
                    threshold does not. */}
                {(d.ratios || []).map((rr, j) => (
                  <Clause key={`r${j}`} title={t('defs.ratioTooltip')}
                          onRemove={() => patch(i, {
                            ratios: d.ratios.filter((_, x) => x !== j),
                          })}>
                    <span style={{ width: 40 }}>
                      {rr.numerator}:{rr.denominator}
                    </span>
                    <input type="number" style={NUM} value={rr.min_ratio ?? ''}
                           placeholder={t('defs.min')}
                           onChange={(e) => patch(i, {
                             ratios: d.ratios.map((x, y) => (y === j
                               ? { ...x, min_ratio: e.target.value === '' ? null
                                 : Number(e.target.value) } : x)),
                           })} />
                    <span style={{ color: C.textSecondary }}>-</span>
                    <input type="number" style={NUM} value={rr.max_ratio ?? ''}
                           placeholder={t('defs.max')}
                           onChange={(e) => patch(i, {
                             ratios: d.ratios.map((x, y) => (y === j
                               ? { ...x, max_ratio: e.target.value === '' ? null
                                 : Number(e.target.value) } : x)),
                           })} />
                  </Clause>
                ))}

                {/* Enrichment measures against THIS map's own background, so
                    the factor travels between datasets where an absolute at%
                    does not. Measured on a real particle: "Si >= 2x
                    background" held 96.3% of it while letting 5.3% of the map
                    through; "Si >= 40 at%" held 46.3% at 0.9%. */}
                {(d.enrichment || []).map((en, j) => (
                  <Clause key={`x${j}`} title={t('defs.enrichmentTooltip')}
                          onRemove={() => patch(i, {
                            enrichment: d.enrichment.filter((_, x) => x !== j),
                          })}>
                    <span style={{ width: 28 }}>{en.element}</span>
                    <span style={{ color: C.textSecondary }}>&ge;</span>
                    <input type="number" style={NUM} value={en.min_factor ?? ''}
                           placeholder={t('defs.min')}
                           onChange={(e) => patch(i, {
                             enrichment: d.enrichment.map((x, y) => (y === j
                               ? { ...x, min_factor: e.target.value === '' ? null
                                 : Number(e.target.value) } : x)),
                           })} />
                    <span style={{ color: C.textSecondary }}>
                      {t('defs.timesBackground')}
                    </span>
                  </Clause>
                ))}

                <div style={{ display: 'flex', gap: 4, alignItems: 'center',
                              marginTop: 2 }}>
                  <select value={newEl} onChange={(e) => setNewEl(e.target.value)}
                          style={{ fontSize: '8.5pt', flex: 1, minWidth: 0 }}>
                    <option value="">{t('defs.pickElement')}</option>
                    {elements.map((el) => (
                      <option key={el} value={el}>{el}</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={!newEl}
                    onClick={() => {
                      // Read the pick NOW. Inside the updater the controlled
                      // select has already been reset, and the clause would
                      // arrive with an empty element — the same trap the
                      // matrix dropdown fell into.
                      const el = newEl;
                      patch(i, {
                        elements: [...(d.elements || []),
                          { element: el, min_at_pct: null, max_at_pct: null }],
                      });
                      setNewEl('');
                    }}
                    style={{ fontSize: '8.5pt', padding: '1px 8px',
                             borderRadius: 3, cursor: newEl ? 'pointer' : 'default',
                             opacity: newEl ? 1 : 0.45,
                             background: 'transparent', color: C.text,
                             border: `1px solid ${C.border}` }}>
                    {t('defs.addClause')}
                  </button>
                </div>

                <div style={{ display: 'flex', gap: 4, alignItems: 'center',
                              flexWrap: 'wrap' }}>
                  <select
                    value={ratioNum}
                    aria-label={t('defs.addRatio')}
                    onChange={(e) => setRatioNum(e.target.value)}
                    style={{ fontSize: '8.5pt', flex: 1, minWidth: 0 }}
                  >
                    <option value="">{t('defs.addRatio')}</option>
                    {elements.map((el) => (
                      <option key={el} value={el}>{el}</option>
                    ))}
                  </select>
                  {ratioNum && (
                    <select
                      value=""
                      aria-label={t('defs.addRatioDenominator')}
                      onChange={(e) => {
                        // Read before the updater runs: the controlled select
                        // is reset by then and the clause would arrive with
                        // no denominator in it.
                        const den = e.target.value;
                        if (!den) return;
                        patch(i, {
                          ratios: [...(d.ratios || []),
                            { numerator: ratioNum, denominator: den,
                              min_ratio: null, max_ratio: null }],
                        });
                        setRatioNum('');
                      }}
                      style={{ fontSize: '8.5pt', flex: 1, minWidth: 0 }}
                    >
                      <option value="">
                        {t('defs.ratioPerElement', { element: ratioNum })}
                      </option>
                      {elements.filter((el) => el !== ratioNum).map((el) => (
                        <option key={el} value={el}>{el}</option>
                      ))}
                    </select>
                  )}
                  <select
                    value=""
                    aria-label={t('defs.addEnrichment')}
                    onChange={(e) => {
                      const el = e.target.value;
                      if (!el) return;
                      patch(i, {
                        enrichment: [...(d.enrichment || []),
                          { element: el, min_factor: 2, max_factor: null }],
                      });
                      e.target.value = '';
                    }}
                    style={{ fontSize: '8.5pt', flex: 1, minWidth: 0 }}
                  >
                    <option value="">{t('defs.addEnrichment')}</option>
                    {elements.map((el) => (
                      <option key={el} value={el}>{el}</option>
                    ))}
                  </select>
                </div>
                <Label secondary small style={{ display: 'block' }}>
                  {t('defs.clauseHint')}
                </Label>

                {/* Naming is optional: empty means "group these pixels, then
                    let the library matcher name them as usual". */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 4,
                              marginTop: 3, fontSize: '8.5pt' }}>
                  <span style={{ color: C.textSecondary }}>
                    {t('defs.phaseLabel')}
                  </span>
                  <select
                    value={d.phase_key || ''}
                    onChange={(e) => {
                      const picked = e.target.value;
                      patch(i, { phase_key: picked });
                    }}
                    style={{ flex: 1, minWidth: 0, fontSize: '8.5pt' }}
                  >
                    <option value="">{t('defs.phaseAuto')}</option>
                    {allPhases.map((p) => (
                      <option key={p.key || p.cif_filename}
                              value={p.key || p.cif_filename}>
                        {p.cif_filename}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {/* --- what happens to everything undeclared ------------------------- */}
      {list.length > 0 && (
        <label style={{ display: 'flex', alignItems: 'center', gap: 6,
                        fontSize: '8.5pt', color: C.text }}
               title={t('defs.remainderTooltip')}>
          <input type="checkbox" checked={!!clusterRemainder}
                 onChange={(e) => setClusterRemainder(e.target.checked)} />
          {t('defs.remainder')}
        </label>
      )}

      {previewErr && (
        <div style={{ fontSize: '8pt', color: C.red || '#ff526f' }}>
          {previewErr}
        </div>
      )}
      {preview && (
        <Label secondary small style={{ display: 'block' }}>
          {t('defs.previewSummary', {
            px: preview.unclaimed_pixels.toLocaleString(),
            pct: preview.unclaimed_percentage,
          })}
        </Label>
      )}

      <div style={{ display: 'flex', gap: 4 }}>
        <button type="button" onClick={runPreview}
                disabled={busy || previewing || list.length === 0}
                title={t('defs.previewTooltip')}
                style={{ flex: 1, fontSize: '8.5pt', padding: '3px 6px',
                         borderRadius: 3,
                         cursor: (busy || !list.length) ? 'default' : 'pointer',
                         opacity: (busy || !list.length) ? 0.45 : 1,
                         background: 'transparent', color: C.text,
                         border: `1px solid ${C.border}` }}>
          {previewing ? t('defs.previewing') : t('defs.preview')}
        </button>
        <button type="button" onClick={onReclassify} disabled={busy}
                title={t('defs.applyTooltip')}
                style={{ flex: 1, fontSize: '8.5pt', padding: '3px 6px',
                         borderRadius: 3, cursor: busy ? 'default' : 'pointer',
                         background: alpha(C.cyan, 15), color: C.text,
                         border: `1px solid ${alpha(C.cyan, 45)}` }}>
          {t('defs.apply')}
        </button>
      </div>
    </div>
  );
}
