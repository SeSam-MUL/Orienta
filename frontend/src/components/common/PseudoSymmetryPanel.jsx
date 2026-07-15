import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi } from '../../services/api';
import { colors as C } from '../../theme/tokens';

/**
 * Universal manual pseudo-symmetry correction — SHARED panel mounted in both
 * pattern-match dialogs (Indexing page + Phase Maps page).
 *
 * Flow: load candidate orientations for the clicked pixel (current +
 * crystallographic pseudo-variants + Hough), each rendered through the SHT
 * forward model with its render-NCC — the user picks the one whose simulated
 * pattern matches, then applies it to the whole grain. The backend snap fill
 * (grain-flip v2) flips every grain pixel to ITS OWN best variant, so one
 * apply also fixes variant-split grains and intra-grain distortion survives.
 *
 * Contextual prominence: when the match looks wrong (orientation from Hough /
 * poor R) the entry button is accent-highlighted with a hint — visible without
 * scrolling instead of buried at the bottom of the dialog.
 *
 * Props:
 *   selectedPixel — {row, col} | null
 *   matchData     — pattern-match response (indexing_method, r_quality,
 *                   orientation_source, ...); panel renders only for spherical
 *   onApplied     — called after a successful apply/undo so the host dialog
 *                   can refetch the match and flush orientation-coloured map
 *                   layers (IPF) — Phase Maps refresh immediately.
 */
export default function PseudoSymmetryPanel({ selectedPixel, matchData, onApplied }) {
  const { t } = useTranslation('indexing');
  const [variants, setVariants] = useState(null);     // {candidates:[...], point_group} | null
  const [variantsBusy, setVariantsBusy] = useState(false);
  const [chosenVariant, setChosenVariant] = useState(null);
  const [grainThreshold, setGrainThreshold] = useState(5);
  const [refine, setRefine] = useState(true);
  const [grainBusy, setGrainBusy] = useState(false);
  const [grainMsg, setGrainMsg] = useState(null);
  const [undoAvailable, setUndoAvailable] = useState(false);
  // Free reference pixel (foreign-basin fixes): the user names any indexed
  // pixel whose orientation should join the gallery as a candidate.
  const [refRowIn, setRefRowIn] = useState('');
  const [refColIn, setRefColIn] = useState('');

  const row = selectedPixel?.row;
  const col = selectedPixel?.col;
  useEffect(() => {
    // reset the flip UI when the user clicks another pixel; prefill the
    // reference inputs with the CURRENT pixel so the (col, row) order is
    // unmistakable — empty placeholders let the user type them swapped.
    setVariants(null); setChosenVariant(null); setGrainMsg(null);
    setUndoAvailable(false);
    setRefColIn(col != null ? String(col) : '');
    setRefRowIn(row != null ? String(row) : '');
  }, [row, col]);

  if (!selectedPixel || matchData?.indexing_method !== 'spherical') return null;

  const suspicious = matchData?.orientation_source === 'hough'
    || matchData?.r_quality === 'poor';
  // Compare-phases evidence (when the user has it on): if the top re-indexed
  // result is the pixel's OWN phase within <2° of the stored orientation,
  // the poor R is a sub-degree refinement problem, NOT a wrong variant —
  // point the user at the re-index candidate instead of variant flipping.
  // (disorientation_deg is only ever set on same-phase rows.)
  const compareTop = matchData?.phase_results?.[0];
  const refineOnly = suspicious
    && compareTop?.disorientation_deg != null
    && compareTop.disorientation_deg < 2;

  const loadVariants = (opts = {}) => {
    const { ref = null, reindex = false } = opts;
    setVariantsBusy(true); setGrainMsg(null);
    indexApi.patternMatchVariants(row, col, {
      ...(ref ? { refRow: ref.row, refCol: ref.col } : {}),
      ...(reindex ? { reindex: true } : {}),
    })
      .then(r => {
        setVariants(r.data);
        // A silently missing reference tile cost the user a debugging
        // session — surface the backend's reason loudly.
        if (r.data?.reference_error) {
          setGrainMsg({ err: true,
            text: t('matchesDialog.refPickFailed', { msg: r.data.reference_error }) });
        }
        // Auto-select the best NON-current candidate: applying 'current' is a
        // no-op (it is the orientation already on the map), so pre-picking it
        // would make the Apply button do nothing. Fall back to the best
        // overall only if every candidate is 'current'.
        const cands = r.data?.candidates || [];
        const best = cands.find(c => c.kind !== 'current' && c.label !== 'current')
          || cands[0] || null;
        setChosenVariant(best);
      })
      .catch(e => setGrainMsg({ err: true, text: e?.response?.data?.detail || String(e) }))
      .finally(() => setVariantsBusy(false));
  };

  const applyToGrain = (propagate = false) => {
    if (!chosenVariant) return;
    setGrainBusy(true);
    // No cap override needed even for foreign-basin candidates (neighbour
    // grain / reference pixel): the backend's anti-drift cap is measured
    // relative to the TARGET, and the rigid-C fill lands every coherent
    // grain pixel near the target by construction.
    indexApi.applyVariantToGrain({
      row, col, quat: chosenVariant.quat, thresholdDeg: grainThreshold, refine,
      propagateSimilar: propagate,
    })
      .then(r => {
        const d = r.data;
        let text = t('matchesDialog.grainApplied', { n: d.n_changed });
        const rf = d.refine;
        if (rf?.status === 'applied') {
          text += ' ' + t('matchesDialog.refineApplied', {
            n: rf.n_refined, delta: (rf.median_render_ncc_delta ?? 0).toFixed(3) });
        } else if (rf?.status === 'rejected') {
          text += ' ' + t('matchesDialog.refineRejected');
        } else if (rf?.status === 'skipped' || rf?.status === 'error') {
          text += ' ' + t('matchesDialog.refineSkipped', { reason: rf.reason || '' });
        }
        const p = d.propagate;
        if (p) {
          text += ' ' + (p.n_candidates === 0
            ? t('matchesDialog.propagateNone')
            : t('matchesDialog.propagateDone', {
                found: p.n_candidates, adopted: p.n_adopted,
                rejected: p.n_rejected, px: p.n_pixels,
              }));
          if (p.truncated) text += ' ' + t('matchesDialog.propagateTruncated');
        }
        setGrainMsg({ err: false, text });
        setUndoAvailable(!!d.undo_available);
        onApplied?.();
      })
      .catch(e => setGrainMsg({ err: true, text: e?.response?.data?.detail || String(e) }))
      .finally(() => setGrainBusy(false));
  };

  const undoGrain = () => {
    setGrainBusy(true);
    indexApi.undoGrainFlip()
      .then(r => {
        setGrainMsg({ err: false, text: t('matchesDialog.undoDone', { n: r.data.n_restored }) });
        setUndoAvailable(false);
        onApplied?.();
      })
      .catch(e => setGrainMsg({ err: true, text: e?.response?.data?.detail || String(e) }))
      .finally(() => setGrainBusy(false));
  };

  return (
    <div style={{ marginTop: 8, borderTop: `1px solid ${C.border}`, paddingTop: 6 }}>
      {!variants ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <button onClick={() => loadVariants()} disabled={variantsBusy}
            title={t('matchesDialog.tryVariantsTip')}
            style={{
              fontSize: suspicious ? '9pt' : '8pt', padding: suspicious ? '5px 12px' : '3px 10px',
              background: suspicious ? '#ffb86c22' : C.bg,
              border: `1px solid ${suspicious ? '#ffb86c' : C.border}`,
              borderRadius: 3, color: suspicious ? '#ffb86c' : C.text,
              cursor: 'pointer', fontWeight: suspicious ? 700 : 400,
            }}>
            {variantsBusy ? t('matchesDialog.variantsLoading') : `⬡ ${t('matchesDialog.tryVariants')}`}
          </button>
          {suspicious && (
            <span style={{ fontSize: '8pt', color: refineOnly ? '#8be9fd' : '#ffb86c' }}>
              {refineOnly
                ? t('matchesDialog.suspicionHintRefine', {
                    delta: compareTop.disorientation_deg.toFixed(2) })
                : t('matchesDialog.suspicionHint')}
            </span>
          )}
        </div>
      ) : (
        <>
          <div style={{ fontSize: '8pt', color: '#6272a4', marginBottom: 6 }}>
            {t('matchesDialog.variantsHint', { pg: variants.point_group || '?' })}
          </div>

          {/* Large side-by-side: experimental vs the SELECTED candidate, so the
              user can actually judge the match at size (the thumbnail row below
              is only for picking). */}
          {chosenVariant && (
            <div style={{ display: 'flex', gap: 10, justifyContent: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
              {matchData?.experimental && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '8pt', color: '#f8f8f2', marginBottom: 2 }}>{t('matchesDialog.expLabel')}</div>
                  <img src={`data:image/png;base64,${matchData.experimental}`} alt="experimental"
                    style={{ width: 200, height: 200, objectFit: 'contain', display: 'block', borderRadius: 4, border: `1px solid ${C.border}`, background: '#1a1b26' }} />
                </div>
              )}
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '8pt', color: '#50fa7b', marginBottom: 2 }}>
                  {t('matchesDialog.selectedLabel', { label: chosenVariant.label })}
                </div>
                <img src={`data:image/png;base64,${chosenVariant.thumbnail}`} alt={chosenVariant.label}
                  style={{ width: 200, height: 200, objectFit: 'contain', display: 'block', borderRadius: 4, border: '2px solid #50fa7b', background: '#1a1b26' }} />
                <div style={{ fontSize: '8pt', fontWeight: 700, marginTop: 2, color: chosenVariant.r_score >= 0.3 ? '#50fa7b' : chosenVariant.r_score >= 0.15 ? '#ffb86c' : '#ff5555' }}>
                  R={chosenVariant.r_score?.toFixed(3)}
                </div>
              </div>
            </div>
          )}

          {/* Thumbnail picker row — larger tiles, best-first. Neighbour-grain
              and reference-pixel candidates get a coloured badge: they cover
              foreign basins the classic variants/Hough can't reach. */}
          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
            {(variants.candidates || []).map((c, i) => {
              const kindText = c.kind === 'neighbour'
                ? `↖ ${t('matchesDialog.kindNeighbour')}`
                : c.kind === 'reference'
                  ? `⌖ ${t('matchesDialog.kindReference')}`
                  : c.kind === 'reindex'
                    ? `↻ ${t('matchesDialog.kindReindex')}`
                    : c.label;
              const kindColor = c.kind === 'neighbour' ? '#8be9fd'
                : c.kind === 'reference' ? '#bd93f9'
                  : c.kind === 'reindex' ? '#f1fa8c'
                    : (c.label === 'current' ? '#ffb86c' : '#6272a4');
              return (
                <div key={i} onClick={() => setChosenVariant(c)}
                  title={`${c.label} — Euler (${(c.euler || []).map(a => a?.toFixed(1)).join(', ')})°`
                    + (c.disorientation_deg != null ? ` — Δ ${c.disorientation_deg}°` : '')}
                  style={{ border: chosenVariant === c ? '2px solid #50fa7b' : `1px solid ${C.border}`, borderRadius: 4, padding: 3, cursor: 'pointer', minWidth: 96, textAlign: 'center', flexShrink: 0 }}>
                  <img src={`data:image/png;base64,${c.thumbnail}`} alt={c.label} style={{ width: 90, height: 90, objectFit: 'contain', display: 'block' }} />
                  <div style={{ fontSize: '8pt', fontWeight: 700, color: c.r_score >= 0.3 ? '#50fa7b' : c.r_score >= 0.15 ? '#ffb86c' : '#ff5555' }}>R={c.r_score?.toFixed(2)}</div>
                  <div style={{ fontSize: '7pt', color: kindColor }}>{kindText}</div>
                </div>
              );
            })}
          </div>

          {/* Escalation ladder below the gallery (top→bottom): re-index this
              pixel (full search, same location) → reference another pixel
              (cross-grain, last resort). */}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>
            <button
              onClick={() => loadVariants({ reindex: true })}
              disabled={variantsBusy}
              title={t('matchesDialog.reindexTip')}
              style={{ fontSize: '8pt', padding: '2px 10px', background: 'transparent', border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: variantsBusy ? 'wait' : 'pointer' }}>
              {variantsBusy ? t('matchesDialog.reindexBusy') : `↻ ${t('matchesDialog.reindexBtn')}`}
            </button>
          </div>

          {/* Free reference pixel: fetch the gallery again with any indexed
              pixel's stored orientation as an extra candidate — for cases
              where the correct grain does NOT touch the wrong one. Order
              matches the dialog's "Pixel (col, row)" display; inputs are
              prefilled with the current pixel so the order is unmistakable. */}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 4, flexWrap: 'wrap' }}
            title={t('matchesDialog.refPickTip')}>
            <span style={{ fontSize: '8pt', color: '#6272a4' }}>
              {t('matchesDialog.refPickLabel')}{' '}
              <span style={{ color: '#8be9fd' }}>(col, row)</span>
            </span>
            <input type="number" min={0} placeholder={t('matchesDialog.refPickCol')} value={refColIn}
              onChange={e => setRefColIn(e.target.value)}
              aria-label={t('matchesDialog.refPickCol')}
              style={{ width: 58, fontSize: '8pt', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, padding: '2px 4px' }} />
            <input type="number" min={0} placeholder={t('matchesDialog.refPickRow')} value={refRowIn}
              onChange={e => setRefRowIn(e.target.value)}
              aria-label={t('matchesDialog.refPickRow')}
              style={{ width: 58, fontSize: '8pt', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, padding: '2px 4px' }} />
            <button
              onClick={() => {
                const rr = parseInt(refRowIn, 10);
                const rc = parseInt(refColIn, 10);
                if (Number.isFinite(rr) && Number.isFinite(rc)) loadVariants({ ref: { row: rr, col: rc } });
              }}
              disabled={variantsBusy || refRowIn === '' || refColIn === ''}
              style={{ fontSize: '8pt', padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}>
              {variantsBusy ? t('matchesDialog.variantsLoading') : t('matchesDialog.refPickAdd')}
            </button>
          </div>
          {chosenVariant && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>
              <label style={{ fontSize: '8pt', color: C.text }}>
                {t('matchesDialog.grainThreshold')}:
                <input type="number" min={1} max={20} step={0.5} value={grainThreshold}
                  onChange={e => setGrainThreshold(Number(e.target.value))}
                  style={{ width: 50, marginLeft: 4, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text }} />°
              </label>
              <label style={{ fontSize: '8pt', color: C.text, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
                title={t('matchesDialog.refineTip')}>
                <input type="checkbox" checked={refine} onChange={e => setRefine(e.target.checked)} />
                {t('matchesDialog.refineLabel')}
              </label>
              {(() => {
                const isCur = chosenVariant.kind === 'current' || chosenVariant.label === 'current';
                return (
                  <>
                    <button onClick={() => applyToGrain(false)} disabled={grainBusy || isCur}
                      title={isCur ? t('matchesDialog.currentNoop') : t('matchesDialog.applyGrainTip')}
                      style={{ fontSize: '8pt', padding: '3px 10px', background: isCur ? '#44475a' : '#50fa7b22', border: `1px solid ${isCur ? C.border : '#50fa7b'}`, borderRadius: 3, color: isCur ? '#6272a4' : '#50fa7b', cursor: isCur ? 'not-allowed' : 'pointer', fontWeight: 700 }}>
                      {grainBusy ? t('matchesDialog.grainApplying') : t('matchesDialog.applyGrain', { label: chosenVariant.label })}
                    </button>
                    {/* Second BUTTON (not a hidden checkbox mode): scope is
                        part of what's clicked. Map-wide siblings of the same
                        wrong orientation get the same fix, each render-
                        verified server-side. */}
                    <button onClick={() => applyToGrain(true)} disabled={grainBusy || isCur}
                      title={t('matchesDialog.applyAllTip')}
                      style={{ fontSize: '8pt', padding: '3px 10px', background: 'transparent', border: `1px solid ${isCur ? C.border : '#8be9fd'}`, borderRadius: 3, color: isCur ? '#6272a4' : '#8be9fd', cursor: isCur ? 'not-allowed' : 'pointer', fontWeight: 600 }}>
                      {grainBusy ? t('matchesDialog.applyAllBusy') : t('matchesDialog.applyAllBtn')}
                    </button>
                    {isCur && (
                      <span style={{ fontSize: '8pt', color: '#ffb86c' }}>{t('matchesDialog.currentNoop')}</span>
                    )}
                  </>
                );
              })()}
            </div>
          )}
        </>
      )}
      {(grainMsg || undoAvailable) && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 4, flexWrap: 'wrap' }}>
          {grainMsg && (
            <span style={{ fontSize: '8pt', color: grainMsg.err ? '#ff5555' : '#50fa7b' }}>{grainMsg.text}</span>
          )}
          {undoAvailable && (
            <button onClick={undoGrain} disabled={grainBusy}
              title={t('matchesDialog.undoTip')}
              style={{ fontSize: '8pt', padding: '2px 8px', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 3, color: C.text, cursor: 'pointer' }}>
              ↩ {t('matchesDialog.undoBtn')}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
