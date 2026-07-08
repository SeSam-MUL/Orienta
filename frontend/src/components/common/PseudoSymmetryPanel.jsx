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

  const row = selectedPixel?.row;
  const col = selectedPixel?.col;
  useEffect(() => {
    // reset the flip UI when the user clicks another pixel
    setVariants(null); setChosenVariant(null); setGrainMsg(null);
    setUndoAvailable(false);
  }, [row, col]);

  if (!selectedPixel || matchData?.indexing_method !== 'spherical') return null;

  const suspicious = matchData?.orientation_source === 'hough'
    || matchData?.r_quality === 'poor';

  const loadVariants = () => {
    setVariantsBusy(true); setGrainMsg(null);
    indexApi.patternMatchVariants(row, col)
      .then(r => {
        setVariants(r.data);
        // Auto-select the best NON-current candidate: applying 'current' is a
        // no-op (it is the orientation already on the map), so pre-picking it
        // would make the Apply button do nothing. Fall back to the best
        // overall only if every candidate is 'current'.
        const cands = r.data?.candidates || [];
        const best = cands.find(c => c.label !== 'current') || cands[0] || null;
        setChosenVariant(best);
      })
      .catch(e => setGrainMsg({ err: true, text: e?.response?.data?.detail || String(e) }))
      .finally(() => setVariantsBusy(false));
  };

  const applyToGrain = () => {
    if (!chosenVariant) return;
    setGrainBusy(true);
    indexApi.applyVariantToGrain({
      row, col, quat: chosenVariant.quat, thresholdDeg: grainThreshold, refine,
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
          <button onClick={loadVariants} disabled={variantsBusy}
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
            <span style={{ fontSize: '8pt', color: '#ffb86c' }}>
              {t('matchesDialog.suspicionHint')}
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

          {/* Thumbnail picker row — larger tiles, best-first. */}
          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
            {(variants.candidates || []).map((c, i) => (
              <div key={i} onClick={() => setChosenVariant(c)}
                title={`${c.label} — Euler (${(c.euler || []).map(a => a?.toFixed(1)).join(', ')})°`}
                style={{ border: chosenVariant === c ? '2px solid #50fa7b' : `1px solid ${C.border}`, borderRadius: 4, padding: 3, cursor: 'pointer', minWidth: 96, textAlign: 'center', flexShrink: 0 }}>
                <img src={`data:image/png;base64,${c.thumbnail}`} alt={c.label} style={{ width: 90, height: 90, objectFit: 'contain', display: 'block' }} />
                <div style={{ fontSize: '8pt', fontWeight: 700, color: c.r_score >= 0.3 ? '#50fa7b' : c.r_score >= 0.15 ? '#ffb86c' : '#ff5555' }}>R={c.r_score?.toFixed(2)}</div>
                <div style={{ fontSize: '7pt', color: c.label === 'current' ? '#ffb86c' : '#6272a4' }}>{c.label}</div>
              </div>
            ))}
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
              <button onClick={applyToGrain} disabled={grainBusy || chosenVariant.label === 'current'}
                title={chosenVariant.label === 'current' ? t('matchesDialog.currentNoop') : t('matchesDialog.applyGrainTip')}
                style={{ fontSize: '8pt', padding: '3px 10px', background: chosenVariant.label === 'current' ? '#44475a' : '#50fa7b22', border: `1px solid ${chosenVariant.label === 'current' ? C.border : '#50fa7b'}`, borderRadius: 3, color: chosenVariant.label === 'current' ? '#6272a4' : '#50fa7b', cursor: chosenVariant.label === 'current' ? 'not-allowed' : 'pointer', fontWeight: 700 }}>
                {grainBusy ? t('matchesDialog.grainApplying') : t('matchesDialog.applyGrain', { label: chosenVariant.label })}
              </button>
              {chosenVariant.label === 'current' && (
                <span style={{ fontSize: '8pt', color: '#ffb86c' }}>{t('matchesDialog.currentNoop')}</span>
              )}
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
