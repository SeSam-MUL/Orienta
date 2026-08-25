/**
 * Structures: what the map is made of, before anything is named.
 *
 * The user's framing, and the reason this panel exists: group pixels that
 * belong together by element ratio alone, then put a CIF on each group. The
 * previous design went straight to phase names, and two things collapsed on
 * the way — the cluster count was chosen to keep the picture tidy (which
 * forces the smallest count), and several distinct chemistries mapped onto the
 * same degenerate CIF name. Measured on SampleB: 7 structures painted as 3
 * colours.
 *
 * The four tools below exist because no clustering gets the grouping right on
 * this data. EDS taken during an EBSD session has an interaction volume far
 * larger than the features, so a small particle reads as a dilution gradient
 * and gets cut into concentric rings — on SampleB one Si particle became three
 * structures at Si 24 / 36 / 52 at%. How much rim belongs to the particle is a
 * judgement, so it is offered as a control rather than decided.
 */
import React, { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button, Label, colors as C, alpha } from '../../theme/components';
import PhaseRollup from './PhaseRollup';
import OpacitySlider from '../common/OpacitySlider';
import { BG_NONE } from './useMapBackground';

/** Short, readable composition: the elements that actually carry signal. */
export function describeComposition(meanAtPct, maxElements = 4) {
  const entries = Object.entries(meanAtPct || {})
    .filter(([, v]) => v >= 0.5)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxElements);
  return entries.map(([el, v]) => `${el} ${v.toFixed(1)}`).join(' · ');
}

function ToolRow({ children, title }) {
  return (
    <div
      title={title}
      style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4,
               flexWrap: 'wrap' }}
    >
      {children}
    </div>
  );
}

export default function StructurePanel({ handle }) {
  const { t } = useTranslation('eds');
  const {
    phaseMap, structureBusy,
    selectedStructureId, setSelectedStructureId,
    scale, setScale,
    nClusters, setNClusters,
    handleMergeStructures, handleSplitStructure,
    handleGrowStructure, handleSnapEdges,
    setMapView, background, backgroundChoices,
    mode, phaseMap: pm,
  } = handle;

  const [mergeInto, setMergeInto] = useState(null);
  const [snapStrength, setSnapStrength] = useState(2);

  const structures = phaseMap?.structures || [];
  const selected = useMemo(
    () => structures.find((s) => s.structure_id === selectedStructureId) || null,
    [structures, selectedStructureId],
  );

  const pick = useCallback((sid) => {
    setSelectedStructureId(selectedStructureId === sid ? null : sid);
    setMergeInto(null);
  }, [selectedStructureId, setSelectedStructureId]);

  if (!phaseMap?.loaded) return null;

  if (structures.length === 0) {
    return (
      <Label secondary small style={{ display: 'block', marginTop: 4 }}>
        {t('structures.none')}
      </Label>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {/* --- what this page is doing, in three sentences. Open by
          default: the user's report was "ich habe keine Ahnung was ich
          hier sehe", and a closed box does not fix that. --- */}
      <details open style={{ marginBottom: 2 }}>
        <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                          color: C.cyan, userSelect: 'none' }}>
          {t('structures.howTitle')}
        </summary>
        <div style={{ fontSize: '8pt', color: C.textSecondary,
                      lineHeight: 1.45, marginTop: 3 }}>
          <div>{t('structures.howStep1')}</div>
          <div style={{ marginTop: 3 }}>{t('structures.howStep2')}</div>
          <div style={{ marginTop: 3 }}>{t('structures.howStep3')}</div>
        </div>
      </details>

      {/* --- what to lay the map over ----------------------------------
          Placing a region is easier against the picture it came from, so the
          map can sit over one element map or one electron image. The MAP is
          what fades - it is the thing being placed. --- */}
      {backgroundChoices?.length > 0 && background && (
        <div style={{ marginBottom: 4 }}>
          <ToolRow title={t('background.tooltip')}>
            <Label secondary small style={{ minWidth: 62 }}>
              {t('background.label')}
            </Label>
            <select
              value={background.backgroundId}
              onChange={(e) => background.setBackgroundId(e.target.value)}
              aria-label={t('background.label')}
              style={{ flex: 1, minWidth: 0, fontSize: '8.5pt',
                       padding: '2px 4px', background: 'transparent',
                       color: C.text, borderRadius: 3,
                       border: `1px solid ${alpha(C.purple, 25)}` }}
            >
              <option value={BG_NONE}>{t('background.none')}</option>
              {backgroundChoices.map((g) => (
                <optgroup key={g.label} label={g.label}>
                  {g.items.map((it) => (
                    <option key={it.id} value={it.id}>{it.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
          </ToolRow>
          {background.isSet && (
            <>
              <ToolRow title={t('background.opacityTooltip')}>
                <Label secondary small style={{ minWidth: 62 }}>
                  {t('background.opacity')}
                </Label>
                <OpacitySlider
                  value={background.opacity}
                  onChange={background.setOpacity}
                  color={C.cyan}
                  aria-label={t('background.opacity')}
                />
              </ToolRow>
              {background.error && (
                <Label secondary small style={{ display: 'block', color: C.red }}>
                  {background.error}
                </Label>
              )}
              {background.fovWarning && (
                <Label secondary small style={{ display: 'block',
                                               color: C.orange || '#f0b429' }}>
                  {t('background.fovWarning', { pct: background.fovWarning })}
                </Label>
              )}
            </>
          )}
        </div>
      )}

      {/* --- how the grouping was made ---------------------------------- */}
      <ToolRow title={t('structures.scaleTooltip')}>
        <Label secondary small style={{ minWidth: 62 }}>
          {t('structures.scale')}
        </Label>
        <input
          type="range"
          min={0}
          max={11}
          step={2}
          value={scale ?? 5}
          onChange={(e) => setScale(Number(e.target.value))}
          aria-label={t('structures.scale')}
          style={{ flex: 1, minWidth: 70 }}
        />
        <span style={{ fontSize: '8.5pt', color: C.textSecondary,
                       fontVariantNumeric: 'tabular-nums', minWidth: 34 }}>
          {(scale ?? 5) <= 1 ? t('structures.scaleOff') : `${scale ?? 5} px`}
        </span>
      </ToolRow>

      {/* Switching to per-pixel throws the structures away, and with them
          every name the user gave by hand. Verified: the sidecar drops to an
          empty structure grid and comes back only on the next cluster run.
          Not a fault - per-pixel HAS no groups - but it must not be silent. */}
      {handle.mode === 'pixel' && (structures?.length === 0)
        && (phaseMap?.n_classified > 0) && (
        <Label secondary small style={{ display: 'block',
                                        color: C.orange || '#f0b429' }}>
          {t('structures.pixelModeNoStructures')}
        </Label>
      )}

      <ToolRow title={t('structures.countTooltip')}>
        <Label secondary small style={{ minWidth: 62 }}>
          {t('structures.count')}
        </Label>
        <input
          type="number"
          min={2}
          max={24}
          value={nClusters ?? ''}
          placeholder={t('structures.countAuto')}
          onChange={(e) => setNClusters(
            e.target.value === '' ? null : Number(e.target.value))}
          aria-label={t('structures.count')}
          style={{ width: 62, fontSize: '8.5pt', padding: '2px 4px',
                   background: 'transparent', color: C.text,
                   border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3 }}
        />
        <span style={{ fontSize: '8pt', color: C.textSecondary }}>
          {t('structures.countNow', { count: structures.length })}
        </span>
      </ToolRow>
      <Label secondary small style={{ display: 'block' }}>
        {t('structures.reclassifyHint')}
      </Label>

      {/* --- the structures --------------------------------------------- */}
      <div
        className="thin-scrollbar"
        style={{ display: 'flex', flexDirection: 'column', gap: 2,
                 maxHeight: 210, overflowY: 'auto', marginTop: 4,
                 border: `1px solid ${C.border}`, borderRadius: 4, padding: 4 }}
      >
        {structures.map((s) => {
          const active = selectedStructureId === s.structure_id;
          return (
            <div
              key={s.structure_id}
              role="button"
              tabIndex={0}
              onClick={() => pick(s.structure_id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  pick(s.structure_id);
                }
              }}
              title={t('structures.rowTooltip', {
                px: s.n_pixels,
                comp: describeComposition(s.mean_at_pct, 8) || '—',
              })}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 6px', borderRadius: 3, cursor: 'pointer',
                background: active ? alpha(C.cyan, 18) : 'transparent',
                border: active ? `1px solid ${alpha(C.cyan, 50)}`
                               : '1px solid transparent',
              }}
            >
              <span style={{ width: 14, height: 14, borderRadius: 3,
                             flexShrink: 0, background: s.color,
                             border: `1px solid ${C.border}` }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '8.5pt', color: C.text,
                              overflow: 'hidden', textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap' }}>
                  {describeComposition(s.mean_at_pct) || t('structures.noChemistry')}
                </div>
                <div style={{ fontSize: '7.5pt',
                              color: s.cif_filename ? C.green : C.textSecondary,
                              overflow: 'hidden', textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap' }}>
                  {s.cif_filename || t('structures.unnamed')}
                </div>
              </div>
              <span style={{ fontSize: '8pt', color: C.textSecondary,
                             flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
                {s.percentage}%
              </span>
            </div>
          );
        })}
      </div>

      {/* --- many regions, few phases: the collapse made visible -------- */}
      <PhaseRollup
        structures={structures}
        allPhases={phaseMap?.all_phases}
        totalPixels={(phaseMap?.n_rows || 0) * (phaseMap?.n_cols || 0)}
        onShowPhases={() => setMapView?.('phases')}
      />

      {/* --- what to do with the selected one ---------------------------- */}
      {selected && (
        <div style={{ marginTop: 4, padding: '4px 6px', borderRadius: 4,
                      border: `1px solid ${alpha(C.cyan, 30)}` }}>
          {/* The composition, the candidates and the neighbours live in the
              inspector under the map, where they fit side by side. Repeating
              them here cost a second ranking definition that could disagree
              with the backend's. */}
          {/* Merge — the tool this data needs most. */}
          <ToolRow title={t('structures.mergeTooltip')}>
            <Label secondary small style={{ minWidth: 62 }}>
              {t('structures.merge')}
            </Label>
            <select
              value={mergeInto ?? ''}
              onChange={(e) => setMergeInto(
                e.target.value === '' ? null : Number(e.target.value))}
              aria-label={t('structures.merge')}
              style={{ flex: 1, minWidth: 0, fontSize: '8.5pt',
                       padding: '2px 4px', background: 'transparent',
                       color: C.text, borderRadius: 3,
                       border: `1px solid ${alpha(C.purple, 25)}` }}
            >
              <option value="">{t('structures.mergePick')}</option>
              {structures
                .filter((o) => o.structure_id !== selected.structure_id)
                .map((o) => (
                  <option key={o.structure_id} value={o.structure_id}>
                    {describeComposition(o.mean_at_pct, 3) || `#${o.structure_id}`}
                    {` (${o.percentage}%)`}
                  </option>
                ))}
            </select>
            <Button
              variant="secondary"
              disabled={mergeInto == null || structureBusy}
              onClick={() => {
                handleMergeStructures(selected.structure_id, mergeInto);
                setMergeInto(null);
              }}
            >
              {t('structures.mergeGo')}
            </Button>
          </ToolRow>

          {/* Split — local, so the rest of the map survives. */}
          <ToolRow title={t('structures.splitTooltip')}>
            <Label secondary small style={{ minWidth: 62 }}>
              {t('structures.split')}
            </Label>
            {[2, 3, 4].map((n) => (
              <Button
                key={n}
                variant="secondary"
                disabled={structureBusy}
                onClick={() => handleSplitStructure(selected.structure_id, n)}
              >
                {t('structures.splitInto', { count: n })}
              </Button>
            ))}
          </ToolRow>

          {/* Grow / shrink — the literal boundary nudge. */}
          <ToolRow title={t('structures.growTooltip')}>
            <Label secondary small style={{ minWidth: 62 }}>
              {t('structures.grow')}
            </Label>
            <Button
              variant="secondary"
              disabled={structureBusy}
              onClick={() => handleGrowStructure(selected.structure_id, -1)}
            >
              −1 px
            </Button>
            <Button
              variant="secondary"
              disabled={structureBusy}
              onClick={() => handleGrowStructure(selected.structure_id, 1)}
            >
              +1 px
            </Button>
          </ToolRow>
        </div>
      )}

      {/* --- map-wide: let the data place the boundaries ------------------ */}
      <ToolRow title={t('structures.snapTooltip')}>
        <Label secondary small style={{ minWidth: 62 }}>
          {t('structures.snap')}
        </Label>
        <input
          type="range"
          min={1}
          max={5}
          step={1}
          value={snapStrength}
          onChange={(e) => setSnapStrength(Number(e.target.value))}
          aria-label={t('structures.snap')}
          style={{ flex: 1, minWidth: 60 }}
        />
        <span style={{ fontSize: '8.5pt', color: C.textSecondary,
                       fontVariantNumeric: 'tabular-nums', minWidth: 30 }}>
          {snapStrength} px
        </span>
        <Button
          variant="secondary"
          disabled={structureBusy}
          onClick={() => handleSnapEdges(snapStrength)}
        >
          {t('structures.snapGo')}
        </Button>
      </ToolRow>
    </div>
  );
}
