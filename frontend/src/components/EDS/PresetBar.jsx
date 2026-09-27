/**
 * The analysis recipe as a portable file — and the gate that stops it being
 * applied to a scan it does not fit.
 *
 * The compatibility check IS the feature. A preset that applies silently to
 * the wrong dataset is worse than no preset at all: it produces a plausible
 * map from settings that mean something else, and nothing on screen says so.
 * Four of six consulted users independently asked for a refusal rather than a
 * warning, and the design spec lists the four refusable conditions (missing
 * element, wrong pinned matrix, missing CIF phase, and — as warnings — a
 * different element list or step size).
 *
 * Rules this file follows, each for a recorded reason:
 *
 *  - **Check before apply, always.** `POST /presets/{name}/check` runs first
 *    and nothing is written into the controls until it answers.
 *  - **`ok: false` refuses.** The blockers stay on screen as a panel, not a
 *    toast: a warning that disappears in half a second is a warning nobody
 *    read. Overriding is a second, separate click.
 *  - **An override is recorded.** The refusal report travels into
 *    `POST /api/eds/export` as `compatibility`, so `provenance.json` carries
 *    "this was applied over a refusal" rather than losing it at dialog close.
 *  - **Identity is shown.** Author, date and a short content hash. A user put
 *    it plainly: "a preset with no author is not evidence, it's a rumour."
 *  - **`tolerance` is excluded.** Verified inert (`auto_classify_pixels`
 *    never reads it, `eds_clustering.py` references it zero times). A preset
 *    that stored it would have the user tune a control, see no change, and
 *    conclude the preset was broken.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { edsExportApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import {
  colors as C, alpha, Button, Select, Input, Label,
  useConfirm, ConfirmDialog,
} from '../../theme/components';

/** Everything a preset may set, and nothing that is bound to one scan. */
export const PRESET_FIELDS = [
  'mode', 'scale', 'scale_um', 'n_clusters', 'min_score', 'element_weights',
  'region_defs', 'rules', 'phase_keys', 'cluster_remainder',
];

/**
 * Read the current controls into a settings object.
 *
 * `tolerance` is deliberately absent — see the header. `n_clusters_pinned`
 * carries the difference between "8 clusters" and "let it choose, which
 * happened to give 8": storing only the number would turn every automatic
 * preset into a pinned one the moment it was saved.
 */
export function presetSettingsFrom(handle = {}) {
  // A physical smoothing width is the one setting that means the SAME
  // analysis on two scans with different step sizes — which is the entire
  // reason a recipe is portable at all. It travels only when the user is
  // actually working in um: storing a stale um value beside a pixel count
  // would let the backend's precedence rule silently pick the one the user
  // was not looking at.
  const um = handle.scaleUnit === 'um' ? Number(handle.scaleUm) : NaN;
  return {
    mode: handle.mode ?? 'cluster',
    scale: handle.scale ?? null,
    scale_um: (Number.isFinite(um) && um > 0) ? um : null,
    n_clusters: handle.nClusters ?? null,
    n_clusters_pinned: handle.nClusters != null,
    min_score: handle.minScore ?? null,
    element_weights: handle.elementWeights || {},
    region_defs: handle.regionDefs || [],
    rules: handle.rules || null,
    // Explicit, never "all": a preset saying "everything" would silently
    // include phases the library gained after it was written.
    phase_keys: [...(handle.selectedPhaseKeys || [])],
    cluster_remainder: handle.clusterRemainder !== false,
  };
}

/**
 * Write a preset's settings into the phase-map controls.
 *
 * Returns the list of fields it actually touched, so a caller (and a test)
 * can tell "applied nothing" from "applied everything". Absent fields are
 * left alone rather than reset — a preset that only pins the cluster count
 * must not also wipe the region definitions.
 *
 * `scale_px` wins over `scale` when the backend resolved a physical
 * `scale_um` against this scan's step size: smoothing is a length, and the
 * pixel count that realises it differs per dataset.
 */
export function applyPresetSettings(settings, handle = {}) {
  const s = settings || {};
  const done = [];

  if (s.mode === 'cluster' || s.mode === 'pixel') {
    handle.setMode?.(s.mode);
    done.push('mode');
  }
  const px = s.scale_px ?? s.scale;
  if (px != null && Number.isFinite(Number(px))) {
    handle.setScale?.(Number(px));
    done.push('scale');
  }
  // Restore the UNIT as well as the number. A preset written in microns that
  // came back as a pixel count would be the very defect the field exists to
  // fix, one step further down the chain. An ABSENT key is left alone — a
  // preset written before this field existed must not silently re-declare
  // the user's current unit.
  if ('scale_um' in s) {
    const um = Number(s.scale_um);
    if (Number.isFinite(um) && um > 0) {
      handle.setScaleUm?.(um);
      handle.setScaleUnit?.('um');
    } else {
      handle.setScaleUnit?.('px');
    }
    done.push('scale_um');
  }
  if ('n_clusters' in s || 'n_clusters_pinned' in s) {
    const pinned = s.n_clusters_pinned !== false && s.n_clusters != null;
    handle.setNClusters?.(pinned ? Number(s.n_clusters) : null);
    done.push('n_clusters');
  }
  if (s.min_score != null && Number.isFinite(Number(s.min_score))) {
    handle.setMinScore?.(Number(s.min_score));
    done.push('min_score');
  }
  if (s.element_weights && typeof s.element_weights === 'object') {
    handle.setElementWeights?.({ ...s.element_weights });
    done.push('element_weights');
  }
  if (Array.isArray(s.region_defs)) {
    handle.setRegionDefs?.(s.region_defs.map((d) => ({ ...d })));
    done.push('region_defs');
  }
  if ('rules' in s) {
    handle.setRules?.(s.rules ? { ...s.rules } : null);
    done.push('rules');
  }
  if (Array.isArray(s.phase_keys) && s.phase_keys.length) {
    handle.setSelectedPhaseKeys?.(new Set(s.phase_keys));
    done.push('phase_keys');
  }
  if (typeof s.cluster_remainder === 'boolean') {
    handle.setClusterRemainder?.(s.cluster_remainder);
    done.push('cluster_remainder');
  }
  // `tolerance` is never applied, even when a hand-written preset file
  // carries one. Silently doing nothing with it is the honest option; the
  // control itself is labelled inert in the panel.
  return done;
}

/** First eight characters — enough to compare two presets by eye. */
export function shortHash(hash) {
  const h = String(hash || '');
  return h ? h.slice(0, 8) : '';
}

function utf8ToBase64(text) {
  const bytes = new TextEncoder().encode(String(text ?? ''));
  let bin = '';
  for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

/**
 * Write a text file, through Electron when it is there and a browser
 * download when it is not.
 *
 * `saveImage` is the only channel in the preload that writes BYTES; the
 * name is historical, the handler is format-agnostic (`main.js` writes the
 * buffer and forwards the filters it is given).
 */
export async function saveTextFile(text, filename) {
  const api = typeof window !== 'undefined' ? window.electronAPI : null;
  if (api?.saveImage) {
    const path = await api.saveImage({
      defaultPath: filename,
      base64: utf8ToBase64(text),
      filters: [{ name: 'JSON', extensions: ['json'] }],
    });
    return path ? { via: 'electron', path } : null;
  }
  if (typeof document === 'undefined' || typeof URL?.createObjectURL !== 'function') {
    return null;
  }
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return { via: 'download', path: filename };
}

const errText = (e) => e?.response?.data?.detail || e?.message || String(e);

/* ------------------------------------------------------------------------ *
 * Finding one preset among a hundred.
 *
 * `material_class` and `tags` were collected on save, written to disk and
 * returned by the list endpoint - and shown nowhere, while the tooltip that
 * asks for a material class promises it "is what makes a preset findable once
 * there are a hundred of them". A service lab expects 60-100 of them after a
 * year; a flat list of names is not a way to find one.
 *
 * The shape is deliberately the one `PhaseMap/AddLayerPicker.jsx` already
 * uses - a trigger, a search box, a filter, a grouped list - rather than a
 * second idiom for the same job.
 * ------------------------------------------------------------------------ */

/** Sentinel for "presets with no material class". Not '', which means "all". */
export const NO_MATERIAL_CLASS = '\u0000none';

export const presetMaterial = (p) => String(p?.material_class || '').trim();

export const presetTags = (p) =>
  (Array.isArray(p?.tags) ? p.tags : []).map((x) => String(x).trim()).filter(Boolean);

/**
 * Every word of the query must appear somewhere in the preset's identity -
 * name, material class, tags, author or notes.
 *
 * Word-wise rather than as one substring, so "al steel" finds a preset named
 * "matrix" tagged `steel` in material class "low-alloy steel" without
 * demanding the user guess the order the fields were concatenated in.
 */
export function matchesPresetQuery(p, query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return true;
  const hay = [
    p?.name, presetMaterial(p), presetTags(p).join(' '), p?.author, p?.notes,
  ].filter(Boolean).join(' ').toLowerCase();
  return q.split(/\s+/).every((w) => hay.includes(w));
}

/**
 * Narrow by free text, by material class and by tags.
 *
 * Tags are ANDed: ticking a second one asks for presets carrying both, which
 * is the only reading under which adding a tag ever narrows anything.
 */
export function filterPresets(list, { query = '', material = '', tags = [] } = {}) {
  const want = (tags || []).map((x) => String(x).toLowerCase());
  return (list || []).filter((p) => {
    if (!p?.name) return false;
    const mat = presetMaterial(p);
    if (material === NO_MATERIAL_CLASS) { if (mat) return false; }
    else if (material && mat !== material) return false;
    if (want.length) {
      const have = presetTags(p).map((x) => x.toLowerCase());
      if (!want.every((x) => have.includes(x))) return false;
    }
    return matchesPresetQuery(p, query);
  });
}

/** Distinct material classes present, alphabetical. */
export function collectMaterials(list) {
  return [...new Set((list || []).map(presetMaterial).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b));
}

/** Distinct tags present, alphabetical. */
export function collectTags(list) {
  return [...new Set((list || []).flatMap(presetTags))]
    .sort((a, b) => a.localeCompare(b));
}

/**
 * Group for display, alphabetically by material class, with the unlabelled
 * ones LAST - they are the ones somebody still has to label, not the ones to
 * read first.
 */
export function groupPresetsByMaterial(list) {
  const byMat = new Map();
  for (const p of list || []) {
    const key = presetMaterial(p);
    if (!byMat.has(key)) byMat.set(key, []);
    byMat.get(key).push(p);
  }
  const named = [...byMat.keys()].filter(Boolean).sort((a, b) => a.localeCompare(b));
  const out = named.map((m) => ({ material: m, items: byMat.get(m) }));
  if (byMat.has('')) out.push({ material: '', items: byMat.get('') });
  return out;
}

const chipStyle = (on) => ({
  fontSize: '7.5pt', padding: '1px 6px', borderRadius: 8, cursor: 'pointer',
  background: on ? alpha(C.cyan, 30) : 'transparent',
  color: on ? C.text : C.textSecondary,
  border: `1px solid ${alpha(C.cyan, on ? 70 : 25)}`,
});

const LOCK = '\u{1F512} ';

/**
 * No compatibility answer on screen, and the file it is an answer about.
 *
 * Module level so the identity is stable: a fresh object per render would
 * make every consumer of the derived value re-run for nothing.
 */
export const NO_CHECK = { file: '', report: null, warnings: [] };

/**
 * Read a comma-separated tag line the way a user types it.
 *
 * Free text on purpose - the same reason `material_class` is. She needs
 * group names, project names and instrument names ("6xxx extrusion",
 * "customer 4711", "Symmetry S2"), and any enum we could write would be
 * wrong within a week.
 *
 * Trimmed, empties dropped (a trailing comma is how people type), and
 * de-duplicated CASE-INSENSITIVELY because `filterPresets` already compares
 * tags in lower case - "QA" and "qa" are one tag to the filter, so storing
 * both would put two chips in the row that select the same presets. The
 * first spelling wins and the typed ORDER is kept: it is the only ordering
 * information the user gave us.
 */
export function parseTags(text) {
  const out = [];
  const seen = new Set();
  for (const raw of String(text ?? '').split(',')) {
    const tag = raw.trim();
    if (!tag) continue;
    const key = tag.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(tag);
  }
  return out;
}

/**
 * The preset picker.
 *
 * Module level, never nested inside `PresetBar`: a component declared in a
 * component is a NEW type on every render, so React unmounts and remounts the
 * subtree - the open list would lose focus mid-keystroke. This repo has been
 * bitten by exactly that before (the annotation toolbar, 2026-08-14).
 */
export function PresetPicker({ presets = [], selected = '', onSelect, t }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [material, setMaterial] = useState('');
  const [tags, setTags] = useState([]);
  const inputRef = useRef(null);

  const materials = useMemo(() => collectMaterials(presets), [presets]);
  const allTags = useMemo(() => collectTags(presets), [presets]);
  const hasUnlabelled = useMemo(
    () => (presets || []).some((p) => p?.name && !presetMaterial(p)), [presets]);
  const filtered = useMemo(
    () => filterPresets(presets, { query, material, tags }),
    [presets, query, material, tags]);
  const groups = useMemo(() => groupPresetsByMaterial(filtered), [filtered]);

  const current = (presets || []).find((p) => p.name === selected) || null;
  const currentMat = presetMaterial(current);

  const pick = (name) => {
    onSelect?.(name);
    setOpen(false);
    setQuery('');
  };

  const toggleTag = (tag) => setTags((prev) => (
    prev.includes(tag) ? prev.filter((x) => x !== tag) : [...prev, tag]));

  const onKeyDown = (e) => {
    if (e.key === 'Escape') { setOpen(false); return; }
    // Enter takes the first hit - that is what a search box is for.
    if (e.key === 'Enter' && filtered.length) pick(filtered[0].name);
  };

  return (
    <div data-preset-picker style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <button
        type="button"
        aria-label={t('presets.title')}
        aria-expanded={open}
        data-preset-trigger
        title={t('presets.pickTooltip')}
        onClick={() => {
          const next = !open;
          setOpen(next);
          if (next) setTimeout(() => inputRef.current?.focus(), 0);
        }}
        style={{
          width: '100%', minWidth: 0, textAlign: 'left',
          fontSize: '9pt', height: 24, padding: '1px 6px',
          background: 'transparent', color: C.text,
          border: `1px solid ${C.border}`, borderRadius: 3, cursor: 'pointer',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}
      >
        {selected
          ? `${(current?.builtin ?? current?.read_only) ? LOCK : ''}${selected}${
              currentMat ? ` \u00b7 ${currentMat}` : ''}`
          : t('presets.none')}
      </button>

      {open && (
        <div
          data-preset-list
          onKeyDown={onKeyDown}
          style={{
            border: `1px solid ${C.border}`, borderRadius: 3,
            padding: 4, display: 'flex', flexDirection: 'column', gap: 4,
            background: alpha(C.cyan, 4),
          }}
        >
          <input
            ref={inputRef}
            type="search"
            value={query}
            aria-label={t('presets.search')}
            placeholder={t('presets.searchPlaceholder')}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              fontSize: '9pt', height: 24, padding: '2px 6px', minWidth: 0,
              background: 'transparent', color: C.text,
              border: `1px solid ${C.border}`, borderRadius: 3,
            }}
          />

          <Select
            value={material}
            aria-label={t('presets.filterMaterial')}
            onChange={(e) => setMaterial(e.target.value)}
            options={[
              { value: '', label: t('presets.allMaterials') },
              ...materials.map((m) => ({ value: m, label: m })),
              ...(hasUnlabelled
                ? [{ value: NO_MATERIAL_CLASS, label: t('presets.noMaterial') }]
                : []),
            ]}
            style={{ height: 24, fontSize: '8.5pt' }}
          />

          {allTags.length > 0 && (
            <div data-preset-tag-filter
                 style={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ fontSize: '7.5pt', color: C.textSecondary }}>
                {t('presets.tags')}
              </span>
              {allTags.map((tag) => (
                <button
                  key={tag}
                  type="button"
                  data-preset-tag={tag}
                  onClick={() => toggleTag(tag)}
                  aria-pressed={tags.includes(tag)}
                  title={t('presets.tagFilterTooltip')}
                  style={chipStyle(tags.includes(tag))}
                >
                  {tag}
                </button>
              ))}
            </div>
          )}

          <div style={{ maxHeight: 240, overflowY: 'auto' }}>
            <button
              type="button"
              data-preset-option=""
              onClick={() => pick('')}
              style={{
                display: 'block', width: '100%', textAlign: 'left',
                fontSize: '8.5pt', padding: '3px 6px', cursor: 'pointer',
                background: 'transparent', color: C.textSecondary, border: 'none',
              }}
            >
              {t('presets.none')}
            </button>
            {filtered.length === 0 && (
              <div style={{ fontSize: '8pt', color: C.textSecondary, padding: '6px 4px' }}>
                {t('presets.noMatches')}
              </div>
            )}
            {groups.map((g) => (
              <div key={g.material || NO_MATERIAL_CLASS}>
                <div style={{
                  fontSize: '7.5pt', color: C.textSecondary,
                  padding: '4px 4px 2px', textTransform: 'uppercase',
                  letterSpacing: 0.4,
                }}>
                  {g.material || t('presets.noMaterial')}
                </div>
                {g.items.map((p) => (
                  <button
                    key={p.name}
                    type="button"
                    data-preset-option={p.name}
                    onClick={() => pick(p.name)}
                    title={p.notes || p.name}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left',
                      fontSize: '9pt', padding: '3px 6px', cursor: 'pointer',
                      background: p.name === selected ? alpha(C.purple, 20) : 'transparent',
                      color: C.text, border: 'none', borderRadius: 2,
                    }}
                  >
                    <span style={{ whiteSpace: 'nowrap', overflow: 'hidden',
                                   textOverflow: 'ellipsis', display: 'block' }}>
                      {(p.builtin ?? p.read_only) ? LOCK : ''}{p.name}
                    </span>
                    {presetTags(p).length > 0 && (
                      <span style={{ fontSize: '7.5pt', color: C.textSecondary }}>
                        {presetTags(p).map((x) => `#${x}`).join(' ')}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            ))}
          </div>

          <div style={{ fontSize: '7.5pt', color: C.textSecondary }}>
            {t('presets.countShown', {
              shown: filtered.length,
              total: (presets || []).filter((p) => p?.name).length,
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * The preset row above the classification controls.
 *
 * `onApplied({ name, compatibility })` hands the applied preset and its
 * report up so the export dialog can put both into the provenance. A refused
 * preset that was overridden arrives with `compatibility.ok === false`.
 */
export default function PresetBar({ handle = {}, onApplied }) {
  const { t } = useTranslation('eds');
  // WHICH scan every answer below is about. Read from the store the rest of
  // the page reads it from, not passed in: a caller that forgot the prop
  // would silently get the stale-refusal bug back.
  const filePath = useDataStore((s) => s.filePath);

  const [presets, setPresets] = useState([]);
  const [selected, setSelected] = useState('');
  const [busy, setBusy] = useState(false);
  const [listError, setListError] = useState(null);
  // The last compatibility answer, and WHICH scan it was an answer about.
  // The refusal inside it is kept on screen until it is fixed or overridden.
  // Never a toast — the whole point is that it does not go away by itself.
  const [checked, setChecked] = useState(NO_CHECK);
  const [status, setStatus] = useState(null);   // { kind, text }
  const [showSave, setShowSave] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [saveAuthor, setSaveAuthor] = useState('');
  const [saveNotes, setSaveNotes] = useState('');
  // Findability and the strongest safety blocker. A service lab expects on
  // the order of a hundred presets; without a material class they are a list
  // of names, and without a pinned matrix element `check_compatibility` has
  // nothing to refuse a wrong-alloy preset with.
  const [saveMaterial, setSaveMaterial] = useState('');
  const [saveMatrix, setSaveMatrix] = useState('');
  // Free-text tags, comma separated. Carried by the API and the file format
  // since the feature shipped — and collected by no input, so every preset
  // saved through this dialog was born with `tags: []` and the tag filter in
  // the picker had nothing to filter on.
  const [saveTags, setSaveTags] = useState('');
  const fileRef = useRef(null);
  const [askDelete, confirmProps] = useConfirm();

  const refresh = useCallback(async (keep) => {
    try {
      const res = await edsExportApi.listPresets();
      const list = res?.data?.presets || [];
      setPresets(list);
      setListError(null);
      if (keep && list.some((p) => p.name === keep)) setSelected(keep);
      return list;
    } catch (e) {
      // Presets are an accelerator; the page works without them. Say so
      // once rather than breaking the panel.
      setPresets([]);
      setListError(errText(e));
      return [];
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const current = presets.find((p) => p.name === selected) || null;
  const readOnly = !!(current?.builtin ?? current?.read_only);

  // Every tag the library already uses, as completions for the save form.
  const knownTags = useMemo(() => collectTags(presets), [presets]);

  // A compatibility answer describes ONE scan, so it is DERIVED against the
  // file currently loaded rather than reset by an effect. Nothing used to
  // reset it at all: refuse on scan A, switch to B, and B still showed A's
  // blockers — press "apply anyway" and A's refusal was stamped with B's
  // path. It could never fake an `ok: true` (`doOverride` hardcodes
  // `ok: false`), but a refusal attributed to the wrong scan UNDERSTATES B's
  // own incompatibilities, which is the same lie pointed the other way.
  //
  // Deriving beats an effect for the same reason it does in `PhaseMapPanel`:
  // there is no render in which the stale answer is still live, and no
  // second copy to forget.
  const live = (checked.file || '') === (filePath || '') ? checked : NO_CHECK;
  const blockReport = live.report;
  const warnings = live.warnings;

  /** Fetch the preset and push it into the controls. */
  const applyResolved = useCallback(async (report) => {
    const res = await edsExportApi.getPreset(selected);
    const preset = res?.data?.preset || res?.data || {};
    const fields = applyPresetSettings(preset.settings || {}, handle);
    setChecked({ file: filePath || '', report: null, warnings: report?.warnings || [] });
    onApplied?.({ name: selected, compatibility: report || null });
    return fields;
  }, [selected, handle, onApplied, filePath]);

  const doApply = useCallback(async () => {
    if (!selected) { setStatus({ kind: 'warn', text: t('presets.noneToApply') }); return; }
    setBusy(true);
    setStatus(null);
    setChecked(NO_CHECK);
    try {
      // Check FIRST. Nothing reaches the controls before this answers.
      const chk = await edsExportApi.checkPreset(selected);
      const report = chk?.data || null;
      if (report && report.ok === false) {
        // Stamped with the file it was checked against, so it cannot be
        // read as an answer about the next one.
        setChecked({ file: filePath || '', report, warnings: [] });
        setStatus(null);
        return;
      }
      await applyResolved(report);
      setStatus({ kind: 'ok', text: t('presets.applied', { name: selected }) });
    } catch (e) {
      setStatus({ kind: 'error', text: errText(e) });
    } finally {
      setBusy(false);
    }
  }, [selected, applyResolved, filePath, t]);

  const doOverride = useCallback(async () => {
    // Deliberately reuses the stored refusal rather than re-checking: the
    // record that has to reach provenance.json is the refusal being
    // overridden, not whatever a second check happens to say.
    const report = { ...(blockReport || {}), ok: false, overridden: true };
    setBusy(true);
    try {
      await applyResolved(report);
      setStatus({ kind: 'warn', text: t('presets.appliedOverride', { name: selected }) });
    } catch (e) {
      setStatus({ kind: 'error', text: errText(e) });
    } finally {
      setBusy(false);
    }
  }, [blockReport, applyResolved, selected, t]);

  const doSave = useCallback(async () => {
    const name = saveName.trim();
    if (!name) return;
    setBusy(true);
    setStatus(null);
    try {
      await edsExportApi.savePreset({
        name,
        settings: presetSettingsFrom(handle),
        overwrite: true,
        author: saveAuthor.trim(),
        notes: saveNotes.trim(),
        // What this recipe was authored AGAINST. `check_compatibility` gates
        // `element_set_differs` on the first and `step_size_differs` on the
        // second, so a preset saved without them carries two warnings that
        // can never fire. Both are omitted rather than defaulted when this
        // scan cannot supply them — see `savePreset` in api.js.
        authoredElements: handle.authoredElements || [],
        authoredStepUm: handle.authoredStepUm ?? null,
        materialClass: saveMaterial.trim(),
        matrixElement: saveMatrix.trim(),
        // Parsed here, not in the request layer: what the user typed is a
        // line of text, and what the format stores is a list.
        tags: parseTags(saveTags),
      });
      setShowSave(false);
      setStatus({ kind: 'ok', text: t('presets.saved', { name }) });
      await refresh(name);
    } catch (e) {
      setStatus({ kind: 'error', text: errText(e) });
    } finally {
      setBusy(false);
    }
  }, [saveName, saveAuthor, saveNotes, saveMaterial, saveMatrix, saveTags,
      handle, refresh, t]);

  const doDelete = useCallback(async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await edsExportApi.deletePreset(selected);
      setSelected('');
      setChecked(NO_CHECK);
      await refresh();
    } catch (e) {
      setStatus({ kind: 'error', text: errText(e) });
    } finally {
      setBusy(false);
    }
  }, [selected, refresh]);

  const doImport = useCallback(async (file) => {
    if (!file) return;
    setBusy(true);
    setStatus(null);
    try {
      const text = await file.text();
      const res = await edsExportApi.importPreset(text);
      const name = res?.data?.preset?.name || '';
      setStatus({ kind: 'ok', text: t('presets.imported', { name }) });
      await refresh(name);
    } catch (e) {
      setStatus({ kind: 'error', text: errText(e) });
    } finally {
      setBusy(false);
    }
  }, [refresh, t]);

  const doExportFile = useCallback(async () => {
    if (!selected) return;
    setBusy(true);
    try {
      const res = await edsExportApi.exportPresetFile(selected);
      const text = res?.data?.json_text ?? '';
      const filename = res?.data?.filename || `${selected}.json`;
      await saveTextFile(text, filename);
    } catch (e) {
      setStatus({ kind: 'error', text: errText(e) });
    } finally {
      setBusy(false);
    }
  }, [selected]);

  const statusColour = { ok: C.green, warn: C.orange, error: C.red };

  return (
    <div
      data-preset-bar
      style={{
        display: 'flex', flexDirection: 'column', gap: 5,
        padding: '6px 8px', borderRadius: 4,
        border: `1px solid ${C.border}`, background: alpha(C.purple, 5),
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Label secondary small>{t('presets.title')}</Label>
        <PresetPicker
          presets={presets}
          selected={selected}
          t={t}
          onSelect={(name) => {
            setSelected(name);
            // A refusal describes ONE preset. Carrying it to the next
            // selection would refuse a preset nobody has checked yet.
            setChecked(NO_CHECK);
            setStatus(null);
          }}
        />
      </div>

      {/* Identity, and what the recipe is FOR.
          "Priya, 2026-08-27, 81aee0db" tells a reader who cannot ask Priya
          nothing about whether this is a steel recipe or an aluminium one.
          `notes` and `material_class` were saved, stored and returned by the
          API all along and rendered nowhere - the one thing a non-author
          needs in order to judge a preset she did not write. */}
      {current && (
        <div data-preset-identity style={{ fontSize: '8pt', color: C.textSecondary }}>
          <div>
            {presetMaterial(current)
              ? <span data-preset-material style={{ color: C.cyan }}>
                  {t('presets.forMaterial', { material: presetMaterial(current) })}
                </span>
              : <span style={{ color: C.orange }}>{t('presets.noMaterial')}</span>}
            {presetTags(current).length > 0 && (
              <span data-preset-tags>
                {` · ${presetTags(current).map((x) => `#${x}`).join(' ')}`}
              </span>
            )}
          </div>
          <div data-preset-notes style={{ fontStyle: 'italic' }}>
            {current.notes ? current.notes : t('presets.noNotes')}
          </div>
          <div>
            {current.author
              ? t('presets.by', { author: current.author })
              : <span style={{ color: C.orange }}>{t('presets.noAuthor')}</span>}
            {current.created ? ` · ${current.created}` : ''}
            {current.content_hash
              ? ` · ${t('presets.hash', { hash: shortHash(current.content_hash) })}`
              : ''}
            {readOnly ? ` · ${t('presets.builtin')}` : ''}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        <Button small variant="primary" onClick={doApply} disabled={busy || !selected}
                title={t('presets.applyTooltip')}>
          {busy ? t('presets.checking') : t('presets.apply')}
        </Button>
        <Button small
                onClick={() => {
                  setSaveName(selected || '');
                  // The rules already pin a matrix element when the user set
                  // one; offering it back is a suggestion, not a second
                  // source of truth — the backend falls back to exactly this
                  // when the field is blank.
                  setSaveMatrix((v) => v || handle.rules?.matrix_elements?.[0] || '');
                  setShowSave((v) => !v);
                }}
                disabled={busy} title={t('presets.saveAsTooltip')}>
          {t('presets.saveAs')}
        </Button>
        <Button small onClick={() => fileRef.current?.click()} disabled={busy}
                title={t('presets.importTooltip')}>
          {t('presets.import')}
        </Button>
        <Button small onClick={doExportFile} disabled={busy || !selected}
                title={t('presets.exportFileTooltip')}>
          {t('presets.exportFile')}
        </Button>
        <Button
          small
          variant="danger"
          disabled={busy || !selected || readOnly}
          title={readOnly ? t('presets.readOnly') : t('presets.deleteTooltip')}
          onClick={() => askDelete({
            title: t('presets.delete'),
            message: t('presets.deleteConfirm', { name: selected }),
            confirmLabel: t('presets.delete'),
            onConfirm: doDelete,
          })}
        >
          {t('presets.delete')}
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          data-preset-import-input
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = '';
            doImport(f);
          }}
        />
      </div>

      {/* --- the refusal. A panel, not a toast. -------------------------- */}
      {blockReport && (
        <div
          data-preset-blockers
          role="alert"
          style={{
            padding: '6px 8px', borderRadius: 4, fontSize: '8.5pt',
            color: C.text, background: alpha(C.red, 8),
            border: `1px solid ${alpha(C.red, 40)}`,
          }}
        >
          <div style={{ color: C.red, fontWeight: 600, marginBottom: 3 }}>
            {t('presets.blockedTitle')}
          </div>
          {(blockReport.blockers || []).map((b, i) => (
            <div key={b.code || i} style={{ marginBottom: 2 }}>· {b.message}</div>
          ))}
          <div style={{ color: C.textSecondary, marginTop: 4 }}>
            {t('presets.blockedHint')}
          </div>
          <div style={{ display: 'flex', gap: 4, marginTop: 5 }}>
            <Button small variant="danger" onClick={doOverride} disabled={busy}
                    title={t('presets.applyAnywayTooltip')}>
              {t('presets.applyAnyway')}
            </Button>
            <Button small
                    onClick={() => setChecked((c) => ({ ...c, report: null }))}
                    disabled={busy}>
              {t('presets.dismiss')}
            </Button>
          </div>
        </div>
      )}

      {/* Warnings inform; they never stop the apply. */}
      {warnings.length > 0 && (
        <div data-preset-warnings style={{ fontSize: '8pt', color: C.orange }}>
          <div>{t('presets.warnTitle')}</div>
          {warnings.map((w, i) => (
            <div key={w.code || i}>· {w.message}</div>
          ))}
        </div>
      )}

      {showSave && (
        <div data-preset-save-form style={{
          display: 'flex', flexDirection: 'column', gap: 4,
          padding: '5px 6px', borderRadius: 4,
          border: `1px solid ${alpha(C.cyan, 35)}`,
        }}>
          <Input
            value={saveName}
            aria-label={t('presets.saveName')}
            placeholder={t('presets.saveNamePlaceholder')}
            onChange={(e) => setSaveName(e.target.value)}
            style={{ height: 24, fontSize: '9pt' }}
          />
          <Input
            value={saveAuthor}
            aria-label={t('presets.saveAuthor')}
            placeholder={t('presets.saveAuthorPlaceholder')}
            onChange={(e) => setSaveAuthor(e.target.value)}
            style={{ height: 24, fontSize: '9pt' }}
          />
          <Input
            value={saveNotes}
            aria-label={t('presets.saveNotes')}
            placeholder={t('presets.saveNotesPlaceholder')}
            onChange={(e) => setSaveNotes(e.target.value)}
            style={{ height: 24, fontSize: '9pt' }}
          />
          <Input
            value={saveMaterial}
            aria-label={t('presets.saveMaterial')}
            placeholder={t('presets.saveMaterialPlaceholder')}
            title={t('presets.saveMaterialTooltip')}
            onChange={(e) => setSaveMaterial(e.target.value)}
            style={{ height: 24, fontSize: '9pt' }}
          />
          {/* Comma-separated free text, with the tags already in the library
              offered as completions. NOT a closed list: group, project and
              instrument names are exactly what she wants to tag by, and no
              enum can anticipate them. The datalist is only there so
              "as-cast" does not become "as cast" on the fortieth preset —
              a tag spelt two ways is two chips that each find half the
              presets. */}
          <Input
            value={saveTags}
            aria-label={t('presets.saveTags')}
            placeholder={t('presets.saveTagsPlaceholder')}
            title={t('presets.saveTagsTooltip')}
            list="eds-preset-tag-options"
            data-preset-save-tags
            onChange={(e) => setSaveTags(e.target.value)}
            style={{ height: 24, fontSize: '9pt' }}
          />
          <datalist id="eds-preset-tag-options">
            {knownTags.map((tag) => <option key={tag} value={tag} />)}
          </datalist>
          {/* A list, not a text field: the symbols come from what this scan
              measured, so a preset cannot be pinned to an element that is
              not there — and the Aztec line names ("Al Kα1") never reach the
              backend, which keys everything on plain symbols. */}
          <Select
            value={saveMatrix}
            aria-label={t('presets.saveMatrix')}
            title={t('presets.saveMatrixTooltip')}
            onChange={(e) => setSaveMatrix(e.target.value)}
            options={[
              { value: '', label: t('presets.saveMatrixNone') },
              ...(handle.authoredElements || []).map(
                (el) => ({ value: el, label: el })),
            ]}
            style={{ height: 24, fontSize: '9pt' }}
          />
          <Label secondary small style={{ display: 'block' }}>
            {t('presets.saveHint')}
          </Label>
          {/* Say what is being recorded ABOUT the scan, and say when it is
              not available. "Authored against nothing" is what makes the two
              compatibility warnings unfireable, and the user is the only one
              who can fix it — by running a classification first. */}
          <Label secondary small style={{ display: 'block' }}>
            {(handle.authoredElements || []).length
              ? (handle.authoredStepUm != null
                  ? t('presets.authoredAgainstStep', {
                      elements: (handle.authoredElements || []).join(', '),
                      step: Number(handle.authoredStepUm).toFixed(3),
                    })
                  : t('presets.authoredAgainst', {
                      elements: (handle.authoredElements || []).join(', '),
                    }))
              : t('presets.authoredAgainstNothing')}
          </Label>
          <div style={{ display: 'flex', gap: 4 }}>
            <Button small variant="primary" onClick={doSave}
                    disabled={busy || !saveName.trim()}>
              {t('presets.save')}
            </Button>
            <Button small onClick={() => setShowSave(false)} disabled={busy}>
              {t('presets.cancel')}
            </Button>
          </div>
        </div>
      )}

      {listError && (
        <div style={{ fontSize: '8pt', color: C.orange }}>
          {t('presets.loadFailed', { detail: listError })}
        </div>
      )}
      {status && (
        <div data-preset-status style={{ fontSize: '8pt', color: statusColour[status.kind] || C.textSecondary }}>
          {status.text}
        </div>
      )}

      {/* The "tolerance is not part of a preset" line is gone with the
          slider it described (PhaseMapPanel no longer draws one). The key is
          kept: the backend still records the value in provenance.json. */}

      <ConfirmDialog {...confirmProps} cancelLabel={t('presets.cancel')} />
    </div>
  );
}
