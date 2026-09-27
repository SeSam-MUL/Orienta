/**
 * Get the EDS numbers out of the app, with everything needed to defend them.
 *
 * Six user personas were asked what they need before this was designed and
 * five of six ranked export above presets, unprompted. The single most
 * repeated point, in six phrasings: the settings must live INSIDE the
 * exported file, not be referenced by a preset name. That part is the
 * backend's `provenance.json`; this dialog's job is to make the choice that
 * feeds it, and to be honest about what the choice produces.
 *
 * Three things here are not decoration:
 *
 *  1. **The counts are shown before anything is written.** `export/preview`
 *     is a dry run, so the dialog can say "1 842 particles" rather than
 *     making the user export to find out what they would get.
 *  2. **A missing step size is stated, loudly.** Without it every µm column
 *     is left out — `area_um2`, `ecd_um`, the lot. A wrong scale is worse
 *     than no scale, so the omission is deliberate, and an omission the user
 *     has to discover in the CSV is a bug report.
 *  3. **Comma-decimal with comma-delimiter is refused before the request.**
 *     Two personas independently raised the German-Excel case: `3.14` read
 *     as `314` is a wrong number in a report. With both set to a comma every
 *     value would additionally split across two columns.
 *
 * `min_particle_px` never deletes rows — it sets `below_size_limit` and the
 * counts still include them. Silent exclusion was named a disqualifier, so
 * the note beside the field says so rather than leaving it to the manual.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { edsExportApi } from '../../services/api';
import { fileLine } from './writtenFileLine';
import {
  colors as C, alpha, Button, NumberInput, Input, Label,
} from '../../theme/components';

/** A value no translation can equal, so a missing key is detectable.
 *  i18next here has returnEmptyString:false, which makes '' unusable. */
const I18N_MISS = '\u0000miss';

/**
 * The artefacts, in the order the folder lists them.
 *
 * `labels` and `pixels` default off because both are large and neither is
 * needed to answer "what phases, how much, how big" — the questions the
 * other five tables exist for.
 */
export const ARTEFACTS = [
  { id: 'phases', on: true },
  { id: 'regions', on: true },
  { id: 'particles', on: true },
  { id: 'definitions', on: true },
  { id: 'xlsx', on: true },
  { id: 'labels', on: false },
  { id: 'pixels', on: false },
];

export const DEFAULT_ARTEFACTS = Object.fromEntries(
  ARTEFACTS.map((a) => [a.id, a.on]),
);

/**
 * Would this pair of separators produce ambiguous numbers?
 *
 * Only the comma/comma case: a comma decimal with a semicolon or a tab is
 * exactly what a German Excel expects, and a dot decimal never collides.
 */
export function separatorClash(decimal, delimiter) {
  return decimal === ',' && delimiter === ',';
}

/** Are these settings safe to send? Pure, so the test does not need a DOM. */
export function canRunExport({ destDir, decimal, delimiter, busy }) {
  if (busy) return false;
  if (!String(destDir || '').trim()) return false;
  return !separatorClash(decimal, delimiter);
}

const n = (v) => (Number.isFinite(Number(v)) ? Number(v).toLocaleString() : '\u2014');

/* ------------------------------------------------------------------------ *
 * The one line somebody pastes under a figure.
 *
 * Two testers asked for it independently - a failure analyst for figure
 * captions, a group leader for a grant report - and both were left to
 * assemble it by hand out of provenance.json. Everything in it is already in
 * the export response; nothing here computes a new number, and no clause is
 * printed with a placeholder. A clause whose value the backend did not send
 * is OMITTED, because a caption that says "- x - um" is a caption somebody
 * pastes into a paper.
 * ------------------------------------------------------------------------ */

/**
 * A number, or null when there is not one.
 *
 * NOT `Number(v)`: `Number(null)` is 0 and `Number('')` is 0, so a missing
 * step size would print "0 um step" and a missing count would print "0
 * regions" - a placeholder wearing the clothes of a measurement, which is
 * the one thing this summary must never do.
 */
export function num(v) {
  if (v === null || v === undefined || v === '') return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

/** A number for prose: at most two decimals, no trailing zeros, no locale. */
export function fmtNum(v, dp = 2) {
  const x = num(v);
  if (x === null) return null;
  return String(Number(x.toFixed(dp)));
}

/** File name without directory or extension. Null rather than a guess. */
export function scanName(path) {
  const raw = String(path || '').trim();
  if (!raw) return null;
  const base = raw.split(/[\\/]/).pop() || '';
  return base.replace(/\.[^.]+$/, '') || null;
}

/** "Orienta 2026-08-27 (f1080b69)" out of whatever the version block holds. */
export function appLabel(app) {
  if (!app) return null;
  const parts = [app.app, app.version].filter(Boolean).map(String);
  if (!parts.length) return null;
  const line = parts.join(' ');
  const commit = app.commit ? String(app.commit) : '';
  return (commit && !line.includes(commit)) ? `${line} (${commit})` : line;
}

/**
 * The clauses of the summary line, as translation keys plus their values.
 *
 * Returned rather than rendered so the decision "which clauses does this
 * export support" is testable without a DOM and without a locale.
 */
export function summaryClauses(result) {
  const prov = result?.provenance || {};
  const grid = prov.grid || {};
  const step = prov.step || {};
  const counts = prov.counts || {};
  const cls = prov.classification || {};
  const sm = cls.smoothing || {};
  const parts = prov.particles || {};
  const out = [];

  // 1. what was measured, and how big it is.
  const name = scanName(prov.source?.path);
  const rows = num(grid.n_rows);
  const cols = num(grid.n_cols);
  const sx = num(step.x_um);
  const sy = num(step.y_um);
  const haveGrid = rows !== null && cols !== null;
  const haveStep = sx !== null && sx > 0 && sy !== null && sy > 0;
  if (haveGrid && haveStep) {
    const vars = {
      width: fmtNum(cols * sx, 1), height: fmtNum(rows * sy, 1),
      step: fmtNum(sx, 3),
    };
    out.push(name ? { key: 'export.sum.scan', vars: { name, ...vars } }
                  : { key: 'export.sum.sizeUm', vars });
  } else if (haveGrid) {
    const vars = { cols: String(cols), rows: String(rows) };
    out.push(name ? { key: 'export.sum.scanPx', vars: { name, ...vars } }
                  : { key: 'export.sum.sizePx', vars });
  } else if (name) {
    out.push({ key: 'export.sum.scanOnly', vars: { name } });
  }

  // 2. regions to phases - the two-level structure of this map.
  const nReg = num(counts.n_regions);
  const nPh = num(counts.n_phases_with_pixels);
  if (nReg !== null && nPh !== null) {
    out.push({ key: 'export.sum.regions',
               vars: { regions: String(nReg), phases: String(nPh) } });
  }

  // 3. particles. The two qualifiers are only printed when the backend
  //    actually counted them - they are what makes a size distribution
  //    defensible, and inventing either would be worse than silence.
  const nPart = num(counts.n_particles);
  if (nPart !== null) {
    const below = num(counts.n_particles_below_size_limit);
    const edge = num(counts.n_particles_touching_edge);
    const limit = num(parts.min_particle_px);
    if (below !== null && edge !== null && limit !== null && limit > 0) {
      out.push({
        key: 'export.sum.particlesFlagged',
        vars: {
          particles: String(nPart), below: String(below),
          limit: String(limit), edge: String(edge),
        },
      });
    } else {
      out.push({ key: 'export.sum.particles', vars: { particles: String(nPart) } });
    }
  }

  // 4. the averaging box. "5 px" is not a length a reader can judge; the
  //    micrometre width is the number that says whether the box was larger
  //    than the features being counted.
  const px = num(sm.scale_px);
  const box = num(sm.box_um);
  if (px !== null && px > 0) {
    out.push(box !== null && box > 0
      ? { key: 'export.sum.smoothing', vars: { px: String(px), um: fmtNum(box, 2) } }
      : { key: 'export.sum.smoothingPx', vars: { px: String(px) } });
  }

  // 5. which build produced it.
  const app = appLabel(prov.app);
  if (app) out.push({ key: 'export.sum.app', vars: { app } });

  // 6. reproducibility, only when the record actually claims it.
  const det = prov.determinism || {};
  if (det.deterministic === true && det.random_state != null) {
    out.push({ key: 'export.sum.deterministic',
               vars: { seed: String(det.random_state) } });
  }

  // 7. the caveat that must not be dropped when the numbers are quoted:
  //    standardless Cliff-Lorimer, no ZAF, no standard calibration.
  out.push({ key: 'export.sum.semiQuant', vars: {} });
  return out;
}

/** The clauses, translated and joined. */
export function buildSummaryLine(result, t) {
  return summaryClauses(result)
    .map(({ key, vars }) => t(key, vars))
    .filter((x) => x && String(x).trim())
    .join(' ');
}

/**
 * Put text on the clipboard, with the pre-`navigator.clipboard` path kept.
 *
 * Electron's renderer has the async API; a plain browser over http:// (the
 * dev server) does not, and there the button would silently do nothing.
 */
export async function copyText(text, node) {
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the selection path */
  }
  try {
    if (node?.select) {
      node.select();
      return !!document.execCommand?.('copy');
    }
  } catch {
    /* nothing else to try */
  }
  return false;
}

function Section({ title, children, style: extra = {} }) {
  return (
    <div style={{ marginBottom: 12, ...extra }}>
      <div style={{
        fontSize: '8.5pt', color: C.textSecondary, marginBottom: 4,
        textTransform: 'uppercase', letterSpacing: '.04em',
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function Check({ id, checked, onChange, label, hint }) {
  return (
    <label
      title={hint}
      style={{
        display: 'flex', alignItems: 'flex-start', gap: 6,
        fontSize: '9pt', color: C.text, cursor: 'pointer', marginBottom: 3,
      }}
    >
      <input
        type="checkbox"
        data-artefact={id}
        checked={!!checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: C.purple, marginTop: 2, flexShrink: 0 }}
      />
      <span>{label}</span>
    </label>
  );
}

/** Two-or-three-way radio row. A `<select>` hides the alternatives. */
function Choice({ name, value, onChange, options }) {
  return (
    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }} role="radiogroup" aria-label={name}>
      {options.map((o) => {
        const on = String(o.value) === String(value);
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={on}
            data-choice={`${name}:${o.value}`}
            onClick={() => onChange(o.value)}
            title={o.title}
            style={{
              fontSize: '8.5pt', padding: '2px 10px', borderRadius: 3,
              cursor: 'pointer',
              background: on ? alpha(C.purple, 30) : 'transparent',
              color: on ? C.text : C.textSecondary,
              border: `1px solid ${alpha(C.purple, on ? 70 : 25)}`,
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * An export warning, in the reader's language.
 *
 * The backend has sent a stable `code` beside its English `message` since the
 * export was built — and nothing read it: the dialog printed the message, so
 * a German user got English warnings under a German heading. The M5 tester
 * reported exactly that (report, part B, row "B2 Export").
 *
 * Warnings whose text is fixed are translated. The ones that quote an
 * exception, a file name or a count still fall back to the English sentence,
 * because the backend does not send those pieces separately yet; that is
 * visible here rather than hidden, and the fallback is a true sentence.
 */
export function warnText(w, t) {
  if (!w) return '';
  if (!w.code) return w.message || '';
  // See I18N_MISS: an empty defaultValue returns the key, not ''.
  const out = t(`export.warn.${w.code}`, { defaultValue: I18N_MISS });
  return out === I18N_MISS ? (w.message || '') : out;
}

/**
 * Pick artefacts, number locale and destination; write the folder.
 *
 * `presetName` / `compatibility` are carried straight through to the request
 * so a preset applied over a refusal lands in the provenance rather than
 * being forgotten the moment the dialog closed.
 */
export default function ExportDialog({
  open, onClose, presetName = '', compatibility = null, onExported,
}) {
  const { t } = useTranslation('eds');

  const [artefacts, setArtefacts] = useState(DEFAULT_ARTEFACTS);
  const [decimal, setDecimal] = useState('.');
  const [delimiter, setDelimiter] = useState(',');
  const [connectivity, setConnectivity] = useState(8);
  const [minParticlePx, setMinParticlePx] = useState(0);
  const [destDir, setDestDir] = useState('');

  const [preview, setPreview] = useState(null);
  const [previewErr, setPreviewErr] = useState(null);
  const [previewing, setPreviewing] = useState(false);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [copied, setCopied] = useState(null);   // null | true | false
  const summaryRef = useRef(null);

  // Built from what the export actually returned, so it cannot describe a
  // run that did not happen.
  const summary = useMemo(
    () => (result ? buildSummaryLine(result, t) : ''), [result, t]);

  const doCopy = useCallback(async () => {
    setCopied(await copyText(summary, summaryRef.current));
  }, [summary]);

  // Fresh state on every open. A dialog that remembers last time's
  // destination but not last time's counts shows one and implies the other.
  useEffect(() => {
    if (!open) return;
    setError(null);
    setResult(null);
    setPreviewErr(null);
    setCopied(null);
  }, [open]);

  // The dry run. Re-read when the two options that change the particle COUNT
  // change, on a short delay so typing a limit does not fire per keystroke.
  useEffect(() => {
    if (!open) return undefined;
    let cancelled = false;
    setPreviewing(true);
    const timer = setTimeout(() => {
      (async () => {
        try {
          const res = await edsExportApi.exportPreview({ connectivity, minParticlePx });
          if (!cancelled) { setPreview(res?.data || null); setPreviewErr(null); }
        } catch (e) {
          if (!cancelled) {
            setPreview(null);
            setPreviewErr(e?.response?.data?.detail || e?.message || String(e));
          }
        } finally {
          if (!cancelled) setPreviewing(false);
        }
      })();
    }, 200);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [open, connectivity, minParticlePx]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape' && !busy) { e.stopPropagation(); onClose?.(); }
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [open, busy, onClose]);

  const browse = useCallback(async () => {
    // An export destination: the next save dialog starts here too.
    const picked = await window?.electronAPI?.openFolder?.({ remember: true });
    if (picked) setDestDir(picked);
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await edsExportApi.runExport({
        destDir: destDir.trim(),
        artefacts,
        decimal,
        delimiter,
        connectivity,
        minParticlePx,
        presetName,
        compatibility,
      });
      setResult(res?.data || null);
      setCopied(null);
      onExported?.(res?.data || null);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }, [destDir, artefacts, decimal, delimiter, connectivity, minParticlePx,
      presetName, compatibility, onExported]);

  if (!open) return null;

  const clash = separatorClash(decimal, delimiter);
  const ready = canRunExport({ destDir, decimal, delimiter, busy });
  const areaMissing = preview && preview.area_available === false;
  const hasElectronFolder = typeof window !== 'undefined'
    && !!window.electronAPI?.openFolder;

  return createPortal(
    <div
      data-eds-export-backdrop
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'fixed', inset: 0, zIndex: 3600,
        background: 'rgba(0,0,0,.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 20,
      }}
    >
      <div
        data-eds-export-dialog
        role="dialog"
        aria-modal="true"
        aria-label={t('export.title')}
        style={{
          background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8,
          width: 'min(620px, 96vw)', maxHeight: '92vh',
          display: 'flex', flexDirection: 'column',
          boxShadow: '0 24px 64px rgba(0,0,0,.5)', color: C.text,
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 14px', borderBottom: `1px solid ${C.border}`,
          flexShrink: 0,
        }}>
          <span style={{ color: C.cyan, fontWeight: 600, fontSize: '11pt' }}>
            {t('export.title')}
          </span>
          <div style={{ flex: 1 }} />
          <Button small onClick={() => !busy && onClose?.()} title={t('export.cancel')}>✕</Button>
        </div>

        {/* Body */}
        <div style={{ padding: 14, overflowY: 'auto', flex: 1, minHeight: 0 }}>
          {/* --- what you would get, before writing anything --------------- */}
          <Section title={t('export.whatYouGet')}>
            <div
              data-export-preview
              style={{
                border: `1px solid ${C.border}`, borderRadius: 4,
                padding: '6px 8px', fontSize: '9pt',
                background: alpha(C.cyan, 6),
              }}
            >
              {previewErr ? (
                <span style={{ color: C.red }}>
                  {t('export.previewFailed', { detail: previewErr })}
                </span>
              ) : previewing && !preview ? (
                <span style={{ color: C.textSecondary }}>{t('export.previewLoading')}</span>
              ) : preview ? (
                <>
                  <div>
                    {t('export.counts', {
                      particles: n(preview.n_particles),
                      regions: n(preview.n_regions),
                      phases: n(preview.n_phases),
                    })}
                  </div>
                  <div style={{ color: C.textSecondary, fontSize: '8.5pt' }}>
                    {t('export.classified', {
                      classified: n(preview.n_classified_px),
                      total: n(preview.n_total_px),
                    })}
                  </div>
                </>
              ) : (
                <span style={{ color: C.textSecondary }}>{t('export.previewLoading')}</span>
              )}
            </div>

            {/* An omission the user meets in the CSV is a bug report. */}
            {areaMissing && (
              <div
                data-export-area-warning
                role="status"
                style={{
                  marginTop: 6, padding: '6px 8px', borderRadius: 4,
                  fontSize: '8.5pt', lineHeight: 1.45,
                  color: C.text,
                  background: alpha(C.orange, 10),
                  border: `1px solid ${alpha(C.orange, 45)}`,
                }}
              >
                <strong style={{ color: C.orange }}>{t('export.areaMissingTitle')}</strong>
                <div>{t('export.areaMissing')}</div>
              </div>
            )}
            {preview && preview.area_available && (
              <Label secondary small style={{ display: 'block', marginTop: 4 }}>
                {t('export.areaOk', {
                  x: preview.step_x_um, y: preview.step_y_um,
                })}
              </Label>
            )}
            {(preview?.warnings || []).length > 0 && (
              <div data-export-preview-warnings style={{ marginTop: 6 }}>
                {preview.warnings.map((w, i) => (
                  <div key={`${w.code || 'w'}-${i}`} style={{ fontSize: '8.5pt', color: C.orange }}>
                    · {warnText(w, t)}
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* --- which files ------------------------------------------------ */}
          <Section title={t('export.artefacts')}>
            {ARTEFACTS.map((a) => (
              <Check
                key={a.id}
                id={a.id}
                checked={artefacts[a.id]}
                onChange={(v) => setArtefacts((s) => ({ ...s, [a.id]: v }))}
                label={t(`export.art.${a.id}`)}
                hint={t(`export.artTip.${a.id}`)}
              />
            ))}
            <Label secondary small style={{ display: 'block', marginTop: 4 }}>
              {t('export.alwaysWritten')}
            </Label>
          </Section>

          {/* --- numbers ---------------------------------------------------- */}
          <Section title={t('export.locale')}>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}>
                  {t('export.decimal')}
                </Label>
                <Choice
                  name="decimal"
                  value={decimal}
                  onChange={setDecimal}
                  options={[
                    { value: '.', label: t('export.decimalDot') },
                    { value: ',', label: t('export.decimalComma') },
                  ]}
                />
              </div>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}>
                  {t('export.delimiter')}
                </Label>
                <Choice
                  name="delimiter"
                  value={delimiter}
                  onChange={setDelimiter}
                  options={[
                    { value: ',', label: t('export.delimComma') },
                    { value: ';', label: t('export.delimSemicolon') },
                    { value: '\t', label: t('export.delimTab') },
                  ]}
                />
              </div>
            </div>
            {clash && (
              <div
                data-export-separator-clash
                role="alert"
                style={{
                  marginTop: 6, padding: '5px 8px', borderRadius: 4,
                  fontSize: '8.5pt', color: C.red,
                  background: alpha(C.red, 8),
                  border: `1px solid ${alpha(C.red, 40)}`,
                }}
              >
                {t('export.clash')}
              </div>
            )}
          </Section>

          {/* --- particles --------------------------------------------------- */}
          <Section title={t('export.particles')}>
            <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}
                       title={t('export.connectivityTooltip')}>
                  {t('export.connectivity')}
                </Label>
                <Choice
                  name="connectivity"
                  value={connectivity}
                  onChange={(v) => setConnectivity(Number(v))}
                  options={[
                    { value: 4, label: t('export.conn4'), title: t('export.conn4Tooltip') },
                    { value: 8, label: t('export.conn8'), title: t('export.conn8Tooltip') },
                  ]}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <Label secondary small>{t('export.minPx')}</Label>
                <NumberInput
                  value={minParticlePx}
                  min={0}
                  aria-label={t('export.minPx')}
                  onChange={(e) => setMinParticlePx(Math.max(0, Number(e.target.value) || 0))}
                  style={{ width: 76 }}
                />
                <span style={{ fontSize: '8.5pt', color: C.textSecondary }}>
                  {t('export.minPxUnit')}
                </span>
              </div>
            </div>
            <Label secondary small style={{ display: 'block', marginTop: 4 }}>
              {t('export.minPxNote')}
            </Label>
          </Section>

          {/* --- where ------------------------------------------------------- */}
          <Section title={t('export.dest')}>
            <div style={{ display: 'flex', gap: 6 }}>
              <Input
                value={destDir}
                aria-label={t('export.dest')}
                placeholder={t('export.destPlaceholder')}
                onChange={(e) => setDestDir(e.target.value)}
                style={{ flex: 1, minWidth: 0 }}
              />
              {hasElectronFolder && (
                <Button small onClick={browse} title={t('export.browse')}>
                  {t('export.browse')}
                </Button>
              )}
            </div>
            {!destDir.trim() && (
              <Label secondary small style={{ display: 'block', marginTop: 3 }}>
                {t('export.destNeeded')}
              </Label>
            )}
          </Section>

          {presetName && (
            <Label secondary small style={{ display: 'block' }}>
              {compatibility && compatibility.ok === false
                ? t('export.overrideNote', { name: presetName })
                : t('export.presetNote', { name: presetName })}
            </Label>
          )}

          {/* Which scan the compatibility report was checked against, and
              when. A report that names no file is unfalsifiable: it looks
              authoritative for whatever scan happens to be loaded. */}
          {compatibility?.checked_file && (
            <div
              data-export-checked-against
              style={{ fontSize: '9pt', color: C.textSecondary, wordBreak: 'break-all' }}
            >
              {t('export.checkedAgainst', {
                file: compatibility.checked_file,
                when: compatibility.checked_at || '',
              })}
            </div>
          )}

          {/* --- result ------------------------------------------------------ */}
          {result && (
            <div
              data-export-result
              style={{
                marginTop: 10, padding: '8px 10px', borderRadius: 4,
                background: alpha(C.green, 8),
                border: `1px solid ${alpha(C.green, 40)}`,
              }}
            >
              <div style={{ fontSize: '9pt', marginBottom: 4 }}>
                {t('export.doneTitle')}
              </div>
              <div style={{
                fontFamily: 'monospace', fontSize: '8.5pt',
                color: C.cyan, wordBreak: 'break-all', marginBottom: 6,
              }}>
                {result.folder}
              </div>
              {(result.files || []).map((f) => (
                <div key={f.name} style={{ fontSize: '8.5pt', color: C.textSecondary }}>
                  {fileLine(f, t, n)}
                </div>
              ))}

              {/* The pasteable line. Read-only and selectable, so the copy
                  button is a convenience rather than the only way out. */}
              {summary && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: '8pt', color: C.textSecondary, marginBottom: 3 }}>
                    {t('export.summaryTitle')}
                  </div>
                  <textarea
                    ref={summaryRef}
                    data-export-summary
                    readOnly
                    value={summary}
                    aria-label={t('export.summaryTitle')}
                    rows={3}
                    onFocus={(e) => e.target.select()}
                    style={{
                      width: '100%', boxSizing: 'border-box', resize: 'vertical',
                      fontSize: '8.5pt', lineHeight: 1.45, padding: '5px 6px',
                      background: C.bg, color: C.text,
                      border: `1px solid ${C.border}`, borderRadius: 3,
                    }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                    <Button small onClick={doCopy} title={t('export.copyTooltip')}>
                      {t('export.copy')}
                    </Button>
                    <span data-export-copy-status style={{
                      fontSize: '8pt',
                      color: copied === false ? C.orange : C.green,
                    }}>
                      {copied === true ? t('export.copied')
                        : copied === false ? t('export.copyFailed') : ''}
                    </span>
                  </div>
                  <Label secondary small style={{ display: 'block', marginTop: 2 }}>
                    {t('export.summaryHint')}
                  </Label>
                </div>
              )}
              {(result.warnings || []).length > 0 && (
                <div style={{ marginTop: 5 }}>
                  <div style={{ fontSize: '8.5pt', color: C.orange }}>
                    {t('export.warningsTitle')}
                  </div>
                  {result.warnings.map((w, i) => (
                    <div key={`${w.code || 'w'}-${i}`} style={{ fontSize: '8.5pt', color: C.orange }}>
                      · {warnText(w, t)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '10px 14px', borderTop: `1px solid ${C.border}`,
          flexShrink: 0,
        }}>
          {error && (
            <span data-export-error style={{ color: C.red, fontSize: '8.5pt' }}>
              {t('export.failed', { detail: error })}
            </span>
          )}
          <div style={{ flex: 1 }} />
          <Button onClick={() => !busy && onClose?.()} disabled={busy}>
            {result ? t('export.close') : t('export.cancel')}
          </Button>
          <Button
            variant="primary"
            onClick={run}
            disabled={!ready}
            title={clash ? t('export.clash') : t('export.run')}
          >
            {busy ? t('export.running') : t('export.run')}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
