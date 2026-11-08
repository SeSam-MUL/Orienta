/**
 * LoadDialog - choose which measurement and which kind of patterns of an HDF5 EBSD file to open.
 *
 * Shown only for a file that offers more than one variant (a measurement with one pattern source;
 * `GET /api/ebsd/inspect` answers `kind: 'choices'` for exactly those, and the backend decides what
 * is a choice). One row per measurement; inside it one option per pattern source, all in ONE group of
 * radio buttons, because a file is open as one variant at a time. Each source says how many patterns
 * it holds, their size and bit depth, whether it has a background image and, for a source that cannot
 * be opened, why. A small pattern from the middle of the scan helps to tell the sources apart; a
 * preview that cannot be had never blocks the dialog.
 *
 * Nothing is chosen for the user beyond the preselection: the variant opened last time for this file
 * (`remembered`) or, failing that, what a plain load takes. Enter opens the preselection. Choosing the
 * unprocessed patterns changes nothing else: the dialog only points out that they are not background
 * corrected. An EDAX APEX file has its own dialog (`ApexLoadDialog.jsx`, which also pairs pattern files).
 */
import { useCallback, useEffect, useMemo, useRef, useState, useId } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { codedText } from '../../services/problemText';
import { choiceJobs, choiceRows, defaultChoice, limitConcurrent, sourceBits } from './loadChoices';

const basename = (p) => String(p || '').split(/[\\/]/).pop();

/** How many pictures are asked for at the same moment. */
const PREVIEWS_AT_ONCE = 2;

/** Stands in for a preview function that was not given (one identity, so the effect does not rerun). */
const NO_PREVIEW = () => Promise.reject(new Error('no preview function'));

/** What Tab can reach in the dialog, in order: of a group of radio buttons only the chosen one. */
const tabStops = (root) => [...(root?.querySelectorAll('input:not([disabled]), button:not([disabled])') || [])]
  .filter((el) => el.type !== 'radio' || el.checked);

/** A small picture of one source; `…` while it loads, `noneText` when it cannot be had. */
function Preview({ path, measurement, source, loadPreview, queue, alt, title, noneText }) {
  // The file is part of the key: another file has measurements and sources of the same names.
  const wanted = `${path}|${measurement}|${source}`;
  const [state, setState] = useState({ wanted: null, url: null, failed: false });
  useEffect(() => {
    let url = null;
    let cancelled = false;
    // Leaving (another file, the dialog closing) aborts the question: a waiting one never starts.
    const controller = new AbortController();
    queue(() => loadPreview(measurement, source, controller.signal), controller.signal)
      .then((blob) => {
        if (cancelled) return;
        if (!blob || typeof URL.createObjectURL !== 'function') throw new Error('no preview');
        url = URL.createObjectURL(blob);
        setState({ wanted, url, failed: false });
      })
      .catch(() => { if (!cancelled) setState({ wanted, url: null, failed: true }); });
    return () => {
      cancelled = true;
      controller.abort();
      if (url && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(url);
    };
  }, [measurement, source, loadPreview, queue, wanted]);
  // An answer for another file, measurement or source is not this picture.
  const mine = state.wanted === wanted ? state : { url: null, failed: false };
  const box = { width: 96, height: 72, flex: 'none' };
  return mine.url ? (
    <img src={mine.url} alt={alt} title={title}
         style={{ ...box, objectFit: 'contain', imageRendering: 'pixelated',
                  border: `1px solid ${colors.border}`, background: colors.bg }} />
  ) : (
    <div title={title} role="img" aria-label={alt}
         style={{ ...box, border: `1px dashed ${colors.border}`, color: colors.textSecondary, fontSize: '8pt',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center',
                  boxSizing: 'border-box' }}>
      {mine.failed ? noneText : '…'}
    </div>
  );
}

export default function LoadDialog({
  isOpen, path, inspection, remembered = null, onClose, onLoad, loadPreview, escapeBlocked = false,
}) {
  const { t: tr, i18n } = useTranslation('ebsdviewer');
  const t = useCallback((key, opts) => tr(`loadDialog.${key}`, opts), [tr]);
  const group = `load-variant-${useId()}`;
  const dialogRef = useRef(null);

  const rows = useMemo(() => choiceRows(inspection), [inspection]);
  // One queue per preview function (that is per file): the pictures are asked for a few at a time.
  const previewQueue = useMemo(() => limitConcurrent(PREVIEWS_AT_ONCE), [loadPreview]);
  const [choice, setChoice] = useState(null);
  const [seen, setSeen] = useState(null);
  const [wasRemembered, setWasRemembered] = useState(false);
  // A new file starts over BEFORE it is first drawn (no flash of the previous file's choice).
  if (!seen || seen.inspection !== inspection || seen.path !== path) {
    const first = defaultChoice(inspection, remembered);
    setSeen({ inspection, path });
    setChoice(first ? { measurement: first.measurement, pattern_source: first.pattern_source } : null);
    setWasRemembered(!!first?.remembered);
  }

  const jobs = useMemo(() => choiceJobs(path, inspection, choice), [path, inspection, choice]);
  const canLoad = jobs.length > 0;
  const load = useCallback(() => { if (canLoad) onLoad(jobs); }, [canLoad, onLoad, jobs]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape' && !e.defaultPrevented && !escapeBlocked) onClose();
      // Keep Tab inside the dialog (it is modal); a question on top of it (escapeBlocked) has its own Tab.
      if (e.key === 'Tab' && !e.defaultPrevented && !escapeBlocked) {
        const root = dialogRef.current;
        const stops = tabStops(root);
        if (!root || stops.length === 0) return;
        const first = stops[0];
        const last = stops[stops.length - 1];
        const at = document.activeElement;
        if (!root.contains(at)) {
          e.preventDefault();
          (e.shiftKey ? last : first).focus();
        } else if (e.shiftKey && at === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && at === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose, escapeBlocked]);

  // Whatever had the focus when the dialog opened gets it back when the dialog closes.
  // (Declared before the effect below: it must read the focus before the dialog takes it.)
  useEffect(() => {
    if (!isOpen) return undefined;
    const opener = document.activeElement;
    return () => {
      if (opener && typeof opener.focus === 'function' && document.contains(opener)) opener.focus();
    };
  }, [isOpen]);

  // The preselected option has the focus, so Enter opens it at once; with nothing to open, Cancel has it.
  useEffect(() => {
    if (!isOpen) return;
    const root = dialogRef.current;
    (root?.querySelector('input[type="radio"]:checked') || root?.querySelector('[data-cancel]'))?.focus();
  }, [isOpen, path, inspection]);

  const nf = useMemo(() => {
    try { return new Intl.NumberFormat(i18n.language || 'en'); } catch { return new Intl.NumberFormat('en'); }
  }, [i18n.language]);

  if (!isOpen) return null;
  const file = basename(path);
  const sub = { color: colors.textSecondary, fontSize: '8.5pt' };
  const sourceName = (key) => t(`source.${key}`, { defaultValue: key });
  const bitsText = (source) => {
    const bits = sourceBits(source);
    if (bits.kind === 'none') return null;
    return bits.kind === 'bits' ? t('bits.other', { bits: bits.bits }) : t(`bits.${bits.kind}`);
  };
  const onDialogKey = (e) => {
    // Enter opens the preselection; a focused button keeps its own meaning (Cancel, Open).
    if (e.key !== 'Enter' || e.target?.tagName === 'BUTTON') return;
    e.preventDefault();
    load();
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex',
                  alignItems: 'center', justifyContent: 'center', zIndex: 9998 }}>
      <div ref={dialogRef} role="dialog" aria-modal="true" aria-label={t('title', { file })} onKeyDown={onDialogKey}
           style={{ background: colors.bgSecondary, border: `1px solid ${colors.border}`, borderRadius: 8,
                    padding: 18, width: 'min(760px, 94vw)', maxHeight: '92vh', overflow: 'auto',
                    boxSizing: 'border-box', boxShadow: '0 8px 40px rgba(0,0,0,0.7)', color: colors.text }}>
        <div style={{ fontSize: '12pt', fontWeight: 600, marginBottom: 6 }}>{t('title', { file })}</div>
        <p style={{ ...sub, fontSize: '9.5pt', margin: '0 0 4px 0' }}>{t('intro')}</p>
        <p data-testid="load-pc-note" style={{ ...sub, fontSize: '9.5pt', margin: '0 0 8px 0' }}>{t('pcNote')}</p>
        {wasRemembered && choice && (
          <p role="status" style={{ color: colors.accent, fontSize: '9pt', margin: '0 0 8px 0' }}>{t('remembered')}</p>
        )}
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '9.5pt' }}>
          <thead>
            <tr style={{ color: colors.textSecondary, textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', fontWeight: 500 }}>{t('columns.measurement')}</th>
              <th style={{ padding: '6px 8px', fontWeight: 500 }}>{t('columns.patterns')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => {
              const cell = { padding: '7px 8px', borderBottom: `1px solid ${colors.border}`, verticalAlign: 'top' };
              return (
                <tr key={row.name} data-testid={`measurement-${row.name}`}>
                  <td style={cell}>
                    <b>{row.label}</b>
                    {row.nRows > 0 && row.nCols > 0 && (
                      <div style={sub}>
                        {tr('apexDialog.grid', {
                          rows: tr('apexDialog.rowsCount', { count: row.nRows }),
                          points: tr('apexDialog.pointsCount', { count: row.nCols }) })}
                      </div>
                    )}
                  </td>
                  <td style={cell}>
                    {row.sources.map((s, si) => {
                      const checked = !!choice && choice.measurement === row.name && choice.pattern_source === s.key;
                      const name = sourceName(s.key);
                      const facts = [
                        s.nPatterns > 0 ? t('patternCount', { count: s.nPatterns, grouped: nf.format(s.nPatterns) }) : null,
                        s.size ? t('patternSize', { width: nf.format(s.size.width), height: nf.format(s.size.height) }) : null,
                      ].filter(Boolean).join(' · ');
                      const bits = bitsText(s);
                      const factsId = `${group}-facts-${ri}-${si}`;
                      const problem = !s.loadable ? (codedText(s.problem, tr, 'ebsdviewer') || s.problem || '') : '';
                      return (
                        <label key={s.key} style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 8,
                                                    opacity: s.loadable ? 1 : 0.6,
                                                    cursor: s.loadable ? 'pointer' : 'not-allowed' }}>
                          <input type="radio" name={group} checked={checked} disabled={!s.loadable}
                                 onChange={() => { if (s.loadable) setChoice({ measurement: row.name, pattern_source: s.key }); }}
                                 aria-label={t('option', { measurement: row.label, source: name })}
                                 aria-describedby={factsId}
                                 style={{ marginTop: 4 }} />
                          <Preview path={path} measurement={row.name} source={s.key}
                                   loadPreview={typeof loadPreview === 'function' ? loadPreview : NO_PREVIEW} queue={previewQueue}
                                   alt={t('preview.alt', { measurement: row.label, source: name })}
                                   title={t('preview.title')} noneText={t('preview.none')} />
                          <div>
                            <b>{name}</b>
                            <div id={factsId}>
                              {facts && <div style={sub}>{facts}</div>}
                              {bits && <div style={sub}>{bits}</div>}
                              <div style={sub}>{s.hasBackground ? t('background.has') : t('background.hasNot')}</div>
                              {s.key === 'unprocessed' && (
                                <div style={{ ...sub, color: colors.orange }}>{t('rawHint')}</div>
                              )}
                              {problem && <div role="alert" style={{ color: colors.red, fontSize: '8.5pt' }}>{problem}</div>}
                            </div>
                          </div>
                        </label>
                      );
                    })}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 8, marginTop: 14 }}>
          {!canLoad && (
            <span data-testid="load-none" style={{ ...sub, alignSelf: 'center' }}>{t('noneOpenable')}</span>
          )}
          <button type="button" data-cancel="" onClick={onClose} style={{ padding: '6px 14px', cursor: 'pointer', fontSize: '10pt' }}>
            {t('cancel')}
          </button>
          <button type="button" disabled={!canLoad} onClick={load}
                  style={{ padding: '6px 14px', fontSize: '10pt', cursor: canLoad ? 'pointer' : 'not-allowed' }}>
            {t('load')}
          </button>
        </div>
      </div>
    </div>
  );
}
