/**
 * ApexLoadDialog - choose which measurements of an EDAX APEX metadata file to open.
 *
 * The metadata `.h5` lists measurements and holds no patterns; the patterns of each are in a
 * `.up1`/`.up2` file whose header carries the measurement's ID. One table, one row per measurement
 * with a tick, an overview thumbnail, the measurement with
 * a Hexagonal/Square chip and the date its scan ended, its grid and step, and the state of its
 * pattern file; "Details" opens a second row under it.
 *
 * What can be opened is the backend's call (`loadable`, `problem` of `GET /api/ebsd/inspect`);
 * `apexLoad.js` turns that answer into rows and this component only shows them. A pattern file the
 * user chooses counts only after the backend has described it as fitting. The hexagonal line and
 * the list in Details come from the backend's own list of limits (`apexHexLimits.js`), never from
 * a list written here. Numbers are written in the language of the page. Everything that touches the
 * backend or the file system comes in as a prop, so the dialog is tested without either.
 */
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha } from '../../theme/components';
import { problemText } from '../../services/problemText';
import {
  buildLoadJobs, candidateAsChoice, defaultSelection, measurementTitle, mergeFolderSearch,
  patternSize, refusalOf, rowContext, rowKey, rowOf, scanEndDate, SIZE_MISMATCH, sizeMismatch, sizeOf,
  stepParts, summaryOf,
} from './apexLoad';
import { groupHexLimits } from './apexHexLimits';

const basename = (p) => String(p || '').split(/[\\/]/).pop();

/** "a, b and c" in the language of the page. */
function listText(items, language) {
  try {
    return new Intl.ListFormat(language || 'en', { style: 'long', type: 'conjunction' }).format(items);
  } catch {
    return items.join(', ');
  }
}

/** Number formats of the page's language: counts, free numbers, steps (3 digits), the pattern centre. */
function useNumberFormats(language) {
  return useMemo(() => {
    const make = (options) => {
      try { return new Intl.NumberFormat(language || 'en', options); } catch { return new Intl.NumberFormat('en', options); }
    };
    const count = make({});
    const free = make({ maximumFractionDigits: 2 });
    const step = make({ maximumSignificantDigits: 3 });
    const pc = make({ minimumFractionDigits: 4, maximumFractionDigits: 4 });
    return {
      count: (n) => count.format(Number(n)),
      free: (n) => free.format(Number(n)),
      step: (n) => step.format(Number(n)),
      pc: (n) => pc.format(Number(n)),
    };
  }, [language]);
}

/** A small picture of a measurement; shows `…` while it loads and `noneText` when it cannot be had. */
function Preview({ measurementId, kind, loadPreview, alt, title, width, height, noneText }) {
  const wanted = `${measurementId}|${kind}`;
  const [state, setState] = useState({ wanted: null, url: null, failed: false });
  useEffect(() => {
    let url = null;
    let cancelled = false;
    Promise.resolve()
      .then(() => loadPreview(measurementId, kind))
      .then((blob) => {
        if (cancelled || !blob || typeof URL.createObjectURL !== 'function') return;
        url = URL.createObjectURL(blob);
        setState({ wanted, url, failed: false });
      })
      .catch(() => { if (!cancelled) setState({ wanted, url: null, failed: true }); });
    return () => {
      cancelled = true;
      if (url && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(url);
    };
  }, [measurementId, kind, loadPreview, wanted]);
  // An answer for another measurement or kind is not this picture.
  const mine = state.wanted === wanted ? state : { url: null, failed: false };
  return mine.url ? (
    <img src={mine.url} alt={alt} title={title}
         style={{ width, height, objectFit: 'contain', imageRendering: 'pixelated',
                  border: `1px solid ${colors.border}`, background: colors.bg }} />
  ) : (
    <div title={title} role="img" aria-label={alt}
         style={{ width, height, border: `1px dashed ${colors.border}`, color: colors.textSecondary,
                  fontSize: '8pt', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  textAlign: 'center', boxSizing: 'border-box' }}>
      {mine.failed ? noneText : '…'}
    </div>
  );
}

export default function ApexLoadDialog({
  isOpen, h5Path, inspection, onClose, onLoad, choosePatternFile, chooseFolder, describePatternFile,
  searchFolder, loadPreview, loadHexLimits, escapeBlocked = false,
}) {
  const { t: tr, i18n } = useTranslation('ebsdviewer');
  // Every text of the dialog lives under `apexDialog` of the namespace, except the refusal codes,
  // which are `errors.<code>` there (the load window and this dialog share them).
  const t = useCallback(
    (key, opts) => tr(key.startsWith('errors.') ? key : `apexDialog.${key}`, opts), [tr]);
  const fmt = useNumberFormats(i18n.language);

  const base = useMemo(() => ({
    measurements: inspection?.measurements || [],
    unmatched: inspection?.unmatched_pattern_files || [],
    folder: inspection?.folder || '',
    folderOwn: true,
  }), [inspection]);
  const [searched, setSearched] = useState(null);           // {inspection, listing} after a folder search
  const [chosen, setChosen] = useState({});                 // accepted pattern files, by row
  const [attempts, setAttempts] = useState({});             // files being checked or refused, by row
  const [selected, setSelected] = useState([]);
  const [openRows, setOpenRows] = useState([]);
  const [searching, setSearching] = useState(false);
  const [folderError, setFolderError] = useState(null);
  const [hexData, setHexData] = useState(null);
  const [discardFor, setDiscardFor] = useState(null);
  const [seen, setSeen] = useState(null);
  // A new file starts everything over BEFORE it is first drawn (no flash of the previous file); an
  // answer that arrives for the old one is ignored (epoch, and a sequence per row).
  if (!seen || seen.inspection !== inspection || seen.h5Path !== h5Path) {
    setSeen({ inspection, h5Path });
    setSearched(null);
    setChosen({});
    setAttempts({});
    setOpenRows([]);
    setSearching(false);
    setFolderError(null);
    setDiscardFor(null);
    setSelected(defaultSelection(base.measurements, rowContext({}, base.unmatched)));
  }
  const listing = searched && searched.inspection === inspection ? searched.listing : base;
  const epoch = useRef(0);
  const choiceSeq = useRef({});
  const latest = useRef({});
  useEffect(() => {
    epoch.current += 1;
    choiceSeq.current = {};
  }, [h5Path, inspection]);
  useEffect(() => { latest.current = { listing, chosen }; });

  const list = listing.measurements;
  const ctx = useMemo(() => rowContext(chosen, listing.unmatched, attempts), [chosen, attempts, listing]);
  const jobs = useMemo(
    () => buildLoadJobs(h5Path, list, selected, ctx), [h5Path, list, selected, ctx]);
  const summary = useMemo(() => summaryOf(list, ctx), [list, ctx]);

  // Escape closes; with something ticked or chosen the first Esc only says so.
  const signature = `${selected.join('\n')}|${Object.keys(chosen).join('\n')}`;
  const hasChoices = selected.length > 0 || Object.keys(chosen).length > 0;
  const discardAsked = discardFor === signature;
  useEffect(() => {
    if (!isOpen) return undefined;
    const onKey = (e) => {
      if (e.key !== 'Escape' || e.defaultPrevented || escapeBlocked) return;
      if (hasChoices && !discardAsked) { setDiscardFor(signature); return; }
      onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose, escapeBlocked, hasChoices, discardAsked, signature]);

  const anyHex = list.some((m) => m.hex);
  useEffect(() => {
    if (!isOpen || !anyHex || typeof loadHexLimits !== 'function') return undefined;
    let cancelled = false;
    Promise.resolve()
      .then(() => loadHexLimits())
      .then((data) => { if (!cancelled) setHexData(groupHexLimits(data)); })
      .catch(() => { if (!cancelled) setHexData(null); });
    return () => { cancelled = true; };
  }, [isOpen, anyHex, loadHexLimits]);

  const toggle = useCallback((key) => {
    setSelected((prev) => (prev.includes(key) ? prev.filter((x) => x !== key) : [...prev, key]));
  }, []);
  const toggleDetails = useCallback((key) => {
    setOpenRows((prev) => (prev.includes(key) ? prev.filter((x) => x !== key) : [...prev, key]));
  }, []);

  const choose = useCallback(async (m) => {
    const key = rowKey(m);
    const mine = epoch.current;
    const path = await choosePatternFile();
    if (!path || mine !== epoch.current) return;
    const name = basename(path);
    choiceSeq.current[key] = (choiceSeq.current[key] || 0) + 1;
    const seq = choiceSeq.current[key];
    const current = () => mine === epoch.current && choiceSeq.current[key] === seq;
    setAttempts((a) => ({ ...a, [key]: { state: 'checking', path, name } }));
    try {
      const entry = await describePatternFile(m, path);
      if (!current()) return;
      // A file that carries no measurement ID is tied to the row by its count alone: another pattern
      // size proves it is not this measurement's. A file that names the measurement keeps the ID's word.
      const sizeOff = entry?.matches_grid && entry.pairing !== 'guid' ? sizeMismatch(m, entry) : null;
      if (sizeOff) {
        setAttempts((a) => ({ ...a, [key]: { state: 'refused', path, name, entry: { ...entry, matches_grid: false, problem: SIZE_MISMATCH } } }));
      } else if (entry?.matches_grid) {
        setChosen((c) => ({ ...c, [key]: { state: 'ok', path, name, entry } }));
        setAttempts((a) => { const { [key]: _gone, ...rest } = a; return rest; });
        setSelected((prev) => (prev.includes(key) ? prev : [...prev, key]));
      } else {
        setAttempts((a) => ({ ...a, [key]: { state: 'refused', path, name, entry } }));
      }
    } catch (err) {
      if (!current()) return;
      setAttempts((a) => ({
        ...a, [key]: { state: 'error', path, name, message: problemText(err, tr, 'ebsdviewer') },
      }));
    }
  }, [choosePatternFile, describePatternFile, tr]);

  const pickCandidate = useCallback((m, file) => {
    const key = rowKey(m);
    choiceSeq.current[key] = (choiceSeq.current[key] || 0) + 1;
    setChosen((c) => ({ ...c, [key]: candidateAsChoice(file) }));
    setAttempts((a) => { const { [key]: _gone, ...rest } = a; return rest; });
    setSelected((prev) => (prev.includes(key) ? prev : [...prev, key]));
  }, []);

  const searchElsewhere = useCallback(async () => {
    const mine = epoch.current;
    const folder = await chooseFolder();
    if (!folder || mine !== epoch.current) return;
    setSearching(true);
    setFolderError(null);
    try {
      const data = await searchFolder(folder);
      if (mine !== epoch.current) return;
      const { listing: now, chosen: accepted } = latest.current;
      const merged = mergeFolderSearch(now, data, accepted);
      setSearched({ inspection, listing: { ...merged, folder: data?.folder || folder, folderOwn: false } });
    } catch (err) {
      if (mine !== epoch.current) return;
      setFolderError(problemText(err, tr, 'ebsdviewer'));
    } finally {
      if (mine === epoch.current) setSearching(false);
    }
  }, [chooseFolder, searchFolder, tr, inspection]);

  if (!isOpen) return null;
  const file = basename(h5Path);
  const errorText = (code) => t(`errors.${code}`, { defaultValue: code });
  // "40 rows × 30 points", always in words: "(40 × 30)" was read as columns × rows
  const gridText = (m) => t('grid', {
    rows: t('rowsCount', { count: m.n_rows }), points: t('pointsCount', { count: m.n_cols }) });
  /** "This measurement needs a pattern file with 2,000 patterns (50 rows × 40 points)." or null. */
  const needsText = (m) => {
    const expected = Number(m.n_rows) * Number(m.n_cols);
    return expected > 0 ? t('pair.needs', { count: expected, grouped: fmt.count(expected), grid: gridText(m) }) : null;
  };
  const titleOfKey = (key) => {
    const owner = list.find((x) => rowKey(x) === key);
    return owner ? measurementTitle(owner) : '';
  };

  /** What the pattern-file cell says for a pattern file the backend described and refused. */
  const refusalText = (entry, m) => {
    const name = entry?.name || '';
    const r = refusalOf(entry, m, list);
    if (!r) return '';
    if (r.kind === 'tooFew' || r.kind === 'tooMany') {
      const text = t(`pair.${r.kind}`, { name, count: r.have, grouped: fmt.count(r.have) });
      // what the measurement needs first, then what the file holds, then (too short) where a file comes from
      return `${needsText(m)} ${text}${r.kind === 'tooFew' ? ` ${t('pair.reexport')}` : ''}`;
    }
    if (r.kind === 'size') {
      const px = (z) => ({ w: fmt.count(z.width), h: fmt.count(z.height) });
      return t('pair.sizeMismatch', { name, fw: px(r.have).w, fh: px(r.have).h, mw: px(r.need).w, mh: px(r.need).h });
    }
    if (r.kind === 'otherMeasurement') {
      return r.other ? t('pair.otherMeasurement', { name, other: r.other })
                     : t('pair.otherUnknown', { name });
    }
    return t('pair.code', { name, reason: errorText(r.code) });
  };

  /** {tone, symbol, core, subs[]}: the state of a row's pattern file, in words. */
  const pairingOfRow = (row, m) => {
    const f = row.file;
    const needs = needsText(m);
    if (row.state === 'blocked') {
      const damaged = ['apexUnreadable', 'apexBadId'].includes(row.problem);
      return {
        tone: 'bad', symbol: '✗',
        // a damaged entry says what to do: the other measurements of the file still open
        subs: damaged ? [t('pair.damagedHelp', { file })] : [],
        core: damaged ? t('pair.damaged', { file }) : t('pair.blocked', { reason: errorText(row.problem) }),
      };
    }
    if (row.state === 'ready' || row.state === 'chosen') {
      const byId = f.pairing === 'guid';
      const params = { count: f.n_patterns, grouped: fmt.count(f.n_patterns) };
      const key = row.state === 'ready'
        ? (byId ? 'pair.matchedById' : 'pair.matchedByCount')
        : (byId ? 'pair.chosenById' : 'pair.chosenByCount');
      const core = row.state === 'ready' ? f.name : t('pair.chosen', { name: f.name });
      // matched by ID: green. Matched by count only, nothing proves it is this measurement's file:
      // it keeps a warning look and says so, also after the user has chosen it
      if (byId) return { tone: 'ok', symbol: '✓', subs: [t(key, params)], core };
      const both = sizeOf(m) && sizeOf(f);
      return {
        tone: 'warn', symbol: '!', core,
        subs: [t(key, params), t('pair.candidateWarning'), both ? t('pair.sizeCompare', compareParams(m, f)) : null].filter(Boolean),
      };
    }
    if (row.state === 'wrong') {
      // too few / too many already begin with what the measurement needs
      const says = ['tooFew', 'tooMany'].includes(refusalOf(f, m, list)?.kind);
      return { tone: 'bad', symbol: '✗', core: refusalText(f, m), subs: says ? [] : [needs].filter(Boolean) };
    }
    if (row.state === 'folder') {
      return { tone: 'warn', symbol: '', core: t('pair.folderFailed'), subs: [t('pair.folderFailedHelp')] };
    }
    return {
      tone: 'warn', symbol: '', core: t('pair.missing'),
      subs: [needs, t('pair.missingHelp'), t('pair.reexport')].filter(Boolean),
    };
  };

  const compareParams = (m, f) => {
    const a = sizeOf(m);
    const b = sizeOf(f);
    return { mw: fmt.count(a.width), mh: fmt.count(a.height), fw: fmt.count(b.width), fh: fmt.count(b.height) };
  };

  const toneColor = (tone) => ({ ok: colors.green, warn: colors.orange, bad: colors.red }[tone] || colors.text);
  const sub = { color: colors.textSecondary, fontSize: '8.5pt' };

  // ONE short sentence in the row, from the backend's list: that indexing works. Everything else,
  // saving included, is in Details.
  const hexLine = (grouping) => (grouping ? t('hexRowNone') : t('hexRowGeneric'));

  /** What the user cannot do with a limit, in user terms: the kinds it touches, never an option name. */
  const itemNotes = (item) => {
    const notes = [];
    if (item.affects.length) {
      notes.push(t('detail.hexOnlyFor', {
        list: listText(item.affects.map((a) => t(`hexLimits.affects.${a}`, { defaultValue: a })), i18n.language),
      }));
    } else if (item.partial && item.parameters.length) {
      notes.push(t('detail.hexSomeSettings'));
    }
    return notes;
  };

  const hexDetails = (grouping) => {
    const categories = grouping ? grouping.categories : [];
    const technical = categories.flatMap((c) => c.items).filter((i) => i.parameters.length > 0);
    const keyName = (key) => t(`hexLimits.keys.${key}`, { defaultValue: key });
    return (
      <>
        <dt>{t('detail.hexGrid')}</dt>
        <dd>
          <p style={{ margin: '0 0 4px 0' }}>{t('detail.hexLayout')}</p>
          <p style={{ margin: '0 0 4px 0' }}>{t('detail.hexMapDrawn')}</p>
          {grouping && <p style={{ margin: '0 0 4px 0' }}>{t('detail.hexNotAffected')}</p>}
          <p style={{ margin: '0 0 4px 0' }}>{t('detail.hexSaving')}</p>
          {grouping ? (
            <>
              {grouping.works.length > 0 && (
                <p style={{ margin: '0 0 4px 0' }}>
                  {t('detail.hexStillWorks', {
                    list: listText(grouping.works.map((k) => t(`hexLimits.works.${k}`, { defaultValue: k })), i18n.language),
                  })}
                </p>
              )}
              {categories.length > 0 && (
                <>
                  <p style={{ margin: '0 0 2px 0' }}>{t('detail.hexOff')}</p>
                  <ul data-testid="hex-limit-list" style={{ margin: '0 0 4px 0', paddingLeft: 18 }}>
                    {categories.map((c) => (
                      <li key={c.category}>
                        <b>{t(`hexLimits.categories.${c.category}`, { defaultValue: c.category })}</b>
                        <ul style={{ margin: 0, paddingLeft: 16 }}>
                          {c.items.map((item) => {
                            const notes = itemNotes(item);
                            return (
                              <li key={item.key}>{notes.length ? `${keyName(item.key)}: ${notes.join('; ')}` : keyName(item.key)}</li>
                            );
                          })}
                        </ul>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {technical.length > 0 && (
                <details data-testid="hex-technical" style={{ margin: 0 }}>
                  <summary>{t('detail.hexTechnical')}</summary>
                  {technical.map((item) => (
                    <p key={item.key} style={{ margin: '2px 0 0 0' }}>
                      {t('detail.hexTechnicalLine', {
                        name: keyName(item.key),
                        list: listText(item.parameters.map((q) => t(`hexLimits.parameters.${q}`, { defaultValue: q })), i18n.language),
                      })}
                    </p>
                  ))}
                </details>
              )}
            </>
          ) : (
            <p style={{ margin: 0 }}>{t('detail.hexListMissing')}</p>
          )}
        </dd>
      </>
    );
  };

  /** What can be checked about a file by eye: its pattern count and, from its header, its pattern size. */
  const fileFacts = (m, c) => {
    const size = patternSize(c);
    const p = { count: c.n_patterns, grouped: fmt.count(c.n_patterns) };
    if (size && sizeOf(m)) {
      const own = sizeOf(m);
      return t('pair.candidateFactsCompare', { ...p, width: fmt.count(size.width), height: fmt.count(size.height),
        mw: fmt.count(own.width), mh: fmt.count(own.height) });
    }
    return size
      ? t('pair.candidateFacts', { ...p, width: fmt.count(size.width), height: fmt.count(size.height) })
      : t('pair.candidateFactsNoSize', p);
  };

  /** Buttons "Use <file>" with what can be checked about each; a file another row took says so. */
  const fileButtons = (m, files) => files.map((c) => (
    c.usedFor ? (
      <div key={c.path} style={sub}>
        {t('pair.usedFor', { name: c.name, row: titleOfKey(c.usedFor) })}
      </div>
    ) : (
      <div key={c.path} style={{ marginTop: 2 }}>
        <button type="button" onClick={() => pickCandidate(m, c)} title={t('pair.useCandidateTooltip')}
                style={{ fontSize: '9pt', padding: '1px 8px', marginRight: 6, cursor: 'pointer' }}>
          {t('pair.useCandidate', { name: c.name })}
        </button>
        <span>{fileFacts(m, c)}</span>
      </div>
    )));

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex',
                  alignItems: 'center', justifyContent: 'center', zIndex: 9998 }}>
      <div role="dialog" aria-modal="true" aria-label={t('title', { file })}
           style={{ background: colors.bgSecondary, border: `1px solid ${colors.border}`, borderRadius: 8,
                    padding: 18, width: 'min(980px, 94vw)', maxHeight: '92vh', overflow: 'auto',
                    boxSizing: 'border-box', boxShadow: '0 8px 40px rgba(0,0,0,0.7)', color: colors.text }}>
        <div style={{ fontSize: '12pt', fontWeight: 600, marginBottom: 6 }}>{t('title', { file })}</div>
        <p data-testid="apex-top-note" style={{ ...sub, fontSize: '9.5pt', margin: '0 0 4px 0' }}>{t('topNote')}</p>
        <p style={{ ...sub, fontSize: '9.5pt', margin: '0 0 4px 0' }}>{t('intro')}</p>
        <p data-testid="apex-folder" style={{ ...sub, margin: '0 0 4px 0' }}>
          {t('folderLabel')}{' '}
          <code style={{ color: colors.text }}>{listing.folder}</code>{' '}
          {listing.folderOwn ? t('folderOwn', { file }) : t('folderChosen')}{' '}
          <button type="button" onClick={searchElsewhere} disabled={searching}
                  title={t('searchOtherTooltip')}
                  style={{ fontSize: '9pt', padding: '1px 8px', cursor: searching ? 'wait' : 'pointer' }}>
            {searching ? t('searching') : t('searchOther')}
          </button>
        </p>
        {folderError && (
          <div role="alert" style={{ color: colors.red, fontSize: '9pt', margin: '2px 0 4px 0' }}>{folderError}</div>
        )}
        {summary.total > 0 && (
          <p data-testid="apex-summary" style={{ fontSize: '9.5pt', margin: '6px 0 10px 0' }}>
            {summary.ready === summary.total
              ? t('summary.allReady', { count: summary.total })
              : t('summary.line', {
                count: summary.total,
                parts: [
                  summary.ready > 0 ? t('summary.ready', { count: summary.ready }) : null,
                  summary.needFile > 0 ? t('summary.needFile', { count: summary.needFile }) : null,
                  summary.wrongFile > 0 ? t('summary.wrongFile', { count: summary.wrongFile }) : null,
                  summary.blocked > 0 ? t('summary.blocked', { count: summary.blocked }) : null,
                ].filter(Boolean).join(t('summary.separator')),
              })}
            {' '}{t('summary.ticked', { count: jobs.length })}
          </p>
        )}
        {discardAsked && (
          <div role="status" style={{ color: colors.orange, fontSize: '9pt', margin: '0 0 6px 0' }}>
            {t('discardHint')}
          </div>
        )}

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '9.5pt', minWidth: 680 }}>
            <thead>
              <tr style={{ color: colors.textSecondary, textAlign: 'left' }}>
                <th style={{ padding: '6px 8px' }} />
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>{t('columns.overview')}</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>{t('columns.measurement')}</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>{t('columns.size')}</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>{t('columns.patternFile')}</th>
                <th style={{ padding: '6px 8px' }} />
              </tr>
            </thead>
            <tbody>
              {list.map((m, index) => {
                const key = rowKey(m);
                const row = rowOf(m, ctx);
                const pair = pairingOfRow(row, m);
                const title = measurementTitle(m);
                const date = scanEndDate(m);
                const { step, rowStep } = stepParts(m);
                const size = patternSize(row.file);
                const detailsOpen = openRows.includes(key);
                const reasonId = `apex-pair-${index}`;
                const cell = { padding: '7px 8px', borderBottom: `1px solid ${colors.border}`, verticalAlign: 'top' };
                const attempt = row.attempt;
                const attemptText = attempt && (attempt.state === 'checking'
                  ? t('pair.checking', { name: attempt.name })
                  : attempt.state === 'error' ? attempt.message : refusalText(
                    { ...(attempt.entry || {}), name: attempt.name }, m));
                const canChoose = row.state !== 'blocked' && !(row.state === 'ready' && row.extra === 0);
                // a damaged entry or a grid that is not valid has no grid to name
                const noGrid = ['apexUnreadable', 'apexBadId', 'apexUnknownGrid'].includes(m.measurement_problem)
                  || m.consistent === false;
                const showHex = m.hex && row.state !== 'blocked' && !noGrid;
                // a damaged entry has nothing to show beyond the problem: no Details toggle
                const damagedEntry = ['apexUnreadable', 'apexBadId'].includes(m.measurement_problem);
                const pcOk = Array.isArray(m.pc) && m.pc.length === 3;
                return (
                  <Fragment key={key}>
                    <tr data-testid={`measurement-${key}`}>
                      <td style={cell}>
                        <input type="checkbox" checked={selected.includes(key) && row.loadable}
                               disabled={!row.loadable} onChange={() => toggle(key)}
                               aria-label={t('select', { name: title })}
                               aria-describedby={reasonId}
                               title={row.loadable ? undefined : [pair.core, ...pair.subs].filter(Boolean).join(' ')} />
                      </td>
                      <td style={cell}>
                        {m.has_sem && m.id ? (
                          <Preview measurementId={m.id} kind="sem" loadPreview={loadPreview}
                                   alt={t('thumbnail.alt', { name: title })} title={t('thumbnail.title')}
                                   width={64} height={48} noneText={t('thumbnail.failed')} />
                        ) : (
                          <div style={{ ...sub, width: 64, height: 48, border: `1px dashed ${colors.border}`,
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        textAlign: 'center', fontSize: '8pt', boxSizing: 'border-box' }}>
                            {t('thumbnail.none')}
                          </div>
                        )}
                      </td>
                      <td style={cell}>
                        <b>{title}</b>
                        {!noGrid && (
                          <span style={{ display: 'inline-block', fontSize: '8.5pt', padding: '0 6px', borderRadius: 8,
                                         marginLeft: 6, border: `1px solid ${m.hex ? colors.accent : colors.border}`,
                                         color: m.hex ? colors.accent : colors.textSecondary }}>
                            {m.hex ? t('hex') : t('square')}
                          </span>
                        )}
                        {date && <div style={sub}>{t('scanEnded', { date })}</div>}
                        {showHex && (
                          <div data-testid={`hexline-${key}`}
                               style={{ color: colors.accent, fontSize: '8.5pt', marginTop: 3 }}>
                            {hexLine(hexData)} {t('hexRowAssumes')}
                          </div>
                        )}
                      </td>
                      <td style={cell}>
                        {noGrid || !(m.n_rows > 0 && m.n_cols > 0) ? (
                          <span style={sub}>—</span>
                        ) : (
                          <>
                            {gridText(m)}
                            <div style={sub}>
                              {step ? (rowStep
                                ? t(m.hex ? 'stepRowsHex' : 'stepRows',
                                  { step: fmt.step(step), rowStep: fmt.step(rowStep), factor: fmt.step(0.866) })
                                : t('step', { step: fmt.step(step) })) : null}
                              {size ? <>{step ? ' · ' : ''}<span title={t('patternSizeTooltip')}>
                                {t('patternSize', { width: fmt.count(size.width), height: fmt.count(size.height) })}</span></> : null}
                            </div>
                          </>
                        )}
                      </td>
                      <td style={cell} id={reasonId}>
                        <span style={{ color: toneColor(pair.tone) }}
                              title={damagedEntry && m.problem_detail ? m.problem_detail : undefined}>
                          {pair.symbol ? `${pair.symbol} ` : ''}{pair.core}
                        </span>
                        {pair.subs.map((line) => <div key={line} style={sub}>{line}</div>)}
                        {row.extra > 0 && (
                          <div style={sub}>
                            {t('pair.more', { count: row.extra, names: row.others.map((f) => f.name).join(', ') })}
                          </div>
                        )}
                        {row.candidates.length > 0 && row.state !== 'ready' && row.state !== 'chosen' && (
                          <div style={{ ...sub, marginTop: 4 }}>
                            {t('pair.candidates')}
                            {fileButtons(m, row.candidates)}
                            <div style={{ color: colors.orange, marginTop: 3 }}>{t('pair.candidateWarning')}</div>
                          </div>
                        )}
                        {row.sizeRejected.map((f) => {
                          const own = sizeOf(m);
                          const size = sizeOf(f);
                          return (
                            <div key={f.path} style={{ ...sub, marginTop: 4 }}>
                              {t('pair.sizeRejected', { name: f.name, fw: fmt.count(size.width), fh: fmt.count(size.height),
                                mw: fmt.count(own.width), mh: fmt.count(own.height) })}
                            </div>
                          );
                        })}
                        {row.alternatives.length > 0 && (
                          <div style={{ ...sub, marginTop: 4 }}>
                            {t('pair.alternatives', { folder: m.alternativesFolder })}
                            {fileButtons(m, row.alternatives)}
                          </div>
                        )}
                        {attemptText && (
                          <div role="alert" style={{ color: attempt.state === 'checking' ? colors.textSecondary : colors.red,
                                                     fontSize: '8.5pt', marginTop: 4 }}>
                            {attempt.state === 'refused' ? '✗ ' : ''}{attemptText}
                          </div>
                        )}
                        {canChoose && (
                          <div style={{ marginTop: 4 }}>
                            <button type="button" onClick={() => choose(m)}
                                    disabled={attempt?.state === 'checking'}
                                    style={{ fontSize: '9pt', padding: '1px 8px', cursor: 'pointer' }}>
                              {row.state === 'ready' || row.state === 'chosen' ? t('pair.chooseAnother') : t('pair.choose')}
                            </button>
                          </div>
                        )}
                      </td>
                      <td style={cell}>
                        {!damagedEntry && (
                          <button type="button" onClick={() => toggleDetails(key)} aria-expanded={detailsOpen}
                                  style={{ fontSize: '9pt', padding: '1px 8px', cursor: 'pointer' }}>
                            {detailsOpen ? t('detailsLess') : t('details')}
                          </button>
                        )}
                      </td>
                    </tr>
                    {detailsOpen && !damagedEntry && (
                      <tr data-testid={`details-${key}`} style={{ background: alpha(colors.border, 20) }}>
                        <td />
                        <td colSpan={5} style={{ padding: '7px 8px', borderBottom: `1px solid ${colors.border}`,
                                                 color: colors.textSecondary, fontSize: '9pt' }}>
                          <dl style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '3px 12px', margin: 0 }}>
                            {showHex && hexDetails(hexData)}
                            {pcOk && (
                              <>
                                <dt>{t('detail.patternCentre')}</dt>
                                <dd style={{ margin: 0, color: colors.text }}>
                                  {t('detail.patternCentreValue', {
                                    x: fmt.pc(m.pc[0]), y: fmt.pc(m.pc[1]), z: fmt.pc(m.pc[2]) })}
                                  <div style={{ color: colors.textSecondary }}>
                                    {t(m.hex ? 'detail.patternCentreNoteHex' : 'detail.patternCentreNote')}
                                  </div>
                                </dd>
                              </>
                            )}
                            {!m.measurement_problem && (
                              <>
                                <dt>{t('detail.geometry')}</dt>
                                <dd style={{ margin: 0, color: colors.text }}>
                                  {t('detail.geometryValue', { tilt: fmt.free(m.sample_tilt), elevation: fmt.free(m.elevation) })}
                                  {' · '}
                                  {m.beam_kv ? t('detail.beamValue', { kv: fmt.free(m.beam_kv) }) : t('detail.beamUnknown')}
                                </dd>
                              </>
                            )}
                            {(m.phases || []).length > 0 ? (
                              <>
                                <dt>{t('detail.phases')}</dt>
                                <dd style={{ margin: 0, color: colors.text }}>
                                  {(m.phases || []).map((p) => p.name || p.formula).join(', ')}
                                  {' '}<span style={{ color: colors.textSecondary }}>({t('detail.phasesNote')})</span>
                                </dd>
                              </>
                            ) : (
                              <dd style={{ margin: 0, gridColumn: '1 / -1' }}>{t('detail.phasesNoneLine')}</dd>
                            )}
                            {date && (
                              <>
                                <dt>{t('detail.scanEnded')}</dt>
                                <dd style={{ margin: 0, color: colors.text }}>{t('detail.scanEndedValue', { date })}</dd>
                              </>
                            )}
                            {m.id && (
                              <>
                                <dt>{t('detail.measurementId')}</dt>
                                <dd style={{ margin: 0, color: colors.text }}>
                                  <code>{m.id}</code>
                                  <div style={{ color: colors.textSecondary }}>{t('detail.measurementIdNote')}</div>
                                </dd>
                              </>
                            )}
                            {m.problem_detail && (
                              <>
                                <dt>{t('detail.technical')}</dt>
                                <dd style={{ margin: 0, color: colors.text }}><code>{m.problem_detail}</code></dd>
                              </>
                            )}
                            <dt>{t('detail.imageQuality')}</dt>
                            <dd style={{ margin: 0 }}>
                              {m.has_iq && m.id ? (
                                <>
                                  <Preview measurementId={m.id} kind="iq" loadPreview={loadPreview}
                                           alt={t('detail.imageQuality')} title={t('detail.imageQualityCaption')}
                                           width={128} height={96} noneText={t('thumbnail.failed')} />
                                  <div>{m.hex ? t('detail.imageQualityHexCaption') : t('detail.imageQualityCaption')}</div>
                                </>
                              ) : t('detail.imageQualityNone')}
                            </dd>
                          </dl>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 12,
                      marginTop: 14, flexWrap: 'wrap' }}>
          <div style={{ ...sub, maxWidth: 560 }}>
            <div>{t('footer.opensHere')}</div>
            <div>{t('footer.stayClosed', { file })}</div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" onClick={onClose}
                    style={{ padding: '6px 14px', cursor: 'pointer', fontSize: '10pt' }}>
              {t('cancel')}
            </button>
            {jobs.length === 0 && (
              <span data-testid="apex-load-hint" style={{ ...sub, alignSelf: 'center' }}>
                {summary.ready === 0 ? t('loadHintNone') : t('loadHint')}
              </span>
            )}
            <button type="button" disabled={jobs.length === 0} onClick={() => onLoad(jobs)}
                    style={{ padding: '6px 14px', fontSize: '10pt',
                             cursor: jobs.length === 0 ? 'not-allowed' : 'pointer' }}>
              {t('load', { count: jobs.length })}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
