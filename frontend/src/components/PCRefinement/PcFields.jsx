// The PC fields of the Detector Settings: a person types the three numbers of a PC in the kind of numbers
// they have (Bruker, EDAX/TSL, Oxford), sees what Orienta will take them as, and presses "Use these numbers".
// Typing alone applies nothing. The fields start empty; the PC in use stands beside them, in the same kind of
// numbers. Every conversion is pcApi.convert (kikuchipy's own), never a formula in the page.
import React, { useState, useEffect, useRef } from 'react';
import { useTranslation, Trans } from 'react-i18next';
import { pcApi } from '../../services/api';
import { problemText } from '../../services/problemText';
import { colors, alpha, spacing, Button } from '../../theme/components';
import PcGrant, { PcUndoDone } from './PcGrant';
import { samePc, formatPc } from './pcFormat';
import { pcMapActionOf } from './pcMapNote';
import { CONVENTIONS, parseNum, fmt4, patternSizeOf } from './pcConvention';

const AXES = ['x', 'y', 'z'];
const ECHO_DELAY_MS = 300;

const inputStyle = {
  width: 90, background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 4, color: colors.text,
  fontSize: '10pt', padding: '4px 8px', height: spacing.buttonHeight, outline: 'none', boxSizing: 'border-box',
};
const noteStyle = { fontSize: '9pt', color: colors.textSecondary, margin: '2px 0', wordBreak: 'break-word' };
const rowStyle = { display: 'grid', gridTemplateColumns: '110px 1fr', gap: '4px 12px', alignItems: 'start', margin: '6px 0' };
const labelStyle = { fontSize: '9pt', color: colors.textSecondary, paddingTop: 6 };
const cellStyle = { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '4px 10px', minWidth: 0 };
const fullWidth = { flex: '1 1 100%', margin: 0, fontSize: '9pt', color: colors.textSecondary };

/** The text of the badge beside the heading: null (nothing typed), 'unused' or 'inUse'. */
export function PcBadge({ state }) {
  const { t } = useTranslation('pcrefinement');
  if (!state) return null;
  const inUse = state === 'inUse';
  const colour = inUse ? colors.green : colors.orange;
  return (
    <span
      data-testid="pc-typed-badge"
      style={{
        fontSize: '9pt', fontWeight: 400, color: colour, border: `1px solid ${colour}`, borderRadius: 10,
        padding: '0 8px', marginLeft: 8, whiteSpace: 'nowrap', verticalAlign: 'middle',
      }}
    >
      {inUse ? t('convention.badgeInUse') : t('convention.badgeUnused')}
    </span>
  );
}

/**
 * @param {string|null}   fileName    the open file (its dataset name)
 * @param {string}        resetKey    changes with the open file: the fields, the select and the lines under them start again
 * @param {number|string} rows, cols  the pattern size of the open file
 * @param {number[]|null|undefined} usedPc  the PC in use (Bruker x, y, z), null when the file has none, undefined
 *                                    while the backend has not said yet (nothing is claimed beside the fields)
 * @param {string|null}   fileConvention  the convention the open file's vendor uses, when the backend knows it
 * @param {Function}      onBadge     `(null | 'unused' | 'inUse') => void`, for the heading of the group
 * @param {Function}      onUse       `async (brukerPc) => {data} | {error}`: applies the PC, as the page always did
 * @param {Function}      onUndo      `async (file) => {data} | {gone} | {error}`: one step back
 * @param {Function|null} onGoToIndexing  "Go to indexing" after the bands button; null leaves it out
 * @param {Function}      renderBands `(caption) => node`: the pattern with the bands of the PC in use
 */
export default function PcFields({
  fileName, resetKey, rows, cols, usedPc, fileConvention, onBadge, onUse, onUndo, onGoToIndexing, renderBands,
  onSave = null,
}) {
  const { t } = useTranslation('pcrefinement');
  const [conv, setConv] = useState('bruker');
  const [typed, setTyped] = useState(['', '', '']);
  const [notice, setNotice] = useState(null);          // 'switched' | 'notConverted' | null: what the select just did
  const [echo, setEcho] = useState(null);              // {key, pc}: the typed numbers as Bruker numbers
  const [inUseConv, setInUseConv] = useState(null);    // {key, pc}: the PC in use in the chosen convention
  const [convError, setConvError] = useState(null);
  const [useError, setUseError] = useState(null);
  const [busyKey, setBusyKey] = useState(null);         // the numbers of a press that has no answer yet
  const [done, setDone] = useState(null);              // the last "Use these numbers" that the store took
  const [undone, setUndone] = useState(null);
  const [bandsShown, setBandsShown] = useState(false);
  const pickedRef = useRef(false);                     // the person chose the kind of numbers: the file's is not forced back
  const typedRef = useRef(typed);
  typedRef.current = typed;
  const convSeq = useRef(0);
  const useSeq = useRef(0);

  const size = patternSizeOf(rows, cols);
  const sizeKey = size ? size.join('x') : '';
  const nums = typed.map(parseNum);
  const allNums = nums.every((n) => Number.isFinite(n));
  const anyTyped = typed.some((s) => s.trim() !== '');
  const badAt = typed.findIndex((s, i) => s.trim() !== '' && !Number.isFinite(nums[i]));
  const typedKey = `${conv}|${typed.map((s) => s.trim()).join('|')}|${sizeKey}`;

  const resetAll = (nextConv) => {
    convSeq.current += 1;
    useSeq.current += 1;
    setConv(nextConv);
    setTyped(['', '', '']);
    setNotice(null); setEcho(null); setInUseConv(null); setConvError(null); setUseError(null);
    setBusyKey(null); setDone(null); setUndone(null); setBandsShown(false);
  };

  // Another file starts the form again; the same file keeps what is typed. The select starts at the kind of
  // numbers of the file's vendor (once the backend has said it), unless the person chose one, or has typed
  // numbers that would change their meaning under them.
  const lastFile = useRef(undefined);
  useEffect(() => {
    if (lastFile.current !== resetKey) {
      lastFile.current = resetKey;
      pickedRef.current = false;
      resetAll(fileConvention || 'bruker');
      return;
    }
    if (!pickedRef.current && typedRef.current.every((s) => s.trim() === '')) setConv(fileConvention || 'bruker');
  }, [resetKey, fileConvention]);

  const fail = (err) => problemText(err, t, 'pcrefinement') || err?.message || String(err);

  // The live echo: the typed numbers as Bruker numbers, asked from the backend a moment after the last key.
  // It is shown only for the numbers it was asked for, so a late answer never stands under other digits.
  useEffect(() => {
    if (conv === 'bruker' || !allNums || !size) return undefined;
    let live = true;
    const timer = setTimeout(() => {
      pcApi.convert(nums, conv, 'bruker', size).then((res) => {
        const out = res?.data?.pc;
        if (!live) return;
        if (Array.isArray(out) && out.length === 3) { setEcho({ key: typedKey, pc: out }); setConvError(null); }
      }).catch((err) => { if (live) setConvError(fail(err)); });
    }, ECHO_DELAY_MS);
    return () => { live = false; clearTimeout(timer); };
  }, [typedKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // The PC in use, in the chosen convention.
  const usedKey = usedPc ? `${usedPc.join(',')}|${conv}|${sizeKey}` : '';
  useEffect(() => {
    if (!usedPc || conv === 'bruker' || !size) return undefined;
    let live = true;
    pcApi.convert(usedPc, 'bruker', conv, size).then((res) => {
      const out = res?.data?.pc;
      if (live && Array.isArray(out) && out.length === 3) { setInUseConv({ key: usedKey, pc: out }); setConvError(null); }
    }).catch((err) => { if (live) setConvError(fail(err)); });
    return () => { live = false; };
  }, [usedKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const usedShown = (() => {
    if (!usedPc) return null;
    if (conv === 'bruker') return usedPc.map(fmt4);
    return inUseConv && inUseConv.key === usedKey ? inUseConv.pc.map(fmt4) : null;
  })();

  const applied = !!done && done.stored && samePc(done.pc, usedPc);
  const badge = anyTyped ? (applied ? 'inUse' : 'unused') : null;
  useEffect(() => { if (onBadge) onBadge(badge); }, [badge]); // eslint-disable-line react-hooks/exhaustive-deps

  const editing = () => { convSeq.current += 1; useSeq.current += 1; setBusyKey(null); setNotice(null); setUseError(null); setDone(null); setUndone(null); setBandsShown(false); };
  const onField = (i) => (e) => {
    const value = e.target.value;
    editing();
    setTyped((prev) => prev.map((s, j) => (j === i ? value : s)));
  };

  // The select converts what the fields hold; it does not replace it.
  const onSelect = (e) => {
    const next = e.target.value;
    if (next === conv) return;
    pickedRef.current = true;
    const seq = ++convSeq.current;
    setUseError(null);
    if (!allNums) {
      setConv(next);
      setNotice(anyTyped ? 'notConverted' : null);
      return;
    }
    if (!size) { setConvError(t('convention.needsSize')); return; }
    pcApi.convert(nums, conv, next, size).then((res) => {
      if (seq !== convSeq.current) return;
      const out = res?.data?.pc;
      if (!Array.isArray(out) || out.length !== 3) return;
      setConv(next);
      setTyped(out.map(fmt4));
      setNotice('switched');
      setConvError(null);
    }).catch((err) => { if (seq === convSeq.current) setConvError(fail(err)); });
  };

  // The same numbers are not sent twice while the first press is on its way; other numbers may be.
  const canUse = allNums && busyKey !== typedKey && (conv === 'bruker' || !!size);
  const typedText = typed.map((s) => s.trim()).join(', ');

  const use = async () => {
    if (!canUse) return;
    const mine = ++useSeq.current;                        // an edit or a newer press makes this answer history
    const current = () => mine === useSeq.current;
    setBusyKey(typedKey); setUseError(null); setNotice(null); setUndone(null); setBandsShown(false);
    try {
      let bruker = nums;
      if (conv !== 'bruker') {
        const res = await pcApi.convert(nums, conv, 'bruker', size);
        bruker = res?.data?.pc;
        if (!Array.isArray(bruker) || bruker.length !== 3) {
          if (current()) setUseError(t('convention.failed', { detail: '' }));
          return;
        }
      }
      // An edit, a changed select or another file during the conversion voids this press: the numbers that
      // were converted are not the ones in the fields any more, and must not go to the open dataset.
      if (!current()) return;
      const answer = await onUse(bruker);
      if (!current()) return;
      if (answer?.error) { setUseError(answer.error); setDone(null); return; }
      setDone({
        file: fileName, pc: bruker, previous: answer?.data?.previous ?? null, conv, typedText,
        stored: answer?.data?.pc_stored !== false,
        perPixel: pcMapActionOf(answer?.data?.pc_map_action) === 'shifted',     // the map stays, its mean is this PC
      });
    } catch (err) {
      if (current()) setUseError(t('convention.failed', { detail: fail(err) }));
    } finally {
      if (current()) setBusyKey(null);
    }
  };

  // PcGrant hands over the name it shows; the backend is asked for the dataset itself (null: the open one).
  const undo = async () => {
    const answer = await onUndo(fileName);
    if (answer?.data) {
      const d = answer.data;
      setUndone({ file: fileLabel, previous: { pc: d.pc ?? null, had_pc: !!d.had_pc }, notRestored: Array.isArray(d.not_restored) ? d.not_restored : [] });
      setDone(null);
      setBandsShown(false);
      return undefined;
    }
    return answer;
  };

  const fileLabel = fileName || t('fileHead.noFile');
  const label = (c, axis) => t(`convention.fields.${c}.${axis}`);
  const hint = (c, axis) => t(`convention.hints.${c}.${axis}`);
  const shortName = (c) => t(`convention.short.${c}`);
  const badLabel = badAt >= 0 ? label(conv, AXES[badAt]).replace(/:$/, '') : '';

  let echoNode = null;
  if (allNums) {
    if (conv === 'bruker') {
      echoNode = <span>{t('convention.echoBruker', { typed: typedText })}</span>;
    } else if (echo && echo.key === typedKey && size) {
      echoNode = (
        <Trans
          t={t} i18nKey="convention.echoOther" components={{ b: <b /> }}
          values={{ name: shortName(conv), typed: typedText, rows: size[0], cols: size[1], bruker: formatPc(echo.pc) }}
        />
      );
    }
  }

  return (
    <div data-testid="pc-fields">
      <div style={noteStyle}>{t('convention.intro')}</div>

      <div style={rowStyle}>
        <label htmlFor="pc-convention" style={labelStyle}>{t('convention.label')}</label>
        <div style={cellStyle}>
          <select
            id="pc-convention" value={conv} onChange={onSelect}
            style={{ ...inputStyle, width: 'auto', padding: '4px 8px', cursor: 'pointer' }}
          >
            {CONVENTIONS.map((c) => <option key={c} value={c}>{t(`convention.names.${c}`)}</option>)}
          </select>
          {fileConvention && conv === fileConvention && !pickedRef.current && (
            <span data-testid="pc-conv-preselected" style={{ ...fullWidth, flex: '1 1 auto' }}>
              {t(`convention.preselected.${fileConvention}`)}
            </span>
          )}
        </div>
      </div>

      <div style={rowStyle}>
        <span style={labelStyle}>{t('convention.sizeLabel')}</span>
        <div style={cellStyle}>
          <b data-testid="pc-pattern-size" style={{ fontSize: '10pt', color: colors.text }}>
            {size ? t('convention.size', { rows: size[0], cols: size[1] }) : t('convention.sizeUnknown')}
          </b>
          {size && <span style={fullWidth}>{t('convention.sizeNote', { file: fileLabel })}</span>}
        </div>
      </div>

      {usedPc && <div data-testid="pc-same-note" style={noteStyle}>{t('convention.samePc')}</div>}

      {AXES.map((axis, i) => (
        <div key={axis} style={rowStyle}>
          <label htmlFor={`pc-field-${axis}`} style={labelStyle}>{label(conv, axis)}</label>
          <div style={cellStyle}>
            <input
              id={`pc-field-${axis}`} type="text" inputMode="decimal" autoComplete="off" spellCheck={false}
              value={typed[i]} onChange={onField(i)}
              onKeyDown={(e) => { if (e.key === 'Enter') use(); }}
              style={inputStyle}
            />
            <span data-testid={`pc-in-use-${axis}`} style={{ fontSize: '9pt', color: colors.text, whiteSpace: 'nowrap' }}>
              {usedPc === undefined ? '' : usedPc
                ? (usedShown ? t('convention.inUse', { convention: shortName(conv), value: usedShown[i] }) : '')
                : t('convention.noPcInUse')}
            </span>
            <span data-testid={`pc-hint-${axis}`} style={fullWidth}>{hint(conv, axis)}</span>
          </div>
        </div>
      ))}

      {notice === 'switched' && (
        <div data-testid="pc-switched" role="status" style={{ ...noteStyle, color: colors.green }}>{t('convention.switched')}</div>
      )}
      {notice === 'notConverted' && (
        <div data-testid="pc-not-converted" role="status" style={{ ...noteStyle, color: colors.green }}>{t('convention.notConverted')}</div>
      )}
      {!allNums && badAt < 0 && <div data-testid="pc-fill-hint" style={noteStyle}>{t('convention.fillAll')}</div>}
      {conv !== 'bruker' && !size && (
        <div data-testid="pc-needs-size" role="alert" style={{ ...noteStyle, color: colors.red }}>{t('convention.needsSize')}</div>
      )}
      {badAt >= 0 && (
        <div data-testid="pc-field-error" role="alert" style={{ ...noteStyle, color: colors.red }}>
          {t('convention.notANumber', { value: typed[badAt].trim(), label: badLabel })}
        </div>
      )}
      {convError && (
        <div data-testid="pc-conv-error" role="alert" style={{ ...noteStyle, color: colors.red }}>
          {t('convention.failed', { detail: convError })}
        </div>
      )}
      {useError && <div data-testid="pc-use-error" role="alert" style={{ ...noteStyle, color: colors.red }}>{useError}</div>}
      {echoNode && (
        <div
          data-testid="pc-echo"
          style={{
            margin: '6px 0', padding: '6px 8px', fontSize: '9pt', color: colors.text, wordBreak: 'break-word',
            border: `1px solid ${colors.accent}`, borderRadius: 4, background: alpha(colors.accent, 7),
          }}
        >
          {echoNode}
        </div>
      )}

      <Button variant="primary" disabled={!canUse} onClick={use} data-testid="pc-use-button" style={{ marginTop: 4 }}>
        {t('convention.use')}
      </Button>

      {applied && (
        <div data-testid="pc-typed-grant" style={{ marginTop: 6 }}>
          <div style={noteStyle}>
            {done.conv === 'bruker'
              ? t('convention.noteBruker', { typed: done.typedText })
              : t('convention.noteOther', { name: shortName(done.conv), typed: done.typedText, bruker: formatPc(done.pc) })}
          </div>
          <PcGrant
            files={[fileLabel]}
            pc={done.pc}
            perPixel={done.perPixel}
            previous={[done.previous]}
            bands={{ shown: bandsShown, onToggle: () => setBandsShown((v) => !v) }}
            onGoToIndexing={onGoToIndexing}
            actions={onSave ? [{ key: 'save', label: t('grant.saveAsCalibration'), onClick: () => onSave(done.pc) }] : []}
            onUndo={undo}
          />
          {bandsShown && renderBands && renderBands(t('bands.caption'))}
        </div>
      )}
      {undone && (
        <div style={{ marginTop: 6 }}>
          <PcUndoDone file={undone.file} previous={undone.previous} notRestored={undone.notRestored} />
        </div>
      )}
    </div>
  );
}
