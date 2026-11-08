// "Take the pattern centre from...": a saved calibration, or another file. Nothing changes until a button of a row
// (or "Use this PC") is pressed. A calibration that fits has the main button; one that does not fit, or whose
// fit cannot be told, has the second naming "Use anyway" and the request names the file again (`accept_mismatch`).
// A PC read from another file is shown in both conventions with what was assumed, and used only on "Use this PC".
import React, { useEffect, useRef, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { pcCalibrationsApi } from '../../../services/api';
import { colors, Button } from '../../../theme/components';
import { formatPc } from '../pcFormat';
import AppliedGrant from './AppliedGrant';
import { CalibrationFacts, noteStyle, factStyle, alertStyle, warnStyle } from './CalibrationFacts';
import {
  fitState, differenceLines, settingNameKey, noteWords, sourceKindOf, baseName, copyKeptOf,
} from './calibrationText';
import { refusalWords } from './calibrationWords';

const boxStyle = {
  border: `1px solid ${colors.border}`, background: colors.bg, borderRadius: 4, padding: '8px 10px', margin: '8px 0',
  fontSize: '9pt', color: colors.text,
};
const rowStyle = { display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'space-between', padding: '8px 0' };

function CalibrationRow({ item, first, busy, onTake }) {
  const { t } = useTranslation('pcrefinement');
  const state = fitState(item);
  const diffs = state === 'no' ? differenceLines(item) : [];
  return (
    <div
      data-testid="take-calibration-row" data-id={item.id} data-state={state}
      style={{ ...rowStyle, borderTop: first ? 'none' : `1px solid ${colors.border}` }}
    >
      <div style={{ flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
        <CalibrationFacts item={item} />
        {diffs.map((d) => (
          <div key={d.key} role="alert" style={alertStyle} data-testid="take-difference">
            {d.aspect ? t('calibration.diff.aspect') : t('calibration.diff.take', {
              setting: t(settingNameKey(d.key)),
              file: d.file ?? t('calibration.settings.notStated'),
              calibration: d.calibration ?? t('calibration.settings.notStated'),
            })}
          </div>
        ))}
        {state === 'unstated' && <div style={warnStyle}>{t('calibration.take.unstated')}</div>}
      </div>
      {state === 'fits' ? (
        <Button small variant="primary" disabled={busy} onClick={() => onTake(item, false)}>{t('calibration.offer.use')}</Button>
      ) : (
        <Button small variant="danger" disabled={busy} onClick={() => onTake(item, true)}>{t('calibration.offer.useAnyway')}</Button>
      )}
    </div>
  );
}

/** The PC another file holds, in the file's own numbers and in Orienta's, with what was assumed on the way. */
function Proposal({ proposal, dataset, busy, onUse }) {
  const { t } = useTranslation('pcrefinement');
  const d = proposal.data;
  const conv = ['tsl', 'oxford', 'bruker'].includes(d.convention) ? d.convention : 'bruker';
  const kind = sourceKindOf(d.source_kind);
  const items = Array.isArray(d.note_items)
    ? d.note_items
    : (Array.isArray(d.notes) ? d.notes.map((text) => ({ code: 'other', params: {}, text })) : []);
  const context = { kind: t(`takePc.kinds.${kind}`), file: dataset };
  const mismatch = d.shape_matches === false && Array.isArray(d.shape) && Array.isArray(d.target_shape);
  return (
    <div data-testid="take-proposal">
      <div style={factStyle} data-testid="take-stated">
        <Trans
          t={t} i18nKey="takePc.stated" components={{ b: <b /> }}
          values={{
            kind: t(`takePc.kinds.${kind}`), convention: t(`convention.names.${conv}`),
            labels: t(`takePc.axes.${conv}`), values: formatPc(d.stated),
          }}
        />
      </div>
      {conv !== 'bruker' && (
        <div style={factStyle} data-testid="take-in-orienta">
          <Trans t={t} i18nKey="takePc.inOrienta" components={{ b: <b /> }} values={{ values: formatPc(d.pc) }} />
        </div>
      )}
      {mismatch && (
        <div style={warnStyle} data-testid="take-shape-differs">
          {t('takePc.shapeDiffers', { rows: d.shape[0], cols: d.shape[1], targetRows: d.target_shape[0], targetCols: d.target_shape[1] })}
        </div>
      )}
      {items.map((item, i) => {
        const words = noteWords(item, context);
        return (
          <div key={`${item.code}-${i}`} style={noteStyle} data-testid="take-note" data-code={item.code}>
            {words ? t(`takePc.notes.${words.key}`, words.params) : item.text}
          </div>
        );
      })}
      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
        <Button small variant="primary" disabled={busy} onClick={onUse}>{t('takePc.usePc')}</Button>
      </div>
    </div>
  );
}

/**
 * @param {string}   dataset   the open file
 * @param {Function} onClose
 * @param {Function} onPcChanged  `(answer)`: the PC of the file changed; the page loads it again from the backend
 * @param {Function} onUsePc   `(pc) => Promise<{data} | {error}>`: the page's way of giving the file a PC
 * @param {Function} onUndo    `(file) => Promise<{data} | {gone} | {error}>`: the page's put back
 * @param {Function} pickFile  `(current, done)`: the app's file picker; `done(path)` with the chosen path
 */
export function TakePcDialog({
  dataset, onClose, onPcChanged = null, onUsePc, onUndo, onGoToIndexing = null, renderBands = null,
  pickFile = null, grantActions = [],
}) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  const [list, setList] = useState({ state: 'loading', items: [], error: '' });
  const [calGrant, setCalGrant] = useState(null);       // {answer}
  const [calError, setCalError] = useState('');
  const [busy, setBusy] = useState(false);
  const [path, setPath] = useState('');
  const [pathError, setPathError] = useState('');
  const [reading, setReading] = useState(false);
  const [proposal, setProposal] = useState(null);       // {path, data}
  const [fileGrant, setFileGrant] = useState(null);     // {pc, previous, from, mapAction, stored}
  const [useError, setUseError] = useState('');
  const datasetRef = useRef(dataset);
  datasetRef.current = dataset;
  const readSeq = useRef(0);

  // The window belongs to the file it was opened for: another file closes it.
  const openedFor = useRef(dataset);
  useEffect(() => { if (openedFor.current !== dataset) onClose(); }, [dataset]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  useEffect(() => {
    let live = true;
    setList({ state: 'loading', items: [], error: '' });
    pcCalibrationsApi.suggest(dataset, true).then((res) => {
      // each offer names the FILE it is applied to: for an open copy, the file the copy was made from
      if (live) {
        const found = Array.isArray(res.data?.suggestions) ? res.data.suggestions : [];
        setList({ state: 'ready', items: found.map((s) => ({ ...s, target: res.data?.target || dataset })), error: '' });
      }
    }).catch((err) => {
      if (live) setList({ state: 'error', items: [], error: refusalWords(err, t) });
    });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  const take = async (item, anyway) => {
    const forDataset = dataset;
    const target = item.target || forDataset;          // apply refuses a copy: the calibration goes to its file
    setBusy(true);
    setCalError('');
    try {
      const res = await pcCalibrationsApi.apply(item.id, [target], anyway ? [target] : [], item.version);
      if (datasetRef.current !== forDataset) return;
      const answer = (res.data?.results || [])[0];
      if (answer?.status === 'applied') {
        setCalGrant({ answer, opened: forDataset });
        if (onPcChanged) onPcChanged(answer);
      } else {
        setCalError(t(`calibration.apply.${answer?.status || 'failed'}`, { defaultValue: t('calibration.apply.failed') }));
        if (onPcChanged) onPcChanged();             // a refusal: the page asks the backend what the PC is now
      }
    } catch (err) {
      if (datasetRef.current === forDataset) setCalError(refusalWords(err, t));
      if (onPcChanged) onPcChanged();
    } finally {
      if (datasetRef.current === forDataset) setBusy(false);
    }
  };

  const browse = () => {
    if (!pickFile) return;
    pickFile(path, (picked) => {
      if (typeof picked === 'string' && picked.trim()) { setPath(picked.trim()); setPathError(''); setProposal(null); }
    });
  };

  const read = async () => {
    const typed = path.trim();
    setUseError('');
    setFileGrant(null);
    if (!typed) { setPathError(t('takePc.pathEmpty')); setProposal(null); return; }
    setPathError('');
    const forDataset = dataset;
    const mine = ++readSeq.current;
    setReading(true);
    try {
      const res = await pcCalibrationsApi.fromFile(typed, forDataset);
      if (mine !== readSeq.current || datasetRef.current !== forDataset) return;
      setProposal({ path: typed, dataset: forDataset, data: res.data || {} });
    } catch (err) {
      if (mine !== readSeq.current || datasetRef.current !== forDataset) return;
      setProposal(null);
      setPathError(refusalWords(err, t));
    } finally {
      if (mine === readSeq.current) setReading(false);
    }
  };

  const usePc = async () => {
    const pc = proposal?.data?.pc;
    if (!proposal || proposal.dataset !== dataset || !Array.isArray(pc) || pc.length !== 3) return;
    const forDataset = dataset;
    const from = baseName(proposal.path);
    setBusy(true);
    setUseError('');
    try {
      const answer = await onUsePc(pc);
      if (datasetRef.current !== forDataset) return;
      if (answer?.error) { setUseError(answer.error); setFileGrant(null); return; }
      const d = answer?.data || {};
      setFileGrant({ pc, previous: d.previous ?? null, from, mapAction: d.pc_map_action, stored: d.pc_stored !== false });
      if (onPcChanged) onPcChanged(d);
    } finally {
      if (datasetRef.current === forDataset) setBusy(false);
    }
  };

  const items = list.items;
  return (
    <div
      role="dialog" aria-modal="true" aria-label={t('takePc.title')} data-testid="take-pc-dialog"
      style={{
        position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        className="thin-scrollbar"
        style={{
          background: colors.bgSecondary || colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8, padding: 16,
          width: 'min(640px, 94vw)', maxHeight: '90vh', overflowY: 'auto', color: colors.text,
        }}
      >
        <h3 style={{ margin: '0 0 6px', fontSize: '12pt' }}>{t('takePc.title')}</h3>
        <div style={noteStyle}>{t('takePc.nothingChanges')}</div>

        <div style={boxStyle} data-testid="take-calibrations">
          <b>{t('takePc.calibrations')}</b>
          {list.state === 'loading' && <div style={noteStyle} role="status">{t('takePc.loading')}</div>}
          {list.state === 'error' && (
            <div role="alert" style={alertStyle} data-testid="take-list-error">{t('takePc.listFailed', { detail: list.error })}</div>
          )}
          {list.state === 'ready' && items.length === 0 && (
            <div style={noteStyle} data-testid="take-list-empty">{t('takePc.none')}</div>
          )}
          {items.map((item, i) => (
            <CalibrationRow key={`${item.id}-${item.version}`} item={item} first={i === 0} busy={busy} onTake={take} />
          ))}
          {calError && <div role="alert" style={alertStyle} data-testid="take-calibration-error">{calError}</div>}
          {calGrant && (
            <div data-testid="take-calibration-grant">
              <AppliedGrant
                file={calGrant.answer.dataset || dataset} pc={calGrant.answer.pc} previous={calGrant.answer.previous ?? null}
                mapAction={calGrant.answer.pc_map_action} copyKept={copyKeptOf(calGrant.answer, calGrant.opened)} onUndo={onUndo} onGoToIndexing={onGoToIndexing}
                renderBands={renderBands} actions={grantActions}
              />
            </div>
          )}
        </div>

        <div style={boxStyle} data-testid="take-file">
          <b>{t('takePc.anotherFile')}</b>
          <div style={noteStyle}>{t('takePc.fileNote')}</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
            <input
              type="text" value={path} aria-label={t('takePc.pathLabel')} placeholder={t('takePc.pathPlaceholder')}
              onChange={(e) => { setPath(e.target.value); setProposal(null); setFileGrant(null); }}
              onKeyDown={(e) => { if (e.key === 'Enter') read(); }}
              style={{
                flex: 1, minWidth: 0, background: colors.bg, color: colors.text, border: `1px solid ${colors.border}`,
                borderRadius: 3, padding: '3px 6px', fontSize: '9pt',
              }}
            />
            {pickFile && <Button small onClick={browse}>{t('takePc.browse')}</Button>}
            <Button small disabled={reading} onClick={read}>{t('takePc.read')}</Button>
          </div>
          {pathError && <div role="alert" style={alertStyle} data-testid="take-path-error">{pathError}</div>}
          {proposal && proposal.dataset === dataset && (
            <Proposal proposal={proposal} dataset={dataset} busy={busy} onUse={usePc} />
          )}
          {useError && <div role="alert" style={alertStyle} data-testid="take-use-error">{useError}</div>}
          {fileGrant && !fileGrant.stored && (
            <div role="status" style={warnStyle} data-testid="take-not-stored">{t('warnings.pcNotStored')}</div>
          )}
          {fileGrant && fileGrant.stored && (
            <div data-testid="take-file-grant">
              <AppliedGrant
                file={dataset} pc={fileGrant.pc} previous={fileGrant.previous} mapAction={fileGrant.mapAction}
                onUndo={onUndo} onGoToIndexing={onGoToIndexing} renderBands={renderBands} actions={grantActions}
              >
                <div style={noteStyle} data-testid="take-taken-from">{t('takePc.takenFrom', { file: fileGrant.from })}</div>
              </AppliedGrant>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
          <Button onClick={onClose}>{t('common:close')}</Button>
        </div>
      </div>
    </div>
  );
}

export default TakePcDialog;
