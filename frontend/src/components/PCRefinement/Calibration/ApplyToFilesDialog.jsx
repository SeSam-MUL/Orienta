// "Apply <calibration, version, PC> to other files". Nothing is applied until Apply is pressed; a file whose settings
// differ or cannot be told is never ticked for the user, and a file that differs in a setting needs "Allow this file"
// before its tick box can be used. The window belongs to the calibration, not to the file in front of the user:
// it is kept in `applyFlow` and shown again by the page that mounts after "Find PC for it" opened another file.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pcCalibrationsApi } from '../../../services/api';
import useDataStore from '../../../stores/useDataStore';
import { colors, Button } from '../../../theme/components';
import PcGrant, { PcUndoDone } from '../PcGrant';
import { pcMapActionOf, pcMapNote } from '../pcMapNote';
import { formatPc } from '../pcFormat';
import { needsDecision, warningOf } from '../pcResultFacts';
import ModalShell from './ModalShell';
import { noteStyle, factStyle, alertStyle, warnStyle } from './CalibrationFacts';
import { applyRows, versionMeta, settingNameKey } from './calibrationText';
import { refusalWords } from './calibrationWords';
import useApplyFlow, { isFarFile } from './applyFlow';
import switchLoadedFile from '../../common/switchLoadedFile';

const boxStyle = {
  border: `1px solid ${colors.border}`, background: colors.bg, borderRadius: 4, padding: '8px 10px', margin: '8px 0',
  fontSize: '9pt', color: colors.text,
};
const cell = { padding: '3px 6px', borderBottom: `1px solid ${colors.border}`, textAlign: 'left', overflowWrap: 'anywhere' };
const pathStyle = { display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'space-between', margin: '6px 0' };

/** Why a file was not applied, in words: the part in brackets after "not applied". */
function whyText(row, t) {
  if (row.why === 'sameVersion') return t('applyDialog.why.sameVersion');
  if (row.why === 'differs') {
    const first = row.differences[0];
    // the row itself says "Other pattern shape" where the height-to-width ratio differs: the same words here
    if (first?.aspect) return t('applyDialog.why.differsShape');
    return t('applyDialog.why.differs', { setting: first ? t(settingNameKey(first.key)) : '' });
  }
  if (row.why === 'unstated') return t('applyDialog.why.unstated');
  return '';
}

/** A file that is already on this version ("no change") has nothing to apply: its box cannot be ticked. */
const canTick = (row) => row.why !== 'sameVersion';

/**
 * @param {Function} onClose
 * @param {Function} onPcChanged called after anything was applied: the page loads its PC from the backend again
 * @param {Function} onUndo      `(file) => Promise<{data} | {gone} | {error}>`, the page's put back
 * @param {Function} onSave      `({dataset, pc, quality, fromFar})`: the page's Save as calibration window
 * @param {Function} onFindPc    the page's Find PC (starts patterns and search)
 * @param {boolean}  canFindPc   the page can start Find PC now
 * @param {boolean}  pageReady   the page knows the PC of its file
 * @param {object}   findStatus  `{running, message, error}`: what the page says about its Find PC run
 * @param {object}   findResult  the result of the page's last search, or null
 * @param {object}   savedFar    `{dataset, name}` once the far file's result was saved as a calibration
 */
export default function ApplyToFilesDialog({
  onClose, onPcChanged = null, onUndo, onSave, onFindPc, canFindPc = false, pageReady = false,
  findStatus = null, findResult = null, savedFar = null,
}) {
  const { t } = useTranslation(['pcrefinement', 'common', 'ebsdviewer']);
  const flow = useApplyFlow((s) => s.flow);
  const calibration = flow?.calibration;
  const far = flow?.far || null;
  // The far file is told by its path: switching back to a file restores the dataset that was active there last,
  // which may be a derived copy and not the one the row names.
  const openPath = useDataStore((s) => s.filePath);
  const farOpen = isFarFile(far, openPath);
  const [list, setList] = useState({ state: 'loading', rows: { fit: [], others: [] }, error: '' });
  const [ticked, setTicked] = useState(null);                 // Set of dataset names, set once the list is read
  const [allowed, setAllowed] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState(null);               // {lines: [...], applied: [...]}
  const [error, setError] = useState('');
  const [farUndone, setFarUndone] = useState(null);
  const onFindRef = useRef(onFindPc);
  onFindRef.current = onFindPc;

  const calId = calibration?.id;
  const resultKey = findResult ? `${findResult.dataset}|${(findResult.mean_pc || []).join(',')}` : '';

  // The list of files: read when the window opens and again when the far file's search has ended (its PC changed).
  useEffect(() => {
    if (!calId) return undefined;
    let live = true;
    pcCalibrationsApi.candidates(calId).then((res) => {
      if (!live) return;
      const rows = applyRows(res.data?.candidates, flow.origin);
      setList({ state: 'ready', rows, error: '' });
      setTicked((prev) => {
        const known = new Set([...rows.fit, ...rows.others].filter(canTick).map((r) => r.dataset));
        if (prev === null) return new Set(rows.fit.filter((r) => r.ticked).map((r) => r.dataset));
        return new Set([...prev].filter((n) => known.has(n)));
      });
    }).catch((err) => {
      if (live) setList({ state: 'error', rows: { fit: [], others: [] }, error: refusalWords(err, t) });
    });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [calId, resultKey]);

  // Find PC for the far file starts once, in the page that is mounted for that file (whichever of its datasets is open).
  useEffect(() => {
    if (!far || far.step !== 'starting' || !pageReady || !canFindPc || !farOpen) return;
    if (useApplyFlow.getState().claimStart(openPath)) onFindRef.current();
  }, [far, pageReady, canFindPc, farOpen, openPath]);

  const { fit, others } = list.rows;
  const all = useMemo(() => [...fit, ...others], [fit, others]);
  const tickedList = all.filter((r) => canTick(r) && ticked && ticked.has(r.dataset));
  const count = tickedList.length;
  const done = !!outcome;
  const meta = calibration ? versionMeta(calibration) : null;

  if (!calibration) return null;

  const toggle = (name, on) => setTicked((prev) => {
    const next = new Set(prev || []);
    if (on) next.add(name); else next.delete(name);
    return next;
  });
  const fitBoxes = fit.filter((r) => r.kind === 'fits' && canTick(r));
  const selectAll = fitBoxes.length > 0 && fitBoxes.every((r) => ticked && ticked.has(r.dataset));
  const toggleAll = (on) => setTicked((prev) => {
    const next = new Set(prev || []);
    fitBoxes.forEach((r) => { if (on) next.add(r.dataset); else next.delete(r.dataset); });
    return next;
  });

  const apply = async () => {
    const names = tickedList.map((r) => r.dataset);
    const accept = tickedList.filter((r) => r.kind !== 'fits').map((r) => r.dataset);
    setBusy(true);
    setError('');
    try {
      const res = await pcCalibrationsApi.apply(calibration.id, names, accept, calibration.version);
      const byName = new Map((res.data?.results || []).map((a) => [a.dataset, a]));
      const lines = [];
      const applied = [];
      all.forEach((row) => {
        const answer = byName.get(row.dataset);
        if (answer?.status === 'applied') {
          applied.push({ file: row.dataset, previous: answer.previous ?? null, mapAction: pcMapActionOf(answer.pc_map_action) });
          lines.push({ key: row.dataset, ok: true, file: row.dataset });
        } else if (answer) {
          lines.push({ key: row.dataset, ok: false, file: row.dataset, why: t(`applyDialog.status.${answer.status}`, { defaultValue: t('applyDialog.status.failed') }) });
        } else {
          lines.push({ key: row.dataset, ok: false, file: row.dataset, why: whyText(row, t) });
        }
      });
      setOutcome({ lines, applied });
      if (onPcChanged) onPcChanged();
    } catch (err) {
      setError(refusalWords(err, t));
      if (onPcChanged) onPcChanged();         // a refusal: the page asks the backend what the PC is now
    } finally {
      setBusy(false);
    }
  };

  const openFar = async (row) => {
    useApplyFlow.getState().setFar({ dataset: row.dataset, path: row.path, step: 'starting' });
    // the step this press made: gone when the window was closed meanwhile or another file was chosen
    const mine = () => {
      const f = useApplyFlow.getState().flow?.far;
      return !!f && f.dataset === row.dataset && f.path === row.path;
    };
    try {
      const answer = await switchLoadedFile(row.path, (k, o) => t(`ebsdviewer:${k}`, o));
      if (!mine()) return;
      if (answer.filePath) useApplyFlow.getState().patchFar({ openedPath: answer.filePath });
      // another load took the file's place: the step does not stay 'starting', or opening the file by hand later
      // would start Find PC with no press (a step that already searched stays)
      if (!answer.opened && useApplyFlow.getState().flow.far.step === 'starting') {
        useApplyFlow.getState().patchFar({ step: 'error', error: t('applyDialog.openReplaced') });
      }
    } catch (err) {
      if (mine()) useApplyFlow.getState().patchFar({ step: 'error', error: refusalWords(err, t) });
    }
  };

  const heading = t('applyDialog.title', { name: calibration.name, version: calibration.version, pc: formatPc(calibration.pc) });

  // ---- the row of a file that differs: two ways, and what became of the first
  const farRow = (row) => {
    const isFar = far && far.dataset === row.dataset;
    const failed = isFar && far.step === 'error';
    // the result of the page of the far file, whichever of that file's datasets the search ran on
    const result = isFar && !failed && farOpen && findResult && findResult.mode === 'find' ? findResult : null;
    const resultPc = result ? (result.mean_pc || result.pc) : null;
    // where the search wrote the PC: the dataset it ran on (the row's name, or a copy of it)
    const target = result?.dataset || row.dataset;
    return (
      <>
        {(!isFar || failed) && row.path && (
          <div style={pathStyle} data-testid="far-recommended">
            <div style={factStyle}><b>{t('applyDialog.recommended')}</b> {t('applyDialog.recommendedText')}</div>
            <Button small variant="primary" disabled={busy || done} onClick={() => openFar(row)}>{t('applyDialog.findForIt')}</Button>
          </div>
        )}
        {failed && <div role="alert" style={alertStyle} data-testid="far-open-failed">{t('applyDialog.openFailed', { detail: far.error })}</div>}
        {isFar && !failed && !result && (
          <div data-testid="far-progress" style={{ margin: '6px 0' }}>
            <div role="status" style={{ ...factStyle, color: colors.green }}>{t('applyDialog.openNow', { file: row.dataset })}</div>
            {far.step === 'starting' && farOpen && pageReady && !canFindPc && (
              <div style={warnStyle}>{t('applyDialog.needsPhase')}</div>
            )}
            {far.step === 'searching' && findStatus?.running && <div style={noteStyle}>{t('applyDialog.searching')}</div>}
            {far.step === 'searching' && !findStatus?.running && findStatus?.message && (
              <div role={findStatus.error ? 'alert' : 'status'} style={findStatus.error ? alertStyle : noteStyle}>{findStatus.message}</div>
            )}
          </div>
        )}
        {isFar && result && resultPc && (
          <div data-testid="far-result" style={{ margin: '6px 0' }}>
            <div style={factStyle}>
              <span>{t('applyDialog.pcBruker')}</span> <b data-testid="far-result-pc">{formatPc(resultPc)}</b>
            </div>
            {farUndone ? (
              <PcUndoDone file={target} previous={farUndone.previous} notRestored={farUndone.notRestored} />
            ) : (
              <>
                {needsDecision(result) || warningOf(result)?.kind === 'secondPosition' ? (
                  <div style={warnStyle} data-testid="far-unsure">{t('applyDialog.unsure', { file: row.dataset })}</div>
                ) : (
                  <div style={{ ...factStyle, color: colors.green }} data-testid="far-usable">{t('applyDialog.usable', { file: row.dataset })}</div>
                )}
                <PcGrant
                  files={[target]} pc={resultPc} previous={[result.previous || null]}
                  actions={savedFar && savedFar.dataset === target ? [] : [{
                    key: 'save', label: t('grant.saveAsCalibration'),
                    onClick: () => onSave({ dataset: target, pc: resultPc, quality: result.quality || null, fromFar: true }),
                  }]}
                  onUndo={async (file) => {
                    const a = await onUndo(file);
                    if (a?.data) {
                      setFarUndone({ previous: { pc: a.data.pc ?? null, had_pc: !!a.data.had_pc },
                        notRestored: Array.isArray(a.data.not_restored) ? a.data.not_restored : [] });
                      return undefined;
                    }
                    return a;
                  }}
                />
                {savedFar && savedFar.dataset === target && (
                  <div role="status" style={{ ...factStyle, color: colors.green }} data-testid="far-saved">
                    {t('applyDialog.savedFar', { name: savedFar.name, wd: row.wd || t('calibration.settings.notStated') })}
                  </div>
                )}
              </>
            )}
          </div>
        )}
        {!allowed.has(row.dataset) ? (
          <div style={pathStyle}>
            <div style={factStyle}><b>{t('applyDialog.usuallyWrong')}</b> {t('applyDialog.usuallyWrongText')}</div>
            <Button small disabled={busy || done} onClick={() => setAllowed((s) => new Set(s).add(row.dataset))}>{t('applyDialog.allow')}</Button>
          </div>
        ) : (
          <div style={warnStyle} data-testid="far-allowed">
            {/* "not ticked yet" only while it is not: once ticked, or once the window is done, it would be untrue */}
            {(ticked && ticked.has(row.dataset)) || done ? t('applyDialog.allowedShort') : t('applyDialog.allowed')}
          </div>
        )}
      </>
    );
  };

  const nothing = list.state === 'ready' && all.length === 0;
  // the sentence about the ticks follows the ticks: files that fit and would change are ticked at the start
  const topLine = count === 0 ? t('applyDialog.noneTicked') : (fit.some((r) => r.ticked) ? t('applyDialog.someTicked') : '');

  return (
    <ModalShell title={heading} testId="apply-dialog" onClose={onClose}>
      {list.state === 'loading' && <div style={noteStyle} role="status">{t('applyDialog.loading')}</div>}
      {list.state === 'error' && <div role="alert" style={alertStyle} data-testid="apply-list-error">{list.error}</div>}
      {nothing && <div style={noteStyle} data-testid="apply-none">{t('applyDialog.noOtherFile')}</div>}
      {list.state === 'ready' && !nothing && (
        <>
          {(done || topLine) && <div style={factStyle} data-testid="apply-top">{done ? t('applyDialog.done') : topLine}</div>}
          <div style={noteStyle} data-testid="apply-version">{t(`calibration.inline.${meta.key}`, { name: calibration.name, ...meta.params })}</div>
          {!done && <div style={noteStyle} data-testid="apply-nothing-until">{t('applyDialog.nothingUntil')}</div>}

          {fit.length > 0 && (
            <>
              <label style={{ ...factStyle, display: 'flex', gap: 6, alignItems: 'center', margin: '6px 0' }}>
                <input type="checkbox" checked={selectAll} disabled={done || fitBoxes.length === 0} onChange={(e) => toggleAll(e.target.checked)} />
                {t('applyDialog.selectAll')}
              </label>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '9pt', color: colors.text }} data-testid="apply-table">
                <thead>
                  <tr style={{ color: colors.textSecondary }}>
                    <th style={cell} />
                    <th style={cell}>{t('applyDialog.colFile')}</th>
                    <th style={cell}>{t('applyDialog.colWd')}</th>
                    <th style={cell}>{t('applyDialog.colOld')}</th>
                    <th style={cell}>{t('applyDialog.colNew')}</th>
                  </tr>
                </thead>
                <tbody>
                  {fit.map((row) => (
                    <tr key={row.dataset} data-file={row.dataset}>
                      <td style={cell}>
                        <input
                          type="checkbox" aria-label={row.dataset} checked={canTick(row) && !!ticked && ticked.has(row.dataset)}
                          disabled={done || !canTick(row)}
                          onChange={(e) => toggle(row.dataset, e.target.checked)}
                        />
                      </td>
                      <td style={cell}>{row.dataset}</td>
                      <td style={cell} data-col="wd">
                        {row.wdCheck === 'notStated' ? t('calibration.settings.notStated') : `${row.wd}, ${t(`calibration.settings.${row.wdCheck}`)}`}
                      </td>
                      <td style={cell} data-col="old">
                        {row.pc ? formatPc(row.pc) : t('applyDialog.noPc')}
                        {row.ref && <div style={noteStyle}>{t('applyDialog.version', { version: row.ref.version })}</div>}
                      </td>
                      <td style={cell} data-col="new">
                        {row.why === 'sameVersion' ? t('applyDialog.noChange') : formatPc(calibration.pc)}
                        {row.why === 'sameVersion' && <div style={noteStyle}>{t('applyDialog.version', { version: calibration.version })}</div>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {others.map((row) => (
            <div key={row.dataset} style={boxStyle} data-testid="apply-other" data-file={row.dataset} data-kind={row.kind}>
              <label style={{ ...factStyle, display: 'flex', gap: 6, alignItems: 'center' }}>
                <input
                  type="checkbox" aria-label={row.dataset} checked={!!ticked && ticked.has(row.dataset)}
                  disabled={done || row.kind === 'same' || (row.kind === 'differs' && !allowed.has(row.dataset))}
                  onChange={(e) => toggle(row.dataset, e.target.checked)}
                />
                {row.dataset}
              </label>
              {row.kind === 'differs' ? (
                <>
                  {row.differences.map((d) => (
                    <div key={d.key} role="alert" style={alertStyle} data-testid="apply-difference">
                      {d.aspect ? t('calibration.diff.aspect') : t('calibration.diff.offer', {
                        setting: t(settingNameKey(d.key)),
                        file: d.file ?? t('calibration.settings.notStated'),
                        calibration: d.calibration ?? t('calibration.settings.notStated'),
                      })}
                    </div>
                  ))}
                  {farRow(row)}
                </>
              ) : (
                <div style={noteStyle}>{t(row.kind === 'same' ? 'applyDialog.noChange' : 'applyDialog.unstatedNote')}</div>
              )}
            </div>
          ))}
        </>
      )}

      {error && <div role="alert" style={alertStyle} data-testid="apply-error">{error}</div>}
      {outcome && (
        <div data-testid="apply-result" style={{ marginTop: 8 }}>
          {outcome.lines.map((l) => (
            l.ok
              ? <div key={l.key} role="status" style={{ ...factStyle, color: colors.green }} data-testid="apply-line" data-ok="1">{t('applyDialog.applied', { file: l.file })}</div>
              : <div key={l.key} style={noteStyle} data-testid="apply-line" data-ok="0">{l.why ? t('applyDialog.notAppliedWhy', { file: l.file, why: l.why }) : t('applyDialog.notApplied', { file: l.file })}</div>
          ))}
          {outcome.applied.map((a) => {
            const note = pcMapNote(null, a.mapAction);
            return note ? (
              <div key={a.file} role="status" style={noteStyle} data-testid="apply-map-note" data-file={a.file}>
                {a.file}: {t(`calibration.map.${note.key}`, note.params)}
              </div>
            ) : null;
          })}
          {outcome.applied.length > 0 && (
            <PcGrant
              files={outcome.applied.map((a) => a.file)} pc={calibration.pc}
              previous={outcome.applied.map((a) => a.previous)} onUndo={onUndo}
            />
          )}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 8, marginTop: 12 }}>
        {list.state === 'ready' && !nothing && !done && count === 0 && (
          <span style={noteStyle} data-testid="apply-tick-hint">{t('applyDialog.tickHint')}</span>
        )}
        <Button onClick={onClose}>{nothing || done ? t('common:close') : t('common:cancel')}</Button>
        {list.state === 'ready' && !nothing && !done && count > 0 && (
          <Button variant="primary" disabled={busy} onClick={apply}>{t('applyDialog.apply', { count })}</Button>
        )}
      </div>
    </ModalShell>
  );
}
