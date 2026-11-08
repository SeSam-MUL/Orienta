// The question after a refine at a file that was on a saved calibration: other files are still on that version,
// should they use the new PC too? Nothing is passed on unless a button says so; "Only this file" is the one to
// choose when unsure. The change is shown first, with its size in pixels, and the fit contrast before and after.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pcCalibrationsApi } from '../../../services/api';
import { colors, alpha, Button } from '../../../theme/components';
import PcGrant from '../PcGrant';
import { pcMapActionOf, pcMapNote } from '../pcMapNote';
import { noteStyle, factStyle, alertStyle, warnStyle } from './CalibrationFacts';
import {
  followerQuestion, questionKey, pcChange, changeInPixels, formatPixels, isTinyChange, formatSigned, contrastText, applyRows, isoDate,
  settingNameKey,
} from './calibrationText';
import { refusalWords } from './calibrationWords';

const askBox = {
  border: `1px solid ${colors.accent}`, background: alpha(colors.accent, 4), borderRadius: 4,
  padding: '8px 10px', fontSize: '9pt', color: colors.text, display: 'flex', flexDirection: 'column', gap: 4,
};
// A button and its sentence stand in one row: the sentence takes the width that is left and wraps inside it, it does
// not drop under the button (the sentence of "New version" is the longest of the three).
const actStyle = { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'nowrap' };
const beside = { ...noteStyle, flexGrow: 1, flexShrink: 1, flexBasis: '0px', minWidth: 0 };

/**
 * @param {object}   result     the answer of /api/pc/optimize
 * @param {Function} onAnswered `(state)`: 'only' | 'newVersion': the question is answered and Save may show
 * @param {Function} onSaveNew  the page's Save as calibration window for the open file
 * @param {boolean}  savedNew   the new PC was saved as a calibration of its own: the question is answered
 * @param {Function} onApplyToOthers `(calibration)`: the window "Apply to other files" for the new version
 * @param {Function} onPcChanged the page loads its PC from the backend again
 * @param {Function} onUndo     the page's put back
 * @param {Function} onOnlyThis `({id, name, version, latest})`: "Only this file" was pressed; the file keeps its own PC
 *                              and the calibration it left is not offered to it again until it is loaded again
 */
export default function FollowerQuestion({
  result, onAnswered, onSaveNew, savedNew = false, onApplyToOthers, onPcChanged, onUndo, onOnlyThis = null,
}) {
  const { t, i18n } = useTranslation(['pcrefinement', 'common']);
  const question = followerQuestion(result?.calibration);
  const key = questionKey(result);
  const [facts, setFacts] = useState(null);          // {from, notIncluded: [{file, reasons}]}
  const [factsError, setFactsError] = useState('');
  const [answer, setAnswer] = useState(null);        // {kind, made?, applied?}
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  // (the page gives this component the key of the question, so a new question starts with a clean state)
  useEffect(() => {
    if (!question) return undefined;
    let live = true;
    const { id, version } = question.calibration;
    Promise.all([pcCalibrationsApi.get(id), pcCalibrationsApi.candidates(id)]).then(([got, cands]) => {
      if (!live) return;
      const v = (got.data?.versions || []).find((x) => Number(x.version) === Number(version));
      const rows = applyRows(cands.data?.candidates, result.dataset);
      const notIncluded = rows.others
        .filter((r) => r.kind === 'differs' && !question.followers.includes(r.dataset))
        .map((r) => ({ file: r.dataset, reasons: r.differences }));
      setFacts({ from: Array.isArray(v?.pc) ? v.pc : null, notIncluded });
    }).catch((err) => { if (live) setFactsError(refusalWords(err, t)); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (!question) return null;
  const { calibration, followers, maps } = question;
  const to = result.mean_pc;
  const change = facts?.from ? pcChange(facts.from, to) : null;
  const pixels = facts?.from ? changeInPixels(facts.from, to, result.pattern_size) : null;
  const after = contrastText(result.quality?.contrast);
  const before = contrastText(result.quality?.contrast_before);
  const list = (items) => {
    try {
      return new Intl.ListFormat(i18n.language, { style: 'long', type: 'conjunction' }).format(items);
    } catch {
      return items.join(', ');
    }
  };
  const reasonText = (r) => t('question.reason', {
    setting: t(settingNameKey(r.key)), file: r.file ?? t('calibration.settings.notStated'),
    calibration: r.calibration ?? t('calibration.settings.notStated'),
  });
  const notIncludedText = (facts?.notIncluded || [])
    .map((n) => `${n.file} (${n.reasons.map(reasonText).join('; ')})`).join(', ');

  const newVersion = async () => {
    setBusy(true); setError('');
    try {
      // the followers were put on this calibration by the user before; they follow it to its new version
      // `from_version`: the version the followers were on when this was asked; one that has another PC since is left alone
      const res = await pcCalibrationsApi.newVersion(calibration.id, result.dataset, followers, followers, calibration.version);
      const made = res.data?.calibration || {};
      const answers = res.data?.applied || [];
      const applied = answers.filter((a) => a.status === 'applied');
      const left = answers.filter((a) => a.status === 'left').map((a) => a.dataset);
      setAnswer({ kind: 'newVersion', made, applied, left, failed: answers.length - applied.length - left.length });
      onAnswered('newVersion');
      if (onPcChanged) onPcChanged();
    } catch (err) {
      setError(refusalWords(err, t));
    } finally {
      setBusy(false);
    }
  };

  if (savedNew) {
    return (
      <div role="status" style={factStyle} data-testid="question-answered">
        {t('question.saveNewDone', { version: calibration.version, name: calibration.name })}
      </div>
    );
  }
  if (answer?.kind === 'only') {
    return (
      <div role="status" style={factStyle} data-testid="question-answered">
        {t('question.onlyDone', { count: followers.length, file: result.dataset, files: list(followers), version: calibration.version })}
      </div>
    );
  }
  if (answer?.kind === 'newVersion') {
    const made = answer.made;
    return (
      <div data-testid="question-answered" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div role="status" style={{ ...factStyle, color: colors.green }} data-testid="question-version-done">
          {t('question.newVersionDone', { name: made.name || calibration.name, version: made.version, date: isoDate(made.version_created) })}
        </div>
        {answer.failed > 0 && <div style={alertStyle}>{t('question.partly', { count: answer.failed })}</div>}
        {answer.left.map((file) => (
          <div key={file} role="status" style={warnStyle} data-testid="question-left" data-file={file}>{t('question.left', { file })}</div>
        ))}
        {answer.applied.map((a) => {
          const note = pcMapNote(null, pcMapActionOf(a.pc_map_action));
          return note ? (
            <div key={a.dataset} role="status" style={noteStyle} data-testid="question-map-note" data-file={a.dataset}>
              {a.dataset}: {t(`calibration.map.${note.key}`, note.params)}
            </div>
          ) : null;
        })}
        {facts?.notIncluded?.length > 0 && (
          <div style={warnStyle} data-testid="question-not-included-after">{t('question.notIncludedAfter', { text: notIncludedText })}</div>
        )}
        {onApplyToOthers && (
          <div>
            <Button small variant="primary" onClick={() => onApplyToOthers(made)}>{t('question.applyToOthers')}</Button>
          </div>
        )}
        {answer.applied.length > 0 && (
          <PcGrant
            files={answer.applied.map((a) => a.dataset)} pc={made.pc || to}
            previous={answer.applied.map((a) => a.previous ?? null)} onUndo={onUndo}
          />
        )}
      </div>
    );
  }

  return (
    <div data-testid="follower-question" style={askBox} role="group" aria-label={t('question.title')}>
      <div><b>{t('question.ask', { files: list(followers) })}</b></div>
      {change && (
        <div style={factStyle} data-testid="question-change">
          {t(pixels == null ? 'question.changeNoPixels' : (isTinyChange(pixels) ? 'question.changeTiny' : 'question.change'), {
            x: formatSigned(change.delta[0]), y: formatSigned(change.delta[1]), z: formatSigned(change.delta[2]),
            pixels: pixels != null ? formatPixels(pixels) : '',
          })}
          {before != null && after != null && ` ${t('question.contrast', { before, after })}`}
        </div>
      )}
      {factsError && <div role="alert" style={alertStyle}>{factsError}</div>}
      <div style={factStyle} data-testid="question-files">
        {t('question.onVersion', { version: calibration.version, name: calibration.name })} <b>{followers.join(', ')}</b>
      </div>
      {/* before the click: which followers hold a PC for every map point, and what "New version" does to that map */}
      {maps.grid.length > 0 && (
        <div style={noteStyle} data-testid="question-map-grid">{t('question.mapGrid', { files: list(maps.grid), count: maps.grid.length })}</div>
      )}
      {maps.header.length > 0 && (
        <div style={noteStyle} data-testid="question-map-header">{t('question.mapHeader', { files: list(maps.header), count: maps.header.length })}</div>
      )}
      {facts?.notIncluded?.length > 0 && (
        <div style={warnStyle} data-testid="question-not-included">{t('question.notIncluded', { text: notIncludedText })}</div>
      )}
      <div style={actStyle}>
        <Button small variant="primary" disabled={busy} onClick={() => { setAnswer({ kind: 'only' }); onAnswered('only'); if (onOnlyThis) onOnlyThis(calibration); }}>
          {t('question.onlyThis')}
        </Button>
        <span style={beside} data-testid="question-only-note">{t('question.onlyThisNote')}</span>
      </div>
      <div style={actStyle} data-testid="question-new-version-row">
        <Button small disabled={busy || !facts} onClick={newVersion}>{t('question.newVersion', { count: followers.length })}</Button>
        <span style={beside} data-testid="question-new-version-note">
          {t('question.newVersionNote', { count: followers.length, next: Number(calibration.latest ?? calibration.version) + 1, files: list(followers) })}
        </span>
      </div>
      <div style={actStyle}>
        <Button small disabled={busy} onClick={onSaveNew}>{t('question.saveNew')}</Button>
        <span style={beside} data-testid="question-save-new-note">{t('question.saveNewNote', { version: calibration.version })}</span>
      </div>
      {error && <div role="alert" style={alertStyle} data-testid="question-error">{error}</div>}
    </div>
  );
}
