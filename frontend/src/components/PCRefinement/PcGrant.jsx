// What the page says every time it gives a file a PC: which PC, for how long, that the file on disk is
// untouched, what the PC was before and the one step back ("Put it back" / "Remove it", each with its
// confirmation). Used by the Find PC result; the other ways of giving a PC (a typed one, a saved
// calibration, another file) hand over their own extra buttons through `actions`.
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../theme/components';
import { formatPc } from './pcFormat';
import { usePhaseLoaded } from './phaseContext';

const okBox = {
  fontSize: '9pt', color: colors.green, background: alpha(colors.green, 7),
  border: `1px solid ${alpha(colors.green, 33)}`, borderRadius: 3, padding: '4px 8px',
  wordBreak: 'break-word',
};

const quietBox = { fontSize: '9pt', color: colors.textSecondary, wordBreak: 'break-word' };

/**
 * One file's step back. `previous` is what the backend knows of the PC before the last write:
 * `{pc: [x, y, z] | null, had_pc: boolean}`. A file that had no PC offers "Remove it", one that had
 * offers "Put it back" with the numbers; both ask first.
 *
 * `onUndo(file)` does the work and returns nothing when it is done (the caller then replaces this
 * line), `{data}` (the backend's answer) when it is done and this line is to say so itself, `{gone: true}` when
 * the step no longer exists (a file was loaded since) or `{error: text}`.
 */
export function PcUndoLine({ file, previous, many = false, onUndo, testId = 'pc-undo' }) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  const [phase, setPhase] = useState('idle');     // 'idle' | 'ask' | 'busy' | 'gone' | 'done'
  const [error, setError] = useState('');
  const [done, setDone] = useState(null);         // {previous: {pc, had_pc}, notRestored}
  const had = !!previous?.had_pc;
  const pc = formatPc(previous?.pc);

  const run = async () => {
    setPhase('busy');
    setError('');
    let answer;
    try {
      answer = await onUndo(file);
    } catch (err) {
      answer = { error: err?.message || String(err) };
    }
    if (answer?.gone) { setPhase('gone'); return; }
    if (answer?.error) { setError(answer.error); setPhase('idle'); return; }
    if (answer?.data) {
      const d = answer.data;
      setDone({ previous: { pc: d.pc ?? null, had_pc: !!d.had_pc }, notRestored: Array.isArray(d.not_restored) ? d.not_restored : [] });
      setPhase('done');
    }
  };

  let body;
  if (phase === 'done') {
    return <PcUndoDone file={file} previous={done.previous} notRestored={done.notRestored} />;
  }
  if (phase === 'gone') {
    body = <span role="status">{t('pcrefinement:grant.gone')}</span>;
  } else if (phase === 'ask' || phase === 'busy') {
    body = (
      <>
        <span>
          {had
            ? t('pcrefinement:grant.askPutBack', { file, pc })
            : t('pcrefinement:grant.askRemove', { file })}
        </span>{' '}
        <Button small variant="primary" disabled={phase === 'busy'} onClick={run}>
          {had ? t('pcrefinement:grant.yesPutBack') : t('pcrefinement:grant.yesRemove')}
        </Button>{' '}
        <Button small disabled={phase === 'busy'} onClick={() => setPhase('idle')}>
          {t('common:cancel')}
        </Button>
        {phase === 'busy' && (
          <div role="status" data-testid="pc-undo-working">{had ? t('pcrefinement:grant.working') : t('pcrefinement:grant.workingRemove')}</div>
        )}
      </>
    );
  } else {
    body = (
      <>
        <span>
          {had
            ? t(many ? 'pcrefinement:grant.previousMany' : 'pcrefinement:grant.previousOne', { file, pc })
            : t(many ? 'pcrefinement:grant.noPcBeforeMany' : 'pcrefinement:grant.noPcBeforeOne', { file })}
        </span>{' '}
        <Button small onClick={() => setPhase('ask')}>
          {had ? t('pcrefinement:grant.putBack') : t('pcrefinement:grant.remove')}
        </Button>
      </>
    );
  }
  return (
    <div data-testid={testId} data-file={file} style={{ fontSize: '9pt', color: colors.textSecondary }}>
      {body}
      {error && <div role="alert" style={{ color: colors.red, marginTop: 2 }}>{error}</div>}
    </div>
  );
}

/** The words beside a "Show the bands over a pattern" that is off because no phase is loaded. */
export function BandsNeedPhase() {
  const { t } = useTranslation('pcrefinement');
  return <div data-testid="pc-bands-need-phase" role="status" style={{ ...quietBox, color: colors.orange }}>{t('grant.bandsNeedPhase')}</div>;
}

/** What the page says after the step back: the previous PC is back, or the PC is gone. */
export function PcUndoDone({ file, previous, notRestored = [] }) {
  const { t } = useTranslation('pcrefinement');
  const had = !!previous?.had_pc;
  return (
    <div data-testid="pc-undo-done">
      <div role="status" style={okBox}>
        {had
          ? t('grant.donePutBack', { file, pc: formatPc(previous.pc) })
          : t('grant.doneRemove', { file })}
      </div>
      {notRestored.length > 0 && (
        <div data-testid="pc-not-restored" style={{ fontSize: '9pt', color: colors.textSecondary, marginTop: 4 }}>
          {t('grant.notRestored', { names: notRestored.join(', ') })}
        </div>
      )}
    </div>
  );
}

/**
 * @param {string[]}  files     the files that now use `pc`
 * @param {number[]}  pc        the PC (Bruker x, y, z)
 * @param {Array}     previous  per file: `{pc, had_pc}` or null/undefined when there is no step back
 * @param {Array}     actions   extra buttons `{key, label, onClick, title?}` shown after the main button and
 *                              "Go to indexing" (Save as calibration, ...)
 * @param {object}    bands     `{shown, onToggle}` for "Show the bands over a pattern", the main button, first on
 *                              the left; null leaves it out (a result that carries the bands beside it anyway)
 * @param {Function}  onGoToIndexing  "Go to indexing" after the bands button; null leaves it out
 * @param {Function}  onUndo    `(file) => Promise<void | {gone} | {error}>`, see PcUndoLine; null: no step back
 * @param {boolean}   perPixel  the one file holds a PC for every map point and `pc` is the mean of its map ('full' only)
 * @param {string}    variant   'full': the line names the PC. 'nonum': the same line without the numbers (a
 *                              result that shows them in the row above). 'quiet': that line in grey, no buttons
 *                              and no step back (the result is still to be judged, the decision box has them)
 */
export default function PcGrant({
  files, pc, previous = [], actions = [], bands = null, onGoToIndexing = null, onUndo = null, variant = 'full',
  perPixel = false,
}) {
  const { t } = useTranslation('pcrefinement');
  const phaseLoaded = usePhaseLoaded();
  const many = files.length > 1;
  const quiet = variant === 'quiet';
  const withNumbers = variant === 'full';
  // the numbers of a file that holds a PC for every map point are the mean of its map: the line says so
  const numbersKey = many ? 'grant.inUseMany' : (perPixel ? 'grant.inUseOneMap' : 'grant.inUseOne');
  const text = withNumbers
    ? t(numbersKey, { files: files.join(', '), pc: formatPc(pc) })
    : t(many ? 'grant.inUseManyNoPc' : 'grant.inUseOneNoPc', { files: files.join(', ') });
  const hasButtons = !quiet && (actions.length > 0 || bands || onGoToIndexing);
  return (
    <div data-testid="pc-grant" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div role="status" data-testid="pc-grant-text" style={quiet ? quietBox : okBox}>{text}</div>
      {hasButtons && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {bands && (
            <Button small variant="primary" disabled={!phaseLoaded} onClick={bands.onToggle}>
              {t('grant.showBands')}
            </Button>
          )}
          {onGoToIndexing && (
            <Button small onClick={onGoToIndexing}>{t('grant.goToIndexing')}</Button>
          )}
          {actions.map((a) => (
            <Button small key={a.key} onClick={a.onClick} title={a.title}>{a.label}</Button>
          ))}
        </div>
      )}
      {bands && !quiet && !phaseLoaded && <BandsNeedPhase />}
      {onUndo && !quiet && files.map((file, i) => (previous[i] ? (
        <PcUndoLine key={file} file={file} previous={previous[i]} many={many} onUndo={onUndo} />
      ) : null))}
    </div>
  );
}
