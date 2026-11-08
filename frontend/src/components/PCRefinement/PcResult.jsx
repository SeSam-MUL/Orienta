// The Result panel of the PC page after a search (Find PC or Global PC Refine): one bold line that says how
// the search ended, the PC with the facts that help to judge it, what the page does about it (in use at once,
// or the pattern with its bands and the ways on), and Details with the sentences that depend on this result.
//
// The panel knows nothing of the backend: the page hands over the result and the handlers.
import React, { useState } from 'react';
import { useTranslation, Trans } from 'react-i18next';
import { colors, alpha, Button, FormRow, Label } from '../../theme/components';
import PcGrant, { PcUndoLine, BandsNeedPhase } from './PcGrant';
import { formatPc } from './pcFormat';
import { usePhaseLoaded } from './phaseContext';
import {
  warningOf, needsDecision, secondPositionOf, contrastOf, patternCountOf, patternSpread,
} from './pcResultFacts';

const grey = { fontSize: '9pt', color: colors.textSecondary };
const fact = { fontSize: '9pt', color: colors.text, margin: '2px 0' };

// The words for the numbers of the file's own convention, in the order x, y, z.
const OTHER_CONVENTION_KEY = { oxford: 'result.asOxford', tsl: 'result.asTsl' };

// The warning code each bold line says in its own words: no second, grey copy of it.
const HEAD_CODE = { edge: 'pcOnBoxEdge', notAMinimum: 'pcNotAMinimum', secondPosition: 'pcSecondPosition' };

// A scan split into 3 x 3 blocks gives nine patterns at most; fewer means blocks stayed empty.
const BLOCK_COUNT = 9;

function joinAxes(axes, language) {
  try {
    return new Intl.ListFormat(language, { style: 'long', type: 'conjunction' }).format(axes);
  } catch {
    return axes.join(', ');
  }
}

/**
 * @param {object}   result       the answer of /api/pc/optimize
 * @param {string}   file         the dataset the PC was given to
 * @param {number[]} pc           the result's PC (Bruker)
 * @param {boolean}  inUse        the PC in use for the file still is this result and the store took it
 * @param {object}   inConvention `{convention, pc}`: the PC in the open file's own convention, or null
 * @param {object}   patternsFrom `{kind: 'pick', noQuality}` when the list is the page's own pick of a scan split into blocks, `{kind: 'all'}` for a scan used completely, else
 *                                `{kind: 'user'}`
 * @param {string}   mapAction    'shifted' | 'replaced' | null: what the result did to a per-pixel PC map
 * @param {boolean}  busy         a search or a pick is running: the buttons that start one are off
 * @param {boolean}  bandsShown   "Show the bands over a pattern" is open
 * @param {Function} onToggleBands
 * @param {Function} renderBands  `(caption) => node`: the pattern with the bands of this PC
 * @param {Function} onUndo       `(file) => Promise`, see PcUndoLine
 * @param {Function} onFindAgain  Find PC again with other patterns
 * @param {Function} onKeep       "Keep this PC anyway"
 * @param {string}   keptFor      the file "Keep this PC anyway" was pressed for
 * @param {Function} onGoToIndexing  null: the page cannot switch pages, the button is left out
 * @param {Function} onSave       "Save as calibration…" (the PC of this result); null leaves it out
 * @param {Function} onCompare    "Compare with a saved calibration" in the decision box; null: no calibration is saved yet
 * @param {string}   warningText  the backend's warning in the reader's language, for a warning this panel has no line of its own for
 * @param {Array}    warningItems `[{code, text}]`: every warning of the result in the reader's language; the ones the bold line does not say are shown under it
 */
export default function PcResult({
  result, file, pc, inUse, inConvention = null, patternsFrom, mapAction = null, busy = false,
  bandsShown, onToggleBands, renderBands, onUndo, onFindAgain, onKeep, keptFor, onGoToIndexing = null, warningText = '',
  warningItems = [], onSave = null, onCompare = null,
}) {
  const { t, i18n } = useTranslation('pcrefinement');
  const phaseLoaded = usePhaseLoaded();
  const warning = warningOf(result);
  const decide = needsDecision(result);
  const notStored = result.pc_stored === false;
  const codes = Array.isArray(result.pc_warning_codes) ? result.pc_warning_codes : [];
  const otherWarning = !warning && !notStored && (codes.length > 0 || !!result.pc_warning);
  const second = secondPositionOf(result);
  const count = patternCountOf(result);
  const contrast = contrastOf(result);
  const spread = patternSpread(result);
  const search = result.search || {};

  // ---- the one bold line, and the grey words under it
  let head;
  if (notStored) {
    head = { tone: 'warn', bold: t('warnings.pcNotStored'), note: '' };
  } else if (warning?.kind === 'edge') {
    head = {
      tone: 'warn',
      bold: t('result.edgeTitle', { axes: joinAxes(warning.axes, i18n.language) }),
      note: `${t('result.edgeNote')} ${t('result.cannotJudge')}`,
    };
  } else if (warning?.kind === 'notAMinimum') {
    head = { tone: 'warn', bold: t('result.notMinTitle'), note: `${t('result.notMinNote')} ${t('result.cannotJudge')}` };
  } else if (warning?.kind === 'secondPosition') {
    head = { tone: 'warn', bold: t('result.secondTitle'), note: t('result.secondNote') };
  } else if (otherWarning) {
    head = { tone: 'warn', bold: warningText, note: '' };
  } else {
    head = { tone: 'ok', bold: t('result.endedNormally'), note: t('result.cannotJudge') };
  }
  const toneColor = head.tone === 'ok' ? colors.green : colors.orange;

  // The other warnings of the result, as grey lines under the bold one. A line that says the same as the bold
  // one is left out; a PC the store did not take is the bold line itself or, under another bold line, one of these.
  const headCode = notStored ? 'pcNotStored' : HEAD_CODE[warning?.kind];
  const otherLines = head.tone === 'warn' && !otherWarning
    ? warningItems.filter((w) => w.code !== headCode && w.text !== head.bold)
    : [];

  // ---- the sentences of Details that depend on this result
  const details = [];
  if (second && second.cost != null && Number.isFinite(result.quality?.best_cost)) {
    details.push(['pc-second-position', t('result.secondPosition', {
      pc: formatPc(second.pc), cost: second.cost.toFixed(2), best: result.quality.best_cost.toFixed(2) })]);
  }
  details.push(['pc-explain-contrast', t('result.explainContrast')]);
  if (patternsFrom?.kind === 'pick') {
    // After "other patterns" a block gives its best position that was left over, not its best one; with
    // fewer than nine patterns some blocks stayed empty (a scan of one or two rows or columns has no nine).
    const left = !!patternsFrom.excluded;
    const few = Number.isFinite(count) && count < BLOCK_COUNT;
    let key;
    if (patternsFrom.noQuality) key = 'result.patternsPickedNoQuality';
    else if (few) key = left ? 'result.patternsFewBlocksLeft' : 'result.patternsFewBlocks';
    else key = left ? 'result.patternsPickedLeft' : 'result.patternsPicked';
    details.push(['pc-explain-patterns', t(key, { count })]);
  } else if (patternsFrom?.kind === 'all') {
    details.push(['pc-explain-patterns', t('result.patternsAll', { count })]);
  } else {
    details.push(['pc-explain-patterns', t('result.patternsChosen')]);
  }
  if (mapAction === 'shifted') details.push(['pc-explain-map', t('result.mapShifted', { count })]);
  else if (mapAction === 'replaced') details.push(['pc-explain-map', t('result.mapReplaced')]);
  else details.push(['pc-explain-map', t('result.mapOne', { count })]);
  if (search.far_box_used) details.push(['pc-far-box', t('find.farBox')]);
  else if (search.recentred > 0) details.push(['pc-box-moved', t('find.boxMoved')]);
  if (result.file_pc && Number.isFinite(result.file_pc_deviation)) {
    details.push(['pc-file-vs-result', t('result.fileHasPc', {
      pc: formatPc(result.file_pc), deviation: result.file_pc_deviation.toFixed(4) })]);
  }

  const bands = { shown: bandsShown, onToggle: onToggleBands };
  const sentenceStyle = { margin: '1px 0' };

  return (
    <div data-testid="pc-result" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: '9.5pt', color: toneColor }} data-testid="pc-result-head">
        <b>{head.bold}</b>
        {head.note && head.tone === 'ok' && <>{' '}<span style={grey} data-testid="pc-cannot-judge">{head.note}</span></>}
      </div>
      {head.note && head.tone !== 'ok' && <div style={grey} data-testid="pc-result-note">{head.note}</div>}
      {otherLines.map((w) => <div key={w.code} style={grey} data-testid="pc-result-warning" data-code={w.code}>{w.text}</div>)}
      {head.tone === 'ok' && !notStored && (
        <div style={grey} data-testid="pc-good-enough">{t('result.goodEnough')}</div>
      )}

      <FormRow label={t('controls.meanPcLabel')}>
        <Label style={{ fontFamily: 'monospace', color: colors.accent, fontSize: '9pt' }}>
          <b data-testid="pc-result-pc">{formatPc(pc)}</b>
          {count != null && <span style={{ ...grey, marginLeft: 8, fontFamily: 'inherit' }}>{t('result.fromPatterns', { count })}</span>}
        </Label>
      </FormRow>
      {inConvention && OTHER_CONVENTION_KEY[inConvention.convention] && (
        <div style={fact} data-testid="pc-in-convention">
          {t(OTHER_CONVENTION_KEY[inConvention.convention], {
            a: inConvention.pc[0].toFixed(4), b: inConvention.pc[1].toFixed(4), c: inConvention.pc[2].toFixed(4) })}
        </div>
      )}
      {contrast != null && (
        <div style={fact} data-testid="pc-contrast">
          <Trans t={t} i18nKey="result.contrast" values={{ value: contrast.toFixed(1) }} components={{ b: <b /> }} />
        </div>
      )}
      {spread && (
        <div style={fact} data-testid="pc-spread">
          {spread.pixels == null && (
            <Trans t={t} i18nKey="result.disagree" values={{ value: spread.value.toFixed(2) }} components={{ b: <b /> }} />
          )}
          {spread.pixels === 0 && (
            <Trans t={t} i18nKey="result.disagreeUnderPx" values={{ value: spread.value.toFixed(2) }} components={{ b: <b /> }} />
          )}
          {spread.pixels > 0 && (
            <Trans t={t} i18nKey="result.disagreePx" count={spread.pixels}
                   values={{ value: spread.value.toFixed(2) }} components={{ b: <b /> }} />
          )}
        </div>
      )}

      {/* What the page does with the result: in use at once, or the pattern with the bands and the ways on */}
      {inUse && !decide && warning?.kind !== 'secondPosition' && (
        <PcGrant
          files={[file]} pc={pc} variant="nonum" previous={[result.previous || null]}
          bands={{ shown: bandsShown, onToggle: onToggleBands }}
          onGoToIndexing={onGoToIndexing} onUndo={onUndo}
          actions={onSave ? [{ key: 'save', label: t('grant.saveAsCalibration'), onClick: onSave }] : []}
        />
      )}
      {inUse && warning?.kind === 'secondPosition' && !decide && (
        <>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <Button small variant="primary" disabled={busy} onClick={onFindAgain}>{t('decide.findAgain')}</Button>
            <Button small disabled={!phaseLoaded} onClick={onToggleBands}>{t('grant.showBands')}</Button>
          </div>
          {!phaseLoaded && <BandsNeedPhase />}
          <PcGrant files={[file]} pc={pc} variant="quiet" />
          {result.previous && <PcUndoLine file={file} previous={result.previous} onUndo={onUndo} />}
        </>
      )}
      {inUse && decide && (
        <>
          <PcGrant files={[file]} pc={pc} variant="quiet" />
          <div data-testid="pc-decide" style={{
            border: `1px solid ${colors.accent}`, borderRadius: 4, padding: '8px 10px',
            display: 'flex', flexDirection: 'column', gap: 6,
          }}>
            <b style={{ fontSize: '9.5pt' }}>{t('decide.title')}</b>
            {renderBands(t('decide.caption'))}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <Button small variant="primary" disabled={busy} onClick={onFindAgain}>{t('decide.findAgain')}</Button>
              <span style={{ display: 'inline-flex', border: `1px solid ${colors.red}`, borderRadius: 4 }}>
                <Button small onClick={onKeep}>{t('decide.keep')}</Button>
              </span>
            </div>
            {result.previous && (
              <PcUndoLine file={file} previous={result.previous} onUndo={onUndo} testId="pc-decide-undo" />
            )}
            {onCompare && (
              <div>
                <Button small onClick={onCompare}>{t('decide.compare')}</Button>
              </div>
            )}
            {keptFor && (
              <div data-testid="pc-kept">
                <div role="status" style={{
                  fontSize: '9pt', color: colors.green, background: alpha(colors.green, 7),
                  border: `1px solid ${alpha(colors.green, 33)}`, borderRadius: 3, padding: '4px 8px',
                }}>
                  {t('decide.kept', { file: keptFor })}
                </div>
                {(onGoToIndexing || onSave) && (
                  <div style={{ marginTop: 4, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {onGoToIndexing && <Button small onClick={onGoToIndexing}>{t('grant.goToIndexing')}</Button>}
                    {onSave && <Button small onClick={onSave}>{t('grant.saveAsCalibration')}</Button>}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
      {inUse && !decide && bandsShown && renderBands(t('bands.caption'))}

      <details data-testid="pc-details" style={{ ...grey, margin: '4px 0' }}>
        <summary>{t('find.details')}</summary>
        {details.map(([id, text]) => <div key={id} data-testid={id} style={sentenceStyle}>{text}</div>)}
        {inUse && warning && onSave && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 4 }} data-testid="pc-details-save">
            <Button small onClick={onSave}>{t('grant.saveAsCalibration')}</Button>
            <span>{t('result.keepsWarning')}</span>
          </div>
        )}
      </details>
    </div>
  );
}
