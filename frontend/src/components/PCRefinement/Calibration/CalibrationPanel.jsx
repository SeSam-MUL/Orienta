// The calibration panel of the PC page: a saved calibration is OFFERED for the open file, and the window
// "Take the pattern centre from..." (a saved calibration or another file) is mounted here.
//
// THE ONE RULE: nothing changes without a button. An offer is only a question. "Use it" gives the file the PC of
// the calibration when the settings agree; a calibration whose settings differ or cannot be checked needs the
// second naming, "Use anyway", and the request then names the file again (`accept_mismatch`). "Not now" hides the
// offer until the file is loaded again. The backend enforces the same, so a page that forgot would be refused.
//
// What saving a calibration and applying it to other files hangs on:
//   `grantActions`    extra buttons after "Go to indexing" on the line that follows a use: [{key, label, onClick}]
//   `renderAfterUse`  `({offer, answer}) => node`, shown under "The PC of the saved calibration is in use"
//   `onApplied`       `({offer, answer})` once a calibration was applied to the open file
//   `onApplyToOthers` `({id, name, version, pc})`: the button of the line "This file is on <name>, version n."
//   `silenced`        ids of calibrations that are not offered for this file now (the page ends it with the load)
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pcCalibrationsApi } from '../../../services/api';
import useDataStore from '../../../stores/useDataStore';
import { colors, alpha, Button } from '../../../theme/components';
import { samePc } from '../pcFormat';
import AppliedGrant from './AppliedGrant';
import {
  CalibrationFacts, SettingsTable, SettingsFolded, noteStyle, factStyle, alertStyle,
} from './CalibrationFacts';
import {
  offerKind, differenceLines, replacesOf, formatPixels, isTinyChange, versionMeta, isoDate, settingNameKey, copyKeptOf,
} from './calibrationText';
import { refusalWords, wasAnswered } from './calibrationWords';
import { TakePcDialog } from './TakePcDialog';

const askBox = {
  border: `1px solid ${colors.accent}`, background: alpha(colors.accent, 4), borderRadius: 4,
  padding: '6px 8px', fontSize: '9pt', color: colors.text, display: 'flex', flexDirection: 'column', gap: 2,
};
const rowStyle = { display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 };

/** The sentence under the table of each kind of offer. */
const STATEMENT = { fits: 'offer.fits', unstated: 'offer.unstated' };

export const offerKey = (dataset, offer) => `${dataset}|${offer.id}|${offer.version}`;

export default function CalibrationPanel({
  dataset, usedPc, pcInUse, takePcOpen, onTakePcOpenChange, onPcChanged, onUsePc, onUndo, onGoToIndexing = null,
  renderBands = null, onFindPc = null, canFindPc = true, pickFile = null, grantActions = [], renderAfterUse = null,
  onApplied = null, onApplyToOthers = null, silenced = null,
}) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  const [offer, setOffer] = useState(null);
  const [applied, setApplied] = useState(null);       // {dataset, offer, answer, confirmed, undone}
  const [dismissed, setDismissed] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  // A new load of a file brings back the offers that were put off, and ends what was said about the last one.
  // The mark is the load's own (the id of its request) or, where the answer has none, the file it opened: NOT the
  // `ebsdInfo` object, which is also replaced when the grid is edited and says nothing of a load.
  const loadMark = useDataStore((s) => s.ebsdInfo?.request_id ?? s.ebsdInfo?.file_path ?? null);
  const datasetRef = useRef(dataset);
  datasetRef.current = dataset;

  useEffect(() => { setDismissed(new Set()); setApplied(null); setError(''); }, [loadMark, dataset]);

  // Asked again whenever the file or its PC changes: what the offer would replace is the PC in use now.
  const pcKey = useMemo(() => JSON.stringify([
    usedPc ?? null, pcInUse?.kind ?? null, pcInUse?.calibration?.id ?? null, pcInUse?.calibration?.version ?? null,
  ]), [usedPc, pcInUse]);
  useEffect(() => {
    // not before the page knows its PC in use: what an offer would replace is that PC
    if (!dataset || usedPc === undefined) { setOffer(null); return undefined; }
    let live = true;
    pcCalibrationsApi.suggest(dataset).then((res) => {
      if (!live) return null;
      // a name the backend no longer knows (the file the page has just left): no offer, and nothing to say about it
      if (res.data?.unknown_dataset === true) return null;
      const list = res.data?.suggestions;
      // the offer names the FILE it is applied to: for an open copy, the file the copy was made from
      if (Array.isArray(list) && list.length > 0) return { ...list[0], target: res.data?.target || dataset };
      if (res.data?.unverified !== true) return null;
      // A file that states none of its settings (an .up1, an .ebsp) gets no fitting calibration, as nothing but the
      // pattern size could be compared. The calibrations of that size are asked for and offered as "not stated"
      // (kind `unstated`), the newest first; one of another size is not, and not the version the file is on.
      return pcCalibrationsApi.suggest(dataset, true).then((all) => {
        const current = all.data?.current;
        const same = (Array.isArray(all.data?.suggestions) ? all.data.suggestions : []).find((s) => (
          s.unverified === true && (s.differences || []).length === 0
          && !(current && current.id === s.id && current.version === s.version)));
        return same ? { ...same, target: all.data?.target || dataset } : null;
      });
    }).then((found) => {
      if (live) setOffer(found);
    }).catch((err) => {
      if (!live) return;
      setOffer(null);
      // a backend that said no says why; one that could not be reached is the page's other requests' news
      if (wasAnswered(err)) setError(refusalWords(err, t));
    });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, pcKey, loadMark]);

  // The grant stays true only while the PC in use is the one it names: once the page has shown that PC, any other
  // PC (Find PC, a typed one, another file) ends it.
  useEffect(() => {
    if (!applied || applied.undone) return;
    const same = samePc(usedPc, applied.answer.pc);
    if (same && !applied.confirmed) setApplied((a) => (a ? { ...a, confirmed: true } : a));
    else if (!same && applied.confirmed) setApplied(null);
  }, [usedPc, applied]);

  const use = async (item, anyway) => {
    const forDataset = dataset;
    const target = item.target || forDataset;          // apply refuses a copy: the calibration goes to its file
    setBusy(true);
    setError('');
    try {
      const res = await pcCalibrationsApi.apply(item.id, [target], anyway ? [target] : [], item.version);
      if (datasetRef.current !== forDataset) return;
      const answer = (res.data?.results || [])[0];
      if (answer?.status === 'applied') {
        setApplied({ dataset: forDataset, offer: item, answer, kind: offerKind(item), confirmed: false, undone: false });
        if (onApplied) onApplied({ offer: item, answer });
        if (onPcChanged) onPcChanged(answer);       // the page loads its PC from the backend, as after a put back
      } else {
        setError(t(`calibration.apply.${answer?.status || 'failed'}`, { defaultValue: t('calibration.apply.failed') }));
        if (onPcChanged) onPcChanged();             // a refusal: the page asks the backend what the PC is now
      }
    } catch (err) {
      if (datasetRef.current === forDataset) setError(refusalWords(err, t));
      if (onPcChanged) onPcChanged();
    } finally {
      if (datasetRef.current === forDataset) setBusy(false);
    }
  };

  const showApplied = applied && applied.dataset === dataset;
  const hidden = offer && dismissed.has(offerKey(dataset, offer));
  // a calibration the person has just chosen to leave ("Only this file") is not offered again, whatever its version,
  // and no "comes back" note either: nothing was put off, the answer was no
  const silent = !!offer && Array.isArray(silenced) && silenced.includes(offer.id);
  const showOffer = offer && !hidden && !silent && (!showApplied || applied.undone);
  const kind = offer ? offerKind(offer) : null;

  let ask = null;
  if (showOffer) {
    const replaces = kind === 'fits' ? replacesOf(offer) : null;
    const diffs = kind === 'differs' ? differenceLines(offer) : [];
    const calDate = isoDate(offer.version_created);
    const fileDate = isoDate(offer.file_date);
    const statement = kind === 'differs'
      ? (diffs.length === 1 ? t('calibration.offer.usedBeforeOne') : t('calibration.offer.usedBeforeMany', { n: diffs.length }))
      : t(`calibration.${STATEMENT[kind]}`);
    ask = (
      <div data-testid="calibration-offer" data-kind={kind} role="group" aria-label={t('calibration.title')} style={askBox}>
        <div style={factStyle}>{statement}</div>
        <CalibrationFacts item={offer} />
        {kind === 'fits' && <SettingsFolded offer={offer} />}
        {diffs.map((d) => (
          <div key={d.key} role="alert" style={alertStyle} data-testid="calibration-difference">
            {d.aspect ? t('calibration.diff.aspect') : t('calibration.diff.offer', {
              setting: t(settingNameKey(d.key)),
              file: d.file ?? t('calibration.settings.notStated'),
              calibration: d.calibration ?? t('calibration.settings.notStated'),
            })}
          </div>
        ))}
        {kind !== 'fits' && <SettingsTable offer={offer} />}
        {kind === 'unstated' && <div style={noteStyle}>{t('calibration.offer.unstatedUse')}</div>}
        {replaces && (
          <div style={factStyle} data-testid="calibration-replaces">
            {t(`calibration.replaces.${replaces.kind}`)}
            {replaces.pixels != null && ` ${isTinyChange(replaces.pixels)
              ? t('calibration.replaces.changeTiny')
              : t('calibration.replaces.change', { pixels: formatPixels(replaces.pixels) })}`}
          </div>
        )}
        {kind === 'fits' && calDate && fileDate && (
          <div style={noteStyle} data-testid="calibration-dates">
            {t('calibration.offer.dates', { version: offer.version, calibrationDate: calDate, fileDate })}
          </div>
        )}
        <div style={rowStyle}>
          {kind === 'fits' ? (
            <Button small variant="primary" disabled={busy} onClick={() => use(offer, false)}>{t('calibration.offer.use')}</Button>
          ) : (
            <>
              {onFindPc && (
                <Button small variant="primary" disabled={!canFindPc || busy} onClick={onFindPc}>{t('find.button')}</Button>
              )}
              <Button small variant="danger" disabled={busy} onClick={() => use(offer, true)}>{t('calibration.offer.useAnyway')}</Button>
            </>
          )}
          <Button small disabled={busy} onClick={() => setDismissed((s) => new Set(s).add(offerKey(dataset, offer)))}>
            {t('calibration.offer.notNow')}
          </Button>
        </div>
      </div>
    );
  }

  let after = null;
  if (showApplied) {
    const meta = versionMeta(applied.offer);
    after = (
      <div data-testid="calibration-applied">
        <AppliedGrant
          file={applied.answer.dataset || applied.dataset}
          pc={applied.answer.pc}
          previous={applied.answer.previous ?? null}
          mapAction={applied.answer.pc_map_action}
          copyKept={copyKeptOf(applied.answer, applied.dataset)}
          onUndo={onUndo}
          onUndone={() => setApplied((a) => (a ? { ...a, undone: true } : a))}
          onGoToIndexing={onGoToIndexing}
          renderBands={renderBands}
          actions={grantActions}
        >
          {/* the offer that fit says which version it took; "Use anyway" ends in the grant line alone */}
          {applied.kind === 'fits' && (
            <>
              <div style={factStyle} data-testid="calibration-in-use">{t('calibration.offer.inUse')}</div>
              <div style={noteStyle} data-testid="calibration-in-use-from">
                {t(`calibration.inline.${meta.key}`, { name: applied.offer.name, ...meta.params })}
              </div>
            </>
          )}
          {renderAfterUse && renderAfterUse({ offer: applied.offer, answer: applied.answer })}
        </AppliedGrant>
      </div>
    );
  }

  // A file that runs on a saved calibration says so on every visit, not only in the moment after Save or Use it:
  // the way to "Apply to other files…" must not depend on which button was pressed last. The grant that follows a
  // use has this button already, so the line waits while that grant is shown.
  const onCal = pcInUse?.calibration;
  const calLine = onApplyToOthers && onCal?.id && Array.isArray(usedPc) && usedPc.length === 3
    && !(showApplied && !applied.undone);

  const later = offer && hidden;
  const anything = ask || after || error || later || calLine;
  return (
    <>
      {anything && (
        <div data-testid="calibration-panel" style={{ display: 'flex', flexDirection: 'column', gap: 4, margin: '2px 0 6px' }}>
          {ask}
          {after}
          {calLine && (
            <div data-testid="calibration-on" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={factStyle}>{t('calibration.onVersion', { name: onCal.name ?? onCal.id, version: onCal.version })}</span>
              <Button small onClick={() => onApplyToOthers({ ...onCal, name: onCal.name ?? onCal.id, pc: usedPc })}>
                {t('calibration.applyToOthers')}
              </Button>
            </div>
          )}
          {later && !showApplied && (
            <div style={noteStyle} data-testid="calibration-later">{t('calibration.offer.later')}</div>
          )}
          {error && <div role="alert" data-testid="calibration-error" style={alertStyle}>{error}</div>}
        </div>
      )}
      {takePcOpen && (
        <TakePcDialog
          dataset={dataset}
          onClose={() => onTakePcOpenChange && onTakePcOpenChange(false)}
          onPcChanged={onPcChanged}
          onUsePc={onUsePc}
          onUndo={onUndo}
          onGoToIndexing={onGoToIndexing}
          renderBands={renderBands}
          pickFile={pickFile}
          grantActions={grantActions}
        />
      )}
    </>
  );
}
