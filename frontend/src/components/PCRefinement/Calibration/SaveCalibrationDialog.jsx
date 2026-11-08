// "Save as calibration": the PC of a file, with the settings it was acquired with, as a named calibration (version 1).
// The window says which PC, the fit contrast and the warning of the result it comes from; the name is checked
// while it is typed; after Save it offers "Apply to other files…". Nothing about the file changes.
import React, { useEffect, useRef, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { pcCalibrationsApi } from '../../../services/api';
import { Button, Input } from '../../../theme/components';
import { formatPc } from '../pcFormat';
import ModalShell from './ModalShell';
import { noteStyle, factStyle, alertStyle, warnStyle } from './CalibrationFacts';
import { contrastText, isoDate, savedWarning } from './calibrationText';
import { refusalWords } from './calibrationWords';

const nameKey = (v) => String(v || '').split(/\s+/).filter(Boolean).join(' ').toLowerCase();

/**
 * @param {string}   dataset  the file whose PC is saved
 * @param {number[]} pc       the PC shown in the window (what the line that opened it gave)
 * @param {object}   quality  `quality` of the result the PC came from (contrast, warning), or null for a PC typed or taken
 * @param {boolean}  perPixel the file holds a PC for every map point: the calibration keeps their mean
 * @param {Function} onClose
 * @param {Function} onSaved  `(created)` once the calibration exists
 * @param {Function} onApplyToOthers  `(created)`: the button after Save; null leaves it out
 */
export default function SaveCalibrationDialog({
  dataset, pc, quality = null, perPixel = false, onClose, onSaved = null, onApplyToOthers = null,
}) {
  const { t, i18n } = useTranslation(['pcrefinement', 'common']);
  const [name, setName] = useState('');
  const [taken, setTaken] = useState(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [created, setCreated] = useState(null);
  const input = useRef(null);
  useEffect(() => { input.current?.focus(); }, []);

  useEffect(() => {
    let live = true;
    pcCalibrationsApi.list().then((res) => {
      if (live) setTaken(new Set((res.data?.calibrations || []).map((c) => nameKey(c.name))));
    }).catch(() => {});       // the backend checks the name again when Save is pressed
    return () => { live = false; };
  }, []);

  const clean = name.trim();
  const nameTaken = clean !== '' && taken.has(nameKey(clean));
  const contrast = contrastText(quality?.contrast);
  const warning = savedWarning(quality);

  const save = async () => {
    if (!clean || nameTaken || busy || created) return;
    setBusy(true);
    setError('');
    try {
      const res = await pcCalibrationsApi.create(clean, dataset);
      setCreated(res.data);
      if (onSaved) onSaved(res.data);
    } catch (err) {
      setError(refusalWords(err, t));
    } finally {
      setBusy(false);
    }
  };

  let savedText = null;
  if (created) {
    const savedContrast = contrastText(created.quality?.contrast);
    const savedWarn = savedWarning(created.quality);
    const holds = [t('calibration.save.holdsPc', { pc: formatPc(created.pc) })];
    if (savedContrast != null) holds.push(t('calibration.save.holdsContrast', { value: savedContrast }));
    if (savedWarn) holds.push(t(`calibration.save.holdsWarning.${savedWarn}`));
    holds.push(t('calibration.save.holdsSettings', { file: created.source_name || dataset }));
    let list;
    try {
      list = new Intl.ListFormat(i18n.language, { style: 'long', type: 'conjunction' }).format(holds);
    } catch {
      list = holds.join(', ');
    }
    savedText = t('calibration.save.saved', {
      name: created.name, date: isoDate(created.version_created), version: created.version, holds: list,
    });
  }

  return (
    <ModalShell title={t('calibration.save.title')} testId="calibration-save-dialog" onClose={onClose} zIndex={1100}>
      <div style={noteStyle}>{t('calibration.save.intro', { pc: formatPc(pc) })}</div>
      {perPixel && <div style={noteStyle} data-testid="save-per-pixel">{t('calibration.save.perPixel')}</div>}
      {contrast != null && (
        <div style={factStyle} data-testid="save-contrast">
          <Trans t={t} i18nKey="result.contrast" values={{ value: contrast }} components={{ b: <b /> }} />
        </div>
      )}
      {warning && (
        <>
          <div style={warnStyle} data-testid="save-warning"><b>{t(`calibration.save.warning.${warning}`)}</b></div>
          <div style={noteStyle}>{t('calibration.save.keepsWarning')}</div>
        </>
      )}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '8px 0 4px' }}>
        <label htmlFor="calibration-name" style={{ minWidth: 48 }}>{t('calibration.save.nameLabel')}</label>
        <Input
          id="calibration-name" ref={input} value={name} disabled={!!created} style={{ flex: 1 }}
          placeholder={t('calibration.save.namePlaceholder')}
          onChange={(e) => { setName(e.target.value); setError(''); }}
          onKeyDown={(e) => { if (e.key === 'Enter') save(); }}
        />
      </div>
      {nameTaken && <div role="alert" style={alertStyle} data-testid="save-name-taken">{t('calibration.save.nameTaken')}</div>}
      {error && <div role="alert" style={alertStyle} data-testid="save-error">{error}</div>}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 10 }}>
        <Button onClick={onClose}>{created ? t('common:close') : t('common:cancel')}</Button>
        <Button variant="primary" disabled={!clean || nameTaken || busy || !!created} onClick={save}>
          {t('calibration.save.save')}
        </Button>
      </div>
      {created && (
        <div style={{ marginTop: 8 }} data-testid="save-done">
          <div role="status" style={{ ...factStyle, color: 'inherit' }} data-testid="save-done-text">{savedText}</div>
          {onApplyToOthers && (
            <div style={{ marginTop: 6 }}>
              <Button small variant="primary" onClick={() => onApplyToOthers(created)}>{t('calibration.save.applyToOthers')}</Button>
            </div>
          )}
        </div>
      )}
    </ModalShell>
  );
}
