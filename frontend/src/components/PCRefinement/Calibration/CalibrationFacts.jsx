// What the offer and the "Take PC from..." window both say about one saved calibration: its name with the
// badge (fits / does not fit / not stated), the version line, the PC, the fit contrast and the warning it was
// saved with; and the table of the settings of the file against the calibration's.
import React from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';
import { formatPc } from '../pcFormat';
import {
  fitState, versionMeta, savedWarning, savedContrast, settingRows, settingNameKey, sameCount,
} from './calibrationText';

export const noteStyle = { fontSize: '9pt', color: colors.textSecondary, margin: '2px 0', wordBreak: 'break-word' };
export const factStyle = { fontSize: '9pt', color: colors.text, margin: '2px 0', wordBreak: 'break-word' };
export const alertStyle = { fontSize: '9pt', color: colors.red, margin: '2px 0', wordBreak: 'break-word' };
export const warnStyle = { fontSize: '9pt', color: colors.orange, margin: '2px 0', wordBreak: 'break-word' };

const BADGE_COLOR = { fits: colors.green, no: colors.red, unstated: colors.orange };

/** The small label after the name: "fits", "does not fit" or "not stated". */
export function FitBadge({ state }) {
  const { t } = useTranslation('pcrefinement');
  return (
    <span
      data-testid="calibration-badge" data-state={state}
      style={{
        fontSize: '8.5pt', color: BADGE_COLOR[state], border: `1px solid ${BADGE_COLOR[state]}`,
        borderRadius: 10, padding: '0 7px', marginLeft: 6, whiteSpace: 'nowrap',
      }}
    >
      {t(`calibration.badge.${state}`)}
    </span>
  );
}

/** Name, badge, version line, PC, fit contrast and saved warning of a calibration answer. */
export function CalibrationFacts({ item }) {
  const { t } = useTranslation('pcrefinement');
  const meta = versionMeta(item);
  const contrast = savedContrast(item.quality);
  const warning = savedWarning(item.quality);
  return (
    <div data-testid="calibration-facts">
      <div style={factStyle}><b>{item.name}</b><FitBadge state={fitState(item)} /></div>
      <div style={noteStyle} data-testid="calibration-meta">{t(`calibration.meta.${meta.key}`, meta.params)}</div>
      <div style={noteStyle} data-testid="calibration-pc">{t('calibration.pcLine', { pc: formatPc(item.pc) })}</div>
      {contrast != null && (
        <div style={factStyle} data-testid="calibration-contrast">
          <Trans t={t} i18nKey="result.contrast" values={{ value: contrast }} components={{ b: <b /> }} />
        </div>
      )}
      {warning && (
        <div style={warnStyle} data-testid="calibration-saved-warning"><b>{t(`calibration.saved.${warning}`)}</b></div>
      )}
    </div>
  );
}

const cell = { padding: '2px 4px', borderBottom: `1px solid ${colors.border}`, textAlign: 'left', overflowWrap: 'anywhere' };
const CHECK_COLOR = { same: colors.green, different: colors.red, notStated: colors.orange };

/** The settings of the file against the calibration's, and the sentence about the instrument under it. */
export function SettingsTable({ offer }) {
  const { t } = useTranslation('pcrefinement');
  const rows = settingRows(offer);
  const notStated = t('calibration.settings.notStated');
  return (
    <div data-testid="calibration-settings">
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '9pt', margin: '4px 0', color: colors.text }}>
        <thead>
          <tr style={{ color: colors.textSecondary }}>
            <th style={cell}>{t('calibration.settings.setting')}</th>
            <th style={cell}>{t('calibration.settings.thisFile')}</th>
            <th style={cell}>{t('calibration.settings.calibration')}</th>
            <th style={cell}>{t('calibration.settings.check')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-row={r.key} data-check={r.check}>
              <td style={cell}>{t(settingNameKey(r.key))}</td>
              <td style={cell}>{r.file ?? notStated}</td>
              <td style={cell}>{r.calibration ?? notStated}</td>
              <td style={{ ...cell, color: CHECK_COLOR[r.check], fontWeight: r.check === 'different' ? 600 : 400 }}>
                {t(`calibration.settings.${r.check}`)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={noteStyle} data-testid="calibration-instrument-note">{t('calibration.settings.instrumentNote')}</div>
    </div>
  );
}

/** "All 6 settings same" with the table folded away behind "Show the settings". */
export function SettingsFolded({ offer }) {
  const { t } = useTranslation('pcrefinement');
  return (
    <>
      <div style={factStyle} data-testid="calibration-all-same">{t('calibration.settings.allSame', { n: sameCount(offer) })}</div>
      <details style={{ margin: '4px 0', fontSize: '9pt', color: colors.textSecondary }}>
        <summary style={{ cursor: 'pointer', color: colors.accent }}>{t('calibration.settings.show')}</summary>
        <SettingsTable offer={offer} />
      </details>
    </>
  );
}
