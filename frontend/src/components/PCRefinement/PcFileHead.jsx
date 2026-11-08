// The line at the top of the PC page: which file is open and which PC is in use for it. It stands on
// every state of the page, so a number on the screen is never read as the PC of another file.
import React from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { formatPc } from './pcFormat';

/**
 * @param {string|null}   file      the open file (its dataset name), or null
 * @param {number[]|null} pc        the PC in use (Bruker x, y, z), or null when the file has none
 * @param {boolean}       perPixel  the file uses a per-pixel map and `pc` is its mean: the line says so,
 *                                  so three numbers are not read as the one PC of the whole scan
 * @param {object|null}   origin    `pc_in_use` of /api/pc/status (`kind`, `calibration`): where the PC came from
 * @param {boolean}       showConvention  name the kind of numbers after the PC: "(Bruker)". Every number the page
 *                                  keeps is Bruker's, so a person who types Oxford or EDAX numbers below sees
 *                                  that the head speaks another language than the fields they fill in.
 */

// The i18n key of each kind of origin; "none" has no origin to name.
const ORIGIN_KEYS = {
  file: 'fileHead.originFile', find: 'fileHead.originFind', refine: 'fileHead.originRefine',
  typed: 'fileHead.originTyped', grid: 'fileHead.originGrid', calibration: 'fileHead.originCalibration',
};

export default function PcFileHead({ file, pc, perPixel = false, origin = null, showConvention = false }) {
  const { t } = useTranslation('pcrefinement');
  const text = formatPc(pc);
  const shown = text && perPixel ? t('fileHead.pcMapMean', { pc: text }) : text;
  const key = text ? ORIGIN_KEYS[origin?.kind] : null;
  const calibration = origin?.calibration;
  const named = origin?.kind === 'calibration' && calibration ? { name: calibration.name, version: calibration.version } : null;
  // a calibration whose name is not known is not named: no "from undefined, version undefined"
  const originText = key && (origin.kind !== 'calibration' || named) ? t(key, named || {}) : '';
  return (
    <div
      data-testid="pc-file-head"
      style={{
        fontSize: '9pt', color: colors.textSecondary, padding: '2px 2px 4px',
        wordBreak: 'break-word',
      }}
    >
      {t('fileHead.file')}{' '}
      <b data-testid="pc-file-head-file" style={{ color: colors.text }}>{file || t('fileHead.noFile')}</b>
      {' · '}
      {t('fileHead.pc')}{' '}
      <b data-testid="pc-file-head-pc" style={{ color: colors.text }}>{shown || t('fileHead.none')}</b>
      {showConvention && text && <span data-testid="pc-file-head-convention"> ({t('convention.short.bruker')})</span>}
      {originText && <span data-testid="pc-file-head-origin"> ({originText})</span>}
    </div>
  );
}
