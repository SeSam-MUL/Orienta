// Stored results that were computed with an older version of the calibration in use. They are not changed;
// this box only says so and offers the way to the results list. Results computed at the current version are not listed.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pcCalibrationsApi } from '../../../services/api';
import { Button, colors } from '../../../theme/components';
import { noteStyle, factStyle } from './CalibrationFacts';
import { olderResults } from './calibrationText';

/**
 * @param {string}   calibrationId the calibration the open file's PC is
 * @param {string}   refreshKey    changes when the calibration or its version changed: the list is read again
 * @param {Function} onGoToIndexing  null leaves the button out
 */
export default function OlderResults({ calibrationId, refreshKey, onGoToIndexing = null }) {
  const { t } = useTranslation('pcrefinement');
  // the list belongs to the calibration and version it was read for; another one shows nothing until its own is read
  const [read, setRead] = useState({ key: '', older: [] });
  const key = `${calibrationId}|${refreshKey}`;
  useEffect(() => {
    if (!calibrationId) return undefined;
    let live = true;
    pcCalibrationsApi.results(calibrationId).then((res) => {
      if (live) setRead({ key, older: olderResults(res.data?.results) });
    }).catch(() => {});       // information only: a failed read shows nothing
    return () => { live = false; };
  }, [calibrationId, refreshKey, key]);
  const older = read.key === key ? read.older : [];

  if (older.length === 0) return null;
  const versions = [...new Set(older.map((r) => r.version))];
  const latest = older[0].latest;
  return (
    <div
      data-testid="older-results"
      style={{ border: `1px solid ${colors.border}`, background: colors.bg, borderRadius: 4, padding: '8px 10px', fontSize: '9pt', color: colors.text }}
    >
      <div style={factStyle} data-testid="older-results-text">
        {versions.length === 1
          ? t('older.text', { count: older.length, version: versions[0] })
          : t('older.textMany', { count: older.length, latest })}
      </div>
      <div style={noteStyle}>{older.map((r) => `${r.source_file || r.result_id} (v${r.version})`).join(', ')}</div>
      {onGoToIndexing && (
        <div style={{ marginTop: 4 }}>
          <Button small onClick={onGoToIndexing}>{t('older.goToIndexing')}</Button>
        </div>
      )}
    </div>
  );
}
