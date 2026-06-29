/**
 * DictionaryGPUPage — stand-alone GPU Dictionary Generator.
 *
 * Single-route page: pick a master pattern, configure detector + orientations,
 * hit Generate, watch progress, see existing dictionaries. No global stores —
 * this tool stands alone so we can iterate on it without coupling to the rest
 * of the app.
 *
 * Lifecycle:
 *   - On mount: fetch existing dictionaries.
 *   - On Generate: POST /api/dictionary-gpu/generate, store task_id.
 *   - While task_id is set and status is running/pending: poll /progress every
 *     500 ms. Stop polling on done/error.
 *   - On done: refresh the list.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { dictionaryGpuApi } from '../../services/api';
import { colors, spacing, Button } from '../../theme/components';
import MasterSelector from './MasterSelector';
import DetectorSettings from './DetectorSettings';
import OrientationSettings from './OrientationSettings';
import GenerateProgress from './GenerateProgress';
import DictionaryList from './DictionaryList';

const POLL_INTERVAL_MS = 500;

const DEFAULT_DETECTOR = {
  H: '60',
  W: '60',
  pcx: '0.5',
  pcy: '0.5',
  pcz: '0.5',
  sampleTilt: '70',
};

const DEFAULT_ORIENTATION = {
  resolutionDeg: '5.0',
  normalize: false,
};

function buildPayload(masterPath, d, o) {
  // Flat shape that matches backend GenerateRequest Pydantic model.
  return {
    master_path: masterPath,
    detector_shape: [parseInt(d.H, 10), parseInt(d.W, 10)],
    pc: [parseFloat(d.pcx), parseFloat(d.pcy), parseFloat(d.pcz)],
    sample_tilt: parseFloat(d.sampleTilt),
    resolution_deg: parseFloat(o.resolutionDeg),
    normalize: !!o.normalize,
  };
}

export default function DictionaryGPUPage() {
  const { t } = useTranslation('dictionarygpu');

  // --- Form state ----------------------------------------------------------
  const [masterPath, setMasterPath] = useState('');
  const [detector, setDetector] = useState(DEFAULT_DETECTOR);
  const [orientation, setOrientation] = useState(DEFAULT_ORIENTATION);

  // --- Generation state ----------------------------------------------------
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  // --- List state ----------------------------------------------------------
  const [dictionaries, setDictionaries] = useState([]);
  const [listLoading, setListLoading] = useState(false);

  const pollTimerRef = useRef(null);

  // -- List loader ---------------------------------------------------------
  const loadList = useCallback(async () => {
    setListLoading(true);
    try {
      const r = await dictionaryGpuApi.list();
      setDictionaries(r.data?.dictionaries ?? []);
    } catch (e) {
      // Non-fatal: show empty list rather than crash the page.
      setDictionaries([]);
      console.error('Failed to load dictionary list', e);
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  // -- Polling -------------------------------------------------------------
  useEffect(() => {
    if (!taskId) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const r = await dictionaryGpuApi.progress(taskId);
        if (cancelled) return;
        setProgress(r.data);

        const status = r.data?.status;
        if (status === 'done' || status === 'error') {
          // Stop polling. If done, refresh the list so the new dictionary shows up.
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          if (status === 'done') loadList();
        }
      } catch (e) {
        if (cancelled) return;
        // Treat transient failures as non-fatal: keep polling, surface message.
        setProgress((prev) => ({
          ...(prev || {}),
          status: 'error',
          error: t('errors.progressPollFailed', { message: e?.message || e }),
        }));
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
      }
    };

    // Fire once immediately, then on an interval.
    poll();
    pollTimerRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [taskId, loadList, t]);

  // -- Generate ------------------------------------------------------------
  const isRunning = progress && (progress.status === 'running' || progress.status === 'pending');
  const canGenerate = !!masterPath && !isRunning;

  const handleGenerate = async () => {
    setSubmitError(null);
    const payload = buildPayload(masterPath, detector, orientation);

    // Basic client-side guards — backend re-validates.
    const [h, w] = payload.detector_shape;
    if (!Number.isFinite(h) || !Number.isFinite(w)) {
      setSubmitError(t('errors.detectorIntegers'));
      return;
    }
    if (!Number.isFinite(payload.resolution_deg) || payload.resolution_deg <= 0) {
      setSubmitError(t('errors.resolutionPositive'));
      return;
    }

    try {
      const r = await dictionaryGpuApi.generate(payload);
      const newTaskId = r.data?.task_id;
      if (!newTaskId) {
        setSubmitError(t('errors.noTaskId'));
        return;
      }
      setProgress({ status: 'pending', progress: 0, message: t('progress.submitting') });
      setTaskId(newTaskId);
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || String(e);
      setSubmitError(t('errors.generateFailed', { detail }));
    }
  };

  // -- Delete --------------------------------------------------------------
  const handleDelete = async (name) => {
    try {
      await dictionaryGpuApi.delete(name);
      await loadList();
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || String(e);
      window.alert(t('errors.deleteFailed', { detail }));
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        height: '100%',
        overflow: 'auto',
        padding: '0 4px',
      }}
    >
      <header style={{ marginBottom: 4 }}>
        <h1
          style={{
            fontSize: '18px',
            fontWeight: 600,
            color: colors.accent,
            margin: '0 0 6px',
          }}
        >
          {t('page.title')}
        </h1>
        <p style={{ color: colors.textSecondary, fontSize: '12px', margin: 0 }}>
          {t('page.description')}
        </p>
      </header>

      <MasterSelector
        masterPath={masterPath}
        onChange={setMasterPath}
        disabled={isRunning}
      />

      <DetectorSettings
        detector={detector}
        onChange={setDetector}
        disabled={isRunning}
      />

      <OrientationSettings
        orientation={orientation}
        onChange={setOrientation}
        disabled={isRunning}
      />

      <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing }}>
        <Button
          variant="primary"
          onClick={handleGenerate}
          disabled={!canGenerate}
          title={!masterPath ? t('generate.tooltipPickMaster') : isRunning ? t('generate.tooltipInProgress') : t('generate.tooltip')}
        >
          {isRunning ? t('generate.generating') : t('generate.button')}
        </Button>
        {taskId && (
          <span style={{ fontSize: '8pt', color: colors.textSecondary, fontFamily: 'monospace' }}>
            {t('generate.taskLabel', { taskId })}
          </span>
        )}
      </div>

      {submitError && (
        <div
          role="alert"
          style={{
            fontSize: '9pt',
            color: colors.red,
            padding: '6px 10px',
            border: `1px solid ${colors.red}`,
            borderRadius: 4,
            background: 'rgba(255,85,85,0.08)',
          }}
        >
          {submitError}
        </div>
      )}

      <GenerateProgress progress={progress} />

      <DictionaryList
        dictionaries={dictionaries}
        loading={listLoading}
        onRefresh={loadList}
        onDelete={handleDelete}
      />
    </div>
  );
}
