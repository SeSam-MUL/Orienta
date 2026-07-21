/**
 * GenerateDictionaryDialog
 *
 * Modal triggered from the Indexing page that generates a dictionary
 * via the GPU pipeline (backend/dictionary_gpu). Reuses the five
 * sub-components from the (now-retired) standalone DictionaryGPUPage:
 * MasterSelector, DetectorSettings, OrientationSettings, GenerateProgress.
 *
 * State ownership:
 *   - Form state (master path, detector, orientation): local to the dialog.
 *   - Generation state (task_id, progress): lifted to the parent via
 *     props, so closing the dialog mid-run does not lose the task.
 *
 * Backend toggle:
 *   - GPU (default, only working option): POST /api/dictionary-gpu/generate.
 *   - CPU: currently no kikuchipy generation endpoint exists in this
 *     project. Rendered as a disabled radio with a tooltip. (When a CPU
 *     endpoint lands, swap the disabled flag and route the request.)
 */

import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { dictionaryGpuApi } from '../../services/api';
import { colors, spacing, Button } from '../../theme/components';
import MasterSelector from '../DictionaryGPU/MasterSelector';
import DetectorSettings from '../DictionaryGPU/DetectorSettings';
import OrientationSettings from '../DictionaryGPU/OrientationSettings';
import GenerateProgress from '../DictionaryGPU/GenerateProgress';

const DEFAULT_DETECTOR = { H: '60', W: '60', pcx: '0.5', pcy: '0.5', pcz: '0.5', sampleTilt: '70' };
const DEFAULT_ORIENTATION = { resolutionDeg: '5.0', normalize: false };

function buildPayload(masterPath, d, o) {
  return {
    master_path: masterPath,
    detector_shape: [parseInt(d.H, 10), parseInt(d.W, 10)],
    pc: [parseFloat(d.pcx), parseFloat(d.pcy), parseFloat(d.pcz)],
    sample_tilt: parseFloat(d.sampleTilt),
    resolution_deg: parseFloat(o.resolutionDeg),
    normalize: !!o.normalize,
  };
}

export default function GenerateDictionaryDialog({
  isOpen,
  onClose,
  // Parent-owned generation state (so closing the dialog doesn't lose the run):
  taskId,
  setTaskId,
  progress,
  setProgress,
  // Called on done so the parent can refresh file discovery:
  onGenerationDone,
  // Master path to pre-fill from (the master selected on the Indexing page,
  // either in the top file field or the Selected Phases list). Optional.
  initialMasterPath = '',
}) {
  const { t } = useTranslation('indexing');
  const [masterPath, setMasterPath] = useState('');

  // Seed the master from the page's selection when the dialog opens, so the
  // user doesn't have to re-pick a master that's already chosen on the page.
  // Only fills an empty field (never clobbers a master the user typed here).
  useEffect(() => {
    if (isOpen && initialMasterPath && !masterPath) setMasterPath(initialMasterPath);
  }, [isOpen, initialMasterPath]);  // eslint-disable-line react-hooks/exhaustive-deps
  const [detector, setDetector] = useState(DEFAULT_DETECTOR);
  const [orientation, setOrientation] = useState(DEFAULT_ORIENTATION);
  const [submitError, setSubmitError] = useState(null);
  const [mode, setMode] = useState('gpu');

  const isRunning = progress && (progress.status === 'running' || progress.status === 'pending');
  const canGenerate = !!masterPath && !isRunning && mode === 'gpu';

  // Notify parent when generation finishes successfully. One-shot latch:
  // the effect would otherwise re-fire on every parent render that keeps
  // progress.status === 'done' (inline-arrow onGenerationDone has fresh
  // identity each render), triggering redundant discovery refreshes. The
  // latch is reset whenever status leaves 'done' so a second generation
  // run in the same dialog session fires onGenerationDone again.
  const doneFiredRef = useRef(false);
  useEffect(() => {
    if (progress?.status === 'done' && onGenerationDone && !doneFiredRef.current) {
      doneFiredRef.current = true;
      onGenerationDone();
    }
    if (progress?.status !== 'done') {
      doneFiredRef.current = false;
    }
  }, [progress?.status, onGenerationDone]);

  // Esc-to-close, but only when no run is in flight (so the user can't
  // accidentally drop their generation by tapping Esc).
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e) => {
      if (e.key === 'Escape' && !isRunning) onClose?.();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, isRunning, onClose]);

  if (!isOpen) return null;

  const handleGenerate = async () => {
    setSubmitError(null);
    const payload = buildPayload(masterPath, detector, orientation);
    const [h, w] = payload.detector_shape;
    if (!Number.isFinite(h) || !Number.isFinite(w)) {
      setSubmitError(t('genDictDialog.errDetectorInt'));
      return;
    }
    if (!Number.isFinite(payload.resolution_deg) || payload.resolution_deg <= 0) {
      setSubmitError(t('genDictDialog.errResolutionPositive'));
      return;
    }
    try {
      const r = await dictionaryGpuApi.generate(payload);
      const newTaskId = r.data?.task_id;
      if (!newTaskId) { setSubmitError(t('genDictDialog.errNoTaskId')); return; }
      setProgress({ status: 'pending', progress: 0, message: t('genDictDialog.submitting') });
      setTaskId(newTaskId);
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || String(e);
      setSubmitError(t('genDictDialog.errGenerateFailed', { detail }));
    }
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
    }}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="gen-dict-title"
        style={{
          background: colors.bg, border: `1px solid ${colors.border}`,
          borderRadius: 6, padding: 16, width: 'min(620px, 92vw)',
          maxHeight: '90vh', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12,
        }}
      >
        <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h2 id="gen-dict-title" style={{ fontSize: '14pt', margin: 0, color: colors.accent }}>{t('genDictDialog.title')}</h2>
          <button onClick={onClose} title={t('hoverTips.genDictClose')} aria-label={t('hoverTips.genDictClose')} style={{ background: 'transparent', border: 'none', color: colors.textSecondary, fontSize: '14pt', cursor: 'pointer' }}>×</button>
        </header>

        {/* Backend toggle */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ color: colors.text }} title={t('hoverTips.genDictBackendGpu')}>
            <input type="radio" checked={mode === 'gpu'} onChange={() => setMode('gpu')} /> {t('genDictDialog.backendGpu')}
          </label>
          <label
            aria-disabled="true"
            title={t('genDictDialog.backendCpuTip')}
            style={{ color: colors.textSecondary, opacity: 0.5, cursor: 'not-allowed' }}
          >
            <input type="radio" disabled readOnly checked={false} /> {t('genDictDialog.backendCpu')}
          </label>
        </div>

        <MasterSelector masterPath={masterPath} onChange={setMasterPath} disabled={isRunning} />
        <DetectorSettings detector={detector} onChange={setDetector} disabled={isRunning} />
        <OrientationSettings orientation={orientation} onChange={setOrientation} disabled={isRunning} />

        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing }}>
          <Button
            variant="primary"
            onClick={handleGenerate}
            disabled={!canGenerate}
            title={!masterPath ? t('genDictDialog.pickMasterFirst') : isRunning ? t('genDictDialog.inProgress') : t('genDictDialog.startTip')}
          >
            {isRunning ? t('genDictDialog.generating') : t('genDictDialog.generate')}
          </Button>
          {taskId && (
            <span style={{ fontSize: '8pt', color: colors.textSecondary, fontFamily: 'monospace' }}>
              {t('genDictDialog.task', { id: taskId })}
            </span>
          )}
        </div>

        {submitError && (
          <div role="alert" style={{
            fontSize: '9pt', color: colors.red, padding: '6px 10px',
            border: `1px solid ${colors.red}`, borderRadius: 4, background: 'rgba(255,85,85,0.08)',
          }}>{submitError}</div>
        )}

        <GenerateProgress progress={progress} />
      </div>
    </div>
  );
}
