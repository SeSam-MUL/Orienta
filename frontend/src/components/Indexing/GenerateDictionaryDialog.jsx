/**
 * GenerateDictionaryDialog
 *
 * Modal triggered from the Indexing page that generates a dictionary
 * via the GPU pipeline (backend/dictionary_gpu). Reuses the sub-components
 * from the (now-retired) standalone DictionaryGPUPage: DetectorSettings,
 * OrientationSettings, GenerateProgress.
 *
 * Two things this dialog must get right, because both were silently wrong
 * before and made the whole feature useless:
 *
 *   1. WHICH PHASE. A single free-text master path seeded from "the first
 *      .h5 in the phase list" always resolved to the first phase, so a
 *      second phase (Si next to Al) could never get a dictionary. There is
 *      now an explicit phase picker, and the per-phase button on each phase
 *      card preselects its own entry.
 *   2. WHICH DETECTOR. The defaults were a hardcoded 60x60 at PC
 *      (0.5, 0.5, 0.5) — never the loaded dataset. A dictionary generated
 *      for the wrong geometry correlates against nothing. Detector shape,
 *      PC, energy and angular resolution are now seeded from the page and
 *      the loaded dataset, and the header states what they came from.
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

const FALLBACK_DETECTOR = { H: '60', W: '60', pcx: '0.5', pcy: '0.5', pcz: '0.5', sampleTilt: '70' };
const FALLBACK_RESOLUTION = '5.0';
const FALLBACK_ENERGY = '20';

/** Detector form values seeded from the loaded dataset + the page's PC. */
function seedDetector({ detectorShape, pc, sampleTilt }) {
  const d = { ...FALLBACK_DETECTOR };
  if (Array.isArray(detectorShape) && detectorShape.length >= 2) {
    d.H = String(detectorShape[0]);
    d.W = String(detectorShape[1]);
  }
  if (Array.isArray(pc) && pc.length >= 3) {
    d.pcx = String(pc[0]);
    d.pcy = String(pc[1]);
    d.pcz = String(pc[2]);
  }
  if (Number.isFinite(sampleTilt)) d.sampleTilt = String(sampleTilt);
  return d;
}

function buildPayload(masterPath, d, o, energyKv, mode, geom) {
  return {
    master_path: masterPath,
    detector_shape: [parseInt(d.H, 10), parseInt(d.W, 10)],
    pc: [parseFloat(d.pcx), parseFloat(d.pcy), parseFloat(d.pcz)],
    sample_tilt: parseFloat(d.sampleTilt),
    // Camera geometry of the loaded dataset. Leaving these at 0 while the real
    // detector is tilted produces a dictionary for a detector that does not
    // exist — measured NCC 0.017 at 3.44 deg for the SAME orientation.
    detector_tilt_deg: parseFloat(geom.detectorTilt) || 0,
    azimuthal_deg: parseFloat(geom.azimuthal) || 0,
    energy_kv: parseFloat(energyKv),
    resolution_deg: parseFloat(o.resolutionDeg),
    normalize: !!o.normalize,
    // Land in Database/Dictionary_Library under the canonical name, so the
    // Selected-Phases card actually finds the result. Without this the file
    // goes to tasks/ where nothing looks for it.
    save_to_library: true,
    backend: mode === 'cpu' ? 'cpu' : 'gpu',
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
  // Master path to pre-fill from. Optional.
  initialMasterPath = '',
  // One entry per selected phase — see dictionaryTargets().
  phaseTargets = [],
  // Geometry of the loaded dataset / page settings.
  detectorShape = null,
  pc = null,
  sampleTilt = 70,
  detectorTilt = 0,
  azimuthal = 0,
  energyKv = FALLBACK_ENERGY,
  resolutionDeg = FALLBACK_RESOLUTION,
  datasetName = '',
  // Lets the page remember what was actually used, so the kV-mismatch check
  // and the next dialog open follow the user instead of a fixed default.
  onSettingsUsed,
}) {
  const { t } = useTranslation('indexing');
  const [masterPath, setMasterPath] = useState('');
  const [detector, setDetector] = useState(FALLBACK_DETECTOR);
  const [orientation, setOrientation] = useState({ resolutionDeg: FALLBACK_RESOLUTION, normalize: false });
  const [energy, setEnergy] = useState(FALLBACK_ENERGY);
  const [camTilt, setCamTilt] = useState('0');
  const [submitError, setSubmitError] = useState(null);
  const [mode, setMode] = useState('gpu');

  // Seed the whole form from the page each time the dialog opens. Re-seeding
  // on open (rather than once on mount) is what makes the per-phase buttons
  // work: clicking Si's button after Al's must switch the master, and a PC
  // refined between two runs must be picked up.
  useEffect(() => {
    if (!isOpen) return;
    setMasterPath(initialMasterPath || '');
    setDetector(seedDetector({ detectorShape, pc, sampleTilt }));
    setOrientation(o => ({ ...o, resolutionDeg: String(resolutionDeg ?? FALLBACK_RESOLUTION) }));
    setEnergy(String(energyKv ?? FALLBACK_ENERGY));
    setCamTilt(String(detectorTilt ?? 0));
    setSubmitError(null);
  }, [isOpen, initialMasterPath]);  // eslint-disable-line react-hooks/exhaustive-deps

  const isRunning = progress && (progress.status === 'running' || progress.status === 'pending');
  const canGenerate = !!masterPath && !isRunning;

  // Which phase the current master belongs to (drives the picker's value).
  const selectedTarget = phaseTargets.find(p => p.masterPath && p.masterPath === masterPath);

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
    const payload = buildPayload(masterPath, detector, orientation, energy, mode, {
      detectorTilt: camTilt, azimuthal,
    });
    const [h, w] = payload.detector_shape;
    if (!Number.isFinite(h) || !Number.isFinite(w)) {
      setSubmitError(t('genDictDialog.errDetectorInt'));
      return;
    }
    if (!Number.isFinite(payload.resolution_deg) || payload.resolution_deg <= 0) {
      setSubmitError(t('genDictDialog.errResolutionPositive'));
      return;
    }
    if (!Number.isFinite(payload.energy_kv) || payload.energy_kv <= 0) {
      setSubmitError(t('genDictDialog.errEnergyPositive'));
      return;
    }
    onSettingsUsed?.({
      energyKv: payload.energy_kv,
      resolutionDeg: payload.resolution_deg,
    });
    try {
      const r = await dictionaryGpuApi.generate(payload);
      const newTaskId = r.data?.task_id;
      if (!newTaskId) { setSubmitError(t('genDictDialog.errNoTaskId')); return; }
      setProgress({
        status: 'pending', progress: 0,
        message: t('genDictDialog.submitting'),
        output_path: r.data?.output_path || '',
      });
      setTaskId(newTaskId);
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || String(e);
      setSubmitError(t('genDictDialog.errGenerateFailed', { detail }));
    }
  };

  const geometryHint = detectorShape
    ? t('genDictDialog.geometryFrom', {
        dataset: datasetName || t('genDictDialog.loadedDataset'),
        h: detectorShape[0], w: detectorShape[1],
      })
    : t('genDictDialog.geometryUnknown');

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

        {/* Phase picker — the point of the whole dialog: WHICH phase am I
            generating for. Hidden when there is nothing selected to pick from. */}
        {phaseTargets.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <label htmlFor="gen-dict-phase" style={{ color: colors.text, fontSize: '10pt' }}>
              {t('genDictDialog.phaseLabel')}
            </label>
            <select
              id="gen-dict-phase"
              value={selectedTarget ? String(selectedTarget.index) : ''}
              disabled={isRunning}
              title={t('genDictDialog.phaseTip')}
              onChange={(e) => {
                const tgt = phaseTargets.find(p => String(p.index) === e.target.value);
                setMasterPath(tgt?.masterPath || '');
              }}
              style={{
                background: colors.bgSecondary, color: colors.text,
                border: `1px solid ${colors.border}`, borderRadius: 4,
                padding: '3px 6px', fontSize: '10pt', minWidth: 220,
              }}
            >
              <option value="">{t('genDictDialog.phaseCustom')}</option>
              {phaseTargets.map(p => (
                <option key={p.index} value={String(p.index)} disabled={!p.hasMaster}>
                  {p.hasMaster
                    ? `${p.label} — ${p.masterFilename}`
                    : t('genDictDialog.phaseNoMaster', { label: p.label })}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Backend toggle. Both paths write the same file layout — GPU is the
            in-process PyTorch projection, CPU is kikuchipy's get_patterns
            (the routine that built every dictionary already in the library). */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ color: colors.text }} title={t('hoverTips.genDictBackendGpu')}>
            <input
              type="radio" name="gen_dict_backend"
              checked={mode === 'gpu'} disabled={isRunning}
              onChange={() => setMode('gpu')}
            /> {t('genDictDialog.backendGpu')}
          </label>
          <label style={{ color: colors.text }} title={t('genDictDialog.backendCpuTip')}>
            <input
              type="radio" name="gen_dict_backend"
              checked={mode === 'cpu'} disabled={isRunning}
              onChange={() => setMode('cpu')}
            /> {t('genDictDialog.backendCpu')}
          </label>
        </div>
        {mode === 'cpu' && (
          <div style={{ fontSize: '8.5pt', color: colors.textSecondary, marginTop: -6 }}>
            {t('genDictDialog.backendCpuNote')}
          </div>
        )}

        <MasterSelector masterPath={masterPath} onChange={setMasterPath} disabled={isRunning} />

        <div style={{ fontSize: '8.5pt', color: colors.textSecondary, marginTop: -6 }}>
          {geometryHint}
        </div>
        <DetectorSettings detector={detector} onChange={setDetector} disabled={isRunning} />

        {/* Camera tilt. Separate from DetectorSettings' sample tilt, and the
            single most damaging value to get wrong — see buildPayload. */}
        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing, flexWrap: 'wrap' }}>
          <label htmlFor="gen-dict-camtilt" style={{ color: colors.text, fontSize: '10pt' }}>
            {t('genDictDialog.detectorTiltLabel')}
          </label>
          <input
            id="gen-dict-camtilt"
            type="number" min={-30} max={30} step={0.01}
            value={camTilt}
            disabled={isRunning}
            title={t('genDictDialog.detectorTiltTip')}
            onChange={(e) => setCamTilt(e.target.value)}
            style={{
              background: colors.bgSecondary, color: colors.text,
              border: `1px solid ${colors.border}`, borderRadius: 4,
              padding: '3px 6px', width: 80, fontSize: '10pt',
            }}
          />
          <span style={{ color: colors.textSecondary, fontSize: '9pt' }}>°</span>
          {Math.abs(parseFloat(camTilt) || 0) > 0.005 && (
            <span style={{ color: colors.textSecondary, fontSize: '8.5pt' }}>
              {t('genDictDialog.detectorTiltFromFile')}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing }}>
          <label htmlFor="gen-dict-energy" style={{ color: colors.text, fontSize: '10pt' }}>
            {t('genDictDialog.energyLabel')}
          </label>
          <input
            id="gen-dict-energy"
            type="number" min={1} max={40} step={1}
            value={energy}
            disabled={isRunning}
            title={t('genDictDialog.energyTip')}
            onChange={(e) => setEnergy(e.target.value)}
            style={{
              background: colors.bgSecondary, color: colors.text,
              border: `1px solid ${colors.border}`, borderRadius: 4,
              padding: '3px 6px', width: 70, fontSize: '10pt',
            }}
          />
          <span style={{ color: colors.textSecondary, fontSize: '9pt' }}>{t('genDictDialog.kv')}</span>
        </div>

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
