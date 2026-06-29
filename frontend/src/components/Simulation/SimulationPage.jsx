import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { simApi } from '../../services/api';
import CrystalPickerModal from './CrystalPickerModal';
import SimulateMissingDialog from './SimulateMissingDialog';
import useDataStore from '../../stores/useDataStore';
import {
  colors, spacing, alpha,
  Button, Input, NumberInput, Select,
  Tabs, TabPanel,
  GroupBox, CollapsibleGroup, FormRow,
  Label, ProgressBar, Separator, useConfirm, ConfirmDialog,
  usePrompt, PromptDialog,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const DEFAULT_PARAMS = {
  crystalFile:        '',
  voltage:            20.0,
  sampleTilt:         70.0,
  totalElectrons:     500000000,
  resolution:         501,
  dMin:               0.05,
  outputType:         'both',
  computeMode:        'auto',
  masterNpx:          500,
  nthreads:           10,
  globalworkgrpsz:    150,
  // Advanced — Monte Carlo
  omega:              0.0,
  depthmax:           100.0,
  depthstep:          1.0,
  // Advanced — Master Pattern
  combinesites:       false,
  useEnergyWeighting: false,
  doLegendre:         false,
  esel:               -1,
  uniform:            false,
};

const OUTPUT_TYPE_OPTIONS = [
  { value: 'sht_only',    label: 'SHT only (EMSphinx)' },
  { value: 'master_only', label: 'Master Pattern (Kikuchipy DI)' },
  { value: 'both',        label: 'Both (SHT + Master)' },
];

const COMPUTE_MODE_OPTIONS = [
  { value: 'auto', label: 'Auto (GPU if available)' },
  { value: 'gpu',  label: 'GPU only (OpenCL)' },
  { value: 'cpu',  label: 'CPU only (OpenMP)' },
];

// Single source of truth for "this job is not finished — keep polling/counting it".
// The backend uses 'pending' for queued batch jobs and 'running' while a job is
// active; the frontend also seeds new jobs as 'queued'. Leaving ANY of these out of
// a poll/count filter caused the Queue to freeze at "1 done · N pending" — once the
// snapshot held only completed + 'pending' jobs, polling stopped and a fully-
// completing backend batch looked dead-ended. Centralised so the filters can never
// drift apart again.
export const isActiveJobStatus = (status) =>
  status === 'running' || status === 'queued' || status === 'pending';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function statusColor(status) {
  return {
    running:   colors.cyan,
    queued:    colors.orange,
    completed: colors.green,
    failed:    colors.red,
    stopped:   colors.textSecondary,
  }[status] ?? colors.textSecondary;
}

function StatusBadge({ status }) {
  const { t } = useTranslation('simulation');
  const col = statusColor(status);
  const labelKey = {
    running:   'statusBadge.running',
    queued:    'statusBadge.queued',
    completed: 'statusBadge.completed',
    failed:    'statusBadge.failed',
    stopped:   'statusBadge.stopped',
  }[status];
  const label = labelKey ? t(labelKey) : status;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: '9pt',
      fontWeight: 600,
      background: alpha(col, 13),
      color: col,
      border: `1px solid ${alpha(col, 27)}`,
    }}>
      {status === 'running' && (
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: col, animation: 'pulse 1.5s ease-in-out infinite', flexShrink: 0 }} />
      )}
      {label}
    </span>
  );
}

function CheckRow({ label, checked, onChange, title }) {
  return (
    <label
      title={title}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        cursor: 'pointer',
        userSelect: 'none',
        fontSize: '10pt',
        color: colors.text,
        marginBottom: spacing.innerSpacing,
        padding: '2px 4px',
        borderRadius: 3,
        transition: 'background 0.12s',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = alpha(colors.border, 13); }}
      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        style={{ accentColor: colors.purple, width: 14, height: 14, cursor: 'pointer' }}
      />
      {label}
    </label>
  );
}

function outputPreviewText(params, t) {
  if (!params.crystalFile) return null;
  const base = params.crystalFile.split(/[\\/]/).pop()?.replace(/\.xtal$/i, '') ?? '';
  const tag = `{${params.voltage}kV ${params.sampleTilt}deg}`;
  if (params.outputType === 'sht_only')    return `${base} ${tag}.sht`;
  if (params.outputType === 'master_only') return `${base} ${tag}.h5${t('outputType.previewMaster')}`;
  return `${base} ${tag}.sht  +  ${base} ${tag}.h5${t('outputType.previewMasterShort')}`;
}

function estimatedTime(params, t) {
  const baseHours = 0.5;
  const resFactor = Math.pow(Number(params.resolution || 501) / 501, 2);
  const elFactor  = Number(params.totalElectrons || 5e8) / 5e8;
  const hours = baseHours * resFactor * elFactor;
  return hours < 1
    ? t('estimates.timeMinutes', { minutes: Math.round(hours * 60) })
    : t('estimates.timeHours', { hours: hours.toFixed(1) });
}

// ---------------------------------------------------------------------------
// Server Mode Panel (inline settings — replaces PyQt5 ServerModeDialog)
// ---------------------------------------------------------------------------
function ServerModePanel() {
  const { t } = useTranslation('simulation');
  const [cfg, setCfg] = useState({ enabled: false, database_root: '', offline_mode: false, discovered: {} });
  const [testResult, setTestResult] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    simApi.serverConfig()
      .then(r => setCfg(r.data || {}))
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await simApi.updateServerConfig({
        enabled: cfg.enabled,
        database_root: cfg.database_root,
        offline_mode: cfg.offline_mode,
      });
      const r = await simApi.serverConfig();
      setCfg(r.data || {});
    } catch { /* ignore */ }
    setSaving(false);
  };

  const handleTest = async () => {
    setTestResult(null);
    try {
      const r = await simApi.testServerConnection();
      setTestResult(r.data);
    } catch (err) {
      setTestResult({ connected: false, message: err.message });
    }
  };

  const handleBrowse = async () => {
    if (window.electronAPI?.openDirectory) {
      const dir = await window.electronAPI.openDirectory();
      if (dir) setCfg(prev => ({ ...prev, database_root: dir }));
    }
  };

  const disc = cfg.discovered || {};
  const statusColor = !cfg.enabled ? colors.textSecondary
    : testResult?.connected ? colors.green
    : testResult ? colors.red
    : colors.yellow;
  const statusText = !cfg.enabled ? t('serverMode.statusDisabled')
    : testResult?.connected ? t('serverMode.statusConnected')
    : testResult ? testResult.message || t('serverMode.statusOffline')
    : cfg.database_root ? t('serverMode.statusNotTested') : t('serverMode.statusSetupNeeded');

  return (
    <CollapsibleGroup title={t('serverMode.title')} defaultOpen={false}>
      <FormRow label={t('serverMode.enableLabel')}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={!!cfg.enabled}
            onChange={e => setCfg(prev => ({ ...prev, enabled: e.target.checked }))}
            title={t('hoverTips.serverEnable')} />
          <span>{t('serverMode.enableText')}</span>
        </label>
      </FormRow>

      {cfg.enabled && (
        <>
          <FormRow label={t('serverMode.statusLabel')}>
            <Label style={{ color: statusColor, fontFamily: 'monospace' }}>● {statusText}</Label>
          </FormRow>

          <FormRow label={t('serverMode.databaseRootLabel')}>
            <div style={{ display: 'flex', gap: 6, flex: 1 }}>
              <Input value={cfg.database_root || ''} style={{ flex: 1 }}
                onChange={e => setCfg(prev => ({ ...prev, database_root: e.target.value }))}
                placeholder={t('serverMode.databaseRootPlaceholder')}
                title={t('hoverTips.databaseRoot')} />
              <Button variant="default" onClick={handleBrowse} title={t('hoverTips.serverBrowse')}>{t('serverMode.browse')}</Button>
            </div>
          </FormRow>

          {Object.keys(disc).length > 0 && (
            <FormRow label={t('serverMode.discoveredLabel')}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: '9pt' }}>
                {Object.entries(disc).map(([key, val]) => (
                  <span key={key} style={{ color: val.exists ? colors.green : colors.textSecondary }}>
                    {val.exists ? '✓' : '✗'} {key.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </FormRow>
          )}

          <FormRow label={t('serverMode.offlineModeLabel')}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox" checked={!!cfg.offline_mode}
                onChange={e => setCfg(prev => ({ ...prev, offline_mode: e.target.checked }))}
                title={t('hoverTips.offlineMode')} />
              <span>{t('serverMode.offlineModeText')}</span>
            </label>
          </FormRow>

          <div style={{ display: 'flex', gap: spacing.buttonSpacing, marginTop: 6 }}>
            <Button variant="default" onClick={handleTest} title={t('hoverTips.testConnection')}>{t('serverMode.testConnection')}</Button>
            <Button variant="primary" onClick={handleSave} disabled={saving} title={t('hoverTips.saveServerConfig')}>
              {saving ? t('serverMode.saving') : t('common:save')}
            </Button>
          </div>
        </>
      )}
    </CollapsibleGroup>
  );
}

// ---------------------------------------------------------------------------
// Configure Tab
// ---------------------------------------------------------------------------
function ConfigureTab({ onStarted, isActive = false, onShowQueue, engine = 'ours' }) {
  // "Ours" routes to our self-contained GPU/CPU pipeline (startGpu / batchStartGpu /
  // run_gpu_simulation). "EMsoft" routes to the EMsoft WSL pipeline. There is no
  // per-phase EMsoft-vs-ours auto-routing any more — the manual switch decides.
  const { t } = useTranslation(['simulation', 'common']);
  const useOurs = engine !== 'emsoft';
  const [params, setParams]           = useState(DEFAULT_PARAMS);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [statusText, setStatusText]   = useState(t('status.ready'));
  const [missingDialog, setMissingDialog] = useState(null);  // missing-phases list for the Simulate-All-Missing pre-launch dialog
  const [simDone, setSimDone]         = useState(false);
  const [oclInfo, setOclInfo]         = useState(null);
  const [sysStatus, setSysStatus]     = useState(null);
  const [sysLoading, setSysLoading]   = useState(false);
  const [pickerOpen, setPickerOpen]   = useState(false);

  const set = useCallback((key, val) => setParams(prev => ({ ...prev, [key]: val })), []);

  // Pre-fill from EBSD metadata if available
  const { detector, beamEnergy } = useDataStore();
  useEffect(() => {
    if (detector?.sample_tilt != null) set('sampleTilt', Math.round(detector.sample_tilt * 100) / 100);
    if (beamEnergy != null) set('voltage', beamEnergy);
  }, [detector, beamEnergy, set]);

  // Pick up pending XTAL path from Crystal Database cross-navigation
  useEffect(() => {
    const pending = useDataStore.getState().pendingSimXtal;
    if (pending && isActive) {
      setParams(p => ({ ...p, crystalFile: pending }));
      useDataStore.getState().setPendingSimXtal(null);
    }
  }, [isActive]);

  const showMasterNpx = params.outputType === 'master_only' || params.outputType === 'both';

  // Load simulation defaults from backend config + system status when page becomes active
  useEffect(() => {
    if (!isActive) return;
    // 1) Load INI config defaults (totnum_el, numsx, dmin, npx, omega, depthmax, depthstep, globalworkgrpsz)
    simApi.getConfig()
      .then(resp => {
        const cfg = resp.data?.config ?? {};
        const mc = cfg['DefaultSimulationParametersEMMCOpenCL'] ?? {};
        const sht = cfg['DefaultSimulationParametersEMEBSDmasterSHT'] ?? {};
        const master = cfg['DefaultSimulationParametersEMEBSDmaster'] ?? {};
        if (mc.totnum_el)       set('totalElectrons', parseInt(mc.totnum_el, 10));
        if (mc.numsx)           set('resolution', parseInt(mc.numsx, 10));
        if (mc.omega != null)   set('omega', parseFloat(mc.omega));
        if (mc.globalworkgrpsz) set('globalworkgrpsz', parseInt(mc.globalworkgrpsz, 10));
        if (mc.depthmax)        set('depthmax', parseFloat(mc.depthmax));
        if (mc.depthstep)       set('depthstep', parseFloat(mc.depthstep));
        if (sht.dmin)           set('dMin', parseFloat(sht.dmin));
        if (master.npx)         set('masterNpx', parseInt(master.npx, 10));
      })
      .catch(() => { /* backend not reachable — keep hardcoded defaults */ });

    // 2) OpenCL / system status (nthreads, globalworkgrpsz from detection)
    simApi.systemStatus()
      .then(resp => {
        const d = resp.data ?? {};
        const gpuText =
          d.gpu_name         ? d.gpu_name :
          d.opencl_device    ? d.opencl_device :
          d.has_gpu          ? t('ocl.gpuDetected', { gb: d.gpu_memory_gb?.toFixed(1) ?? '?' }) :
          d.opencl_available ? t('ocl.openclAvailable') :
          t('ocl.noGpu');
        const ok = !!(d.has_gpu || d.opencl_available);
        setOclInfo({ text: gpuText, ok, raw: d });
        if (d.cpu_count) {
          set('nthreads', Math.max(1, Math.min(32, (d.cpu_count - 2) || 4)));
        }
        // globalworkgrpsz from hardware detection overrides INI value
        const rec = d.recommended_settings ?? {};
        if (rec.globalworkgrpsz) {
          set('globalworkgrpsz', rec.globalworkgrpsz);
        }
      })
      .catch(() => {
        setOclInfo({ text: t('ocl.backendUnreachable'), ok: false });
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  // Our pipeline needs FAR fewer electrons than EMsoft's 500M — the master
  // pattern uses only the normalised depth/energy profile, which converges by
  // ~10-25M (the backend also caps our MC path at 50M). Swap the electron-count
  // default when the engine changes so the form shows a sensible value. A ref
  // guards the initial mount so we don't clobber the INI-loaded EMsoft default.
  const prevEngineRef = useRef(engine);
  useEffect(() => {
    if (prevEngineRef.current === engine) return;
    prevEngineRef.current = engine;
    set('totalElectrons', engine === 'emsoft' ? 500000000 : 25000000);
  }, [engine, set]);

  const handleBrowse = async () => {
    if (window.electronAPI?.openFile) {
      const selected = await window.electronAPI.openFile({
        filters: [
          { name: 'XTAL Files', extensions: ['xtal'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (selected) set('crystalFile', selected);
    }
  };

  const handleCheckSystem = async () => {
    setSysLoading(true);
    setSysStatus(null);
    try {
      const resp = await simApi.systemStatus();
      setSysStatus(resp.data);
    } catch (err) {
      setSysStatus({ error: err.response?.data?.detail || err.message });
    } finally {
      setSysLoading(false);
    }
  };

  const handleSimulateAllMissing = async () => {
    setStatusText(t('status.scanningMissing'));
    setLoading(true);
    try {
      const resp = await simApi.scanMissing(params.outputType, Number(params.voltage));
      const missing = resp.data?.missing || [];
      if (missing.length === 0) {
        setStatusText(t('status.allHaveResults'));
        setLoading(false);
        return;
      }
      setLoading(false);
      // Open the pre-launch settings dialog (per-phase adaptive dmin) instead
      // of blindly launching all phases with one global dmin.
      setMissingDialog(missing);
    } catch (err) {
      setError(t('errors.batchScanFailed', { message: err.response?.data?.detail || err.message }));
      setStatusText(t('status.errorScanningMissing'));
      setLoading(false);
    }
  };

  // Shared batch payload (everything except xtal_paths/dmin_overrides). Extracted
  // so single + batch dispatch send identical params regardless of pipeline.
  const commonBatchParams = useCallback(() => ({
    ekev: Number(params.voltage), sig: Number(params.sampleTilt),
    totnum_el: Number(params.totalElectrons), numsx: Number(params.resolution),
    output_type: params.outputType, compute_mode: params.computeMode,
    dmin: Number(params.dMin), npx: Number(params.masterNpx),
    nthreads: Number(params.nthreads), omega: Number(params.omega),
    depthmax: Number(params.depthmax), depthstep: Number(params.depthstep),
    combinesites: params.combinesites, globalworkgrpsz: Number(params.globalworkgrpsz || 150),
    use_energy_weighting: params.useEnergyWeighting,
    do_legendre: params.doLegendre, esel: Number(params.esel), uniform: params.uniform,
  }), [params]);

  // Fire one batch endpoint for a group of phases. Each item is
  // {stem, xtal_path, dmin}. Returns the job_count (0 if empty). The endpoint is
  // chosen by the manual engine switch: "Ours" -> batchStartGpu (our pipeline),
  // "EMsoft" -> batchStart. There is no per-phase partitioning any more.
  const fireBatchGroup = useCallback(async (phases) => {
    if (!phases || phases.length === 0) return 0;
    const fn = useOurs ? simApi.batchStartGpu : simApi.batchStart;
    const resp = await fn({
      xtal_paths: phases.map(s => s.xtal_path),
      dmin_overrides: Object.fromEntries(phases.map(s => [s.stem, s.dmin])),
      ...commonBatchParams(),
    });
    const bd = resp.data || {};
    (bd.jobs || []).forEach(j => onStarted({
      task_id: j.task_id, status: 'queued', crystalFile: j.xtal_path,
      _batch_id: bd.batch_id,
    }));
    return bd.job_count || (bd.jobs || []).length || 0;
  }, [commonBatchParams, onStarted, useOurs]);

  // Launch the batch from the Simulate-All-Missing dialog. `selection` is the
  // included phases only: [{stem, xtal_path, dmin}] (disordered ones excluded).
  // ALL selected phases route to the single chosen pipeline — no EMsoft-vs-ours
  // partitioning (superseded by the manual engine switch + user decision).
  const launchMissingBatch = async (selection) => {
    setMissingDialog(null);
    if (!selection || selection.length === 0) { setStatusText(t('status.batchCancelledNoPhases')); return; }
    setStatusText(t('status.startingBatch'));
    try {
      const n = await fireBatchGroup(selection);
      setStatusText(t('status.batchStarted', { count: n }));
    } catch (err) {
      setError(t('errors.batchStartFailed', { message: err.response?.data?.detail || err.message }));
      setStatusText(t('status.errorStartingBatch'));
    }
  };

  const handleBatchSelect = async () => {
    let selectedPaths = [];

    if (window.electronAPI?.openFiles) {
      selectedPaths = (await window.electronAPI.openFiles({
        filters: [{ name: 'XTAL Files', extensions: ['xtal'] }],
        properties: ['openFile', 'multiSelections'],
      })) || [];
    } else if (window.electronAPI?.openFile) {
      const single = await window.electronAPI.openFile({
        filters: [{ name: 'XTAL Files', extensions: ['xtal'] }],
      });
      if (single) selectedPaths = [single];
    } else {
      const path = window.prompt(t('dialogs.batchSelectPrompt'));
      if (path) selectedPaths = path.split(';').map(p => p.trim()).filter(Boolean);
    }

    if (selectedPaths.length === 0) return;

    const names = selectedPaths.map(p => p.split(/[\\/]/).pop()).join(', ');
    if (!window.confirm(t('dialogs.batchSelectConfirm', { count: selectedPaths.length, names }))) {
      return;
    }

    setStatusText(t('status.startingBatchForFiles', { count: selectedPaths.length }));
    // Each user-picked file becomes a phase {stem, xtal_path, dmin} using the
    // form dmin (the file picker has no per-phase recommendation). All route to
    // the single chosen pipeline (no EMsoft-vs-ours partitioning).
    const stemOf = (p) => p.split(/[\\/]/).pop().replace(/\.xtal$/i, '');
    const toPhase = (p) => ({ stem: stemOf(p), xtal_path: p, dmin: Number(params.dMin) });
    try {
      const n = await fireBatchGroup(selectedPaths.map(toPhase));
      setStatusText(t('status.batchStarted', { count: n }));
    } catch (err) {
      setStatusText(t('status.batchFailed', { message: err.response?.data?.detail || err.message }));
    }
  };

  const handleStart = async () => {
    if (!params.crystalFile.trim()) {
      setError(t('errors.crystalFileRequired'));
      return;
    }
    if (Number(params.resolution) % 2 === 0) {
      setError(t('errors.resolutionMustBeOdd'));
      return;
    }
    setError(null);
    setLoading(true);
    setSimDone(false);
    setStatusText(t('status.startingSimulation'));
    try {
      const payload = {
        xtal_path:            params.crystalFile,
        ekev:                 Number(params.voltage),
        sig:                  Number(params.sampleTilt),
        totnum_el:            Number(params.totalElectrons),
        numsx:                Number(params.resolution),
        output_type:          params.outputType,
        compute_mode:         params.computeMode,
        dmin:                 Number(params.dMin),
        npx:                  Number(params.masterNpx),
        nthreads:             Number(params.nthreads),
        omega:                Number(params.omega),
        depthmax:             Number(params.depthmax),
        depthstep:            Number(params.depthstep),
        combinesites:         params.combinesites,
        globalworkgrpsz:      Number(params.globalworkgrpsz || 150),
        use_energy_weighting: params.useEnergyWeighting,
        do_legendre:          params.doLegendre,
        esel:                 Number(params.esel),
        uniform:              params.uniform,
      };
      // "Ours" routes to our self-contained GPU/CPU pipeline; "EMsoft" routes to
      // the EMsoft WSL pipeline. The manual switch decides — no auto-routing.
      const resp = useOurs
        ? await simApi.startGpu(payload)
        : await simApi.start(payload);
      setStatusText(t('status.queuedSuccessfully'));
      setSimDone(true);
      onStarted({ ...params, ...resp.data });
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('errors.failedToStart');
      setError(msg);
      setStatusText(t('status.errorPrefix', { message: msg }));
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadNml = async () => {
    const xtName = params.crystalFile
      ? params.crystalFile.split(/[\\/]/).pop().replace(/\.xtal$/i, '')
      : 'crystal';
    try {
      const resp = await simApi.getNmlTemplate({
        xtal_name:        xtName,
        ekev:             Number(params.voltage),
        sig:              Number(params.sampleTilt),
        omega:            Number(params.omega),
        numsx:            Number(params.resolution),
        totnum_el:        Number(params.totalElectrons),
        dmin:             Number(params.dMin),
        npx:              Number(params.masterNpx),
        nthreads:         Number(params.nthreads),
        output_type:      params.outputType,
        platid:           1,
        devid:            1,
        globalworkgrpsz:  Number(params.globalworkgrpsz || 150),
        depthmax:         Number(params.depthmax),
        depthstep:        Number(params.depthstep),
      });
      const blob = new Blob([resp.data], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${xtName}_${params.voltage}kV.nml`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(t('errors.nmlDownloadFailed', { message: err.response?.data?.detail || err.message }));
    }
  };

  // OpenCL label color
  const oclColor = !oclInfo
    ? colors.textSecondary
    : oclInfo.ok ? colors.green : colors.red;

  // Output preview
  const preview = outputPreviewText(params, t);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing }}>

      {/* Pipeline Header — shows active steps + GPU status */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 14px',
        borderRadius: 8,
        background: alpha(colors.bgSecondary, 80),
        border: `1px solid ${colors.border}`,
      }}>
        {/* Pipeline steps */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Label secondary small style={{ marginRight: 4 }}>{t('pipeline.label')}</Label>
          {[
            { label: 'MC', active: true, tooltip: t('pipeline.mcTooltip') },
            { label: 'Master', active: params.outputType === 'master_only' || params.outputType === 'both', tooltip: t('pipeline.masterTooltip') },
            { label: 'SHT', active: params.outputType === 'sht_only' || params.outputType === 'both', tooltip: t('pipeline.shtTooltip') },
          ].map((step, i, arr) => (
            <span key={step.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span
                title={step.tooltip}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '3px 12px',
                  borderRadius: 12,
                  fontSize: '9pt',
                  fontWeight: 600,
                  fontFamily: 'monospace',
                  border: `1.5px solid ${step.active ? colors.cyan : alpha(colors.border, 40)}`,
                  background: step.active ? alpha(colors.cyan, 10) : 'transparent',
                  color: step.active ? colors.cyan : alpha(colors.textSecondary, 40),
                  transition: 'all 0.2s ease',
                }}
              >
                {step.label}
              </span>
              {i < arr.length - 1 && (
                <span style={{ color: step.active && arr[i + 1].active ? colors.cyan : alpha(colors.border, 40), fontSize: '10pt' }}>&rarr;</span>
              )}
            </span>
          ))}
        </div>

        {/* GPU status chip */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '4px 12px',
          borderRadius: 12,
          background: oclInfo?.ok ? alpha(colors.green, 8) : alpha(colors.orange, 8),
          border: `1px solid ${oclInfo?.ok ? alpha(colors.green, 20) : alpha(colors.orange, 20)}`,
          fontSize: '9pt',
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
            background: oclInfo === null ? colors.yellow : oclInfo.ok ? colors.green : colors.orange,
            animation: oclInfo === null ? 'pulse 1.5s ease-in-out infinite' : 'none',
          }} />
          <span style={{ color: oclInfo?.ok ? colors.green : colors.orange, fontFamily: 'monospace' }}>
            {oclInfo === null ? t('pipeline.gpuDetecting')
              : oclInfo.ok
                ? `${oclInfo.raw?.gpu_name || 'GPU'} ${oclInfo.raw?.gpu_memory_gb ? `(${oclInfo.raw.gpu_memory_gb.toFixed(0)} GB)` : ''}`
                : t('pipeline.gpuCpuOnly')}
          </span>
        </div>
      </div>

      {/* Input Crystal Structure */}
      <GroupBox title={t('inputCrystal.groupTitle')}>
        <FormRow label={t('inputCrystal.fileLabel')}>
          <div style={{ display: 'flex', gap: spacing.buttonSpacing }}>
            <Input
              value={params.crystalFile}
              onChange={e => set('crystalFile', e.target.value)}
              placeholder={t('inputCrystal.filePlaceholder')}
              title={t('hoverTips.crystalFileInput')}
              style={{ flex: 1 }}
            />
            <Button onClick={() => setPickerOpen(true)} variant="default" title={t('inputCrystal.databaseTooltip')}>{t('inputCrystal.database')}</Button>
            <Button onClick={handleBrowse} variant="default" title={t('inputCrystal.browseTooltip')}>{t('inputCrystal.browse')}</Button>
          </div>
        </FormRow>
        <CrystalPickerModal
          open={pickerOpen}
          onClose={() => setPickerOpen(false)}
          onSelect={(path) => set('crystalFile', path)}
          currentKv={params.voltage}
        />
        {missingDialog && (
          <SimulateMissingDialog
            missing={missingDialog}
            defaultDmin={Number(params.dMin)}
            engine={engine}
            onLaunch={launchMissingBatch}
            onClose={() => { setMissingDialog(null); setStatusText('Batch cancelled by user.'); }}
          />
        )}
        {preview && (
          <div style={{
            marginTop: spacing.innerSpacing,
            padding: '5px 8px',
            borderRadius: 4,
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            fontSize: '9pt',
            color: colors.textSecondary,
            animation: 'fadeSlideIn 0.2s ease-out',
          }}>
            <span style={{ fontWeight: 600, color: colors.text }}>{t('inputCrystal.outputPrefix')}</span>
            <span style={{ color: colors.accent }}>{preview}</span>
          </div>
        )}
      </GroupBox>

      {/* Output Type — Visual Cards (before parameters so selection gates what appears) */}
      <GroupBox title={t('outputType.groupTitle')}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: spacing.groupSpacing }}>
          {[
            { value: 'sht_only', icon: '\u25CE', title: t('outputType.shtTitle'), sub: t('outputType.shtSub'), ext: t('outputType.shtExt'), tooltip: t('outputType.shtTooltip') },
            { value: 'master_only', icon: '\u229E', title: t('outputType.masterTitle'), sub: t('outputType.masterSub'), ext: t('outputType.masterExt'), tooltip: t('outputType.masterTooltip') },
            { value: 'both', icon: '\u25C8', title: t('outputType.bothTitle'), sub: t('outputType.bothSub'), ext: t('outputType.bothExt'), tooltip: t('outputType.bothTooltip') },
          ].map(opt => {
            const sel = params.outputType === opt.value;
            return (
              <div
                key={opt.value}
                onClick={() => set('outputType', opt.value)}
                title={opt.tooltip}
                style={{
                  cursor: 'pointer',
                  padding: '14px 16px',
                  borderRadius: 8,
                  border: `1.5px solid ${sel ? colors.purple : colors.border}`,
                  background: sel ? alpha(colors.purple, 8) : 'transparent',
                  transition: 'all 0.15s ease',
                  userSelect: 'none',
                }}
                onMouseEnter={e => { if (!sel) { e.currentTarget.style.borderColor = alpha(colors.purple, 50); e.currentTarget.style.transform = 'translateY(-1px)'; } }}
                onMouseLeave={e => { if (!sel) { e.currentTarget.style.borderColor = colors.border; e.currentTarget.style.transform = 'translateY(0)'; } }}
              >
                <div style={{ fontSize: '20px', marginBottom: 6, color: sel ? colors.purple : colors.textSecondary }}>{opt.icon}</div>
                <div style={{ fontSize: '11pt', fontWeight: 600, color: sel ? colors.text : colors.textSecondary, marginBottom: 2 }}>{opt.title}</div>
                <div style={{ fontSize: '9pt', color: colors.textSecondary, marginBottom: 8 }}>{opt.sub}</div>
                <div style={{ fontSize: '9pt', fontFamily: 'monospace', color: sel ? colors.accent : alpha(colors.accent, 50) }}>{opt.ext}</div>
              </div>
            );
          })}
        </div>
      </GroupBox>

      {/* Simulation Parameters */}
      <GroupBox title={t('params.groupTitle')}>
        {/* Row 1: Voltage + Sample Tilt */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: spacing.groupSpacing, marginBottom: spacing.groupSpacing }}>
          <FormRow label={t('params.voltageLabel')}>
            <NumberInput
              value={params.voltage}
              onChange={e => set('voltage', e.target.value)}
              min={5} max={50} step={0.5}
              style={{ width: '100%' }}
              title={t('params.voltageTooltip')}
            />
          </FormRow>
          <FormRow label={t('params.sampleTiltLabel')}>
            <NumberInput
              value={params.sampleTilt}
              onChange={e => set('sampleTilt', e.target.value)}
              min={0} max={90} step={1}
              style={{ width: '100%' }}
              title={t('params.sampleTiltTooltip')}
            />
          </FormRow>
        </div>

        {/* Row 2: Resolution + Total Electrons */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: spacing.groupSpacing, marginBottom: spacing.groupSpacing }}>
          <FormRow label={t('params.numsxLabel')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <NumberInput
                value={params.resolution}
                onChange={e => set('resolution', e.target.value)}
                min={101} max={1001} step={100}
                style={{ width: '100%', borderColor: Number(params.resolution) % 2 === 0 ? colors.red : undefined }}
                title={t('params.numsxTooltip')}
              />
              {Number(params.resolution) % 2 === 0 && (
                <span style={{ fontSize: '8pt', color: colors.red, fontWeight: 600, whiteSpace: 'nowrap' }}>{t('params.mustBeOddInline')}</span>
              )}
            </div>
          </FormRow>
          <FormRow label={<span>totnum_el <span style={{ fontSize: '8pt', color: colors.textSecondary, fontWeight: 400 }}>({Number(params.totalElectrons).toExponential(1)})</span></span>}>
            <NumberInput
              value={params.totalElectrons}
              onChange={e => set('totalElectrons', e.target.value)}
              min={1000000} step={1000000}
              style={{ width: '100%' }}
              title={t('params.totnumElTooltip')}
            />
          </FormRow>
        </div>

        {useOurs && (
          <div style={{
            marginTop: -spacing.innerSpacing,
            marginBottom: spacing.groupSpacing,
            padding: '6px 10px',
            borderRadius: 4,
            background: alpha(colors.cyan, 6),
            border: `1px solid ${alpha(colors.cyan, 18)}`,
            fontSize: '8.5pt',
            color: colors.cyan,
            lineHeight: 1.4,
          }}>
            {t('params.oursElectronNote')}
          </div>
        )}

        {/* Row 3: dmin (CRITICAL) + Master npx (conditional) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: spacing.groupSpacing }}>
          <FormRow>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Label secondary small>
                {t('params.dminLabel')}<sub>min</sub>{t('params.dminUnit')}
                <span style={{ color: colors.yellow, fontWeight: 700 }}>{t('params.dminCritical')}</span>
              </Label>
              <NumberInput
                value={params.dMin}
                onChange={e => set('dMin', e.target.value)}
                min={0.01} max={2.0} step={0.005}
                style={{ width: '100%' }}
                title={t('params.dminTooltip')}
              />
            </div>
          </FormRow>
          {showMasterNpx && (
            <FormRow label={t('params.masterNpxLabel')}>
              <NumberInput
                value={params.masterNpx}
                onChange={e => set('masterNpx', e.target.value)}
                min={100} max={2000} step={50}
                style={{ width: '100%' }}
                title={t('params.masterNpxTooltip')}
              />
            </FormRow>
          )}
        </div>
      </GroupBox>

      {/* Compute Settings — Toggle + Context Card.
          EMsoft-ONLY: the compute_mode (auto/gpu/cpu OpenCL↔OpenMP) toggle drives
          the EMsoft Fortran binaries' OpenCL device / OpenMP thread selection. Our
          forward runner ignores compute_mode entirely (it auto-picks GPU/CPU per
          step), so this card is dead UI under the "Ours" engine — hide it there.
          NB: this is NOT the engine toggle (that lives in the header). */}
      {engine === 'emsoft' && (
      <GroupBox title={t('compute.groupTitle')}>
        {/* Mode toggle buttons */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          {COMPUTE_MODE_OPTIONS.map(opt => {
            const active = params.computeMode === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => set('computeMode', opt.value)}
                title={opt.value === 'auto' ? t('hoverTips.computeModeAuto') : opt.value === 'gpu' ? t('hoverTips.computeModeGpu') : t('hoverTips.computeModeCpu')}
                style={{
                  flex: 1,
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: `1.5px solid ${active ? colors.cyan : colors.border}`,
                  background: active ? alpha(colors.cyan, 12) : 'transparent',
                  color: active ? colors.cyan : colors.textSecondary,
                  fontSize: '10pt',
                  fontWeight: active ? 600 : 400,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  fontFamily: 'inherit',
                }}
              >
                {opt.value === 'auto' ? t('compute.auto') : opt.value === 'gpu' ? t('compute.gpu') : t('compute.cpu')}
              </button>
            );
          })}
        </div>

        {/* Context card — changes based on mode */}
        {(params.computeMode === 'gpu' || (params.computeMode === 'auto' && oclInfo?.ok)) && (
          <div style={{
            padding: '10px 14px',
            borderRadius: 6,
            border: `1px solid ${alpha(colors.cyan, 25)}`,
            background: alpha(colors.cyan, 5),
            marginBottom: 8,
            fontSize: '9pt',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: colors.green, flexShrink: 0 }} />
              <span style={{ color: colors.text, fontWeight: 600 }}>
                {oclInfo?.raw?.gpu_name || oclInfo?.text || t('compute.gpuDetected')}
              </span>
              {oclInfo?.raw?.gpu_memory_gb > 0 && (
                <span style={{ color: colors.textSecondary }}>
                  {t('compute.vram', { gb: oclInfo.raw.gpu_memory_gb.toFixed(1) })}
                </span>
              )}
            </div>
            <div style={{ color: colors.textSecondary, display: 'flex', gap: 16 }}>
              <span>{t('compute.platform')}</span>
              <span>{t('compute.blockSize', { size: oclInfo?.raw?.gpu_memory_gb >= 16 ? 32 : oclInfo?.raw?.gpu_memory_gb >= 8 ? 16 : 8 })}</span>
            </div>
            {oclInfo?.raw?.gpu_memory_gb > 0 && oclInfo.raw.gpu_memory_gb < 16 && (
              <div style={{ color: colors.yellow, marginTop: 6, fontSize: '8.5pt' }}>
                {t('compute.lowVramWarning', { gb: oclInfo.raw.gpu_memory_gb.toFixed(0) })}
              </div>
            )}
          </div>
        )}

        {(params.computeMode === 'cpu' || (params.computeMode === 'auto' && !oclInfo?.ok)) && (
          <div style={{
            padding: '10px 14px',
            borderRadius: 6,
            border: `1px solid ${alpha(colors.purple, 25)}`,
            background: alpha(colors.purple, 5),
            marginBottom: 8,
            fontSize: '9pt',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: colors.purple, flexShrink: 0 }} />
              <span style={{ color: colors.text, fontWeight: 600 }}>{t('compute.cpuModeTitle')}</span>
              <span style={{ color: colors.textSecondary }}>
                {t('compute.coresAvailable', { count: oclInfo?.raw?.cpu_count || '?' })}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Label secondary small>{t('compute.threadsLabel')}</Label>
              <NumberInput
                value={params.nthreads}
                onChange={e => set('nthreads', e.target.value)}
                min={1} max={128} step={1}
                style={{ width: 70 }}
                title={t('compute.threadsTooltip')}
              />
              <Label secondary small style={{ color: colors.textSecondary }}>{t('compute.autoDetected')}</Label>
            </div>
          </div>
        )}

        {/* GPU mode: show threads info (read-only, auto-adjusted to 4N+3) */}
        {params.computeMode === 'gpu' && (
          <div style={{ fontSize: '9pt', color: colors.textSecondary, display: 'flex', alignItems: 'center', gap: 10 }}>
            <Label secondary small>{t('compute.cpuHelperThreadsLabel')}</Label>
            <span style={{ fontFamily: 'monospace', color: colors.cyan }}>
              {(() => {
                const c = oclInfo?.raw?.cpu_count || 24;
                const base = Math.max(7, c - 2);
                for (let n = base; n >= 7; n--) { if ((n - 3) % 4 === 0) return n; }
                return 7;
              })()}
            </span>
            <Label secondary small>{t('compute.helperThreadsNote')}</Label>
          </div>
        )}
      </GroupBox>
      )}

      {/* Server Mode */}
      <ServerModePanel />

      {/* Advanced Options */}
      <CollapsibleGroup title={t('advanced.groupTitle')} defaultCollapsed={true}>
        {/* Monte Carlo section */}
        <div style={{ marginBottom: spacing.groupSpacing }}>
          <Label style={{ color: colors.purple, fontWeight: 700, display: 'block', marginBottom: spacing.innerSpacing }}>
            {t('advanced.monteCarloHeading')}
          </Label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: spacing.groupSpacing }}>
            <FormRow label={t('advanced.omegaLabel')}>
              <NumberInput
                value={params.omega}
                onChange={e => set('omega', e.target.value)}
                min={0} max={90} step={1}
                style={{ width: '100%' }}
                title={t('advanced.omegaTooltip')}
              />
            </FormRow>
            <FormRow label={t('advanced.depthMaxLabel')}>
              <NumberInput
                value={params.depthmax}
                onChange={e => set('depthmax', e.target.value)}
                min={10} max={500} step={1}
                style={{ width: '100%' }}
                title={t('advanced.depthMaxTooltip')}
              />
            </FormRow>
            <FormRow label={t('advanced.depthStepLabel')}>
              <NumberInput
                value={params.depthstep}
                onChange={e => set('depthstep', e.target.value)}
                min={0.1} max={10} step={0.1}
                style={{ width: '100%' }}
                title={t('advanced.depthStepTooltip')}
              />
            </FormRow>
          </div>
        </div>

        <Separator />

        {/* Master Pattern section */}
        <div style={{ marginTop: spacing.groupSpacing }}>
          <Label style={{ color: colors.purple, fontWeight: 700, display: 'block', marginBottom: spacing.innerSpacing }}>
            {t('advanced.masterPatternHeading')}
          </Label>
          <FormRow label={t('advanced.eselLabel')}>
            <NumberInput
              value={params.esel}
              onChange={e => set('esel', e.target.value)}
              min={-1} max={50} step={1}
              style={{ width: 80 }}
              title={t('advanced.eselTooltip')}
            />
          </FormRow>
          <div style={{ marginTop: spacing.innerSpacing }}>
            <CheckRow
              label={t('advanced.combinesitesLabel')}
              checked={params.combinesites}
              onChange={v => set('combinesites', v)}
              title={t('advanced.combinesitesTooltip')}
            />
            <CheckRow
              label={t('advanced.useEnergyWeightingLabel')}
              checked={params.useEnergyWeighting}
              onChange={v => set('useEnergyWeighting', v)}
              title={t('advanced.useEnergyWeightingTooltip')}
            />
            <CheckRow
              label={t('advanced.doLegendreLabel')}
              checked={params.doLegendre}
              onChange={v => set('doLegendre', v)}
              title={t('advanced.doLegendreTooltip')}
            />
            <CheckRow
              label={t('advanced.uniformLabel')}
              checked={params.uniform}
              onChange={v => set('uniform', v)}
              title={t('advanced.uniformTooltip')}
            />
          </div>
        </div>
      </CollapsibleGroup>

      {/* Estimates */}
      <div style={{
        display: 'flex',
        gap: 12,
        fontSize: '9pt',
        color: colors.textSecondary,
        alignItems: 'center',
      }}>
        <span style={{
          padding: '2px 8px', borderRadius: 8,
          background: alpha(colors.cyan, 8), border: `1px solid ${alpha(colors.cyan, 15)}`,
          color: colors.cyan,
          transition: 'transform 0.12s, border-color 0.15s',
          display: 'inline-block',
        }} title={t('estimates.timeTooltip')}
          onMouseEnter={e => { e.currentTarget.style.transform = 'scale(1.06)'; e.currentTarget.style.borderColor = colors.cyan; }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; e.currentTarget.style.borderColor = alpha(colors.cyan, 15); }}
        >
          {'\u23F1'} {estimatedTime(params, t)}
        </span>
        <span style={{
          padding: '2px 8px', borderRadius: 8,
          background: alpha(colors.textSecondary, 8), border: `1px solid ${alpha(colors.textSecondary, 15)}`,
          transition: 'transform 0.12s',
          display: 'inline-block',
        }} title={t('estimates.diskTooltip')}
          onMouseEnter={e => { e.currentTarget.style.transform = 'scale(1.06)'; }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; }}
        >
          {'\uD83D\uDCBE'} {t('estimates.diskValue')}
        </span>
        {Number(params.resolution) % 2 === 0 && (
          <span style={{
            color: colors.red, fontWeight: 600,
            padding: '2px 8px', borderRadius: 8,
            background: alpha(colors.red, 8), border: `1px solid ${alpha(colors.red, 15)}`,
          }} title={t('estimates.numsxOddBadgeTooltip')}>
            {'\u26A0'} {t('estimates.numsxOddBadge')}
          </span>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div role="alert" style={{
          background: alpha(colors.red, 13),
          border: `1px solid ${alpha(colors.red, 27)}`,
          borderRadius: 4,
          padding: '7px 12px',
          color: colors.red,
          fontSize: '9pt',
          marginBottom: spacing.outerSpacing,
          animation: 'fadeSlideIn 0.25s ease-out',
        }}>
          {error}
        </div>
      )}

      {/* Primary action buttons */}
      <div style={{ display: 'flex', gap: spacing.buttonSpacing, flexWrap: 'wrap', alignItems: 'center' }}>
        <Button
          variant="default"
          onClick={handleSimulateAllMissing}
          title={useOurs
            ? t('actions.simulateAllMissingTooltipOurs')
            : t('actions.simulateAllMissingTooltipEmsoft')}
        >
          {t('actions.simulateAllMissing')}
        </Button>
        <Button
          variant="default"
          onClick={handleBatchSelect}
          title={useOurs
            ? t('actions.batchSelectTooltipOurs')
            : t('actions.batchSelectTooltipEmsoft')}
        >
          {t('actions.batchSelect')}
        </Button>
        <div style={{ flex: 1 }} />
        <Button
          variant="ghost"
          onClick={engine === 'emsoft' ? handleDownloadNml : undefined}
          disabled={engine !== 'emsoft'}
          title={engine === 'emsoft'
            ? t('actions.downloadNmlTooltipEmsoft')
            : t('actions.downloadNmlTooltipDisabled')}
        >
          {t('actions.downloadNml')}
        </Button>
        <Button
          variant="primary"
          onClick={handleStart}
          disabled={loading || Number(params.resolution) % 2 === 0}
          title={useOurs
            ? t('actions.startTooltipOurs')
            : t('actions.startTooltipEmsoft')}
        >
          {loading ? <span className="btn-loading">{t('actions.starting')}</span> : t('actions.startSimulation')}
        </Button>
      </div>

      <Separator />

      {/* System check loading indicator */}
      {sysLoading && (
        <div style={{
          marginTop: spacing.outerSpacing,
          padding: '12px 16px',
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: 4,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block', fontSize: '14pt' }}>&#x21BB;</span>
          <Label secondary small>{t('systemCheck.checkingInline')}</Label>
        </div>
      )}

      {/* System status panel (shown after Check) */}
      {sysStatus && !sysLoading && (
        <GroupBox title={t('systemCheck.statusGroupTitle')}>
          {sysStatus.error ? (
            <Label style={{ color: colors.red }}>{'\u26A0'} {sysStatus.error}</Label>
          ) : (
            Object.entries(sysStatus).map(([key, val]) => (
              <div
                key={key}
                className="table-row-hover"
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '5px 4px',
                  borderBottom: `1px solid ${colors.border}`,
                  borderRadius: 2,
                }}
              >
                <Label secondary small style={{ textTransform: 'capitalize' }}>
                  {key.replace(/_/g, ' ')}
                </Label>
                <Label small style={{
                  color: typeof val === 'boolean'
                    ? (val ? colors.green : colors.red)
                    : colors.accent,
                  fontFamily: 'monospace',
                }}>
                  {typeof val === 'boolean'
                    ? (val ? t('systemCheck.valueOk') : t('systemCheck.valueNotFound'))
                    : Array.isArray(val)
                      ? (val.length === 0 ? '—' : val.join('; '))
                      : (typeof val === 'object' && val !== null
                        ? Object.entries(val).map(([k, v]) => `${k}: ${v}`).join(', ')
                        : (key === 'gpu_memory_gb' && typeof val === 'number'
                          ? t('systemCheck.gbSuffix', { value: val.toFixed(1) })
                          : String(val)))}
                </Label>
              </div>
            ))
          )}
        </GroupBox>
      )}

      {/* Status bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: colors.bgSecondary,
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        padding: `6px 12px`,
        marginTop: spacing.outerSpacing,
        minHeight: 36,
        gap: spacing.buttonSpacing,
      }}>
        <Label secondary small style={{ flex: 1 }}>{statusText}</Label>
        <Button
          variant="default"
          onClick={handleCheckSystem}
          disabled={sysLoading}
          title={t('hoverTips.checkSystem')}
          small
        >
          {sysLoading ? t('systemCheck.checking') : t('systemCheck.checkSystem')}
        </Button>
        <Button
          variant="ghost"
          disabled={!simDone}
          onClick={() => onShowQueue && onShowQueue()}
          title={simDone
            ? t('systemCheck.viewMasterTooltipReady')
            : t('systemCheck.viewMasterTooltipNotReady')}
          small
        >
          {t('systemCheck.viewMasterPattern')}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Queue & Running Tab
// ---------------------------------------------------------------------------
function QueueTab({ jobs, onStop, onCancelBatch, onBackToConfig }) {
  const { t } = useTranslation('simulation');
  const logRef = useRef(null);
  const [logCopied, setLogCopied] = useState(false);

  // Collect log messages: show only the most recent/active job's logs,
  // with separators between jobs if multiple have logs
  const logLines = (() => {
    const activeJob = jobs.find(j => j.status === 'running');
    if (activeJob) return activeJob.logLines ?? [];
    // No running job — show the most recent job that has logs
    const withLogs = jobs.filter(j => (j.logLines ?? []).length > 0);
    if (withLogs.length > 0) return withLogs[0].logLines ?? [];
    return [];
  })();

  // Auto-scroll log to bottom when new lines appear
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines.length]);

  const hasActive = jobs.some(j => j.status === 'running' || j.status === 'queued');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing }}>

      {/* Jobs table */}
      {jobs.length === 0 ? (
        <div style={{ textAlign: 'center', color: colors.textSecondary, padding: '40px 0', fontSize: '10pt' }}>
          <div style={{ fontSize: '24pt', opacity: 0.3, marginBottom: 6 }}>{'\u26A1'}</div>
          <div>{t('queueTab.emptyTitle')}</div>
          <div style={{ fontSize: '8pt', marginTop: 4, opacity: 0.6 }}>
            {t('queueTab.emptyHint')}
          </div>
        </div>
      ) : (
        <div style={{
          border: `1px solid ${colors.border}`,
          borderRadius: 6,
          overflow: 'hidden',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: colors.bgSecondary }}>
                {[
                  { key: 'crystal', label: t('queueTab.colCrystal') },
                  { key: 'status', label: t('queueTab.colStatus') },
                  { key: 'progress', label: t('queueTab.colProgress') },
                  { key: 'actions', label: t('queueTab.colActions') },
                ].map(h => (
                  <th key={h.key} style={{
                    padding: '7px 10px',
                    textAlign: 'left',
                    fontSize: '9pt',
                    fontWeight: 600,
                    color: colors.textSecondary,
                    textTransform: 'uppercase',
                    letterSpacing: 0.6,
                    borderBottom: `1px solid ${colors.border}`,
                  }}>
                    {h.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {jobs.map((job, idx) => (
                <tr
                  key={job.task_id || idx}
                  className="table-row-hover"
                  style={{ background: idx % 2 === 0 ? 'transparent' : alpha(colors.bgSecondary, 53) }}
                >
                  <td title={job.crystal_file || job.task_id} style={{ padding: '8px 10px', fontSize: '10pt', color: colors.accent, fontFamily: 'monospace', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {job.crystal_file?.split(/[\\/]/).pop() || job.task_id}
                  </td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>
                    <StatusBadge status={job.status} />
                  </td>
                  <td style={{ padding: '8px 10px', minWidth: 160 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ flex: 1 }}>
                        <ProgressBar
                          value={job.progress ?? 0}
                          color={job.status === 'failed' ? colors.red : colors.purple}
                        />
                      </div>
                      <Label secondary small style={{ whiteSpace: 'nowrap', minWidth: 36, textAlign: 'right' }}>
                        {job.progress ?? 0}%
                      </Label>
                    </div>
                    {job.message && (
                      <Label secondary small style={{ display: 'block', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {job.message}
                      </Label>
                    )}
                    {job.eta && (
                      <Label secondary small style={{ display: 'block' }}>{t('queueTab.etaPrefix', { eta: job.eta })}</Label>
                    )}
                  </td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', display: 'flex', gap: 4 }}>
                    {(job.status === 'running' || job.status === 'queued') && (
                      <Button variant="danger" small onClick={() => onStop(job.task_id)} title={t('hoverTips.stopJob')}>
                        {t('queueTab.stop')}
                      </Button>
                    )}
                    {job.status === 'completed' && job.result && (
                      <Button variant="accent" small title={t('hoverTips.viewResult')} onClick={() => {
                        const p = job.result?.master || job.result?.sht || job.result?.h5;
                        if (!p) return;
                        // Open the finished file in the in-app HDF5 viewer. The
                        // old `/api/h5/viewer` URL never existed (404 in a new
                        // tab); stash the path + ask App.jsx to open the viewer.
                        try { sessionStorage.setItem('h5_preload_path', p); } catch { /* unavailable */ }
                        window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'h5viewer' } }));
                      }}>
                        {t('queueTab.viewResult')}
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Simulation Log */}
      <GroupBox title={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <span>{t('queueTab.logTitle')}</span>
          <button
            onClick={() => {
              if (logLines.length === 0) return;
              navigator.clipboard.writeText(logLines.join('\n')).then(() => {
                setLogCopied(true);
                setTimeout(() => setLogCopied(false), 1200);
              }).catch(() => {});
            }}
            disabled={logLines.length === 0}
            title={logCopied ? t('queueTab.logCopied') : t('queueTab.logCopyTooltip')}
            aria-label={t('queueTab.logCopyAria')}
            style={{
              background: 'none', border: 'none', cursor: logLines.length ? 'pointer' : 'default',
              color: logCopied ? colors.green : colors.textSecondary, fontSize: '9pt', padding: '1px 4px',
              opacity: logLines.length ? 0.7 : 0.3, fontWeight: 400, borderRadius: 3,
              transition: 'color 0.2s, opacity 0.15s, background 0.15s',
            }}
            onMouseEnter={(e) => { if (logLines.length) { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = `${colors.border}55`; } }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = logLines.length ? '0.7' : '0.3'; e.currentTarget.style.background = 'none'; }}
          >
            {logCopied ? '\u2713' : '\uD83D\uDCCB'}
          </button>
        </div>
      }>
        <div
          ref={logRef}
          className="thin-scrollbar"
          style={{
            minHeight: 180,
            maxHeight: 280,
            overflowY: 'auto',
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 4,
            padding: '8px 10px',
            fontFamily: 'monospace',
            fontSize: '9pt',
            color: colors.text,
            whiteSpace: 'pre-wrap',
          }}
        >
          {logLines.length === 0
            ? <span style={{ color: colors.textSecondary, opacity: 0.6 }}>{t('queueTab.logEmpty')}</span>
            : logLines.map((line, i) => (
              <div key={i} style={{
                display: 'flex', gap: 8,
                color: line.includes('ERROR') ? colors.red
                  : line.includes('complete') || line.includes('Success') ? colors.green
                  : line.includes('WARNING') ? colors.orange
                  : colors.text,
              }}>
                <span style={{ color: colors.textSecondary, opacity: 0.3, minWidth: 20, textAlign: 'right', userSelect: 'none' }}>{i + 1}</span>
                <span>{line}</span>
              </div>
            ))}
        </div>
      </GroupBox>

      {/* Button bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: spacing.buttonSpacing }}>
        <Button variant="danger" onClick={onCancelBatch} disabled={!hasActive} title={t('hoverTips.cancelRemaining')}>
          {t('queueTab.cancelRemaining')}
        </Button>
        <div style={{ flex: 1, textAlign: 'center', fontSize: '9pt', color: colors.textSecondary }}>
          {jobs.length > 0 && (() => {
            const done = jobs.filter(j => j.status === 'completed').length;
            const running = jobs.filter(j => j.status === 'running').length;
            const pending = jobs.filter(j => j.status === 'queued' || j.status === 'pending').length;
            const failed = jobs.filter(j => j.status === 'failed').length;
            const parts = [];
            if (done) parts.push(t('queueTab.summaryDone', { count: done }));
            if (running) parts.push(t('queueTab.summaryRunning', { count: running }));
            if (pending) parts.push(t('queueTab.summaryPending', { count: pending }));
            if (failed) parts.push(t('queueTab.summaryFailed', { count: failed }));
            return parts.join(' \u00b7 ');
          })()}
        </div>
        <Button variant="default" onClick={onBackToConfig} title={t('hoverTips.backToConfigure')}>
          {t('queueTab.backToConfigure')}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload History Tab
// ---------------------------------------------------------------------------
function HistoryTab({ history, onClear }) {
  const { t } = useTranslation('simulation');
  const thStyle = {
    padding: '7px 10px',
    textAlign: 'left',
    fontSize: '9pt',
    fontWeight: 600,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    borderBottom: `1px solid ${colors.border}`,
  };
  const tdStyle = {
    padding: '8px 10px',
    fontSize: '10pt',
    borderBottom: `1px solid ${alpha(colors.border, 13)}`,
    color: colors.text,
    verticalAlign: 'middle',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing }}>
      <Label secondary small>{t('historyTab.intro')}</Label>

      {history.length === 0 ? (
        <div style={{ textAlign: 'center', color: colors.textSecondary, padding: '40px 0', fontSize: '10pt' }}>
          <div style={{ fontSize: '24pt', opacity: 0.3, marginBottom: 6 }}>{'\u2637'}</div>
          <div>{t('historyTab.emptyTitle')}</div>
          <div style={{ fontSize: '8pt', marginTop: 4, opacity: 0.6 }}>
            {t('historyTab.emptyHint')}
          </div>
        </div>
      ) : (
        <div style={{
          border: `1px solid ${colors.border}`,
          borderRadius: 6,
          overflow: 'hidden',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: colors.bgSecondary }}>
                <th style={thStyle}>{t('historyTab.colFile')}</th>
                <th style={thStyle}>{t('historyTab.colMaterial')}</th>
                <th style={thStyle}>{t('historyTab.colStatus')}</th>
                <th style={thStyle}>{t('historyTab.colTime')}</th>
                <th style={thStyle}>{t('historyTab.colSize')}</th>
              </tr>
            </thead>
            <tbody>
              {history.map((entry, idx) => {
                // Support both legacy fields (crystal_file/material) and new persisted fields (crystal/method)
                const crystalFile = entry.crystal || entry.crystal_file || '';
                const fileName = crystalFile.split(/[\\/]/).pop() || entry.output_file || '—';
                const material = entry.material || crystalFile.split(/[\\/]/).pop()?.replace(/\.xtal$/i, '') || '—';
                const timeStr = entry.started_at ? new Date(entry.started_at).toLocaleString() : '—';
                return (
                  <tr
                    key={entry.id || entry.task_id || idx}
                    className="table-row-hover"
                    style={{ background: idx % 2 === 0 ? 'transparent' : alpha(colors.bgSecondary, 53) }}
                  >
                    <td title={crystalFile || fileName} style={{ ...tdStyle, fontFamily: 'monospace', color: colors.accent, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {fileName}
                    </td>
                    <td style={tdStyle}>{material}</td>
                    <td style={tdStyle}>
                      <StatusBadge status={entry.status} />
                    </td>
                    <td style={{ ...tdStyle, color: colors.textSecondary }}>{timeStr}</td>
                    <td style={{ ...tdStyle, color: colors.textSecondary }}>
                      {entry.size_mb != null ? entry.size_mb.toFixed(1) : (entry.method || '—')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="default" small onClick={onClear} title={t('hoverTips.clearHistory')}>
          {t('historyTab.clearHistory')}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function SimulationPage({ isActive = false }) {
  const { t } = useTranslation('simulation');
  const [activeTab, setActiveTab] = useState('configure');
  const [jobs, setJobs]           = useState([]);
  const [history, setHistory]     = useState([]);
  // Two-way engine switch: 'ours' (default, self-contained GPU/CPU pipeline,
  // no WSL/EMsoft needed) | 'emsoft' (optional WSL+EMsoft pipeline).
  const [engine, setEngine]       = useState('ours');
  // System capability — drives the "Ours" hardware-path label + whether the
  // EMsoft toggle option is selectable. {emsoft_available, has_gpu, gpu_name,
  // cpu_count, ...} from GET /api/simulation/system-status. null = not yet known.
  const [cap, setCap]             = useState(null);
  const [askConfirm, confirmProps] = useConfirm();
  const [askPrompt, promptProps] = usePrompt();

  // Probe system capability when the page becomes active (cached 5 min backend
  // side). EMsoft stays disabled until we positively learn it's available.
  useEffect(() => {
    if (!isActive) return;
    let alive = true;
    simApi.systemStatus()
      .then(resp => { if (alive) setCap(resp.data || {}); })
      .catch(() => { if (alive) setCap({}); });
    return () => { alive = false; };
  }, [isActive]);

  // EMsoft is OPTIONAL — only enabled once we positively detect WSL+EMsoft.
  const emsoftAvailable = !!(cap && cap.emsoft_available);

  // If the user is on EMsoft but the capability check reports it's unavailable,
  // fall back to 'ours' (the always-available, self-contained pipeline).
  useEffect(() => {
    if (cap && !emsoftAvailable && engine === 'emsoft') setEngine('ours');
  }, [cap, emsoftAvailable, engine]);

  // Auto-selected hardware path for the "Ours" pipeline (display only — the
  // backend resolves cupy-GPU -> numba/pytorch-CPU per step at run time).
  const oursHardwarePath = (() => {
    if (!cap) return null;
    if (cap.has_gpu && cap.gpu_name) return t('engine.hardwareGpuNamed', { name: cap.gpu_name });
    if (cap.has_gpu) return t('engine.hardwareGpu');
    if (cap.cpu_count) return t('engine.hardwareCpuCores', { count: cap.cpu_count });
    return t('engine.hardwareCpu');
  })();

  // Poll active jobs every 3 s
  const pollJobs = useCallback(async () => {
    // Include 'pending' — backend batch jobs are 'pending' until the worker flips
    // them to 'running'. Excluding it caused the Queue to FREEZE: the moment the
    // frontend snapshot held only completed + pending jobs (the transient window
    // between one batch job finishing and the next going 'running', or after a
    // machine sleep paused the poll timer), this returned early and never polled
    // again — so a fully-completing backend batch looked stuck at "1 done · N
    // pending". As long as ANY job is still pending we must keep polling.
    const active = jobs.filter(j => isActiveJobStatus(j.status));
    if (active.length === 0) return;

    // Group by batch_id for batch polling, individual for single jobs
    const batchIds = [...new Set(active.map(j => j._batch_id).filter(Boolean))];
    const singleJobs = active.filter(j => !j._batch_id);

    // Poll batch jobs
    for (const bid of batchIds) {
      try {
        const resp = await simApi.batchStatus(bid);
        const batchJobs = resp.data?.jobs || [];
        setJobs(prev => prev.map(j => {
          if (j._batch_id !== bid) return j;
          const match = batchJobs.find(b => b.task_id === j.task_id);
          if (!match) return j;
          return { ...j, status: match.status, progress: match.progress || 0, message: match.message || '', result: match.result || j.result, logLines: match.logLines || j.logLines || [] };
        }));
      } catch { /* ignore */ }
    }

    // Poll single jobs
    const updated = await Promise.all(
      singleJobs.map(async j => {
        try {
          const resp = await simApi.getStatus(j.task_id);
          return { ...j, ...resp.data };
        } catch {
          return j;
        }
      })
    );
    if (updated.length > 0) {
      setJobs(prev => prev.map(j => {
        const u = updated.find(u => u.task_id === j.task_id);
        return u ?? j;
      }));
    }
  }, [jobs]);

  useEffect(() => {
    // Poll while this page is visible, OR while jobs are still running in the
    // background (so progress keeps updating even if the user navigated away).
    // A list of only terminal jobs (all completed/failed) no longer keeps the
    // 1s interval alive — that was needless background load, since the page
    // never unmounts (App.jsx keeps all pages mounted and just toggles
    // display).
    const hasRunningJobs = jobs.some((j) => isActiveJobStatus(j.status));
    if (!isActive && !hasRunningJobs) return undefined;
    const id = setInterval(pollJobs, 1000);
    return () => clearInterval(id);
  }, [isActive, pollJobs, jobs]);

  // Load history when switching to History tab
  const loadHistory = useCallback(async () => {
    try {
      const resp = await simApi.history();
      setHistory(resp.data?.jobs || resp.data || []);
    } catch {
      // History endpoint may not be present yet — ignore silently
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'history') loadHistory();
  }, [activeTab, loadHistory]);

  const handleStarted = useCallback((data) => {
    setJobs(prev => [
      {
        task_id:      data.task_id,
        _batch_id:    data._batch_id,
        status:       data.status || 'queued',
        progress:     0,
        message:      'Queued',
        crystal_file: data.crystalFile || data.xtal_path || '',
        logLines:     [],
      },
      ...prev,
    ]);
    setActiveTab('queue');
  }, []);

  const handleStop = useCallback(async (taskId) => {
    try {
      await simApi.stop(taskId);
      setJobs(prev =>
        prev.map(j => j.task_id === taskId ? { ...j, status: 'stopped' } : j)
      );
    } catch {
      // ignore
    }
  }, []);

  const handleCancelBatch = useCallback(async () => {
    // Cancel batch jobs via batch endpoint
    const batchIds = [...new Set(jobs.filter(j => j._batch_id && isActiveJobStatus(j.status)).map(j => j._batch_id))];
    for (const bid of batchIds) {
      try { await simApi.batchCancel(bid); } catch { /* ignore */ }
    }
    // Cancel remaining single jobs
    jobs
      .filter(j => !j._batch_id && (j.status === 'running' || j.status === 'queued'))
      .forEach(j => handleStop(j.task_id));
  }, [jobs, handleStop]);

  const handleClearHistory = useCallback(() => {
    askConfirm({
      title: t('historyTab.clearTitle'),
      message: t('historyTab.clearMessage'),
      confirmLabel: t('historyTab.clearConfirmLabel'),
      onConfirm: async () => {
        try { await simApi.clearHistory(); } catch { /* ignore */ }
        setHistory([]);
      },
    });
  }, [askConfirm, t]);

  const activeCount = jobs.filter(j => isActiveJobStatus(j.status)).length;

  const TAB_DEFS = [
    { id: 'configure', label: t('tabs.configure'),   tip: t('tabs.configureTip') },
    { id: 'queue',     label: activeCount > 0 ? t('tabs.queueWithCount', { count: activeCount }) : t('tabs.queue'), tip: t('tabs.queueTip') },
    { id: 'history',   label: t('tabs.history'), tip: t('tabs.historyTip') },
  ];

  return (
    <div style={{
      background: colors.bg,
      color: colors.text,
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      fontSize: '10pt',
      padding: `${spacing.outerMargin}px ${spacing.outerMargin}px 0`,
    }}>
      {/* Page header */}
      <div style={{
        padding: `0 0 ${spacing.outerSpacing}px 0`,
        borderBottom: `1px solid ${colors.border}`,
        marginBottom: spacing.outerSpacing,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <h1 style={{ fontSize: '18pt', fontWeight: 700, color: colors.accent, margin: 0 }}>
            {engine === 'emsoft'
              ? t('header.titleEmsoft')
              : oursHardwarePath
                ? t('header.titleOursWithHardware', { hardware: oursHardwarePath })
                : t('header.titleOursPlain')}
          </h1>
          {jobs.filter(j => j.status === 'running').length > 0 && (
            <span style={{
              fontSize: '8pt', fontWeight: 600, padding: '2px 8px', borderRadius: 10,
              background: `${colors.green}1a`, color: colors.green,
              animation: 'pulse 2s ease-in-out infinite',
            }}>
              {t('header.runningBadge', { count: jobs.filter(j => j.status === 'running').length })}
            </span>
          )}
          {/* Engine toggle — matches COMPUTE_MODE segmented button style.
              Two-way: "Ours" (default, self-contained, always available) and
              "EMsoft" (optional — disabled until WSL+EMsoft is detected). */}
          <div
            role="group"
            aria-label={t('engine.groupLabel')}
            style={{ display: 'flex', gap: 6, marginLeft: 'auto' }}
          >
            {[
              {
                value: 'ours',
                label: oursHardwarePath ? t('engine.oursLabelWithHardware', { hardware: oursHardwarePath }) : t('engine.oursLabel'),
                disabled: false,
                tip: t('engine.oursTooltip'),
              },
              {
                value: 'emsoft',
                label: t('engine.emsoftLabel'),
                disabled: !emsoftAvailable,
                tip: emsoftAvailable
                  ? t('engine.emsoftTooltipAvailable')
                  : t('engine.emsoftTooltipUnavailable'),
              },
            ].map(opt => {
              const active = engine === opt.value;
              return (
                <button
                  key={opt.value}
                  onClick={() => { if (!opt.disabled) setEngine(opt.value); }}
                  title={opt.tip}
                  aria-pressed={active}
                  disabled={opt.disabled}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 6,
                    border: `1.5px solid ${active ? colors.cyan : colors.border}`,
                    background: active ? alpha(colors.cyan, 12) : 'transparent',
                    color: opt.disabled ? alpha(colors.textSecondary, 40) : active ? colors.cyan : colors.textSecondary,
                    fontSize: '10pt',
                    fontWeight: active ? 600 : 400,
                    cursor: opt.disabled ? 'not-allowed' : 'pointer',
                    opacity: opt.disabled ? 0.6 : 1,
                    transition: 'all 0.15s ease',
                    fontFamily: 'inherit',
                  }}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>
        <div style={{ fontSize: '10pt', color: colors.textSecondary, marginTop: 4 }}>
          {engine === 'emsoft'
            ? t('header.subtitleEmsoft')
            : t('header.subtitleOurs')}
        </div>
      </div>

      {/* Tab bar */}
      <Tabs
        tabs={TAB_DEFS}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        style={{ marginBottom: spacing.outerSpacing }}
      />

      {/* Tab content */}
      <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: `0 0 ${spacing.outerMargin}px 0` }}>
        <TabPanel visible={activeTab === 'configure'} style={{ padding: 0 }}>
          <ConfigureTab onStarted={handleStarted} isActive={isActive} onShowQueue={() => setActiveTab('queue')} engine={engine} />
        </TabPanel>

        <TabPanel visible={activeTab === 'queue'} style={{ padding: 0 }}>
          <QueueTab
            jobs={jobs}
            onStop={handleStop}
            onCancelBatch={handleCancelBatch}
            onBackToConfig={() => setActiveTab('configure')}
          />
        </TabPanel>

        <TabPanel visible={activeTab === 'history'} style={{ padding: 0 }}>
          <HistoryTab history={history} onClear={handleClearHistory} />
        </TabPanel>
      </div>
      <ConfirmDialog {...confirmProps} />
      <PromptDialog {...promptProps} />
    </div>
  );
}
