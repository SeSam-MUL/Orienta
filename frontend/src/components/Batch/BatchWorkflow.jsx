import { useState, useEffect, useCallback } from 'react';
import { useTranslation, Trans } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import { Button, GroupBox, Input } from '../../theme/components';
import { batchV2Api, indexApi } from '../../services/api';
import useBatchConfigStore from '../../stores/useBatchConfigStore';
import useBatchStore from '../../stores/useBatchStore';
import BatchFileTree from './BatchFileTree';
import BatchDashboard from './BatchDashboard';

// ---------------------------------------------------------------------------
const STEPS = [
  { num: 1, labelKey: 'steps.files' },
  { num: 2, labelKey: 'steps.pc' },
  { num: 3, labelKey: 'steps.preprocess' },
  { num: 4, labelKey: 'steps.phases' },
  { num: 5, labelKey: 'steps.method' },
  { num: 6, labelKey: 'steps.queue' },
  { num: 7, labelKey: 'steps.progress' },
];
const METHODS = [
  { value: 'hough', labelKey: 'methods.hough' },
  { value: 'spherical', labelKey: 'methods.spherical' },
  { value: 'dictionary', labelKey: 'methods.dictionary' },
];

// ---------------------------------------------------------------------------
function StepHeader({ currentStep, onStepClick }) {
  const { t } = useTranslation('batch');
  return (
    <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
      {STEPS.map((s) => (
        <button key={s.num} onClick={() => onStepClick(s.num)} title={t('nav.stepTabTooltip')} style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '4px 10px', borderRadius: 4, cursor: 'pointer',
          border: `1px solid ${s.num === currentStep ? colors.accent : colors.border}`,
          background: s.num === currentStep ? colors.accent : 'transparent',
          color: s.num === currentStep ? colors.textOnAccent : colors.textSecondary,
          fontSize: '9pt', fontWeight: s.num === currentStep ? 700 : 400,
        }}>
          <span style={{
            width: 18, height: 18, borderRadius: '50%', display: 'flex',
            alignItems: 'center', justifyContent: 'center', fontSize: '8pt',
            background: s.num === currentStep ? 'rgba(0,0,0,0.2)' : colors.bgSecondary,
          }}>{s.num}</span>
          {t(s.labelKey)}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
function PhaseChip({ label, detail, selected, onToggle, title }) {
  return (
    <button onClick={onToggle} title={title} style={{
      padding: '3px 8px', borderRadius: 10, fontSize: '9pt', cursor: 'pointer',
      border: `1px solid ${selected ? colors.accent : colors.border}`,
      background: selected ? colors.accent : colors.bgSecondary,
      color: selected ? colors.textOnAccent : colors.text,
      whiteSpace: 'nowrap',
    }}>
      {label}
      {detail && <span style={{ fontSize: '8pt', opacity: 0.7, marginLeft: 3 }}>{detail}</span>}
    </button>
  );
}

// ===========================================================================
export default function BatchWorkflow({ isActive = true }) {
  const { t } = useTranslation('batch');
  const store = useBatchConfigStore();
  const batchStore = useBatchStore();
  const { files, activeFileIndex, currentStep, method, globalPhases } = store;

  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState(null);
  const [discoveredPhases, setDiscoveredPhases] = useState([]);
  const [selectedPhaseIndices, setSelectedPhaseIndices] = useState(new Set());
  const [loadingPhases, setLoadingPhases] = useState(false);

  // Phase discovery. Fires on method change AND whenever this page
  // becomes visible again — BatchPage uses display:none so a newly
  // simulated SHT/CIF won't appear until we actively re-scan when the
  // user navigates back here from e.g. the Simulation module.
  const refreshPhases = useCallback(() => {
    setLoadingPhases(true);
    indexApi.discoverFiles(method)
      .then((res) => setDiscoveredPhases(res.data?.files ?? []))
      .catch(() => {/* keep what we had */})
      .finally(() => setLoadingPhases(false));
  }, [method]);

  useEffect(() => {
    setDiscoveredPhases([]);
    setSelectedPhaseIndices(new Set());
    refreshPhases();
  }, [method, refreshPhases]);

  useEffect(() => {
    if (isActive) refreshPhases();
  }, [isActive, refreshPhases]);

  // Sync selectedPhaseIndices with globalPhases
  useEffect(() => {
    if (globalPhases.length === 0 || discoveredPhases.length === 0) return;
    const sel = new Set();
    globalPhases.forEach((gp) => {
      const idx = discoveredPhases.findIndex((dp) => dp.path === gp.path);
      if (idx >= 0) sel.add(idx);
    });
    setSelectedPhaseIndices(sel);
  }, [globalPhases, discoveredPhases]);

  // --- Handlers ---
  const handleAddFolder = async () => {
    let target = '';
    if (window.electronAPI?.openFolder) {
      target = await window.electronAPI.openFolder();
      if (!target) return;
    } else {
      target = prompt(t('step1.enterFolderPrompt'));
      if (!target) return;
    }
    setScanning(true);
    setError(null);
    try {
      const res = await batchV2Api.scanFolder(target);
      const found = res.data?.files ?? [];
      const details = res.data?.details ?? [];
      if (found.length === 0) { setError(t('step1.noFilesFound')); return; }
      store.addFiles(found);
      // Stash scan details (incl. size_mb) into file metadata so the Export
      // size estimate can show an honest number before the batch starts.
      details.forEach((d) => {
        const idx = useBatchConfigStore.getState().files.findIndex((f) => f.file_path === d.path);
        if (idx >= 0) store.updateFileMetadata(idx, { size_mb: d.size_mb });
      });
      // Quick-load each for metadata
      for (const fp of found) {
        try {
          const m = await batchV2Api.quickLoad(fp);
          const d = m.data;
          const idx = useBatchConfigStore.getState().files.findIndex((f) => f.file_path === fp);
          if (idx < 0) continue;
          store.updateFileMetadata(idx, { loaded: true, eds_available: d.has_eds, eds_elements: d.eds_elements });
          if (d.pc) store.updateFilePCStatus(idx, { pc_value: d.pc, pc_status: d.pc_source || 'header' });
        } catch { /* skip */ }
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message || t('step1.scanFailed'));
    } finally {
      setScanning(false);
    }
  };

  const handleRefreshPC = async () => {
    try {
      const res = await batchV2Api.pcStatus(files.map((f) => f.file_name).join(','));
      (res.data || []).forEach((s) => {
        const idx = files.findIndex((f) => f.file_name === s.dataset_name);
        if (idx >= 0 && s.pc) store.updateFilePCStatus(idx, { pc_value: s.pc, pc_status: s.pc_source, pc_source_file: s.parent_name });
      });
    } catch { /* */ }
  };

  const handleApplyPhases = () => {
    const phases = [...selectedPhaseIndices].map((idx) => {
      const p = discoveredPhases[idx];
      return { name: p.formula || p.name || p.filename, path: p.path, method };
    });
    store.applyGlobalPhases(phases);
  };

  const handleStartBatch = async () => {
    setError(null);
    const ready = store.getReadyFiles();
    if (ready.length === 0) {
      setError(t('step6.noReadyFiles'));
      return;
    }
    const cfg = ready[0].config;
    const preprocessing = {
      frame_averaging: cfg.frame_averaging, frame_averaging_window: cfg.frame_averaging_window,
      background_removal: cfg.background_removal, background_method: cfg.background_method,
      gauss_background: cfg.gauss_background, circular_mask: cfg.circular_mask, nregions_ahe: cfg.nregions_ahe,
    };
    try {
      const postprocessing = {
        ci_threshold: cfg.postproc_ci_threshold || 0,
        uncertainty_threshold: cfg.postproc_uncertainty_threshold || 0,
        min_cluster_size: cfg.postproc_min_cluster_size || 0,
      };
      const data = await batchStore.createBatch(
        ready.map((f) => f.file_path), ready[0].config.phases,
        {
          auto_export: cfg.auto_export,
          export_dir: cfg.export_dir,
          export_formats: cfg.export_formats,
          // Opt-in: copy per-element counts + X/Y/Header from the source
          // h5oina into the light result so EDS-aware downstream tools
          // can work without touching the multi-GB source file.
          include_eds: !!cfg.include_eds_in_export,
          preprocessing,
          postprocessing,
        },
      );
      // Preflight gate: refuse to start if any check failed.
      // Delete the just-created orphan batch so the user can fix and retry
      // without polluting the batch list with a never-started shell.
      const pf = data?.preflight;
      if (pf && pf.can_start === false) {
        const failedChecks = (pf.checks || [])
          .filter((c) => c.status === 'fail')
          .map((c) => `${c.name}: ${c.message}`);
        try {
          await batchV2Api.deleteBatch(data.batch_id);
        } catch { /* best-effort cleanup */ }
        batchStore.reset();
        setError(
          t('step6.preflightFailed') +
          (failedChecks.length ? failedChecks.join('\n') : t('step6.preflightNoDetail')),
        );
        return;
      }
      await batchStore.startBatch();
      store.setStep(7);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || t('step6.startFailed'));
    }
  };

  // --- Step renderers ---
  const renderStep1 = () => (
    <GroupBox title={t('step1.title')}>
      <Button onClick={handleAddFolder} variant="primary" disabled={scanning} title={t('step1.addFolderTooltip')}>
        {scanning ? t('step1.scanning') : t('step1.addFolder')}
      </Button>
      <div style={{ marginTop: 8, fontSize: '9pt', color: colors.textSecondary }}>
        {files.length > 0 ? t('step1.filesLoaded', { count: files.length }) : t('step1.hint')}
      </div>
      {files.length > 0 && (
        <div style={{ maxHeight: 300, overflow: 'auto', marginTop: 8, fontSize: '9pt', border: `1px solid ${colors.border}`, borderRadius: 4, padding: 4 }}>
          {files.map((f) => (
            <div key={f.file_path} style={{ padding: '2px 4px', borderBottom: `1px solid ${colors.border}`, color: f.loaded ? colors.text : colors.textSecondary }}>
              {f.file_name}
              {f.loaded && <span style={{ color: '#50fa7b', marginLeft: 6, fontSize: '8pt' }}>{'\u2713'}</span>}
              {f.eds_available && <span style={{ color: colors.accent, marginLeft: 4, fontSize: '8pt' }}>EDS</span>}
            </div>
          ))}
        </div>
      )}
    </GroupBox>
  );

  const renderStep2 = () => (
    <GroupBox title={t('step2.title')}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <Button onClick={handleRefreshPC} variant="default" small title={t('step2.refreshPcTooltip')}>{t('step2.refreshPc')}</Button>
        <Button onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'pcrefinement' } }))} variant="primary" small title={t('step2.openPcRefinementTooltip')}>
          {t('step2.openPcRefinement')}
        </Button>
      </div>
      <div style={{ maxHeight: 250, overflow: 'auto', fontSize: '9pt' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr style={{ color: colors.accent, borderBottom: `1px solid ${colors.border}` }}>
            <th style={{ textAlign: 'left', padding: 3 }}>{t('step2.colFile')}</th>
            <th style={{ textAlign: 'left', padding: 3 }}>{t('step2.colPc')}</th>
            <th style={{ textAlign: 'left', padding: 3 }}>{t('step2.colSource')}</th>
          </tr></thead>
          <tbody>
            {files.map((f) => (
              <tr key={f.file_path} style={{ borderBottom: `1px solid ${colors.border}`, color: colors.text }}>
                <td style={{ padding: 3, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={f.file_name}>{f.file_name}</td>
                <td style={{ padding: 3, fontFamily: 'monospace', fontSize: '8pt' }}>
                  {f.pc_value ? f.pc_value.map((v) => v.toFixed(3)).join(', ') : '\u2014'}
                </td>
                <td style={{ padding: 3, color: f.pc_status === 'refined' ? '#50fa7b' : f.pc_status === 'inherited' ? colors.accent : colors.textSecondary }}>
                  {f.pc_status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8, fontSize: '9pt', color: colors.textSecondary }}>
        {t('step2.footer')}
      </div>
    </GroupBox>
  );

  const renderStep3 = () => {
    const cfg = files[0]?.config || {};
    const totalSourceMb = files.reduce((s, f) => {
      // Source size from quick-load metadata if available; otherwise skip estimate
      return s + (f.size_mb || 0);
    }, 0);

    const toggleFormat = (fmt, on) => {
      const current = new Set(cfg.export_formats || []);
      if (on) current.add(fmt); else current.delete(fmt);
      files.forEach((_, i) => store.updateFileConfig(i, { export_formats: [...current] }));
    };
    const formats = new Set(cfg.export_formats || []);

    // Size estimate (MB) — matches backend preflight math
    const nPhases = cfg.phases?.length || 0;
    const nFiles = files.length;
    let estMb = 0;
    if (cfg.auto_export) {
      if (formats.has('h5_rich'))  estMb += totalSourceMb * 1.05;
      if (formats.has('h5_light')) estMb += (nPhases * 5 + 5) * nFiles;
      if (formats.has('ang'))      estMb += 20 * nFiles;
      if (formats.has('ctf'))      estMb += 20 * nFiles;
    }
    const estStr = estMb >= 1024
      ? `${(estMb / 1024).toFixed(1)} GB`
      : `${estMb.toFixed(0)} MB`;

    return (
      <>
      <GroupBox title={t('step3.preprocessingTitle')}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '10pt', color: colors.text }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} title={t('step3.frameAveragingTooltip')}>
            <input type="checkbox" checked={cfg.frame_averaging || false}
              onChange={(e) => store.applyGlobalPreprocessing({ frame_averaging: e.target.checked })}
              style={{ accentColor: colors.accent }} />
            {t('step3.frameAveraging')}
            {cfg.frame_averaging && (
              <select value={cfg.frame_averaging_window || 3} title={t('step3.frameAveragingWindowTooltip')}
                onChange={(e) => store.applyGlobalPreprocessing({ frame_averaging_window: parseInt(e.target.value) })}
                style={{ background: colors.bg, border: `1px solid ${colors.border}`, color: colors.text, borderRadius: 4, padding: '2px 6px', marginLeft: 8 }}>
                <option value={3}>{t('step3.window3')}</option><option value={5}>{t('step3.window5')}</option><option value={7}>{t('step3.window7')}</option>
              </select>
            )}
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} title={t('step3.backgroundRemovalTooltip')}>
            <input type="checkbox" checked={cfg.background_removal || false}
              onChange={(e) => store.applyGlobalPreprocessing({ background_removal: e.target.checked })}
              style={{ accentColor: colors.accent }} />
            {t('step3.backgroundRemoval')}
            {cfg.background_removal && (
              <select value={cfg.background_method || 'dynamic'} title={t('step3.backgroundMethodTooltip')}
                onChange={(e) => store.applyGlobalPreprocessing({ background_method: e.target.value })}
                style={{ background: colors.bg, border: `1px solid ${colors.border}`, color: colors.text, borderRadius: 4, padding: '2px 6px', marginLeft: 8 }}>
                <option value="dynamic">{t('step3.dynamic')}</option><option value="static">{t('step3.static')}</option>
              </select>
            )}
          </label>
        </div>
        <p style={{ fontSize: '9pt', color: colors.textSecondary, marginTop: 8 }}>
          {t('step3.preprocessingNote')}
        </p>
      </GroupBox>

      <GroupBox title={t('step3.postprocessingTitle')}>
        <p style={{ fontSize: '9pt', color: colors.textSecondary, margin: '0 0 8px 0' }}>
          <Trans i18nKey="step3.postprocessingIntro" t={t} components={{ code: <code />, b: <b /> }} />
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '160px 1fr 70px', gap: '6px 10px', fontSize: '10pt', alignItems: 'center', color: colors.text }}>
          <span>{t('step3.ciThreshold')}</span>
          <input
            type="range" min="0" max="1" step="0.01"
            title={t('step3.ciThresholdTooltip')}
            value={cfg.postproc_ci_threshold || 0}
            onChange={(e) => files.forEach((_, i) => store.updateFileConfig(i, { postproc_ci_threshold: parseFloat(e.target.value) }))}
            style={{ accentColor: colors.accent, width: '100%' }}
          />
          <span style={{ fontFamily: 'monospace', fontSize: '9pt' }}>{(cfg.postproc_ci_threshold || 0).toFixed(2)}</span>

          <span>{t('step3.uncertaintyThreshold')}</span>
          <input
            type="range" min="0" max="0.5" step="0.01"
            title={t('step3.uncertaintyThresholdTooltip')}
            value={cfg.postproc_uncertainty_threshold || 0}
            onChange={(e) => files.forEach((_, i) => store.updateFileConfig(i, { postproc_uncertainty_threshold: parseFloat(e.target.value) }))}
            style={{ accentColor: colors.accent, width: '100%' }}
          />
          <span style={{ fontFamily: 'monospace', fontSize: '9pt' }}>{(cfg.postproc_uncertainty_threshold || 0).toFixed(2)}</span>

          <span>{t('step3.minClusterSize')}</span>
          <input
            type="range" min="0" max="50" step="1"
            title={t('step3.minClusterSizeTooltip')}
            value={cfg.postproc_min_cluster_size || 0}
            onChange={(e) => files.forEach((_, i) => store.updateFileConfig(i, { postproc_min_cluster_size: parseInt(e.target.value, 10) }))}
            style={{ accentColor: colors.accent, width: '100%' }}
          />
          <span style={{ fontFamily: 'monospace', fontSize: '9pt' }}>{cfg.postproc_min_cluster_size || 0}</span>
        </div>
        <p style={{ fontSize: '9pt', color: colors.textSecondary, margin: '6px 0 0 0' }}>
          {t('step3.postprocessingRecommendation')}
        </p>
      </GroupBox>

      <GroupBox title={t('step3.exportTitle')}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '10pt' }} title={t('step3.autoExportTooltip')}>
          <input type="checkbox" checked={cfg.auto_export !== false}
            onChange={(e) => files.forEach((_, i) => store.updateFileConfig(i, { auto_export: e.target.checked }))}
            style={{ accentColor: colors.accent }} />
          {t('step3.autoExport')}
        </label>

        {cfg.auto_export !== false && (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 8, fontSize: '10pt' }}>
              <label style={{ display: 'flex', gap: 8, cursor: 'pointer' }} title={t('step3.lightH5Tooltip')}>
                <input type="checkbox" checked={formats.has('h5_light')}
                  onChange={(e) => toggleFormat('h5_light', e.target.checked)}
                  style={{ accentColor: colors.accent }} />
                <span>
                  <b>{t('step3.lightH5')}</b>
                  <span style={{ color: colors.textSecondary, marginLeft: 6 }}>
                    {t('step3.lightH5Desc')}
                  </span>
                </span>
              </label>
              <label style={{ display: 'flex', gap: 8, cursor: 'pointer' }} title={t('step3.angTooltip')}>
                <input type="checkbox" checked={formats.has('ang')}
                  onChange={(e) => toggleFormat('ang', e.target.checked)}
                  style={{ accentColor: colors.accent }} />
                <span>
                  <b>{t('step3.ang')}</b>
                  <span style={{ color: colors.textSecondary, marginLeft: 6 }}>{t('step3.angDesc')}</span>
                </span>
              </label>
              <label style={{ display: 'flex', gap: 8, cursor: 'pointer' }} title={t('step3.ctfTooltip')}>
                <input type="checkbox" checked={formats.has('ctf')}
                  onChange={(e) => toggleFormat('ctf', e.target.checked)}
                  style={{ accentColor: colors.accent }} />
                <span>
                  <b>{t('step3.ctf')}</b>
                  <span style={{ color: colors.textSecondary, marginLeft: 6 }}>{t('step3.ctfDesc')}</span>
                </span>
              </label>
              <label style={{ display: 'flex', gap: 8, cursor: 'pointer' }} title={t('step3.richH5Tooltip')}>
                <input type="checkbox" checked={formats.has('h5_rich')}
                  onChange={(e) => toggleFormat('h5_rich', e.target.checked)}
                  style={{ accentColor: colors.accent }} />
                <span>
                  <b style={{ color: colors.yellow }}>{t('step3.richH5')}</b>
                  <span style={{ color: colors.textSecondary, marginLeft: 6 }}>
                    {t('step3.richH5Desc')}
                  </span>
                  {totalSourceMb > 0 && formats.has('h5_rich') && (
                    <span style={{ color: colors.red, marginLeft: 6, fontWeight: 600 }}>
                      {t('step3.richH5SourceEstimate', { gb: (totalSourceMb / 1024).toFixed(1) })}
                    </span>
                  )}
                </span>
              </label>
            </div>

            {/* EDS copy-through: copies per-element Window Integral + X/Y
                + Header from the source h5oina into the light .h5. Raw
                Spectrum is intentionally skipped (multi-GB on big maps). */}
            {formats.has('h5_light') && (
              <label style={{
                display: 'flex', gap: 8, cursor: 'pointer', marginTop: 8,
                fontSize: '10pt', paddingLeft: 24,
                borderLeft: `2px solid ${colors.border}`,
              }} title={t('step3.includeEdsTooltip')}>
                <input type="checkbox" checked={!!cfg.include_eds_in_export}
                  onChange={(e) => files.forEach((_, i) =>
                    store.updateFileConfig(i, { include_eds_in_export: e.target.checked }))}
                  style={{ accentColor: colors.accent }} />
                <span>
                  {t('step3.includeEds')}
                  <span style={{ color: colors.textSecondary, marginLeft: 6 }}>
                    {t('step3.includeEdsDesc')}
                  </span>
                </span>
              </label>
            )}

            <div style={{ marginTop: 10, fontSize: '10pt', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: colors.textSecondary }}>{t('step3.outputDir')}</span>
              <Input
                value={cfg.export_dir || ''}
                placeholder={t('step3.outputDirPlaceholder')}
                title={t('step3.outputDirTooltip')}
                onChange={(e) => files.forEach((_, i) => store.updateFileConfig(i, { export_dir: e.target.value }))}
              />
              <Button
                variant="default"
                small
                title={t('step3.browseTooltip')}
                onClick={async () => {
                  if (window.electronAPI?.openFolder) {
                    const p = await window.electronAPI.openFolder();
                    if (p) files.forEach((_, i) => store.updateFileConfig(i, { export_dir: p }));
                  } else {
                    const p = prompt(t('step3.outputDirPrompt'));
                    if (p) files.forEach((_, i) => store.updateFileConfig(i, { export_dir: p }));
                  }
                }}
              >
                {t('step3.browse')}
              </Button>
            </div>

            <div style={{
              marginTop: 10, padding: '6px 10px', borderRadius: 4,
              background: estMb > 10240 ? `${colors.red}22` : `${colors.bg}`,
              border: `1px solid ${estMb > 10240 ? colors.red : colors.border}`,
              fontSize: '9pt', color: estMb > 10240 ? colors.red : colors.text,
            }}>
              <b>{t('step3.estTotalOutput')}</b> {estStr}
              {' '}{t('step3.estForFiles', { files: nFiles, formats: formats.size })}
              {estMb > 10240 && t('step3.estUncheckRich')}
            </div>
          </>
        )}
      </GroupBox>
      </>
    );
  };

  const renderStep4 = () => (
    <>
      <GroupBox title={t('step4.methodTitle')}>
        <div style={{ display: 'flex', gap: 6 }}>
          {METHODS.map((m) => (
            <button key={m.value} onClick={() => store.setMethod(m.value)} title={t('step4.methodTooltip')} style={{
              flex: 1, padding: '6px', borderRadius: 6, fontSize: '10pt', cursor: 'pointer',
              border: `1px solid ${method === m.value ? colors.accent : colors.border}`,
              background: method === m.value ? colors.accent : 'transparent',
              color: method === m.value ? colors.textOnAccent : colors.text,
            }}>{t(m.labelKey)}</button>
          ))}
        </div>
      </GroupBox>
      <GroupBox title={t('step4.availablePhasesTitle')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <Button small onClick={refreshPhases} disabled={loadingPhases}
            title={t('step4.refreshTooltip')}>
            {loadingPhases ? t('step4.refreshScanning') : t('step4.refresh')}
          </Button>
          <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
            {t('step4.autoRescanNote')}
            {discoveredPhases.length > 0 && t('step4.phasesFound', { count: discoveredPhases.length })}
          </span>
        </div>
        {loadingPhases && discoveredPhases.length === 0 ? <span style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('step4.loading')}</span> :
         discoveredPhases.length === 0 ? <span style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('step4.noPhaseFiles')}</span> : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {discoveredPhases.map((p, idx) => (
              <PhaseChip key={p.path} label={p.formula || p.name || p.filename} detail={p.space_group}
                title={t('step4.phaseChipTooltip')}
                selected={selectedPhaseIndices.has(idx)}
                onToggle={() => setSelectedPhaseIndices((prev) => { const n = new Set(prev); n.has(idx) ? n.delete(idx) : n.add(idx); return n; })} />
            ))}
          </div>
        )}
        {selectedPhaseIndices.size > 0 && (
          <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('step4.selected', { count: selectedPhaseIndices.size })}</span>
            <Button onClick={handleApplyPhases} variant="primary" small title={t('step4.applyToAllTooltip')}>{t('step4.applyToAll')}</Button>
          </div>
        )}
      </GroupBox>
    </>
  );

  const renderStep5 = () => {
    const cfg = files[0]?.config || {};
    const u = (patch) => files.forEach((_, i) => store.updateFileConfig(i, patch));
    return (
      <>
        {method === 'spherical' && <GroupBox title={t('step5.sphericalTitle')}>
          <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '6px 12px', fontSize: '10pt', color: colors.text, alignItems: 'center' }}>
            <span>{t('step5.bandwidth')}</span><Input value={cfg.bandwidth ?? 88} type="number" title={t('step5.bandwidthTooltip')} onChange={(e) => u({ bandwidth: +e.target.value || 88 })} />
            <span>{t('step5.nRegions')}</span><Input value={cfg.nregions ?? 10} type="number" title={t('step5.nRegionsTooltip')} onChange={(e) => u({ nregions: +e.target.value || 10 })} />
            <span>{t('step5.refine')}</span><input type="checkbox" checked={cfg.refine ?? true} title={t('step5.refineTooltip')} onChange={(e) => u({ refine: e.target.checked })} style={{ accentColor: colors.accent }} />
          </div>
        </GroupBox>}
        {method === 'dictionary' && <GroupBox title={t('step5.dictionaryTitle')}>
          <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '6px 12px', fontSize: '10pt', color: colors.text, alignItems: 'center' }}>
            <span>{t('step5.keepN')}</span><Input value={cfg.keep_n ?? 20} type="number" title={t('step5.keepNTooltip')} onChange={(e) => u({ keep_n: +e.target.value || 20 })} />
            <span>{t('step5.metric')}</span>
            <select value={cfg.metric || 'ncc'} title={t('step5.metricTooltip')} onChange={(e) => u({ metric: e.target.value })} style={{ background: colors.bg, border: `1px solid ${colors.border}`, color: colors.text, borderRadius: 4, padding: '2px 6px' }}>
              <option value="ncc">{t('step5.ncc')}</option><option value="ndp">{t('step5.ndp')}</option>
            </select>
          </div>
        </GroupBox>}
        {method === 'hough' && <GroupBox title={t('step5.houghTitle')}>
          <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '6px 12px', fontSize: '10pt', color: colors.text, alignItems: 'center' }}>
            <span>{t('step5.nBands')}</span><Input value={cfg.n_bands ?? 12} type="number" title={t('step5.nBandsTooltip')} onChange={(e) => u({ n_bands: +e.target.value || 12 })} />
          </div>
        </GroupBox>}
      </>
    );
  };

  const renderStep6 = () => {
    const ready = files.filter((f) => f.ready);
    const totalJobs = ready.reduce((s, f) => s + f.config.phases.length, 0);
    return (
      <GroupBox title={t('step6.title')}>
        <div style={{ maxHeight: 250, overflow: 'auto', fontSize: '9pt' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr style={{ color: colors.accent, borderBottom: `1px solid ${colors.border}` }}>
              <th style={{ textAlign: 'left', padding: 3 }}>{t('step6.colFile')}</th>
              <th style={{ textAlign: 'center', padding: 3 }}>{t('step6.colPc')}</th>
              <th style={{ textAlign: 'center', padding: 3 }}>{t('step6.colPhases')}</th>
              <th style={{ textAlign: 'center', padding: 3 }}>{t('step6.colOk')}</th>
            </tr></thead>
            <tbody>{files.map((f) => (
              <tr key={f.file_path} style={{ borderBottom: `1px solid ${colors.border}`, color: colors.text }}>
                <td style={{ padding: 3, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.file_name}</td>
                <td style={{ padding: 3, textAlign: 'center', color: f.pc_status !== 'none' ? '#50fa7b' : '#ff5555' }}>{f.pc_status !== 'none' ? '\u2713' : '\u2717'}</td>
                <td style={{ padding: 3, textAlign: 'center' }}>{f.config.phases.length}</td>
                <td style={{ padding: 3, textAlign: 'center', color: f.ready ? '#50fa7b' : '#ff5555' }}>{f.ready ? '\u2713' : '\u2717'}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '10pt', color: colors.text }}>{t('step6.readySummary', { ready: ready.length, total: files.length, jobs: totalJobs })}</span>
          <Button onClick={handleStartBatch} variant="primary" disabled={ready.length === 0} title={t('step6.startBatchTooltip')}>
            {t('step6.startBatch')}
          </Button>
        </div>
        {error && <div style={{ color: '#ff5555', fontSize: '10pt', marginTop: 6, whiteSpace: 'pre-line' }}>{error}</div>}
      </GroupBox>
    );
  };

  const renderStep7 = () => batchStore.batchId
    ? <BatchDashboard />
    : <div style={{ color: colors.textSecondary, textAlign: 'center', marginTop: 40 }}>{t('step7.noBatchRunning')}</div>;

  const stepRenderers = { 1: renderStep1, 2: renderStep2, 3: renderStep3, 4: renderStep4, 5: renderStep5, 6: renderStep6, 7: renderStep7 };

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      <div style={{ width: 260, minWidth: 260, borderRight: `1px solid ${colors.border}`, overflow: 'auto', background: colors.bg }}>
        <BatchFileTree />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: spacing.outerMargin, overflow: 'auto' }}>
        <StepHeader currentStep={currentStep} onStepClick={store.setStep} />
        <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {stepRenderers[currentStep]?.()}
        </div>
        {currentStep < 7 && (
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, paddingTop: 8, borderTop: `1px solid ${colors.border}` }}>
            <Button onClick={() => store.setStep(Math.max(1, currentStep - 1))} variant="default" disabled={currentStep <= 1} title={t('nav.previousTooltip')}>
              {'\u2190'} {t('nav.previous')}
            </Button>
            <Button onClick={() => store.setStep(Math.min(7, currentStep + 1))} variant="primary" disabled={currentStep === 1 && files.length === 0} title={t('nav.nextTooltip')}>
              {t('nav.next')} {'\u2192'}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
