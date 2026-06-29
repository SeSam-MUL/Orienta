import { useEffect, useRef } from 'react';
import { useTranslation, Trans } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import { Button, GroupBox, ProgressBar } from '../../theme/components';
import useBatchStore from '../../stores/useBatchStore';
import JobTable from './JobTable';
import PreFlightReport from './PreFlightReport';

// ---------------------------------------------------------------------------
// RAM gauge — shows process RSS and system percent
// ---------------------------------------------------------------------------
function RamGauge({ memory }) {
  const { t } = useTranslation('batch');
  if (!memory) return null;
  const pct = memory.percent ?? 0;
  const rss  = memory.process_rss_mb != null ? `${memory.process_rss_mb.toFixed(0)} MB` : null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9pt', color: colors.textSecondary }}>
        <span>{t('dashboard.ram')}</span>
        <span>{rss && t('dashboard.ramProcess', { rss })}{t('dashboard.ramSystem', { pct: pct.toFixed(1) })}</span>
      </div>
      <ProgressBar
        value={pct}
        max={100}
        color={pct > 85 ? colors.red : pct > 65 ? colors.yellow : colors.cyan}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// ETA label
// ---------------------------------------------------------------------------
function EtaLabel({ jobs, completed, total }) {
  const { t } = useTranslation('batch');
  if (completed <= 0 || total <= 0) return null;
  const doneDurations = jobs
    .filter((j) => j.status === 'done' && j.duration_sec != null)
    .map((j) => j.duration_sec);
  if (doneDurations.length === 0) return null;
  const avg = doneDurations.reduce((a, b) => a + b, 0) / doneDurations.length;
  const remaining = total - completed;
  const etaSec = Math.round(avg * remaining);
  const minutes = Math.floor(etaSec / 60);
  const seconds = etaSec % 60;
  return (
    <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
      {t('dashboard.eta', { minutes: minutes > 0 ? t('dashboard.etaMinutes', { minutes }) : '', seconds })}
    </span>
  );
}

// ---------------------------------------------------------------------------
// BatchDashboard
// ---------------------------------------------------------------------------
export default function BatchDashboard({ onNavigateToRefinement }) {
  const { t } = useTranslation('batch');
  const {
    status, jobs, memory, totalJobs, completed, failed,
    batchId, preflight, connectionError, preprocessing, config,
    pauseBatch, resumeBatch, stopBatch,
    startPolling, stopPolling,
  } = useBatchStore();

  const notifiedRef = useRef(false);
  const rehydrate = useBatchStore((s) => s.rehydrate);

  // Start polling when mounted, and rehydrate from backend if we lost
  // in-memory state (e.g. page refresh, app restart) but have a batchId.
  useEffect(() => {
    if (batchId && status === null) {
      // Stored batchId from localStorage but no live state — try to rehydrate
      rehydrate();
    } else if (batchId && (status === 'running' || status === 'paused')) {
      startPolling();
    }
    return () => stopPolling();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [batchId]);

  // System notification on completion
  useEffect(() => {
    if ((status === 'completed' || status === 'failed') && !notifiedRef.current) {
      notifiedRef.current = true;
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(t('dashboard.notification.title'), {
          body: status === 'completed'
            ? t('dashboard.notification.completed', { completed, total: totalJobs })
            : t('dashboard.notification.failed', { failed }),
        });
      }
    }
  }, [status, completed, totalJobs, failed]);

  const progressPct = totalJobs > 0 ? Math.round((completed / totalJobs) * 100) : 0;

  const currentJob = jobs.find((j) => j.status === 'running') ?? null;

  const statusColor = {
    running:   colors.cyan,
    paused:    colors.yellow,
    completed: colors.green,
    failed:    colors.red,
    pending:   colors.textSecondary,
  }[status] ?? colors.textSecondary;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.groupSpacing, height: '100%', overflow: 'auto' }}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '13pt', fontWeight: 700, color: colors.accent }}>{t('dashboard.title')}</span>
          <span style={{
            fontSize: '9pt', fontWeight: 600, padding: '2px 8px',
            borderRadius: 10, background: `${statusColor}22`, color: statusColor,
          }}>
            {status ?? t('dashboard.statusIdle')}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {status === 'running' && (
            <Button onClick={pauseBatch} variant="warning" small title={t('hoverTips.pauseBatch')}>{t('dashboard.pause')}</Button>
          )}
          {status === 'paused' && (
            <Button onClick={resumeBatch} variant="primary" small title={t('hoverTips.resumeBatch')}>{t('dashboard.resume')}</Button>
          )}
          {(status === 'running' || status === 'paused') && (
            <Button onClick={stopBatch} variant="danger" small title={t('hoverTips.stopBatch')}>{t('dashboard.stop')}</Button>
          )}
          {(status === 'completed' || status === 'failed') && batchId && (
            <Button
              onClick={() => {
                const url = `${window.location.origin}/api/batch-v2/${batchId}/report`;
                if (window.electronAPI?.openExternal) window.electronAPI.openExternal(url);
                else window.open(url, '_blank');
              }}
              variant="default"
              small
              title={t('hoverTips.viewReport')}
            >
              {t('dashboard.viewReport')}
            </Button>
          )}
        </div>
      </div>

      {/* Connection error banner */}
      {connectionError && (
        <div style={{
          padding: '8px 12px', borderRadius: 4,
          background: `${colors.red}22`, border: `1px solid ${colors.red}55`,
          color: colors.red, fontSize: '10pt',
        }}>
          {t('dashboard.connectionError')}
        </div>
      )}

      {/* Preflight report (if any check warned/failed) */}
      {preflight && preflight.checks?.some((c) => c.status !== 'pass') && (
        <GroupBox title={t('dashboard.preflightTitle')}>
          <PreFlightReport checks={preflight.checks} />
        </GroupBox>
      )}

      {/* Overall progress */}
      <GroupBox title={t('dashboard.overallProgressTitle')}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <span style={{ fontSize: '10pt', color: colors.text }}>
            {t('dashboard.jobsProgress', { completed, total: totalJobs })}  ({failed > 0 ? <span style={{ color: colors.red }}>{t('dashboard.failedCount', { count: failed })}</span> : t('dashboard.zeroFailed')})
          </span>
          <EtaLabel jobs={jobs} completed={completed} total={totalJobs} />
        </div>
        <ProgressBar value={completed} max={totalJobs || 1} label={`${progressPct}%`} />
      </GroupBox>

      {/* RAM gauge */}
      <GroupBox title={t('dashboard.memoryTitle')}>
        <RamGauge memory={memory} />
      </GroupBox>

      {/* Preprocessing applied — visible proof that frame averaging / BG
          removal actually ran (or was skipped, with reason). */}
      {preprocessing && Object.keys(preprocessing).length > 0 && (
        <GroupBox title={t('dashboard.preprocessingAppliedTitle')}>
          <div style={{ fontSize: '10pt', color: colors.text }}>
            {Object.entries(preprocessing).map(([key, values]) => (
              <div key={key} style={{ marginBottom: 3 }}>
                <span style={{ color: colors.accent, fontWeight: 600 }}>{key}:</span>{' '}
                <span style={{
                  color: values.some((v) => v.startsWith('FAILED') || v === 'spherical-bypass')
                    ? colors.yellow
                    : colors.green,
                }}>
                  {values.join(', ')}
                </span>
              </div>
            ))}
          </div>
        </GroupBox>
      )}

      {/* Current job */}
      {currentJob && (
        <GroupBox title={t('dashboard.currentJobTitle')}>
          <div style={{ fontSize: '10pt', color: colors.text }}>
            <span style={{ color: colors.textSecondary }}>{t('dashboard.fileLabel')}</span>{currentJob.file_name}
          </div>
          <div style={{ fontSize: '10pt', color: colors.text }}>
            <span style={{ color: colors.textSecondary }}>{t('dashboard.phaseLabel')}</span>{currentJob.phase_name}
          </div>
        </GroupBox>
      )}

      {/* Job table */}
      <GroupBox title={t('dashboard.allJobsTitle', { count: jobs.length })} style={{ flex: 1 }}>
        {jobs.length === 0 ? (
          <span style={{ fontSize: '10pt', color: colors.textSecondary }}>{t('dashboard.noJobsYet')}</span>
        ) : (
          <JobTable jobs={jobs} />
        )}
      </GroupBox>

      {/* Per-file open-in buttons (visible after completion) */}
      {status === 'completed' && jobs.length > 0 && (
        <ResultExplorer
          jobs={jobs}
          exportDir={config?.export_dir || ''}
          exportFormats={config?.export_formats}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ResultExplorer — pick a completed file and open it in another module.
// Hands off via sessionStorage (same mechanism as Dashboard's "recent files"
// → EBSDViewer auto-load), targeting the *_light.h5 export so we don't
// reload multi-GB rich files.
// ---------------------------------------------------------------------------
function ResultExplorer({ jobs, exportDir, exportFormats }) {
  const { t } = useTranslation('batch');
  // Which file extensions did the batch actually write? When the user
  // ticked only .ctf in the setup, handing off `result_<stem>.ang` sends
  // Phase Map to a nonexistent file and the open fails with
  // "No filename matches...". Pick the first format that was written.
  const formats = Array.isArray(exportFormats) && exportFormats.length > 0
    ? exportFormats
    : ['h5_light', 'ang', 'ctf'];
  const pickedFormat = ['ang', 'h5_light', 'ctf'].find((f) => formats.includes(f)) || 'ang';
  const formatSuffix = {
    ang: (stem) => `result_${stem}.ang`,
    ctf: (stem) => `result_${stem}.ctf`,
    h5_light: (stem) => `result_${stem}_light.h5`,
  }[pickedFormat];
  // Group jobs by file_path (one open-in row per source file)
  const fileMap = new Map();
  jobs.forEach((j) => {
    if (!fileMap.has(j.file_path)) {
      fileMap.set(j.file_path, { name: j.file_name, allDone: true });
    }
    if (j.status !== 'done' && j.status !== 'skipped') {
      fileMap.get(j.file_path).allDone = false;
    }
  });
  const files = [...fileMap.entries()]
    .filter(([, v]) => v.allDone)
    .map(([path, v]) => ({ path, name: v.name }));

  if (files.length === 0) return null;

  // Build paths per format. All exports sit next to source (or in
  // export_dir, but we don't know that here without re-reading config).
  // Convention from result_exporter.py:
  //   .ang/.ctf  →  result_<stem>.ang / .ctf
  //   light h5   →  result_<stem>_light.h5
  // Refinement reads the _multiphase.h5 checkpoint directly via stem.
  const splitPath = (p) => {
    const parts = p.replace(/\\/g, '/').split('/');
    const filename = parts.pop();
    const stem = filename.replace(/\.[^.]+$/, '');
    return { dir: parts.join('/'), stem };
  };

  // Exports land in either the per-batch export_dir (when set) or next to
  // the source file (default). This must mirror result_exporter.py's
  // behaviour, otherwise the handoff 404s silently and the user sees
  // "No filename matches..." in Phase Map.
  const resolveExportDir = (sourceDir) => {
    const ed = (exportDir || '').trim().replace(/\\/g, '/');
    return ed.length > 0 ? ed : sourceDir;
  };

  const open = (target, sourcePath) => {
    const { dir, stem } = splitPath(sourcePath);
    const outDir = resolveExportDir(dir);
    let path = '';
    if (target === 'refinement') {
      // Refinement reads the <stem>_multiphase.h5 checkpoint. The checkpoint
      // is ALWAYS written next to the source (not in export_dir) — see
      // CheckpointWriter._derive_path — so keep using the source dir here.
      try {
        sessionStorage.setItem('refinement_handoff_stem', stem);
        sessionStorage.setItem('refinement_handoff_dir', dir);
      } catch { /* unavailable */ }
    } else if (target === 'analysis' || target === 'phasemap') {
      // Analysis backend uses orix.io.load() which understands .ang / .ctf
      // natively and falls back to our custom .h5 reader. Hand off whichever
      // format the batch actually wrote — .ang is preferred for MTEX, but
      // if the user unticked it we use h5_light / .ctf instead so the
      // hand-off doesn't 404 on a file that was never written.
      path = `${outDir}/${formatSuffix(stem)}`;
      try {
        sessionStorage.setItem('batch_handoff_path', path);
        sessionStorage.setItem('batch_handoff_source', sourcePath);
        // Persist so the "Apply to checkpoint" button in Phase Map still
        // knows which _multiphase.h5 to write to after a page reload.
        localStorage.setItem('last_batch_source', sourcePath);
      } catch { /* unavailable */ }
    }
    window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: target } }));
  };

  const displayExportDir = (exportDir || '').trim();
  return (
    <GroupBox title={t('results.title')}>
      <div style={{ fontSize: '10pt', color: colors.textSecondary, marginBottom: 8 }}>
        <Trans i18nKey="results.intro" t={t} components={{ code: <code /> }} />
        {displayExportDir && (
          <>
            <br/>
            <span style={{ color: colors.accent }}>{t('results.exportDirLabel')}</span>
            <code style={{ fontSize: '9pt' }}>{displayExportDir}</code>
          </>
        )}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10pt' }}>
        <thead>
          <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
            <th style={{ textAlign: 'left', padding: '4px 6px' }}>{t('results.colFile')}</th>
            <th style={{ textAlign: 'right', padding: '4px 6px' }}>{t('results.colOpenIn')}</th>
          </tr>
        </thead>
        <tbody>
          {files.map((f) => (
            <tr key={f.path} style={{ borderBottom: `1px solid ${colors.border}22` }}>
              <td
                style={{
                  padding: '4px 6px', color: colors.text,
                  maxWidth: 320, overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}
                title={f.path}
              >
                {f.name}
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'right', display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                <Button onClick={() => open('refinement', f.path)} variant="default" small title={t('hoverTips.openRefinement')}>{t('results.refinement')}</Button>
                <Button onClick={() => open('analysis',   f.path)} variant="default" small title={t('hoverTips.openAnalysis')}>{t('results.analysis')}</Button>
                <Button onClick={() => open('phasemap',   f.path)} variant="default" small title={t('hoverTips.openPhaseMap')}>{t('results.phaseMap')}</Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </GroupBox>
  );
}
