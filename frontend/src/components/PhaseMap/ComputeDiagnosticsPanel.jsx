/**
 * ComputeDiagnosticsPanel - bandwidth dropdown + Compute button + progress UI.
 *
 * Lives in PhaseMapPage above the layer stack.
 * Uses forwardDiagApi.{compute, progress, summary, cancel}.
 */
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import useDataStore from '../../stores/useDataStore';
import { forwardDiagApi } from '../../services/api';
import { colors, spacing, Button, Select } from '../../theme/components';

export default function ComputeDiagnosticsPanel({ resultId, onOpenBrowser }) {
  const { t } = useTranslation('phasemap');
  const BANDWIDTH_OPTIONS = [
    { value: 128, label: t('phasemap:diagnostics.bwFast') },
    { value: 256, label: t('phasemap:diagnostics.bwStandard') },
    { value: 384, label: t('phasemap:diagnostics.bwSharp') },
  ];
  const [bandwidth, setBandwidth] = useState(256);
  const [error, setError] = useState(null);
  const job = useDataStore((s) => s.diagnosticsJob);
  const computedMap = useDataStore((s) => s.diagnosticsComputed);
  const setJob = useDataStore((s) => s.setDiagnosticsJob);
  const setComputed = useDataStore((s) => s.setDiagnosticsComputed);

  const computed = resultId ? computedMap[resultId] : null;
  const pollRef = useRef(null);

  // Poll progress while a job is running for this result.
  // We read the latest job snapshot inside the interval via useDataStore.getState()
  // so the deps array can stay minimal (no infinite-loop trigger from setJob).
  useEffect(() => {
    if (!job || job.result_id !== resultId || job.state !== 'running') {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return undefined;
    }
    const jobId = job.job_id;
    pollRef.current = setInterval(async () => {
      try {
        const p = await forwardDiagApi.progress(jobId);
        const currentJob = useDataStore.getState().diagnosticsJob;
        if (currentJob && currentJob.job_id === jobId) {
          setJob({ ...currentJob, ...p });
        }
        if (p.state === 'completed') {
          const s = await forwardDiagApi.summary(resultId);
          if (s) setComputed(resultId, { bandwidth: s.bandwidth, computed_at: s.computed_at });
          setJob(null);
        }
        if (p.state === 'failed' || p.state === 'cancelled') {
          if (p.error) setError(p.error);
          setJob(null);
        }
      } catch (e) {
        setError(e.message || String(e));
        setJob(null);
      }
    }, 500);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.job_id, job?.state, job?.result_id, resultId]);

  const onCompute = async () => {
    if (!resultId) return;
    setError(null);
    try {
      const forceRecompute = !!(computed && computed.bandwidth !== bandwidth);
      const r = await forwardDiagApi.compute(resultId, bandwidth, forceRecompute);
      if (r.already_computed) {
        setComputed(resultId, {
          bandwidth: r.existing_bandwidth,
          computed_at: r.computed_at,
        });
        if (r.warning) setError(r.warning);
        return;
      }
      setJob({
        result_id: resultId,
        job_id: r.job_id,
        state: 'running',
        progress: 0,
        current_phase: '',
      });
    } catch (e) {
      setError(e.message || String(e));
    }
  };

  const onCancel = async () => {
    if (!job) return;
    try {
      await forwardDiagApi.cancel(job.job_id);
    } catch {
      /* swallow — we still clear local job state below */
    }
    setJob(null);
  };

  const isRunning = !!(job && job.result_id === resultId && job.state === 'running');
  const progressPct = Math.round((job?.progress || 0) * 100);

  return (
    <div style={{
      background: colors.bgTertiary,
      border: `1px solid ${colors.border}`,
      padding: spacing.groupMargin,
      marginBottom: spacing.outerSpacing,
      borderRadius: 6,
    }}>
      <div style={{
        fontWeight: 700,
        fontSize: '11pt',
        color: colors.accent,
        marginBottom: spacing.groupSpacing,
      }}>
        {t('phasemap:diagnostics.title')}
      </div>

      {!isRunning && (
        <>
          <label
            htmlFor="forward-diag-bw"
            style={{
              display: 'block',
              fontSize: '9pt',
              color: colors.textSecondary,
              marginBottom: 4,
            }}
          >
            {t('phasemap:diagnostics.bandwidth')}
          </label>
          <Select
            value={String(bandwidth)}
            onChange={(e) => setBandwidth(Number(e.target.value))}
            disabled={isRunning}
            options={BANDWIDTH_OPTIONS.map((b) => ({
              value: String(b.value),
              label: b.label,
            }))}
            style={{ width: '100%' }}
            title={t('phasemap:hoverTips.diagBandwidth')}
          />
          <div style={{ marginTop: spacing.innerSpacing, display: 'flex', gap: spacing.innerSpacing, flexWrap: 'wrap' }}>
            <Button
              onClick={onCompute}
              disabled={!resultId}
              variant="primary"
              title={!resultId ? t('phasemap:diagnostics.noActiveResult') : undefined}
            >
              {computed ? t('phasemap:diagnostics.recompute') : t('phasemap:diagnostics.compute')}
            </Button>
            <Button
              onClick={onOpenBrowser}
              disabled={!computed}
              title={!computed ? t('phasemap:diagnostics.computeFirst') : undefined}
            >
              {t('phasemap:diagnostics.openBrowser')}
            </Button>
          </div>
          {computed && (
            <div style={{
              fontSize: '8.5pt',
              color: colors.textSecondary,
              marginTop: 6,
            }}>
              {t('phasemap:diagnostics.computedAt', { bandwidth: computed.bandwidth, time: computed.computed_at })}
            </div>
          )}
        </>
      )}

      {isRunning && (
        <div>
          <div style={{ fontSize: '10pt', color: colors.text }}>
            {t('phasemap:diagnostics.computingPhase', { phase: job?.current_phase || '…', pct: progressPct })}
          </div>
          <div style={{
            background: colors.bgSecondary,
            height: 6,
            borderRadius: 3,
            marginTop: 4,
            overflow: 'hidden',
            border: `1px solid ${colors.border}`,
          }}>
            <div style={{
              background: colors.accent,
              height: '100%',
              width: `${progressPct}%`,
              borderRadius: 3,
              transition: 'width 0.2s ease',
            }} />
          </div>
          <div style={{ marginTop: spacing.innerSpacing }}>
            <Button onClick={onCancel} variant="default" small title={t('phasemap:hoverTips.diagCancel')}>{t('phasemap:diagnostics.cancel')}</Button>
          </div>
        </div>
      )}

      {error && (
        <div style={{
          color: colors.red,
          fontSize: '9pt',
          marginTop: spacing.innerSpacing,
          padding: '4px 6px',
          border: `1px solid ${colors.red}`,
          borderRadius: 4,
          background: `${colors.red}11`,
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
