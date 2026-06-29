/**
 * RefinementPanel - smoothness slider + Refine button + progress + summary display.
 * Lives below ComputeDiagnosticsPanel in PhaseMapPage right settings panel.
 *
 * Mirrors ComputeDiagnosticsPanel (Phase A) for state-management patterns, but
 * uses refinementApi (Phase B joint R+PC refinement) instead of forwardDiagApi.
 */
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import useDataStore from '../../stores/useDataStore';
import { refinementApi } from '../../services/api';
import { colors, Button } from '../../theme/components';

// Log-scale slider mapping: slider value 0..100 -> lambda in [0.001, 1.0]
const _LAMBDA_MIN_LOG = Math.log10(0.001);
const _LAMBDA_MAX_LOG = Math.log10(1.0);
const sliderToLambda = (v) =>
  Math.pow(10, _LAMBDA_MIN_LOG + (v / 100) * (_LAMBDA_MAX_LOG - _LAMBDA_MIN_LOG));
const lambdaToSlider = (l) =>
  ((Math.log10(l) - _LAMBDA_MIN_LOG) / (_LAMBDA_MAX_LOG - _LAMBDA_MIN_LOG)) * 100;

export default function RefinementPanel({ resultId, onSwitchToRefined }) {
  const { t } = useTranslation('phasemap');
  const [error, setError] = useState(null);
  const job = useDataStore((s) => s.refinementJob);
  const computedMap = useDataStore((s) => s.refinementComputed);
  const setJob = useDataStore((s) => s.setRefinementJob);
  const setComputed = useDataStore((s) => s.setRefinementComputed);
  const sliderLambda = useDataStore((s) => s.refinementLambdaPending);
  const setSliderLambda = useDataStore((s) => s.setRefinementLambda);

  const computed = resultId ? computedMap[resultId] : null;
  const pollRef = useRef(null);

  // Poll progress while a job is running for this result.
  // Latest job snapshot is read inside the interval via useDataStore.getState()
  // so the deps array stays minimal (no infinite-loop trigger from setJob).
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
        const p = await refinementApi.progress(jobId);
        const currentJob = useDataStore.getState().refinementJob;
        if (currentJob && currentJob.job_id === jobId) {
          setJob({ ...currentJob, ...p });
        }
        if (p.state === 'completed') {
          const s = await refinementApi.summary(resultId);
          if (s) {
            setComputed(resultId, {
              lambda: s.smoothness_lambda,
              computed_at: s.computed_at,
              refined_result_id: s.refined_result_id,
              summary: s.summary,
              stage_timings: s.stage_timings,
            });
          }
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

  const onRefine = async () => {
    if (!resultId) return;
    setError(null);
    try {
      const r = await refinementApi.compute(resultId, sliderLambda, false);
      setJob({
        result_id: resultId,
        job_id: r.job_id,
        state: 'running',
        progress: 0,
        current_stage: 'stage_1',
        outer_iter: 0,
      });
    } catch (e) {
      setError(e.message || String(e));
    }
  };

  const onResmooth = async () => {
    if (!resultId) return;
    setError(null);
    try {
      const r = await refinementApi.resmooth(resultId, sliderLambda);
      setJob({
        result_id: resultId,
        job_id: r.job_id,
        state: 'running',
        progress: 0,
        current_stage: 'stage_2',
        outer_iter: 0,
      });
    } catch (e) {
      setError(e.message || String(e));
    }
  };

  const onCancel = async () => {
    if (!job) return;
    try {
      await refinementApi.cancel(job.job_id);
    } catch {
      /* swallow - we still clear local job state below */
    }
    setJob(null);
  };

  const isRunning = !!(job && job.result_id === resultId && job.state === 'running');
  const lambdaChanged = !!(computed && Math.abs(computed.lambda - sliderLambda) > 1e-6);
  const hasCache = computed != null;
  const progressPct = Math.round((job?.progress || 0) * 100);

  return (
    <div style={{
      background: colors.bgTertiary,
      border: `1px solid ${colors.border}`,
      padding: 12,
      marginBottom: 12,
      borderRadius: 6,
    }}>
      <div style={{
        fontWeight: 700,
        fontSize: '11pt',
        color: colors.accent,
        marginBottom: 8,
      }}>
        {t('phasemap:refinement.title')}
      </div>

      {!isRunning && (
        <>
          <label style={{
            display: 'block',
            fontSize: '9pt',
            color: colors.textSecondary,
            marginBottom: 4,
          }}>
            {t('phasemap:refinement.smoothness', { value: sliderLambda.toExponential(2) })}
          </label>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={lambdaToSlider(sliderLambda)}
            onChange={(e) => setSliderLambda(sliderToLambda(Number(e.target.value)))}
            style={{ width: '100%' }}
            title={t('phasemap:hoverTips.smoothnessSlider')}
          />
          <div style={{
            fontSize: '8pt',
            color: colors.textSecondary,
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>0.001</span><span>0.01</span><span>1.0</span>
          </div>
          <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {!hasCache ? (
              <Button onClick={onRefine} disabled={!resultId} variant="primary" title={t('phasemap:hoverTips.refineRun')}>
                {t('phasemap:refinement.refine')}
              </Button>
            ) : !lambdaChanged ? (
              <>
                <Button onClick={onRefine} disabled={!resultId} title={t('phasemap:hoverTips.reRefine')}>
                  {t('phasemap:refinement.reRefine')}
                </Button>
                <Button onClick={onSwitchToRefined} disabled={!computed} title={t('phasemap:hoverTips.openRefined')}>
                  {t('phasemap:refinement.openRefined')}
                </Button>
              </>
            ) : (
              <Button onClick={onResmooth} disabled={!resultId} variant="primary" title={t('phasemap:hoverTips.reSmooth')}>
                {t('phasemap:refinement.reSmooth', { value: sliderLambda.toExponential(2) })}
              </Button>
            )}
          </div>
          {computed && (
            <div style={{
              fontSize: '8.5pt',
              color: colors.textSecondary,
              marginTop: 6,
            }}>
              <div>{t('phasemap:refinement.lambdaTime', { lambda: computed.lambda.toExponential(2), time: computed.computed_at })}</div>
              <div>
                {t('phasemap:refinement.stageTimings', {
                  s1: computed.stage_timings?.stage_1?.toFixed(0),
                  s2: computed.stage_timings?.stage_2?.toFixed(0),
                  s3: computed.stage_timings?.stage_3?.toFixed(0),
                })}
              </div>
              <div>
                {t('phasemap:refinement.nccImprovement', { value: computed.summary?.ncc_improvement?.median?.toFixed(2) })}
              </div>
              <div>
                {t('phasemap:refinement.pixelSummary', {
                  refined: computed.summary?.n_pixels_refined,
                  converged: computed.summary?.n_converged,
                  failed: computed.summary?.n_failed,
                })}
              </div>
            </div>
          )}
        </>
      )}

      {isRunning && (
        <div>
          <div style={{ fontSize: '10pt', color: colors.text }}>
            {t('phasemap:refinement.stageProgress', {
              stage: job.current_stage || 'stage_1',
              outer: job.outer_iter ? t('phasemap:refinement.outerIter', { n: job.outer_iter }) : '',
              pct: progressPct,
            })}
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
          <div style={{ marginTop: 8 }}>
            <Button onClick={onCancel} title={t('phasemap:hoverTips.refineCancel')}>{t('phasemap:refinement.cancel')}</Button>
          </div>
        </div>
      )}

      {error && (
        <div style={{
          color: colors.red,
          fontSize: '9pt',
          marginTop: 8,
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
