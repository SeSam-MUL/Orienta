import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { analysisApi } from '../../services/api';
import useResultStore from '../../stores/useResultStore';
import { taskOutcome, pollErrorOutcome, progressPercent } from './grainStats';

/**
 * The page's watch of the reconstruction the backend is running (`reconTaskId` in the store: started by the Grain
 * tab, or found by the /status sync): ONE poll per task id, whichever tab is open. When the task ends, the
 * grains exist (flag, version, statistics) or its failure is recorded; the Grain tab only displays that.
 * "Run Complete Analysis" waits for its own reconstruction, so the watch stays out of its way.
 */
export function useReconstructionWatch({ onComplete = () => {} } = {}) {
  const { t } = useTranslation('analysis');
  const reconTaskId = useResultStore((state) => state.reconTaskId);
  const runAllBusy = useResultStore((state) => state.runAllBusy);
  useEffect(() => {
    if (!reconTaskId || runAllBusy) return undefined;
    const iv = setInterval(async () => {
      const store = useResultStore.getState();
      const epoch = store.grainSizeEpoch;
      try {
        const res = await analysisApi.getTaskStatus(reconTaskId);
        const now = useResultStore.getState();
        // ended meanwhile (sync, new map), or a map was loaded / a reconstruction started while the answer was on its
        // way: it belongs to other grains. The next poll asks again.
        if (now.reconTaskId !== reconTaskId || now.grainSizeEpoch !== epoch) return;
        const d = res.data;
        const percent = progressPercent(d.progress);
        if (percent !== undefined) store.setReconProgress(percent);
        const out = taskOutcome(d);
        if (out.done) {
          clearInterval(iv);
          store.finishReconstruction({ stats: out.stats, text: t('analysis:grain.reconstructedMsg', { count: out.stats.count }) });
          onComplete('grains');
        } else if (out.failed) {
          clearInterval(iv);
          store.failReconstruction(out.message || t('analysis:grain.reconstructionFailed'));
        }
      } catch (err) {
        if (pollErrorOutcome(err).stop && useResultStore.getState().reconTaskId === reconTaskId) {
          // The backend forgot the task (restart / eviction): nothing to wait for.
          clearInterval(iv);
          store.failReconstruction(t('analysis:grain.reconstructionFailed'));
        }
        // other errors (network blip): keep polling
      }
    }, 1500);
    return () => clearInterval(iv);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reconTaskId, runAllBusy]);
}
