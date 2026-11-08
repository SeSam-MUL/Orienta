/**
 * The backend refuses to change the calibration set or the detector while a PC
 * refinement or grid calibration runs (409, code `pc_job_running`): the run works
 * on them for minutes, and a pattern added or removed meanwhile would shift the
 * indices under it. These helpers recognise that answer and word it in the
 * reader's language, wherever it arrives (viewer, PC page).
 */
export const PC_JOB_RUNNING = 'pc_job_running';

/** True for the backend's "a PC run is going on" refusal. */
export function isPcJobRunning(err) {
  return err?.response?.status === 409 && err?.response?.data?.detail?.code === PC_JOB_RUNNING;
}

/** The sentence for it; `t` may come from any namespace (all are loaded). */
export function pcJobRunningText(t) {
  return t('pcrefinement:errors.pc_job_running');
}
