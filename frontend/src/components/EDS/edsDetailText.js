/**
 * The text of a failed EDS request's `detail`, whatever shape it arrived in.
 *
 * Every error panel on the EDS page used to do
 * `setError(err.response?.data?.detail || ...)` and render the result as a
 * React child. That only works while `detail` is a string. Two answers are
 * not strings:
 *
 *  - the spectrum fit's refusals (HTTP 503 while it runs, 500 after it
 *    failed) carry `{code, progress | error, message}` so the page can tell
 *    "still evaluating" from a real error;
 *  - FastAPI's own validation errors (HTTP 422) carry a list of
 *    `{loc, msg, type}`.
 *
 * Rendering either object is "Objects are not valid as a React child": the
 * whole module crashes into the error boundary instead of saying "the spectra
 * are still being evaluated". So every site goes through here.
 *
 * Returns `null` when there is no `detail` at all, so a call site keeps its
 * own fallback chain exactly as it was: `edsDetailText(err, t) || err.message`.
 *
 * With `t` (the `eds` namespace) the spectrum fit's two codes are translated,
 * falling back to the backend's own English sentence; without it, the
 * backend's sentence.
 */
const FIT_KEYS = {
  eds_spectrum_fit_running: 'spectrumFit.running',
  eds_spectrum_fit_failed: 'spectrumFit.failed',
};

/**
 * Is `err` the spectrum fit's refusal (HTTP 503 while it runs, 500 after it
 * failed)? For callers outside the EDS page that translate everything else
 * their own way (`problemText`) and only these two codes through
 * {@link edsDetailText}.
 */
export function isSpectrumFitRefusal(err) {
  const detail = err?.response?.data?.detail;
  return !!(detail && typeof detail === 'object' && !Array.isArray(detail)
            && FIT_KEYS[detail.code]);
}

/** `t` of any namespace, re-pointed at the `eds` namespace's keys. */
export function edsNamespaceT(t) {
  return typeof t === 'function' ? (key, opts) => t(`eds:${key}`, opts) : t;
}

export function edsDetailText(err, t) {
  const detail = err?.response?.data?.detail;
  if (detail == null || detail === '') return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => (typeof d === 'string'
      ? d : (d?.msg || d?.message || JSON.stringify(d))));
    return parts.join('; ') || null;
  }
  if (typeof detail === 'object') {
    const prose = typeof detail.message === 'string' ? detail.message : null;
    const key = FIT_KEYS[detail.code];
    if (key && typeof t === 'function') {
      return t(key, {
        percent: Math.round(100 * (Number(detail.progress) || 0)),
        error: detail.error ?? '',
        ...(prose ? { defaultValue: prose } : {}),
      });
    }
    if (prose) return prose;
    if (typeof detail.error === 'string') return detail.error;
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }
  return String(detail);
}

/**
 * The sentence for an `eds_status` marker (`fit_status_marker` in
 * eds_spectrum_service): the EDS part of a composite answer was left out
 * because the spectrum fit is still running, or because it failed.
 */
export function edsStatusText(status, t) {
  if (status?.state === 'failed') {
    return t('spectrumFit.edsFailed', { error: status.error || '?' });
  }
  return t('spectrumFit.edsPending', {
    percent: Math.round(100 * (Number(status?.progress) || 0)),
  });
}

export default edsDetailText;
