/**
 * Turn the backend's answer to POST /api/install/wsl-install into what the
 * wizard shows.
 *
 * The backend answers HTTP 200 in every case with `{success, stage, message}`
 * — a denied UAC prompt is `success: false`, not an HTTP error — and its
 * message says which stage ran (the Windows feature, elevated; or the
 * distribution, as the user) and what to do next. The generic i18n strings
 * are only the fallback for an answer without a message.
 *
 * @param {object|undefined} data   response body
 * @param {(key: string) => string} t   i18n
 * @returns {{ ok: boolean, text: string, refresh: boolean }}
 */
export function installOutcome(data, t) {
  // Only an explicit success:true is success. A missing or garbled body is
  // the same class of bug as success:false — do not show "started".
  if (data && data.success === true) {
    return { ok: true, text: data.message || t('settings:install.step1.installStarted'), refresh: true };
  }
  const message = data && data.message;
  return { ok: false, text: message || t('settings:install.step1.installFailed'), refresh: false };
}
