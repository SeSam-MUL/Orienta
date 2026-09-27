/**
 * The app's own version, asked for until the backend answers.
 *
 * WHY THIS IS A HOOK AND NOT TWO useEffects. Both callers fetched it once on
 * mount and treated a failed fetch as an answer. The first start after a
 * runtime update is exactly the moment the backend is not up yet, so both
 * broke together, and neither recovered:
 *
 *   - Settings → About showed "unknown (no git information found)" forever,
 *     pointing at git, while GET /api/system/version was answering v0.4.6.
 *     Sebastian saw this after installing build 3.
 *   - ProblemReportDialog swallowed the failure in an empty catch. A bug
 *     report filed right after an update therefore carried no version at
 *     all, and `canOpenIssue` — gated on `versionInfo?.repo_url` — stayed
 *     false, so the user could not open a GitHub issue either. That is the
 *     worse half: it degrades bug reports at exactly the moment bug reports
 *     are most likely and most useful.
 *
 * A failed fetch is not an answer. Keep asking, doubling the wait from 1 s to
 * a minute so a genuinely dead backend costs one request a minute rather than
 * a flood. Any answer stops the retrying, INCLUDING one that says the version
 * is unknown: that is a real answer and the caller should show it as such.
 *
 * Returns `{ info, unreachable }`:
 *   info === null                  nothing has come back yet
 *   unreachable === true           ...and at least one attempt has failed,
 *                                  so the honest thing to say is "the backend
 *                                  is not answering", not "no git information"
 *   info.version === 'unknown'     the backend answered and does not know
 */
import { useEffect, useState } from 'react';
import { getAppVersion } from '../services/api';
import { FIRST_DELAY_MS, MAX_DELAY_MS, nextDelay } from '../services/retryBackoff';

// Re-exported because this hook's tests name them, and because it is still
// the place a reader looks first. The schedule itself now lives in
// services/retryBackoff.js, shared with the Indexing phase list, which had
// the same defect in a worse form: a failed fetch there did not just fail to
// answer, it emptied the list and reported "No phases found".
export { FIRST_DELAY_MS, MAX_DELAY_MS };

export default function useAppVersion() {
  const [info, setInfo] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    let delay = FIRST_DELAY_MS;

    const attempt = () => {
      getAppVersion()
        // `|| {}` so a reply with no body still counts as an answer and ends
        // the retrying; versionText then reports it as unknown, which is what
        // it is.
        .then((v) => { if (!cancelled) setInfo(v || {}); })
        .catch(() => {
          if (cancelled) return;
          setFailed(true);
          timer = setTimeout(attempt, delay);
          delay = nextDelay(delay);
        });
    };
    attempt();

    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  return { info, unreachable: info === null && failed };
}
