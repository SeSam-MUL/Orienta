// The words for a calibration request that was refused or failed. A PC search that runs refuses every request
// that changes a PC; that has its own sentence, a code the build knows is translated, anything else shows the
// backend's own English sentence, and a request that never reached the backend shows the error's message.
import { problemText } from '../../../services/problemText';
import { isPcJobRunning, pcJobRunningText } from '../jobRunning';

/** True when the backend answered (a refusal in its own words); false for a network error with no answer. */
export const wasAnswered = (err) => !!err?.response;

export function refusalWords(err, t) {
  if (isPcJobRunning(err)) return pcJobRunningText(t);
  return problemText(err, t, 'pcrefinement') || err?.message || '';
}
