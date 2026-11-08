/**
 * The Indexing page's "chemistry mask failed" line.
 *
 * `POST /api/eds/chemistry-mask` reads EDS numbers, and on a file with a
 * spectrum cube it answers 503 / 500 with a `{code, message}` detail while
 * the fit runs or after it failed. Put straight into the sentence that was
 * "[object Object]". The detail goes through
 * {@link edsDetailText}: the fit's two codes in the page's language, any
 * other detail as its text.
 */
import { edsDetailText, edsNamespaceT } from '../EDS/edsDetailText';

export function chemMaskErrorText(err, t) {
  return t('mask.maskFailed', {
    error: edsDetailText(err, edsNamespaceT(t)) || err?.message || '',
  });
}

export default chemMaskErrorText;
