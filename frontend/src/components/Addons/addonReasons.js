/**
 * The backend's machine-readable `reason` codes, as sentences.
 *
 * Thirteen stable codes exist and, until this file, nothing in the app read
 * one: every caller branched on English `detail` text or on a status number.
 * This is their first consumer.
 *
 * TWO LAYERS, and the order matters. The TITLE is ours and is translated --
 * "this add-on is switched off" is a fact about Orienta, the same in every
 * language. The DETAIL is the server's own sentence and is passed through
 * VERBATIM: for a failed run it carries the add-on's own message, which is the
 * only text in the response that says what actually went wrong, and no locale
 * of ours could ever contain it.
 */

//: The shape a code has: lowercase words joined by underscores. Checked rather
//: than assumed, because these arrive from the wire — a key built out of a
//: path, a sentence or a number would be looked up, missed, and printed raw.
const CODE = /^[a-z][a-z0-9_]*$/;

export function reasonKey(reason) {
  if (typeof reason !== 'string' || !CODE.test(reason)) return null;
  return `addons:reason.${reason}`;
}

/**
 * @param {{reason?: string, detail?: string}} failure — a PLAIN object, not an
 *   axios error and not a job dict. Both of those exist in this app and they
 *   disagree about field names; adapting them at the two call sites is what
 *   stops this function from growing two shapes and drifting.
 * @param {(key: string, opts?: object) => string} t — i18next's, or any lookup.
 */
export function describeFailure(failure, t) {
  const reason = failure && failure.reason;
  const serverDetail = (failure && typeof failure.detail === 'string'
    && failure.detail.trim()) ? failure.detail.trim() : '';

  const key = reasonKey(reason);
  // defaultValue, so a locale that has not caught up yet shows the server's
  // sentence rather than the key. Printing "addons:reason.x" at a user is
  // worse than printing English.
  const titled = key ? t(key, { defaultValue: '' }) : '';

  if (titled) {
    // Our sentence on top, the server's underneath — two different facts.
    return {
      title: titled,
      detail: serverDetail || t('addons:failure.noDetail', {
        defaultValue: 'The server gave no further explanation.' }),
    };
  }

  // No sentence of ours, so the server's becomes the headline — and the
  // detail is EMPTY rather than the same words again. A panel that printed
  // both read "The add-on list could not be loaded.The add-on list could not
  // be loaded.": a second line that pretends to add information and adds
  // none. The caller renders nothing when detail is empty.
  if (serverDetail) return { title: serverDetail, detail: '' };

  return {
    title: t('addons:failure.unknownTitle', {
      defaultValue: 'The request was refused.' }),
    detail: t('addons:failure.noDetail', {
      defaultValue: 'The server gave no further explanation.' }),
  };
}
