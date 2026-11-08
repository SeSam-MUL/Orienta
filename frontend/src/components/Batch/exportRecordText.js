/**
 * The words of the export record, in the page's language.
 *
 * The backend writes each note about an export and each reason for a missing file as one English sentence,
 * and sends beside it `{code, params, text}`: a stable code, the numbers the sentence is made of, and the
 * sentence itself. The code is looked up under `exports.notes.<code>` or `exports.errors.<code>` of the batch
 * namespace. A code this build has no words for, or an item without the params its words need, shows the
 * backend's sentence unchanged, so a newer backend never shows a translation key.
 */

// An empty defaultValue is not a value when returnEmptyString is false, so a sentinel no translation can
// equal is the only reliable "missing" (the same device as `codedText`).
const MISS = '\u0000miss';

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/** The translation of `exports.<section>.<code>` with `params`, or null when there is none. */
function words(code, section, t, params) {
  if (!code) return null;
  const out = t(`exports.${section}.${code}`, { ...params, defaultValue: MISS });
  return out === MISS ? null : out;
}

/**
 * One note of a file's record, `{code, params, text}`, in the page's language.
 *
 * The resampling note is the sentence with its numbers, and, when `direction_assumed` says no file states the
 * direction of the row offset, the sentence that this is an assumption and on which side the odd rows lie.
 * That second sentence is added for exactly that case.
 */
export function noteText(item, t) {
  const p = item?.params || {};
  switch (item?.code) {
    case 'resampledFromHex': {
      if (!finite(p.step) || !finite(p.source_points) || !finite(p.written) || !finite(p.unused)) break;
      const base = words(item.code, 'notes', t, {
        step: p.step, sourcePoints: p.source_points, written: p.written, unused: p.unused,
      });
      if (base === null) break;
      if (!p.direction_assumed) return base;
      const side = words(p.side === 'left' ? 'rowOffsetAssumedLeft' : 'rowOffsetAssumedRight', 'notes', t, {});
      if (side === null) break;
      return `${base} ${side}`;
    }
    case 'cleanupFailed':
    case 'autoAssignmentFailed': {
      // the text of the exception follows the translated sentence unchanged
      if (typeof p.detail !== 'string') break;
      const out = words(item.code, 'notes', t, { detail: p.detail });
      if (out !== null) return out;
      break;
    }
    default: {
      const out = words(item?.code, 'notes', t, {});
      if (out !== null) return out;
    }
  }
  return item?.text ?? '';
}

/**
 * The reason a format was not written, `{code, params, text}`, in the page's language. `detail` is the text of
 * a real exception: it follows the translated lead-in unchanged.
 */
export function reasonText(item, t) {
  const p = item?.params || {};
  if (item?.code === 'writerFailed' || item?.code === 'exportFailed') {
    if (typeof p.detail === 'string') {
      const out = words(item.code, 'errors', t, { detail: p.detail });
      if (out !== null) return out;
    }
  } else {
    const out = words(item?.code, 'errors', t, {});
    if (out !== null) return out;
  }
  return item?.text ?? '';
}
