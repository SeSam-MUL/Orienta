/**
 * ResultCaveats - notes a finished indexing run carries about its own result.
 *
 * The backend sends them as `caveats: [{ code, phases, text }]` (task result and
 * `/result/last`): Hough on one of the seven trigonal space groups PyEBSDIndex gets
 * wrong (`houghTrigonalSetting`), Hough on the Laue classes -3 and 6/m, where it often
 * returns a variant of the hexagonal lattice that is not the crystal's
 * (`houghHexagonalLaue`), and the same for the other groups below 6/mmm in a smaller
 * share of the patterns (`houghHexagonalVariants`, a softer hint: accent colour and an
 * info sign instead of the warning colour). A note, not a refusal: the run finished and
 * its numbers stand; this says what to check.
 *
 * The sentence comes from `indexing:caveats.<code>` in the user's language (a plural
 * pair on the number of phases). A code this build has no sentence for shows the
 * backend's English `text`, so a newer backend never produces a bare key.
 */
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';

const CODE_SHAPE = /^[A-Za-z][A-Za-z0-9]*$/;

/** Codes that are a hint, not a warning (the same set as `SOFT_CODES` in hough_caveats.py). */
export const SOFT_CODES = new Set(['houghHexagonalVariants']);

/** The caveats in `source` that the page can show: objects with a plain code. */
export function caveatsFrom(source) {
  const list = source?.caveats;
  if (!Array.isArray(list)) return [];
  return list.filter((c) => c && typeof c === 'object' && typeof c.code === 'string'
    && CODE_SHAPE.test(c.code));
}

/** One caveat as a sentence: the locale's text for its code, else the backend text. */
export function caveatText(caveat, t, exists) {
  const phases = Array.isArray(caveat.phases) ? caveat.phases.filter((p) => typeof p === 'string') : [];
  const opts = { ns: 'indexing', count: Math.max(phases.length, 1), phases: phases.join(', ') };
  const key = `caveats.${caveat.code}`;
  if (exists(key, opts)) return t(key, opts);
  return typeof caveat.text === 'string' ? caveat.text : '';
}

export default function ResultCaveats({ caveats }) {
  const { t, i18n } = useTranslation('indexing');
  const list = caveatsFrom({ caveats });
  if (list.length === 0) return null;
  const exists = (key, opts) => i18n.exists(key, opts);

  return (
    <div role="status" aria-label={t('caveats.title')}
      style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
      {list.map((caveat) => {
        const soft = SOFT_CODES.has(caveat.code);
        const tone = soft ? C.accent : C.yellow;
        return (
          <div
            key={caveat.code + (caveat.phases || []).join('|')}
            data-severity={soft ? 'info' : 'warning'}
            style={{
              display: 'flex', alignItems: 'flex-start', gap: 8,
              padding: '5px 8px', borderRadius: 4,
              border: `1px solid ${tone}`,
              background: 'rgba(255,255,255,0.04)',
              borderLeft: `3px solid ${tone}`,
            }}
          >
            <span aria-hidden="true" style={{ flexShrink: 0 }}>{soft ? 'ⓘ' : '⚠'}</span>
            <span style={{ flex: 1, minWidth: 0, color: C.text, fontSize: '9pt' }}>
              {caveatText(caveat, t, exists)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
