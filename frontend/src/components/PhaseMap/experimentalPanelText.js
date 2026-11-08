import { codedText } from '../../services/problemText';

/**
 * What the experimental panel of the Pattern Match window says when it has no pattern: the sentence for
 * the code the answer carries (`experimental_error_code`: the result's own file was refused, e.g. its
 * metadata pairing is lost or it opened as another variant), else the general "not available".
 * @param {object|null} matchData the answer of GET /api/indexing/pattern-match
 * @param {(key: string, options?: object) => string} t i18next's t
 */
export function experimentalMissingText(matchData, t) {
  const general = t('phasemap:matches.notAvailable');
  const code = matchData?.experimental_error_code;
  return (code && codedText(code, t, 'common')) || general;
}
