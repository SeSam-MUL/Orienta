/**
 * What the status bar says about the tilt distortion check of an .h5oina
 * (backend services/tilt_distortion.py). Pure, so it is testable without React.
 *
 * `lang` formats the numbers. Returns null when there is nothing to say: no check, an undistorted file, or
 * an untilted file the check could not judge. Otherwise
 * `{ level, key, params, titleKey, titleParams }` for `t()` in the shell namespace.
 */

export const TILT_X = 'X gestaucht';
export const TILT_Y = 'Y gestaucht';
export const TILT_OK = 'unverzerrt';
export const TILT_UNCLEAR = 'unklar';

export function tiltBadge(check, lang = 'en') {
  if (!check || typeof check !== 'object') return null;
  // Numbers in the reader's locale (2,92 in German).
  const fmt = (x, digits = 2) => (typeof x === 'number' && Number.isFinite(x)
    ? new Intl.NumberFormat(lang || 'en', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x)
    : '—');
  const befund = check.befund;
  const tilt = check.kippung_wirksam_grad ?? check.kippung_header_grad;
  const se = (check.bilder || []).find((b) => b.kanal === 'SE' && b.lx_ly_median != null)
    || (check.bilder || []).find((b) => b.lx_ly_median != null);
  const ebsd = (check.karten || []).find((k) => k.typ === 'EBSD') || (check.karten || [])[0];
  const common = {
    tilt: fmt(tilt, 0),
    ratio: fmt(se?.lx_ly_median),
    image: se?.name || '—',
  };
  if (check.fehler) {
    return { level: 'warn', key: 'status.tiltFailed', params: {},
      titleKey: 'status.tiltFailedTitle', titleParams: { error: String(check.fehler) } };
  }
  if (befund === TILT_X || befund === TILT_Y) {
    const dir = befund === TILT_X ? 'X' : 'Y';
    const factor = fmt(check.faktor_korrektur);
    const nominal = ebsd?.step_nominal_um;
    const trueStep = ebsd?.wahrer_schritt_um?.[dir === 'X' ? 0 : 1];
    const inherits = ebsd?.erbt_bildverzerrung === true;
    return {
      level: 'warn',
      key: 'status.tiltDistorted',
      params: { axis: dir, factor },
      titleKey: inherits ? 'status.tiltDistortedMapTitle' : 'status.tiltDistortedTitle',
      titleParams: { ...common, axis: dir, factor, nominal: fmt(nominal, 3), trueStep: fmt(trueStep, 3),
        target: (se?.zielgroesse_px || []).join(' × ') || '—' },
    };
  }
  if (befund === TILT_UNCLEAR && typeof tilt === 'number' && tilt >= 10) {
    return { level: 'warn', key: 'status.tiltUnclear', params: {},
      titleKey: 'status.tiltUnclearTitle', titleParams: common };
  }
  return null;
}
