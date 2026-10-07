/**
 * The tooltip that says WHY a phase won a calibration pattern.
 *
 * With several phases PyEBSDIndex indexes the pattern against each one and keeps
 * the phase with the largest (3 deg - fit) * matched bands. It is reported with
 * every phase's own numbers so the reader can see the competition; the CI is
 * shown but does not decide.
 *
 * @param {{name: string, ci: number|null, fit: number|null, n_bands: number,
 *          score: number|null}[]|null|undefined} fits  one entry per loaded phase
 * @param {string} winner  name of the phase that won
 * @param {(key: string, opts?: object) => string} t  i18next `t`
 * @returns {string|null}  null when there is nothing to show (single phase)
 */
export function phaseFitsTooltip(fits, winner, t) {
  if (!Array.isArray(fits) || fits.length < 2) return null;
  const lines = [t('pcrefinement:preview.phaseFitsTitle')];
  for (const f of fits) {
    let line = f.score == null
      ? t('pcrefinement:preview.phaseFitsNone', { name: f.name, bands: f.n_bands })
      : t('pcrefinement:preview.phaseFitsRow', {
        name: f.name,
        ci: f.ci.toFixed(3),
        fit: f.fit.toFixed(2),
        bands: f.n_bands,
        score: f.score.toFixed(1),
      });
    if (f.name === winner) line += t('pcrefinement:preview.phaseFitsWinner');
    lines.push(line);
  }
  lines.push(t('pcrefinement:preview.phaseFitsRule'));
  return lines.join('\n');
}
