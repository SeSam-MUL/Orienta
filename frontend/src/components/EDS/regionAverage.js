/**
 * The Region Average panel's words: its column headers (the table and its
 * "Copy to Clipboard" text share them) and whether its composition is a sum
 * or a mean. Plain functions, kept out of the component module.
 */
import { COMPOSITION_SUMMED_NETS } from './compositionMethod';

/**
 * The `eds` i18n keys of the Region Average columns, for the table AND its
 * "Copy to Clipboard" text -- one definition. On the summed source wt% and
 * at% are ONE composition of the region's summed net counts; only their ±
 * (and min/max) is the pixels'. A "(mean±σ)" header over them would be
 * false. Counts stay a pixel mean (net counts on the spectrum source). The
 * copy must not keep saying "(mean±σ)" once the table has stopped.
 */
export function regionHeaderKeys(data) {
  const net = data?.quantification?.source === 'spectrum';
  const summed = data?.composition_method === COMPOSITION_SUMMED_NETS;
  return [
    'table.element',
    net ? 'region.netCountsHeader' : 'region.countsHeader',
    summed ? 'region.wtPctSummedHeader' : 'region.wtPctHeader',
    summed ? 'region.atPctSummedHeader' : 'region.atPctHeader',
  ];
}

/** The tab-separated "Copy to Clipboard" text of a `/region-quantify` answer. */
export function regionClipboardText(data, t) {
  const fmtStat = (s) => {
    if (!s) return '';
    if (typeof s === 'number') return s.toFixed(2);
    return `${s.mean?.toFixed(1) ?? ''} ± ${s.std?.toFixed(1) ?? ''}`;
  };
  const header = regionHeaderKeys(data).map((k) => t(k)).join('\t');
  const rows = Object.entries(data?.data || {}).map(([el, vals]) =>
    `${el}\t${fmtStat(vals.counts)}\t${fmtStat(vals.wt_pct)}\t${fmtStat(vals.at_pct)}`);
  return [header, ...rows].join('\n');
}

/**
 * Is the Region Average a summed composition? The answer's
 * `composition_method` says it; before anything is computed, the file's
 * source does: a file whose spectra are being or have been evaluated (fit
 * running or ready) gives summed compositions, a file without gives the
 * mean of pixels. A FAILED fit gives none at all -- `/region-quantify`
 * answers 500 and the page says the fit failed -- so the panel does not
 * promise a summed composition there.
 */
export function regionAverageIsSummed(data, fitState) {
  if (data?.composition_method) return data.composition_method === COMPOSITION_SUMMED_NETS;
  return fitState === 'running' || fitState === 'ready';
}
