/**
 * How a loaded file is named where the user switches between them.
 *
 * An EDAX measurement is loaded from its pattern file, and the registry names an entry by that
 * file's stem, so two measurements of one session would read as two unrelated pattern files and
 * the name the user ticked in the measurement dialog would never reappear. An entry loaded with an
 * APEX pairing carries `apex.measurement_name` ("Area 1 / OIM Map 1"); it leads the label, and the
 * pattern file's name follows in brackets. Any other entry keeps its name.
 *
 * A file opened as a variant other than its default (another measurement, or the unprocessed
 * patterns) is named by the backend with the parts added (`scan · 2 · unprocessed`) and carries the
 * `variant` it was opened as. The measurement part is the file's own; the word for the unprocessed
 * source is English from the backend and is written in the language of the page here.
 */

const UNPROCESSED_TAIL = ' · unprocessed';

/** The measurement name of a loaded-files entry, or ''. */
export const apexMeasurementName = (entry) => {
  const name = entry?.apex?.measurement_name;
  return typeof name === 'string' ? name.trim() : '';
};

/** True for an entry opened as a variant that is not its file's default: its name carries the variant. */
export const isNonDefaultVariant = (entry) =>
  !!entry?.variant && typeof entry.name === 'string' && entry.name.includes(' · ');

/**
 * "Area 1 / OIM Map 1 (scan_a.up1)" for a measurement of an APEX file; "scan · 2 · unprocessed" for a
 * non-default variant of a file; else the entry's name.
 * @param {object} entry an entry of GET /loaded-files
 * @param {(key: string, options?: object) => string} [tt] translates a key of the `ebsdviewer` namespace;
 *        without it the backend's English parts stay as they are
 */
export function loadedFileLabel(entry, tt) {
  const name = entry?.name || '';
  const measurement = apexMeasurementName(entry);
  if (measurement) return `${measurement} (${name})`;
  if (typeof tt === 'function' && isNonDefaultVariant(entry) && entry.variant.pattern_source === 'unprocessed'
      && name.endsWith(UNPROCESSED_TAIL)) {
    return `${name.slice(0, -UNPROCESSED_TAIL.length)} · ${tt('loadDialog.unprocessedTag')}`;
  }
  return name;
}
