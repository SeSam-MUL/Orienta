/**
 * Which discovered files carry the EDS phase map's phases, for the method now
 * selected.
 *
 * The phase map names its phases by CIF filename and nothing else, but a
 * Spherical run is made of SHT files and a Dictionary run of dictionary /
 * master `.h5` files. The discovery attaches to every record `linked_cif`, the
 * basename of the CIF the file belongs to (a CIF is its own; the backend's
 * `phase_metadata.strict_linked_cif`), and that is the join: a phase-map phase
 * `Al.cif` is carried by the SHT or dictionary whose `linked_cif` is `Al.cif`.
 * The backend joins the request's files the same way
 * (`phase_map_file_links`), so what is pre-selected here is what the run will
 * route.
 *
 * At most ONE file per phase-map phase, by the rule the page already uses for
 * the method: the CIF itself for Hough, the first SHT for Spherical, and for
 * Dictionary the dictionary that fits the detector and pattern centre
 * (`selectBestDict`, as the page resolves a phase's dictionary). A phase that
 * has only a master pattern has nothing to index with yet and is reported as
 * needing a dictionary, not selected.
 *
 * Pure, so it can be tested without mounting the 4000-line page.
 */
import { selectBestDict, dictShapeMatches, dictGeometryMatches } from './dictionaryTargets';

/** Extensions a method runs on, lower case, without the dot. */
const METHOD_EXTENSIONS = {
  hough: ['cif'],
  spherical: ['sht'],
  dictionary: ['h5', 'hdf5'],
};

/** The locale key (messages.*) naming the file kind of each method. */
const KIND_KEYS = {
  hough: 'messages.handoffKindHough',
  spherical: 'messages.handoffKindSpherical',
  dictionary: 'messages.handoffKindDictionary',
};

/** The short method name the page already shows elsewhere (no backend in it). */
const METHOD_LABEL_KEYS = {
  hough: 'batchDialog.methodHough',
  spherical: 'batchDialog.methodSpherical',
  dictionary: 'batchDialog.methodDictionary',
};

/** Locale key of the short display name of ``method`` ("Spherical", not "Spherical (EMSphinx)"). */
export function handoffMethodLabelKey(method) {
  return METHOD_LABEL_KEYS[method] || METHOD_LABEL_KEYS.hough;
}

/** Locale key of the file kind of ``method`` (``messages.handoffKindHough`` ...). */
export function handoffFileKindKey(method) {
  return KIND_KEYS[method] || KIND_KEYS.hough;
}

function extensionOf(file) {
  const name = String(file?.filename || file?.path || '');
  return name.toLowerCase().split('.').pop();
}

const baseName = (f) => String(f?.filename || f?.path || '');

/**
 * @param {Array<object>} discoveredFiles  records of the CURRENT method's list
 * @param {string} method                  'hough' | 'spherical' | 'dictionary'
 * @param {string[]} cifFilenames          the phase map's CIF basenames
 * @param {{currentPc?: number[]|null, detectorShape?: number[]|null, geom?: object|null}} [ctx]
 *   what a Dictionary has to fit (the page's PC, detector shape and geometry)
 * @returns {{
 *   matched: object[],            files to select, one per phase, in the map order
 *   missing: string[],            phases with no file for this method
 *   needsDictionary: string[],    Dictionary: a master pattern but no dictionary that fits
 *   several: {name: string, taken: string, others: string[]}[],
 *                                 phases for which several files fit (which one was taken)
 *   dictPaths: Object<string, string>,  Dictionary: file path -> the dictionary to run
 * }}
 */
export function pickHandoffFiles(discoveredFiles, method, cifFilenames, ctx = {}) {
  const files = Array.isArray(discoveredFiles) ? discoveredFiles : [];
  const names = Array.isArray(cifFilenames) ? cifFilenames : [];
  const out = { matched: [], missing: [], needsDictionary: [], several: [], dictPaths: {} };

  if (method === 'hough') {
    // Unchanged from the Hough-only hand-off: every discovered CIF that is a
    // phase-map phase, and a phase is missing when no CIF of that name exists.
    const cifFiles = files.filter(
      (f) => f.file_type === 'phase' || (f.filename || '').toLowerCase().endsWith('.cif'),
    );
    out.matched = cifFiles.filter((f) => names.includes(f.filename));
    out.missing = names.filter((name) => !cifFiles.some((f) => f.filename === name));
    return out;
  }

  const extensions = METHOD_EXTENSIONS[method] || [];
  const pool = files.filter((f) => extensions.includes(extensionOf(f)));
  const { currentPc = null, detectorShape = null, geom = null } = ctx || {};

  for (const name of names) {
    const linked = pool.filter((f) => f.linked_cif === name);

    if (method === 'dictionary') {
      const dicts = linked.filter((f) => f.file_type === 'dictionary');
      const masters = linked.filter((f) => f.file_type === 'master');
      if (dicts.length === 0) {
        // A master pattern alone is not something a run can index with.
        (masters.length > 0 ? out.needsDictionary : out.missing).push(name);
        continue;
      }
      const best = selectBestDict(dicts, currentPc, detectorShape, geom);
      if (!best) {
        out.needsDictionary.push(name);       // dictionaries exist, none fits this detector
        continue;
      }
      out.matched.push(best);
      out.dictPaths[best.path] = best.path;   // ship exactly this one, not the formula's pick
      const fits = dicts.filter(
        (d) => d !== best && dictShapeMatches(d, detectorShape) && dictGeometryMatches(d, geom),
      );
      if (fits.length > 0) {
        out.several.push({ name, taken: baseName(best), others: fits.map(baseName) });
      }
      continue;
    }

    if (linked.length === 0) {
      out.missing.push(name);
      continue;
    }
    out.matched.push(linked[0]);
    if (linked.length > 1) {
      out.several.push({
        name, taken: baseName(linked[0]), others: linked.slice(1).map(baseName),
      });
    }
  }
  return out;
}
