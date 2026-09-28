/**
 * What the profile card shows, decided here so it can be tested without a
 * browser (spec §2.3).
 *
 * NAMED `phaseCardRows`, because `PhaseCard.jsx` sits beside it and Windows
 * cannot tell two names apart that differ only in case. THIRD time in this
 * folder in one day -- ElementFacets, IdentityLine, and this -- each time
 * the import silently returned the logic module and every test that
 * rendered the component died on "Element type is invalid ... undefined".
 * The convention is written in `identityParts.js` too, and writing it down
 * has not been enough; `caseCollisions.test.js` walks all of src/ and the
 * PhaseLibrary suite now carries the same check, so a narrow run catches it.
 *
 * ONE CARD, EVERYWHERE THE SAME. Today there is none: mockup B showed files
 * and simulation parameters without a cell or a composition, C did the
 * opposite. So this module lists the rows once and every caller draws the
 * same list.
 *
 * TWO RULES THAT ARE NOT LAYOUT:
 *
 * 1. NEVER AN ABSOLUTE PATH. The endpoint's SHT provenance carries one, and
 *    on a developer machine it reads `C:\Users\<account>\...`. This
 *    project has already put a local path into a manuscript, and the
 *    guarantee written afterwards turned out to be weaker off-Windows than
 *    on it. The card shows the file's NAME, and the repo-relative `rel` from
 *    `deep_link` when a path is what the reader wants.
 *
 * 2. A CARD FOR A BROKEN FILE IS NOT AN EMPTY CARD. `sd_1816951` parses to
 *    two different compositions and the backend refuses it, so it has no
 *    Hough capability -- but it has an .xtal, an .sht and a master, and the
 *    card shows what those give. It says the CIF is unreadable and it
 *    invents nothing.
 */

/** Lattice constants, at a precision that is a tenth of a picometre. */
export function cellRows(cell) {
  if (!cell) return [];
  const rows = [];
  for (const k of ['a', 'b', 'c']) {
    if (typeof cell[k] === 'number') {
      rows.push({ key: k, text: `${Number(cell[k].toFixed(4))} Å` });
    }
  }
  for (const k of ['alpha', 'beta', 'gamma']) {
    if (typeof cell[k] === 'number') {
      rows.push({ key: k, text: `${Number(cell[k].toFixed(4))}°` });
    }
  }
  return rows;
}

/**
 * The composition twice, side by side, with the difference computed rather
 * than curated.
 *
 * c1's idea, and it is the right one: two phases in this library are named
 * for an element their structure does not contain, and a maintained list of
 * "known discrepancies" would go stale the day someone adds a CIF. A
 * computed difference cannot.
 */
export function compositionRows(card) {
  const structure = card.elements_structure || [];
  const label = card.elements_label || [];
  const all = [...new Set([...structure, ...label])].sort();
  return all.map((element) => ({
    element,
    inStructure: structure.includes(element),
    inLabel: label.includes(element),
    disagrees: structure.includes(element) !== label.includes(element),
  }));
}

/** The four file slots, in the order the pipeline produces them. */
export const FILE_SLOTS = ['cif', 'xtal', 'master', 'sht', 'dictionary'];

/**
 * One row per file: present or not, its name, its date, and a
 * repo-relative path -- never an absolute one.
 *
 * A cross rather than a colour, because red alone reads as an error and a
 * phase without a dictionary is not an error.
 */
export function fileRows(card) {
  const files = card.files || {};
  const links = card.deep_link || {};
  return FILE_SLOTS.map((slot) => {
    const f = files[slot];
    return {
      slot,
      present: Boolean(f),
      name: f ? f.name : null,
      date: f ? f.date : null,
      bytes: f ? f.bytes : null,
      rel: (links[slot] && links[slot].rel) || (f && f.rel) || null,
      category: (links[slot] && links[slot].category) || null,
    };
  });
}

/**
 * The simulation parameters worth showing, out of a namelist with fifty
 * entries.
 *
 * Chosen, not dumped: these are the ones that decide whether a master is
 * usable for a given measurement. `EkeV` because a 20 kV master is no use
 * at 15, `dmin` because it sets the reflector range and is the parameter
 * that makes a large cell take days, `npx` for the resolution, `sig` for
 * the sample tilt the simulation assumed.
 */
export function masterParameters(card) {
  const nml = ((card.simulation || {}).master) || {};
  const ebsd = nml.EBSDMasterNameList || {};
  const mc = nml.MCCLNameList || {};
  const out = [];
  const push = (key, value, unit) => {
    if (value !== undefined && value !== null) out.push({ key, value, unit });
  };
  push('energy', mc.EkeV, 'kV');
  push('dmin', ebsd.dmin ? Number(ebsd.dmin.toFixed(4)) : ebsd.dmin, 'nm');
  push('npx', ebsd.npx, null);
  push('tilt', mc.sig, '°');
  push('electrons', mc.totnum_el || null, null);
  push('program', nml.program, null);
  return out;
}

/** The same for the .sht, whose sidecar states them directly. */
export function shtParameters(card) {
  const sht = ((card.simulation || {}).sht) || {};
  const p = sht.parameters || {};
  const out = [];
  const push = (key, value, unit) => {
    if (value !== undefined && value !== null) out.push({ key, value, unit });
  };
  push('energy', p.voltage_kV, 'kV');
  push('dmin', p.dmin, 'nm');
  push('bandwidth', p.bandwidth, null);
  push('tilt', p.sig, '°');
  push('electrons', p.electrons, null);
  push('engine', p.engine, null);
  return out;
}

/**
 * Where the .sht came from, WITHOUT the absolute paths the endpoint sends.
 *
 * `found` is the useful half anyway: it is measured locally, so it answers
 * "is the file this was built from still here" -- which a path does not.
 */
export function shtProvenance(card) {
  const prov = (((card.simulation || {}).sht) || {}).provenance || {};
  const sources = [];
  for (const [slot, key] of [['xtal', 'source_xtal'], ['cif', 'source_cif']]) {
    const s = prov[key];
    // `elsewhere` is the one fact the server went to trouble to keep:
    // when the source cannot be found here AND the sidecar recorded an
    // absolute path from outside this project, the master was built
    // against a LIBRARY THAT IS NOT THIS ONE. The path itself is thrown
    // away -- it names somebody's account -- and this boolean is what
    // survived. Nothing in the frontend read it, so the card said only
    // "not found", which reads as a missing file rather than as a master
    // whose provenance belongs to another machine.
    if (s && s.name) {
      sources.push({ slot, name: s.name, found: Boolean(s.found),
        elsewhere: Boolean(s.recorded_elsewhere) });
    }
  }
  return {
    sources,
    reference: prov.reference || null,
    engine: prov.engine || null,
    origin: prov.origin || null,
  };
}

/**
 * What the card must SAY rather than show, as a list of notices.
 *
 * Each one is a sentence the reader needs before they trust the rest of the
 * card, and each corresponds to something measured in this library.
 */
export function notices(card) {
  const out = [];
  if (card.parse_error) out.push({ kind: 'parseError', detail: card.parse_error });
  if (card.elements_disagree) out.push({ kind: 'elementsDisagree' });
  if (card.cell_setting_conflict) {
    out.push({ kind: 'cellConflict', detail: card.cell_setting_conflict });
  }
  if ((card.cell_settings_available || []).length > 1) {
    out.push({
      kind: 'severalSettings',
      detail: (card.cell_settings_available || []).join(', '),
      shown: card.cell_setting || null,
    });
  }
  if (card.reference_rejected) {
    out.push({ kind: 'referenceRejected', detail: card.reference_rejected });
  }
  return out;
}
