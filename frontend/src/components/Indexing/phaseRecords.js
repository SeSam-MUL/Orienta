/**
 * One record per phase for the indexing request.
 *
 * The page holds a list of phases, each with the file it brings — a CIF for
 * Hough, a master .h5 for Dictionary, an .sht for Spherical. The request used
 * to carry three lists built by three independent `.filter()` passes over the
 * flat path list, one per extension. That loses which phase is which: a run
 * with
 *
 *     [Al.cif, alpha.sht, Cu.h5]
 *
 * ships `cif_paths: [Al.cif]`, `sht_paths: [alpha.sht]`,
 * `master_h5_paths: [Cu.h5]` — three lists of length 1 whose first entries are
 * three DIFFERENT phases. The backend then runs the method's own list and
 * names the phases from `cif_paths or master_h5_paths or sht_paths`, so a
 * spherical run of the alpha phase came back labelled "Al" — and the per-phase
 * master was matched back from that label, so it rendered against Al's
 * master too. Both lists are well-formed; nothing downstream can see it.
 *
 * The backend accepts both forms and derives the three lists from these
 * records when they are present (see PhaseFiles in routes/indexing.py), so
 * this is the only place the split by extension happens.
 */

/** Which field a path belongs in, or null for something we do not run. */
export function phaseFileField(path) {
  const p = String(path || '').toLowerCase();
  if (p.endsWith('.cif')) return 'cif';
  if (p.endsWith('.sht')) return 'sht';
  if (/\.(h5|hdf5)$/.test(p)) return 'master';
  return null;
}

/**
 * `[{name?, cif?, sht?, master?}]` — one entry per phase that brings a file.
 *
 * A path no method runs on (anything but .cif/.sht/.h5/.hdf5 — the file picker
 * allows any file when the method is "embedding") yields NO record. Nothing
 * indexes `phases` positionally: the backend derives the per-method list and
 * the per-phase name source from the records themselves, so an omission here
 * cannot shift anything, while an EMPTY record can only be a phase nothing can
 * run — which the backend refuses outright rather than let it reduce the phase
 * count in silence.
 *
 * @param paths  the effective file of each phase, in the order the page shows
 *               them (Dictionary substitution already applied).
 * @param labels optional human label per path, for the backend log only. The
 *               backend derives its own names from the files, deliberately:
 *               the name on the map and the name a later lookup matches
 *               against have to come from one derivation.
 */
export function buildPhaseRecords(paths, labels = {}) {
  const records = [];
  for (const path of paths || []) {
    const field = phaseFileField(path);
    if (!field) continue;
    const record = { [field]: path };
    const label = labels?.[path];
    if (label) record.name = label;
    records.push(record);
  }
  return records;
}
