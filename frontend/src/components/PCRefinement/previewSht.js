/**
 * Which master (.sht) the forward-sim preview should render for the loaded phase.
 *
 * The preview used to take the first file the library listed. With Al and Ni
 * both simulated, loading Ni for the calibration showed a Ni pattern next to an
 * Al simulation and a red "NCC = 0.03" — a correct PC, reported as a bad one.
 *
 * Match on the phase's own identity first (material folder, display label,
 * formula), then on the file name's leading formula ("Ni (Ni) [cF4] {20kV}.sht") or
 * the CIF stem in its parentheses ("… (sd_0302719) [cI168] …"),
 * and only then fall back to the first file, so a phase without its own master
 * still gets a preview rather than none.
 *
 * @param {Array<{path:string, filename?:string, material?:string,
 *                display_label?:string, formula?:string}>} files
 * @param {string|null} phaseName
 * @returns {string|null} the chosen path
 */
export function pickShtForPhase(files, phaseName) {
  const list = Array.isArray(files) ? files.filter((f) => f && f.path) : [];
  if (list.length === 0) return null;
  const want = String(phaseName || '').trim().toLowerCase();
  if (want) {
    const same = (v) => String(v || '').trim().toLowerCase() === want;
    const byIdentity = list.find(
      (f) => same(f.material) || same(f.display_label) || same(f.formula),
    );
    if (byIdentity) return byIdentity.path;
    const byName = list.find((f) => {
      const name = String(f.filename || f.path.split(/[\\/]/).pop() || '').toLowerCase();
      // `(sd_0302719)`: a phase loaded from its CIF is named by the CIF stem,
      // which the master's file name carries in parentheses after the formula.
      return name.startsWith(`${want} (`) || name.startsWith(`${want}.`)
        || name.includes(`(${want})`);
    });
    if (byName) return byName.path;
  }
  return list[0].path;
}
