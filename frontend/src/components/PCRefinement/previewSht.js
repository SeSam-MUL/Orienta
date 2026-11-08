/**
 * Which master (.sht) the forward-sim preview should render for the loaded phase.
 *
 * The preview used to take the first file the library listed. With Al and Ni
 * both simulated, loading Ni for the calibration showed a Ni pattern next to an
 * Al simulation and a red "NCC = 0.03" - a correct PC, reported as a bad one.
 *
 * Match on the phase's own identity: its display label or formula, the CIF
 * stem in the file name's parentheses (". (sd_0302719) [cI168] ."), the file
 * name's leading formula ("Ni (Ni) [cF4] {20kV}.sht"), and last its material
 * folder - which is shared: the "Al" folder also holds Al6Fe, Al7FeCu2, Al2Cu,
 * MnAl6 and the pi phase. A rule that names more than one master decides
 * nothing.
 *
 * A phase without a master of its own gets NONE (null), never another phase's:
 * alpha- and beta-Ti loaded from their own CIFs were shown the
 * Mg master and an NCC of 0.016, which reads as a bad pattern centre. The page
 * says that this phase has no master and lets the user pick one on purpose.
 *
 * @param {Array<{path:string, filename?:string, material?:string,
 *                display_label?:string, formula?:string}>} files
 * @param {string|null} phaseName
 * @returns {string|null} the chosen path, or null when no master is the phase's own
 */
export function pickShtForPhase(files, phaseName) {
  const list = Array.isArray(files) ? files.filter((f) => f && f.path) : [];
  const want = String(phaseName || '').trim().toLowerCase();
  if (list.length === 0 || !want) return null;
  const same = (v) => String(v || '').trim().toLowerCase() === want;
  const fileName = (f) => String(f.filename || f.path.split(/[\\/]/).pop() || '').toLowerCase();
  // Each rule answers only when it names exactly one master: "Ti" is both the
  // alpha and the beta master's label, "Fe" both ferrite's and austenite's.
  const rules = [
    (f) => same(f.display_label) || same(f.formula),
    // `(sd_0302719)`: a phase loaded from its CIF is named by the CIF stem,
    // which the master's file name carries in parentheses after the formula.
    (f) => fileName(f).includes(`(${want})`),
    (f) => fileName(f).startsWith(`${want} (`) || fileName(f).startsWith(`${want}.`),
    (f) => same(f.material),
  ];
  // Several hits that are one phase simulated more than once (the name before
  // "{20kV}" is the same, only the beam energy differs) are that phase's own
  // masters: take the first, as before. Hits from different phases ("Ti" is the
  // label of alpha AND beta titanium) decide nothing.
  const head = (f) => fileName(f).split('{')[0].trim();
  for (const rule of rules) {
    const hits = list.filter(rule);
    if (hits.length === 1) return hits[0].path;
    if (hits.length > 1 && hits.every((f) => head(f) === head(hits[0]))) return hits[0].path;
  }
  return null;
}
