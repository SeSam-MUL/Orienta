/**
 * X-ray emission lines for elements likely to appear in EBSD samples.
 * Source: NIST X-ray Transition Energies (https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html).
 * Energies in keV. Only the practically-useful lines for EDS are listed.
 */
export const EDS_LINES = {
  H:  { Ka1: 0.0136 },
  Be: { Ka1: 0.108 },
  B:  { Ka1: 0.183 },
  C:  { Ka1: 0.277 },
  N:  { Ka1: 0.392 },
  O:  { Ka1: 0.525 },
  F:  { Ka1: 0.677 },
  Na: { Ka1: 1.041, Kb1: 1.071 },
  Mg: { Ka1: 1.254, Kb1: 1.302 },
  Al: { Ka1: 1.487, Kb1: 1.557 },
  Si: { Ka1: 1.740, Kb1: 1.836 },
  P:  { Ka1: 2.013, Kb1: 2.139 },
  S:  { Ka1: 2.308, Kb1: 2.464 },
  Cl: { Ka1: 2.622, Kb1: 2.815 },
  K:  { Ka1: 3.314, Kb1: 3.589 },
  Ca: { Ka1: 3.692, Kb1: 4.013 },
  Ti: { Ka1: 4.511, Kb1: 4.932 },
  V:  { Ka1: 4.952, Kb1: 5.427 },
  Cr: { Ka1: 5.415, Kb1: 5.947 },
  Mn: { Ka1: 5.899, Kb1: 6.490 },
  Fe: { Ka1: 6.404, Kb1: 7.058, La1: 0.705 },
  Co: { Ka1: 6.930, Kb1: 7.649 },
  Ni: { Ka1: 7.478, Kb1: 8.265 },
  Cu: { Ka1: 8.048, Kb1: 8.905, La1: 0.930 },
  Zn: { Ka1: 8.639, Kb1: 9.572, La1: 1.012 },
  Ga: { Ka1: 9.252, Kb1: 10.264 },
  Ge: { Ka1: 9.886, Kb1: 10.982 },
  As: { Ka1: 10.544, Kb1: 11.726 },
  Mo: { Ka1: 17.479, La1: 2.293 },
  Ag: { Ka1: 22.163, La1: 2.984 },
  Sn: { Ka1: 25.271, La1: 3.444 },
  W:  { La1: 8.398, Lb1: 9.672, Ma1: 1.775 },
  Pt: { La1: 9.443, Ma1: 2.051 },
  Au: { La1: 9.713, Ma1: 2.123 },
  Pb: { La1: 10.551, Ma1: 2.342 },
  Nb: { Ka1: 16.615, La1: 2.166 },
  Zr: { Ka1: 15.775, La1: 2.042 },
  Hf: { La1: 7.899, Ma1: 1.645 },
  Ta: { La1: 8.146, Ma1: 1.710 },
  Y:  { Ka1: 14.958, La1: 1.923 },
  La: { La1: 4.651 },
  Ce: { La1: 4.840 },
};

export function getElementLines(symbol) {
  const el = EDS_LINES[symbol];
  if (!el) return [];
  const lines = Object.entries(el).map(([line, energy]) => ({ line, energy }));
  lines.sort((a, b) => a.energy - b.energy);
  return lines;
}
