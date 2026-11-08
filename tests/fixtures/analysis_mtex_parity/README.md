# MTEX reference numbers for analysis/ (synthetic map)

- Input: `tests/synthetic_grain_map.py`, `add_defects(make_map())`, default seeds, then
  every indexed pixel moved to a random symmetry branch (`scramble_branches`, seed 5),
  as files store reduced orientations. The stored `quats` are that scrambled input.
- Convention: an orix quaternion is the conjugate of the MTEX rotation of the same
  orientation (`orix_to_mtex = "conjugate"`), decided by the script's KAM gate, which
  only the conjugate mapping passes on the scrambled input.
- Output: computed by MTEX for Python 1.0.0 (GPL-2.0-only), run as a separate
  program (a generator script that is not part of the published tree) in its own
  Python 3.12 environment. This file holds only numbers; no MTEX code is part
  of Orienta, and no Orienta module or test imports mtex.
- Parameter sets: p0 = 5 deg / minPixel 3 / alpha 3.1, p1 = 5 deg / 3 / 1.5,
  p2 = 15 deg / 5 / 3.1. KAM order 2, threshold 7 deg.
- Per parameter set and grain: GOS, numPixel, area, aspectRatio, perimeter,
  perimeterWithInclusion, `<p>_is_boundary` (`grains.isBoundary`: the grain touches the map border or a
  not-indexed area open to it; one entry per MTEX grain, not-indexed grains included); grain boundary segments between two kept measurements
  (`<p>_gb_pairs` flat indices a < b, summed `segLength` in um, misorientation angle in deg,
  NaN across a phase boundary); `<p>_fill_grain`: the grain `fill(ebsd, grains)` gives each
  not-indexed pixel it fills (0 = not filled).
- Regenerate: create the environment (`conda create -p <dir> -c conda-forge
  --override-channels python=3.12 pip`, then `pip install "mtex[all]==1.0.0" orix h5py`)
  and run the generator script (not distributed) with that interpreter.

# cells_probes.npz (cell decomposition, small probe maps)

- Input: 19 probe maps built with numpy by a generator script
  (bands, crosses, notches at the border, rectangles, a disc, a diagonal band, a corner block,
  an island, random speckle, smoothed random blobs; alpha 3.1, three maps also at 1.5). Every
  pixel has its own orientation 0.5-2 deg from the identity (not stored; no stored number depends
  on the values).
- Output: computed by MTEX for Python 1.0.0 as a separate program in its own environment, as above.
  Angle 1e-4 deg (nothing joins): pairs of neighbouring measurement cells from `grains.boundary`
  with their summed `segLength`. Angle 5 deg (every connected part one grain): per grain `area`,
  `perimeter`, `perimeterWithInclusion`, `aspectRatio`, `numPixel`, and the grain of every pixel
  `fill(ebsd, grains)` fills.
- Regenerate: the generator script is not distributed; the numbers are fixed reference data.

# real_is_boundary.npz (MTEX `isBoundary` per pixel, four real maps)

- Maps (four real scans that are not distributed: two Oxford `.h5oina`, two `.ang`): an edited `.ang`,
  a ground-truth `.h5oina`, an Oxford `.h5oina` and a batch `.ang`;
  5 deg / minPixel 3 / alpha 3.1.
- Output: MTEX for Python 1.0.0 run as a separate program, reduced by
  a generator script to numbers only: per map the shape, the pixels MTEX
  puts in a grain of an indexed phase, and `isBoundary` of that grain for each of them (bit-packed,
  row-major). Used by `tests/test_grain_edge_rules.py`, which skips without the data files.

# texture_components.npz (texture components as a reference MTEX texture script builds and counts them)

- Script: the reference MTEX texture script, lines 17-30 (`sym_X = unique(specimenSymmetry('mmm') *
  orientation.byMiller(hkl, uvw, CS))`, `byEuler` for the two Euler components) and 154-167 (`volEBSD_X` = sum of
  `volume(oriR, sym_X(i), tol)*100` over the first 1, 2 or 4 variants).
- Output: MTEX for Python 1.0.0 run as a separate program by a generator script
  in its own environment; numbers only. In the file, quaternions are MTEX rotations (an orix quaternion is
  their conjugate).
  - `ideal_mtex` (14, 4): `byMiller` / `byEuler` of the 14 FCC preset components, `names` in the preset's order;
    `variants_mtex` (14, 4, 4): `unique(rs * ideal)` in MTEX's order (NaN padded), `n_distinct` its length, `n_used` the
    number the script sums (equal for every component, so the choice of "the first n" never selects).
  - `quats_mtex` (9000, 4): 3000 uniformly random orientations and 6000 within 25 deg of a random variant of a random
    component, each through a random cubic operator; `vol_pct_5|10|15` (14,): the script's `volEBSD_X` for them.
  - `edge_eps`, `edge_volume`: `volume` of a single rotation about z by tol + eps; `nan_quats`, `nan_volume`: a list
    with a NaN entry (counted in the length, in no count).
- Measured semantics (all from outputs): `byMiller(hkl, uvw)` takes `hkl` to z and the symmetry equivalent of `uvw`
  closest to perpendicular to `hkl` to x (all perpendicular equivalents give the same set of variants for the 12 Miller
  components); `specimenFrame('mmm') * ori` is not a list in MTEX for Python, the list is `rotation(...) * ori` over the
  eight elements of `specimenFrame('mmm').rot`; `unique` keeps the first of equal orientations (modulo the crystal
  symmetry); `volume(list, centre, tol)` is the count of list entries within `tol` of `centre` modulo the crystal
  symmetry, divided by the list length (a NaN entry is in the length only), crossing at `tol` within 1e-15 rad.
- Used by `tests/test_texture_components_mtex.py`.
