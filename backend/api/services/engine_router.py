"""Auto-routing of a forward-sim job to the EMsoft or GPU engine.

The decision is the measured heuristic (2026-06-19 benchmark; the thresholds are
*defaults* the user can still override via the manual EMsoft/GPU toggle).  The
three inputs are:

* the reflection count at the job's **effective** dmin (the per-phase
  recommended dmin or the user override — NOT the 0.05 floor; at the floor the
  counts are 2-8x larger and the heuristic mis-routes — verified 2026-06-19),
* the crystal point-group order (number of unique rotation matrices in the
  space group), and
* EMsoft/WSL availability.

**ORDER OF THE RULES IS LOAD-BEARING** (see :func:`recommend_engine`): the
``reflections >= 8000 -> gpu`` test must run BEFORE the
``point_group_order >= 12 -> emsoft`` test, otherwise the T-phase
(pg=24, refl=12196) routes to EMsoft — which did-not-finish in the benchmark.

Caveats (documented; see ``tasks/forward_sim/iterations/AUTOROUTE/design.md`` §5):

1. All benchmark data is npx=50; production npx (≈500) shifts the absolute
   crossovers (relative ordering is robust).
2. The 4000 / 8000 reflection crossovers are coarse and UNMEASURED between
   cells.  The triclinic cell ``Fe3_Al2_Si3`` sits on the crossover (gpu at
   dmin>=0.07, emsoft at dmin=0.05) — this module honours the effective dmin so
   the route tracks the actual workload.
3. Reflection count MUST be computed at the effective job dmin.
4. The Monte-Carlo step stays EMsoft when available (EMsoft GPU-MC ≈ 500x
   faster); auto routing concerns the MASTER/dynamical step + which pipeline a
   job uses.
5. The manual EMsoft/GPU toggle remains the override; auto is opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Routing thresholds — measured-benchmark defaults (npx=50; coarse between cells).
REFL_GPU_FORCE = 8000       # >= this many reflections -> GPU (EMsoft impractical)
PG_ORDER_EMSOFT = 12        # >= this point-group order -> EMsoft (symmetry wedge)
REFL_GPU_LOWSYM = 4000      # <= this many refl AND low symmetry -> GPU


@dataclass
class EngineRecommendation:
    """The outcome of :func:`recommend_engine`."""

    engine: str               # 'emsoft' | 'gpu'
    reason: str               # human-readable, includes the deciding numbers
    reflections: int          # refl count at the effective dmin
    point_group_order: int    # unique rotation matrices in the space group
    emsoft_available: bool


def point_group_order(space_group: int) -> int:
    """Number of UNIQUE rotation matrices (R parts) of the space group.

    Uses the same machinery as
    :func:`backend.forward_sim.crystal.structure_matrix._expand_symmetry_orbit`:
    ``diffpy.structure.spacegroups.GetSpaceGroup(sg).symop_list``, deduping the
    rotation part ``op.R`` (the translation ``op.t`` is irrelevant to the point
    group).

    Reference values: cubic m-3m=48, 4/mmm=16, -6m2=12, m-3=24, 2/m=4, -1=2.

    Args:
        space_group: international space-group number (1..230).

    Returns:
        The point-group order (count of distinct rotation matrices).
    """
    from diffpy.structure.spacegroups import GetSpaceGroup

    sg = GetSpaceGroup(int(space_group))
    seen = set()
    for op in sg.symop_list:
        seen.add(tuple(np.round(np.asarray(op.R, dtype=float), 6).ravel()))
    return len(seen)


def recommend_engine(
    *,
    reflections: int,
    point_group_order: int,
    emsoft_available: bool,
) -> EngineRecommendation:
    """Pure decision function (no I/O) — ORDER OF THE TESTS IS LOAD-BEARING.

    1. ``not emsoft_available``              -> gpu    (only engine)
    2. ``reflections >= REFL_GPU_FORCE``     -> gpu    (EMsoft impractical at N)
    3. ``point_group_order >= PG_ORDER_EMSOFT`` -> emsoft (symmetry-wedge wins)
    4. ``reflections <= REFL_GPU_LOWSYM``    -> gpu    (low symmetry + moderate N)
    5. else                                  -> emsoft (low sym but refl-heavy)

    Args:
        reflections: reflection count at the job's EFFECTIVE dmin.
        point_group_order: number of unique rotation matrices in the space group.
        emsoft_available: whether EMsoft/WSL can be used at all.

    Returns:
        An :class:`EngineRecommendation` whose ``reason`` embeds the deciding
        numbers.
    """
    n = int(reflections)
    pg = int(point_group_order)
    if not emsoft_available:
        return EngineRecommendation(
            "gpu",
            "EMsoft/WSL not available — GPU is the only engine",
            n, pg, False,
        )
    if n >= REFL_GPU_FORCE:
        return EngineRecommendation(
            "gpu",
            f"very large reflection list ({n}) — EMsoft impractical at this size",
            n, pg, True,
        )
    if pg >= PG_ORDER_EMSOFT:
        return EngineRecommendation(
            "emsoft",
            f"high symmetry (PG order {pg}) — EMsoft's symmetry-wedge reduction "
            f"is far faster",
            n, pg, True,
        )
    if n <= REFL_GPU_LOWSYM:
        return EngineRecommendation(
            "gpu",
            f"low symmetry (PG {pg}) + moderate size ({n} refl) — GPU faster "
            f"(EMsoft has little symmetry to exploit)",
            n, pg, True,
        )
    return EngineRecommendation(
        "emsoft",
        f"low symmetry but reflection-heavy ({n} refl) — EMsoft's lower "
        f"per-reflection cost wins (our dense solve is memory-bound)",
        n, pg, True,
    )


def recommend_engine_for_xtal(
    xtal_path: str,
    dmin: float,
    *,
    emsoft_available: Optional[bool] = None,
) -> EngineRecommendation:
    """I/O wrapper: read the ``.xtal``, count reflections AT ``dmin``, route.

    Args:
        xtal_path: absolute path to the ``.xtal`` (or master ``.h5``) file.
        dmin: the EFFECTIVE dmin the job will run with (nm).
        emsoft_available: ``None`` -> probe via
            :func:`backend.api.services.gpu_sim_runner._wsl_available` (a cheap
            ``test -f`` probe, ~ms-100ms).  Pass an explicit bool to skip the
            probe (the batch / scan-missing path probes ONCE and reuses it for
            all phases).

    Returns:
        An :class:`EngineRecommendation`.

    Raises:
        FileNotFoundError: if ``xtal_path`` does not exist.
    """
    from backend.forward_sim.crystal.xtal_io import read_crystal_structure
    from backend.forward_sim.crystal.structure_matrix import reflection_list

    if not Path(xtal_path).exists():
        raise FileNotFoundError(f"xtal file not found: {xtal_path}")
    if emsoft_available is None:
        from backend.api.services.gpu_sim_runner import _wsl_available
        emsoft_available = _wsl_available()
    structure = read_crystal_structure(xtal_path)
    n = int(reflection_list(structure, float(dmin)).shape[0])
    pg = point_group_order(structure.space_group)
    return recommend_engine(
        reflections=n,
        point_group_order=pg,
        emsoft_available=bool(emsoft_available),
    )
