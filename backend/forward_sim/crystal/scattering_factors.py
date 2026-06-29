"""Weickenmeier-Kohl (WK) complex electron scattering factor (SP0, Task 3).

This module provides :func:`wk_scattering_factor`, the most NCC-critical primitive
of the forward model.  It returns the **complex** electron atomic scattering
factor ``f = f_el + i·f_abs`` in Angstrom, parametrised exactly as EMsoft does
(``others.f90::WEKO/FSCATT`` + ``diffraction.f90::CalcUcg``):

* **Elastic (real) part** — the WEKO function
  ``f_el(S) = Σ_{i=1..4} A_i·(1 - exp(-B_i·S²)) / S²`` with the 4 ``(A_i, B_i)``
  coefficient pairs per element, taken verbatim from the EMsoft ``GETWK`` DATA
  table (see :data:`WK_COEFFS`).  The ``S`` variable is scaled from the input
  ``s = |g| = 1/d`` (the full reciprocal-vector length, in nm⁻¹) by
  ``S = 0.05·s`` (the reconciliation below).
* **Debye-Waller damping** — ``exp(-0.5·UL²·G²)`` with ``UL = sqrt(B·dwwk)``,
  ``dwwk = 100/(8π²)`` mapping the EMsoft B-factor (nm²) to the RMS displacement
  ``UL`` (Å), and ``G = s·swk`` the FSCATT argument (``swk = 0.1·2π``).
* **Relativistic correction** — multiply the elastic part by ``γ = (V_kV+511)/511``
  (EMsoft ``FSCATT`` with ``ACCFLG = .TRUE.``).
* **Absorptive (imaginary) part** — the WK phonon (thermal-diffuse) form factor
  ``FPHON`` (EMsoft ``others.f90``), implemented faithfully (NOT a placeholder).
  See the note on :func:`_fphon` about its CPU/non-autograd nature and its role
  as a diagnose-loop knob.

Units & convention
------------------
``s = |g| = 1/d`` in **nm⁻¹** — the full reciprocal-vector length (EMsoft
``rlp%g``), NOT the half-angle ``sin(θ)/λ``.  The internal WEKO variable
``S = 0.05·s = 0.05·|g|`` (= ``rlp%g·swk/4π``) IS the physical ``sin(θ)/λ`` in
**Å⁻¹**.  The returned scattering factor is in **Angstrom** (physical ``f_e``).
The EMsoft ``4π``/``pref``/``preg`` normalisation that turns ``f_e`` into the
potential coefficient ``U_g`` lives downstream in ``compute_Ug_table`` (Task 5),
NOT here.

S-variable reconciliation (AMP_2 fix, verified 2026-06-18)
---------------------------------------------------------
EMsoft ``diffraction.f90::CalcUcg`` (WK branch) sets ``s = rlp%g·swk`` with
``rlp%g = CalcLength(hkl,'r') = |g| = 1/d`` (nm⁻¹) and ``swk = 0.1·2π``, then calls
``FSCATT(s, …)`` whose documented argument is ``G/(4π) = S = sin(θ)/λ``.  Hence the
WEKO scattering variable is ``S = rlp%g·swk/(4π) = 0.05·|g|`` — with ``|g|`` in nm⁻¹
this is exactly the physical ``sin(θ)/λ`` expressed in Å⁻¹ (the units WK's A[Å],
B[Å²] coefficients require).  Because ``S_SCALE = SWK/4π = 0.05`` already bundles
BOTH the nm⁻¹→Å⁻¹ factor (×0.1) AND the ``|g|→sin(θ)/λ`` factor (×0.5) — i.e.
``0.05 = 0.1·0.5`` — the caller must pass ``s = |g|`` (full length), NOT ``|g|/2``.

An earlier draft (pre-AMP_2) passed ``s = |g|/2`` from ``compute_Ug_table``, which
applied the ÷2 a SECOND time → ``S = 0.025·|g|``, half the EMsoft value.  This was
masked at low ``s`` (the Kirkland 5 % cross-check still "passed"), but inflated
``f_e`` |g|-dependently (Doyle-Turner ratio rose 1.07→3.9 toward high ``|g|``),
compressing dynamical band contrast on dense cells.  The fix (caller passes
``s = |g|``) restores the EMsoft WEKO ``S`` and the absorptive FPHON/DEWA/FCORE
``G = swk·s`` simultaneously.  The s→0 analytic limit is unchanged: ``f_el(0) =
Σ A_i·B_i`` (matches Kirkland ``f_e(0)`` to <0.7 %).  Full details + citations in
``tasks/forward_sim/lessons.md``.  Sources: Weickenmeier & Kohl 1991 (Acta Cryst
A47 590); Callahan & De Graef 2013 (Microsc Microanal 19 1255); Doyle & Turner 1968
(Acta Cryst A24 390); EMsoft ``others.f90`` FSCATT/WEKO + ``diffraction.f90``
CalcUcg.

Sources
-------
* A. Weickenmeier & H. Kohl, *Acta Cryst.* A47, 590-597 (1991).
* EMsoft ``others.f90`` (WEKO/FSCATT/FPHON/GETWK) and ``diffraction.f90``
  (CalcUcg): https://github.com/EMsoft-org/EMsoft
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

# --- WEKO elastic coefficients (A_i, B_i), verbatim from EMsoft others.f90 ----
# Full GETWK ``AA(4,98)``/``BB(4,98)`` ``reshape`` arrays — every element Z=1..98,
# transcribed verbatim from EMsoft ``Source/EMsoftLib/others.f90`` and cross-checked
# bit-for-bit against EMsoftOO ``Source/EMsoftOOLib/mod_others.f90`` (fetched
# 2026-06-11).  A in Å, B in Å² (WK units).  The Al(13)/Ni(28) rows are unchanged
# from the SP0 table (verified 2026-06-10); Mg(12)/Fe(26)/Zn(30) match the
# iter9-injected values (lessons.md §21).  This unlocks master-pattern builds for
# ANY crystal/CIF whose species fall in Z=1..98 (no more Al/Ni-only restriction).
WK_COEFFS: Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    1: (  # H
        (0.00427, 0.00957, 0.00802, 0.00209),
        (4.17218, 16.05892, 26.78365, 69.45643),
    ),
    2: (  # He
        (0.01217, 0.02616, -0.00884, 0.01841),
        (1.83008, 7.20225, 16.13585, 18.75551),
    ),
    3: (  # Li
        (0.00251, 0.03576, 0.00988, 0.02370),
        (0.02620, 2.00907, 10.80597, 130.49226),
    ),
    4: (  # Be
        (0.01596, 0.02959, 0.04024, 0.01001),
        (0.38968, 1.99268, 46.86913, 108.84167),
    ),
    5: (  # B
        (0.03652, 0.01140, 0.05677, 0.01506),
        (0.50627, 3.68297, 27.90586, 74.98296),
    ),
    6: (  # C
        (0.04102, 0.04911, 0.05296, 0.00061),
        (0.41335, 10.98289, 34.80286, 177.19113),
    ),
    7: (  # N
        (0.04123, 0.05740, 0.06529, 0.00373),
        (0.29792, 7.84094, 22.58809, 72.59254),
    ),
    8: (  # O
        (0.03547, 0.03133, 0.10865, 0.01615),
        (0.17964, 2.60856, 11.79972, 38.02912),
    ),
    9: (  # F
        (0.03957, 0.07225, 0.09581, 0.00792),
        (0.16403, 3.96612, 12.43903, 40.05053),
    ),
    10: (  # Ne
        (0.02597, 0.02197, 0.13762, 0.05394),
        (0.09101, 0.41253, 5.02463, 17.52954),
    ),
    11: (  # Na
        (0.03283, 0.08858, 0.11688, 0.02516),
        (0.06008, 2.07182, 7.64444, 146.00952),
    ),
    12: (  # Mg
        (0.03833, 0.17124, 0.03649, 0.04134),
        (0.07424, 2.87177, 18.06729, 97.00854),
    ),
    13: (  # Al
        (0.04388, 0.17743, 0.05047, 0.03957),
        (0.09086, 2.53252, 30.43883, 98.26737),
    ),
    14: (  # Si
        (0.03812, 0.17833, 0.06280, 0.05605),
        (0.05396, 1.86461, 22.54263, 72.43144),
    ),
    15: (  # P
        (0.04166, 0.17817, 0.09479, 0.04463),
        (0.05564, 1.62500, 24.45354, 64.38264),
    ),
    16: (  # S
        (0.04003, 0.18346, 0.12218, 0.03753),
        (0.05214, 1.40793, 23.35691, 53.59676),
    ),
    17: (  # Cl
        (0.04245, 0.17645, 0.15814, 0.03011),
        (0.04643, 1.15677, 19.34091, 52.88785),
    ),
    18: (  # Ar
        (0.05011, 0.16667, 0.17074, 0.04358),
        (0.07991, 1.01436, 15.67109, 39.60819),
    ),
    19: (  # K
        (0.04058, 0.17582, 0.20943, 0.02922),
        (0.03352, 0.82984, 14.13679, 200.97722),
    ),
    20: (  # Ca
        (0.04001, 0.17416, 0.20986, 0.05497),
        (0.02289, 0.71288, 11.18914, 135.02390),
    ),
    21: (  # Sc
        (0.09685, 0.14777, 0.20981, 0.04852),
        (0.12527, 1.34248, 12.43524, 131.71112),
    ),
    22: (  # Ti
        (0.06667, 0.17356, 0.22710, 0.05957),
        (0.05198, 0.86467, 10.59984, 103.56776),
    ),
    23: (  # V
        (0.05118, 0.16791, 0.26700, 0.06476),
        (0.03786, 0.57160, 8.30305, 91.78068),
    ),
    24: (  # Cr
        (0.03204, 0.18460, 0.30764, 0.05052),
        (0.00240, 0.44931, 7.92251, 86.64058),
    ),
    25: (  # Mn
        (0.03866, 0.17782, 0.31329, 0.06898),
        (0.01836, 0.41203, 6.73736, 76.30466),
    ),
    26: (  # Fe
        (0.05455, 0.16660, 0.33208, 0.06947),
        (0.03947, 0.43294, 6.26864, 71.29470),
    ),
    27: (  # Co
        (0.05942, 0.17472, 0.34423, 0.06828),
        (0.03962, 0.43253, 6.05175, 68.72437),
    ),
    28: (  # Ni
        (0.06049, 0.16600, 0.37302, 0.07109),
        (0.03558, 0.39976, 5.36660, 62.46894),
    ),
    29: (  # Cu
        (0.08034, 0.15838, 0.40116, 0.05467),
        (0.05475, 0.45736, 5.38252, 60.43276),
    ),
    30: (  # Zn
        (0.02948, 0.19200, 0.42222, 0.07480),
        (0.00137, 0.26535, 4.48040, 54.26088),
    ),
    31: (  # Ga
        (0.16157, 0.32976, 0.18964, 0.06148),
        (0.10455, 2.18391, 9.04125, 75.16958),
    ),
    32: (  # Ge
        (0.16184, 0.35705, 0.17618, 0.07133),
        (0.09890, 2.06856, 9.89926, 68.13783),
    ),
    33: (  # As
        (0.06190, 0.18452, 0.41600, 0.12793),
        (0.01642, 0.32542, 3.51888, 44.50604),
    ),
    34: (  # Se
        (0.15913, 0.41583, 0.13385, 0.10549),
        (0.07669, 1.89297, 11.31554, 46.32082),
    ),
    35: (  # Br
        (0.16514, 0.41202, 0.12900, 0.13209),
        (0.08199, 1.76568, 9.87254, 38.10640),
    ),
    36: (  # Kr
        (0.15798, 0.41181, 0.14254, 0.14987),
        (0.06939, 1.53446, 8.98025, 33.04365),
    ),
    37: (  # Rb
        (0.16535, 0.44674, 0.24245, 0.03161),
        (0.07044, 1.59236, 17.53592, 215.26198),
    ),
    38: (  # Sr
        (0.16039, 0.44470, 0.24661, 0.05840),
        (0.06199, 1.41265, 14.33812, 152.80257),
    ),
    39: (  # Y
        (0.16619, 0.44376, 0.25613, 0.06797),
        (0.06364, 1.34205, 13.66551, 125.72522),
    ),
    40: (  # Zr
        (0.16794, 0.44505, 0.27188, 0.07313),
        (0.06565, 1.25292, 13.09355, 109.50252),
    ),
    41: (  # Nb
        (0.16552, 0.45008, 0.30474, 0.06161),
        (0.05921, 1.15624, 13.24924, 98.69958),
    ),
    42: (  # Mo
        (0.17327, 0.44679, 0.32441, 0.06143),
        (0.06162, 1.11236, 12.76149, 90.92026),
    ),
    43: (  # Tc
        (0.16424, 0.45046, 0.33749, 0.07766),
        (0.05081, 0.99771, 11.28925, 84.28943),
    ),
    44: (  # Ru
        (0.18750, 0.44919, 0.36323, 0.05388),
        (0.05120, 1.08672, 12.23172, 85.27316),
    ),
    45: (  # Rh
        (0.16081, 0.45211, 0.40343, 0.06140),
        (0.04662, 0.85252, 10.51121, 74.53949),
    ),
    46: (  # Pd
        (0.16599, 0.43951, 0.41478, 0.08142),
        (0.04933, 0.79381, 9.30944, 41.17414),
    ),
    47: (  # Ag
        (0.16547, 0.44658, 0.45401, 0.05959),
        (0.04481, 0.75608, 9.34354, 67.91975),
    ),
    48: (  # Cd
        (0.17154, 0.43689, 0.46392, 0.07725),
        (0.04867, 0.71518, 8.40595, 64.24400),
    ),
    49: (  # In
        (0.15752, 0.44821, 0.48186, 0.08596),
        (0.03672, 0.64379, 7.83687, 73.37281),
    ),
    50: (  # Sn
        (0.15732, 0.44563, 0.48507, 0.10948),
        (0.03308, 0.60931, 7.04977, 64.83582),
    ),
    51: (  # Sb
        (0.16971, 0.42742, 0.48779, 0.13653),
        (0.04023, 0.58192, 6.29247, 55.57061),
    ),
    52: (  # Te
        (0.14927, 0.43729, 0.49444, 0.16440),
        (0.02842, 0.50687, 5.60835, 48.28004),
    ),
    53: (  # I
        (0.18053, 0.44724, 0.48163, 0.15995),
        (0.03830, 0.58340, 6.47550, 47.08820),
    ),
    54: (  # Xe
        (0.13141, 0.43855, 0.50035, 0.22299),
        (0.02097, 0.41007, 4.52105, 37.18178),
    ),
    55: (  # Cs
        (0.31397, 0.55648, 0.39828, 0.04852),
        (0.07813, 1.45053, 15.05933, 199.48830),
    ),
    56: (  # Ba
        (0.32756, 0.53927, 0.39830, 0.07607),
        (0.08444, 1.40227, 13.12939, 160.56676),
    ),
    57: (  # La
        (0.30887, 0.53804, 0.42265, 0.09559),
        (0.07206, 1.19585, 11.55866, 127.31371),
    ),
    58: (  # Ce
        (0.28398, 0.53568, 0.46662, 0.10282),
        (0.05717, 0.98756, 9.95556, 117.31874),
    ),
    59: (  # Pr
        (0.35160, 0.56889, 0.42010, 0.07246),
        (0.08249, 1.43427, 12.37363, 150.55968),
    ),
    60: (  # Nd
        (0.33810, 0.58035, 0.44442, 0.07413),
        (0.07081, 1.31033, 11.44403, 144.17706),
    ),
    61: (  # Pm
        (0.35449, 0.59626, 0.43868, 0.07152),
        (0.07442, 1.38680, 11.54391, 143.72185),
    ),
    62: (  # Sm
        (0.35559, 0.60598, 0.45165, 0.07168),
        (0.07155, 1.34703, 11.00432, 140.09138),
    ),
    63: (  # Eu
        (0.38379, 0.64088, 0.41710, 0.06708),
        (0.07794, 1.55042, 11.89283, 142.79585),
    ),
    64: (  # Gd
        (0.40352, 0.64303, 0.40488, 0.08137),
        (0.08508, 1.60712, 11.45367, 116.64063),
    ),
    65: (  # Tb
        (0.36838, 0.64761, 0.47222, 0.06854),
        (0.06520, 1.32571, 10.16884, 134.69034),
    ),
    66: (  # Dy
        (0.38514, 0.68422, 0.44359, 0.06775),
        (0.06850, 1.43566, 10.57719, 131.88972),
    ),
    67: (  # Ho
        (0.37280, 0.67528, 0.47337, 0.08320),
        (0.06264, 1.26756, 9.46411, 107.50194),
    ),
    68: (  # Er
        (0.39335, 0.70093, 0.46774, 0.06658),
        (0.06750, 1.35829, 9.76480, 127.40374),
    ),
    69: (  # Tm
        (0.40587, 0.71223, 0.46598, 0.06847),
        (0.06958, 1.38750, 9.41888, 122.10940),
    ),
    70: (  # Yb
        (0.39728, 0.73368, 0.47795, 0.06759),
        (0.06574, 1.31578, 9.13448, 120.98209),
    ),
    71: (  # Lu
        (0.40697, 0.73576, 0.47481, 0.08291),
        (0.06517, 1.29452, 8.67569, 100.34878),
    ),
    72: (  # Hf
        (0.40122, 0.78861, 0.44658, 0.08799),
        (0.06213, 1.30860, 9.18871, 91.20213),
    ),
    73: (  # Ta
        (0.41127, 0.76965, 0.46563, 0.10180),
        (0.06292, 1.23499, 8.42904, 77.59815),
    ),
    74: (  # W
        (0.39978, 0.77171, 0.48541, 0.11540),
        (0.05693, 1.15762, 7.83077, 67.14066),
    ),
    75: (  # Re
        (0.39130, 0.80752, 0.48702, 0.11041),
        (0.05145, 1.11240, 8.33441, 65.71782),
    ),
    76: (  # Os
        (0.40436, 0.80701, 0.48445, 0.12438),
        (0.05573, 1.11159, 8.00221, 57.35021),
    ),
    77: (  # Ir
        (0.38816, 0.80163, 0.51922, 0.13514),
        (0.04855, 0.99356, 7.38693, 51.75829),
    ),
    78: (  # Pt
        (0.39551, 0.80409, 0.53365, 0.13485),
        (0.04981, 0.97669, 7.38024, 44.52068),
    ),
    79: (  # Au
        (0.40850, 0.83052, 0.53325, 0.11978),
        (0.05151, 1.00803, 8.03707, 45.01758),
    ),
    80: (  # Hg
        (0.40092, 0.85415, 0.53346, 0.12747),
        (0.04693, 0.98398, 7.83562, 46.51474),
    ),
    81: (  # Tl
        (0.41872, 0.88168, 0.54551, 0.09404),
        (0.05161, 1.02127, 9.18455, 64.88177),
    ),
    82: (  # Pb
        (0.43358, 0.88007, 0.52966, 0.12059),
        (0.05154, 1.03252, 8.49678, 58.79463),
    ),
    83: (  # Bi
        (0.40858, 0.87837, 0.56392, 0.13698),
        (0.04200, 0.90939, 7.71158, 57.79178),
    ),
    84: (  # Po
        (0.41637, 0.85094, 0.57749, 0.16700),
        (0.04661, 0.87289, 6.84038, 51.36000),
    ),
    85: (  # At
        (0.38951, 0.83297, 0.60557, 0.20770),
        (0.04168, 0.73697, 5.86112, 43.78613),
    ),
    86: (  # Rn
        (0.41677, 0.88094, 0.55170, 0.21029),
        (0.04488, 0.83871, 6.44020, 43.51940),
    ),
    87: (  # Fr
        (0.50089, 1.00860, 0.51420, 0.05996),
        (0.05786, 1.20028, 13.85073, 172.15909),
    ),
    88: (  # Ra
        (0.47470, 0.99363, 0.54721, 0.09206),
        (0.05239, 1.03225, 11.49796, 143.12303),
    ),
    89: (  # Ac
        (0.47810, 0.98385, 0.54905, 0.12055),
        (0.05167, 0.98867, 10.52682, 112.18267),
    ),
    90: (  # Th
        (0.47903, 0.97455, 0.55883, 0.14309),
        (0.04931, 0.95698, 9.61135, 95.44649),
    ),
    91: (  # Pa
        (0.48351, 0.98292, 0.58877, 0.12425),
        (0.04748, 0.93369, 9.89867, 102.06961),
    ),
    92: (  # U
        (0.48664, 0.98057, 0.61483, 0.12136),
        (0.04660, 0.89912, 9.69785, 100.23434),
    ),
    93: (  # Np
        (0.46078, 0.97139, 0.66506, 0.13012),
        (0.04323, 0.78798, 8.71624, 92.30811),
    ),
    94: (  # Pu
        (0.49148, 0.98583, 0.67674, 0.09725),
        (0.04641, 0.85867, 9.51157, 111.02754),
    ),
    95: (  # Am
        (0.50865, 0.98574, 0.68109, 0.09977),
        (0.04918, 0.87026, 9.41105, 104.98576),
    ),
    96: (  # Cm
        (0.46259, 0.97882, 0.73056, 0.12723),
        (0.03904, 0.72797, 8.00506, 86.41747),
    ),
    97: (  # Bk
        (0.46221, 0.95749, 0.76259, 0.14086),
        (0.03969, 0.68167, 7.29607, 75.72682),
    ),
    98: (  # Cf
        (0.48500, 0.95602, 0.77234, 0.13374),
        (0.04291, 0.69956, 7.38554, 77.18528),
    ),
}

# --- EMsoft unit-conversion constants (diffraction.f90::CalcUcg) ---------------
_TWOPI = 2.0 * math.pi
SWK = 0.1 * _TWOPI            # g[nm⁻¹] -> FSCATT argument G (WK angular units)
DWWK = 100.0 / (8.0 * math.pi**2)  # B[nm²] -> UL[Å]:  UL = sqrt(B·DWWK)
FOURPI = 4.0 * math.pi
# Verified S-scaling: WEKO argument S = S_SCALE · s, with s = |g| (full reciprocal
# vector length) in nm⁻¹.  EMsoft S_weko = rlp%g·swk/4π = |g|·0.05; S_SCALE = 0.05
# folds BOTH the nm⁻¹→Å⁻¹ (×0.1) AND |g|→sin(θ)/λ (×0.5) factors, so S = 0.05·|g|
# IS the physical sin(θ)/λ in Å⁻¹ (AMP_2 fix — see module docstring).
S_SCALE = SWK / FOURPI        # == 0.05


def _weko_real(A: torch.Tensor, B: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """WEKO elastic sum ``Σ A_i·(1-exp(-B_i·S²))/S²`` (Å), torch-native.

    Mirrors EMsoft ``others.f90::WEKO`` including its numerically-conditioned
    branches:

    * ``B_i·S² < 0.1``: Taylor ``A_i·B_i·(1 - 0.5·B_i·S²)`` (avoids 0/0 at S→0
      and catastrophic cancellation of ``(1-exp(-x))/x`` for small x).  This also
      yields the analytic S→0 limit ``Σ A_i·B_i``.
    * ``B_i·S² > 20``: ``A_i/S²`` (the exponential term is negligible).
    * otherwise: ``A_i·(1-exp(-B_i·S²))/S²``.

    Args:
        A, B: shape ``(4,)`` coefficient tensors (same dtype/device as ``S``).
        S: WEKO scattering variable, any shape.

    Returns:
        Real tensor, same shape as ``S``.
    """
    S = S.to(dtype=A.dtype)
    S2 = S * S
    # Guard 1/S² at S=0; the S→0 region is always served by the Taylor branch.
    inv_S2 = torch.where(S2 > 0, 1.0 / torch.clamp(S2, min=torch.finfo(A.dtype).tiny),
                         torch.zeros_like(S2))

    out = torch.zeros_like(S)
    for i in range(4):
        argu = B[i] * S2  # = B_i · S²
        taylor = A[i] * B[i] * (1.0 - 0.5 * argu)
        large = A[i] * inv_S2
        mid = A[i] * (1.0 - torch.exp(-argu)) * inv_S2
        term = torch.where(
            argu < 0.1,
            taylor,
            torch.where(argu > 20.0, large, mid),
        )
        out = out + term
    return out


def _fphon(G: torch.Tensor, UL: float, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """WK phonon (thermal-diffuse) absorptive form factor ``f_abs`` (EMsoft FPHON).

    Faithful port of EMsoft ``others.f90::FPHON`` (+ helpers ``RI1/RI2/RIH1/RIH2``).
    The coefficients are rescaled exactly as EMsoft does inside FPHON:
    ``A1 = A·(4π)²``, ``B1 = B/(4π)²``.  The double sum over the 4×4 coefficient
    pairs accumulates the analytic Einstein-phonon convolution integrals.

    .. note::

       **Non-autograd / CPU bridge — flagged as a diagnose-loop knob.**  The WK
       phonon integrals require the exponential integral ``Ei(x)`` and a
       table-interpolated helper (``RIH2``).  PyTorch has no ``Ei``, so this
       function evaluates the imaginary part via ``scipy.special.expi`` on a
       NumPy bridge and returns a detached tensor.  Consequences:

       * It is **not differentiable** w.r.t. ``G`` (the elastic real part *is*).
       * It runs on CPU regardless of the input device (then moved back).

       This is acceptable for SP0+SP1: the absorptive part is a contrast/sharpness
       knob, not band geometry, and the real (elastic) part is the NCC driver.
       The physics is honest (full FPHON), not a faked constant.  If absorption
       turns out to matter for the NCC gate, this is the obvious place to (a) push
       the integrals onto the GPU or (b) swap in the simpler Hashimoto-Howie-Whelan
       absorptive model — both are diagnose-loop replacements, tracked in lessons.md.
    """
    import numpy as np
    from scipy.special import expi as _expi  # exponential integral Ei(x)

    g_arr = G.detach().to("cpu", dtype=torch.float64).numpy()
    Anp = A.detach().to("cpu", dtype=torch.float64).numpy()
    Bnp = B.detach().to("cpu", dtype=torch.float64).numpy()

    fp2 = float(FOURPI) * float(FOURPI)
    A1 = Anp * fp2
    B1 = Bnp / fp2
    U = float(UL)

    # WK ``RIH2`` lookup table (others.f90 DATA F/0:20/).
    _F = np.array(
        [1.000000, 1.005051, 1.010206, 1.015472, 1.020852, 1.026355, 1.031985,
         1.037751, 1.043662, 1.049726, 1.055956, 1.062364, 1.068965, 1.075780,
         1.082830, 1.090140, 1.097737, 1.105647, 1.113894, 1.122497, 1.131470],
        dtype=np.float64,
    )
    C = 0.5772157  # Euler-Mascheroni

    # ---- scalar helpers, one g at a time (mirror the Fortran exactly) --------
    def _rih2(x: float) -> float:
        x1 = 1.0 / x
        i = int(200.0 * x1)
        if i > 19:
            i = 19
        if i < 0:
            i = 0
        return _F[i] + 200.0 * (_F[i + 1] - _F[i]) * (x1 - 0.5e-3 * i)

    def _rih1(x1: float, x2: float, x3: float) -> float:
        if x2 <= 20.0 and x3 <= 20.0:
            return math.exp(-x1) * (_expi(x2) - _expi(x3))
        if x2 > 20.0:
            r = math.exp(x2 - x1) * _rih2(x2) / x2
        else:
            r = math.exp(-x1) * _expi(x2)
        if x3 > 20.0:
            r = r - math.exp(x3 - x1) * _rih2(x3) / x3
        else:
            r = r - math.exp(-x1) * _expi(x3)
        return r

    def _ri1(bi: float, bj: float, g_: float) -> float:
        g2 = g_ * g_
        eps = max(bi, bj) * g2
        if eps <= 0.1:
            ri1 = bi * math.log((bi + bj) / bi) + bj * math.log((bi + bj) / bj)
            ri1 = ri1 * math.pi
            if g_ == 0.0:
                return ri1
            bi2 = bi * bi
            bj2 = bj * bj
            temp = 0.5 * bi2 * math.log(bi / (bi + bj)) + 0.5 * bj2 * math.log(bj / (bi + bj))
            temp += 0.75 * (bi2 + bj2) - 0.25 * (bi + bj) * (bi + bj)
            temp -= 0.5 * (bi - bj) * (bi - bj)
            return ri1 + math.pi * g2 * temp
        big2 = bi * g2
        bjg2 = bj * g2
        ri1 = 2.0 * C + math.log(big2) + math.log(bjg2) - 2.0 * _expi(-bi * bj * g2 / (bi + bj))
        ri1 += _rih1(big2, big2 * bi / (bi + bj), big2)
        ri1 += _rih1(bjg2, bjg2 * bj / (bi + bj), bjg2)
        return ri1 * math.pi / g2

    def _ri2(bi: float, bj: float, g_: float, u: float) -> float:
        u2 = u * u
        u22 = 0.5 * u2
        g2 = g_ * g_
        biuh = bi + 0.5 * u2
        bjuh = bj + 0.5 * u2
        biu = bi + u2
        bju = bj + u2
        eps = max(bi, bj, u2) * g2
        if eps <= 0.1:
            ri2 = (bi + u2) * math.log((bi + bj + u2) / (bi + u2))
            ri2 += bj * math.log((bi + bj + u2) / (bj + u2))
            if u2 > 0.0:
                ri2 += u2 * math.log(u2 / (bj + u2))
            ri2 *= math.pi
            if g_ == 0.0:
                return ri2
            temp = 0.0
            if u2 > 0.0:
                temp += 0.5 * u22 * u22 * math.log(biu * bju / (u2 * u2))
            temp += 0.5 * biuh * biuh * math.log(biu / (biuh + bjuh))
            temp += 0.5 * bjuh * bjuh * math.log(bju / (biuh + bjuh))
            temp += 0.25 * biu * biu + 0.5 * bi * bi
            temp += 0.25 * bju * bju + 0.5 * bj * bj
            temp -= 0.25 * (biuh + bjuh) * (biuh + bjuh)
            temp -= 0.5 * ((bi * biu - bj * bju) / (biuh + bjuh)) ** 2
            temp -= u22 * u22
            return ri2 + math.pi * g2 * temp
        ri2 = _expi(-0.5 * u2 * g2 * biuh / biu) + _expi(-0.5 * u2 * g2 * bjuh / bju)
        ri2 -= _expi(-biuh * bjuh * g2 / (biuh + bjuh)) + _expi(-0.25 * u2 * g2)
        ri2 *= 2.0
        ri2 += _rih1(0.5 * u2 * g2, 0.25 * u2 * g2, 0.25 * u2 * u2 * g2 / biu)
        ri2 += _rih1(0.5 * u2 * g2, 0.25 * u2 * g2, 0.25 * u2 * u2 * g2 / bju)
        ri2 += _rih1(biuh * g2, biuh * biuh * g2 / (biuh + bjuh), biuh * biuh * g2 / biu)
        ri2 += _rih1(bjuh * g2, bjuh * bjuh * g2 / (biuh + bjuh), bjuh * bjuh * g2 / bju)
        return ri2 * math.pi / g2

    def _fphon_scalar(g_: float) -> float:
        # u2==0 (no Debye-Waller) makes the RI2 g==0/log(u2) terms singular and is
        # unphysical for a thermal-diffuse (phonon) integral; clamp to a tiny UL.
        u = U if U > 0.0 else math.sqrt(1e-6 * DWWK)
        dewa = math.exp(-0.5 * u * u * g_ * g_)
        val = 0.0
        for j in range(4):
            val += A1[j] * A1[j] * (dewa * _ri1(B1[j], B1[j], g_) - _ri2(B1[j], B1[j], g_, u))
            for i in range(j):
                val += 2.0 * A1[j] * A1[i] * (
                    dewa * _ri1(B1[i], B1[j], g_) - _ri2(B1[i], B1[j], g_, u)
                )
        return val

    flat = g_arr.reshape(-1)
    out = np.array([_fphon_scalar(float(gv)) for gv in flat], dtype=np.float64)
    out = np.maximum(out, 0.0)  # absorption convention: f_abs >= 0
    out = out.reshape(g_arr.shape)
    return torch.from_numpy(np.ascontiguousarray(out)).to(G.device)


# --- EMsoft FCORE constants (others.f90, single-precision parameters) ----------
_FCORE_A0 = 0.5289      # Bohr radius (Å); EMsoft DATA A0=.5289
_FCORE_PI = 3.1415927   # EMsoft DATA PI=3.1415927 (single-precision π)


def _fcore(G: torch.Tensor, Z: int, voltage_kV: float) -> torch.Tensor:
    """WK core-loss absorptive form factor ``FCORE`` (EMsoft ``others.f90::FCORE``).

    Faithful port of the EMsoft ``FCORE`` function (modified 1992-01-13 by
    A. Weickenmeier), which implements H. Rose, *Optik* **45**, 139–158 (1976):
    the core-electron contribution to the absorptive potential via a Yukawa-screened
    atomic core and a characteristic energy loss ``ΔE ≈ 6Z eV``.

    Formula (verbatim from the ``develop`` source, verified 2026-06-10)::

        K0     = 0.5068·sqrt(1022·V + V²)            [Å⁻¹, angular = 2π/λ_rel]
        ΔE     = 6·Z·1e-3                            [keV]
        θ_E    = ΔE/(2V)·(2V + 1022)/(V + 1022)
        R      = 0.885·A0 / Z^(1/3)                  [Å],  A0 = 0.5289 Å
        θ_A    = 1/(K0·R)
        θ_B    = G/(2·K0)
        Ω      = 2·θ_B/θ_A   (= G·R)
        κ      = θ_E/θ_A
        X1     = Ω/[(1+Ω²)·√(Ω²+4κ²)]·ln[(Ω+√(Ω²+4κ²))/(2κ)]
        X2     = 1/√[(1+Ω²)²+4κ²Ω²]·ln[(1+2κ²+Ω²+√((1+Ω²)²+4κ²Ω²))/(2κ√(1+κ²))]
        X3     = 1/[Ω·√(Ω²+4(1+κ²))]·ln[(Ω+√(Ω²+4(1+κ²)))/(2√(1+κ²))]   (Ω > 0.01)
               = 1/[4(1+κ²)]                                              (Ω ≤ 0.01)
        HI     = (2Z/θ_A²)·(−X1 + X2 − X3)
        FCORE  = (4/A0²)·(2π/K0²)·HI                 [Å]

    The input ``G`` is the FSCATT scattering-vector magnitude in Å⁻¹ (the same
    ``G = swk·s`` argument fed to :func:`_fphon`).  Like FPHON, FCORE's return
    value carries **no net 4π** (it has a single ``2π``, which is intrinsic to the
    Rose azimuth-averaged solid-angle integral and is accounted for by HI), so it
    is fed through the identical ``γ²/K0`` and ``/(4π)`` scaling as FPHON — see the
    ``wk_scattering_factor`` absorptive block and ``lessons.md`` §2/§12.

    .. note::

       Evaluated on a NumPy/CPU bridge (the EMsoft single-precision logarithms are
       scalar) and returned as a detached tensor — same non-autograd/CPU contract
       as :func:`_fphon`.  The absorptive part is a contrast knob, not a band-
       geometry driver.

    Args:
        G: scattering-vector magnitude ``|g|`` in Å⁻¹ (FSCATT convention), any shape.
        Z: atomic number.
        voltage_kV: accelerating voltage in kV.

    Returns:
        Real tensor of FCORE values (Å), same shape as ``G``, clamped ``>= 0``.
    """
    import numpy as np

    g_arr = G.detach().to("cpu", dtype=torch.float64).numpy()
    V = float(voltage_kV)
    Zf = float(Z)

    k0 = 0.5068 * math.sqrt(1022.0 * V + V * V)        # Å⁻¹ (angular)
    de = 6.0 * Zf * 1.0e-3                              # keV
    thetae = de / (2.0 * V) * (2.0 * V + 1022.0) / (V + 1022.0)
    r = 0.885 * _FCORE_A0 / (Zf ** (1.0 / 3.0))         # Å
    ta = 1.0 / (k0 * r)
    kappa = thetae / ta
    k2 = kappa * kappa

    def _fcore_scalar(g_: float) -> float:
        tb = g_ / (2.0 * k0)
        omega = 2.0 * tb / ta
        o2 = omega * omega
        # X1: vanishes as Ω→0 (the Ω prefactor), so the transmitted beam g=0 is safe.
        if omega > 0.0:
            x1 = (
                omega / ((1.0 + o2) * math.sqrt(o2 + 4.0 * k2))
                * math.log((omega + math.sqrt(o2 + 4.0 * k2)) / (2.0 * kappa))
            )
        else:
            x1 = 0.0
        x2 = (
            1.0 / math.sqrt((1.0 + o2) * (1.0 + o2) + 4.0 * k2 * o2)
            * math.log(
                (1.0 + 2.0 * k2 + o2
                 + math.sqrt((1.0 + o2) * (1.0 + o2) + 4.0 * k2 * o2))
                / (2.0 * kappa * math.sqrt(1.0 + k2))
            )
        )
        if omega > 1.0e-2:
            x3 = (
                1.0 / (omega * math.sqrt(o2 + 4.0 * (1.0 + k2)))
                * math.log(
                    (omega + math.sqrt(o2 + 4.0 * (1.0 + k2)))
                    / (2.0 * math.sqrt(1.0 + k2))
                )
            )
        else:
            x3 = 1.0 / (4.0 * (1.0 + k2))
        hi = (2.0 * Zf) / (ta * ta) * (-x1 + x2 - x3)
        return 4.0 / (_FCORE_A0 * _FCORE_A0) * 2.0 * _FCORE_PI / (k0 * k0) * hi

    flat = g_arr.reshape(-1)
    out = np.array([_fcore_scalar(float(gv)) for gv in flat], dtype=np.float64)
    out = np.maximum(out, 0.0)  # absorption convention: f_abs >= 0
    out = out.reshape(g_arr.shape)
    return torch.from_numpy(np.ascontiguousarray(out)).to(G.device)


def wk_scattering_factor(
    Z: int,
    s: torch.Tensor,
    B: float,
    voltage_kV: float,
    *,
    relativistic: bool = True,
    absorptive: bool = True,
    absflg: int = 1,
) -> torch.Tensor:
    """Weickenmeier-Kohl complex electron scattering factor (Å).

    Args:
        Z: atomic number (must be in :data:`WK_COEFFS`).
        s: scattering variable ``|g| = 1/d`` (the full reciprocal-vector length,
            EMsoft ``rlp%g``) in **nm⁻¹**, any shape.  The internal WEKO variable
            ``S = 0.05·s = 0.05·|g|`` is the physical ``sin(θ)/λ`` in Å⁻¹ — pass the
            full ``|g|``, NOT the half-angle ``|g|/2`` (see the AMP_2 module-docstring
            note; the 0.05 already folds in the ×0.5 half-angle factor).
        B: Debye-Waller B-factor in **nm²** (EMsoft AtomData convention).  Use
            ``B=0`` to disable Debye-Waller damping of the *elastic* part (e.g. for
            cross-checking against published room-temperature-independent tables).
            The phonon (absorptive) integral is intrinsically thermal, so when
            ``B=0`` and ``absorptive=True`` a tiny non-zero ``UL`` is used for the
            FPHON term only (the elastic part stays undamped).
        voltage_kV: accelerating voltage in kV (used for the relativistic γ and the
            core-loss absorptive term).
        relativistic: if True (default), multiply the elastic part by
            ``γ = (V_kV + 511)/511`` (EMsoft ``ACCFLG = .TRUE.``).
        absorptive: if True (default), add the WK imaginary part (see ``absflg``).
            Set False to get the pure elastic factor.
        absflg: EMsoft absorption flag selecting the imaginary channel
            (EMsoft ``FSCATT``):

            * ``1`` (**default**) — phonon (thermal-diffuse) only, :func:`_fphon`.
              Byte-identical to the historical behaviour.
            * ``3`` — phonon **+** core-loss: ``FCORE(G,Z,V)·DEWA + FPHON(...)``,
              the full absorptive potential (EMsoft EBSD production setting).
              ``DEWA = exp(-0.5·UL²·G²)`` multiplies FCORE (core absorption is
              localised to bound electrons that move with the nucleus); FPHON is
              already DW-corrected internally.  The combined sum passes through the
              identical ``γ²/K0`` and ``/(4π)`` scaling as the phonon-only case
              (both channels are 4π-free — see ``lessons.md`` §12).

            ``absflg`` only affects the imaginary part and only when
            ``absorptive=True``.  Other values raise ``ValueError``.

    Returns:
        Complex tensor (``complex128`` if ``s`` is float64, else ``complex64``),
        same shape as ``s``.  ``real`` = elastic ``f_el`` (Å, DW- and
        γ-corrected), ``imag`` = absorptive ``f_abs >= 0`` (Å).

    Raises:
        ValueError: if ``Z`` is not in the WK coefficient table, or ``absflg`` is
            not one of ``{1, 3}``.
    """
    if absflg not in (1, 3):
        raise ValueError(
            f"absflg must be 1 (phonon only) or 3 (phonon + core), got {absflg!r}"
        )
    if Z not in WK_COEFFS:
        raise ValueError(
            f"no Weickenmeier-Kohl coefficients for Z={Z}; "
            f"available: {sorted(WK_COEFFS)}"
        )

    if not torch.is_tensor(s):
        s = torch.as_tensor(s)
    real_dtype = s.dtype if s.dtype.is_floating_point else torch.float64
    s = s.to(dtype=real_dtype)

    A_list, B_list = WK_COEFFS[Z]
    A = torch.tensor(A_list, dtype=real_dtype, device=s.device)
    Bc = torch.tensor(B_list, dtype=real_dtype, device=s.device)

    # WEKO scattering variable S = 0.05·s = 0.05·|g|  (s = |g| in nm⁻¹) — the
    # physical sin(θ)/λ in Å⁻¹; see module docstring (AMP_2 fix).
    S = S_SCALE * s

    # Elastic part (Å).
    f_el = _weko_real(A, Bc, S)

    # Debye-Waller damping: exp(-0.5·UL²·G²); UL[Å]=sqrt(B[nm²]·dwwk); G=s·swk.
    if B and B > 0.0:
        UL = math.sqrt(B * DWWK)  # Å
        G = SWK * s               # FSCATT argument
        dewa = torch.exp(-0.5 * (UL * UL) * (G * G))
        f_el = f_el * dewa
    else:
        UL = 0.0

    # Relativistic γ = (V_kV + m_e c²)/m_e c², computed ONCE (used by both the
    # elastic γ-scaling and the absorptive γ²/K0 scaling below).
    gamma = (voltage_kV + 511.0) / 511.0

    # Relativistic correction.
    if relativistic:
        f_el = f_el * gamma

    # Absorptive (imaginary) part — WK phonon term (CPU/non-autograd, flagged).
    if absorptive:
        G = SWK * s
        f_abs = _fphon(G, UL, A, Bc)
        if absflg == 3:
            # absflg=3 → FIMA = FCORE(G,Z,V)·DEWA + FPHON(...)  (EMsoft FSCATT).
            # FCORE carries the Debye-Waller damping (core absorption is localised
            # to bound electrons that move with the nucleus); FPHON is already
            # DW-corrected internally.  DEWA uses the SAME UL = sqrt(B·dwwk) as the
            # elastic part (UL=0 → DEWA=1 when B=0, matching EMsoft).  This sum
            # happens BEFORE the single γ²/K0 (and /4π) scaling below — exactly as
            # EMsoft assembles FIMA — so the FCORE channel inherits the identical
            # scaling as FPHON (both are 4π-free; see lessons.md §12).
            G_arr = SWK * s
            dewa_abs = torch.exp(
                -0.5 * (UL * UL) * (G_arr * G_arr)
            ).to(dtype=f_abs.dtype)
            f_core = _fcore(G_arr, Z, voltage_kV).to(dtype=f_abs.dtype) * dewa_abs
            f_abs = f_abs + f_core
        if relativistic:
            # EMsoft FSCATT scales FIMA by γ²/K0 (no 4π — unlike FREAL which
            # carries an explicit 4π).  K0 = 0.5068·sqrt(1022·V + V²) is the
            # relativistic |k| = 2π/λ in Å⁻¹ (FSCATT works in Å; the nm→Å factor
            # of ten is baked into the 0.5068 prefactor — see lessons.md §2).
            k0 = 0.5068 * math.sqrt(1022.0 * voltage_kV + voltage_kV * voltage_kV)
            # The additional /FOURPI mirrors EMsoft's single `pref = .../(4π)`:
            # that /(4π) is matched to the 4π *inside* FREAL, but it also divides
            # FIMA (which has NO 4π).  Our downstream PREF_EFF = pref·4π
            # compensates the 4π-free f_el correctly, but would over-scale the
            # 4π-free f_abs by exactly 4π ≈ 12.57× — so we divide it out here.
            f_abs = f_abs * (gamma * gamma / k0) / FOURPI
        f_abs = f_abs.to(dtype=real_dtype)
    else:
        f_abs = torch.zeros_like(f_el)

    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    return torch.complex(f_el.to(real_dtype), f_abs.to(real_dtype)).to(complex_dtype)
