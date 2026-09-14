r"""
EDS Utilities - Extract and convert EDS data from H5OINA files

Provides:
- Raw counts extraction from H5OINA Window Integral datasets
- Standardless Cliff-Lorimer quantification (counts -> Wt.%)
- Wt.% -> At.% conversion using atomic masses
- Phase suggestion based on measured At.% composition

Quantification convention (nail this down before touching anything below)
------------------------------------------------------------------------
Cliff-Lorimer is

    C_A / C_B = k_AB * (I_A / I_B)        with  k_AB = k_A / k_B

so a weight fraction is **proportional to k times intensity**::

    C_i = k_i * I_i / sum_j (k_j * I_j) * 100

Until 2026-09-12 this module divided by k instead of multiplying, while the
table it divided by was labelled "typical Oxford/Bruker defaults" -- i.e. the
values were inverted on top of being flat. See
``tasks/eds-quantification-wrong-2026-09-10.md``.

What the k-factors are now
--------------------------
``k_factor()`` builds the factor per **(element, X-ray line)** from physics and
then applies one measured instrument-response curve:

    k  =  A / (omega * p * n * b * ln(U) / E_c)   *   R(E_line)
          \___________ emission ______________/       \_ response _/

* ``A``       atomic mass -- intensity is per unit *mass*, so atoms/gram enters
* ``omega``   fluorescence yield of the ionised shell (Krause 1979)
* ``p``       fraction of that shell's emission inside the measured window
* ``n, b``    electrons in the shell and the Bethe constant (Green & Cosslett)
* ``U``       overvoltage ``E_0 / E_c``; ``E_c`` the absorption-edge energy
* ``R``       measured response, see below

The ``n*b*ln(U)/E_c`` term is the Bethe ionisation cross-section integrated
along the electron trajectory at constant stopping power: with
``Q ~ n*b*ln(U) / (U * E_c^2)`` and a path ``~ E_0 / S = U * E_c / S`` the
overvoltage cancels once and ``ln(U)/E_c`` remains. This is the term the old
flat table lacked entirely, and it is the dominant part of the error: at 20 kV
Fe K (edge 7.11 keV) sits at U = 2.8 while Si K sits at U = 10.9, so a given
weight of Fe produces far fewer counts. With this term alone, and *no* fitting
whatsoever, the Si/Mn/Fe ratios of the Aztec Tru-Q reference are reproduced to
within 6 % (Al to 22 %) -- measured 2026-09-12, see the report next to the
finding.

``R(E_line)`` is everything the emission model does not describe: detector
window transmission and Si dead-layer losses, absorption of the emitted photon
on its way out of the sample, and the residual error of the tabulated yields.
It is a *measured* curve, interpolated log-log in line energy between the
anchors in ``_CALIBRATION_STANDARD`` and clamped outside them. Its shape --
0.21 at 0.93 keV rising to a plateau of ~1 above 1.7 keV -- is the shape of a
SATW window transmission curve, which is what it mostly is.

Limits, stated plainly because the numbers are displayed and exported
--------------------------------------------------------------------
* **There is no ZAF / phi(rho z) matrix correction.** ``R`` was measured in an
  Al-rich matrix at 20 kV and silently carries that matrix's absorption. On a
  very different matrix (an oxide, a steel) the absorption part of ``R`` is
  wrong and the numbers drift. Every return value carries this in
  ``.provenance``; do not present an At.% from here as a standards-quality
  analysis.
* **Window integrals, not spectra.** We quantify the vendor's pre-integrated
  windows, so overlapping lines cannot be separated. The known case in this
  data set is the Zn L window sitting under Cu L emission; it is reported in
  ``.provenance["spectral_overlaps"]`` rather than absorbed into a k-factor,
  because a k that swallowed it would read ~4x too low on a Cu-free sample.
  Fixing it needs ``1/EDS/Data/Spectrum``.

Sources
-------
* Absorption edges and emission-line energies: Bearden & Burr (1967),
  Rev. Mod. Phys. 39, 125; Thompson et al., X-Ray Data Booklet, LBNL (2009).
* Fluorescence yields: Krause (1979), J. Phys. Chem. Ref. Data 8, 307.
* K-beta/K-alpha intensity ratios: Salem, Panossian & Krause (1974),
  At. Data Nucl. Data Tables 14, 91.
* Ionisation cross-section: Bethe (1930), parameterised by Green & Cosslett
  (1961), Proc. Phys. Soc. 78, 1206; as presented in Goldstein et al.,
  "Scanning Electron Microscopy and X-Ray Microanalysis".
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Atomic masses (g/mol), IUPAC 2021 abridged standard atomic weights; for the
# elements with no stable nuclide the mass number of the longest-lived isotope.
# Complete for Z = 1..92: a missing element here raises
# :class:`UnknownLineError` out of the quantification, and one exotic window in
# a file should not be able to take a whole EDS page down (2026-09-12).
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03,
}

# ---------------------------------------------------------------------------
# X-ray line physics
# ---------------------------------------------------------------------------
# Per (line family, element):
#     edge_keV       absorption edge of the ionised shell (K, L3 or M5)
#     line_keV       energy of the measured emission line (Kalpha1 / Lalpha1 /
#                    Malpha1) -- this is what the detector and the sample
#                    absorption see
#     omega          fluorescence yield of that shell (Krause 1979)
#     p              fraction of the shell's radiative emission that falls in
#                    the measured window
#
# For K lines, p = 1 / (1 + I(Kbeta)/I(Kalpha)); below Z = 11 there is no
# Kbeta to speak of, so p = 1.
#
# For L lines, p is held at the nominal 0.80 and omega is the L3 yield. This
# under-describes reality: L3 is additionally fed by Coster-Kronig transitions
# from L1/L2, and for the 3d metals "Lalpha" is a valence (3d -> 2p)
# transition whose tabulated strength is poor. That is a known weakness and it
# is exactly what the measured response curve absorbs -- which is why the L
# anchor sits at 0.21 rather than near 1. L3 yields below Z ~ 30 carry >30 %
# uncertainty in the literature.
#
# M lines (heavy elements at SEM voltages) are the weakest entry here: the M5
# yields are order-of-magnitude values and p is a nominal 0.70. There is no
# calibration anchor anywhere near them, so a k-factor for an M line is a
# model value only and is flagged as extrapolated in the provenance.
# For Po, At, Rn, Fr, Ra, Ac and Pa the M5 edge and the M-alpha energy are
# interpolated in Z between Bi/Th/U as well -- not just the yield. Nobody does
# SEM-EDS on those, and inventing four digits would have read as transcribed
# data; the rows are marked so the next reader knows which they are.
_P_L_ALPHA = 0.80
_P_M_ALPHA = 0.70

#: family -> symbol -> (edge_keV, line_keV, omega, p)
_XRAY_LINES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "K": {
        "Be": (0.1115, 0.1085, 0.00045, 1.0),
        "B":  (0.1880, 0.1833, 0.00101, 1.0),
        "C":  (0.2838, 0.2770, 0.00255, 1.0),
        "N":  (0.4016, 0.3924, 0.00520, 1.0),
        "O":  (0.5320, 0.5249, 0.00830, 1.0),
        "F":  (0.6854, 0.6768, 0.01300, 1.0),
        "Ne": (0.8701, 0.8486, 0.01800, 1.0),
        "Na": (1.0721, 1.0410, 0.02300, 1.0 / 1.010),
        "Mg": (1.3050, 1.2536, 0.03000, 1.0 / 1.013),
        "Al": (1.5596, 1.4867, 0.03570, 1.0 / 1.0126),
        "Si": (1.8389, 1.7400, 0.05000, 1.0 / 1.0272),
        "P":  (2.1455, 2.0137, 0.06300, 1.0 / 1.0510),
        "S":  (2.4720, 2.3078, 0.07800, 1.0 / 1.0640),
        "Cl": (2.8224, 2.6224, 0.09700, 1.0 / 1.0790),
        "Ar": (3.2029, 2.9577, 0.11800, 1.0 / 1.0930),
        "K":  (3.6074, 3.3138, 0.14000, 1.0 / 1.1030),
        "Ca": (4.0381, 3.6917, 0.16300, 1.0 / 1.1090),
        "Sc": (4.4928, 4.0906, 0.18800, 1.0 / 1.1150),
        "Ti": (4.9664, 4.5108, 0.21400, 1.0 / 1.1190),
        "V":  (5.4651, 4.9522, 0.24300, 1.0 / 1.1210),
        "Cr": (5.9892, 5.4147, 0.27500, 1.0 / 1.1230),
        "Mn": (6.5390, 5.8988, 0.30800, 1.0 / 1.1250),
        "Fe": (7.1120, 6.4038, 0.34000, 1.0 / 1.1304),
        "Co": (7.7089, 6.9303, 0.37300, 1.0 / 1.1320),
        "Ni": (8.3328, 7.4781, 0.40600, 1.0 / 1.1340),
        "Cu": (8.9789, 8.0478, 0.44000, 1.0 / 1.1350),
        "Zn": (9.6586, 8.6389, 0.47400, 1.0 / 1.1370),
        "Ga": (10.3671, 9.2517, 0.50700, 1.0 / 1.1400),
        "Ge": (11.1031, 9.8864, 0.53500, 1.0 / 1.1420),
        "As": (11.8667, 10.5437, 0.56200, 1.0 / 1.1450),
        "Se": (12.6578, 11.2224, 0.58900, 1.0 / 1.1470),
        "Br": (13.4737, 11.9242, 0.61800, 1.0 / 1.1500),
        "Kr": (14.3256, 12.6490, 0.64300, 1.0 / 1.1520),
        "Rb": (15.1997, 13.3953, 0.66700, 1.0 / 1.1550),
        "Sr": (16.1046, 14.1650, 0.69000, 1.0 / 1.1580),
        "Y":  (17.0384, 14.9584, 0.71000, 1.0 / 1.1600),
        "Zr": (17.9976, 15.7751, 0.73000, 1.0 / 1.1630),
        "Nb": (18.9856, 16.6151, 0.74700, 1.0 / 1.1650),
        "Mo": (19.9995, 17.4793, 0.76500, 1.0 / 1.1680),
    },
    "L": {
        "Ti": (0.4555, 0.4522, 0.0021, _P_L_ALPHA),
        "V":  (0.5129, 0.5113, 0.0026, _P_L_ALPHA),  # omega interpolated
        "Cr": (0.5745, 0.5728, 0.0030, _P_L_ALPHA),
        "Mn": (0.6403, 0.6374, 0.0036, _P_L_ALPHA),  # omega interpolated
        "Fe": (0.7080, 0.7050, 0.0041, _P_L_ALPHA),
        "Co": (0.7786, 0.7762, 0.0050, _P_L_ALPHA),  # omega interpolated
        "Ni": (0.8547, 0.8515, 0.0058, _P_L_ALPHA),  # omega interpolated
        "Cu": (0.9327, 0.9297, 0.0067, _P_L_ALPHA),
        "Zn": (1.0197, 1.0117, 0.0078, _P_L_ALPHA),
        "Ga": (1.1154, 1.0980, 0.0094, _P_L_ALPHA),  # omega interpolated
        "Ge": (1.2167, 1.1880, 0.0109, _P_L_ALPHA),  # omega interpolated
        "As": (1.3230, 1.2820, 0.0125, _P_L_ALPHA),  # omega interpolated
        "Se": (1.4358, 1.3791, 0.0140, _P_L_ALPHA),  # omega interpolated
        "Br": (1.5499, 1.4804, 0.0156, _P_L_ALPHA),
        "Kr": (1.6749, 1.5860, 0.0182, _P_L_ALPHA),  # omega interpolated
        "Rb": (1.8044, 1.6941, 0.0208, _P_L_ALPHA),  # omega interpolated
        "Sr": (1.9396, 1.8065, 0.0234, _P_L_ALPHA),  # omega interpolated
        "Y":  (2.0800, 1.9226, 0.0260, _P_L_ALPHA),  # omega interpolated
        "Zr": (2.2223, 2.0423, 0.0286, _P_L_ALPHA),  # omega interpolated
        "Nb": (2.3705, 2.1659, 0.0312, _P_L_ALPHA),  # omega interpolated
        "Mo": (2.5202, 2.2932, 0.0338, _P_L_ALPHA),
        "Tc": (2.6769, 2.4240, 0.0376, _P_L_ALPHA),  # omega interpolated
        "Ru": (2.8379, 2.5586, 0.0414, _P_L_ALPHA),  # omega interpolated
        "Rh": (3.0038, 2.6967, 0.0453, _P_L_ALPHA),  # omega interpolated
        "Pd": (3.1733, 2.8386, 0.0491, _P_L_ALPHA),  # omega interpolated
        "Ag": (3.3511, 2.9843, 0.0529, _P_L_ALPHA),
        "Cd": (3.5375, 3.1337, 0.0576, _P_L_ALPHA),  # omega interpolated
        "In": (3.7301, 3.2869, 0.0622, _P_L_ALPHA),  # omega interpolated
        "Sn": (3.9288, 3.4440, 0.0669, _P_L_ALPHA),
        "Sb": (4.1322, 3.6047, 0.0722, _P_L_ALPHA),  # omega interpolated
        "Te": (4.3414, 3.7693, 0.0774, _P_L_ALPHA),  # omega interpolated
        "I":  (4.5571, 3.9377, 0.0827, _P_L_ALPHA),
        "Xe": (4.7822, 4.1099, 0.0884, _P_L_ALPHA),  # omega interpolated
        "Cs": (5.0119, 4.2865, 0.0941, _P_L_ALPHA),  # omega interpolated
        "Ba": (5.2470, 4.4663, 0.0998, _P_L_ALPHA),
        "La": (5.4827, 4.6510, 0.1061, _P_L_ALPHA),  # omega interpolated
        "Ce": (5.7234, 4.8402, 0.1124, _P_L_ALPHA),  # omega interpolated
        "Pr": (5.9643, 5.0337, 0.1187, _P_L_ALPHA),  # omega interpolated
        "Nd": (6.2080, 5.2304, 0.1250, _P_L_ALPHA),
        "Pm": (6.4593, 5.4325, 0.1343, _P_L_ALPHA),  # omega interpolated
        "Sm": (6.7162, 5.6361, 0.1436, _P_L_ALPHA),  # omega interpolated
        "Eu": (6.9769, 5.8457, 0.1529, _P_L_ALPHA),  # omega interpolated
        "Gd": (7.2428, 6.0572, 0.1621, _P_L_ALPHA),  # omega interpolated
        "Tb": (7.5140, 6.2728, 0.1714, _P_L_ALPHA),  # omega interpolated
        "Dy": (7.7901, 6.4952, 0.1807, _P_L_ALPHA),  # omega interpolated
        "Ho": (8.0711, 6.7198, 0.1900, _P_L_ALPHA),  # omega interpolated
        "Er": (8.3579, 6.9487, 0.1993, _P_L_ALPHA),  # omega interpolated
        "Tm": (8.6480, 7.1799, 0.2086, _P_L_ALPHA),  # omega interpolated
        "Yb": (8.9436, 7.4156, 0.2179, _P_L_ALPHA),  # omega interpolated
        "Lu": (9.2441, 7.6555, 0.2271, _P_L_ALPHA),  # omega interpolated
        "Hf": (9.5607, 7.8990, 0.2364, _P_L_ALPHA),  # omega interpolated
        "Ta": (9.8811, 8.1461, 0.2457, _P_L_ALPHA),  # omega interpolated
        "W":  (10.2068, 8.3976, 0.2550, _P_L_ALPHA),
        "Re": (10.5353, 8.6525, 0.2652, _P_L_ALPHA),  # omega interpolated
        "Os": (10.8709, 8.9117, 0.2754, _P_L_ALPHA),  # omega interpolated
        "Ir": (11.2152, 9.1751, 0.2856, _P_L_ALPHA),  # omega interpolated
        "Pt": (11.5637, 9.4423, 0.2958, _P_L_ALPHA),  # omega interpolated
        "Au": (11.9187, 9.7133, 0.3060, _P_L_ALPHA),
        "Hg": (12.2839, 9.9888, 0.3163, _P_L_ALPHA),  # omega interpolated
        "Tl": (12.6580, 10.2685, 0.3267, _P_L_ALPHA),  # omega interpolated
        "Pb": (13.0352, 10.5515, 0.3370, _P_L_ALPHA),
        "Bi": (13.4185, 10.8388, 0.3468, _P_L_ALPHA),  # omega interpolated
        "Po": (13.8138, 11.1308, 0.3566, _P_L_ALPHA),  # omega interpolated
        "At": (14.2139, 11.4268, 0.3664, _P_L_ALPHA),  # omega interpolated
        "Rn": (14.6194, 11.7270, 0.3762, _P_L_ALPHA),  # omega interpolated
        "Fr": (15.0311, 12.0313, 0.3860, _P_L_ALPHA),  # omega interpolated
        "Ra": (15.4444, 12.3397, 0.3958, _P_L_ALPHA),  # omega interpolated
        "Ac": (15.8710, 12.6520, 0.4056, _P_L_ALPHA),  # omega interpolated
        "Th": (16.3003, 12.9687, 0.4154, _P_L_ALPHA),  # omega interpolated
        "Pa": (16.7330, 13.2907, 0.4252, _P_L_ALPHA),  # omega interpolated
        "U":  (17.1663, 13.6147, 0.4350, _P_L_ALPHA),
    },
    "M": {
        "Yb": (1.5278, 1.5214, 0.0125, _P_M_ALPHA),
        "Lu": (1.5885, 1.5813, 0.0139, _P_M_ALPHA),  # omega interpolated
        "Hf": (1.6617, 1.6446, 0.0152, _P_M_ALPHA),  # omega interpolated
        "Ta": (1.7351, 1.7096, 0.0166, _P_M_ALPHA),  # omega interpolated
        "W":  (1.8092, 1.7754, 0.0180, _P_M_ALPHA),
        "Re": (1.8829, 1.8420, 0.0192, _P_M_ALPHA),  # omega interpolated
        "Os": (1.9601, 1.9102, 0.0205, _P_M_ALPHA),  # omega interpolated
        "Ir": (2.0407, 1.9799, 0.0217, _P_M_ALPHA),  # omega interpolated
        "Pt": (2.1220, 2.0505, 0.0230, _P_M_ALPHA),
        "Au": (2.2050, 2.1229, 0.0250, _P_M_ALPHA),
        "Hg": (2.2949, 2.1953, 0.0267, _P_M_ALPHA),  # omega interpolated
        "Tl": (2.3893, 2.2706, 0.0283, _P_M_ALPHA),  # omega interpolated
        "Pb": (2.4840, 2.3455, 0.0300, _P_M_ALPHA),
        "Bi": (2.5800, 2.4227, 0.0320, _P_M_ALPHA),
        "Po": (2.6874, 2.5047, 0.0334, _P_M_ALPHA),  # edge, line AND omega interpolated
        "At": (2.7949, 2.5867, 0.0349, _P_M_ALPHA),  # edge, line AND omega interpolated
        "Rn": (2.9023, 2.6687, 0.0363, _P_M_ALPHA),  # edge, line AND omega interpolated
        "Fr": (3.0097, 2.7508, 0.0377, _P_M_ALPHA),  # edge, line AND omega interpolated
        "Ra": (3.1171, 2.8328, 0.0391, _P_M_ALPHA),  # edge, line AND omega interpolated
        "Ac": (3.2246, 2.9148, 0.0406, _P_M_ALPHA),  # edge, line AND omega interpolated
        "Th": (3.3320, 2.9968, 0.0420, _P_M_ALPHA),
        "Pa": (3.4420, 3.0838, 0.0435, _P_M_ALPHA),  # edge, line AND omega interpolated
        "U":  (3.5520, 3.1708, 0.0450, _P_M_ALPHA),
    },
}

#: Electrons in the ionised (sub)shell, and the Bethe constant b for it.
#: Green & Cosslett's K-shell values; L/M scaled in the usual way. c is taken
#: as 1, i.e. ln(U) rather than ln(cU) -- fitting c against the reference made
#: the K family *worse* (Fe went from 6 % to 19 % off), so it stays at the
#: textbook value.
_SHELL_ELECTRONS = {"K": 2.0, "L": 4.0, "M": 6.0}
_SHELL_B = {"K": 0.35, "L": 0.25, "M": 0.20}

#: Below this overvoltage a line is not usefully excited and we refuse to
#: quantify from it rather than return a wild extrapolation.
MIN_OVERVOLTAGE = 1.5

#: Elements that are in ATOMIC_MASSES but have no X-ray line an EDS detector
#: can see. Named so the error message can say why.
_NO_XRAY_LINE = {
    "H": "hydrogen has no core electrons",
    "He": "helium has no accessible X-ray line",
    "Li": "the Li K line at 0.054 keV is below any EDS detector window",
}


class UnknownLineError(ValueError):
    """No usable k-factor for this (element, line, beam voltage).

    A ``ValueError`` so existing ``except ValueError`` keeps working, but its
    own type so a caller can turn it into "this one window is unusable"
    instead of letting it take a whole page down. Carries the pieces
    separately for exactly that:

        try:
            at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts))
        except UnknownLineError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    ``element`` is the symbol, ``line`` the family (or ``None`` when the
    element has no line at all), and ``str(exc)`` names both plus what to do.
    """

    def __init__(self, message: str, element: str = "",
                 line: Optional[str] = None):
        super().__init__(message)
        self.element = element
        self.line = line


#: An X-ray line label as Aztec writes it: the shell letter, optionally a
#: greek (or transliterated) sub-line and its indices -- "K", "Ka1", "Kα1",
#: "Lα1,2", "Ll". Note that "Mg", "La" and "K" also match; callers disambiguate
#: by position, see parse_element_name.
_LINE_LABEL_RE = re.compile(
    r"^[KLM](?:[αβγ]|[abgl])?\d*(?:[,.]\d+)*$"
)


def _is_line_label(token: str) -> bool:
    return bool(_LINE_LABEL_RE.match(token))


class ElementLine(str):
    """An element symbol that remembers which X-ray line it was measured on.

    This *is* the symbol -- it hashes, compares, sorts, formats and
    JSON-serialises exactly like the plain ``str`` it replaces, so every
    existing consumer (``counts[el]``, ``el == "Fe"``, ``{el: value}`` in a
    response body) is untouched. The line rides along as an attribute, so the
    quantification can tell ``"Cu Lalpha"`` from ``"Cu Kalpha"`` -- which is
    the whole point, since those two need k-factors that differ by a factor of
    five.

    ``line`` is one of ``"K"``, ``"L"``, ``"M"`` or ``""`` when the source name
    did not say (a bare ``"Fe"``); ``sub_line`` keeps the rest of the label
    (``"alpha1"``, ``"beta1"``, ...) for the provenance.

    **Equality and hashing are by SYMBOL ONLY, deliberately.** ``Cu`` on K and
    ``Cu`` on L are the *same* key. A composition has one column per element,
    not one per window -- an element is quantified once, from one chosen line
    -- and every consumer downstream (the clustering, the phase rules,
    ``chemistry_fit``, the sidecar, the JSON going to the browser) indexes by
    the bare symbol. Making the line part of the identity would hide ``"Cu"``
    from all of them.

    The price is that ``{a: x, b: y}`` with two windows of one element silently
    keeps ONE entry -- and Python keeps the *first* key object with the *last*
    value, so the surviving key can carry the line of the window whose counts
    were thrown away. Measured 2026-09-12: a Cu K key holding the Cu L counts
    picks up k = 4.13 instead of 1.10, a silent 3.75x. Therefore **do not build
    the counts dict by hand** -- use :func:`build_counts_by_element`, which
    chooses one window per element deterministically and says which it dropped.
    """

    def __new__(cls, symbol: str, line: str = "", sub_line: str = ""):
        obj = super().__new__(cls, symbol)
        obj._line = line
        obj._sub_line = sub_line
        return obj

    # Plain attributes would be lost by copy/pickle round-trips of a str
    # subclass, so expose them read-only and reduce explicitly.
    @property
    def line(self) -> str:
        return self._line

    @property
    def sub_line(self) -> str:
        return self._sub_line

    @property
    def symbol(self) -> str:
        return str(self)

    def __reduce__(self):
        return (ElementLine, (str(self), self._line, self._sub_line))

    def __repr__(self) -> str:
        return f"ElementLine({str(self)!r}, {self._line!r})"


class QuantMap(dict):
    """A ``{element: array}`` result that carries how it was produced.

    A plain dict in every respect; ``.provenance`` is the extra. Point 4 of
    ``tasks/eds-quantification-wrong-2026-09-10.md``: the docstring knew the
    numbers were uncalibrated, the surface did not.
    """

    provenance: Dict

    def __init__(self, *args, provenance: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.provenance = provenance if provenance is not None else {}


@dataclass
class EDSPixelData:
    """EDS data for a single pixel or region."""
    elements: List[str]
    counts: Dict[str, float]         # Raw counts per element
    weight_pct: Dict[str, float]     # Weight percent per element
    atomic_pct: Dict[str, float]     # Atomic percent per element
    provenance: Dict = field(default_factory=dict)


def parse_element_name(raw_name: str) -> ElementLine:
    """
    Extract the element symbol -- and the X-ray line -- from an H5OINA name.

    H5OINA uses forms like ``"Fe Kα1"``, ``"Cu Lα1,2"``, ``"Al Ka1"``,
    ``"Window Integral Al Ka1"`` or just ``"Fe"`` (the older group layout).

    Returns an :class:`ElementLine`, which behaves as the bare symbol
    everywhere a ``str`` is expected, so callers that only want ``"Fe"`` need
    no change. The difference is that the line is no longer thrown away:
    ``parse_element_name("Cu Lα1,2").line == "L"``. Until 2026-09-12 it was,
    and Cu/Zn L lines were therefore quantified with K-line factors.

    A name that carries no line (``"Fe"``) yields ``line == ""``; the
    quantification then picks a line by overvoltage and says so in its
    provenance rather than guessing silently.
    """
    parts = str(raw_name).split()
    if not parts:
        return ElementLine("", "", "")
    if len(parts) == 1:
        return ElementLine(parts[0].strip(), "", "")

    last, prev = parts[-1].strip(), parts[-2].strip()
    # "Mg", "La" and "K" read as line labels as much as as element symbols, so
    # position decides: a trailing label is the line only when the token in
    # front of it is itself an element ("Al K" -> Al on K), otherwise the
    # trailing token is the element ("Window Integral Mg" -> Mg).
    if _is_line_label(last) and (prev in ATOMIC_MASSES or last not in ATOMIC_MASSES):
        return ElementLine(prev, last[0].upper(), last[1:])
    return ElementLine(last if last in ATOMIC_MASSES else parts[0].strip(), "", "")


#: Preference among sub-lines of the same shell, best first. The k-factor
#: model describes the alpha window; a Kbeta window of the same element is a
#: different measurement and is only ever a fallback.
def _sub_line_rank(sub_line: str) -> int:
    s = (sub_line or "").lower()
    if not s or s.startswith(("a", "α")):
        return 0            # alpha (or an unqualified "Fe K")
    if s.startswith(("b", "β")):
        return 1
    return 2


def build_counts_by_element(
    raw_counts,
    beam_kv: float = 20.0,
) -> Dict[ElementLine, np.ndarray]:
    """Build the ``{element: counts}`` dict quantification expects.

    Use this instead of ``counts[parse_element_name(name)] = array`` in a
    loop. :class:`ElementLine` keys compare by symbol, so writing two windows
    of one element into a dict by hand collapses them -- keeping the first
    key object and the last value, which can pair a K-line key with L-line
    counts and a k-factor that is out by 3.75x with no warning at all.

    Here the choice is explicit and deterministic:

    1. group the windows by element symbol;
    2. within one element prefer the line :func:`default_line_for` would pick
       at this beam voltage -- the same single policy used for a bare name --
       falling back to K, then L, then M among the windows actually present;
    3. within one shell prefer the alpha sub-line over beta;
    4. log which window was dropped.

    Args:
        raw_counts: a mapping ``{raw_name: array}`` or any iterable of
            ``(raw_name, array)`` pairs. ``None`` arrays are skipped.
        beam_kv: accelerating voltage, for the line preference.

    Returns:
        ``{ElementLine: np.ndarray}`` with exactly one entry per element.
    """
    items = (raw_counts.items() if hasattr(raw_counts, "items")
             else list(raw_counts))

    by_symbol: Dict[str, list] = {}
    for raw_name, data in items:
        if data is None:
            continue
        key = parse_element_name(raw_name)
        by_symbol.setdefault(str(key), []).append((key, raw_name, data))

    result: Dict[ElementLine, np.ndarray] = {}
    for symbol, windows in by_symbol.items():
        if len(windows) > 1:
            try:
                preferred = default_line_for(symbol, beam_kv)
            except UnknownLineError:
                preferred = ""
            order = [preferred] + [f for f in ("K", "L", "M") if f != preferred]

            def rank(win):
                key = win[0]
                fam = key.line or ""
                return (order.index(fam) if fam in order else len(order),
                        _sub_line_rank(key.sub_line))

            windows.sort(key=rank)
            logger.warning(
                "EDS quantification: %d windows for %s (%s); using %r and "
                "ignoring %s. An element gets one column, so one window has "
                "to win -- see build_counts_by_element.",
                len(windows), symbol, ", ".join(repr(w[1]) for w in windows),
                windows[0][1], ", ".join(repr(w[1]) for w in windows[1:]),
            )
        key, _raw_name, data = windows[0]
        result[key] = data
    return result


def default_line_for(element: str, beam_kv: float = 20.0) -> str:
    """Which line family to assume when the source name did not say.

    Standard microanalysis practice: take the hardest line that is still
    adequately excited, i.e. K if its overvoltage clears
    :data:`MIN_OVERVOLTAGE`, else L, else M. At 20 kV that gives K for
    everything up to about Zn and L above it.

    Raises ``ValueError`` when the element has no line this beam can excite --
    quantifying it would be an invention.
    """
    for family in ("K", "L", "M"):
        entry = _XRAY_LINES[family].get(element)
        if entry is not None and beam_kv / entry[0] >= MIN_OVERVOLTAGE:
            return family
    if element in _NO_XRAY_LINE:
        raise UnknownLineError(
            f"EDS quantification: {element!r} has no usable X-ray line "
            f"({_NO_XRAY_LINE[element]}).",
            element=element,
        )
    known = [f for f in ("K", "L", "M") if element in _XRAY_LINES[f]]
    if known:
        edges = ", ".join(
            f"{f} edge {_XRAY_LINES[f][element][0]:.3f} keV" for f in known
        )
        raise UnknownLineError(
            f"EDS quantification: no line of {element!r} is excited at "
            f"{beam_kv:g} kV (need overvoltage >= {MIN_OVERVOLTAGE}; {edges}). "
            "Raise the beam voltage or drop this element.",
            element=element,
        )
    raise UnknownLineError(
        f"EDS quantification: no X-ray line data for element {element!r}. "
        "Add it to _XRAY_LINES in eds_utils.py (edge, line energy, "
        "fluorescence yield, line fraction) -- refusing to guess a k-factor.",
        element=element,
    )


def _emission_k(element: str, family: str, beam_kv: float) -> float:
    """Physics-only k: atomic mass over emitted line photons per unit mass."""
    entry = _XRAY_LINES.get(family, {}).get(element)
    if entry is None:
        available = [f for f in ("K", "L", "M") if element in _XRAY_LINES[f]]
        hint = (f" (have {', '.join(available)} for {element})"
                if available else "")
        raise UnknownLineError(
            f"EDS quantification: no {family} line data for element "
            f"{element!r}{hint}. Refusing a silent k = 1.0 -- add the line to "
            "_XRAY_LINES in eds_utils.py.",
            element=element, line=family,
        )
    edge_kev, _line_kev, omega, p = entry
    u = beam_kv / edge_kev
    # The same gate default_line_for applies when it *chooses* a line. It has
    # to be here too, or a window that NAMES its line walks straight past it:
    # ln(U) -> 0 as U -> 1, so k -> infinity and that element takes 100 % of
    # the composition. Measured before this guard existed: k("Mo", "K") at
    # 20 kV (U = 1.0000) came out at 2.6e+05 and turned equal Al/Mo counts
    # into Al 0.0003 wt%, Mo 99.9997 wt% with no error at all.
    if u < MIN_OVERVOLTAGE:
        raise UnknownLineError(
            f"EDS quantification: {element} {family} (edge {edge_kev:.3f} keV) "
            f"is not usefully excited at {beam_kv:g} kV -- overvoltage "
            f"{u:.2f}, need at least {MIN_OVERVOLTAGE}. Raise the beam "
            f"voltage, pick another line of {element}, or drop the element.",
            element=element, line=family,
        )
    mass = ATOMIC_MASSES.get(element)
    if mass is None:
        raise UnknownLineError(
            f"EDS quantification: no atomic mass for element {element!r}; "
            "add it to ATOMIC_MASSES in eds_utils.py.",
            element=element, line=family,
        )
    ionisations = _SHELL_ELECTRONS[family] * _SHELL_B[family] * math.log(u) / edge_kev
    emitted_per_unit_mass = omega * p * ionisations / mass
    return 1.0 / emitted_per_unit_mass


# ---------------------------------------------------------------------------
# Instrument response -- the one calibration
# ---------------------------------------------------------------------------
#: The standard the response curve is measured against. Everything the
#: emission model above does not describe is lumped into R(E_line) and
#: measured here, once.
#:
#: Region: the alpha-Al(Fe,Mn)Si interior of SampleB, phase 4 of the 4-phase
#: spherical-GPU map `sampleb_4ph.npz`, eroded twice with a 3x3 element so no
#: boundary pixel (which the 20 kV interaction volume mixes with the Al matrix)
#: contributes: 12744 px -> 10225 px. File: "...B_SA_EBSD Arbeitsbereich 4
#: Elementverteilungsdaten 11.h5oina", 268 x 201, 20 kV, Oxford SATW detector
#: UVA11191. Reference: Aztec Tru-Q on the same raw spectra, normalised.
#: Measured 2026-09-12; see tasks/eds-quantification-wrong-2026-09-10.md.
_CALIBRATION_STANDARD = {
    "label": "SampleB alpha-Al(Fe,Mn)Si interior vs Oxford Aztec Tru-Q",
    "matrix": "Al-rich intermetallic in an Al extrusion alloy",
    "beam_kv": 20.0,
    "n_px": 10225,
    "detector": "Oxford SATW, serial UVA11191",
    "measured_on": "2026-09-12",
    # mean raw Window-Integral counts over the region
    "counts": {
        "Al": 175130.9, "Si": 22751.1, "Fe": 20698.8,
        "Mn": 6619.9, "Cu": 24989.6, "Zn": 10727.2,
    },
    # the line each of those windows is
    "lines": {
        "Al": "K", "Si": "K", "Fe": "K", "Mn": "K", "Cu": "L", "Zn": "L",
    },
    # Aztec Tru-Q At.% for the same region (sums to 100.1 -- Aztec's rounding;
    # only ratios are used, so it does not matter)
    "reference_at_pct": {
        "Al": 66.0, "Si": 10.0, "Fe": 14.0, "Mn": 3.7, "Cu": 5.7, "Zn": 0.7,
    },
    # Excluded from the curve, with the reason. Zn's response would come out at
    # 0.06 against Cu's 0.21 eighty eV away -- a physically impossible dip --
    # because the Zn L window is not a clean Zn measurement here (see
    # _SPECTRAL_OVERLAPS). Letting it anchor the curve would quietly corrupt
    # every other line in the 0.9-1.5 keV band (Na K, Mg K, Ga L, Ge L).
    "excluded": {
        "Zn": "Zn L window contaminated by Cu L emission -- spectral overlap, "
              "not a k-factor; see _SPECTRAL_OVERLAPS",
    },
}

#: Known window interferences. Reported, never silently corrected: the
#: contribution scales with the interfering element's concentration, so
#: folding it into a constant k would be right on this sample and wrong
#: everywhere else.
_SPECTRAL_OVERLAPS = [
    {
        "window": ("Zn", "L"),
        "interferer": ("Cu", "L"),
        "effect": "Zn L reads high wherever Cu is present",
        "measured": "Zn counts ~ 0.29 * Cu counts + const across the three "
                    "phase regions of SampleB (holds to ~5 % over an 8x range "
                    "in Cu), 2026-09-12",
        "fix": "quantify from 1/EDS/Data/Spectrum with peak deconvolution "
               "(point 3 of tasks/eds-quantification-wrong-2026-09-10.md)",
    },
]


def _build_response_curve():
    """Derive R(E_line) from the stored standard.

    Deriving rather than hard-coding the anchors keeps the calibration honest
    if a physics constant above is ever corrected: R is *defined* as
    "measured k over model k", so the pair always reproduces the standard.
    """
    std = _CALIBRATION_STANDARD
    kv = std["beam_kv"]
    ref_at = std["reference_at_pct"]
    # At.% -> Wt.%
    grams = {el: ref_at[el] * ATOMIC_MASSES[el] for el in ref_at}
    total = sum(grams.values())
    ref_wt = {el: 100.0 * g / total for el, g in grams.items()}

    anchors = []
    for el, counts in std["counts"].items():
        if el in std["excluded"]:
            continue
        family = std["lines"][el]
        # measured k (Cliff-Lorimer: C ~ k I) and model k, both up to the same
        # arbitrary global scale -- which cancels in the 100 % normalisation.
        k_measured = ref_wt[el] / counts
        k_model = _emission_k(el, family, kv)
        line_kev = _XRAY_LINES[family][el][1]
        anchors.append((line_kev, k_measured / k_model))

    anchors.sort()
    # Normalise the curve so it reads 1.0 on its plateau; purely cosmetic (a
    # global factor on every k cancels), but it makes the numbers readable as
    # "fraction of the photons the model expected that actually arrived".
    scale = max(r for _, r in anchors)
    energies = np.array([e for e, _ in anchors], dtype=np.float64)
    responses = np.array([r / scale for _, r in anchors], dtype=np.float64)
    return energies, responses


_RESPONSE_ENERGY_KEV, _RESPONSE_VALUE = _build_response_curve()
_LOG_RESPONSE_ENERGY = np.log(_RESPONSE_ENERGY_KEV)
_LOG_RESPONSE_VALUE = np.log(_RESPONSE_VALUE)

#: The energy band the response was actually measured over. Outside it the
#: curve is clamped and the provenance says "extrapolated".
RESPONSE_CALIBRATED_RANGE_KEV = (
    float(_RESPONSE_ENERGY_KEV[0]), float(_RESPONSE_ENERGY_KEV[-1]),
)


def instrument_response(line_kev: float) -> float:
    """Measured response at a line energy: log-log interpolation, clamped."""
    return float(np.exp(np.interp(
        math.log(line_kev), _LOG_RESPONSE_ENERGY, _LOG_RESPONSE_VALUE)))


def k_factor(element: str, line: Optional[str] = None,
             beam_kv: float = 20.0) -> float:
    """Cliff-Lorimer k-factor for one element on one X-ray line.

    Convention: ``C_i = k_i * I_i / sum_j (k_j * I_j)``. Larger k means the
    line is a *weaker* emitter per unit weight and its counts have to be
    scaled up.

    Args:
        element: element symbol, e.g. ``"Fe"``.
        line: ``"K"``, ``"L"`` or ``"M"``. ``None`` or ``""`` picks one by
            overvoltage via :func:`default_line_for`.
        beam_kv: accelerating voltage. The emission term follows it; the
            measured response curve does not (it was taken at 20 kV), so
            quantification well away from 20 kV drifts -- the provenance
            flags it.

    Raises:
        ValueError: for an element or line we hold no data for, or one the
            beam cannot excite. Never returns a silent 1.0.
    """
    element = str(element)
    family = (line or "").upper() or default_line_for(element, beam_kv)
    if family not in _XRAY_LINES:
        raise UnknownLineError(
            f"EDS quantification: unknown X-ray line family {line!r} for "
            f"{element!r}; expected one of K, L, M.",
            element=element, line=str(line),
        )
    k_model = _emission_k(element, family, beam_kv)
    line_kev = _XRAY_LINES[family][element][1]
    # Reference the scale to Si K at the calibration voltage so the returned
    # numbers read as "relative to Si Kalpha", the usual convention.
    k_si = _emission_k("Si", "K", _CALIBRATION_STANDARD["beam_kv"])
    return (k_model / k_si) * instrument_response(line_kev)


def quantification_provenance(beam_kv: float = 20.0) -> Dict:
    """The standing caveats on any At.%/Wt.% this module produces."""
    std = _CALIBRATION_STANDARD
    lo, hi = RESPONSE_CALIBRATED_RANGE_KEV
    return {
        "summary": (
            "k-factors: Bethe/Green-Cosslett ionisation x Krause fluorescence "
            "yields x measured instrument response, calibrated on an Al-rich "
            f"matrix at {std['beam_kv']:g} kV ({std['label']}). "
            "NO ZAF matrix correction -- treat as semi-quantitative."
        ),
        "method": "standardless Cliff-Lorimer, C_i proportional to k_i * I_i",
        "matrix_correction": None,
        "source_data": "vendor Window Integral, not the spectrum cube",
        "beam_kv": float(beam_kv),
        "calibration": {
            "standard": std["label"],
            "matrix": std["matrix"],
            "reference": "Oxford Aztec Tru-Q",
            "beam_kv": std["beam_kv"],
            "n_px": std["n_px"],
            "detector": std["detector"],
            "measured_on": std["measured_on"],
            "response_range_kev": [lo, hi],
            "excluded": dict(std["excluded"]),
        },
        "lines": {},
        "spectral_overlaps": [],
        # Windows that could not be quantified and were left out, one entry
        # each. See _resolve_k_factors: an element with no usable k costs its
        # own window and nothing else.
        "excluded_windows": [],
        "warnings": [],
    }


def _resolve_k_factors(counts, k_factors, beam_kv):
    """Pick a k per element, recording where each one came from.

    **The blast radius of an unusable window is that window.** A stray label,
    an element outside the line tables, or a line this beam cannot excite used
    to raise out of this loop and take every OTHER element with it: a scan with
    Al, Si and one unreadable window answered 400 for the whole tooltip, the
    whole pixel quantification and the whole element map, although the two real
    elements were sitting in the same dict fully quantifiable. That contradicted
    :class:`UnknownLineError`'s own docstring, which promises a caller can turn
    it into "this one window is unusable".

    So each element is resolved on its own. A failure drops that element from
    the returned k map — the caller must then quantify only the elements it got
    back — and is recorded twice: as a structured entry in
    ``provenance["excluded_windows"]`` and as a sentence in
    ``provenance["warnings"]``, because a silent drop would be the worse bug
    (the remaining elements renormalise to 100 %, so nothing on screen says a
    window went missing).

    Only when NOTHING is left does the first failure propagate: an empty answer
    is not a degraded answer. Re-raising the first exception rather than a
    generic "nothing quantifiable" keeps the message specific to what actually
    went wrong.
    """
    prov = quantification_provenance(beam_kv)
    lo, hi = RESPONSE_CALIBRATED_RANGE_KEV
    resolved: Dict[str, float] = {}
    first_failure: Optional[UnknownLineError] = None

    for elem in counts:
        symbol = str(elem)
        try:
            _resolve_one_k(elem, symbol, counts, k_factors, beam_kv,
                           resolved, prov, lo, hi)
        except UnknownLineError as exc:
            if first_failure is None:
                first_failure = exc
            resolved.pop(elem, None)
            prov["lines"].pop(symbol, None)
            line = getattr(elem, "line", "") or getattr(exc, "line", "") or None
            prov["excluded_windows"].append({
                "element": symbol,
                "line": line,
                "reason": str(exc),
            })
            prov["warnings"].append(
                f"{symbol}"
                f"{' ' + str(line) if line else ''} was left out of the "
                f"quantification: {exc}. The remaining elements are "
                "renormalised to 100 % without it."
            )
            # DEBUG, not WARNING: this runs once per PIXEL on the chemistry
            # prior, so a systematic cause would write one identical line per
            # pixel of the map. The loud channel is the provenance above --
            # the routes forward `warnings` and `excluded_windows` to the UI,
            # and callers that fire per pixel (eds_pixel_chemistry) log it
            # once per dataset through their own dedupe.
            logger.debug("EDS quantification: skipping window %s (%s)",
                         symbol, exc)

    if not resolved:
        if first_failure is not None:
            raise first_failure
        return resolved, prov

    if abs(beam_kv - _CALIBRATION_STANDARD["beam_kv"]) > 0.51:
        prov["warnings"].append(
            f"beam voltage {beam_kv:g} kV differs from the "
            f"{_CALIBRATION_STANDARD['beam_kv']:g} kV the response curve was "
            "measured at; the absorption part of the response no longer applies"
        )

    # Only over the windows that survived — a dropped one has no entry in
    # prov["lines"] to read a line off, and it contributes no counts either.
    present = {(str(e), getattr(e, "line", "") or
                prov["lines"][str(e)].get("line")) for e in resolved}
    for overlap in _SPECTRAL_OVERLAPS:
        if overlap["window"] in present and overlap["interferer"] in present:
            prov["spectral_overlaps"].append(dict(overlap))
            prov["warnings"].append(
                "{} {} window overlaps {} {} emission: {}".format(
                    *overlap["window"], *overlap["interferer"],
                    overlap["effect"])
            )
    return resolved, prov


def _resolve_one_k(elem, symbol, counts, k_factors, beam_kv, resolved, prov,
                   lo, hi) -> None:
    """One element's k and provenance, or :class:`UnknownLineError`.

    Split out of :func:`_resolve_k_factors` so that function can catch a
    failure per element without the loop body's ``continue`` having to mean
    two different things. Writes into ``resolved`` and ``prov`` in place; the
    caller undoes a partial write when this raises.
    """
    if k_factors is not None and elem in k_factors:
        resolved[elem] = float(k_factors[elem])
        # Still work out which line this is, so the overlap check has
        # something to match on. A caller-supplied k means we do not know the
        # physics, not that the spectrum stopped overlapping.
        caller_line = getattr(elem, "line", "") or ""
        if not caller_line:
            try:
                caller_line = default_line_for(symbol, beam_kv)
            except UnknownLineError:
                caller_line = ""
                prov["warnings"].append(
                    f"{symbol}: k-factor supplied by the caller and no "
                    "line could be determined, so this element is "
                    "excluded from the spectral-overlap check"
                )
        prov["lines"][symbol] = {"line": caller_line or None,
                                 "k": resolved[elem], "source": "caller"}
        return

    family = getattr(elem, "line", "") or ""
    assumed = False
    if not family:
        family = default_line_for(symbol, beam_kv)
        assumed = True
    k = k_factor(symbol, family, beam_kv)
    line_kev = _XRAY_LINES[family][symbol][1]
    entry = {"line": family, "k": k, "line_kev": line_kev,
             "source": "model+response"}
    sub = getattr(elem, "sub_line", "") or ""
    if _sub_line_rank(sub) != 0:
        # The model's p is the alpha window's line fraction and the
        # response curve was measured on alpha windows. A Kbeta window is
        # a different measurement, some 8x weaker, and quantifying it with
        # the alpha k would read ~8x low -- the module's own stance is
        # that we do not hand back a number we cannot stand behind.
        raise UnknownLineError(
            f"EDS quantification: {symbol} window is {family}{sub}, but "
            f"the k-factor model only describes the {family}-alpha "
            f"window. Use the {family}-alpha window for {symbol} (if the "
            "file has both, build_counts_by_element picks it), or add "
            f"{family}{sub} to _XRAY_LINES.",
            element=symbol, line=family,
        )
    if assumed:
        entry["line_assumed"] = True
        prov["warnings"].append(
            f"{symbol}: the source name carried no X-ray line, assumed "
            f"{family} from the overvoltage at {beam_kv:g} kV"
        )
    if not (lo <= line_kev <= hi):
        entry["source"] = "model+response(extrapolated)"
        prov["warnings"].append(
            f"{symbol} {family} at {line_kev:.3f} keV lies outside the "
            f"calibrated response range {lo:.2f}-{hi:.2f} keV; its "
            "response is clamped to the nearest anchor"
        )
    # Written last, and only once nothing can still raise: a window that is
    # dropped must leave no trace of a k it never got.
    resolved[elem] = k
    prov["lines"][symbol] = entry


def counts_to_weight_pct(
    counts: Dict[str, np.ndarray],
    k_factors: Optional[Dict[str, float]] = None,
    beam_kv: float = 20.0,
) -> QuantMap:
    """
    Convert raw EDS counts to weight percent by standardless Cliff-Lorimer.

        C_i = k_i * I_i / sum_j (k_j * I_j) * 100

    Note the direction: weight fraction is proportional to ``k * I``. Before
    2026-09-12 this divided by k, and the table it divided by was flat, which
    is why Fe read a third of its true value. If you pass ``k_factors``, pass
    them in this convention.

    Args:
        counts: element symbol -> counts array (any shape). Keys produced by
            :func:`parse_element_name` carry the X-ray line, which is what
            lets Cu L be told apart from Cu K; a plain ``str`` key falls back
            to :func:`default_line_for` and says so in the provenance.
        k_factors: optional override per element, in the convention above.
            Elements not listed fall back to :func:`k_factor`.
        beam_kv: accelerating voltage, for the overvoltage term.

    Returns:
        :class:`QuantMap` -- a dict of weight-percent arrays that also carries
        ``.provenance`` (model, calibration, per-line k, warnings, known
        spectral overlaps). Read it before presenting the numbers as
        quantitative.

    Raises:
        ValueError: if an element or line has no physics data, or the beam
            cannot excite it. It never falls back to a silent k = 1.0.
    """
    if not counts:
        return QuantMap(provenance=quantification_provenance(beam_kv))

    resolved, prov = _resolve_k_factors(counts, k_factors, beam_kv)

    # Weight-proportional signal per element: k_i * I_i.
    # Over `resolved`, not over `counts`: a window with no usable k-factor was
    # dropped there and named in the provenance, and it must not reappear here
    # with a fabricated k. The remaining elements renormalise to 100 % without
    # it — see _resolve_k_factors for why that is a warning and not an error.
    weighted: Dict[str, np.ndarray] = {}
    for elem, data in counts.items():
        if elem not in resolved:
            continue
        weighted[elem] = np.asarray(data, dtype=np.float64) * resolved[elem]

    total = None
    for arr in weighted.values():
        total = arr.copy() if total is None else total + arr

    # Avoid division by zero
    total = np.where(total > 0, total, 1.0)

    result = QuantMap(provenance=prov)
    for elem, arr in weighted.items():
        result[elem] = (arr / total) * 100.0
    return result


def weight_pct_to_atomic_pct(
    weight_pct: Dict[str, np.ndarray],
) -> QuantMap:
    """
    Convert weight percent to atomic percent.

    Formula: At.%_i = (Wt.%_i / M_i) / sum(Wt.%_j / M_j) * 100

    Where M_i is the atomic mass of element i.

    Args:
        weight_pct: Dict mapping element symbol -> weight percent array

    Returns:
        :class:`QuantMap` of atomic-percent arrays, carrying through the
        ``.provenance`` of the weight percents it was given.

    Raises:
        ValueError: for an element with no tabulated atomic mass. The old
            default of 1.0 g/mol was not a fallback but a landmine -- it makes
            that element outweigh every other by a factor of 30 or more.
    """
    if not weight_pct:
        return QuantMap(provenance=getattr(weight_pct, "provenance", {}) or {})

    # Compute Wt.%_i / M_i for each element
    molar: Dict[str, np.ndarray] = {}
    for elem, wt_arr in weight_pct.items():
        mass = ATOMIC_MASSES.get(str(elem))
        if mass is None:
            raise UnknownLineError(
                f"EDS quantification: no atomic mass for element {elem!r}; "
                "add it to ATOMIC_MASSES in eds_utils.py.",
                element=str(elem),
            )
        molar[elem] = np.asarray(wt_arr, dtype=np.float64) / mass

    # Sum of all molar fractions (per-pixel)
    total = None
    for arr in molar.values():
        if total is None:
            total = arr.copy()
        else:
            total += arr

    # Avoid division by zero
    total = np.where(total > 0, total, 1.0)

    # Atomic percent
    result = QuantMap(provenance=getattr(weight_pct, "provenance", {}) or {})
    for elem, mol_arr in molar.items():
        result[elem] = (mol_arr / total) * 100.0

    return result


def counts_to_all(
    counts: Dict[str, np.ndarray],
    k_factors: Optional[Dict[str, float]] = None,
    beam_kv: float = 20.0,
) -> Tuple[Dict[str, np.ndarray], QuantMap, QuantMap]:
    """
    Convert raw counts to both weight percent and atomic percent.

    Args:
        counts: Dict mapping element symbol -> counts array
        k_factors: Optional custom k-factors (C ~ k * I convention)
        beam_kv: Accelerating voltage, for the overvoltage term

    Returns:
        Tuple of (counts, weight_pct, atomic_pct); the latter two carry
        ``.provenance``.
    """
    wt_pct = counts_to_weight_pct(counts, k_factors, beam_kv)
    at_pct = weight_pct_to_atomic_pct(wt_pct)
    return counts, wt_pct, at_pct


def get_pixel_composition(
    counts: Dict[str, np.ndarray],
    row: int,
    col: int,
    n_cols: int,
    k_factors: Optional[Dict[str, float]] = None,
    beam_kv: float = 20.0,
) -> EDSPixelData:
    """
    Get full EDS composition for a single pixel.

    Args:
        counts: Dict mapping element symbol -> 1D counts array (flattened)
        row: Pixel row index
        col: Pixel column index
        n_cols: Number of columns in the grid
        k_factors: Optional custom k-factors (C ~ k * I convention)
        beam_kv: Accelerating voltage, for the overvoltage term

    Returns:
        EDSPixelData with counts, wt.%, at.% and the quantification provenance
    """
    idx = row * n_cols + col

    pixel_counts = {}
    for elem, arr in counts.items():
        if idx < len(arr):
            pixel_counts[elem] = float(arr[idx])
        else:
            pixel_counts[elem] = 0.0

    # Convert to arrays for the conversion functions
    pixel_counts_arrays = {
        elem: np.array([val]) for elem, val in pixel_counts.items()
    }

    wt_pct_arrays = counts_to_weight_pct(pixel_counts_arrays, k_factors, beam_kv)
    at_pct_arrays = weight_pct_to_atomic_pct(wt_pct_arrays)

    elements = sorted(pixel_counts.keys())
    wt_pct = {elem: float(wt_pct_arrays[elem][0]) for elem in elements}
    at_pct = {elem: float(at_pct_arrays[elem][0]) for elem in elements}

    return EDSPixelData(
        elements=elements,
        counts=pixel_counts,
        weight_pct=wt_pct,
        atomic_pct=at_pct,
        provenance=at_pct_arrays.provenance,
    )


@dataclass
class PhaseMatch:
    """Result of matching a composition to a phase."""
    phase_name: str
    score: float              # 0.0 (no match) to 1.0 (perfect match)
    expected_composition: Dict[str, float]  # At.% from phase definition


def suggest_phases(
    measured_at_pct: Dict[str, float],
    phase_definitions: Dict[str, Dict[str, float]],
    tolerance: float = 15.0,
) -> List[PhaseMatch]:
    """
    Suggest matching crystal phases based on measured atomic percent composition.

    Compares measured At.% with expected phase compositions using
    a simple distance metric. Only elements present in the phase
    definition are compared.

    Args:
        measured_at_pct: Measured At.% per element (e.g., {"Fe": 50.0, "O": 50.0})
        phase_definitions: Dict of phase_name -> {element: expected_at_pct}
            Example: {"Fe2O3": {"Fe": 40.0, "O": 60.0}}
        tolerance: Maximum allowed deviation per element in At.%

    Returns:
        List of PhaseMatch objects sorted by score (best first)
    """
    matches = []

    for phase_name, expected in phase_definitions.items():
        # Check if all expected elements are present in measurement
        all_present = all(
            elem in measured_at_pct for elem in expected
        )

        if not all_present:
            continue

        # Calculate score based on element-wise deviation
        deviations = []
        for elem, expected_pct in expected.items():
            measured_pct = measured_at_pct.get(elem, 0.0)
            deviation = abs(measured_pct - expected_pct)
            deviations.append(deviation)

        if not deviations:
            continue

        max_deviation = max(deviations)
        avg_deviation = sum(deviations) / len(deviations)

        # Skip if any element is too far off
        if max_deviation > tolerance * 2:
            continue

        # Score: 1.0 = perfect match, 0.0 = at tolerance limit
        score = max(0.0, 1.0 - (avg_deviation / tolerance))

        matches.append(PhaseMatch(
            phase_name=phase_name,
            score=score,
            expected_composition=expected,
        ))

    # Sort by score (highest first)
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


# Common Fe-bearing phase definitions (At.%)
# These are approximate stoichiometric compositions for phase pre-selection
COMMON_FE_PHASES: Dict[str, Dict[str, float]] = {
    "alpha-Fe (Ferrite/BCC Iron)": {"Fe": 100.0},
    "gamma-Fe (Austenite/FCC Iron)": {"Fe": 100.0},
    "Fe3C (Cementite)": {"Fe": 75.0, "C": 25.0},
    "Fe2O3 (Hematite)": {"Fe": 40.0, "O": 60.0},
    "Fe3O4 (Magnetite)": {"Fe": 42.9, "O": 57.1},
    "FeO (Wuestite)": {"Fe": 50.0, "O": 50.0},
    "FeS2 (Pyrite)": {"Fe": 33.3, "S": 66.7},
    "FeTiO3 (Ilmenite)": {"Fe": 20.0, "Ti": 20.0, "O": 60.0},
    "Fe2SiO4 (Fayalite)": {"Fe": 28.6, "Si": 14.3, "O": 57.1},
    "FeCr2O4 (Chromite)": {"Fe": 14.3, "Cr": 28.6, "O": 57.1},
    "Fe-3%Si (Electrical Steel)": {"Fe": 94.1, "Si": 5.9},
    "AISI 304 (Austenitic SS)": {"Fe": 68.0, "Cr": 19.0, "Ni": 10.0, "Mn": 2.0, "Si": 1.0},
    "AISI 316L (Austenitic SS)": {"Fe": 65.0, "Cr": 17.0, "Ni": 12.0, "Mo": 2.5, "Mn": 2.0},
    "Fe-Ni (Taenite)": {"Fe": 70.0, "Ni": 30.0},
    "FeAl (Iron Aluminide)": {"Fe": 50.0, "Al": 50.0},
    "Fe3Al": {"Fe": 75.0, "Al": 25.0},
}

COMMON_AL_PHASES: Dict[str, Dict[str, float]] = {
    "Al (Aluminum)": {"Al": 100.0},
    "Al2O3 (Corundum)": {"Al": 40.0, "O": 60.0},
    "Al2SiO5 (Sillimanite)": {"Al": 25.0, "Si": 12.5, "O": 62.5},
    "AlN (Aluminum Nitride)": {"Al": 50.0, "N": 50.0},
}

COMMON_SI_PHASES: Dict[str, Dict[str, float]] = {
    "Si (Silicon)": {"Si": 100.0},
    "SiO2 (Quartz)": {"Si": 33.3, "O": 66.7},
    "SiC (Silicon Carbide)": {"Si": 50.0, "C": 50.0},
}

# Al-Fe-Si intermetallic phases common in Al extrusion alloys (At.%)
COMMON_ALFE_PHASES: Dict[str, Dict[str, float]] = {
    "Al6Fe": {"Al": 85.7, "Fe": 14.3},
    "alpha-AlFeSi (Al8Fe2Si)": {"Al": 72.7, "Fe": 18.2, "Si": 9.1},
    "beta-AlFeSi (Al5FeSi)": {"Al": 71.4, "Fe": 14.3, "Si": 14.3},
    "Al13Fe4": {"Al": 76.5, "Fe": 23.5},
    "Al3Fe": {"Al": 75.0, "Fe": 25.0},
}

# Combined default phase library
DEFAULT_PHASE_LIBRARY: Dict[str, Dict[str, float]] = {
    **COMMON_FE_PHASES,
    **COMMON_AL_PHASES,
    **COMMON_SI_PHASES,
    **COMMON_ALFE_PHASES,
}


def filter_pixels_by_chemistry(
    counts: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    element: str,
    min_at_pct: float = 0.0,
    max_at_pct: float = 100.0,
    k_factors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Create a boolean mask of pixels matching a chemistry criterion.

    Selects pixels where the At.% of a given element falls within
    [min_at_pct, max_at_pct]. Useful for selective indexing — e.g.,
    only index Fe-rich pixels.

    Args:
        counts: Dict mapping element symbol -> 1D counts array (flattened)
        n_rows: Number of rows in the scan grid
        n_cols: Number of columns in the scan grid
        element: Element symbol to filter on (e.g., "Fe")
        min_at_pct: Minimum At.% threshold (inclusive)
        max_at_pct: Maximum At.% threshold (inclusive)
        k_factors: Optional custom k-factors

    Returns:
        2D boolean mask (n_rows, n_cols) — True where criterion is met
    """
    # Convert all counts to At.%
    _, _, at_pct = counts_to_all(counts, k_factors)

    if element not in at_pct:
        logger.warning("Element '%s' not in EDS data, returning empty mask", element)
        return np.zeros((n_rows, n_cols), dtype=bool)

    elem_at = at_pct[element]
    # Reshape to 2D
    total = n_rows * n_cols
    if len(elem_at) < total:
        padded = np.zeros(total)
        padded[:len(elem_at)] = elem_at
        elem_at = padded

    elem_2d = elem_at[:total].reshape(n_rows, n_cols)
    mask = (elem_2d >= min_at_pct) & (elem_2d <= max_at_pct)
    return mask


def filter_pixels_by_phase(
    counts: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    phase_name: str,
    phase_library: Optional[Dict[str, Dict[str, float]]] = None,
    min_score: float = 0.5,
    k_factors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Create a boolean mask of pixels matching a phase composition.

    For each pixel, computes the phase match score and selects pixels
    where the score exceeds min_score.

    Args:
        counts: Dict mapping element symbol -> 1D counts array (flattened)
        n_rows: Number of rows in the scan grid
        n_cols: Number of columns in the scan grid
        phase_name: Name of the phase in the library
        phase_library: Phase definitions dict; defaults to DEFAULT_PHASE_LIBRARY
        min_score: Minimum match score (0-1) to include pixel
        k_factors: Optional custom k-factors

    Returns:
        2D boolean mask (n_rows, n_cols) — True where phase matches
    """
    if phase_library is None:
        phase_library = DEFAULT_PHASE_LIBRARY

    if phase_name not in phase_library:
        logger.warning("Phase '%s' not in library, returning empty mask", phase_name)
        return np.zeros((n_rows, n_cols), dtype=bool)

    total = n_rows * n_cols
    mask = np.zeros(total, dtype=bool)

    for idx in range(total):
        row, col = divmod(idx, n_cols)
        pixel = get_pixel_composition(counts, row, col, n_cols, k_factors)
        matches = suggest_phases(pixel.atomic_pct, {phase_name: phase_library[phase_name]})
        if matches and matches[0].score >= min_score:
            mask[idx] = True

    return mask.reshape(n_rows, n_cols)


@dataclass
class PhaseRegionComposition:
    """Average EDS composition for a phase region in a consensus map."""
    phase_idx: int
    phase_name: str
    n_pixels: int
    avg_at_pct: Dict[str, float]
    avg_wt_pct: Dict[str, float]
    phase_matches: List[PhaseMatch]


def compute_phase_region_composition(
    counts: Dict[str, np.ndarray],
    phase_map: np.ndarray,
    phase_names: List[str],
    n_cols: int,
    k_factors: Optional[Dict[str, float]] = None,
    phase_library: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[PhaseRegionComposition]:
    """Compute average EDS composition for each phase region in a consensus map.

    Parameters
    ----------
    counts : dict
        Element symbol -> 1D counts array (flattened).
    phase_map : np.ndarray
        2D array (n_rows, n_cols) with phase indices (0, 1, ...) or -1.
    phase_names : list of str
        Phase display names, indexed by phase_map values.
    n_cols : int
        Number of columns in the scan grid.
    k_factors : dict, optional
        Custom k-factors (C ~ k * I); defaults to :func:`k_factor`.
    phase_library : dict, optional
        Phase definitions for suggest_phases(); defaults to DEFAULT_PHASE_LIBRARY.

    Returns
    -------
    list of PhaseRegionComposition
        One entry per phase (excluding unindexed pixels with -1).
    """
    if phase_library is None:
        phase_library = DEFAULT_PHASE_LIBRARY

    # Convert all counts to Wt.% and At.%
    _, wt_pct_arrays, at_pct_arrays = counts_to_all(counts, k_factors)

    flat_map = phase_map.ravel()
    results = []

    for phase_idx in range(len(phase_names)):
        pixels = np.where(flat_map == phase_idx)[0]
        if pixels.size == 0:
            results.append(PhaseRegionComposition(
                phase_idx=phase_idx,
                phase_name=phase_names[phase_idx],
                n_pixels=0,
                avg_at_pct={},
                avg_wt_pct={},
                phase_matches=[],
            ))
            continue

        # Average At.% and Wt.% over the region
        avg_at: Dict[str, float] = {}
        avg_wt: Dict[str, float] = {}
        for elem, arr in at_pct_arrays.items():
            valid = pixels[pixels < len(arr)]
            avg_at[elem] = float(np.mean(arr[valid])) if valid.size > 0 else 0.0
        for elem, arr in wt_pct_arrays.items():
            valid = pixels[pixels < len(arr)]
            avg_wt[elem] = float(np.mean(arr[valid])) if valid.size > 0 else 0.0

        # Phase suggestion based on average At.%
        matches = suggest_phases(avg_at, phase_library)

        results.append(PhaseRegionComposition(
            phase_idx=phase_idx,
            phase_name=phase_names[phase_idx],
            n_pixels=int(pixels.size),
            avg_at_pct=avg_at,
            avg_wt_pct=avg_wt,
            phase_matches=matches,
        ))

    return results
