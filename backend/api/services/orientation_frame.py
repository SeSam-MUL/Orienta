# backend/api/services/orientation_frame.py
"""Vendor-general EBSD reference-frame correction.

Brings orientations from any supported vendor into one canonical frame
(CS1 / raw vendor Euler — the MTEX default import; see
docs/superpowers/specs/2026-05-19-orientation-reference-frame-fix-design.md).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d

logger = logging.getLogger(__name__)

CANONICAL_FRAME_DESC = (
    "CS1 / raw vendor Euler (MTEX default import); Bunge ZXZ; lab2crystal; radians"
)


@dataclass(frozen=True)
class ReferenceFrameTransform:
    """Rotation that maps a vendor's stored orientations into the canonical frame.

    `orientation_rotation` is applied to the stored orientations.
    """
    orientation_rotation: Rotation
    description: str


@dataclass
class FrameOffsetResult:
    """Outcome of measuring the residual frame offset between two orientation sets."""
    best_candidate: str
    best_rotation: Rotation
    median_disorientation_deg: float
    fraction_below_5deg: float
    is_peaked: bool
    mirror_suspected: bool


def _candidate_rotations() -> dict[str, Rotation]:
    """Right-multiply candidates: identity + +/-90/180 about each axis.

    Built via axis-angle (unambiguous for X, Y and Z alike).
    """
    from orix.vector import Vector3d
    c: dict[str, Rotation] = {"identity": Rotation.identity()}
    axes = {"X": Vector3d.xvector(), "Y": Vector3d.yvector(), "Z": Vector3d.zvector()}
    for name, v in axes.items():
        for deg in (90.0, -90.0, 180.0):
            label = f"R{name.lower()}{int(deg):+d}".replace("+", "")
            c[label] = Rotation.from_axes_angles(v, np.deg2rad(deg))
    return c


def measure_frame_offset(our_rot: Rotation, ref_rot: Rotation, symmetry) -> FrameOffsetResult:
    """Grid-search the rotation that best aligns `our_rot` to `ref_rot`.

    Disorientation is symmetry-reduced via orix `Orientation.angle_with`.
    NaN orientations (unindexed pixels) are dropped before comparison.
    """
    our = Orientation(our_rot, symmetry)
    ref = Orientation(ref_rot, symmetry)
    valid = ~(np.isnan(our.data).any(axis=-1) | np.isnan(ref.data).any(axis=-1))
    if valid.sum() < 10:
        raise ValueError(
            f"measure_frame_offset: only {int(valid.sum())} valid pixels — "
            "cannot measure a frame offset"
        )

    best = None
    for label, T in _candidate_rotations().items():
        # MTEX corrections are left-multiply; the legacy -90 deg code is
        # right-multiply. Test both sides and let the data pick.
        for side in ("right", "left"):
            cand = (our_rot[valid] * T) if side == "right" else (T * our_rot[valid])
            corrected = Orientation(cand, symmetry)
            angles_deg = np.rad2deg(corrected.angle_with(ref[valid]))
            median = float(np.median(angles_deg))
            if best is None or median < best[1]:
                std = float(np.std(angles_deg))
                frac5 = float(np.mean(angles_deg < 5.0))
                best = (f"{label} ({side})", median, T, std, frac5)

    label, median, T, std, frac5 = best
    return FrameOffsetResult(
        best_candidate=label,
        best_rotation=T,
        median_disorientation_deg=median,
        fraction_below_5deg=frac5,
        is_peaked=(std < 10.0 and median < 15.0),
        mirror_suspected=(median > 15.0),
    )


def oxford_frame_adapter(header_meta: dict) -> ReferenceFrameTransform:
    """Oxford H5OINA -> canonical CS1 frame.

    CS1 canonical = raw Aztec Euler, matching MTEX's default h5oina import.
    MTEX applies Specimen Orientation Euler only on the opt-in 'CS0' flag and
    never reads Scanning Rotation Angle (see Task 1 research notes), so the
    adapter applies nothing. header_meta is accepted but unused under CS1.
    """
    return ReferenceFrameTransform(
        orientation_rotation=Rotation.identity(),
        description="oxford CS1 (raw Aztec Euler — identity)",
    )


def edax_frame_adapter(header_meta: dict) -> ReferenceFrameTransform:
    """EDAX OIM h5 -> canonical CS1 frame.

    CS1 canonical = raw OIM Phi, matching MTEX's loadEBSD_h5 (which applies no
    coordinate-system correction). coordinate_system_id is carried in
    header_meta for diagnostics and a possible future CS0 mode, but not applied.
    """
    return ReferenceFrameTransform(
        orientation_rotation=Rotation.identity(),
        description="edax CS1 (raw OIM Phi — identity)",
    )


def bruker_frame_adapter(header_meta: dict) -> ReferenceFrameTransform:
    """Bruker h5ebsd -> canonical CS1. UNVERIFIED (no local file)."""
    return ReferenceFrameTransform(
        orientation_rotation=Rotation.identity(),
        description="bruker CS1 (UNVERIFIED — identity, fail-loud guarded)",
    )


@dataclass(frozen=True)
class VendorAdapter:
    fn: Callable[[dict], ReferenceFrameTransform]
    verified: bool


VENDOR_FRAME_ADAPTERS: dict[str, VendorAdapter] = {
    "oxford": VendorAdapter(oxford_frame_adapter, verified=True),
    "edax": VendorAdapter(edax_frame_adapter, verified=True),
    "bruker": VendorAdapter(bruker_frame_adapter, verified=False),
}


# Intrinsic offset of each indexing source relative to the canonical (CS1 / raw
# vendor Euler) frame, applied as rotations_canonical = rotations * intrinsic *
# vendor_transform inside apply_reference_frame_correction.
#
# "vendor"        identity by definition (vendor solution IS the canonical frame).
# "spherical_gpu" identity — measured 2026-05-19 in
#                 tasks/orientation_synthetic_report.md (synthetic round-trip,
#                 cubic Oh, median residual 1.17° at L=88 = indexer quantization
#                 noise; backend natively produces canonical orientations).
# "emsphinx"      identity — empirically verified 2026-05-21 (cross-method
#                 audit on NiLowGain Test_data/LoGainNi.h5). With
#                 Manufacturer='EDAX' (vendorFlip=true) on kikuchipy-loaded
#                 patterns, EMSphInx output is natively in the canonical
#                 frame:
#                   * with R(+90 deg, Z) right-mult: 41.06 deg vs Hough  WRONG
#                   * with identity:                  5.44 deg vs Hough  OK
#                 An A/B test (verify_emsphinx_vendor_diff.py) confirmed:
#                   * Manufacturer='EDAX'  (vendorFlip=true):   5.44 deg vs Hough  OK
#                   * Manufacturer='Bruker'(vendorFlip=false): 40.25 deg vs Hough  WRONG
#                 The historical R(+90 deg, Z) ("Phase 5e: ~37 deg -> ~5 deg
#                 vs Aztec") was a lucky empirical fit on an earlier code
#                 path. With kikuchipy as the uniform loader and
#                 Manufacturer='EDAX' applied unconditionally, no rotation
#                 correction is needed. Residual ~5 deg vs Hough/Dictionary
#                 is shared by BOTH spherical paths (CPU 5.44, GPU 5.69) —
#                 a separate frame residual, not addressed here.
SOURCE_INTRINSIC_OFFSET: dict[str, Rotation] = {
    "vendor": Rotation.identity(),
    "spherical_gpu": Rotation.identity(),
    "emsphinx": Rotation.identity(),
}


def get_source_intrinsic_offset(source: str, vendor: str) -> Rotation:
    """Intrinsic offset for a given indexing source.

    `vendor` is currently unused (all sources are identity), but is kept in
    the signature for future vendor-specific corrections.
    """
    if source in SOURCE_INTRINSIC_OFFSET:
        return SOURCE_INTRINSIC_OFFSET[source]
    raise ValueError(f"unknown source {source!r}")


def apply_reference_frame_correction(
    rotations: Rotation,
    vendor: str,
    header_meta: dict,
    source: str,
    strict: bool = False,
) -> Rotation:
    """Bring `rotations` from (vendor, source) into the canonical frame.

    rotations_canonical = rotations * source_intrinsic * vendor_transform

    The `source_intrinsic` is vendor-aware for `source='emsphinx'`; see
    `get_source_intrinsic_offset` for the rationale.
    """
    vendor = (vendor or "").lower()
    if vendor not in VENDOR_FRAME_ADAPTERS:
        raise ValueError(f"unknown vendor {vendor!r}; expected one of {list(VENDOR_FRAME_ADAPTERS)}")
    # Will raise ValueError for unknown source.
    intrinsic = get_source_intrinsic_offset(source, vendor)

    adapter = VENDOR_FRAME_ADAPTERS[vendor]
    if not adapter.verified:
        msg = (f"reference-frame adapter for vendor {vendor!r} is not verified — "
               "orientation map may be rotated/mirrored")
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    vendor_t = adapter.fn(header_meta).orientation_rotation
    return rotations * intrinsic * vendor_t


# ---------------------------------------------------------------------------
# Export-boundary frame conversion (native indexer frame <-> vendor stored Euler)
# ---------------------------------------------------------------------------
# Our indexers (spherical_gpu, EMSphInx, Hough via PyEBSDIndex/kikuchipy,
# dictionary via kikuchipy) all emit orientations in the EMsoft/EDAX/kikuchipy
# COMMON sample frame. Oxford Aztec (and Bruker) store Euler in a frame rotated
# -90deg about the sample normal (ND/Z) from that common frame — see the
# kikuchipy "Reference frames" tutorial
# (https://kikuchipy.org/en/stable/tutorials/reference_frames.html), whose
# R_sample table lists OXFORD = BRUKER = Rz(-90deg) and EMSOFT = EDAX =
# KIKUCHIPY = identity. MTEX's default loadEBSD_h5oina applies NO rotation, so
# "match Aztec in MTEX" == "write the Oxford-frame Euler".
#
# Empirically pinned 2026-06-05 on Oxford SampleB (tasks/repro_90z_oxford.py):
# our_rotation * Rz(+90deg) (orix right-multiply) drives the symmetry-reduced
# median disorientation vs Aztec from 36deg down to 5deg (the residual being
# the indexer/PC quantization noise) — every other candidate stays at 36deg.
#
# We apply this ONLY at the file-export boundary; the in-memory xmap stays in
# the native frame so the pattern renderer (forward diagnostics, refinement,
# pattern-match preview) keeps matching the measured patterns. The h5 reader
# inverts it on re-import so re-imported working maps return to the native
# frame.
_RZ90 = Rotation.from_axes_angles(Vector3d.zvector(), np.deg2rad(90.0))

# Vendors whose stored Euler frame is the -90deg-about-ND family.
_VENDOR_NEEDS_RZ90 = {"oxford", "bruker"}
# Vendors already in our native (common) frame.
_VENDOR_NATIVE = {"edax", "tsl", "kikuchipy", "emsoft", "ametek"}


def export_frame_offset(vendor: str) -> Rotation:
    """Right-multiply rotation mapping native -> ``vendor`` stored Euler frame.

    Identity for vendors already in the native frame; ``Rz(+90deg)`` for the
    Oxford/Bruker family. Unknown vendors return identity (no silent mis-
    rotation) with a warning.
    """
    v = (vendor or "").lower().strip()
    if v in _VENDOR_NEEDS_RZ90:
        if v == "bruker":
            logger.warning(
                "export_frame_offset: Bruker Rz(+90deg) is from the kikuchipy "
                "reference-frames table but UNVERIFIED on real data here"
            )
        return _RZ90
    if v in _VENDOR_NATIVE:
        return Rotation.identity()
    logger.warning(
        "export_frame_offset: unknown vendor %r -> leaving orientations in the "
        "native frame (no in-plane correction)", vendor,
    )
    return Rotation.identity()


def to_vendor_export_frame(
    rotations: Rotation, vendor: str, r_user: Rotation | None = None
) -> Rotation:
    """Native (EMsoft/kikuchipy) -> vendor stored Euler frame, for file export.

    So that a tool reading the raw exported Euler (e.g. MTEX default
    loadEBSD_h5oina, or h5read on our light .h5) sees orientations aligned with
    the vendor's own solution.

    When ``r_user`` is given (the per-file coordinate-system rotation, opt-in),
    it is composed to the LEFT of the vendor offset:
        rotations * r_user * export_frame_offset(vendor)
    ``r_user=None`` (the default) is byte-identical to the legacy 2-arg call.
    """
    base = rotations if r_user is None else rotations * r_user
    return base * export_frame_offset(vendor)


def from_vendor_export_frame(rotations: Rotation, vendor: str) -> Rotation:
    """Vendor stored Euler frame -> native, for re-import of our own exports.

    Inverse of :func:`to_vendor_export_frame`, so a re-loaded export returns to
    the native working frame the renderer expects.
    """
    return rotations * (~export_frame_offset(vendor))
