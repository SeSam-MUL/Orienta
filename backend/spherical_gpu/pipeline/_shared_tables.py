"""perf Phase 1.4 — SharedSphericalTables.

L+geom+sample_tilt-dependent precomputations that can be shared across
all phases of a multi-phase indexing run at the same bandwidth and
detector geometry. Built ONCE by ``SphericalGPUBackend._ensure_built``
when ``len(phases) > 1``; subsequent ``Tier1Indexer`` instances receive
it via the ``shared_tables=`` constructor param and skip the redundant
``_build_*`` calls (saves ~7.5 s per skipped phase at L=88).

See the perf notes for the full categorization
of which Tier1Indexer attributes are L+geom-only (shareable) vs
master-dependent (per-phase).

Iter-2 status: dataclass scaffold only. Tier1Indexer wiring +
SphericalGPUBackend integration land in iter-3 along with the
validation harness run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .detector import DetectorGeometry


@dataclass
class SharedSphericalTables:
    """Cached precomputations that depend only on (L, geom, sample_tilt).

    Attributes mirror the corresponding ``self._*`` names in
    ``Tier1Indexer.__init__`` so binding inside a shared-tables-aware
    Tier1Indexer is a trivial ``self._x = shared.x`` loop.

    All tensor fields are READ-ONLY in the indexer hot path. They are
    NEVER mutated after construction; if any per-phase code path tries
    to mutate one, it would corrupt other phases sharing the same
    instance. The oracle test catches such regressions because the
    second/third phase's output would diverge from baseline.
    """
    # Identity
    bandwidth: int
    device: torch.device
    sample_tilt_deg: float

    # L-only
    rsht: object                                 # RSHT instance
    wigner_table: torch.Tensor                   # wigner_d_eq_half_pi(L+2)
    d_lmk_full: torch.Tensor                     # (L, 2L-1, 2L-1) f32
    sign_factors_L: torch.Tensor                 # (L,) f32

    # (L, geom)
    dp_inside_flat: torch.Tensor                 # (N_inside,) i64
    dp_alp: torch.Tensor                         # (L, L, N_inside) f32
    dp_phase: torch.Tensor                       # (N_inside,) c64
    dp_omega: torch.Tensor                       # (N_inside,) f32
    dp_grid_xy: torch.Tensor                     # (N_inside, 2) f32

    # GL ring tables (lists are CPU-side; *_dev variants are on device)
    gl_ring_cell_flat: List[torch.Tensor]
    gl_ring_alp: List[torch.Tensor]
    gl_ring_wy: torch.Tensor
    gl_dim: int
    gl_ring_cell_flat_dev: List[torch.Tensor]
    gl_ring_wy_dev: torch.Tensor

    # South hemisphere GL ring map
    gl_south_ring_gxy: List[torch.Tensor]
    gl_south_ring_inside: List[torch.Tensor]
    n_south_inside_per_ring: List[int]
    south_gxy_inside_dev: List[torch.Tensor]
    south_inside_idx_dev: List[torch.Tensor]
    dp_south_grid_xy: torch.Tensor
    dp_south_omega: torch.Tensor

    # Direct-quadrature SHT basis (used by _direct_sht_coefs path)
    dp_sht_basis: torch.Tensor                   # (L*L, N_inside) c64

    # Scalar derived from dp_omega
    rescale_factor: float

    # Tilt-derived geometry
    alpha_tilt: float

    # Backprojection map (DH path)
    bp_grid_xy: torch.Tensor
    bp_north_mask: torch.Tensor
    bp_north_flat: torch.Tensor
    bp_south_flat: torch.Tensor
    bp_antipodal_flat: torch.Tensor
    bp_dh_weights: torch.Tensor
    bp_complete_ring_mask: torch.Tensor

    @classmethod
    def from_indexer(cls, indexer) -> "SharedSphericalTables":
        """Extract shareable tables from an already-constructed Tier1Indexer.

        Used by ``SphericalGPUBackend._ensure_built`` to harvest the first
        phase's tables and pass them to subsequent phases. Avoids
        duplicating the build code: we let the first ``Tier1Indexer``
        construct everything normally, then snapshot its (L, geom)-only
        state into this dataclass for the others to consume.

        The returned dataclass holds REFERENCES to the indexer's tensors
        (no copy). Indexer-1 and the shared dataclass therefore back the
        same memory. Since the tables are read-only in the hot path
        (assert at bind time), aliasing is safe and saves 21 MB/phase
        on rDen alone if a future variant ever shares it (rDen is
        master-dependent, so NOT in this dataclass).
        """
        return cls(
            bandwidth=int(indexer.bandwidth),
            device=indexer.device,
            sample_tilt_deg=float(indexer.sample_tilt_deg),
            # L-only
            rsht=indexer._rsht,
            wigner_table=indexer._wigner_table,
            d_lmk_full=indexer._d_lmk_full,
            sign_factors_L=indexer._sign_factors_L,
            # (L, geom)
            dp_inside_flat=indexer._dp_inside_flat,
            dp_alp=indexer._dp_alp,
            dp_phase=indexer._dp_phase,
            dp_omega=indexer._dp_omega,
            dp_grid_xy=indexer._dp_grid_xy,
            # GL ring tables
            gl_ring_cell_flat=indexer._gl_ring_cell_flat,
            gl_ring_alp=indexer._gl_ring_alp,
            gl_ring_wy=indexer._gl_ring_wy,
            gl_dim=int(indexer._gl_dim),
            gl_ring_cell_flat_dev=indexer._gl_ring_cell_flat_dev,
            gl_ring_wy_dev=indexer._gl_ring_wy_dev,
            # South hemisphere
            gl_south_ring_gxy=indexer._gl_south_ring_gxy,
            gl_south_ring_inside=indexer._gl_south_ring_inside,
            n_south_inside_per_ring=indexer._n_south_inside_per_ring,
            south_gxy_inside_dev=indexer._south_gxy_inside_dev,
            south_inside_idx_dev=indexer._south_inside_idx_dev,
            dp_south_grid_xy=indexer._dp_south_grid_xy,
            dp_south_omega=indexer._dp_south_omega,
            # SHT basis
            dp_sht_basis=indexer._dp_sht_basis,
            # Scalars
            rescale_factor=float(indexer._rescale_factor),
            alpha_tilt=float(indexer._alpha_tilt),
            # Backprojection
            bp_grid_xy=indexer._bp_grid_xy,
            bp_north_mask=indexer._bp_north_mask,
            bp_north_flat=indexer._bp_north_flat,
            bp_south_flat=indexer._bp_south_flat,
            bp_antipodal_flat=indexer._bp_antipodal_flat,
            bp_dh_weights=indexer._bp_dh_weights,
            bp_complete_ring_mask=indexer._bp_complete_ring_mask,
        )

    def assert_compatible(self, *, bandwidth: int, device: torch.device,
                          sample_tilt_deg: float, geom_signature: tuple) -> None:
        """Cheap guard before binding into a new Tier1Indexer.

        Raises ValueError fail-loud (per memory/feedback_fail_loud_not_silent)
        if any identity field mismatches the consumer's parameters. Caller
        is responsible for computing ``geom_signature`` (tuple of geom attrs
        that affect the precomputed tables — pat_h, pat_w, cx, cy, L_scint,
        pixel_size, tilt, vendor — see DetectorGeometry).
        """
        if int(bandwidth) != self.bandwidth:
            raise ValueError(
                f"shared-tables bandwidth {self.bandwidth} != requested {bandwidth}"
            )
        if device != self.device:
            raise ValueError(
                f"shared-tables device {self.device} != requested {device}"
            )
        if abs(float(sample_tilt_deg) - self.sample_tilt_deg) > 1e-6:
            raise ValueError(
                f"shared-tables sample_tilt {self.sample_tilt_deg} != "
                f"requested {sample_tilt_deg}"
            )
        # geom_signature equality is enforced by the caller — we trust the
        # backend to only pass shared_tables when geometry is identical.
