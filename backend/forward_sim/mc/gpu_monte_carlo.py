"""SP2 — GPU-native PyTorch Monte Carlo producing an EMsoft-compatible MCData.

Algorithm:  single-scattering screened-Rutherford + Bethe-CSDA energy loss
(Joy 1995, ch.3), ported from ebsdtorch ``simulation/monte_carlo.py::bse_sim_mc``
(Z. Varley, MIT, fetched 2026-06-11).  The MC drives N_e electrons IN PARALLEL
as (N_e,) PyTorch tensors, resetting each slot the instant the electron exits
(z < 0) or stalls (step counter > n_max_steps).  The loop runs until
``sims_finished >= n_simulations_max``.

Initial un-scattered beam step (EMsoft parity):
    EMsoft takes ONE un-scattered beam step into the sample BEFORE any scattering
    event.  In EMMC.cl L224-229 the first step is computed before the while-loop:
    step = -mfp*log(rand); r_new = r0 + step*c0; E_new += step*rho*de_ds.  The
    electron advances along the un-deflected beam direction c0 by depth
    step*cos(sig) before the first scatter angle is sampled inside the loop
    (L272).  The CPU path mirrors this exactly (EMMC.f90 L490-503).

    Our ``_take_initial_beam_step`` reproduces this: for each affected slot it
    draws a step, advances z along the un-deflected d0, and applies energy loss,
    WITHOUT calling ``_update_directions``.  It is applied to every slot at
    startup and to every reset slot before the next scatter iteration.

Exit convention:
    +z points INTO the sample; the surface is at z = 0.  An electron has
    backscattered when its updated depth z_new < 0.  The sample tilt enters
    ONLY as the initial beam direction (70° by default); the tilt is NOT applied
    as a post-hoc rotation of the exit vector — the trajectory naturally
    produces the correct tilted-geometry exit directions.

Accumulator layout (matches EMsoft oracle):
    ``accum_e`` — (n_dir, n_dir, nE)   Lambert-direction × energy back-scatter counts
    ``accum_z`` — (n_dir_z, n_dir_z, nz, nE)  depth × energy (two coarser direction
                  axes to match EMsoft's 51 × 51 accum_z vs 501 × 501 accum_e).

Direction convention (EMsoft north-hemisphere):
    BSE electrons exit upward through z=0 with cz < 0 (south hemisphere).
    EMsoft stores the *negated* direction (-cx, -cy, -cz) for accum_e/accum_z
    so that the master-pattern lookup works with the standard north-hemisphere
    Lambert map (cz > 0 hemisphere).

    Note: :func:`rosca_lambert` uses ``abs(z)`` internally, so the Lambert
    projection value for direction d is identical to that of -d.  Negating
    the exit direction therefore does NOT change the Lambert xy-coordinates.
    What negation DOES achieve is a point-reflection of the in-plane (x, y)
    components — this flips the directional histogram so that the peak
    appears at the correct Lambert grid location (matching EMsoft's accum_e
    storage convention).  This fix affects ONLY accum_e and accum_z direction
    bins; it has NO effect on the master pattern or the .sht file, both of
    which are derived from lambda_z (a scalar summed over all directions).

Termination (EMsoft-faithful, iter7):
    An electron slot is recycled when (a) z_new < 0 (backscattered) or (b)
    ticks > n_max_steps (the scatter-COUNT bound; default 300 = EMsoft GPU
    `steps`, EMMCOpenCL.f90).  EMsoft does NOT kill electrons at depth_max —
    `depth_max` only bounds the accum_z depth-binning extent (deep exits still
    count in accum_e / eta).  We therefore do NOT transmit-kill at z >= depth_max
    (doing so discarded the deep electrons that random-walk back and backscatter,
    underestimating eta ~21%: 0.447 vs EMsoft 0.569; removing it gives 0.579).
    E < Ehistmin does NOT terminate an electron either — it only prevents binning
    of below-threshold exits.

Energy binning:
    We bin the POST-STEP exit energy E_new against (Ehistmin, Ebinsize) so our
    energy axis is 1:1 with EMsoft's EkeVs = [Ehistmin, Ehistmin+1, ..., Emax].
    ie = nint((E_exit - Ehistmin) / Ebinsize), clipped to [0, nE).

    EMsoft uses nint (round-half-away-from-zero, Fortran convention) and the
    energy AFTER the escaping step (Ec updated at loop top before exit test,
    EMMC.f90 L505-507/555; EMMCOpenCL.f90 L577).  Using floor or pre-step
    energy starves the top bin of the near-elastic 19.5-20 keV population,
    producing the characteristic bin-0 pile-up seen in iter3/iter4.

Lambert projection:
    Uses :func:`backend.spherical_gpu._math.lambert.rosca_lambert` (bit-identical
    to ebsdtorch's embedded copy, verified against EMsoft Lambert.f90 sPio2
    scaling).  Direction binning: ``idx = floor((xy * 0.499999 + 0.5) * n_dir)``,
    0.499999 guards the upper edge.

References:
    Joy, D.C. 1995, *Monte Carlo Modeling for Electron Microscopy and
    Microanalysis*, Oxford University Press, ch.3.

    ebsdtorch (Z. Varley, MIT):
    https://github.com/ZacharyVarley/ebsdtorch (simulation/monte_carlo.py)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from backend.forward_sim.mc.emsoft_mc_input import MCData
from backend.forward_sim.mc.scattering_mc import (
    ElasticModel,
    mean_ionisation_potential_kev,
)
from backend.spherical_gpu._math.lambert import rosca_lambert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MC run configuration
# ---------------------------------------------------------------------------

@dataclass
class MCConfig:
    """Configuration for a single GPU MC run.

    Attributes:
        starting_E_keV: beam (accelerating) voltage in keV.
        Ehistmin: lowest energy bin centre in kV (keV).
        Ebinsize: energy bin width in kV (keV).
        nE: number of energy bins; EkeVs = arange(Ehistmin, Ehistmin+nE*Ebinsize, Ebinsize).
        depth_step: depth bin width in nm.
        depth_max: maximum tracked depth in nm.
        sig_deg: sample tilt in degrees (EBSD standard = 70 deg).
        omega_deg: secondary tilt in degrees (usually 0).
        n_dir: Lambert grid side for accum_e (EMsoft uses 501).
        n_dir_z: Lambert grid side for accum_z coarser grid (EMsoft uses 51).
        n_simulations: total electrons to simulate (controls statistical quality).
        n_electrons_parallel: batch size (number of simultaneous trajectory slots).
        n_max_steps: max steps before an electron slot is recycled (stall guard).
        seed: random seed for reproducibility; None = non-deterministic.
        device: torch device string (``'cpu'``, ``'cuda'``, ``'cuda:0'``, …).
        elastic_model: :class:`~backend.forward_sim.mc.scattering_mc.ElasticModel`
            controlling the elastic cross-section and angular-sampling model.
            Default is ``ElasticModel.JOY`` (screened-Rutherford, EMsoft-faithful).
            ``ElasticModel.BROWNING`` uses the Browning 1994 Mott-fitted analytic
            model (MFP(Ni,20kV) ≈ 3.36 nm, shorter by 1.54× vs Joy).
        engine: Monte-Carlo execution model (LEVER 1 cupy + LEVER 3 numba).
            ``"auto"`` (default) is a fail-safe chain: the one-thread-per-electron
            CuPy RawKernel (:func:`run_gpu_mc_cupy`) when the device is CUDA, cupy
            is importable and the model is Joy → else the numba CPU
            one-electron-per-thread njit loop
            (:func:`backend.forward_sim.mc.numba_mc.run_numba_mc`) when numba
            compiles (the no-GPU fast path) → else the PyTorch step-major loop.
            ``"cupy"`` forces the CUDA kernel (raises if unavailable).
            ``"numba"`` forces the numba CPU loop (raises if numba unavailable).
            ``"pytorch"`` (or any other value) forces the PyTorch loop.  Both
            one-per-electron engines are master-NCC-equivalent to the loop
            (≥ 0.998 on Ni/Al); only Joy is ported (Browning → PyTorch loop).
        cupy_n_el_per_thread: electrons each CUDA thread simulates sequentially
            (mirrors EMsoft's ``num_el``); only used by the cupy engine.
        cupy_chunk_electrons: max electrons per kernel launch (bounds the
            per-launch exit-array memory); only used by the cupy engine.
    """

    starting_E_keV: float = 20.0
    Ehistmin: float = 10.0
    Ebinsize: float = 1.0
    nE: int = 11
    depth_step: float = 1.0
    depth_max: float = 100.0
    sig_deg: float = 70.0
    omega_deg: float = 0.0
    n_dir: int = 501
    n_dir_z: int = 51
    n_simulations: int = 10_000_000
    n_electrons_parallel: int = 100_000
    n_max_steps: int = 300  # EMsoft GPU steps (EMMCOpenCL.f90:240); CPU oracle = 500
    seed: Optional[int] = None
    device: str = "cpu"
    elastic_model: ElasticModel = ElasticModel.JOY
    engine: str = "auto"  # "auto" | "cupy" | "numba" | "pytorch" — MC execution model
    # NB: "auto"/"cupy"/"numba" count each electron once (pool-independent eta ~0.57,
    # EMsoft-faithful); the legacy "pytorch" perpetual-pool loop's absolute eta is
    # pool-size dependent (~0.58-0.64). All engines agree on the NORMALISED depth/
    # energy marginals and the resulting master (NCC 0.9999) — the quality gate.
    cupy_n_el_per_thread: int = 32  # electrons run sequentially per CUDA thread (EMsoft num_el-style)
    cupy_chunk_electrons: int = 8_000_000  # max electrons per kernel launch (bounds exit-array memory)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_direction(sig_deg: float, omega_deg: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the initial beam direction as an (n, 3) tensor.

    The incident beam is tilted by ``sig`` about the omega axis; with omega=0
    the direction is (sin_sig, 0, cos_sig) in the (x, y, z)-sample frame
    where +z points into the sample.

    Args:
        sig_deg: sample tilt in degrees.
        omega_deg: secondary tilt (azimuth about z) in degrees.
        n: batch size.
        device: torch device.
        dtype: floating-point dtype.

    Returns:
        (n, 3) tensor of unit direction vectors, all identical.
    """
    sig = math.radians(sig_deg)
    om = math.radians(omega_deg)
    d0_x = math.sin(sig) * math.cos(om)
    d0_y = math.sin(sig) * math.sin(om)
    d0_z = math.cos(sig)
    d0 = torch.tensor([[d0_x, d0_y, d0_z]], dtype=dtype, device=device)
    return d0.expand(n, 3).clone()


def _update_directions(
    cx: torch.Tensor,
    cy: torch.Tensor,
    cz: torch.Tensor,
    phi: torch.Tensor,
    psi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Joy 1995 direction-cosine update (eqs 3.12a-c) in-place.

    Handles the |cz| > 0.99999 pole case with a vectorised mask.

    Args:
        cx, cy, cz: direction cosines of shape (N,).
        phi: polar scattering angle (N,) in radians.
        psi: azimuth angle (N,) in radians.

    Returns:
        Updated (cx, cy, cz) unit direction cosines, renormalized.
    """
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)

    pole = torch.abs(cz) > 0.99999

    # Pole branch (numerically safe with the sign of cz).
    ca_pole = sin_phi * cos_psi
    cb_pole = sin_phi * sin_psi
    cc_pole = torch.sign(cz) * cos_phi

    # General branch (Joy 1995 eqs 3.12a-c).
    dsq = torch.sqrt(torch.clamp(1.0 - cz * cz, min=0.0))
    dsqi = 1.0 / (dsq + 1e-30)  # avoid div-by-zero (masked out anyway)

    ca_gen = sin_phi * (cx * cz * cos_psi - cy * sin_psi) * dsqi + cx * cos_phi
    cb_gen = sin_phi * (cy * cz * cos_psi + cx * sin_psi) * dsqi + cy * cos_phi
    cc_gen = -sin_phi * cos_psi * dsq + cz * cos_phi

    ca = torch.where(pole, ca_pole, ca_gen)
    cb = torch.where(pole, cb_pole, cb_gen)
    cc = torch.where(pole, cc_pole, cc_gen)

    # Renormalize (numerical drift accumulates over many steps).
    norm = torch.sqrt(torch.clamp(ca * ca + cb * cb + cc * cc, min=1e-30))
    return ca / norm, cb / norm, cc / norm


def _lambert_dir_indices(
    cx: torch.Tensor,
    cy: torch.Tensor,
    cz: torch.Tensor,
    n_dir: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project exit directions to Lambert grid indices (i_x, i_y) in [0, n_dir).

    Uses :func:`rosca_lambert` which maps unit-sphere directions to
    (-1, 1) × (-1, 1).  Index formula:

        idx = floor((xy * 0.499999 + 0.5) * n_dir)

    The 0.499999 (not 0.5) guards the upper edge from rounding to n_dir.

    Args:
        cx, cy, cz: direction cosines (N,).
        n_dir: Lambert grid size.

    Returns:
        (ix, iy) long tensors of shape (N,), values in [0, n_dir).
        Out-of-range (should be rare) is clamped to [0, n_dir-1].
    """
    pts = torch.stack([cx, cy, cz], dim=-1)  # (N, 3)
    xy = rosca_lambert(pts)                   # (N, 2)
    ix = torch.floor((xy[:, 0] * 0.499999 + 0.5) * n_dir).long()
    iy = torch.floor((xy[:, 1] * 0.499999 + 0.5) * n_dir).long()
    ix = ix.clamp(0, n_dir - 1)
    iy = iy.clamp(0, n_dir - 1)
    return ix, iy


def _compute_reset_mask(
    exited: torch.Tensor,
    z_new: torch.Tensor,
    ticks: torch.Tensor,
    depth_max: float,
    n_max_steps: int,
    E_new: Optional[torch.Tensor] = None,
    Ehistmin: float = 0.0,
) -> torch.Tensor:
    """Return the per-slot reset mask (ETA-FIX behaviour contract).

    A slot is reset if and only if:
    * ``exited`` — z_new < 0 (backscattered through surface), OR
    * stalled — ticks > n_max_steps (scatter-count cap, EMsoft-parity 300).

    Explicitly: reaching z >= depth_max does NOT reset a slot.  EMsoft
    never kills an electron on depth (EMMC.cl:238 ``while (counter1 < steps)``
    has no depth test; EMMC.f90:503 ``do while (traj.lt.num)`` likewise).
    depth_max is BINNING-ONLY (sizes accum_z; does not gate accum_e or eta).
    Deep electrons random-walk until they either backscatter or hit the
    scatter-count cap.

    Explicitly: E < Ehistmin does NOT reset a slot.  Ehistmin is the
    lowest BINNING threshold; electrons below it continue scattering.
    ``E_new`` and ``Ehistmin`` are accepted (and passed from
    :func:`run_gpu_mc`) so that the B2 unit-test can exercise the full
    call signature and verify that energy-based termination is absent
    from the production path.

    Args:
        exited: (N,) bool — True where z_new < 0.
        z_new: (N,) float — updated depth after the current step.
        ticks: (N,) int32 — step counts BEFORE incrementing this tick.
        depth_max: maximum tracked depth in nm (unused here; kept for
            interface stability and test-harness compatibility).
        n_max_steps: stall-guard threshold (EMsoft GPU default: 300).
        E_new: (N,) float — updated electron energies (passed for
            test-lethality; NOT used in the reset decision).
        Ehistmin: lowest energy bin threshold in keV (passed for
            test-lethality; NOT used in the reset decision).

    Returns:
        (N,) bool reset mask.
    """
    stalled = (ticks + 1) > n_max_steps  # +1 matches the ticks += 1 in run_gpu_mc
    return exited | stalled


def _bin_exits(
    accum_e: torch.Tensor,
    accum_z: torch.Tensor,
    exited: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    cz: torch.Tensor,
    escape_depth: torch.Tensor,
    energy: torch.Tensor,
    *,
    Ehistmin: float,
    Ebinsize: float,
    nE: int,
    depth_step: float,
    nz: int,
    n_dir: int,
    n_dir_z: int,
) -> None:
    """Scatter-add a batch of backscattered exits into ``accum_e`` / ``accum_z``.

    This is the SINGLE binning path shared by both the step-major PyTorch loop
    (:func:`run_gpu_mc`) and the one-thread-per-electron CuPy kernel
    (:func:`run_gpu_mc_cupy`).  Extracting it guarantees the cupy engine bins
    its kernel-emitted exit scalars with byte-for-byte the validated math —
    EMsoft-faithful ``nint`` energy/depth indices, the ``-cx,-cy,-cz`` Lambert
    negation, and the accum_e (no depth gate) / accum_z (depth-gated) split.

    All tensors are length-N (one slot per electron / trajectory).  Only slots
    with ``exited`` True contribute; non-exited slots are masked out.  Modifies
    ``accum_e`` and ``accum_z`` in place (scatter-add).

    Args:
        accum_e: (n_dir, n_dir, nE) accumulator (modified in place).
        accum_z: (n_dir_z, n_dir_z, nz, nE) accumulator (modified in place).
        exited: (N,) bool — True where the electron backscattered (z_new < 0).
        cx, cy, cz: (N,) exit direction cosines (the NEW post-scatter direction
            at the exit step; negation is applied inside).
        escape_depth: (N,) escape depth in nm (= |z_old / cz_new|, Joy/EMsoft
            convention); only read where ``exited``.
        energy: (N,) POST-step exit energy in keV.
        Ehistmin, Ebinsize, nE: energy binning (keV / keV / count).
        depth_step, nz: depth binning (nm / count).
        n_dir, n_dir_z: Lambert grid sizes for accum_e / accum_z.
    """
    if not bool(exited.any()):
        return
    device = accum_e.device

    # Energy bin — POST-step energy, EMsoft-faithful nint (see _energy_index).
    ie = _energy_index(energy, Ehistmin, Ebinsize)

    # Depth bin — nint(escape_depth/depth_step) = floor(x+0.5) (Fortran nint).
    iz = torch.floor(escape_depth / depth_step + 0.5).long()

    # B1-FIX: negate exit direction before Lambert projection (point-reflects the
    # in-plane components to EMsoft's north-hemisphere accum_e storage convention;
    # rosca_lambert uses abs(z) so the Lambert *value* is unchanged by negation).
    ix_e, iy_e = _lambert_dir_indices(-cx, -cy, -cz, n_dir)
    ix_z, iy_z = _lambert_dir_indices(-cx, -cy, -cz, n_dir_z)

    # ETA-FIX accumulator semantics: accum_e has NO depth gate (any valid-energy
    # exit counts); accum_z is depth-gated (iz in [0, nz)).
    valid_e = exited & (ie >= 0) & (ie < nE)
    valid_z = valid_e & (iz >= 0) & (iz < nz)

    if bool(valid_e.any()):
        v_ie_e = ie[valid_e]
        v_ix_e = ix_e[valid_e]
        v_iy_e = iy_e[valid_e]
        ones_e = torch.ones(v_ie_e.shape[0], dtype=torch.float32, device=device)
        accum_e.index_put_((v_ix_e, v_iy_e, v_ie_e), ones_e, accumulate=True)

    if bool(valid_z.any()):
        v_ie_z = ie[valid_z]
        v_iz = iz[valid_z]
        v_ix_z = ix_z[valid_z]
        v_iy_z = iy_z[valid_z]
        ones_z = torch.ones(v_ie_z.shape[0], dtype=torch.float32, device=device)
        accum_z.index_put_((v_ix_z, v_iy_z, v_iz, v_ie_z), ones_z, accumulate=True)


def _energy_index(E: torch.Tensor, Ehistmin: float, Ebinsize: float) -> torch.Tensor:
    """Map exit energy to energy bin index using EMsoft-faithful nint rounding.

    ie = nint((E - Ehistmin) / Ebinsize)
       = floor((E - Ehistmin) / Ebinsize + 0.5)   [round-half-away-from-zero]

    EMsoft (GPU host EMMCOpenCL.f90 L577; CPU EMMC.f90 L570) uses Fortran
    ``nint``, which rounds half-away-from-zero.  The critical difference vs
    ``floor`` is in the topmost bin: an electron with E=19.6 keV maps to
    floor((19.6-10)/1)=9 (wrong, 19-keV bin) but nint → 10 (correct, 20-keV
    bin).  Using floor starves the top bin of the near-elastic population and
    produces the characteristic bin-0 pile-up.

    The +0.5/floor formulation (rather than Python/torch round-half-to-even)
    is bit-faithful to Fortran nint on all non-half-integer inputs, and on
    half-integer inputs matches the Fortran convention (always rounds up).

    Args:
        E: (N,) exit energies in keV.  Caller is responsible for passing the
            POST-STEP energy (E_new after the escaping step) to match EMsoft's
            convention (EMMC.f90 L505-507: Ec updated before exit test).
        Ehistmin: lowest bin centre in keV.
        Ebinsize: bin width in keV.

    Returns:
        (N,) long tensor of bin indices (may be outside [0, nE); caller must mask).
    """
    return torch.floor((E - Ehistmin) / Ebinsize + 0.5).long()


def _take_initial_beam_step(
    E: torch.Tensor,
    z: torch.Tensor,
    cz0: float,
    mask: torch.Tensor,
    mean_Z: float,
    mean_A: float,
    rho: float,
    J_keV: float,
    use_browning: bool = False,
    _br_mfp_const: float = 0.0,
    _br_c1: float = 0.0,
    _br_c2: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one un-scattered beam step for newly initialised slots (EMsoft parity).

    Mirrors EMMC.cl L224-232 / EMMC.f90 L490-500: before the first scatter
    event each electron travels along the un-deflected beam direction c0 by a
    random step drawn from the elastic MFP distribution.  The direction
    cosines (cx, cy, cz) are left unchanged — only depth z and energy E are
    updated.  No exit test is performed (EMsoft does not test exit after the
    pre-loop step).

    This function operates ONLY on slots selected by ``mask`` (fresh electrons
    at z=0 with direction d0).  It modifies E and z in-place for those slots
    and returns the updated tensors.

    Args:
        E: (N,) float32 energy tensor.  Slots addressed by ``mask`` must have
            E == starting_E (the function reads E[mask] for the MFP calculation).
        z: (N,) float32 depth tensor.  Slots addressed by ``mask`` must have
            z == 0.0.
        cz0: the z-component of the un-deflected beam direction d0 (scalar),
            = cos(sig_deg).  Always positive (beam entering the sample).
        mask: (N,) bool — True for slots that need the initial beam step.
        mean_Z: mean atomic number (material scalar).
        mean_A: mean atomic weight in g/mol.
        rho: density in g/cm^3.
        J_keV: mean ionisation potential in keV (pre-computed).
        use_browning: if True use the Browning MFP formula; otherwise Joy.
        _br_mfp_const: Browning MFP constant ``1e7/(N_cm3)`` in nm·cm^2.
            Used only when ``use_browning=True``.
        _br_c1: Browning denominator coefficient ``0.005*Z^1.7``.
        _br_c2: Browning denominator coefficient ``0.0007*Z^2``.

    Returns:
        Updated (E, z) tensors (same storage, modified in-place for mask slots
        and returned for convenience).

    Raises:
        ValueError: if ``cz0 <= 0`` (beam must have a downward z-component).
    """
    if cz0 <= 0.0:
        raise ValueError(
            f"cz0 (cos(sig)) must be > 0 for the initial beam step, got {cz0:.6f}"
        )

    if not mask.any():
        return E, z

    E_m = E[mask]  # (M,) energies of active slots — all equal to starting_E
    dtype = E.dtype
    device = E.device
    M = int(mask.sum().item())

    # Draw step lengths from Exponential(mfp) — identical sampling to the main loop.
    r1 = torch.empty(M, dtype=dtype, device=device).uniform_()
    r1.clamp_(min=1e-7)

    if use_browning:
        e_sqrt = torch.sqrt(E_m)
        denom_br = E_m + _br_c1 * e_sqrt + _br_c2 / e_sqrt
        # Browning sigma [cm^2/atom bare]:  3.0e-18*Z^1.7 / denom
        # mfp = _br_mfp_const / sigma  (see MCConfig doc-string)
        sigma_m = (3.0e-18 * (mean_Z ** 1.7)) / denom_br
        mfp = _br_mfp_const / sigma_m
    else:
        # Joy 1995 screened-Rutherford (same formula as the main loop)
        alpha_m = 3.4e-3 * (mean_Z ** (2.0 / 3.0)) / E_m
        sigma_m = (
            5.21
            * 602.2
            * (mean_Z / E_m) ** 2
            * (4.0 * math.pi / (alpha_m * (1.0 + alpha_m)))
            * ((511.0 + E_m) / (1024.0 + E_m)) ** 2
        )
        mfp = 1.0e7 * mean_A / (rho * sigma_m)

    step = -mfp * torch.log(r1)

    # Energy loss along the un-deflected beam step.
    J_t = torch.tensor(J_keV, dtype=dtype, device=device)
    rho_t = torch.tensor(rho, dtype=dtype, device=device)
    de_ds = -0.00785 * (mean_Z / (mean_A * E_m)) * torch.log(1.166 * E_m / J_t + 0.9911)
    E_new_m = (E_m + step * rho_t * de_ds).clamp(min=1e-3)

    # Advance depth along the un-deflected beam direction (z-component = cz0 > 0).
    z_new_m = step * cz0  # z was 0 for all mask slots; z_new = 0 + step*cz0

    E[mask] = E_new_m
    z[mask] = z_new_m
    return E, z


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_gpu_mc(
    mean_Z: float,
    mean_A: float,
    rho: float,
    config: MCConfig,
    *,
    elastic_model: Optional[ElasticModel] = None,
) -> MCData:
    """Run the GPU Monte Carlo and return an EMsoft-compatible MCData.

    Implements the perpetual-pool vectorised trajectory loop:
    * N = ``config.n_electrons_parallel`` slots run concurrently.
    * A slot that exits (z < 0) or stalls (steps > n_max_steps) is reset to
      a fresh electron immediately, keeping the GPU saturated.
    * The loop ends when cumulative ``sims_finished >= config.n_simulations``.

    Accumulator update uses ``torch.Tensor.index_put_`` with ``accumulate=True``
    (scatter-add), which is supported on both CPU and CUDA.

    Args:
        mean_Z: occupancy-weighted mean atomic number.
        mean_A: occupancy-weighted mean atomic weight in g/mol.
        rho: density in g/cm^3.
        config: :class:`MCConfig` with all simulation parameters.
        elastic_model: override for the elastic cross-section model.  If
            ``None`` (default), the value from ``config.elastic_model`` is used.
            Explicit keyword argument takes precedence over the config field so
            that callers can switch models without rebuilding the config object.

    Returns:
        A :class:`MCData` with:

        * ``accum_e`` — (n_dir, n_dir, nE) int-valued float32 tensor.
        * ``accum_z`` — (n_dir_z, n_dir_z, nz, nE) int-valued float32 tensor.
        * ``EkeVs`` — (nE,) float32 energies in kV.
        * ``depth_step`` — bin width in nm.
        * ``depth_max`` — max tracked depth in nm.

    Raises:
        ValueError: if physics parameters are non-physical (rho <= 0, etc.).
    """
    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    if mean_A <= 0.0:
        raise ValueError(f"mean_A must be positive, got {mean_A}")
    if mean_Z <= 0.0:
        raise ValueError(f"mean_Z must be positive, got {mean_Z}")

    # Resolve elastic model: explicit kwarg overrides config field.
    _elastic_model: ElasticModel = elastic_model if elastic_model is not None else config.elastic_model
    _use_browning: bool = (_elastic_model == ElasticModel.BROWNING)

    # --- engine dispatch (LEVER 1 cupy + LEVER 3 numba) --------------------------
    # engine="cupy"   -> require the one-thread-per-electron CUDA RawKernel (raise if no CUDA)
    # engine="numba"  -> require the one-electron-per-thread numba CPU njit loop (raise if no numba)
    # engine="auto"   -> fail-safe chain: cupy (CUDA+Joy) -> numba (Joy, when no
    #                    CUDA/cupy but numba present) -> the validated PyTorch loop.
    # engine="pytorch" (or any other value) -> the PyTorch loop below.
    _engine = str(getattr(config, "engine", "auto")).lower()
    _is_cuda = torch.device(config.device).type == "cuda"
    if _engine == "cupy":
        return run_gpu_mc_cupy(mean_Z, mean_A, rho, config, elastic_model=_elastic_model)
    if _engine == "numba":
        from backend.forward_sim.mc.numba_mc import run_numba_mc  # noqa: PLC0415

        return run_numba_mc(mean_Z, mean_A, rho, config, elastic_model=_elastic_model)
    if _engine == "auto" and not _use_browning:
        # 1) Prefer the CUDA kernel when on a CUDA device with cupy available.
        if _is_cuda and cupy_mc_available():
            try:
                return run_gpu_mc_cupy(mean_Z, mean_A, rho, config, elastic_model=_elastic_model)
            except Exception as exc:
                # Never silent-fallback: name the failed engine + the exception so the
                # ~400x-slower downgrade is observable. Keep the broad catch (auto
                # must not crash) and drop through to numba / pytorch.
                logger.warning(
                    "MC engine='auto': cupy CUDA kernel failed (%s: %s); "
                    "downgrading to numba/pytorch.",
                    type(exc).__name__,
                    exc,
                )
        # 2) No CUDA/cupy: use the numba CPU one-electron-per-thread loop when
        #    numba compiles (the no-GPU fast path).  Run on whatever device the
        #    config requests — the numba engine bins into CPU accumulators and
        #    the MCData consumer (build_master) moves them to its own device.
        try:
            from backend.forward_sim.mc.numba_mc import (  # noqa: PLC0415
                numba_mc_available,
                run_numba_mc,
            )

            if numba_mc_available():
                return run_numba_mc(mean_Z, mean_A, rho, config, elastic_model=_elastic_model)
        except Exception as exc:
            # Never silent-fallback: name the failed engine + the exception so the
            # ~400x-slower downgrade to the PyTorch loop is observable. Keep the
            # broad catch (auto must not crash).
            logger.warning(
                "MC engine='auto': numba CPU kernel failed (%s: %s); "
                "downgrading to the PyTorch loop.",
                type(exc).__name__,
                exc,
            )

    device = torch.device(config.device)
    dtype = torch.float32

    # --- pre-compute material constants ---
    J_keV = float(mean_ionisation_potential_kev(mean_Z))
    starting_E = float(config.starting_E_keV)
    Ehistmin = float(config.Ehistmin)
    Ebinsize = float(config.Ebinsize)
    nE = int(config.nE)
    depth_step = float(config.depth_step)
    depth_max = float(config.depth_max)
    nz = int(round(depth_max / depth_step)) + 1  # 0..depth_max inclusive
    n_dir = int(config.n_dir)
    n_dir_z = int(config.n_dir_z)
    n_max_steps = int(config.n_max_steps)
    N = int(config.n_electrons_parallel)
    n_simulations = int(config.n_simulations)

    # Material scalars as device tensors for broadcasting.
    J = torch.tensor(J_keV, dtype=dtype, device=device)
    Z_t = torch.tensor(mean_Z, dtype=dtype, device=device)
    A_t = torch.tensor(mean_A, dtype=dtype, device=device)
    rho_t = torch.tensor(rho, dtype=dtype, device=device)

    # Browning model: pre-compute Z-dependent scalars that are constant
    # throughout the simulation (only Z-varying terms; E-varying terms
    # remain inside the loop as they depend on the per-slot energy).
    if _use_browning:
        # Total-sigma numerator: 3.0e-18 * Z^1.7
        _br_z17 = float(mean_Z ** 1.7)
        _br_num = 3.0e-18 * _br_z17          # cm^2 (bare Browning numerator constant)
        # Denominator coefficients (E-varying part computed in-loop):
        #   denom = E + 0.005*Z^1.7*E^0.5 + 0.0007*Z^2/E^0.5
        _br_c1 = 0.005 * _br_z17              # coefficient of E^0.5
        _br_c2 = 0.0007 * float(mean_Z ** 2)  # coefficient of E^(-0.5)
        # Number density N = 6.022e23 * rho / A [atoms/cm^3]; mfp = 1e7/(N*sigma)
        _br_N_cm3 = 6.022e23 * rho / mean_A   # atoms/cm^3
        _br_mfp_const = 1.0e7 / _br_N_cm3     # nm·cm^2 (mfp = _br_mfp_const / sigma)
        # Angular sampling: anisotropic fraction (Z-only)
        _br_f = float(max(0.0, min(1.0, 1.0 - 0.9 / (mean_Z ** 0.5))))
        # Browning tensors for vectorised inner loop
        _br_num_t  = torch.tensor(_br_num,      dtype=dtype, device=device)
        _br_c1_t   = torch.tensor(_br_c1,       dtype=dtype, device=device)
        _br_c2_t   = torch.tensor(_br_c2,       dtype=dtype, device=device)
        _br_mfp_t  = torch.tensor(_br_mfp_const, dtype=dtype, device=device)
        _br_f_t    = torch.tensor(_br_f,        dtype=dtype, device=device)

    # --- allocators ---
    # accum_e: (n_dir, n_dir, nE)
    accum_e = torch.zeros(n_dir, n_dir, nE, dtype=torch.float32, device=device)
    # accum_z: (n_dir_z, n_dir_z, nz, nE)
    accum_z = torch.zeros(n_dir_z, n_dir_z, nz, nE, dtype=torch.float32, device=device)

    # --- seed ---
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # --- beam direction z-component (scalar) needed for the initial beam step ---
    sig = math.radians(config.sig_deg)
    _cz0 = math.cos(sig)  # = cos(sig_deg), always > 0 for physical tilt < 90 deg

    # --- initial electron state (N slots) ---
    E = torch.full((N,), starting_E, dtype=dtype, device=device)
    z = torch.zeros(N, dtype=dtype, device=device)
    d0 = _init_direction(config.sig_deg, config.omega_deg, N, device, dtype)
    cx = d0[:, 0].clone()
    cy = d0[:, 1].clone()
    cz = d0[:, 2].clone()
    ticks = torch.zeros(N, dtype=torch.int32, device=device)

    # EMsoft parity: take ONE un-scattered beam step for ALL N slots before the
    # first scatter event.  EMMC.cl L224-232 / EMMC.f90 L490-500 do this outside
    # the while-loop; we must mirror it at init AND after each slot reset.
    all_mask = torch.ones(N, dtype=torch.bool, device=device)
    E, z = _take_initial_beam_step(
        E, z, _cz0, all_mask,
        mean_Z, mean_A, rho, J_keV,
        use_browning=_use_browning,
        _br_mfp_const=float(_br_mfp_const) if _use_browning else 0.0,
        _br_c1=float(_br_c1) if _use_browning else 0.0,
        _br_c2=float(_br_c2) if _use_browning else 0.0,
    )

    sims_finished = 0

    # --- perpetual-pool loop ---
    while sims_finished < n_simulations:
        if _use_browning:
            # --- Browning 1994 elastic model ---
            # 1. Total elastic cross-section [cm^2/atom, bare]
            #    sigma = 3.0e-18*Z^1.7 / (E + c1*E^0.5 + c2/E^0.5)
            e_sqrt = torch.sqrt(E)
            denom_br = E + _br_c1_t * e_sqrt + _br_c2_t / e_sqrt
            sigma_E = _br_num_t / denom_br

            # 2. Elastic MFP [nm]  mfp = 1e7 / (N * sigma)
            mfp = _br_mfp_t / sigma_E

            # 3. Step length [nm] ~ Exponential(mfp)
            r1 = torch.empty(N, dtype=dtype, device=device).uniform_()
            r1.clamp_(min=1e-7)
            step = -mfp * torch.log(r1)

            # 4. Energy loss rate — Joy-Luo Bethe (same as Joy model)
            de_ds = -0.00785 * (Z_t / (A_t * E)) * torch.log(1.166 * E / J + 0.9911)

            # 5. Browning two-component polar angle:
            #    branch selector r_branch (R1), angle variate r_angle (R2)
            r_branch = torch.empty(N, dtype=dtype, device=device).uniform_()
            r_angle  = torch.empty(N, dtype=dtype, device=device).uniform_()
            # Anisotropic alpha_B = 7.0e-3 * E^{-0.5}
            alpha_B = 7.0e-3 / e_sqrt
            # Anisotropic (Rutherford CDF) branch:
            cos_phi_aniso = 1.0 - 2.0 * alpha_B * r_angle / (1.0 + alpha_B - r_angle)
            # Isotropic branch:
            cos_phi_iso   = 1.0 - 2.0 * r_angle
            # Select branch per slot:
            cos_phi = torch.where(r_branch <= _br_f_t, cos_phi_aniso, cos_phi_iso)
            cos_phi = cos_phi.clamp(-1.0, 1.0)
            phi = torch.acos(cos_phi)
        else:
            # --- Joy 1995 screened-Rutherford model (EMsoft-faithful default) ---
            # 1. Screening parameter alpha = 3.4e-3 * Z^(2/3) / E
            alpha = 3.4e-3 * (Z_t ** (2.0 / 3.0)) / E  # (N,)

            # 2. Screened-Rutherford cross-section [cm^2/atom]
            sigma_E = (
                5.21
                * 602.2
                * (Z_t / E) ** 2
                * (4.0 * math.pi / (alpha * (1.0 + alpha)))
                * ((511.0 + E) / (1024.0 + E)) ** 2
            )

            # 3. Elastic mean free path [nm]
            mfp = 1.0e7 * A_t / (rho_t * sigma_E)

            # 4. Step length [nm] ~ Exponential(mfp)
            r1 = torch.empty(N, dtype=dtype, device=device).uniform_()
            r1.clamp_(min=1e-7)  # avoid log(0)
            step = -mfp * torch.log(r1)

            # 5. Energy loss rate [keV per (g/cm^2)] — Joy-Luo modified Bethe
            de_ds = -0.00785 * (Z_t / (A_t * E)) * torch.log(1.166 * E / J + 0.9911)

            # 6. Polar scattering angle (screened-Rutherford inverse CDF)
            r2 = torch.empty(N, dtype=dtype, device=device).uniform_()
            cos_phi = 1.0 - 2.0 * alpha * r2 / (1.0 + alpha - r2)
            cos_phi = cos_phi.clamp(-1.0, 1.0)
            phi = torch.acos(cos_phi)

        # 7. Azimuth
        psi = torch.empty(N, dtype=dtype, device=device).uniform_() * (2.0 * math.pi)

        # 8. Update directions (Joy eqs 3.12a-c)
        cx, cy, cz = _update_directions(cx, cy, cz, phi, psi)

        # 9. Advance depth
        z_new = z + step * cz

        # 10. Energy loss
        E_new = E + step * rho_t * de_ds
        E_new = E_new.clamp(min=1e-3)  # physical floor (> 0 keV)

        # 11. Identify backscattered (exited) electrons.
        #     ETA-FIX: we do NOT kill electrons at z >= depth_max.  EMsoft's
        #     trajectory loop has NO depth test (EMMC.cl:238 ``while (counter1 <
        #     steps)``; EMMC.f90:503 ``do while (traj.lt.num)``).  depth_max is
        #     BINNING-ONLY — it sizes accum_z but does not gate accum_e or eta.
        #     Deep electrons random-walk until they either exit through z<0
        #     (backscatter) or hit the scatter-count cap (n_max_steps=300).
        #     NOTE: we also do NOT terminate at E < Ehistmin.  Ehistmin is the
        #     lowest BINNING threshold; premature energy termination would reduce
        #     eta by ~20% vs EMsoft.  The physical floor (E.clamp(min=1e-3) above)
        #     and the stall guard prevent infinite loops.
        exited = z_new < 0.0                                       # (N,) bool — backscattered

        # 12. For exited (backscattered) electrons: bin direction, depth, energy.
        #     escape_depth = |z / cz| (Joy/EMsoft convention: old depth / new cz);
        #     all index/Lambert/scatter-add math lives in the shared _bin_exits
        #     helper, which the CuPy kernel path also calls (verbatim binning).
        if exited.any():
            cz_safe = cz.clone()
            cz_safe[~exited] = 1.0  # dummy for non-exiting slots (masked in helper)
            escape_depth = torch.abs(z / cz_safe)
            _bin_exits(
                accum_e, accum_z, exited, cx, cy, cz, escape_depth, E_new,
                Ehistmin=Ehistmin, Ebinsize=Ebinsize, nE=nE,
                depth_step=depth_step, nz=nz, n_dir=n_dir, n_dir_z=n_dir_z,
            )

        # 13. Reset exited OR stalled slots (ETA-FIX: no depth kill).
        #     Delegate to _compute_reset_mask (single source of truth) so the
        #     B2 unit-test guards the real production path.
        #     _compute_reset_mask expects ticks BEFORE the +1 increment and
        #     applies (ticks+1) > n_max_steps internally.
        reset = _compute_reset_mask(exited, z_new, ticks, depth_max, n_max_steps, E_new, Ehistmin)
        ticks = ticks + 1
        n_reset = int(reset.sum().item())

        if n_reset > 0:
            sims_finished += n_reset
            # Restore direction, energy, depth, and tick counter to fresh-electron
            # values for reset slots.  Energy and depth are set to starting values
            # first so that _take_initial_beam_step can read them as the baseline.
            E[reset] = starting_E
            z[reset] = 0.0
            cx[reset] = d0[reset, 0] if d0.shape[0] > 1 else float(d0[0, 0])
            cy[reset] = d0[reset, 1] if d0.shape[0] > 1 else float(d0[0, 1])
            cz[reset] = d0[reset, 2] if d0.shape[0] > 1 else float(d0[0, 2])
            ticks[reset] = 0

            # EMsoft parity: apply the initial un-scattered beam step for reset
            # slots (mirrors EMMC.cl L224-232; same step taken at startup above).
            # After this call, E[reset] and z[reset] reflect the post-beam-step
            # state; direction (cx/cy/cz) remains d0 (un-deflected).
            E, z = _take_initial_beam_step(
                E, z, _cz0, reset,
                mean_Z, mean_A, rho, J_keV,
                use_browning=_use_browning,
                _br_mfp_const=float(_br_mfp_const) if _use_browning else 0.0,
                _br_c1=float(_br_c1) if _use_browning else 0.0,
                _br_c2=float(_br_c2) if _use_browning else 0.0,
            )

        # 14. Advance state for surviving electrons.
        #     For reset slots, E[reset] and z[reset] already hold the
        #     post-beam-step values (set by _take_initial_beam_step above);
        #     torch.where selects those over E_new / z_new for reset slots.
        E = torch.where(reset, E, E_new)
        z = torch.where(reset, z, z_new)

    # --- assemble EkeVs ---
    EkeVs = torch.arange(nE, dtype=torch.float32, device=device) * Ebinsize + Ehistmin

    return MCData(
        accum_e=accum_e,
        accum_z=accum_z,
        EkeVs=EkeVs,
        depth_step=depth_step,
        depth_max=depth_max,
    )


# ---------------------------------------------------------------------------
# LEVER 1 — one-thread-per-electron CuPy RawKernel engine
# ---------------------------------------------------------------------------

_CUPY = None
_CUPY_AVAILABLE: Optional[bool] = None
_MC_RAWKERNEL = None


def _try_import_cupy():
    """Lazily import cupy and confirm the CUDA runtime actually loads."""
    global _CUPY, _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp  # noqa: PLC0415

            cp.cuda.runtime.runtimeGetVersion()
            _CUPY = cp
            _CUPY_AVAILABLE = True
        except Exception:
            _CUPY = None
            _CUPY_AVAILABLE = False
    return _CUPY


def _get_mc_rawkernel():
    """Lazy-compile (NVRTC) the one-thread-per-electron MC kernel."""
    global _MC_RAWKERNEL
    if _MC_RAWKERNEL is None:
        cp = _try_import_cupy()
        if cp is None:
            raise RuntimeError("cupy not available")
        from backend.forward_sim.mc.cupy_mc_kernel import MC_KERNEL_SOURCE

        _MC_RAWKERNEL = cp.RawKernel(MC_KERNEL_SOURCE, "mc_trajectories")
    return _MC_RAWKERNEL


def cupy_mc_available() -> bool:
    """True iff cupy + CUDA libs are importable AND the MC RawKernel compiles."""
    cp = _try_import_cupy()
    if cp is None:
        return False
    try:
        _get_mc_rawkernel()
        return True
    except Exception:
        return False


def _seed_lfsr113(cp, n_threads: int, seed: Optional[int]):
    """Build a valid (n_threads, 4) uint32 LFSR113 seed array on the GPU.

    LFSR113 requires z1>1, z2>7, z3>15, z4>127 (L'Ecuyer 1999); we OR in the
    minimum bits to guarantee that for every thread.  A SplitMix64-style hash of
    (seed, thread_id, lane) gives well-separated, reproducible per-thread states.
    """
    import numpy as np  # noqa: PLC0415

    GOLD = np.uint64(0x9E3779B97F4A7C15)
    MIX1 = np.uint64(0xBF58476D1CE4E5B9)
    MIX2 = np.uint64(0x94D049BB133111EB)
    MASK32 = np.uint64(0xFFFFFFFF)
    # uint64 wrap-around IS the intended (SplitMix64) arithmetic — suppress the
    # numpy "overflow in scalar multiply" RuntimeWarning for this block only.
    with np.errstate(over="ignore"):
        if seed is None:
            base = GOLD ^ np.uint64(np.random.SeedSequence().entropy & ((1 << 64) - 1))
        else:
            base = np.uint64(seed & ((1 << 64) - 1)) * GOLD + np.uint64(1)

        tid = np.arange(n_threads, dtype=np.uint64)
        seeds = np.empty((n_threads, 4), dtype=np.uint32)
        mins = (np.uint32(2), np.uint32(8), np.uint32(16), np.uint32(128))
        for lane in range(4):
            x = base + tid * GOLD + np.uint64(lane + 1) * MIX1
            # SplitMix64 finaliser
            x = (x ^ (x >> np.uint64(30))) * MIX1
            x = (x ^ (x >> np.uint64(27))) * MIX2
            x = x ^ (x >> np.uint64(31))
            col = (x & MASK32).astype(np.uint32) | mins[lane]
            seeds[:, lane] = col
    return cp.asarray(seeds.reshape(-1))


def run_gpu_mc_cupy(
    mean_Z: float,
    mean_A: float,
    rho: float,
    config: MCConfig,
    *,
    elastic_model: Optional[ElasticModel] = None,
) -> MCData:
    """Run the Monte Carlo with the one-thread-per-electron CuPy RawKernel.

    Each CUDA thread runs ``config.cupy_n_el_per_thread`` electrons' FULL
    trajectories in registers (mirroring EMsoft's EMMCOpenCL ``num_el`` model),
    emitting one exit tuple ``(cx, cy, cz, escape_depth, energy, exited)`` per
    electron.  Those exit scalars are bridged zero-copy to torch (DLPack) and
    fed into the SHARED host binning helper :func:`_bin_exits` — the exact same
    accum_e / accum_z math the PyTorch loop uses — so only *where trajectories
    run* changes.

    Only the default Joy screened-Rutherford model is ported to the kernel;
    ``ElasticModel.BROWNING`` falls back to the PyTorch engine (the kernel raises
    via the caller if Browning is requested).

    Args mirror :func:`run_gpu_mc`.  Requires cupy + CUDA; raises if unavailable.

    Returns:
        A :class:`MCData` identical in shape/contract to :func:`run_gpu_mc`.
    """
    cp = _try_import_cupy()
    if cp is None:
        raise RuntimeError(
            "run_gpu_mc_cupy requires cupy + CUDA; none available. "
            "Use engine='pytorch' or engine='auto' (which falls back)."
        )

    _elastic_model = elastic_model if elastic_model is not None else config.elastic_model
    if _elastic_model == ElasticModel.BROWNING:
        raise ValueError(
            "run_gpu_mc_cupy supports only ElasticModel.JOY; "
            "Browning is not ported to the CUDA kernel (use the PyTorch engine)."
        )

    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    if mean_A <= 0.0:
        raise ValueError(f"mean_A must be positive, got {mean_A}")
    if mean_Z <= 0.0:
        raise ValueError(f"mean_Z must be positive, got {mean_Z}")

    import numpy as np  # noqa: PLC0415

    device = torch.device("cuda")
    dtype = torch.float32

    # --- scalars (identical to the PyTorch path's precomputed values) ---
    J_keV = float(mean_ionisation_potential_kev(mean_Z))
    starting_E = float(config.starting_E_keV)
    Ehistmin = float(config.Ehistmin)
    Ebinsize = float(config.Ebinsize)
    nE = int(config.nE)
    depth_step = float(config.depth_step)
    depth_max = float(config.depth_max)
    nz = int(round(depth_max / depth_step)) + 1
    n_dir = int(config.n_dir)
    n_dir_z = int(config.n_dir_z)
    n_max_steps = int(config.n_max_steps)
    n_simulations = int(config.n_simulations)

    sig = math.radians(config.sig_deg)
    om = math.radians(config.omega_deg)
    cz0 = math.cos(sig)
    # Match _take_initial_beam_step semantics: the beam must have a downward
    # z-component (sig < 90 deg). The CUDA kernel does not guard this, so raise
    # host-side rather than emit a silently-wrong non-entering trajectory.
    if cz0 <= 0.0:
        raise ValueError(
            f"cz0 (cos(sig)) must be > 0 for the initial beam step, got {cz0:.6f} "
            f"(sig_deg={config.sig_deg})"
        )
    d0x = math.sin(sig) * math.cos(om)
    d0y = math.sin(sig) * math.sin(om)
    d0z = math.cos(sig)

    # --- accumulators (torch, shared binning) ---
    accum_e = torch.zeros(n_dir, n_dir, nE, dtype=torch.float32, device=device)
    accum_z = torch.zeros(n_dir_z, n_dir_z, nz, nE, dtype=torch.float32, device=device)

    kernel = _get_mc_rawkernel()
    threads_per_block = 256
    n_el_per_thread = max(1, int(config.cupy_n_el_per_thread))
    chunk_electrons = max(1, int(config.cupy_chunk_electrons))

    # SplitMix-derived per-thread seeds.  We reseed each chunk with an offset so
    # chunks draw disjoint streams (otherwise every chunk would repeat).
    base_seed = config.seed

    remaining = n_simulations
    chunk_idx = 0
    while remaining > 0:
        chunk = min(chunk_electrons, remaining)
        # threads needed for this chunk
        n_threads = (chunk + n_el_per_thread - 1) // n_el_per_thread
        n_blocks = (n_threads + threads_per_block - 1) // threads_per_block
        n_threads_launched = n_blocks * threads_per_block

        chunk_seed = None if base_seed is None else (base_seed + 1_000_003 * (chunk_idx + 1))
        seeds = _seed_lfsr113(cp, n_threads_launched, chunk_seed)

        out_cx = cp.empty(chunk, dtype=cp.float32)
        out_cy = cp.empty(chunk, dtype=cp.float32)
        out_cz = cp.empty(chunk, dtype=cp.float32)
        out_depth = cp.empty(chunk, dtype=cp.float32)
        out_energy = cp.empty(chunk, dtype=cp.float32)
        out_exited = cp.empty(chunk, dtype=cp.int32)

        kernel(
            (n_blocks,), (threads_per_block,),
            (out_cx, out_cy, out_cz, out_depth, out_energy, out_exited,
             seeds,
             np.int64(chunk), np.int32(n_el_per_thread),
             np.float32(starting_E), np.float32(mean_Z), np.float32(mean_A),
             np.float32(rho), np.float32(J_keV), np.float32(cz0),
             np.float32(d0x), np.float32(d0y), np.float32(d0z),
             np.int32(n_max_steps)),
        )

        # Zero-copy bridge cupy -> torch (DLPack).
        t_cx = torch.from_dlpack(out_cx)
        t_cy = torch.from_dlpack(out_cy)
        t_cz = torch.from_dlpack(out_cz)
        t_depth = torch.from_dlpack(out_depth)
        t_energy = torch.from_dlpack(out_energy)
        t_exited = torch.from_dlpack(out_exited).bool()

        # Shared, verbatim binning (escape_depth already = |z_old/cz_new| from kernel).
        _bin_exits(
            accum_e, accum_z, t_exited, t_cx, t_cy, t_cz, t_depth, t_energy,
            Ehistmin=Ehistmin, Ebinsize=Ebinsize, nE=nE,
            depth_step=depth_step, nz=nz, n_dir=n_dir, n_dir_z=n_dir_z,
        )

        remaining -= chunk
        chunk_idx += 1

    EkeVs = torch.arange(nE, dtype=torch.float32, device=device) * Ebinsize + Ehistmin
    return MCData(
        accum_e=accum_e,
        accum_z=accum_z,
        EkeVs=EkeVs,
        depth_step=depth_step,
        depth_max=depth_max,
    )


def gpu_mc_for_structure(
    structure,
    config: Optional[MCConfig] = None,
) -> MCData:
    """Convenience wrapper: derive composition and run the GPU MC.

    Args:
        structure: :class:`~backend.forward_sim.crystal.xtal_io.CrystalStructure`
            from :func:`~backend.forward_sim.crystal.xtal_io.read_crystal_structure`.
        config: :class:`MCConfig` override; if None a default config is used
            (20 kV, 10 M electrons, CPU device).

    Returns:
        :class:`MCData` ready to pass to
        :func:`~backend.forward_sim.dynamical.master_builder.build_master`.
    """
    from backend.forward_sim.mc.composition import mc_composition_from_structure

    comp = mc_composition_from_structure(structure)
    if config is None:
        config = MCConfig()
    return run_gpu_mc(comp.mean_Z, comp.mean_A, comp.rho, config)
