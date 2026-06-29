"""LEVER 3 — numba CPU one-electron-per-thread Monte Carlo (no-GPU fallback).

This module is the CPU analogue of the CuPy one-thread-per-electron kernel
(:mod:`backend.forward_sim.mc.cupy_mc_kernel`): a ``numba.njit(parallel=True,
fastmath=True)`` loop with ``prange`` over electrons, where each thread runs ONE
electron's FULL trajectory in registers and writes its exit scalars to a
preallocated per-electron array — NO shared histogram in the parallel region
(race-safe, mirroring EMsoft's EMMC.f90 OpenMP one-electron-per-thread model).

Why this exists
---------------
The default PyTorch CPU MC (:func:`gpu_monte_carlo.run_gpu_mc`) is *step-major*:
it advances ALL ~50k slots by ONE step, hundreds of times, each step allocating
several (N,) tensors and doing a host-side scatter-add.  On CPU that overhead
dominates and the run is slow.  The trajectory-major numba loop keeps each
electron's whole walk in registers and parallelises trivially across cores —
the same structural win the CuPy kernel gets on the GPU.

Physics (validated, ported verbatim from the CuPy kernel)
---------------------------------------------------------
Joy-1995 screened-Rutherford elastic scattering + Joy-Luo Bethe-CSDA energy
loss, the SAME constants used by both the PyTorch loop
(:func:`gpu_monte_carlo.run_gpu_mc` inner loop) and the CuPy kernel
(:data:`cupy_mc_kernel.MC_KERNEL_SOURCE`).  Per electron:

* Initial un-scattered beam step BEFORE the first scatter (advance z by
  ``step*cz0``, apply energy loss, no direction change, no exit test) —
  EMMC.f90 L490-503 / ``_take_initial_beam_step``.
* Per step: ``alpha = 3.4e-3·Z^(2/3)/E`` ; screened-Rutherford ``sigma`` ;
  ``mfp`` ; ``step = -mfp·log(r1)`` ; ``de_ds`` (Joy-Luo Bethe) ;
  polar ``cos φ`` inverse-CDF ; azimuth ``ψ`` ; direction-cosine update
  (Joy 3.12a-c, ``|cz|>0.99999`` pole branch) ; ``z_new = z + step·cz_new`` ;
  ``E_new = E + step·rho·de_ds`` (floored 1e-3).
* ``escape_depth = |z_old / cz_new|`` (Joy/EMsoft convention; old depth, new cz).
* Record the FIRST backscatter (``z_new < 0``) and break; otherwise loop until
  ``n_max_steps`` (EMsoft GPU = 300; CPU oracle = 500).

RNG
---
Inline LFSR113 (L'Ecuyer 1999), the *exact* RNG the validated CuPy kernel uses,
seeded per-electron via the SAME SplitMix64 host helper
(:func:`gpu_monte_carlo._seed_lfsr113`).  A bit-exact match to the PyTorch
``torch.uniform_`` stream is impossible (different RNG) and is NOT the gate —
statistical equivalence is (master NCC ≥ 0.998 vs the PyTorch/EMsoft master,
eta within ~1-2%, Poisson-consistent marginals).

Binning
-------
The kernel emits ONLY the per-electron exit scalars; ALL accumulator binning
(energy/depth ``nint`` indices, the ``-cx,-cy,-cz`` Lambert negation, the
accum_e / accum_z split) is done by the EXISTING single-threaded host helper
:func:`gpu_monte_carlo._bin_exits` — byte-for-byte the validated path the
PyTorch and CuPy engines also call.  Only Joy is ported; ``ElasticModel.BROWNING``
falls back to the PyTorch loop (the dispatcher in ``run_gpu_mc`` handles this).

References:
    Joy, D.C. 1995, *Monte Carlo Modeling for Electron Microscopy and
    Microanalysis*, Oxford University Press, ch.3.
"""
from __future__ import annotations

import math
from typing import Optional

import numba
import numpy as np
import torch
from numba import njit, prange


# ---------------------------------------------------------------------------
# Inline LFSR113 RNG (verbatim from cupy_mc_kernel.py:lfsr113) — float in (0,1].
# ---------------------------------------------------------------------------

@njit(numba.float32(numba.uint32[:]), fastmath=True, inline="always", cache=True)
def _lfsr113(s):
    """Advance a 4-uint32 LFSR113 state in place; return a float in (0, 1].

    Bit-identical to the CuPy kernel's ``lfsr113`` (EMMC.cl:79-99).  Constraints
    z1>1, z2>7, z3>15, z4>127 are guaranteed by the host SplitMix64 seeder
    (:func:`gpu_monte_carlo._seed_lfsr113`).

    NOTE on the ``& M`` masks: in CUDA C ``unsigned int`` is 32-bit, so every
    ``<<`` truncates to 32 bits automatically (the CuPy kernel relies on this).
    numba promotes ``uint32 << uint32`` to a 64-bit intermediate, so we must mask
    each left-shift result back to 32 bits with ``& 0xFFFFFFFF`` to reproduce the
    CUDA wrap-around exactly.  Without these masks the high bits survive and the
    stream is wrong (returns values ≫ 1, breaking ``-mfp·log(r)``).
    """
    M = numba.uint32(0xFFFFFFFF)
    z1 = s[0]
    z2 = s[1]
    z3 = s[2]
    z4 = s[3]
    b = ((((z1 << numba.uint32(6)) & M) ^ z1) >> numba.uint32(13)) & M
    z1 = ((((z1 & numba.uint32(4294967294)) << numba.uint32(18)) & M) ^ b) & M
    b = ((((z2 << numba.uint32(2)) & M) ^ z2) >> numba.uint32(27)) & M
    z2 = ((((z2 & numba.uint32(4294967288)) << numba.uint32(2)) & M) ^ b) & M
    b = ((((z3 << numba.uint32(13)) & M) ^ z3) >> numba.uint32(21)) & M
    z3 = ((((z3 & numba.uint32(4294967280)) << numba.uint32(7)) & M) ^ b) & M
    b = ((((z4 << numba.uint32(3)) & M) ^ z4) >> numba.uint32(12)) & M
    z4 = ((((z4 & numba.uint32(4294967168)) << numba.uint32(13)) & M) ^ b) & M
    s[0] = z1
    s[1] = z2
    s[2] = z3
    s[3] = z4
    r = (z1 ^ z2 ^ z3 ^ z4) & M
    # Map to (0, 1]; never exactly 0 (guards log(0)).  2^-32.
    u = numba.float32(r) * numba.float32(2.3283064365386963e-10)
    if u <= numba.float32(1e-7):
        u = numba.float32(1e-7)
    return u


# ---------------------------------------------------------------------------
# One-electron-per-thread trajectory kernel (njit parallel, prange over electrons)
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def _mc_trajectories_numba(
    seeds,        # (N, 4) uint32 — per-electron LFSR113 seed (one lane per electron)
    out,          # (N, 6) float32 — [cx, cy, cz, escape_depth, energy, exited]
    startE,       # float32 beam energy keV
    Zeff,         # float32 mean atomic number
    Aeff,         # float32 mean atomic weight g/mol
    rho,          # float32 density g/cm^3
    J,            # float32 mean ionisation potential keV
    cz0,          # float32 cos(sig) — beam z-component
    d0x, d0y, d0z,  # float32 initial beam direction cosines
    n_max_steps,  # int32 scatter-count cap (EMsoft GPU = 300)
):
    """Run N independent electron trajectories, one per ``prange`` iteration.

    RACE-SAFE: each iteration ``i`` reads only ``seeds[i]`` (copied into a thread-
    local register array) and writes only ``out[i]`` — no shared mutable state in
    the parallel region.  The host single-threaded :func:`gpu_monte_carlo._bin_exits`
    then bins ``out`` into the accumulators.

    The physics is a verbatim port of the validated CuPy kernel (Joy 1995
    screened-Rutherford + Joy-Luo Bethe).  ``out[i]`` columns:
    ``[cx, cy, cz, escape_depth, exit_energy, exited(0/1)]``.
    """
    N = out.shape[0]
    PI = numba.float32(3.14159265358979)
    Z23 = numba.float32(Zeff ** numba.float32(0.66666666666))  # Z^(2/3) — constant
    Z2 = numba.float32(Zeff * Zeff)                            # Z^2     — constant

    for i in prange(N):
        # Thread-local RNG state (copy the seed lane into registers).
        s = np.empty(4, dtype=np.uint32)
        s[0] = seeds[i, 0]
        s[1] = seeds[i, 1]
        s[2] = seeds[i, 2]
        s[3] = seeds[i, 3]

        # --- fresh electron state ---
        cx = numba.float32(d0x)
        cy = numba.float32(d0y)
        cz = numba.float32(d0z)
        E = numba.float32(startE)
        zdepth = numba.float32(0.0)

        # --- initial un-scattered beam step (EMMC.f90 L490-503) ---
        alpha = numba.float32(3.4e-3) * Z23 / E
        sigma = (
            numba.float32(5.21) * numba.float32(602.2) * (Z2 / (E * E))
            * (numba.float32(4.0) * PI / (alpha * (numba.float32(1.0) + alpha)))
            * ((numba.float32(511.0) + E) / (numba.float32(1024.0) + E)) ** numba.float32(2.0)
        )
        mfp = numba.float32(1.0e7) * Aeff / (rho * sigma)
        r1 = _lfsr113(s)
        step = -mfp * numba.float32(math.log(r1))
        de_ds = (
            numba.float32(-0.00785) * (Zeff / (Aeff * E))
            * numba.float32(math.log(numba.float32(1.166) * E / J + numba.float32(0.9911)))
        )
        E = E + step * rho * de_ds
        if E < numba.float32(1e-3):
            E = numba.float32(1e-3)
        zdepth = step * cz0  # z was 0; advance along un-deflected beam

        # defaults if the electron never backscatters
        ex_cx = numba.float32(0.0)
        ex_cy = numba.float32(0.0)
        ex_cz = numba.float32(0.0)
        ex_depth = numba.float32(0.0)
        ex_energy = numba.float32(0.0)
        exited = numba.int32(0)

        # --- scatter loop (per-thread register trajectory) ---
        for _t in range(n_max_steps):
            # 1. screening parameter
            alpha = numba.float32(3.4e-3) * Z23 / E
            # 2. screened-Rutherford cross-section
            sigma = (
                numba.float32(5.21) * numba.float32(602.2) * (Z2 / (E * E))
                * (numba.float32(4.0) * PI / (alpha * (numba.float32(1.0) + alpha)))
                * ((numba.float32(511.0) + E) / (numba.float32(1024.0) + E)) ** numba.float32(2.0)
            )
            # 3. elastic mean free path [nm]
            mfp = numba.float32(1.0e7) * Aeff / (rho * sigma)
            # 4. step length ~ Exponential(mfp)
            r1 = _lfsr113(s)
            step = -mfp * numba.float32(math.log(r1))
            # 5. Joy-Luo Bethe energy loss rate
            de_ds = (
                numba.float32(-0.00785) * (Zeff / (Aeff * E))
                * numba.float32(math.log(numba.float32(1.166) * E / J + numba.float32(0.9911)))
            )
            # 6. polar scattering angle (screened-Rutherford inverse CDF)
            r2 = _lfsr113(s)
            cphi = numba.float32(1.0) - numba.float32(2.0) * alpha * r2 / (numba.float32(1.0) + alpha - r2)
            if cphi > numba.float32(1.0):
                cphi = numba.float32(1.0)
            if cphi < numba.float32(-1.0):
                cphi = numba.float32(-1.0)
            phi = numba.float32(math.acos(cphi))
            # 7. azimuth
            r3 = _lfsr113(s)
            psi = r3 * (numba.float32(2.0) * PI)

            # 8. direction-cosine update (Joy 3.12a-c) using OLD (cx,cy,cz)
            sphi = numba.float32(math.sin(phi))
            cph = numba.float32(math.cos(phi))
            cpsi = numba.float32(math.cos(psi))
            spsi = numba.float32(math.sin(psi))
            if abs(cz) > numba.float32(0.99999):
                sgn = numba.float32(1.0) if cz >= numba.float32(0.0) else numba.float32(-1.0)
                ncx = sphi * cpsi
                ncy = sphi * spsi
                ncz = sgn * cph
            else:
                dsq = numba.float32(math.sqrt(numba.float32(1.0) - cz * cz))
                if dsq < numba.float32(1e-30):
                    dsq = numba.float32(1e-30)
                dsqi = numba.float32(1.0) / dsq
                ncx = sphi * (cx * cz * cpsi - cy * spsi) * dsqi + cx * cph
                ncy = sphi * (cy * cz * cpsi + cx * spsi) * dsqi + cy * cph
                ncz = -sphi * cpsi * dsq + cz * cph
            # renormalise (drift over many steps)
            nrm = numba.float32(math.sqrt(ncx * ncx + ncy * ncy + ncz * ncz))
            if nrm < numba.float32(1e-30):
                nrm = numba.float32(1e-30)
            ncx = ncx / nrm
            ncy = ncy / nrm
            ncz = ncz / nrm

            # 9. advance depth (z) and 10. energy loss
            z_new = zdepth + step * ncz
            E_new = E + step * rho * de_ds
            if E_new < numba.float32(1e-3):
                E_new = numba.float32(1e-3)

            # 11. exit test — backscattered when z_new < 0 (strict, PyTorch parity)
            if z_new < numba.float32(0.0):
                # escape_depth = |z_old / cz_new|  (old depth / new cz)
                czs = ncz if ncz != numba.float32(0.0) else numba.float32(1.0)
                ex_depth = abs(zdepth / czs)
                ex_energy = E_new            # POST-step energy (EMsoft nint convention)
                ex_cx = ncx
                ex_cy = ncy
                ex_cz = ncz
                exited = numba.int32(1)
                break

            # 12. advance state
            cx = ncx
            cy = ncy
            cz = ncz
            zdepth = z_new
            E = E_new

        out[i, 0] = ex_cx
        out[i, 1] = ex_cy
        out[i, 2] = ex_cz
        out[i, 3] = ex_depth
        out[i, 4] = ex_energy
        out[i, 5] = numba.float32(exited)


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

_NUMBA_AVAILABLE: Optional[bool] = None


def numba_mc_available() -> bool:
    """True iff numba is importable and the njit MC kernel compiles.

    The compile is cached (``cache=True``) so the first call pays the JIT cost
    once and subsequent calls are cheap.  Any compile failure → False so
    ``engine='auto'`` cleanly falls back to the PyTorch loop.
    """
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            # Trigger compilation on a 1-electron probe (cheap, bounded).
            seeds = np.full((1, 4), 1000, dtype=np.uint32)
            out = np.zeros((1, 6), dtype=np.float32)
            _mc_trajectories_numba(
                seeds, out,
                np.float32(20.0), np.float32(28.0), np.float32(58.7),
                np.float32(8.9), np.float32(0.3), np.float32(0.342),
                np.float32(0.94), np.float32(0.0), np.float32(0.342),
                np.int32(10),
            )
            _NUMBA_AVAILABLE = True
        except Exception:
            _NUMBA_AVAILABLE = False
    return _NUMBA_AVAILABLE


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------

def run_numba_mc(
    mean_Z: float,
    mean_A: float,
    rho: float,
    config,
    *,
    elastic_model=None,
):
    """Run the Monte Carlo with the numba CPU one-electron-per-thread engine.

    Mirrors :func:`gpu_monte_carlo.run_gpu_mc_cupy` exactly, but on CPU:
    chunk the ``n_simulations`` electrons, seed each chunk's electrons with the
    SplitMix64 host seeder, run the njit ``prange`` trajectory kernel into a
    preallocated ``(chunk, 6)`` exit array, bridge to torch, and feed the SHARED
    host binning helper :func:`gpu_monte_carlo._bin_exits` — the exact same
    accum_e / accum_z math the PyTorch and CuPy engines use.

    Args mirror :func:`gpu_monte_carlo.run_gpu_mc`.  Only the Joy model is
    supported; ``ElasticModel.BROWNING`` raises (the dispatcher routes Browning
    to the PyTorch loop instead).

    Returns:
        A :class:`~backend.forward_sim.mc.emsoft_mc_input.MCData` identical in
        shape/contract to :func:`gpu_monte_carlo.run_gpu_mc`.

    Raises:
        RuntimeError: if numba is not importable / the kernel does not compile.
        ValueError: for Browning, or non-physical material parameters.
    """
    # Local imports avoid a circular import at module load
    # (gpu_monte_carlo imports nothing from here; this keeps the dependency
    # one-directional and lets gpu_monte_carlo lazy-import run_numba_mc).
    from backend.forward_sim.mc.emsoft_mc_input import MCData
    from backend.forward_sim.mc.gpu_monte_carlo import _bin_exits, _seed_lfsr113
    from backend.forward_sim.mc.scattering_mc import (
        ElasticModel,
        mean_ionisation_potential_kev,
    )

    if not numba_mc_available():
        raise RuntimeError(
            "run_numba_mc requires numba; the njit MC kernel failed to compile. "
            "Use engine='pytorch' or engine='auto' (which falls back)."
        )

    _elastic_model = elastic_model if elastic_model is not None else config.elastic_model
    if _elastic_model == ElasticModel.BROWNING:
        raise ValueError(
            "run_numba_mc supports only ElasticModel.JOY; "
            "Browning is not ported to the numba kernel (use the PyTorch engine)."
        )

    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    if mean_A <= 0.0:
        raise ValueError(f"mean_A must be positive, got {mean_A}")
    if mean_Z <= 0.0:
        raise ValueError(f"mean_Z must be positive, got {mean_Z}")

    # --- scalars (identical to the PyTorch / cupy paths' precomputed values) ---
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
    # z-component (sig < 90 deg). The numba kernel does not guard this, so raise
    # host-side rather than emit a silently-wrong non-entering trajectory.
    if cz0 <= 0.0:
        raise ValueError(
            f"cz0 (cos(sig)) must be > 0 for the initial beam step, got {cz0:.6f} "
            f"(sig_deg={config.sig_deg})"
        )
    d0x = math.sin(sig) * math.cos(om)
    d0y = math.sin(sig) * math.sin(om)
    d0z = math.cos(sig)

    # --- accumulators (torch CPU, shared binning) ---
    device = torch.device("cpu")
    accum_e = torch.zeros(n_dir, n_dir, nE, dtype=torch.float32, device=device)
    accum_z = torch.zeros(n_dir_z, n_dir_z, nz, nE, dtype=torch.float32, device=device)

    # Chunk size: one exit row per electron.  Reuse cupy_chunk_electrons (an
    # exit-array memory bound) so the same config field controls both CPU/GPU
    # one-thread-per-electron engines.
    chunk_electrons = max(1, int(getattr(config, "cupy_chunk_electrons", 8_000_000)))
    base_seed = config.seed

    remaining = n_simulations
    chunk_idx = 0
    while remaining > 0:
        chunk = min(chunk_electrons, remaining)

        # Per-electron SplitMix64-derived LFSR113 seeds (one lane per electron).
        # _seed_lfsr113 returns a flat (chunk*4,) cupy/numpy array; we need numpy
        # (chunk, 4).  Pass cp=np so it builds a host numpy array directly.
        chunk_seed = None if base_seed is None else (base_seed + 1_000_003 * (chunk_idx + 1))
        seeds_flat = _seed_lfsr113(np, chunk, chunk_seed)
        seeds = np.ascontiguousarray(
            np.asarray(seeds_flat, dtype=np.uint32).reshape(chunk, 4)
        )

        out = np.empty((chunk, 6), dtype=np.float32)
        _mc_trajectories_numba(
            seeds, out,
            np.float32(starting_E), np.float32(mean_Z), np.float32(mean_A),
            np.float32(rho), np.float32(J_keV), np.float32(cz0),
            np.float32(d0x), np.float32(d0y), np.float32(d0z),
            np.int32(n_max_steps),
        )

        # Bridge to torch (zero-copy view of the contiguous numpy columns).
        t_out = torch.from_numpy(out)
        t_cx = t_out[:, 0]
        t_cy = t_out[:, 1]
        t_cz = t_out[:, 2]
        t_depth = t_out[:, 3]
        t_energy = t_out[:, 4]
        t_exited = t_out[:, 5].bool()

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
