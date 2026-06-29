"""LEVER 1 — one-thread-per-electron CuPy RawKernel Monte Carlo.

This module provides a ``cupy.RawKernel`` that runs each electron's FULL
trajectory in registers (mirroring EMsoft's EMMC.cl one-work-item-per-electron
model), instead of the step-major PyTorch loop in :mod:`gpu_monte_carlo` which
advances ALL slots by ONE step ×hundreds with a host sync per step.

The kernel ports the *validated* Joy-1995 screened-Rutherford + Joy-Luo
Bethe-CSDA physics from
:func:`backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc` (the inner loop
~:619-661) verbatim, with one difference that does NOT change the physics: the
RNG is an inline LFSR113 stream (ported from EMMC.cl :79-99) seeded per-thread,
because cuRAND/curand_kernel.h pulls in headers that are awkward under NVRTC and
LFSR113 is dependency-free and EMsoft-faithful.  A bit-exact match to the
PyTorch stream is impossible (different RNG) and is NOT the gate — statistical
equivalence is (master NCC ≥ 0.998 vs the PyTorch-MC / EMsoft-MC master).

The kernel emits ONLY the per-electron exit scalars (cx, cy, cz exit direction
cosines + escape depth + exit energy + an exited flag).  ALL accumulator binning
(energy index, Lambert direction indices with the -cx,-cy,-cz negation, the
accum_e / accum_z split) is then performed by the EXISTING host code in
:func:`gpu_monte_carlo.run_gpu_mc_cupy`, so only *where the trajectories run*
changes — the binning math is byte-for-byte the validated path.

Trajectory semantics (mirrors :func:`gpu_monte_carlo.run_gpu_mc`):
* Initial un-scattered beam step BEFORE the first scatter (advance z by
  step*cz0, apply energy loss, no direction change, no exit test) — EMMC.cl
  :224-232 / our ``_take_initial_beam_step``.
* Per step: alpha = 3.4e-3·Z^(2/3)/E ; screened-Rutherford sigma ; mfp ;
  step = -mfp·log(r1) ; de_ds (Joy-Luo Bethe) ; polar cos φ inverse-CDF ;
  azimuth ψ ; direction-cosine update (Joy eqs 3.12a-c, |cz|>0.99999 pole
  branch) ; z_new = z + step·cz_new ; E_new = E + step·rho·de_ds (floored 1e-3).
* escape_depth = |z_old / cz_new|  (Joy/EMsoft convention; old depth, new cz).
* Record the FIRST backscatter (z_new < 0) and break (a per-thread register
  loop needs no lock-step; the recorded scalar is identical to EMMC.cl's
  counter2-gated first crossing).
* Otherwise advance and loop until n_max_steps (stall cap; EMsoft GPU = 300).

The host then negates (cx,cy,cz) and bins exactly as the PyTorch path does.
"""
from __future__ import annotations

# CUDA-C source for the one-thread-per-electron MC kernel.
#
# Each thread runs ``n_el_per_thread`` electrons sequentially (mirroring
# EMMC.cl's ``num_el`` inner loop) and writes one (cx,cy,cz,depth,energy,exited)
# tuple per electron into the global output arrays.  Total electrons simulated
# = gridDim*blockDim * n_el_per_thread.
#
# Physics constants are passed as kernel scalars so they are *identical* to the
# Python-side precomputed values (J, mean_Z, mean_A, rho, cz0) — no float32
# recompute drift vs the validated PyTorch path.
MC_KERNEL_SOURCE = r"""
extern "C" {

// LFSR113 (L'Ecuyer 1999) — the exact RNG EMMC.cl uses (EMMC.cl:79-99).
// Returns a float in (0, 1].  State is 4 uints; advanced in place.
// Constraints: z1>1, z2>7, z3>15, z4>127 (enforced at seed time on host).
__device__ __forceinline__ float lfsr113(unsigned int* s) {
    unsigned int b;
    unsigned int z1 = s[0], z2 = s[1], z3 = s[2], z4 = s[3];
    b  = ((z1 << 6)  ^ z1) >> 13;
    z1 = ((z1 & 4294967294U) << 18) ^ b;
    b  = ((z2 << 2)  ^ z2) >> 27;
    z2 = ((z2 & 4294967288U) << 2)  ^ b;
    b  = ((z3 << 13) ^ z3) >> 21;
    z3 = ((z3 & 4294967280U) << 7)  ^ b;
    b  = ((z4 << 3)  ^ z4) >> 12;
    z4 = ((z4 & 4294967168U) << 13) ^ b;
    s[0] = z1; s[1] = z2; s[2] = z3; s[3] = z4;
    unsigned int r = z1 ^ z2 ^ z3 ^ z4;
    // Map to (0,1]; never exactly 0 (guards log(0)).  2^-32.
    float u = (float)r * 2.3283064365386963e-10f;
    if (u <= 1e-7f) u = 1e-7f;
    return u;
}

__global__ void mc_trajectories(
    float* __restrict__ out_cx,
    float* __restrict__ out_cy,
    float* __restrict__ out_cz,
    float* __restrict__ out_depth,
    float* __restrict__ out_energy,
    int*   __restrict__ out_exited,
    const unsigned int* __restrict__ seeds,   // 4 uints per thread
    const long long n_electrons,              // total electrons to simulate
    const int   n_el_per_thread,
    const float startE,                       // beam energy keV
    const float Zeff,
    const float Aeff,
    const float rho,
    const float J,                            // mean ionisation potential keV
    const float cz0,                          // cos(sig) — beam z-component
    const float d0x, const float d0y, const float d0z,
    const int   n_max_steps
) {
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // Load this thread's RNG state into registers.
    unsigned int s[4];
    s[0] = seeds[4*gid + 0];
    s[1] = seeds[4*gid + 1];
    s[2] = seeds[4*gid + 2];
    s[3] = seeds[4*gid + 3];

    const float PI = 3.14159265358979f;
    const float Z23 = powf(Zeff, 0.66666666666f);   // Z^(2/3) — constant
    const float Z2  = Zeff * Zeff;                   // Z^2     — constant

    for (int e = 0; e < n_el_per_thread; ++e) {
        const long long eidx = gid * (long long)n_el_per_thread + e;
        if (eidx >= n_electrons) return;

        // --- fresh electron state ---
        float cx = d0x, cy = d0y, cz = d0z;
        float E  = startE;
        float zdepth = 0.0f;

        // --- initial un-scattered beam step (EMMC.cl:224-232) ---
        {
            float alpha = 3.4e-3f * Z23 / E;
            float sigma = 5.21f * 602.2f * (Z2 / (E*E))
                        * (4.0f * PI / (alpha * (1.0f + alpha)))
                        * powf((511.0f + E) / (1024.0f + E), 2.0f);
            float mfp  = 1.0e7f * Aeff / (rho * sigma);
            float r1   = lfsr113(s);
            float step = -mfp * logf(r1);
            float de_ds = -0.00785f * (Zeff / (Aeff * E)) * logf(1.166f * E / J + 0.9911f);
            E = E + step * rho * de_ds;
            if (E < 1e-3f) E = 1e-3f;
            zdepth = step * cz0;   // z was 0; advance along un-deflected beam
        }

        // defaults if the electron never backscatters
        float ex_cx = 0.0f, ex_cy = 0.0f, ex_cz = 0.0f;
        float ex_depth = 0.0f, ex_energy = 0.0f;
        int   exited = 0;

        // --- scatter loop (per-thread register trajectory) ---
        for (int t = 0; t < n_max_steps; ++t) {
            // 1. screening parameter
            float alpha = 3.4e-3f * Z23 / E;
            // 2. screened-Rutherford cross-section
            float sigma = 5.21f * 602.2f * (Z2 / (E*E))
                        * (4.0f * PI / (alpha * (1.0f + alpha)))
                        * powf((511.0f + E) / (1024.0f + E), 2.0f);
            // 3. elastic mean free path [nm]
            float mfp = 1.0e7f * Aeff / (rho * sigma);
            // 4. step length ~ Exponential(mfp)
            float r1 = lfsr113(s);
            float step = -mfp * logf(r1);
            // 5. Joy-Luo Bethe energy loss rate
            float de_ds = -0.00785f * (Zeff / (Aeff * E)) * logf(1.166f * E / J + 0.9911f);
            // 6. polar scattering angle (screened-Rutherford inverse CDF)
            float r2 = lfsr113(s);
            float cphi = 1.0f - 2.0f * alpha * r2 / (1.0f + alpha - r2);
            if (cphi >  1.0f) cphi =  1.0f;
            if (cphi < -1.0f) cphi = -1.0f;
            float phi = acosf(cphi);
            // 7. azimuth
            float r3 = lfsr113(s);
            float psi = r3 * (2.0f * PI);

            // 8. direction-cosine update (Joy 3.12a-c) using OLD (cx,cy,cz)
            float sphi = sinf(phi), cph = cosf(phi);
            float cpsi = cosf(psi), spsi = sinf(psi);
            float ncx, ncy, ncz;
            if (fabsf(cz) > 0.99999f) {
                float sgn = (cz >= 0.0f) ? 1.0f : -1.0f;
                ncx = sphi * cpsi;
                ncy = sphi * spsi;
                ncz = sgn * cph;
            } else {
                float dsq = sqrtf(1.0f - cz*cz);
                if (dsq < 1e-30f) dsq = 1e-30f;
                float dsqi = 1.0f / dsq;
                ncx = sphi * (cx*cz*cpsi - cy*spsi) * dsqi + cx*cph;
                ncy = sphi * (cy*cz*cpsi + cx*spsi) * dsqi + cy*cph;
                ncz = -sphi * cpsi * dsq + cz*cph;
            }
            // renormalise (drift over many steps)
            float nrm = sqrtf(ncx*ncx + ncy*ncy + ncz*ncz);
            if (nrm < 1e-30f) nrm = 1e-30f;
            ncx /= nrm; ncy /= nrm; ncz /= nrm;

            // 9. advance depth (z) and 10. energy loss
            float z_new = zdepth + step * ncz;
            float E_new = E + step * rho * de_ds;
            if (E_new < 1e-3f) E_new = 1e-3f;

            // 11. exit test — backscattered when z_new < 0 (strict, PyTorch parity)
            if (z_new < 0.0f) {
                // escape_depth = |z_old / cz_new|  (old depth / new cz)
                float czs = (ncz != 0.0f) ? ncz : 1.0f;
                ex_depth  = fabsf(zdepth / czs);
                ex_energy = E_new;          // POST-step energy (EMsoft nint convention)
                ex_cx = ncx; ex_cy = ncy; ex_cz = ncz;
                exited = 1;
                break;
            }

            // 12. advance state
            cx = ncx; cy = ncy; cz = ncz;
            zdepth = z_new;
            E = E_new;
        }

        out_cx[eidx]     = ex_cx;
        out_cy[eidx]     = ex_cy;
        out_cz[eidx]     = ex_cz;
        out_depth[eidx]  = ex_depth;
        out_energy[eidx] = ex_energy;
        out_exited[eidx] = exited;
    }
}

}  // extern "C"
"""
