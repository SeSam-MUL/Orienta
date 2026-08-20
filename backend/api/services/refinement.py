"""Joint R+PC refinement service — Stage 1 (per-pixel LM), Stage 2 (smoothness
solve), Stage 3 (R-only). See spec
docs/superpowers/specs/2026-05-14-joint-r-pc-refinement-design.md
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from backend.api.services.refinement_math import (
    so3_exp_to_quat,
    quat_multiply,
    build_neighbor_laplacian,
    disorientation_degrees,
)
from backend.api.services.sht_pattern_renderer import (
    load_or_get_phase,
    get_renderer,
    SHTRenderError,
)
from backend.api.services.forward_diagnostics import (
    _get_experimental_pattern,
    build_phase_id_2d,
    summary_stats,
)
from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft
from backend.api.services import state_version

logger = logging.getLogger(__name__)


# Stage 1 hyperparameters (spec section 4.3)
STAGE1_MAX_ITER = 20
STAGE1_LM_INIT = 1e-2
STAGE1_TOL_ABS = 1e-5
STAGE1_TOL_STEP = 1e-4
STAGE1_BOUND_OMEGA_RAD = 0.0873      # 5 degrees in radians
STAGE1_BOUND_DXY_PX = 10.0
STAGE1_BOUND_DL_UM = 1000.0

# Phase B v1.5: finite-difference step sizes for PC gradient (autograd through
# PC is broken because PatternRenderer.render casts pc_emsoft to Python floats,
# stripping the autograd graph -- same situation Phase A's
# compute_pc_sensitivity_map handles via finite-difference). Steps chosen
# small enough that the linearization is valid, large enough to dominate
# render noise.
FD_EPS_XY_PX = 0.5
FD_EPS_L_UM = 25.0


# Job registry — LRU with capacity (Task 5)
JOB_REGISTRY_CAPACITY = 8
JOB_REGISTRY: "OrderedDict[str, RefinementJob]" = OrderedDict()


class RefinementJob:
    """Tracks one in-flight or finished compute_full_refinement call.

    Lives in module-level JOB_REGISTRY (LRU cap 8). Mirrors the Phase A
    DiagnosticsJob pattern but with multi-stage progress.
    """

    def __init__(self, result_id: str, smoothness_lambda: float):
        self.job_id: str = uuid.uuid4().hex
        self.result_id: str = result_id
        self.smoothness_lambda: float = float(smoothness_lambda)
        self.state: str = "running"     # running | completed | failed | cancelled
        self.progress: float = 0.0
        self.current_stage: str = "stage_1"   # stage_1 | stage_2 | stage_3 | outer_iter_N
        self.outer_iter: int = 0
        self.pixels_done: int = 0
        self.pixels_total: int = 0
        self.error: Optional[str] = None
        self.cancel_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        _register_job(self)

    def set_progress(self, frac: float, current_stage: str = "", outer_iter: Optional[int] = None):
        self.progress = float(max(0.0, min(1.0, frac)))
        if current_stage:
            self.current_stage = current_stage
        if outer_iter is not None:
            self.outer_iter = int(outer_iter)

    def mark_completed(self):
        if self.cancel_event.is_set():
            self.state = "cancelled"
            return
        self.state = "completed"
        self.progress = 1.0

    def mark_failed(self, error_msg: str):
        if self.cancel_event.is_set():
            self.state = "cancelled"
            return
        self.state = "failed"
        self.error = error_msg

    def cancel(self):
        self.state = "cancelled"
        self.cancel_event.set()

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "outer_iter": self.outer_iter,
            "pixels_done": self.pixels_done,
            "pixels_total": self.pixels_total,
            "error": self.error,
            "result_id": self.result_id,
        }


def _register_job(job: "RefinementJob") -> None:
    JOB_REGISTRY[job.job_id] = job
    while len(JOB_REGISTRY) > JOB_REGISTRY_CAPACITY:
        JOB_REGISTRY.popitem(last=False)


def get_active_refinement_job(result_id: str) -> Optional["RefinementJob"]:
    for j in JOB_REGISTRY.values():
        if j.result_id == result_id and j.state == "running":
            return j
    return None


def _unit_normalize_batched(v: torch.Tensor) -> torch.Tensor:
    """Mean-subtract + L2-normalize along the last axis. Shape (B, ...)."""
    flat = v.reshape(v.shape[0], -1).to(torch.float64)
    flat = flat - flat.mean(dim=1, keepdim=True)
    n = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    n_safe = torch.where(n < 1e-12, torch.ones_like(n), n)
    return flat / n_safe


def _ncc_batched(I_exp: torch.Tensor, I_sim: torch.Tensor) -> torch.Tensor:
    """Per-pixel NCC: shape (B, H, W) inputs -> (B,) scalar NCC each."""
    e = _unit_normalize_batched(I_exp)   # (B, H*W)
    s = _unit_normalize_batched(I_sim)
    return (e * s).sum(dim=1)            # (B,)


def _render_serial_per_pc(
    renderer,
    grid,
    quats: torch.Tensor,
    pcs: torch.Tensor,
    detector_shape: tuple,
    pixel_size_um: float,
    tilt_deg: float,
    det_tilt_deg: float,
) -> torch.Tensor:
    """Render N patterns one (quaternion, PC) pair at a time.

    Workaround for ``PatternRenderer.render_batch`` requiring a single shared
    PC across the batch (Bug 1 from Phase B E2E). Returns an
    ``(N, H, W)`` float64 tensor stacked on the quats' device.

    ``tilt_deg`` is the SAMPLE tilt and ``det_tilt_deg`` the DETECTOR
    elevation; the renderer needs both (alpha = 90 - sample_tilt + det_tilt).

    Each per-pixel call goes through ``renderer.render(...)`` which expects
    a single quaternion of shape ``(4,)`` and a single PC tuple. The
    rendered tensor's autograd graph (through ``orientation_quat``) is
    preserved by ``torch.stack``, so Stage 1's ``torch.autograd.grad``
    against ``x_active[:, :3]`` (omega) still works. PC gradients do not
    flow through ``render`` because the renderer casts ``pc_emsoft`` to
    Python floats internally -- this matches the pre-existing behaviour of
    ``render_batch``.
    """
    n = int(quats.shape[0])
    if n == 0:
        H, W = int(detector_shape[0]), int(detector_shape[1])
        return torch.empty((0, H, W), dtype=torch.float64)
    # Detach PC values once (renderer casts them to Python floats anyway, so no
    # autograd flows through PC; detach silences the requires_grad warning).
    pcs_d = pcs.detach()
    out_list = []
    for i in range(n):
        pc_tuple = (float(pcs_d[i, 0]), float(pcs_d[i, 1]), float(pcs_d[i, 2]))
        pat = renderer.render(
            grid=grid,
            orientation_quat=quats[i],
            pc_emsoft=pc_tuple,
            detector_shape=detector_shape,
            pixel_size_um=pixel_size_um,
            tilt_deg=tilt_deg,
            det_tilt_deg=det_tilt_deg,
        )
        if not isinstance(pat, torch.Tensor):
            pat = torch.as_tensor(pat)
        if pat.ndim == 3 and pat.shape[0] == 1:
            pat = pat.squeeze(0)
        out_list.append(pat)
    stacked = torch.stack(out_list, dim=0)
    return stacked


def _compute_jp_finite_difference(
    *,
    renderer,
    grid,
    quats: torch.Tensor,             # (B, 4)
    pcs: torch.Tensor,                # (B, 3) — base PC per pixel
    I_exp: torch.Tensor,              # (B, H, W)
    detector_shape: tuple,
    pixel_size_um: float,
    tilt_deg: float,
    det_tilt_deg: float,
) -> torch.Tensor:                    # returns (B, 3) ∂r/∂PC where r = 1 - NCC
    """Central finite-difference Jacobian of the data residual r=1-NCC w.r.t. PC.

    This is the autograd-free path for the PC columns of Stage 1's Jacobian.
    Costs ``6`` renders per pixel (``3`` axes × ``2`` central-difference
    sides). Used both to seed the LM loop in ``refine_per_pixel`` (the
    "frozen J_p" strategy) and for the orchestrator's outer-loop
    relinearization step.

    Returns ``J_p`` with shape ``(B, 3)`` and dtype ``float64``. Operates
    under ``torch.no_grad()`` since the gradient is computed numerically.
    """
    B = int(quats.shape[0])
    J_p = torch.zeros(B, 3, dtype=torch.float64)
    if B == 0:
        return J_p
    epsilons = [FD_EPS_XY_PX, FD_EPS_XY_PX, FD_EPS_L_UM]
    quats_d = quats.detach()
    pcs_d = pcs.detach().to(torch.float64)
    I_exp_d = I_exp.detach()
    with torch.no_grad():
        for axis in range(3):
            eps = epsilons[axis]
            pcs_plus = pcs_d.clone()
            pcs_plus[:, axis] = pcs_plus[:, axis] + eps
            pcs_minus = pcs_d.clone()
            pcs_minus[:, axis] = pcs_minus[:, axis] - eps
            I_plus = _render_serial_per_pc(
                renderer, grid, quats_d, pcs_plus,
                detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
            )
            I_minus = _render_serial_per_pc(
                renderer, grid, quats_d, pcs_minus,
                detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
            )
            ncc_plus = _ncc_batched(I_exp_d, I_plus.to(torch.float64))
            ncc_minus = _ncc_batched(I_exp_d, I_minus.to(torch.float64))
            # r = 1 - NCC, so ∂r/∂PC = -∂NCC/∂PC
            J_p[:, axis] = -(ncc_plus - ncc_minus) / (2.0 * eps)
    return J_p


def refine_per_pixel(
    *,
    renderer,
    grid,
    I_exp: torch.Tensor,                  # (B, H, W) float64
    q0: torch.Tensor,                     # (B, 4) float64
    pc0: torch.Tensor,                    # (B, 3) float64
    detector_shape: tuple,
    pixel_size_um: float,
    tilt_deg: float,
    det_tilt_deg: float,
    max_iter: int = STAGE1_MAX_ITER,
    pc_only: bool = False,
) -> dict:
    """Stage 1: per-pixel 6-DOF Levenberg-Marquardt refinement.

    Returns dict with:
        omega:            (B, 3) delta omega in axis-angle (relative to q0)
        dpc:              (B, 3) Delta pc (relative to pc0)
        ncc_final:        (B,)   final NCC value
        iter_used:        (B,)   iteration count used per pixel
        convergence_flag: (B,)   0=failed, 1=converged, 2=max_iter
        J_p:              (B, 3) last Jacobian column dr/dPC
    """
    B = I_exp.shape[0]
    device = I_exp.device

    # Optimization variables: x = [omega, dpc] shape (B, 6)
    x = torch.zeros(B, 6, dtype=torch.float64, device=device, requires_grad=True)
    lm_lambda = torch.full((B,), STAGE1_LM_INIT, dtype=torch.float64, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    iter_used = torch.zeros(B, dtype=torch.int64, device=device)
    flag = torch.zeros(B, dtype=torch.int64, device=device)
    last_r = torch.full((B,), float("nan"), dtype=torch.float64, device=device)
    last_J_p = torch.zeros(B, 3, dtype=torch.float64, device=device)

    # Phase B v1.5: frozen finite-difference J_p. Computed once at the START
    # of the LM loop at the initial (q0, pc0) and reused as the PC columns of
    # the Jacobian for every iteration. Cheaper than recomputing every step
    # (6 renders per pixel total, vs. 6 per iter × max_iter), and accurate
    # enough because the LM step caps (5 px / 200 um) keep us inside the
    # region where ∂r/∂PC is roughly linear.
    #
    # The autograd path through PC is BROKEN (PatternRenderer.render casts
    # pc_emsoft to Python floats, stripping the graph) so without this fix
    # the PC columns of J are identically zero and Stage 2's solve degenerates.
    J_p_fixed = _compute_jp_finite_difference(
        renderer=renderer, grid=grid,
        quats=q0, pcs=pc0, I_exp=I_exp,
        detector_shape=detector_shape,
        pixel_size_um=pixel_size_um,
        tilt_deg=tilt_deg,
        det_tilt_deg=det_tilt_deg,
    )                                                # (B, 3) float64

    for it in range(max_iter):
        if not active.any():
            break
        idx_active = torch.where(active)[0]
        x_active = x[idx_active].detach().clone().requires_grad_(True)

        omega = x_active[:, :3]
        dpc = x_active[:, 3:]
        q_pert = so3_exp_to_quat(omega)                          # (B_act, 4)
        q_eff = quat_multiply(q0[idx_active], q_pert)            # (B_act, 4)
        pc_eff = pc0[idx_active] + dpc                           # (B_act, 3)

        # Render — autograd flows through if renderer.render is differentiable.
        # Per-pixel PC requires serial rendering (one render() per pixel) because
        # PatternRenderer.render_batch needs a single PC for all orientations,
        # but each Stage-1 pixel diverges in PC after iteration 1. For mocked
        # renderers the returned tensor has no grad and J falls back to zero.
        I_sim = _render_serial_per_pc(
            renderer, grid, q_eff, pc_eff,
            detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
        )
        if isinstance(I_sim, torch.Tensor):
            I_sim_t = I_sim.to(torch.float64)
            if not I_sim_t.requires_grad:
                # Mocked renderer — fabricate a constant zero-gradient situation
                I_sim_t = I_sim_t.clone()
        else:
            I_sim_t = torch.as_tensor(I_sim, dtype=torch.float64)

        r = 1.0 - _ncc_batched(I_exp[idx_active], I_sim_t)        # (B_act,)
        last_r[idx_active] = r.detach()

        # Jacobian via autograd: J[i] = dr[i]/dx_active[i]
        # Sum trick: d(sum r)/dx equals stacked dr[i]/dx[i] when x_active is per-pixel
        try:
            J = torch.autograd.grad(
                r.sum(), x_active, create_graph=False, retain_graph=False,
                allow_unused=True,
            )[0]
        except RuntimeError:
            J = None
        if J is None:
            # Mocked / non-differentiable path — use zero Jacobian (no update)
            J = torch.zeros_like(x_active)
        # J: (B_act, 6)

        # Phase B v1.5: replace the (broken-zero) PC columns of the autograd
        # Jacobian with the finite-difference values computed once at the
        # start of the LM loop. Omega columns keep their autograd-derived
        # values (omega gradient DOES flow through the renderer's rotation
        # math; only PC was broken by the float-cast).
        J = J.clone()
        J[:, 3:] = J_p_fixed[idx_active].to(J.dtype).to(J.device)
        if pc_only:
            # Freeze orientation: zero the omega Jacobian columns so the LM
            # step in omega is 0 and only PC moves. Without this, the clean
            # autograd omega gradient dominates the noisy finite-difference PC
            # gradient and the orientation simply ABSORBS the PC error
            # (omega rotates ~1.4 deg for a ~1 px PC offset while dpc stays ~0),
            # so the PC is never actually corrected. With R trusted from
            # indexing, pc_only makes per-pixel PC refinement well-posed.
            J[:, :3] = 0.0

        # LM step per pixel: proper Levenberg with diagonal approximation:
        #   step[i] = -J[i] * r / (J[i]^2 + lambda)
        # Earlier formulation `denom = JTJ * (1 + lambda)` reduced to
        # `step = -r / (J * (1 + lambda))`, which blows up whenever J is small
        # (typical for low-contrast PC gradients) and saturates the step bounds
        # on essentially every pixel.
        # NOTE on the detector-distance L column: L is in um and its per-unit
        # residual gradient is ~pixel_size (~287x) smaller than the px columns,
        # so the absolute damping (J^2 + lambda) keeps its per-pixel step small.
        # That is INTENTIONAL here — L is weakly constrained by a SINGLE pattern
        # (a large L offset barely changes NCC), so refining it per-pixel is
        # ill-posed and overshoots/wanders. L is instead pooled across pixels by
        # the Stage-2 spatial-smoothness solve (solve_smooth_pc), which averages
        # out the per-pixel L noise. Stage 1's job for L is to supply J_p[L].
        JTJ = J * J                                              # (B_act, 6)
        JTr = J * r.detach().unsqueeze(-1)                       # (B_act, 6)
        denom = JTJ + lm_lambda[idx_active].unsqueeze(-1)        # (B_act, 6)
        step = -JTr / (denom + 1e-12)                             # (B_act, 6)

        # Step bounds
        step_omega = step[:, :3]
        step_dpc = step[:, 3:]
        omega_norm = torch.linalg.vector_norm(step_omega, dim=-1, keepdim=True)
        bound_scale_o = torch.clamp(STAGE1_BOUND_OMEGA_RAD / (omega_norm + 1e-12), max=1.0)
        step_omega = step_omega * bound_scale_o
        step_dpc = torch.stack([
            torch.clamp(step_dpc[:, 0], -STAGE1_BOUND_DXY_PX, STAGE1_BOUND_DXY_PX),
            torch.clamp(step_dpc[:, 1], -STAGE1_BOUND_DXY_PX, STAGE1_BOUND_DXY_PX),
            torch.clamp(step_dpc[:, 2], -STAGE1_BOUND_DL_UM, STAGE1_BOUND_DL_UM),
        ], dim=-1)
        step_clipped = torch.cat([step_omega, step_dpc], dim=-1)

        # Evaluate at proposed x_new -- clamp CUMULATIVE x_new (not the step):
        # the step-bound version let pixels drift unboundedly across many
        # iterations of accepted steps. Clamping the proposed cumulative state
        # keeps every accepted x within the spec caps.
        x_new = x_active.detach() + step_clipped.detach()
        x_new_omega = x_new[:, :3]
        omega_cum_norm = torch.linalg.vector_norm(x_new_omega, dim=-1, keepdim=True)
        cum_scale_o = torch.clamp(STAGE1_BOUND_OMEGA_RAD / (omega_cum_norm + 1e-12), max=1.0)
        x_new_omega = x_new_omega * cum_scale_o
        x_new_dpc = torch.stack([
            torch.clamp(x_new[:, 3], -STAGE1_BOUND_DXY_PX, STAGE1_BOUND_DXY_PX),
            torch.clamp(x_new[:, 4], -STAGE1_BOUND_DXY_PX, STAGE1_BOUND_DXY_PX),
            torch.clamp(x_new[:, 5], -STAGE1_BOUND_DL_UM, STAGE1_BOUND_DL_UM),
        ], dim=-1)
        x_new = torch.cat([x_new_omega, x_new_dpc], dim=-1)

        q_pert_new = so3_exp_to_quat(x_new[:, :3])
        q_eff_new = quat_multiply(q0[idx_active], q_pert_new)
        pc_eff_new = pc0[idx_active] + x_new[:, 3:]
        I_sim_new = _render_serial_per_pc(
            renderer, grid, q_eff_new, pc_eff_new,
            detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
        )
        if isinstance(I_sim_new, torch.Tensor):
            I_sim_new_t = I_sim_new.to(torch.float64).detach()
        else:
            I_sim_new_t = torch.as_tensor(I_sim_new, dtype=torch.float64)
        r_new = 1.0 - _ncc_batched(I_exp[idx_active], I_sim_new_t)

        # Per-pixel accept/reject (allow no-change: tolerate r_new == r as
        # acceptable so that zero-Jacobian / already-optimal pixels deactivate).
        accept = r_new <= r.detach() + 1e-12
        # Update accepted
        x_data = x.detach()
        x_data[idx_active[accept]] = x_new[accept]
        x = x_data.clone().requires_grad_(True)
        lm_lambda[idx_active[accept]] *= 0.5
        lm_lambda[idx_active[~accept]] *= 2.0

        # Convergence checks
        delta_r = (r.detach() - r_new).abs()
        step_norm = torch.linalg.vector_norm(step_clipped.detach(), dim=-1)
        converged_now = (delta_r < STAGE1_TOL_ABS) | (step_norm < STAGE1_TOL_STEP)
        # Save last Jacobian PC column for newly-deactivating pixels. Phase B
        # v1.5: this is the frozen finite-difference J_p (we replaced the
        # broken autograd PC columns above), so the value handed off to
        # Stage 2 is the genuine ∂r/∂PC at the LM start point.
        last_J_p[idx_active] = J.detach()[:, 3:]

        deactivate = converged_now & accept
        flag[idx_active[deactivate]] = 1
        active[idx_active[deactivate]] = False
        iter_used[idx_active] += 1

    # Mark remaining active pixels as max_iter
    flag[active] = 2
    last_J_p_final = last_J_p

    # Final NCC for all pixels (re-render at converged x)
    with torch.no_grad():
        q_pert = so3_exp_to_quat(x.detach()[:, :3])
        q_eff = quat_multiply(q0, q_pert)
        pc_eff = pc0 + x.detach()[:, 3:]
        I_sim = _render_serial_per_pc(
            renderer, grid, q_eff, pc_eff,
            detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
        )
        I_sim_t = I_sim.to(torch.float64).detach()
        ncc_final = _ncc_batched(I_exp, I_sim_t)

    return {
        "omega": x.detach()[:, :3],
        "dpc":   x.detach()[:, 3:],
        "ncc_final": ncc_final,
        "iter_used": iter_used,
        "convergence_flag": flag,
        "J_p": last_J_p_final,
    }


def solve_smooth_pc(
    *,
    pc_stage1: np.ndarray,     # (N, 3) per-pixel PC from Stage 1
    r_stage1: np.ndarray,      # (N,)   per-pixel data residual
    J_p: np.ndarray,           # (N, 3) dr/dPC at pc_stage1
    mask: np.ndarray,          # (H, W) bool, True for indexed pixels
    lambda_: float,             # smoothness strength
) -> np.ndarray:
    """Stage 2: sparse Tikhonov smoothness solve, per-axis decoupled.

    For each axis (xpc, ypc, L) independently:
        min_pc  sum_i ( r[i] + j[i] * (pc[i] - pc_init[i]) )^2
              + lambda * sum_neighbors ||pc[i] - pc[j]||^2

    Closed-form linear system:
        (diag(j^2) + lambda * L) * pc = j * (j * pc_init - r)

    Per-axis degeneracy guard (Phase B v1.8): when ``median(|j[axis]|)``
    is at noise level, ``j`` is too small for the data term ``diag(j^2)``
    to constrain the rank-deficient ``lambda * Laplacian`` (Laplacian
    nullspace = constant vector), and the implied data-term displacement
    ``r/j`` blows up. spsolve then projects onto the Laplacian nullspace
    with arbitrary magnitude, and the orchestrator's hard-clamp saturates
    every pixel at the same Stage-1 L bound -- the HiGainNi
    ``pc_delta_l_um std=0 at -1000 um`` failure mode.

    Concrete numbers on HiGainNi 16x16 ROI after Stage 1:
        axis L: median|j| ~ 3.7e-5, max|j| ~ 1.6e-4
                ``r/j`` ~ 1.0 / 3.7e-5 ~ 27000 um (vs spec bound 1000 um)
        axis xpc: median|j| ~ 1.3e-2 (healthy)
        axis ypc: median|j| ~ 1.1e-2 (healthy)

    Criterion: ``median(|r/j|) > 10 * max_step`` declares the axis
    degenerate (the implied data correction is 10x larger than what the
    LM/Stage 2 bound would allow anyway). Falls back to ``median(|j|) <
    1e-3`` when ``r`` is near-zero (avoids 0/0 in the unit tests where
    r_stage1 = 0).

    Pre-existing global guard ``max(|J_p|) < 1e-6`` at the orchestrator
    only trips when ALL three axes are degenerate; doesn't catch the
    L-only degeneracy seen here.
    """
    n = pc_stage1.shape[0]
    assert n == int(mask.sum()), \
        f"pc_stage1 length {n} != indexed pixel count {int(mask.sum())}"
    L = build_neighbor_laplacian(mask)   # (n, n) sparse
    pc_refined = pc_stage1.copy()
    # Per-axis max-step from the Stage-1 bounds: xy in px, L in um.
    axis_bound = (STAGE1_BOUND_DXY_PX, STAGE1_BOUND_DXY_PX, STAGE1_BOUND_DL_UM)
    for axis in range(3):
        j = J_p[:, axis].astype(np.float64)            # (n,)
        r = np.asarray(r_stage1, dtype=np.float64)
        # Median |j| as the typical gradient magnitude (max can be a
        # single outlier and is unreliable).
        j_med = float(np.median(np.abs(j))) if n > 0 else 0.0
        j_max = float(np.max(np.abs(j))) if n > 0 else 0.0
        r_med = float(np.median(np.abs(r))) if r.size else 0.0
        bound = float(axis_bound[axis])
        # Implied per-pixel data-term correction magnitude r/j. If this
        # is much larger than the spec's Stage-1 bound for this axis,
        # the data term is asking for a step the LM cannot grant, so the
        # solve will be dominated by the smoothing term and produce
        # near-constant garbage. Tested first because it's the real
        # failure mode on HiGainNi.
        if j_med > 0 and r_med > 0:
            implied = r_med / j_med
            if implied > 10.0 * bound:
                logger.debug(
                    "solve_smooth_pc: axis %d degenerate "
                    "(median|r/j|=%.2e > 10*%g; max|j|=%.2e, median|j|=%.2e, "
                    "median|r|=%.2e); keeping pc_stage1[:, %d]",
                    axis, implied, bound, j_max, j_med, r_med, axis,
                )
                continue
        # Fallback: very small |j| in absolute terms (catches r=0 case
        # without spurious 0/0 division).
        if j_med < 1e-6:
            logger.debug(
                "solve_smooth_pc: axis %d degenerate (median|j|=%.2e < 1e-6); "
                "keeping pc_stage1[:, %d]",
                axis, j_med, axis,
            )
            continue
        logger.debug(
            "solve_smooth_pc: axis %d solving (max|j|=%.2e, median|j|=%.2e, "
            "median|r|=%.2e, lambda=%.2e)",
            axis, j_max, j_med, r_med, float(lambda_),
        )
        j_safe = np.where(np.abs(j) < 1e-8, 1e-8 * np.sign(j + 1e-12), j)
        # System: (diag(j^2) + lambda*L) * pc = j * (j * pc_stage1 - r)
        diag_j2 = j_safe ** 2
        A = diags(diag_j2) + lambda_ * L
        rhs = j_safe * (j_safe * pc_stage1[:, axis] - r_stage1)
        pc_refined[:, axis] = spsolve(A.tocsc(), rhs)
    return pc_refined


STAGE3_MAX_ITER = 30  # raised from 10 -- the per-pixel serial render path
                      # (Bug 1 fix) introduces enough LM step-size churn that
                      # 10 iterations of the bounded LM rarely reaches the
                      # spec's step-norm tolerance on real HiGainNi data.


def refine_r_only(
    *,
    renderer, grid,
    I_exp: torch.Tensor,             # (B, H, W)
    q0: torch.Tensor,                # (B, 4) — initial quaternion per pixel
    pc_fixed: torch.Tensor,          # (B, 3) — smoothed PC from Stage 2
    detector_shape: tuple,
    pixel_size_um: float,
    tilt_deg: float,
    det_tilt_deg: float,
    max_iter: int = STAGE3_MAX_ITER,
) -> dict:
    """Stage 3: R-only LM refinement with fixed per-pixel PC.

    Returns dict with:
        omega:            (B, 3) δω relative to q0
        ncc_final:        (B,)
        iter_used:        (B,)
        convergence_flag: (B,)
    """
    # Standalone 3-DOF version reusing _ncc_batched. PC is frozen at pc_fixed.
    B = I_exp.shape[0]
    omega = torch.zeros(B, 3, dtype=torch.float64, requires_grad=True)
    lm_lambda = torch.full((B,), STAGE1_LM_INIT, dtype=torch.float64)
    active = torch.ones(B, dtype=torch.bool)
    iter_used = torch.zeros(B, dtype=torch.int64)
    flag = torch.zeros(B, dtype=torch.int64)

    for it in range(max_iter):
        if not active.any():
            break
        idx_active = torch.where(active)[0]
        omega_active = omega[idx_active].detach().clone().requires_grad_(True)

        q_eff = quat_multiply(q0[idx_active], so3_exp_to_quat(omega_active))
        pc_eff = pc_fixed[idx_active]
        # Per-pixel PC: render one at a time (PatternRenderer.render_batch
        # requires a single shared PC for the whole batch).
        I_sim = _render_serial_per_pc(
            renderer, grid, q_eff, pc_eff,
            detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
        )
        I_sim_t = I_sim.to(torch.float64)
        r = 1.0 - _ncc_batched(I_exp[idx_active], I_sim_t)
        try:
            J = torch.autograd.grad(r.sum(), omega_active, allow_unused=True)[0]
        except RuntimeError:
            J = None
        if J is None:
            J = torch.zeros_like(omega_active)

        # Proper Levenberg: denom = J^2 + lambda (NOT J^2 * (1+lambda)).
        # See refine_per_pixel for the rationale.
        JTJ = J * J
        JTr = J * r.detach().unsqueeze(-1)
        denom = JTJ + lm_lambda[idx_active].unsqueeze(-1)
        step = -JTr / (denom + 1e-12)
        step_norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        bound_scale = torch.clamp(STAGE1_BOUND_OMEGA_RAD / (step_norm + 1e-12), max=1.0)
        step = step * bound_scale

        omega_new = omega_active.detach() + step.detach()
        # Per-stage Stage-3 omega magnitude bound: keep the LM trust-region
        # well-defined without forcing the iterate onto a hard sphere every
        # step (which destroys ``step_norm``-based convergence). The outer
        # loop in ``compute_full_refinement`` applies a slerp-based
        # CUMULATIVE cap on ``q_current`` against ``q_orig_per_indexed``
        # after each Stage-3 invocation -- that's where the spec's hard 5
        # deg cap is enforced.
        omega_step_norm = torch.linalg.vector_norm(step.detach(), dim=-1, keepdim=True)
        step_scale = torch.clamp(STAGE1_BOUND_OMEGA_RAD / (omega_step_norm + 1e-12), max=1.0)
        # Re-apply the step-only scale (no cumulative clamp here -- step
        # magnitude was already capped via ``bound_scale`` above, this is
        # defensive against any future refactor).
        omega_new = omega_active.detach() + step.detach() * step_scale

        q_eff_new = quat_multiply(q0[idx_active], so3_exp_to_quat(omega_new))
        I_sim_new = _render_serial_per_pc(
            renderer, grid, q_eff_new, pc_eff,
            detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
        )
        I_sim_new_t = I_sim_new.to(torch.float64).detach()
        r_new = 1.0 - _ncc_batched(I_exp[idx_active], I_sim_new_t)
        # Per-pixel accept/reject (allow no-change: tolerate r_new == r as
        # acceptable so that zero-Jacobian / already-optimal pixels deactivate).
        accept = r_new <= r.detach() + 1e-12

        omega_data = omega.detach()
        omega_data[idx_active[accept]] = omega_new[accept]
        omega = omega_data.clone().requires_grad_(True)
        lm_lambda[idx_active[accept]] *= 0.5
        lm_lambda[idx_active[~accept]] *= 2.0

        delta_r = (r.detach() - r_new).abs()
        sn = torch.linalg.vector_norm(step.detach(), dim=-1)
        converged_now = (delta_r < STAGE1_TOL_ABS) | (sn < STAGE1_TOL_STEP)
        deactivate = converged_now & accept
        flag[idx_active[deactivate]] = 1
        active[idx_active[deactivate]] = False
        iter_used[idx_active] += 1

    flag[active] = 2

    with torch.no_grad():
        q_eff = quat_multiply(q0, so3_exp_to_quat(omega.detach()))
        I_sim = _render_serial_per_pc(
            renderer, grid, q_eff, pc_fixed,
            detector_shape, pixel_size_um, tilt_deg, det_tilt_deg,
        )
        I_sim_t = I_sim.to(torch.float64).detach()
        ncc_final = _ncc_batched(I_exp, I_sim_t)

    return {
        "omega": omega.detach(),
        "ncc_final": ncc_final,
        "iter_used": iter_used,
        "convergence_flag": flag,
    }


# ---------------------------------------------------------------------------
# Orchestrator: compute_full_refinement
# ---------------------------------------------------------------------------

import datetime as _dt

MAX_OUTER = 3
TOL_PC_xy_px = 0.1
TOL_PC_L_um = 5.0


class _RefinementCancelledError(RuntimeError):
    """Internal sentinel raised when cancel_event was set."""


class _RefinedResult:
    """Lightweight result wrapper that duck-types as an IndexingResult.

    The full orix CrystalMap reconstruction is heavy and not required for
    downstream consumers — they only need xmap.rotations.data, xmap.phase_id,
    xmap.shape, xmap.phases, original_shape, selection_mask, and metadata.
    """


class _RefinedXMap:
    """Duck-typed xmap for refined results."""


class _RefinedRotation:
    """Duck-typed rotations container for refined results."""


def _gather_experimental_patterns(result, flat_indices, n_cols, H_pat, W_pat):
    """Build an (M, H, W) float64 tensor of experimental patterns.

    Missing patterns are replaced by zeros (logged once per orchestrator run
    by the caller would be ideal — we silently fill here for batch flow).
    """
    patterns = []
    for flat in flat_indices:
        r = int(flat // n_cols)
        c = int(flat % n_cols)
        p = _get_experimental_pattern(result, r, c)
        if p is None:
            patterns.append(np.zeros((H_pat, W_pat), dtype=np.float32))
        else:
            patterns.append(np.asarray(p, dtype=np.float32))
    return torch.tensor(np.stack(patterns), dtype=torch.float64)


def compute_full_refinement(
    result,
    smoothness_lambda: float,
    *,
    progress_callback=None,
    cancel_event=None,
    max_outer_iter: int = MAX_OUTER,
    chunk_size: int = 128,
    force_full_recompute: bool = False,
) -> None:
    """Run all three refinement stages and write metadata atomically.

    Writes to ``result.metadata["refinement"]`` and
    ``result.metadata["_refinement_stage1_cache"]`` ONLY on successful
    completion. Also registers a separate refined ``IndexingResult``-like
    object in ``backend.api.routes.indexing._result_registry`` via
    ``_store_result(refined_result, "refinement")``. The id of that refined
    result is stored as ``refinement.refined_result_id``.

    On exception or cancellation, leaves metadata unchanged.

    See spec section 4 for the algorithm.
    """
    metadata = getattr(result, "metadata", None) or {}
    method = metadata.get("indexing_method")
    if method not in ("spherical", "dictionary"):
        raise SHTRenderError(
            f"refinement requires spherical or dictionary indexing (got {method!r})"
        )
    sht_map = metadata.get("sht_paths_by_phase") or {}
    if not sht_map:
        raise SHTRenderError(
            "result.metadata['sht_paths_by_phase'] is empty -- re-run indexing"
        )
    if not (smoothness_lambda >= 0 and np.isfinite(smoothness_lambda)):
        raise ValueError(
            f"smoothness_lambda must be >= 0 and finite, got {smoothness_lambda}"
        )

    det = metadata.get("detector_geometry") or {}
    n_rows, n_cols = result.original_shape
    H_pat = int(det["pat_height"])
    W_pat = int(det["pat_width"])
    pixel_size_um = float(det.get("pixel_size", 70.0))
    sample_tilt = float(det.get("sample_tilt", 70.0))
    # ``tilt`` is the DETECTOR elevation, a separate angle from the sample
    # tilt (the renderer uses alpha = 90 - sample_tilt + det_tilt). Omitting
    # it renders every pattern rotated by det.tilt, so the LM residual is
    # dominated by a geometry error the solver cannot fix and refinement
    # walks the orientation away from the correct answer.
    det_tilt = float(det.get("tilt", 0.0))
    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
        vendor=str(det.get("vendor", "Bruker")),
        pat_width=W_pat, pat_height=H_pat,
        pixel_size=pixel_size_um, binning=int(det.get("binning", 1)),
    )
    pc_global = np.array([float(xpc), float(ypc), float(L_um)], dtype=np.float64)

    # Collect indexed pixels
    phase_ids_2d = build_phase_id_2d(result)
    indexed_mask = phase_ids_2d >= 0
    indexed_flat_idx = np.where(indexed_mask.ravel())[0]
    n_indexed = int(indexed_flat_idx.size)
    if n_indexed == 0:
        raise SHTRenderError("no indexed pixels to refine")

    # Flat-to-xmap-row mapping (handles both selection_mask and full-grid cases)
    n_total = n_rows * n_cols
    if int(result.xmap.rotations.size) == n_total:
        flat_to_xmap = np.arange(n_total)
    else:
        flat_to_xmap = np.cumsum(indexed_mask.ravel()) - 1
    phase_id_per_pixel = phase_ids_2d.ravel()[indexed_flat_idx]

    # ------------------------------------------------------------------
    # Stage 1: per-pixel LM (per-phase chunked) — or reuse cache
    # ------------------------------------------------------------------
    renderer = get_renderer()
    stage1_t0 = _dt.datetime.utcnow()
    cache = metadata.get("_refinement_stage1_cache") if not force_full_recompute else None

    if cache is not None and cache.get("bandwidth") is not None:
        stage1_data = cache
        stage_1_seconds = 0.0
    else:
        from orix.quaternion import Rotation

        pc_stage1 = np.zeros((n_indexed, 3), dtype=np.float64)
        r_stage1 = np.zeros(n_indexed, dtype=np.float64)
        J_p_stage1 = np.zeros((n_indexed, 3), dtype=np.float64)
        q_stage1 = np.zeros((n_indexed, 4), dtype=np.float64)
        flag_stage1 = np.zeros(n_indexed, dtype=np.int64)
        pc_global_t = torch.tensor(pc_global, dtype=torch.float64).unsqueeze(0)
        eulers_all = np.asarray(result.xmap.rotations.to_euler())

        unique_phases = np.unique(phase_id_per_pixel)
        for phase_id_int in unique_phases:
            if cancel_event is not None and cancel_event.is_set():
                raise _RefinementCancelledError()
            phase_id = int(phase_id_int)
            sht_path = sht_map.get(phase_id)
            if sht_path is None:
                continue
            try:
                grid = load_or_get_phase(str(sht_path), max_bandwidth=256)
            except Exception as e:
                logger.warning("SHT for phase %s unavailable: %s", phase_id, e)
                continue

            ph_local_idx = np.where(phase_id_per_pixel == phase_id)[0]
            ph_global_flat = indexed_flat_idx[ph_local_idx]
            ph_xmap_rows = flat_to_xmap[ph_global_flat]

            eulers = eulers_all[ph_xmap_rows]
            quats_np = Rotation.from_euler(eulers).data
            q0 = torch.tensor(np.asarray(quats_np)[:, :4], dtype=torch.float64)
            pc0_batch = pc_global_t.expand(q0.shape[0], 3).clone()
            I_exp_t = _gather_experimental_patterns(
                result, ph_global_flat, n_cols, H_pat, W_pat
            )

            M = q0.shape[0]
            done = 0
            for cs in range(0, M, chunk_size):
                if cancel_event is not None and cancel_event.is_set():
                    raise _RefinementCancelledError()
                ce = min(cs + chunk_size, M)
                out = refine_per_pixel(
                    renderer=renderer, grid=grid,
                    I_exp=I_exp_t[cs:ce],
                    q0=q0[cs:ce], pc0=pc0_batch[cs:ce],
                    detector_shape=(H_pat, W_pat),
                    pixel_size_um=pixel_size_um, tilt_deg=sample_tilt,
                    det_tilt_deg=det_tilt,
                    max_iter=STAGE1_MAX_ITER,
                )
                local_chunk_idx = ph_local_idx[cs:ce]
                q_pert = so3_exp_to_quat(out["omega"])
                q_final = quat_multiply(q0[cs:ce], q_pert)
                q_stage1[local_chunk_idx] = q_final.numpy()
                pc_stage1[local_chunk_idx] = (pc0_batch[cs:ce] + out["dpc"]).numpy()
                r_stage1[local_chunk_idx] = (1.0 - out["ncc_final"]).numpy()
                J_p_stage1[local_chunk_idx] = out["J_p"].numpy()
                flag_stage1[local_chunk_idx] = out["convergence_flag"].numpy()
                done += (ce - cs)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            min(0.5, 0.5 * done / max(1, n_indexed)),
                            current_stage="stage_1",
                            outer_iter=0,
                        )
                    except Exception:
                        pass

        stage1_data = {
            "pc_stage1": pc_stage1,
            "r_stage1": r_stage1,
            "J_p": J_p_stage1,
            "q_stage1": q_stage1,
            "flag_stage1": flag_stage1,
            "phase_id_per_pixel": phase_id_per_pixel,
            "indexed_flat_idx": indexed_flat_idx,
            "bandwidth": 256,
        }
        stage_1_seconds = (_dt.datetime.utcnow() - stage1_t0).total_seconds()

    # ------------------------------------------------------------------
    # Outer loop: Stage 2 (smoothness) + Stage 3 (R-only) with convergence
    # ------------------------------------------------------------------
    stage_2_t = 0.0
    stage_3_t = 0.0
    pc_current = stage1_data["pc_stage1"].copy()
    q_current = stage1_data["q_stage1"].copy()
    r_current = stage1_data["r_stage1"].copy()
    J_p_current = stage1_data["J_p"].copy()
    flag_current = stage1_data["flag_stage1"].copy()
    # Pre-init so post-loop scatter sees something even if max_outer_iter==0
    ncc_final_all = np.full(n_indexed, np.nan, dtype=np.float64)
    outer_iter = 0

    # Cumulative orientation cap: even though each Stage-3 invocation clips
    # omega to STAGE1_BOUND_OMEGA_RAD, drift accumulates across outer
    # iterations as we compose q_current = q_current * exp(omega). Cap the
    # TOTAL disorientation against the original Stage-0 quaternion at the
    # same 5 deg bound (spec section 4.3, mirrored by tests/test_refinement_e2e.py).
    q_orig_all_flat = np.asarray(result.xmap.rotations.data)
    xmap_rows_for_cap = flat_to_xmap[indexed_flat_idx]
    q_orig_per_indexed_np = q_orig_all_flat[xmap_rows_for_cap]
    # The hyperparameter ``STAGE1_BOUND_OMEGA_RAD = 0.0873`` is the rounded
    # radian value of 5 deg; rad2deg(0.0873) = 5.00191... deg which would
    # let post-loop p95 land just above the spec's 5 deg cap. Use exact 5 deg
    # for the cumulative cap so the spec bound holds tightly.
    STAGE3_CUM_BOUND_DEG = 5.0

    for outer_iter in range(max_outer_iter):
        if cancel_event is not None and cancel_event.is_set():
            raise _RefinementCancelledError()

        # Stage 2: smooth-PC solve (global)
        s2_t0 = _dt.datetime.utcnow()
        # Per-axis diagnostic at debug level (Phase B v1.8): noisy at info
        # level because it fires 3*max_outer_iter times. Use logging.DEBUG
        # if you need it during incident debugging.
        if logger.isEnabledFor(logging.DEBUG):
            for axis_i, axis_name in enumerate(("xpc", "ypc", "L")):
                jp_col = J_p_current[:, axis_i]
                pc_col = pc_current[:, axis_i]
                logger.debug(
                    "outer_iter=%d  pre-Stage2 axis=%s  "
                    "J_p: max|j|=%.3e, median|j|=%.3e  "
                    "pc: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                    outer_iter, axis_name,
                    float(np.max(np.abs(jp_col))) if jp_col.size else 0.0,
                    float(np.median(np.abs(jp_col))) if jp_col.size else 0.0,
                    float(pc_col.mean()), float(pc_col.std()),
                    float(pc_col.min()), float(pc_col.max()),
                )
        # Degeneracy guard: when J_p is essentially zero -- which happens
        # whenever Stage 1's autograd cannot flow through PC (e.g. because
        # the renderer's float-cast of pc_emsoft strips the graph) -- the
        # sparse system in ``solve_smooth_pc`` is dominated by the
        # rank-deficient ``lambda * Laplacian`` and spsolve produces garbage
        # of magnitude ~1e8. In that case skip the solve and keep
        # pc_current; subsequent stages still benefit from Stage 1's
        # pc_stage1 values.
        jp_max = float(np.max(np.abs(J_p_current))) if J_p_current.size else 0.0
        if jp_max < 1e-6:
            pc_new = pc_current.copy()
        else:
            pc_new = solve_smooth_pc(
                pc_stage1=pc_current, r_stage1=r_current, J_p=J_p_current,
                mask=indexed_mask, lambda_=smoothness_lambda,
            )
        # Hard-clamp pc_new to the spec's Stage-1 bounds around pc_global.
        # Defends against any remaining ill-conditioned drift in the rare
        # non-degenerate-but-noisy case.
        pc_new[:, 0] = np.clip(
            pc_new[:, 0],
            pc_global[0] - STAGE1_BOUND_DXY_PX,
            pc_global[0] + STAGE1_BOUND_DXY_PX,
        )
        pc_new[:, 1] = np.clip(
            pc_new[:, 1],
            pc_global[1] - STAGE1_BOUND_DXY_PX,
            pc_global[1] + STAGE1_BOUND_DXY_PX,
        )
        pc_new[:, 2] = np.clip(
            pc_new[:, 2],
            pc_global[2] - STAGE1_BOUND_DL_UM,
            pc_global[2] + STAGE1_BOUND_DL_UM,
        )
        stage_2_t += (_dt.datetime.utcnow() - s2_t0).total_seconds()

        # Stage 3: R-only LM per-phase using fixed PC from Stage 2
        s3_t0 = _dt.datetime.utcnow()
        ncc_final_all = np.zeros(n_indexed, dtype=np.float64)
        omega_all = np.zeros((n_indexed, 3), dtype=np.float64)
        flag_all = flag_current.copy()
        for phase_id_int in np.unique(stage1_data["phase_id_per_pixel"]):
            phase_id = int(phase_id_int)
            sht_path = sht_map.get(phase_id)
            if sht_path is None:
                continue
            try:
                grid = load_or_get_phase(str(sht_path), max_bandwidth=256)
            except Exception as e:
                logger.warning("SHT for phase %s unavailable in Stage 3: %s", phase_id, e)
                continue

            ph_local_idx = np.where(stage1_data["phase_id_per_pixel"] == phase_id)[0]
            ph_global_flat = stage1_data["indexed_flat_idx"][ph_local_idx]

            q0_phase = torch.tensor(q_current[ph_local_idx], dtype=torch.float64)
            pc_fixed_phase = torch.tensor(pc_new[ph_local_idx], dtype=torch.float64)
            I_exp_t = _gather_experimental_patterns(
                result, ph_global_flat, n_cols, H_pat, W_pat
            )

            M = q0_phase.shape[0]
            for cs in range(0, M, chunk_size):
                if cancel_event is not None and cancel_event.is_set():
                    raise _RefinementCancelledError()
                ce = min(cs + chunk_size, M)
                out = refine_r_only(
                    renderer=renderer, grid=grid,
                    I_exp=I_exp_t[cs:ce],
                    q0=q0_phase[cs:ce], pc_fixed=pc_fixed_phase[cs:ce],
                    detector_shape=(H_pat, W_pat),
                    pixel_size_um=pixel_size_um, tilt_deg=sample_tilt,
                    det_tilt_deg=det_tilt,
                )
                local_chunk_idx = ph_local_idx[cs:ce]
                ncc_final_all[local_chunk_idx] = out["ncc_final"].numpy()
                omega_all[local_chunk_idx] = out["omega"].numpy()
                flag_all[local_chunk_idx] = out["convergence_flag"].numpy()

        # Apply Stage 3 omega to q_current
        q_omega_t = so3_exp_to_quat(torch.tensor(omega_all, dtype=torch.float64))
        q_current_t = torch.tensor(q_current, dtype=torch.float64)
        q_current = quat_multiply(q_current_t, q_omega_t).numpy()

        # Cumulative orientation cap (Bug 2 corollary, post-Stage-3): clamp
        # the total disorientation against q_orig back to the spec bound. We
        # slerp q_current toward q_orig along the shorter great circle so each
        # over-cap pixel ends exactly at the bound, preserving the rotation
        # direction Stage 3 chose. Pixels within bound are untouched.
        q_orig_t = torch.tensor(q_orig_per_indexed_np, dtype=torch.float64)
        q_curr_t = torch.tensor(q_current, dtype=torch.float64)
        # Ensure positive dot (choose closer hemisphere) for stable slerp.
        dot_t = (q_orig_t * q_curr_t).sum(dim=-1, keepdim=True)
        q_curr_aligned = torch.where(dot_t < 0, -q_curr_t, q_curr_t)
        dot_aligned = torch.clamp((q_orig_t * q_curr_aligned).sum(dim=-1), -1.0, 1.0)
        half_ang = torch.acos(dot_aligned)                          # [0, pi/2]
        full_deg = torch.rad2deg(2.0 * half_ang)
        bound_half_ang = torch.deg2rad(torch.tensor(STAGE3_CUM_BOUND_DEG, dtype=torch.float64) * 0.5)
        # Allow a small numerical slack so identical-quaternion pixels stay put.
        over = full_deg > (STAGE3_CUM_BOUND_DEG + 1e-9)
        if bool(over.any()):
            t = torch.where(half_ang > 1e-12, bound_half_ang / half_ang, torch.zeros_like(half_ang))
            t = t.clamp(0.0, 1.0)
            sin_half = torch.sin(half_ang).clamp_min(1e-12)
            w_o = (torch.sin((1.0 - t) * half_ang) / sin_half).unsqueeze(-1)
            w_c = (torch.sin(t * half_ang) / sin_half).unsqueeze(-1)
            q_slerp = w_o * q_orig_t + w_c * q_curr_aligned
            # Renormalise (slerp output already unit up to fp noise).
            q_slerp = q_slerp / torch.linalg.vector_norm(q_slerp, dim=-1, keepdim=True).clamp_min(1e-30)
            q_curr_capped = torch.where(over.unsqueeze(-1), q_slerp, q_curr_aligned)
            q_current = q_curr_capped.numpy()

        flag_current = flag_all
        r_current = 1.0 - ncc_final_all
        stage_3_t += (_dt.datetime.utcnow() - s3_t0).total_seconds()

        # Convergence check (normalised against per-axis tolerances)
        delta = max(
            float(np.max(np.abs(pc_new[:, 0] - pc_current[:, 0]))) / TOL_PC_xy_px,
            float(np.max(np.abs(pc_new[:, 1] - pc_current[:, 1]))) / TOL_PC_xy_px,
            float(np.max(np.abs(pc_new[:, 2] - pc_current[:, 2]))) / TOL_PC_L_um,
        )
        pc_current = pc_new
        if progress_callback is not None:
            try:
                progress_callback(
                    min(1.0, 0.5 + 0.5 * (outer_iter + 1) / max_outer_iter),
                    current_stage=f"outer_iter_{outer_iter + 1}",
                    outer_iter=outer_iter + 1,
                )
            except Exception:
                pass
        if delta < 1.0:
            break

    # ------------------------------------------------------------------
    # Scatter per-pixel results into (H, W) maps with NaN for unindexed
    # ------------------------------------------------------------------
    def _scatter_map(values_flat, dtype=np.float32, fill=np.nan):
        m = np.full((n_rows, n_cols), fill, dtype=dtype)
        m.ravel()[indexed_flat_idx] = np.asarray(values_flat, dtype=dtype)
        return m

    ncc_map = _scatter_map(ncc_final_all)
    conv_map = _scatter_map(flag_current.astype(np.float32))
    pc_delta = pc_current - pc_global
    pc_dx_map = _scatter_map(pc_delta[:, 0])
    pc_dy_map = _scatter_map(pc_delta[:, 1])
    pc_dl_map = _scatter_map(pc_delta[:, 2])

    # Orientation delta in degrees
    q_orig_all = np.asarray(result.xmap.rotations.data)
    xmap_rows = flat_to_xmap[indexed_flat_idx]
    q_orig_per_indexed = q_orig_all[xmap_rows]
    orientation_delta_deg = disorientation_degrees(
        torch.tensor(q_orig_per_indexed, dtype=torch.float64),
        torch.tensor(q_current, dtype=torch.float64),
    ).numpy()
    orientation_delta_map = _scatter_map(orientation_delta_deg.astype(np.float32))

    # ------------------------------------------------------------------
    # Build refined result and register it
    # ------------------------------------------------------------------
    refined_xmap_rotations_data = q_orig_all.copy()
    refined_xmap_rotations_data[xmap_rows] = q_current

    refined_xmap = _RefinedXMap()
    refined_xmap.phase_id = result.xmap.phase_id
    refined_xmap.shape = result.xmap.shape
    refined_xmap.phases = result.xmap.phases
    refined_xmap.scores = _scatter_map(ncc_final_all.astype(np.float32))

    refined_rot = _RefinedRotation()
    refined_rot.size = q_orig_all.shape[0]
    refined_rot.data = refined_xmap_rotations_data
    refined_rot.to_euler = result.xmap.rotations.to_euler  # reuse upstream impl
    refined_xmap.rotations = refined_rot

    pc_per_pixel = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    pc_per_pixel.reshape(-1, 3)[indexed_flat_idx] = pc_current

    refined_result = _RefinedResult()
    refined_result.xmap = refined_xmap
    refined_result.original_shape = result.original_shape
    refined_result.selection_mask = result.selection_mask

    # Find original result's id in the registry (needed for refinement_of)
    from backend.api.routes import indexing as ind_routes
    original_rid = None
    for rid_key, rid_val in ind_routes._result_registry.items():
        if rid_val is result:
            original_rid = rid_key
            break

    refined_result.metadata = {
        "indexing_method": method,
        "is_refined": True,
        "refinement_of": original_rid,
        "refinement_smoothness_lambda": float(smoothness_lambda),
        "sht_paths_by_phase": dict(sht_map),
        "detector_geometry": dict(det),
        "pc_per_pixel": pc_per_pixel,
    }
    # A refinement inherits its parent's origin by definition: same pixels,
    # same measurement. Copied off the parent rather than left to
    # _store_result's setdefault, which reads the crop LIVE — clear or change
    # the crop between indexing and refining and parent and child would report
    # different origins for the very same pixels.
    parent_meta = getattr(result, "metadata", None)
    if not isinstance(parent_meta, dict):
        parent_meta = {}
    for _key, _default in ind_routes.NO_SCAN_OFFSET.items():
        refined_result.metadata[_key] = parent_meta.get(_key, _default)

    refined_rid = ind_routes._store_result(refined_result, "refinement")
    # _store_result already bumps; bump again is harmless but the explicit
    # call here keeps the refinement path self-documenting as a state change.
    state_version.bump()

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def _safe_summary(arr):
        """summary_stats wrapper that survives float32 collapsed-range bug.

        np.histogram raises 'Too many bins for data range' when all valid
        values are identical at high magnitude (float32 precision collapses
        bin edges). Fall back to a degenerate histogram in that case.
        """
        try:
            return summary_stats(arr)
        except ValueError:
            arr64 = np.asarray(arr, dtype=np.float64)
            valid = arr64[~np.isnan(arr64)]
            if valid.size == 0:
                v_mean = v_std = v_min = v_max = v_p05 = v_p95 = float("nan")
            else:
                v_mean = float(valid.mean())
                v_std = float(valid.std())
                v_min = float(valid.min())
                v_max = float(valid.max())
                v_p05 = float(np.percentile(valid, 5))
                v_p95 = float(np.percentile(valid, 95))
            return {
                "mean": v_mean, "std": v_std,
                "min": v_min, "max": v_max,
                "p05": v_p05, "p95": v_p95,
                "histogram": {
                    "bins": [v_min, v_max] if valid.size else [0.0, 0.0],
                    "counts": [int(valid.size)] if valid.size else [0],
                },
            }

    prior_ncc = (metadata.get("forward_diagnostics") or {}).get("ncc_map")
    if prior_ncc is None or not hasattr(prior_ncc, "shape") or prior_ncc.shape != ncc_map.shape:
        ncc_improvement = _safe_summary(np.full_like(ncc_map, np.nan))
    else:
        ncc_improvement = _safe_summary(ncc_map - prior_ncc)

    summary = {
        "n_pixels_refined": int(n_indexed),
        "n_converged": int((flag_current == 1).sum()),
        "n_max_iter": int((flag_current == 2).sum()),
        "n_failed": int((flag_current == 0).sum()),
        "ncc_improvement": ncc_improvement,
        "orientation_delta_deg": _safe_summary(orientation_delta_map),
        "pc_delta_x_px": _safe_summary(pc_dx_map),
        "pc_delta_y_px": _safe_summary(pc_dy_map),
        "pc_delta_l_um": _safe_summary(pc_dl_map),
    }

    # ------------------------------------------------------------------
    # Atomic write (last step — keeps result.metadata pristine on error)
    # ------------------------------------------------------------------
    block = {
        "refined_result_id": refined_rid,
        "smoothness_lambda": float(smoothness_lambda),
        "computed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "stage_1_seconds": stage_1_seconds,
        "stage_2_seconds": stage_2_t,
        "stage_3_seconds": stage_3_t,
        "outer_iterations": outer_iter + 1,
        "refined_ncc_map": ncc_map,
        "convergence_status_map": conv_map,
        "orientation_delta_map": orientation_delta_map,
        "pc_delta_x_map": pc_dx_map,
        "pc_delta_y_map": pc_dy_map,
        "pc_delta_l_map": pc_dl_map,
        "summary": summary,
    }
    cache_block = {
        "pc_stage1": stage1_data["pc_stage1"],
        "r_stage1": stage1_data["r_stage1"],
        "J_p": stage1_data["J_p"],
        "q_stage1": stage1_data["q_stage1"],
        "flag_stage1": stage1_data["flag_stage1"],
        "phase_id_per_pixel": stage1_data["phase_id_per_pixel"],
        "indexed_flat_idx": stage1_data["indexed_flat_idx"],
        "bandwidth": stage1_data["bandwidth"],
    }
    result.metadata["refinement"] = block
    result.metadata["_refinement_stage1_cache"] = cache_block
