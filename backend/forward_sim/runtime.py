"""Device selection for the forward-sim package — fail-loud, no silent CPU.

Mirrors ``backend/spherical_gpu/runtime.py``'s philosophy: the forward
model is GPU-native, so we never silently fall back to CPU on a machine
without CUDA (that would be slow and surprising). Instead we raise a clear
``ForwardSimError`` and tell the caller how to opt into CPU explicitly.

Device resolution order (``get_device``):

1. If the ``FORWARD_SIM_DEVICE`` env var is set, honour it verbatim
   (e.g. ``cpu``, ``cuda``, ``cuda:1``). This is the escape hatch for
   CPU-only test/CI runs.
2. Else, if CUDA is available, use ``cuda``.
3. Else raise ``ForwardSimError`` — never a silent CPU fallback
   (per ``feedback_fail_loud_not_silent``).
"""
from __future__ import annotations

import os

import torch


class ForwardSimError(Exception):
    """Raised on a forward-sim contract/runtime violation (e.g. no CUDA)."""


def get_device() -> torch.device:
    """Return the torch device for forward-sim work.

    Honours the ``FORWARD_SIM_DEVICE`` env var if set; otherwise picks
    ``cuda`` when available; otherwise raises ``ForwardSimError`` (no
    silent CPU fallback).

    Returns
    -------
    torch.device
        The resolved compute device.

    Raises
    ------
    ForwardSimError
        If no CUDA device is available and ``FORWARD_SIM_DEVICE`` is unset.
    """
    override = os.environ.get("FORWARD_SIM_DEVICE")
    if override:
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    raise ForwardSimError(
        "no CUDA; set FORWARD_SIM_DEVICE=cpu to override"
    )


def resolve_device_adaptive() -> torch.device:
    """Return the torch device for the HARDWARE-ADAPTIVE "Ours" forward-sim engine.

    Unlike :func:`get_device` (which fail-loud raises ``ForwardSimError`` on a
    CPU-only host — the right behaviour for the GPU-native indexing paths), this
    resolver NEVER raises on a CPU-only machine: the "Ours" simulation engine is
    required to be self-contained and to run on a GPU OR a CPU with no env var.

    Resolution order:

    1. If ``FORWARD_SIM_DEVICE`` is set, honour it verbatim (the explicit
       escape hatch, e.g. ``cpu``, ``cuda``, ``cuda:1``).
    2. Else if CUDA is available, use ``cuda`` (the fast path).
    3. Else gracefully fall back to ``cpu`` (no exception — the numba/PyTorch
       CPU MC + CPU master build keep the engine usable without a GPU).

    Returns
    -------
    torch.device
        The resolved compute device — ``cuda`` when available, else ``cpu``.
    """
    override = os.environ.get("FORWARD_SIM_DEVICE")
    if override:
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
