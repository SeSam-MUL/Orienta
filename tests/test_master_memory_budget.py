"""The master build must size its direction batch to the machine it is on.

A Mac tester's silicon master at npx=500 was killed by the OS twice, ~6 % in, no
traceback and no MemoryError — the signature of a kernel kill rather than a
Python failure. Reproduced on a 64 GB Windows box: the same build peaks at
41-49 GB and killed the process outright at 12 workers.

A live-tensor census during a build put the peak in one place: the **depth
stack**, `(B', n, n, izz)` complex128, several workers holding one each.
`chunk_cells` bounds `B'·n²`, so one chunk is `chunk_cells · izz · 16` bytes —
at the old fixed 4e6 cells and izz≈101 that is 12 GB per worker, 289 GB implied
across 24. Nothing scaled it on a small machine: the value was a constant, and
the only RAM-aware cap in that file is CUDA-only, on the stated belief that
"CPU has host RAM headroom".

Measured after the fix, Si npx=250 on this box: peak 22.44 -> 11.78 GB, build
188 -> 139 s (smaller batches are also faster here), and the master is
bit-identical — `torch.equal`, not `allclose`.
"""
from __future__ import annotations

import pytest

from backend.forward_sim.dynamical.master_builder import auto_chunk_cells

GB = 1024 ** 3


def test_it_only_ever_shrinks_the_request():
    """The caller's value is an upper bound — a big machine must not inflate it."""
    assert auto_chunk_cells(101, 4, 500 * GB, 4_000_000) == 4_000_000
    assert auto_chunk_cells(101, 24, 8 * GB, 4_000_000) < 4_000_000


def test_the_tester_s_machine_gets_a_batch_that_fits():
    """8 GB free and 12 workers: the configuration that was killed."""
    cells = auto_chunk_cells(101, 12, 8 * GB, 4_000_000)
    per_worker = cells * 101 * 16 * 2
    assert per_worker * 12 <= 8 * GB * 0.5 + 1
    # and the old constant would not have fitted, which is the point
    assert 4_000_000 * 101 * 16 * 2 * 12 > 8 * GB


@pytest.mark.parametrize("izz", [1, 11, 101, 501])
def test_more_depth_steps_mean_a_smaller_batch(izz):
    """izz multiplies the depth stack, so it has to enter the budget."""
    cells = auto_chunk_cells(izz, 8, 16 * GB, 10_000_000)
    assert cells * izz * 16 * 2 * 8 <= 16 * GB * 0.5 + 1


def test_more_workers_mean_a_smaller_batch_each():
    """n_workers of them are in flight at once."""
    few = auto_chunk_cells(101, 2, 16 * GB, 10_000_000)
    many = auto_chunk_cells(101, 32, 16 * GB, 10_000_000)
    assert many < few
    assert many >= 1


def test_it_never_returns_zero_on_a_tiny_machine():
    """A batch of zero directions would hang, not fail — always at least one."""
    assert auto_chunk_cells(4001, 64, 64 * 1024 * 1024, 4_000_000) >= 1
    assert auto_chunk_cells(101, 24, 0, 4_000_000) >= 1


def test_the_estimate_matches_the_shape_it_is_about():
    """The budget must describe the tensor the census actually found.

    `(B', n, n, izz)` complex128 with `B'·n² = chunk_cells`, two copies live.
    If the shape or dtype of the depth stack ever changes, this is the number
    that has to change with it.
    """
    cells = auto_chunk_cells(101, 10, 20 * GB, 4_000_000)
    predicted_bytes_per_worker = cells * 101 * 16 * 2
    assert predicted_bytes_per_worker * 10 == pytest.approx(20 * GB * 0.5, rel=1e-3)
