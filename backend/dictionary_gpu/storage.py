"""Write generated dictionaries as kikuchipy-readable h5 in chunks.

We use kikuchipy's preferred 'h5ebsd' / 'kikuchipy_h5ebsd' layout via
HyperSpy's signal save API on the final array. To avoid holding the whole
dict in RAM, we accumulate to a memory-mapped numpy file, then save once.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class ChunkedDictionaryWriter:
    def __init__(
        self,
        output_path: str,
        n_total: int,
        detector_shape: Tuple[int, int],
        dtype: np.dtype = np.float32,
    ):
        self.output_path = Path(output_path)
        self.n_total = n_total
        self.H, self.W = detector_shape
        self.dtype = np.dtype(dtype)

        # Memory-mapped staging file on disk to avoid RAM blowup
        self._tmp = tempfile.NamedTemporaryFile(
            prefix="dictgpu_", suffix=".npy", delete=False
        )
        self._tmp.close()
        self._mmap = np.memmap(
            self._tmp.name,
            mode="w+",
            dtype=self.dtype,
            shape=(n_total, self.H, self.W),
        )
        self._cursor = 0
        self._closed = False

    def append(self, batch: torch.Tensor) -> None:
        if batch.dim() != 3 or batch.shape[1:] != (self.H, self.W):
            raise ValueError(
                f"chunk shape {tuple(batch.shape)} incompatible with detector "
                f"({self.H}, {self.W})"
            )
        n = batch.shape[0]
        if self._cursor + n > self.n_total:
            raise ValueError("appended more patterns than reserved n_total")
        np_batch = batch.detach().cpu().numpy().astype(self.dtype, copy=False)
        self._mmap[self._cursor : self._cursor + n] = np_batch
        self._cursor += n

    def finalize(self, rotations=None, phase=None) -> Path:
        """Write the dictionary, WITH the orientation each pattern was simulated at.

        Parameters
        ----------
        rotations : orix.quaternion.Rotation
            One rotation per pattern, in the same order as they were appended.
            Required: dictionary indexing recovers an orientation by mapping the
            best-matching pattern INDEX through this list, so a dictionary
            without it can only ever report one constant orientation for the
            whole map. Saving a bare ``kp.signals.EBSD(data)`` used to do
            exactly that — kikuchipy synthesises N identity rotations on load,
            which looks like a valid xmap and is silently useless.
        phase : orix.crystal_map.Phase, optional
            Phase of the simulated patterns. Carries the point group that IPF
            colouring and ``pc.phase_list`` need downstream.
        """
        if self._cursor != self.n_total:
            raise ValueError(
                f"finalize called with {self._cursor}/{self.n_total} patterns written"
            )
        if rotations is None:
            raise ValueError(
                "finalize() needs the rotations the patterns were simulated at — "
                "a dictionary without them cannot be used for indexing."
            )
        if rotations.size != self.n_total:
            raise ValueError(
                f"rotation count {rotations.size} does not match the "
                f"{self.n_total} patterns written — every match would be mapped "
                "to the wrong orientation."
            )
        self._mmap.flush()

        # Convert to kikuchipy EBSD signal and save (kikuchipy.save handles
        # the right group structure for kp.load() roundtrip).
        # NOTE: np.asarray returns a view on the memmap (no copy here). hyperspy
        # may still materialize the array internally during save — that's outside
        # our control, but at least we don't add an unnecessary extra copy.
        import kikuchipy as kp
        from orix.crystal_map import CrystalMap, Phase, PhaseList

        data = np.asarray(self._mmap)
        sig = kp.signals.EBSD(data)
        # Same xmap shape kikuchipy's own EBSDMasterPattern.get_patterns()
        # attaches to a dictionary (ebsd_master_pattern.py), so files from this
        # writer and from the CPU generator are interchangeable downstream.
        sig.xmap = CrystalMap(
            rotations=rotations,
            phase_list=PhaseList(phase if phase is not None else Phase()),
        )
        sig.save(str(self.output_path), overwrite=True)

        # Clean up staging file
        self.close()
        return self.output_path

    def close(self) -> None:
        """Release the memmap and remove the staging temp file.

        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return
        # Release the memmap so the file handle on Windows is freed before unlink
        try:
            mmap = getattr(self, "_mmap", None)
            if mmap is not None:
                # numpy memmap exposes the underlying mmap.mmap as ._mmap
                inner = getattr(mmap, "_mmap", None)
                if inner is not None:
                    try:
                        inner.close()
                    except Exception:
                        pass
                del self._mmap
        except Exception:
            pass
        # Unlink temp file
        try:
            tmp = getattr(self, "_tmp", None)
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass
        self._closed = True

    def __enter__(self) -> "ChunkedDictionaryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
