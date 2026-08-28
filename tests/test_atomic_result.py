"""A crash must never leave a half-written result at the final path.

The measurement that motivated this (2026-08-27): a process killed while
writing HDF5 leaves a file that opens, carries format_version, and holds only
the flushed datasets — it looks exactly like a finished export.
"""

import os
import subprocess
import sys
import textwrap

import h5py
import numpy as np
import pytest

from backend.api.services.atomic_result import AtomicResultFile


def test_final_path_appears_only_on_success(tmp_path):
    final = tmp_path / "result.h5"
    with AtomicResultFile(final) as tmp:
        with h5py.File(tmp, "w") as f:
            f.create_dataset("x", data=np.arange(10))
        assert not final.exists(), "must not exist while still being written"
    assert final.exists()
    with h5py.File(final, "r") as f:
        assert list(f["x"][:]) == list(range(10))


def test_failure_leaves_no_file_and_no_debris(tmp_path):
    final = tmp_path / "result.h5"
    with pytest.raises(RuntimeError):
        with AtomicResultFile(final) as tmp:
            with h5py.File(tmp, "w") as f:
                f.create_dataset("x", data=np.arange(10))
            raise RuntimeError("writer blew up")
    assert not final.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_a_failed_re_export_keeps_the_previous_good_result(tmp_path):
    """The case that would hurt most: overwriting a good file and crashing."""
    final = tmp_path / "result.h5"
    with h5py.File(final, "w") as f:
        f.create_dataset("generation", data=1)

    with pytest.raises(RuntimeError):
        with AtomicResultFile(final) as tmp:
            with h5py.File(tmp, "w") as f:
                f.create_dataset("generation", data=2)
            raise RuntimeError("crash halfway")

    with h5py.File(final, "r") as f:
        assert int(f["generation"][()]) == 1, "previous result must survive"


def test_leftover_part_from_an_earlier_crash_is_not_reused(tmp_path):
    final = tmp_path / "result.h5"
    debris = tmp_path / "result.h5.part"
    debris.write_bytes(b"garbage from a previous crash")

    with AtomicResultFile(final) as tmp:
        assert not os.path.exists(tmp), "the stale fragment must be gone"
        with h5py.File(tmp, "w") as f:
            f.create_dataset("x", data=[1])
    with h5py.File(final, "r") as f:
        assert list(f["x"][:]) == [1]


def test_a_killed_writer_leaves_the_final_path_untouched(tmp_path):
    """End to end with a real SIGKILL, because that is the actual scenario."""
    final = tmp_path / "result.h5"
    script = tmp_path / "writer.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        import h5py, numpy as np
        sys.path.insert(0, {str(os.getcwd())!r})
        from backend.api.services.atomic_result import AtomicResultFile
        with AtomicResultFile({str(final)!r}) as tmp:
            with h5py.File(tmp, "w") as f:
                for i in range(60):
                    f.create_dataset("chunk_%d" % i, data=np.random.rand(120, 120))
                    f.flush()
                    time.sleep(0.15)
    """), encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        proc.wait(timeout=2.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)

    assert not final.exists(), (
        "a killed writer must not leave a result at the final path — "
        "that file would open cleanly and look complete"
    )
    # The fragment may remain under .part; it is never read and the next
    # export removes it.
    assert not list(tmp_path.glob("result.h5")), "no final file"


def test_the_fragment_is_what_the_old_behaviour_produced(tmp_path):
    """Pins the hazard itself: writing in place DOES leave a usable-looking
    file. If this ever stops being true the guard can be reconsidered."""
    direct = tmp_path / "direct.h5"
    script = tmp_path / "direct_writer.py"
    script.write_text(textwrap.dedent(f"""
        import time
        import h5py, numpy as np
        with h5py.File({str(direct)!r}, "w") as f:
            f.attrs["format_version"] = "1.3"
            for i in range(60):
                f.create_dataset("chunk_%d" % i, data=np.random.rand(120, 120))
                f.flush()
                time.sleep(0.15)
    """), encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        proc.wait(timeout=2.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)

    assert direct.exists()
    with h5py.File(direct, "r") as f:
        assert f.attrs.get("format_version") == "1.3"
        assert 0 < len(list(f.keys())) < 60, "partial, yet it opens and reads"
