import pytest


def test_sample_orientations_at_5deg_returns_thousands_of_rotations():
    from backend.dict_gpu.pipeline.grid import sample_orientations
    rot = sample_orientations("m-3m", 5.0)
    # orix's exact count is grid-dependent; for cubic at 5° it's ~6500
    assert 800 <= rot.size <= 8000, f"unexpected size {rot.size} at 5°"


def test_sample_orientations_at_2deg_is_finer_than_5deg():
    from backend.dict_gpu.pipeline.grid import sample_orientations
    coarse = sample_orientations("m-3m", 5.0)
    fine = sample_orientations("m-3m", 2.0)
    assert fine.size > coarse.size, (
        f"2° ({fine.size}) should give more rotations than 5° ({coarse.size})"
    )


def test_sample_orientations_accepts_orix_symmetry_object():
    from backend.dict_gpu.pipeline.grid import sample_orientations
    from orix.quaternion import symmetry as orix_sym
    rot = sample_orientations(orix_sym.Oh, 5.0)  # Oh = m-3m in Schoenflies
    assert rot.size > 0


def test_sample_orientations_returns_rotation_object():
    from backend.dict_gpu.pipeline.grid import sample_orientations
    from orix.quaternion import Rotation
    rot = sample_orientations("m-3m", 10.0)
    assert isinstance(rot, Rotation)
