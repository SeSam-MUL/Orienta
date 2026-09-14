"""A1: verify cc_fp64 plumbing reaches Tier1Indexer.

Background: rs2cc_fp64 + Tier1Indexer.cc_fp64 are implemented but never
reachable from the public IndexingConfig API. This test locks that
plumbing in.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from indexing_controller import IndexingConfig
from backend.spherical_gpu.backend import BackendConfig
from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
from backend.spherical_gpu.pipeline.detector import DetectorGeometry
from backend.spherical_gpu.pipeline.sht_io import read_sht_master


def test_indexing_config_has_cc_fp64_field():
    cfg = IndexingConfig()
    assert hasattr(cfg, "cc_fp64"), \
        "IndexingConfig must expose cc_fp64 (default False)"
    assert cfg.cc_fp64 is False, "Default must remain False (FP32 hot path)"


def test_indexing_config_cc_fp64_propagates_to_phase_config():
    cfg = IndexingConfig(sht_file="dummy.sht", cc_fp64=True)
    bcfg = BackendConfig.from_indexing_config(cfg)
    assert len(bcfg.phases) == 1
    assert bcfg.phases[0].cc_fp64 is True, \
        "PhaseConfig must inherit cc_fp64 from IndexingConfig"


def test_indexing_config_cc_fp64_default_does_not_change_phase_config():
    cfg = IndexingConfig(sht_file="dummy.sht")  # default cc_fp64=False
    bcfg = BackendConfig.from_indexing_config(cfg)
    assert bcfg.phases[0].cc_fp64 is False, \
        "PhaseConfig must inherit cc_fp64=False from default IndexingConfig"


def _disorientation_deg(eu_ours, eu_alt, symmetry_group):
    """Min disorientation between two ZXZ Euler triples under a symmetry group.

    Returns a 1D numpy array (one value per pixel). orix exposes symmetry
    instances via the ``_groups`` list (keyed by Hermann-Mauguin name like
    "432"), not as direct module attributes — same lookup pattern as
    test_05_indexer_tier1.py.
    """
    from orix.quaternion import Misorientation, Rotation
    from orix.quaternion.symmetry import _groups

    sym = next((g for g in _groups if g.name == symmetry_group), None)
    if sym is None:
        raise ValueError(
            f"Unknown point group {symmetry_group!r}; expected one of "
            f"{[g.name for g in _groups]}"
        )
    r_a = Rotation.from_euler(eu_ours)
    r_b = Rotation.from_euler(eu_alt)
    miso = Misorientation((r_a * (~r_b)).data, symmetry=(sym, sym))
    miso = miso.map_into_symmetry_reduced_zone()
    return np.rad2deg(miso.angle)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP64 cc-volume parity check is too slow on CPU; CUDA-only.",
)
def test_cc_fp64_improves_alt_oracle_parity(oracle, oracle_h5oina, detector_params, oracle_meta, tmp_path):
    """A1: cc_fp64=True must NOT degrade alt-oracle parity vs cc_fp64=False.

    iter-15B finding: switching to FP64 cc-volume should lift the
    <5deg fraction vs alt-oracle from ~28% to ~38%. This test enforces
    the weaker guarantee 'cc_fp64=True is not worse than cc_fp64=False
    on this dataset', which is a robust regression lock independent
    of the absolute number (which varies with hardware FP rounding).
    """
    if "euler_xyz_alt_oracle" not in oracle:
        pytest.skip("Oracle does not embed alt-oracle eulers (regenerate with BatchIndexDump path)")

    device = torch.device("cuda")
    geom = DetectorGeometry.from_params(detector_params, device=device)

    sht_filename = oracle["embedded_sht_filename"][()]
    if isinstance(sht_filename, bytes):
        sht_filename = sht_filename.decode("utf-8")
    sht_target = tmp_path / sht_filename
    sht_target.write_bytes(bytes(oracle["embedded_sht_bytes"][()]))

    master = read_sht_master(str(sht_target), device=device)
    eu_alt = np.asarray(oracle["euler_xyz_alt_oracle"])
    pg = oracle_meta["symmetry_group"]

    # ----- FP32 baseline -----
    ix_fp32 = Tier1Indexer(geom=geom, master=master, device=device, bandwidth=68, cc_fp64=False)
    res_fp32 = ix_fp32.index_h5oina(oracle_h5oina, batch_size=32)
    eu_ours_fp32 = res_fp32.euler_xyz.cpu().numpy()
    diso_fp32 = _disorientation_deg(eu_ours_fp32, eu_alt, pg)

    # ----- FP64 -----
    ix_fp64 = Tier1Indexer(geom=geom, master=master, device=device, bandwidth=68, cc_fp64=True)
    res_fp64 = ix_fp64.index_h5oina(oracle_h5oina, batch_size=32)
    eu_ours_fp64 = res_fp64.euler_xyz.cpu().numpy()
    diso_fp64 = _disorientation_deg(eu_ours_fp64, eu_alt, pg)

    frac_fp32 = float((diso_fp32 < 5.0).mean())
    frac_fp64 = float((diso_fp64 < 5.0).mean())
    print(f"\n<5deg fraction vs alt-oracle: FP32={frac_fp32:.3f} FP64={frac_fp64:.3f}")
    # Lock: FP64 must be no worse than FP32 minus 1 percentage point.
    assert frac_fp64 >= frac_fp32 - 0.01, \
        f"FP64 regressed: FP32={frac_fp32:.3f} > FP64={frac_fp64:.3f}+0.01"
    # Strong claim from iter-15B: FP64 should lift by at least 3pp.
    # Only assert this on the canonical Oxford dataset (skip elsewhere).
    if oracle_meta.get("dataset_name") == "iter15_oxford":
        assert frac_fp64 >= frac_fp32 + 0.03, \
            f"FP64 should lift by at least 3pp on iter15 dataset; got " \
            f"FP32={frac_fp32:.3f}, FP64={frac_fp64:.3f}"
