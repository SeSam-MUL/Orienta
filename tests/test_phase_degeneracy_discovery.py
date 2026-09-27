"""discover_files_for_method must surface the degeneracy-detection fields."""
import pytest

# Needs the maintainer's crystal library / measurement data, which a clone
# does not have — skip with the missing path named, never fail.
from tests.data_deps import CIF_LIBRARY, requires


@requires(CIF_LIBRARY)
def test_discover_hough_files_carry_degeneracy_fields():
    from indexing_controller import IndexingMethod, discover_files_for_method

    result = discover_files_for_method(IndexingMethod.HOUGH)
    files = result["files"] if isinstance(result, dict) else result
    assert files, "expected at least one CIF discovered for Hough"

    # Every entry must expose the new keys (value may be None when a phase
    # has no lattice data).
    for entry in files:
        for key in ("lattice_a", "lattice_b", "lattice_c",
                    "lattice_alpha", "lattice_beta", "lattice_gamma",
                    "space_group_number", "laue_class", "centering"):
            assert key in entry, f"{entry.get('filename')} missing {key}"

    # MnAl6 is Cmcm — must come through fully populated.
    mnal6 = next((e for e in files if "MnAl6" in (e.get("filename") or "")), None)
    assert mnal6 is not None, "MnAl6 CIF not discovered"
    assert mnal6["laue_class"] == "mmm"
    assert mnal6["centering"] == "C"
    assert mnal6["lattice_a"] is not None and mnal6["lattice_a"] > 0
