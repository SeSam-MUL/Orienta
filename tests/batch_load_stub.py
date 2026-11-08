"""Stand-in for the batch's crystal-map reader in tests that feed it a ready-made xmap."""
from analysis.xmap_loader import step_of_xmap


def patch_batch_load(monkeypatch, bp, read):
    """Make ``BatchProcessor`` read every file with ``read(path) -> xmap`` (what ``analysis.xmap_loader`` would
    return: the map, its step, and the meta of a file that states no step)."""
    def load_xmap_file(path):
        xmap = read(path)
        return xmap, step_of_xmap(xmap), {"format": "stub", "step_from_file": False}

    monkeypatch.setattr(bp, "load_xmap_file", load_xmap_file)
