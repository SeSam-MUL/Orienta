"""A phase file is identified by where it really is, not by how it is spelled.

The library folder is often reached through a link (a junction on Windows, a
symlink elsewhere; a development worktree shares its `Database` this way), so
the listing names `<link>/CIF_Library/Al.cif` while a pasted or stored path
names the real folder. Compared as strings these are two files: the phase
picker offered both, and the second was refused as "a different file named Al".
The server therefore reports the resolved path (`real_path`) of every phase file
it returns, and compares files by it.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from backend.api.services.phase_path import inspect_phase_path
from tests.test_phase_path import AL_CIF


@pytest.fixture
def linked_library(tmp_path, monkeypatch):
    """A library whose listing spelling goes through a directory link.

    Returns ``(link_root, real_root)``; skips where a link cannot be created.
    """
    import path_utils

    real = tmp_path / "real_db"
    (real / "CIF_Library").mkdir(parents=True)
    (real / "CIF_Library" / "Al.cif").write_text(AL_CIF, encoding="utf-8")
    link = tmp_path / "linked_db"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot create a symlink here")
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(real)],
                              capture_output=True)
        if done.returncode != 0:
            pytest.skip("cannot create a junction here")
    monkeypatch.setattr(path_utils, "get_local_database_path", lambda: link)
    return link, real


def test_the_record_carries_the_resolved_path_whichever_way_it_was_reached(linked_library):
    link, real = linked_library
    via_link = inspect_phase_path("hough", str(link / "CIF_Library" / "Al.cif"))
    via_real = inspect_phase_path("hough", str(real / "CIF_Library" / "Al.cif"))
    assert via_link["real_path"] == via_real["real_path"]
    assert via_link["real_path"] == os.path.realpath(real / "CIF_Library" / "Al.cif")
    # and each says which entry of the listing it is, in the listing's spelling
    assert via_real["library_path"] == str(link / "CIF_Library" / "Al.cif")
    assert via_link["library_path"] == str(link / "CIF_Library" / "Al.cif")


def test_a_file_outside_the_library_has_a_resolved_path_too(tmp_path):
    p = tmp_path / "MyAl.cif"
    p.write_text(AL_CIF, encoding="utf-8")
    rec = inspect_phase_path("hough", str(p))
    assert rec["real_path"] == os.path.realpath(p)


def test_the_listing_reports_the_resolved_path_of_each_file(linked_library):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import indexing

    link, real = linked_library
    app = FastAPI()
    app.include_router(indexing.router, prefix="/api/indexing")
    files = TestClient(app).get("/api/indexing/files/hough").json()["files"]
    al = [f for f in files if f["filename"] == "Al.cif"]
    assert len(al) == 1
    assert al[0]["path"].startswith(str(link))
    assert al[0]["real_path"] == os.path.realpath(real / "CIF_Library" / "Al.cif")


def test_the_pc_controller_sees_a_link_and_its_target_as_one_file(linked_library):
    from pc_controller import _same_file

    link, real = linked_library
    assert _same_file(link / "CIF_Library" / "Al.cif", real / "CIF_Library" / "Al.cif")
    assert not _same_file(link / "CIF_Library" / "Al.cif", real / "CIF_Library" / "Other.cif")


def test_a_loaded_phase_reports_its_resolved_path(linked_library):
    from backend.api.routes.pcrefinement import _phase_summary
    from backend.api.routes import pcrefinement

    link, real = linked_library

    class _P:
        name = "Al"
        space_group = "Fm-3m"

    class _Ctrl:
        phase_paths = {"Al": str(link / "CIF_Library" / "Al.cif")}

    orig = pcrefinement._get_controller
    pcrefinement._get_controller = lambda: _Ctrl()
    try:
        s = _phase_summary(_P())
    finally:
        pcrefinement._get_controller = orig
    assert s["real_path"] == os.path.realpath(real / "CIF_Library" / "Al.cif")


def test_adding_the_same_file_through_the_target_is_a_duplicate_not_a_different_file(linked_library):
    from pc_controller import PCController, PhaseSetError

    link, real = linked_library
    ctrl = PCController()
    ctrl.load_phase(str(link / "CIF_Library" / "Al.cif"))
    ctrl.phase_paths["Al"] = os.path.abspath(str(link / "CIF_Library" / "Al.cif"))
    with pytest.raises(PhaseSetError) as ei:
        ctrl.add_phase(str(real / "CIF_Library" / "Al.cif"))
    assert ei.value.code == "phase_duplicate"
