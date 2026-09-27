"""The three gates, against trees built to break them.

Every case here is a real shape the macOS runtime can take, including the two
that made the first version of gate 1 useless:

  * conda-forge ships `libiomp5.dylib` as a second name for `libomp.dylib`,
    so a CORRECT install has two names and one runtime. A gate that counted
    names would fail every correct install and be switched off within a week.
  * a conda environment is full of symlinked directories, and `Path.rglob`
    does not descend into them. The first version reported "exactly one
    runtime" on a tree with two, the second hidden behind
    `site-packages/torch/.dylibs -> elsewhere`. Measured, not deduced.

The tests force `platform_name="darwin"` because these are macOS gates and
the machine that runs the suite is not a Mac. Whether they pass on a real Mac
is T11's question; whether they say the right thing about a given tree is
this file's.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_runtime_health as health  # noqa: E402

MACHO64 = b"\xcf\xfa\xed\xfe rest"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_lib(root: Path, relative: str, content: bytes = MACHO64) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def link(target: Path, name: Path, directory: bool = False) -> bool:
    """Symlink, or tell the caller it is not available.

    Windows refuses symlinks without developer mode or admin rights, and this
    suite runs there. A test that silently became a no-op would be worse than
    one that skips.
    """
    try:
        name.parent.mkdir(parents=True, exist_ok=True)
        name.symlink_to(target, target_is_directory=directory)
        return True
    except (OSError, NotImplementedError):
        return False


class FakeRun:
    """A stand-in for subprocess.run that fails on named paths."""

    def __init__(self, failing=(), report="", returncode_for_report=0, raises=None):
        self.failing = tuple(failing)
        self.report = report
        self.returncode_for_report = returncode_for_report
        self.raises = raises
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if self.raises:
            raise self.raises
        if cmd[:2] == ["codesign", "-dv"]:
            return subprocess.CompletedProcess(
                cmd, self.returncode_for_report, stdout="", stderr=self.report)
        target = cmd[-1]
        if any(bad in target for bad in self.failing):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr=f"{target}: code object is not signed at all")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


# --------------------------------------------------------------------------
# Gate 1 -- files
# --------------------------------------------------------------------------

def test_one_runtime_passes(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    result = health.gate_files(tmp_path, platform_name="darwin")
    assert result.ok, result.detail


def test_the_libiomp5_symlink_is_not_a_second_runtime(tmp_path):
    """The case that decides whether this gate is usable at all."""
    real = make_lib(tmp_path, "lib/libomp.dylib")
    if not link(real, tmp_path / "lib" / "libiomp5.dylib"):
        pytest.skip("symlinks unavailable on this machine")
    result = health.gate_files(tmp_path, platform_name="darwin")
    assert result.ok, result.detail
    assert "2 name(s)" in result.detail


def test_a_runtime_behind_a_symlinked_directory_is_found(tmp_path):
    """The false PASS a reviewer measured: rglob does not descend into a
    symlinked directory, and site-packages is full of them."""
    prefix, outside = tmp_path / "prefix", tmp_path / "outside"
    make_lib(prefix, "lib/libomp.dylib")
    make_lib(outside, "libiomp5.dylib")
    site = prefix / "lib" / "python3.11" / "site-packages" / "torch"
    site.mkdir(parents=True)
    if not link(outside, site / ".dylibs", directory=True):
        pytest.skip("symlinks unavailable on this machine")
    result = health.gate_files(prefix, platform_name="darwin")
    assert not result.ok, "two real runtimes must not read as one"
    assert "2 OpenMP runtimes" in result.detail


def test_a_symlink_loop_does_not_hang_the_walk(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    if not link(tmp_path, tmp_path / "lib" / "back", directory=True):
        pytest.skip("symlinks unavailable on this machine")
    assert health.gate_files(tmp_path, platform_name="darwin").ok


def test_a_hard_linked_alias_is_one_runtime(tmp_path):
    """Two directory entries, one file. Same situation as the symlink, and
    the same wrong answer if identity is taken from the path."""
    real = make_lib(tmp_path, "lib/libomp.dylib")
    try:
        (tmp_path / "lib" / "libiomp5.dylib").hardlink_to(real)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links unavailable on this machine")
    result = health.gate_files(tmp_path, platform_name="darwin")
    assert result.ok, result.detail


def test_two_real_runtimes_fail(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "lib/python3.11/site-packages/torch/.dylibs/libomp.dylib")
    result = health.gate_files(tmp_path, platform_name="darwin")
    assert not result.ok
    assert "2 OpenMP runtimes" in result.detail
    assert len(result.evidence) == 2


def test_an_intel_runtime_beside_the_llvm_one_fails(tmp_path):
    # Two different names AND two different files -- the actual bug report.
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "lib/libiomp5.dylib")
    assert not health.gate_files(tmp_path, platform_name="darwin").ok


def test_a_gnu_runtime_counts_too(tmp_path):
    """libgomp is in the prefix list; without it the glob would miss a GNU
    runtime dragged in by a pip wheel."""
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "lib/libgomp.1.dylib")
    assert not health.gate_files(tmp_path, platform_name="darwin").ok


def test_no_runtime_at_all_fails_rather_than_passing(tmp_path):
    """Zero is not one. An empty prefix must not read as healthy."""
    result = health.gate_files(tmp_path, platform_name="darwin")
    assert not result.ok
    assert "no OpenMP runtime" in result.detail


def test_a_dangling_symlink_is_not_counted(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    if not link(tmp_path / "lib" / "gone.dylib", tmp_path / "lib" / "libiomp5.dylib"):
        pytest.skip("symlinks unavailable on this machine")
    assert health.gate_files(tmp_path, platform_name="darwin").ok


def test_the_only_runtime_living_outside_the_prefix_fails(tmp_path):
    """A symlink out to Homebrew's libomp: one runtime today, and a different
    one the moment Homebrew upgrades it."""
    prefix, outside = tmp_path / "prefix", tmp_path / "brew"
    (prefix / "lib").mkdir(parents=True)
    real = make_lib(outside, "libomp.dylib")
    if not link(real, prefix / "lib" / "libomp.dylib"):
        pytest.skip("symlinks unavailable on this machine")
    result = health.gate_files(prefix, platform_name="darwin")
    assert not result.ok
    assert "OUTSIDE" in result.detail


def test_linux_is_reported_as_not_applicable_never_as_a_pass(tmp_path):
    # On Linux the bundled libgomp copies coexist; globbing .so would fail
    # every correct install. It must say so, not stay quiet.
    result = health.gate_files(tmp_path, platform_name="linux")
    assert result.applicable is False
    assert "not applicable" in result.line()


# --------------------------------------------------------------------------
# Gate 2 -- loaded
# --------------------------------------------------------------------------

def test_loaded_unions_both_sources(tmp_path):
    real = make_lib(tmp_path, "lib/libomp.dylib")
    found = health.loaded_openmp_libraries(
        threadpool_info=[{"user_api": "openmp", "filepath": str(real)}],
        image_names=[],
    )
    assert found == {str(real.resolve())}


def test_a_runtime_only_dyld_can_see_still_counts(tmp_path):
    """threadpoolctl reports runtimes it can drive; a loaded-but-idle one is
    invisible to it and aborts the moment it is used."""
    driven = make_lib(tmp_path, "lib/libomp.dylib")
    idle = make_lib(tmp_path, "torch/.dylibs/libiomp5.dylib")
    found = health.loaded_openmp_libraries(
        threadpool_info=[{"user_api": "openmp", "filepath": str(driven)}],
        image_names=[str(idle), "/usr/lib/libSystem.B.dylib"],
    )
    assert found == {str(driven.resolve()), str(idle.resolve())}


def test_a_gnu_runtime_in_the_image_list_counts(tmp_path):
    idle = make_lib(tmp_path, "x/libgomp.1.dylib")
    assert health.loaded_openmp_libraries(threadpool_info=[],
                                          image_names=[str(idle)])


def test_the_same_runtime_under_two_names_counts_once(tmp_path):
    real = make_lib(tmp_path, "lib/libomp.dylib")
    alias = tmp_path / "lib" / "libiomp5.dylib"
    if not link(real, alias):
        pytest.skip("symlinks unavailable on this machine")
    found = health.loaded_openmp_libraries(
        threadpool_info=[{"user_api": "openmp", "filepath": str(real)}],
        image_names=[str(alias)],
    )
    assert len(found) == 1


def test_the_dyld_list_is_actually_consulted(monkeypatch, tmp_path):
    """Without this, the macOS half of the probe can be deleted and the suite
    stays green, because every other test injects `image_names` by hand."""
    idle = make_lib(tmp_path, "x/libiomp5.dylib")
    monkeypatch.setattr(health, "dyld_image_names", lambda: [str(idle)])
    found = health.loaded_openmp_libraries(threadpool_info=[])   # image_names=None
    assert found == {str(idle.resolve())}


def test_non_openmp_images_are_ignored():
    found = health.loaded_openmp_libraries(
        threadpool_info=[{"user_api": "blas", "filepath": "/x/libopenblas.dylib"}],
        image_names=["/usr/lib/libSystem.B.dylib", "/x/libcompression.dylib"],
    )
    assert found == set()


# The verdict branches are pure logic, so they are driven directly rather than
# by running torch on a machine that has no Mac. Without this seam the whole
# darwin half of the gate could be gutted with a green suite -- a reviewer
# demonstrated exactly that.

def test_one_loaded_runtime_passes():
    result = health.gate_loaded(platform_name="darwin", work=lambda: ["work"],
                                probe=lambda: {"/x/libomp.dylib"})
    assert result.ok, result.detail


def test_two_loaded_runtimes_fail():
    result = health.gate_loaded(platform_name="darwin", work=lambda: ["work"],
                                probe=lambda: {"/x/libomp.dylib", "/y/libiomp5.dylib"})
    assert not result.ok
    assert "2 OpenMP runtimes loaded" in result.detail


def test_no_loaded_runtime_fails_rather_than_passing():
    result = health.gate_loaded(platform_name="darwin", work=lambda: ["work"],
                                probe=lambda: set())
    assert not result.ok
    assert "cannot see what it is meant to check" in result.detail


def test_work_that_raises_is_a_failure_not_a_pass():
    def boom():
        raise RuntimeError("no torch here")
    result = health.gate_loaded(platform_name="darwin", work=boom, probe=lambda: set())
    assert not result.ok
    assert "RuntimeError" in result.detail


def test_a_probe_that_raises_is_also_a_failure():
    """threadpoolctl missing must be a gate failure, not a raw traceback."""
    def boom():
        raise ImportError("threadpoolctl")
    result = health.gate_loaded(platform_name="darwin", work=lambda: [], probe=boom)
    assert not result.ok
    assert "ImportError" in result.detail


def test_the_gate_refuses_to_report_on_the_wrong_interpreter(tmp_path):
    """Its whole premise is 'inside the environment under test'. Run from
    another interpreter it could pass while the runtime being installed is
    broken."""
    result = health.gate_loaded(platform_name="darwin", prefix=tmp_path / "elsewhere",
                                work=lambda: ["work"], probe=lambda: set())
    assert not result.ok
    assert "own interpreter" in result.detail


def test_the_gate_runs_when_the_interpreter_is_the_one_under_test():
    result = health.gate_loaded(platform_name="darwin", prefix=Path(sys.prefix),
                                work=lambda: ["work"], probe=lambda: {"/x/libomp.dylib"})
    assert result.ok, result.detail


def test_the_work_really_runs_and_announces_each_step_first():
    """The only test that executes `do_parallel_work`.

    It cannot run on macOS here, so it does not prove the gate catches a
    duplicate runtime -- that is T11. What it does prove is that the function
    does the work rather than returning an empty list, and that every step is
    announced BEFORE it runs. The announcements are load-bearing: the failure
    this gate exists for is an abort(), which kills the process without an
    exception, so the last line printed is the only evidence of which step
    did it.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("numba")
    said: list[str] = []
    done = health.do_parallel_work(say=said.append)

    assert any("torch" in s for s in done)
    assert any("KMeans" in s for s in done)
    assert any("numba" in s for s in done)
    # announced before finished, for every step
    assert len(said) >= len(done)
    assert said[0].startswith("gate 2: torch")
    # numba may pick tbb over omp; the gate reports which, rather than assume
    assert any("threading layer" in s for s in done)


def test_loaded_gate_is_not_applicable_off_macos():
    assert health.gate_loaded(platform_name="linux").applicable is False


# --------------------------------------------------------------------------
# Gate 3 -- signatures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("magic", [
    b"\xcf\xfa\xed\xfe",  # 64-bit little-endian
    b"\xce\xfa\xed\xfe",  # 32-bit little-endian
    b"\xfe\xed\xfa\xcf",  # 64-bit big-endian
    b"\xfe\xed\xfa\xce",  # 32-bit big-endian
    b"\xca\xfe\xba\xbf",  # 64-bit universal -- missing at first, fails OPEN
])
def test_every_mach_o_magic_is_recognised(magic):
    assert health.is_macho(magic + b"\x00\x00\x00\x02")


def test_a_universal_binary_is_recognised():
    # ca fe ba be, then a plausible architecture count
    assert health.is_macho(b"\xca\xfe\xba\xbe\x00\x00\x00\x02")


def test_a_java_class_file_is_not_a_mach_o():
    """Same magic; the next field tells them apart. Without this a stray
    .class fails a gate about code signatures."""
    assert not health.is_macho(b"\xca\xfe\xba\xbe\x00\x00\x00\x41")  # major 65


@pytest.mark.parametrize("header", [b"plain te", b"!<arch>\n", b"", b"\x7fELF\x02\x01\x01"])
def test_non_mach_o_headers_are_rejected(header):
    assert not health.is_macho(header)


def test_mach_o_files_are_found_by_magic_not_by_extension(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "bin/python")                       # no extension
    (tmp_path / "lib" / "notes.txt").write_bytes(b"plain text, not Mach-O")
    found = {p.name for p in health.macho_files(tmp_path)}
    assert found == {"libomp.dylib", "python"}


def test_an_app_bundle_we_never_launch_is_reported_but_does_not_fail(tmp_path):
    """Measured on the runner, then encoded here.

    Run 35997347703 asked dyld which of the 17 rejected files the backend
    actually maps: sixteen are never loaded, and every Qt developer tool
    (Designer, Linguist, Assistant, qml, pixeltool, qdbusviewer) is among
    them. Orienta launches none of them, so no dlopen of ours can go through
    their main executable.

    They are still listed -- silence would be a different claim -- but they
    do not fail a gate whose whole sentence is about dlopen.
    """
    make_lib(tmp_path, "bin/Designer.app/Contents/MacOS/Designer")
    result = health.gate_signatures(
        tmp_path, run=FakeRun(failing=["Designer"]), platform_name="darwin",
        say=lambda _: None)
    assert result.ok, result.detail
    assert any("Designer" in e for e in (result.evidence or [])), result.evidence


def test_the_named_exception_passes_but_is_still_printed(tmp_path):
    """An accepted exception that stops being printed is one nobody re-reads.

    libocl_icd_wrapper_apple carries a date with it: it is inert only while
    the bundle runs without the hardened runtime, and notarisation turns that
    on. So the gate passes and still says the name and the reason, every run.
    """
    make_lib(tmp_path, "lib/libocl_icd_wrapper_apple.dylib")
    result = health.gate_signatures(
        tmp_path, run=FakeRun(failing=["libocl"]), platform_name="darwin",
        say=lambda _: None)
    assert result.ok, result.detail
    printed = " ".join(result.evidence or [])
    assert "libocl_icd_wrapper_apple" in printed
    # the name, the word that marks it a decision, and WHY -- so a reader who
    # has never seen this file can judge the entry without leaving the output
    assert "accepted" in printed and "pyopencl" in printed
    # and the count is in the summary, so a list that grows is visible even to
    # someone skimming
    assert "named exception" in result.detail


def test_the_exception_is_one_name_and_not_a_licence_for_its_neighbours(tmp_path):
    """The next broken dylib in the same directory must still stop the gate.

    This is the difference between an exception and a silencer, and it is the
    reason KNOWN_UNSIGNED is keyed on an exact file name rather than a path
    prefix or a pattern.
    """
    make_lib(tmp_path, "lib/libocl_icd_wrapper_apple.dylib")
    make_lib(tmp_path, "lib/libsomething_else.dylib")
    result = health.gate_signatures(
        tmp_path, run=FakeRun(failing=["lib"]), platform_name="darwin",
        say=lambda _: None)
    assert not result.ok
    assert "libsomething_else" in " ".join(result.evidence or [])


def test_a_loadable_library_still_fails(tmp_path):
    """Anything loadable, broken and unlisted must stop the gate.

    This used to use libocl_icd_wrapper_apple as its example. That file is now
    a NAMED exception, so the test would have passed for the wrong reason --
    the guard against exactly that is
    test_the_exception_is_one_name_and_not_a_licence_for_its_neighbours. An
    ordinary library is the right example here.

    Why a rejected signature matters at all: macOS enforces library validation
    under the hardened runtime. This application is ad-hoc signed with
    hardenedRuntime false, so such a library loads today. Notarisation
    requires the hardened runtime and makes it fatal at dlopen.
    """
    make_lib(tmp_path, "lib/libtorch_cpu.dylib")
    result = health.gate_signatures(
        tmp_path, run=FakeRun(failing=["libtorch"]), platform_name="darwin",
        say=lambda _: None)
    assert not result.ok
    assert "hardened runtime" in result.detail


def test_link_time_leftovers_are_not_checked(tmp_path):
    """A `.o` and a `.a` are Mach-O and can never be the failure this gate means.

    The gate's claim is about `dlopen`: a dylib whose signature micromamba
    failed to restore kills the process when something loads it. A relocatable
    object and a static archive are link-time inputs; nothing loads them at
    run time, and no user can ever meet the failure through them.

    They carry the magic number regardless, so the first run of this gate on a
    real Mac (35996081610) reported 33 broken signatures of which 18 were
    exactly these -- build leftovers inside conda-forge's Qt packages. A gate
    reporting things it cannot mean is a gate people stop reading.
    """
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "lib/objects-Release/json_reader.cpp.o")
    make_lib(tmp_path, "lib/libQt5Bootstrap.a")
    assert [p.name for p in health.macho_files(tmp_path)] == ["libomp.dylib"]


def test_a_symlink_is_not_verified_twice(tmp_path):
    real = make_lib(tmp_path, "lib/libomp.dylib")
    if not link(real, tmp_path / "lib" / "alias.dylib"):
        pytest.skip("symlinks unavailable on this machine")
    assert [p.name for p in health.macho_files(tmp_path)] == ["libomp.dylib"]


def test_every_file_verifying_passes(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "bin/python")
    run = FakeRun()
    result = health.gate_signatures(tmp_path, run=run, platform_name="darwin",
                                    say=lambda m: None)
    assert result.ok, result.detail
    assert len(run.calls) == 2
    assert run.calls[0][:4] == ["codesign", "--verify", "--deep", "--strict"]


def test_one_broken_signature_fails_and_names_the_file(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    make_lib(tmp_path, "lib/libtorch_cpu.dylib")
    result = health.gate_signatures(tmp_path, run=FakeRun(failing=["libtorch_cpu"]),
                                    platform_name="darwin", say=lambda m: None)
    assert not result.ok
    assert "1 of 2" in result.detail
    assert any("libtorch_cpu" in item for item in result.evidence)


def test_a_hung_codesign_is_a_failure_not_a_hang(tmp_path):
    make_lib(tmp_path, "lib/libomp.dylib")
    run = FakeRun(raises=subprocess.TimeoutExpired(cmd="codesign", timeout=1))
    result = health.gate_signatures(tmp_path, run=run, platform_name="darwin",
                                    say=lambda m: None)
    assert not result.ok
    assert "did not finish" in result.evidence[0]


def test_an_empty_prefix_fails_rather_than_reporting_all_clear(tmp_path):
    result = health.gate_signatures(tmp_path, run=FakeRun(), platform_name="darwin",
                                    say=lambda m: None)
    assert not result.ok
    assert "no Mach-O files" in result.detail


def test_adhoc_bundle_passes(tmp_path):
    run = FakeRun(report="Executable=/x/Orienta.app\nSignature=adhoc\n")
    assert health.gate_bundle_is_adhoc_signed(
        tmp_path / "Orienta.app", run=run, platform_name="darwin").ok


def test_a_bundle_that_is_not_signed_at_all_says_so(tmp_path):
    """What real codesign does: non-zero exit and 'not signed at all'. Before,
    that was reported as 'could not read', which reads as a missing path."""
    run = FakeRun(report="/x/Orienta.app: code object is not signed at all",
                  returncode_for_report=1)
    result = health.gate_bundle_is_adhoc_signed(
        tmp_path / "Orienta.app", run=run, platform_name="darwin")
    assert not result.ok
    assert "not signed at all" in result.detail


def test_a_bundle_signed_by_somebody_else_is_not_adhoc(tmp_path):
    run = FakeRun(report="Executable=/x/Orienta.app\nSignature=Developer ID\n")
    result = health.gate_bundle_is_adhoc_signed(
        tmp_path / "Orienta.app", run=run, platform_name="darwin")
    assert not result.ok
    assert "not ad-hoc" in result.detail


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

def test_a_failing_gate_makes_main_exit_nonzero(tmp_path, monkeypatch, capsys):
    """Driven through darwin rather than through this machine's platform, so
    the exit code is actually exercised where the suite runs."""
    monkeypatch.setattr(health, "gate_files",
                        lambda prefix, **kw: health.GateResult("openmp files", False, "two"))
    code = health.main(["--prefix", str(tmp_path), "--gate", "files"])
    assert code == health.EXIT_FAILED
    assert "FAIL" in capsys.readouterr().out


def test_a_passing_gate_makes_main_exit_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(health, "gate_files",
                        lambda prefix, **kw: health.GateResult("openmp files", True, "one"))
    assert health.main(["--prefix", str(tmp_path), "--gate", "files"]) == health.EXIT_OK
    assert "OK " in capsys.readouterr().out


def test_ok_and_fail_are_not_interchangeable_in_the_output():
    assert "OK " in health.GateResult("x", True, "d").line()
    assert "FAIL" in health.GateResult("x", False, "d").line()
    assert "FAIL" not in health.GateResult("x", True, "d").line()


def test_a_run_where_nothing_applies_does_not_claim_success(tmp_path, capsys):
    """A CI job on Linux would otherwise print 'all gates passed' and go green
    having checked nothing -- the exact failure this file is about."""
    code = health.main(["--prefix", str(tmp_path), "--gate", "files"])
    captured = capsys.readouterr()
    if sys.platform == "darwin":
        pytest.skip("on macOS the gate applies")
    assert code == health.EXIT_NOT_APPLICABLE
    assert "nothing was checked" in captured.err
    assert "all" not in captured.out.lower() or "passed" not in captured.out.lower()


def test_main_refuses_to_do_nothing_quietly(capsys):
    assert health.main(["--gate", "files"]) == health.EXIT_NOTHING_TO_DO
    assert "nothing to check" in capsys.readouterr().err
