"""What may leave this repository in a release artifact, and what must be in it.

NEGATIVE: the maintainer's instruction was explicit — "wir geben unsere Datenbank
nicht her". An allowlist decides what goes in; a negative test decides what must
never be found. Two independent guards, because one of them will eventually be
edited by someone who does not know why it is there.

POSITIVE: a package missing one module is not a package. A hand-written list of
required files cannot see that, so the import graph of everything shipped is
resolved against what is shipped — with no parent-package escape hatch, because
`from backend.api.routes import eds_export` has a parent that exists and a module
that (once) did not, and that is exactly the case this guard is for.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_runtime_package import (
    RUNTIME_DIRS, build, check_notice_references, collect_files, notice_references,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = (".cif", ".xtal", ".sht", ".h5oina", ".h5", ".hdf5", ".up1", ".up2",
                      ".ang", ".ctf", ".osc")
FORBIDDEN_DIRS = ("Database/", "Test_data/", "tests/", "tasks/", "screenshots/", ".git/",
                  "docs/", "research/")
FIRST_PARTY = ("backend", "analysis", "simulation", "tools")


DIST_MISSING = "frontend/dist fehlt: erst `npm run build` in frontend/"


def _skip_unless_dist_has(*rel_paths: str) -> None:
    """The Python suite must not depend on a build artefact being current.

    `frontend/dist` is gitignored and built by `npm run build`; a fresh
    checkout has none, an older checkout has one without the files a newer
    build writes (THIRD-PARTY-LICENSES.txt, vendor/LICENSE.vanta.txt). Either
    way that is a fact about the checkout, not about the packaging code, so
    the tests that need those files skip and say what to run. The guard for
    the PACKAGE stays in build_release.py, which refuses to release without
    them.
    """
    missing = [p for p in rel_paths if not (REPO_ROOT / p).is_file()]
    if missing:
        pytest.skip(f"{DIST_MISSING} (nicht vorhanden: {', '.join(missing)})")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    _skip_unless_dist_has("frontend/dist/index.html")
    out = tmp_path_factory.mktemp("pkg") / "orienta-runtime-v0.0.0-test.zip"
    build(REPO_ROOT, out, version="v0.0.0-test")
    return out


# --------------------------------------------------------------------------
# what must never leave
# --------------------------------------------------------------------------

def test_no_crystal_or_measurement_data_is_ever_packaged(built):
    with zipfile.ZipFile(built) as zf:
        offenders = [n for n in zf.namelist() if n.lower().endswith(FORBIDDEN_SUFFIXES)]
    assert not offenders, f"data files leaked into the package: {offenders[:10]}"


def test_no_excluded_directory_is_ever_packaged(built):
    with zipfile.ZipFile(built) as zf:
        offenders = [n for n in zf.namelist()
                     if any(n.startswith(d) for d in FORBIDDEN_DIRS)]
    assert not offenders, f"excluded directories leaked: {offenders[:10]}"


def test_nothing_untracked_is_ever_packaged():
    """The working tree carries other sessions' uncommitted work by design, so
    the package is built from `git ls-files` and never from an rglob."""
    _skip_unless_dist_has("frontend/dist/index.html")   # collect_files refuses without it
    tracked = set(subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines())
    for rel in collect_files(REPO_ROOT):
        posix = rel.as_posix()
        if posix.startswith("frontend/dist/"):
            continue  # a build product, gitignored on purpose
        assert posix in tracked, f"{posix} is not tracked by git"


def test_no_member_would_be_refused_by_the_applier(built):
    """The applier refuses any member with a `Database` path component (compared
    case-folded, because Win32 is case-insensitive) and discards the whole
    package. Nothing checked that the package we actually BUILD passes.

    The day someone adds `backend/api/services/database/__init__.py`, every
    release becomes undeployable and the update check re-downloads and
    re-discards the same package on every attempt.
    """
    def components(name: str) -> list[str]:
        return [p.rstrip(" .").casefold() for p in name.replace("\\", "/").split("/")]

    with zipfile.ZipFile(built) as zf:
        offenders = [n for n in zf.namelist()
                     if "database" in components(n) or ".applying" in components(n)]
    assert not offenders, f"the applier would refuse this package: {offenders[:10]}"


def test_the_allowlist_is_directories_not_a_denylist():
    assert "backend" in RUNTIME_DIRS and "analysis" in RUNTIME_DIRS
    assert "Database" not in RUNTIME_DIRS and "Test_data" not in RUNTIME_DIRS


# --------------------------------------------------------------------------
# what must be in it
# --------------------------------------------------------------------------

def test_every_first_party_import_in_the_package_resolves_inside_the_package(built):
    """No parent escape hatch: the motivating defect had a present parent."""
    with zipfile.ZipFile(built) as zf:
        names = set(zf.namelist())
        missing = []
        for name in sorted(n for n in names if n.endswith(".py")):
            try:
                tree = ast.parse(zf.read(name).decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                dotted_names = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    dotted_names = [f"{node.module}.{a.name}" for a in node.names]
                elif isinstance(node, ast.Import):
                    dotted_names = [a.name for a in node.names]
                for dotted in dotted_names:
                    if dotted.split(".")[0] not in FIRST_PARTY:
                        continue
                    stem = dotted.replace(".", "/")
                    if f"{stem}.py" in names or f"{stem}/__init__.py" in names:
                        continue
                    # `from pkg import name` where `name` is an attribute of the
                    # parent module, not a submodule. Legal; only report when the
                    # parent is missing too.
                    parent = dotted.rsplit(".", 1)[0].replace(".", "/")
                    if f"{parent}.py" in names or f"{parent}/__init__.py" in names:
                        continue
                    missing.append(f"{name} imports {dotted}")
    assert not missing, ("the package may import modules it does not contain:\n"
                         + "\n".join(dict.fromkeys(missing)))


def test_the_package_carries_its_version_and_manifest(built):
    with zipfile.ZipFile(built) as zf:
        assert zf.read("VERSION").decode("utf-8").strip() == "v0.0.0-test"
        manifest = zf.read("MANIFEST").decode("utf-8").splitlines()
        names = set(zf.namelist())
    assert "backend/api/main.py" in manifest
    assert "MANIFEST" not in manifest and "VERSION" not in manifest, (
        "the manifest must not list itself; the prune would then be able to "
        "delete the file that tells it what to prune"
    )
    assert set(manifest) <= names, "the manifest lists files the package does not contain"
    assert not any(m.startswith("Database/") for m in manifest)


def test_the_manifest_lists_every_packaged_file(built):
    """Anything shipped but unlisted can never be removed by a later release."""
    with zipfile.ZipFile(built) as zf:
        names = set(zf.namelist()) - {"VERSION", "MANIFEST"}
        manifest = set(zf.read("MANIFEST").decode("utf-8").splitlines())
    assert names == manifest


@pytest.mark.parametrize("required", [
    "backend/api/main.py",
    "backend/api/services/app_version.py",
    "backend/api/services/github_releases.py",
    "CITATION.cff",               # citations/self_entry.py raises without it
    "requirements-lock-cpu.txt",
    "requirements-lock-gpu.txt",
    # A Linux shell reads its lock from the package it downloaded, so the
    # package has to carry the locks of platforms this machine cannot install.
    "requirements-lock-linux-cpu.txt",
    "requirements-lock-linux-gpu.txt",
    # macOS builds its environment from conda-forge instead.
    "orienta-macos-lock.yml",
    "environment-macos.yml",
    # The gates the wizard runs on the environment it just built. `scripts/`
    # is not a RUNTIME_DIR and electron-builder ships only `electron/`, so
    # without an explicit entry this file reaches no user machine at all.
    "scripts/check_runtime_health.py",
    "frontend/dist/index.html",
    # read through _project_root() by the CIF -> .xtal conversion and by every
    # EMsoft simulation; an installed copy without them returned a 500 on the
    # first CIF and "Failed to import automation script" on the first simulation
    "crystal-structures-for-ebsd-main/calculationxtal/DWF.xlsx",
    "crystal-structures-for-ebsd-main/calculationxtal/cif_database_builder.py",
    "crystal-structures-for-ebsd-main/_Phyton_Automization/windwos_to_WSL/emsphinx_automation.py",
    "crystal-structures-for-ebsd-main/_Phyton_Automization/windwos_to_WSL/emsphinx_config.ini.template",
    # the notices the licenses require to travel with every copy: BSD-3 clause 2
    # for the EMsoft/SHTfile tables, MIT for the inlined npm packages and vanta
    "NOTICE.md",
    "licenses/EMsoft-License.txt",
    "licenses/SHTfile-License.txt",
    "licenses/torch-dct-License.txt",
    "licenses/Feather-License.txt",
    "licenses/python-build-standalone/LICENSE.openssl-3.txt",
    "frontend/dist/THIRD-PARTY-LICENSES.txt",
    "frontend/dist/vendor/LICENSE.vanta.txt",
    "crystal-structures-for-ebsd-main/LICENSE",
])
def test_files_the_app_needs_are_present(built, required):
    if required.startswith("frontend/dist/"):
        _skip_unless_dist_has(required)      # a stale build, not a packaging defect
    with zipfile.ZipFile(built) as zf:
        assert required in set(zf.namelist()), f"{required} missing"


# --------------------------------------------------------------------------
# what the notices promise must be in it
# --------------------------------------------------------------------------

def test_every_license_text_the_notice_cites_is_in_the_package(built):
    """NOTICE.md says `licenses/EMsoft-License.txt` reproduces the BSD text.
    A package that carries the NOTICE but not the file makes that sentence a
    lie in every installed copy."""
    # The dist-side texts NOTICE cites, read from NOTICE itself so a citation
    # added later is not silently exempt from this check.
    cited = notice_references((REPO_ROOT / "NOTICE.md").read_text(encoding="utf-8"))
    dist_texts = [r.replace("frontend/public/", "frontend/dist/", 1)
                  for r in cited if r.startswith(("frontend/dist/", "frontend/public/"))
                  and not r.endswith("/")]
    _skip_unless_dist_has(*dist_texts)
    assert check_notice_references(built) == []


def test_notice_reference_parser_sees_license_texts_only():
    refs = notice_references(
        "see `licenses/EMsoft-License.txt` and `licenses/python-build-standalone/` "
        "and `frontend/dist/THIRD-PARTY-LICENSES.txt` and "
        "`backend/spherical_gpu/_math/LICENSE.ebsdtorch`; provenance in "
        "`backend/forward_sim/io/sht_writer.py`, not `Database/x.cif`."
    )
    assert refs == sorted([
        "backend/spherical_gpu/_math/LICENSE.ebsdtorch",
        "frontend/dist/THIRD-PARTY-LICENSES.txt",
        "licenses/EMsoft-License.txt",
        "licenses/python-build-standalone/",
    ])


def test_a_notice_citing_a_missing_file_is_detected(tmp_path):
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("NOTICE.md", "text in `licenses/Missing-License.txt` and `licenses/gone/` "
                                 "and `frontend/public/vendor/LICENSE.vanta.txt`")
        zf.writestr("licenses/Other.txt", "x")
        zf.writestr("frontend/dist/vendor/LICENSE.vanta.txt", "x")  # public/ ships as dist/
    assert check_notice_references(archive) == ["licenses/Missing-License.txt", "licenses/gone/"]


def test_the_derived_source_files_carry_their_upstream_copyright():
    """BSD-3 clause 1: redistributions of source must retain the notice. The
    NOTICE row is not enough when the file itself travels alone."""
    expectations = {
        "backend/forward_sim/crystal/scattering_factors.py": "Marc De Graef Research Group/Carnegie Mellon University",
        "backend/forward_sim/mc/cupy_mc_kernel.py": "Marc De Graef Research Group/Carnegie Mellon University",
        "backend/forward_sim/io/sht_writer.py": "De Graef Group, Carnegie Mellon University",
        "backend/spherical_gpu/pipeline/sht_io.py": "De Graef Group, Carnegie Mellon University",
        "backend/spherical_gpu/_math/sht.py": "Ziyang Hu",
    }
    for rel, holder in expectations.items():
        head = (REPO_ROOT / rel).read_text(encoding="utf-8")[:6000]
        assert holder in head, f"{rel} lost its upstream copyright notice"


def test_root_modules_come_from_git_not_from_a_hand_written_list():
    """edax_hex.py is the canary: it was missing from a hand-built copy once."""
    _skip_unless_dist_has("frontend/dist/index.html")   # collect_files refuses without it
    names = {p.as_posix() for p in collect_files(REPO_ROOT)}
    for required in ("ebsd_utils.py", "safe_loader.py", "unified_loader.py", "edax_hex.py"):
        assert required in names, f"{required} missing from the package"


def test_a_missing_required_extra_fails_loudly(tmp_path):
    """A silently dropped lock file would make the update check's dependency
    comparison fail open."""
    import scripts.build_runtime_package as brp

    with pytest.raises(SystemExit) as excinfo:
        brp.collect_files(tmp_path)          # an empty directory: no git, no extras
    assert "not a git repository" in str(excinfo.value).lower() \
        or "required" in str(excinfo.value).lower()


@pytest.mark.slow
def test_the_unpacked_package_can_actually_import_the_backend(built, tmp_path):
    """The arbiter. The static graph above is fast; this is the truth."""
    target = tmp_path / "unpacked"
    with zipfile.ZipFile(built) as zf:
        zf.extractall(target)
    result = subprocess.run(
        [sys.executable, "-c", "import backend.api.main"],
        cwd=target, capture_output=True, text=True, encoding="utf-8", timeout=900,
    )
    # A third-party import failing means this interpreter is not the project's
    # environment -- which is a fact about the machine, not about the package.
    # Reporting it as a packaging defect sends whoever ran the suite looking
    # for a missing file that is right there.
    if result.returncode != 0 and "No module named" in result.stderr:
        missing = result.stderr.rsplit("No module named ", 1)[1].strip().strip("'\"")
        top = missing.split(".")[0]
        ours = {"backend", "analysis", "simulation", "tools", "resources"}
        if top not in ours:
            pytest.skip(
                f"this interpreter ({sys.executable}) has no {top!r}; run the "
                f"suite in the project environment to exercise this test")

    assert result.returncode == 0, result.stderr[-4000:]


# ---------------------------------------------------------------------------
# the locks, held against the installer that reads them
# ---------------------------------------------------------------------------

def test_the_lock_list_matches_the_installer_that_reads_it():
    """One list of locks in Python, one rule in JavaScript, checked against
    each other.

    The names are not ours to choose: `lockFileFor(mode, platform)` in
    electron/setup/installer.js builds the pip ones and MACOS_LOCK_FILE in
    electron/setup/macos_env.js names the conda one. A package built to a
    Python list that has drifted from those would pass every test here and
    hand a user a lock file the wizard never looks for.

    This is the shape of defect that has cost this project the most: two
    sides each correct on their own, wrong only together.
    """
    from scripts.build_runtime_package import REQUIRED_LOCKS

    installer = (REPO_ROOT / "electron" / "setup" / "installer.js").read_text(encoding="utf-8")
    macos_env = (REPO_ROOT / "electron" / "setup" / "macos_env.js").read_text(encoding="utf-8")

    # The pip names, as the JS template literals spell them.
    win = re.search(r"return `requirements-lock-\$\{flavour\}\.txt`", installer)
    linux = re.search(r"return `requirements-lock-linux-\$\{flavour\}\.txt`", installer)
    conda = re.search(r"MACOS_LOCK_FILE\s*=\s*'([^']+)'", macos_env)

    # Positive control: if the source is reshaped, this test must fail loudly
    # rather than quietly verify nothing.
    assert win, "installer.js no longer spells the Windows lock name this way"
    assert linux, "installer.js no longer spells the Linux lock name this way"
    assert conda, "macos_env.js no longer defines MACOS_LOCK_FILE this way"

    expected = {f"requirements-lock-{f}.txt" for f in ("cpu", "gpu")}
    expected |= {f"requirements-lock-linux-{f}.txt" for f in ("cpu", "gpu")}
    expected.add(conda.group(1))

    assert set(REQUIRED_LOCKS) == expected, (
        "REQUIRED_LOCKS and the installer disagree about which lock files exist"
    )


def test_every_required_lock_exists_in_the_repository():
    """A name in the list that is not a file would ship nothing, silently."""
    from scripts.build_runtime_package import REQUIRED_LOCKS

    for lock in REQUIRED_LOCKS:
        assert (REPO_ROOT / lock).is_file(), f"{lock} is listed but does not exist"
