"""Tests for scripts/port_guard.py — the release-port check.

Two things these tests are careful about:

* Each check must **fire on the real thing**. The anchor case is
  ``Ai_Ml/Detailed_Plan_Ai_integration.md``, which went public with v0.1.0:
  an internal design note that sat inside a shipped module, written in
  English, in no held-back folder. Every check except the planning-document
  one would miss it, and there is a test asserting exactly that, so nobody
  later removes the planning check as redundant.
* Each check must **stay quiet on ordinary work**, because a guard that cries
  at every commit gets skipped at exactly the release it was built for. The
  false-positive tests carry real lines from this repository.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.port_guard import (  # noqa: E402
    check_absolute_path,
    check_addressed_to_us,
    check_blocked_path,
    check_binary,
    check_credential,
    check_db_identifier,
    check_dead_secret,
    check_german,
    check_private_tree_ref,
    check_sample_name,
    check_session_residue,
    check_third_party_name,
    check_withdrawals,
    Finding,
    parse_binaries,
    parse_messages,
    check_plan_tokens,
    check_planning_doc,
    format_report,
    main,
    parse_patch,
    scan,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "port_guard.py"


# --------------------------------------------------------------------------
# patch parsing
# --------------------------------------------------------------------------

SIMPLE_PATCH = """\
diff --git a/backend/api/foo.py b/backend/api/foo.py
index 1111111..2222222 100644
--- a/backend/api/foo.py
+++ b/backend/api/foo.py
@@ -10,6 +10,8 @@ def existing():
     context line
+    added_one()
+    added_two()
     another context
"""


def test_parse_patch_reports_added_lines_with_their_new_numbers():
    added = parse_patch(SIMPLE_PATCH)
    assert list(added) == ["backend/api/foo.py"]
    assert added["backend/api/foo.py"] == [(11, "    added_one()"), (12, "    added_two()")]


def test_parse_patch_counts_line_numbers_across_several_hunks():
    patch = SIMPLE_PATCH + """\
@@ -100,3 +102,4 @@ def later():
     ctx
+    late_addition()
"""
    added = parse_patch(patch)
    assert (103, "    late_addition()") in added["backend/api/foo.py"]


def test_parse_patch_lists_a_deleted_file_with_no_added_lines():
    patch = """\
diff --git a/tasks/plan.md b/tasks/plan.md
deleted file mode 100644
index 3333333..0000000
--- a/tasks/plan.md
+++ /dev/null
@@ -1,2 +0,0 @@
-gone
-also gone
"""
    added = parse_patch(patch)
    assert added == {"tasks/plan.md": []}


def test_parse_patch_follows_a_rename_to_its_new_path():
    patch = """\
diff --git a/old/name.md b/docs/new/name.md
similarity index 90%
rename from old/name.md
rename to docs/new/name.md
--- a/old/name.md
+++ b/docs/new/name.md
@@ -1,2 +1,3 @@
 kept
+new line
"""
    added = parse_patch(patch)
    assert added["docs/new/name.md"] == [(2, "new line")]


def test_parse_patch_ignores_the_no_newline_marker():
    patch = SIMPLE_PATCH + "\\ No newline at end of file\n"
    assert len(parse_patch(patch)["backend/api/foo.py"]) == 2


# --------------------------------------------------------------------------
# 1. blocked paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "tasks/SESSIONS.md",
    "docs/superpowers/specs/2026-09-17-addon-contract-design.md",
    "docs/paper/eds-chemistry-prior-methods.md",
    "docs/archive/ralph/ui_standards_audit.md",
    "screenshots/phase-map.png",
    "research/refframe.py",
    "CLAUDE.md",
    "progress.txt",
    ".claude/settings.json",
])
def test_held_back_paths_are_reported(path):
    found = check_blocked_path(path)
    assert found is not None and found.kind == "blocked-path"


@pytest.mark.parametrize("path", [
    "backend/api/routes/indexing.py",
    "docs/user-guide/Indexing.md",
    "docs/ARCHITECTURE.md",
    "frontend/src/components/EDS/EDSPage.jsx",
    "Ai_Ml/ebsd_ai/config.py",
    "tests/test_eds_regions.py",
])
def test_ordinary_paths_are_not_reported(path):
    assert check_blocked_path(path) is None


def test_a_blocked_file_is_reported_once_and_its_contents_are_not(tmp_path):
    patch = """\
diff --git a/tasks/notes.md b/tasks/notes.md
--- a/tasks/notes.md
+++ b/tasks/notes.md
@@ -0,0 +1,2 @@
+C:\\Users\\someone\\thing
+SP3 said so
"""
    findings = scan(parse_patch(patch))
    assert [f.kind for f in findings] == ["blocked-path"]


# --------------------------------------------------------------------------
# 2. plan tokens
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    '"""GPU-native EBSD forward model (SP0+SP1).',
    "# Task 4 gates every reassignment",
    "# FEAT-10 K-factors land later",
    "# iter-15B measured 0.44",
    "# Pillar 2 of the batch work",
])
def test_plan_tokens_are_reported(line):
    found = check_plan_tokens("backend/x.py", 1, line)
    assert found is not None and found.kind == "plan-token"


@pytest.mark.parametrize("line", [
    "# Phase 1 is the aluminium phase in this map",     # crystal phase, not a plan
    "task_id = uuid4()",
    "for task in queue:",
    "# see the feature table",
    "sp = SphericalIndexer()",
])
def test_ordinary_code_is_not_a_plan_token(line):
    assert check_plan_tokens("backend/x.py", 1, line) is None


# --------------------------------------------------------------------------
# 3. German prose outside the German locale
# --------------------------------------------------------------------------

def test_german_prose_in_a_python_file_is_reported():
    line = "# Die Berechnung wird nicht durchgefuehrt, wenn die Datei fehlt"
    found = check_german("backend/api/x.py", 3, line)
    assert found is not None and found.kind == "german"


def test_german_in_the_german_locale_is_correct_and_quiet():
    line = '  "title": "Die Phasenkarte wird nicht neu berechnet, wenn sie schon da ist",'
    assert check_german("frontend/src/locales/de/phasemap.json", 3, line) is None
    assert check_german("frontend/src/locales/ja/phasemap.json", 3, line) is None


def test_english_with_a_stray_german_looking_word_is_quiet():
    # "die casting" is in a real preset; "ist" appears inside identifiers
    assert check_german("backend/x.py", 1, "# Aluminium-Silicon die casting alloy") is None
    assert check_german("backend/x.py", 1, "hist = compute_histogram(values)") is None
    assert check_german(
        "docs/ARCHITECTURE.md", 1,
        "The layer stack is composed in the browser, not on the server.",
    ) is None


def test_one_german_word_is_not_enough():
    """One is still not enough. Two now is, and that is deliberate.

    The threshold was three only because Aztec's German dataset names
    (Elementverteilungsdaten, Arbeitsbereich, Elektronenbild) appear in every
    file that reads one and pushed ordinary English lines over a lower bar.
    Those names are stripped before counting now, so the bar can come down.
    Measured over 403k lines of this repository's English files: three reports
    213 lines, two reports 375, and every one of the 162 extra lines outside
    the German-language vendored folder is real German prose.
    """
    assert check_german("backend/x.py", 1, "# the der value") is None
    assert check_german("backend/x.py", 1, "# der Wert") is not None


def test_aztec_dataset_names_do_not_make_a_line_german():
    """The whole reason the threshold could be lowered."""
    assert check_german(
        "backend/x.py", 1,
        "    # read the Elementverteilungsdaten of Arbeitsbereich 1") is None


def test_a_minified_bundle_is_not_prose():
    """Token soup trips any word-frequency rule; it was the one measured
    false positive outside the German vendored folder."""
    assert check_german("frontend/public/vendor/three.min.js", 6,
                        '!function(t,e){"object"==typeof exports&&der.wert=e}') is None


# --------------------------------------------------------------------------
# 4. absolute paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    r'    r"E:/Some_Thesis_2026_07/h5oina/"',
    r'foldername = "C:\\CDL_DePIct-Al\\_Wissenschaft"',
    '    path = "/home/sesam/emsoft/builds"',
    '    share = "\\\\\\\\wfsrv5.unileoben.domain\\\\nem"',
])
def test_machine_specific_paths_are_reported(line):
    found = check_absolute_path("tests/x.py", 1, line)
    assert found is not None and found.kind == "absolute-path"


@pytest.mark.parametrize("line", [
    'path = Path(__file__).parents[3] / "Database"',
    'url = "https://github.com/SeSam-MUL/Orienta"',
    'rel = "docs/user-guide/Indexing.md"',
    'env = os.environ.get("EMSOFT_ROOT")',
])
def test_relative_and_derived_paths_are_quiet(line):
    assert check_absolute_path("backend/x.py", 1, line) is None


# --------------------------------------------------------------------------
# 5. planning documents — the anchor case
# --------------------------------------------------------------------------

#: Verbatim from Ai_Ml/Detailed_Plan_Ai_integration.md as it was published.
REAL_PLAN_LINES = [
    "### Phase 1: Foundation (2–3 weeks)",
    "□ Master-pattern generation for α-Al (FCC) with EMsoft/kikuchipy",
    "□ Implement ResNet-18 encoder",
    "□ Integration into kikuchipy_GUI as an indexing backend",
]


@pytest.mark.parametrize("line", REAL_PLAN_LINES)
def test_the_file_that_caused_this_guard_would_be_reported(line):
    """It sat in Ai_Ml/, in English, in no held-back folder."""
    found = check_planning_doc("Ai_Ml/Detailed_Plan_Ai_integration.md", 1, line)
    assert found is not None and found.kind == "planning-doc"


def test_the_other_checks_would_have_missed_it():
    """Stated as a test so nobody later deletes the planning check as redundant."""
    for line in REAL_PLAN_LINES:
        assert check_blocked_path("Ai_Ml/Detailed_Plan_Ai_integration.md") is None
        assert check_german("Ai_Ml/Detailed_Plan_Ai_integration.md", 1, line) is None
        assert check_absolute_path("Ai_Ml/Detailed_Plan_Ai_integration.md", 1, line) is None
        assert check_plan_tokens("Ai_Ml/Detailed_Plan_Ai_integration.md", 1, line) is None
        assert check_addressed_to_us("Ai_Ml/Detailed_Plan_Ai_integration.md", 1, line) is None


def test_markdown_checkboxes_outside_the_changelog_are_reported():
    found = check_planning_doc("backend/README.md", 4, "- [ ] wire the second reader")
    assert found is not None


def test_a_changelog_checklist_is_left_alone():
    assert check_planning_doc("CHANGELOG.md", 4, "- [ ] not yet released") is None


def test_ordinary_prose_about_phases_is_quiet():
    assert check_planning_doc(
        "docs/user-guide/PhaseMap.md", 1,
        "Phase 2 of the map shows the Al7Cu2Fe particles.",
    ) is None


# --------------------------------------------------------------------------
# report and exit status
# --------------------------------------------------------------------------

def test_a_clean_patch_reports_nothing_and_exits_zero(tmp_path, capsys):
    p = tmp_path / "clean.patch"
    p.write_text(SIMPLE_PATCH, encoding="utf-8")
    assert main([str(p)]) == 0
    assert "nothing to report" in capsys.readouterr().out


def test_a_dirty_patch_exits_one_and_names_the_path(tmp_path, capsys):
    p = tmp_path / "dirty.patch"
    p.write_text("""\
diff --git a/docs/paper/methods.md b/docs/paper/methods.md
--- /dev/null
+++ b/docs/paper/methods.md
@@ -0,0 +1,1 @@
+text
""", encoding="utf-8")
    assert main([str(p)]) == 1
    out = capsys.readouterr().out
    assert "docs/paper/methods.md" in out and "blocked-path" in out


def test_report_groups_by_kind():
    findings = scan(parse_patch("""\
diff --git a/backend/a.py b/backend/a.py
--- a/backend/a.py
+++ b/backend/a.py
@@ -1,1 +1,3 @@
 ctx
+# Task 7 does this
+p = r"C:\\Users\\sebas\\thing"
"""))
    text = format_report(findings)
    assert "absolute-path: 1" in text and "plan-token: 1" in text


def test_runs_as_a_command_and_reads_stdin():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-"],
        input=SIMPLE_PATCH, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0
    assert "nothing to report" in proc.stdout


# --------------------------------------------------------------------------
# a patch is bytes
#
# This repository is not all UTF-8: matlab_testskripts/DeformationAnalysis.m
# and TextureAnalysis_mtex6.m are cp1252. A patch that touches one of them
# carries those bytes, including in CONTEXT lines the guard never examines.
# --------------------------------------------------------------------------

#: A patch whose context line holds a cp1252 u-umlaut (0xFC), as the real file
#: does. Built as bytes on purpose: writing it as text would let this file's
#: own encoding decide what the test actually exercises.
CP1252_PATCH = (
    b"diff --git a/matlab_testskripts/DeformationAnalysis.m"
    b" b/matlab_testskripts/DeformationAnalysis.m\n"
    b"--- a/matlab_testskripts/DeformationAnalysis.m\n"
    b"+++ b/matlab_testskripts/DeformationAnalysis.m\n"
    b"@@ -160,2 +160,2 @@\n"
    b" % f\xfcr Belinda\n"
    b"-old line\n"
    b"+new line\n"
)


def test_a_cp1252_byte_in_a_context_line_does_not_kill_the_run():
    """The stdin path used to inherit the console's codec instead of decoding
    like the other two, so the SAME patch was accepted from a file and killed
    the run from a pipe. Under PYTHONIOENCODING=utf-8 it died on one 0xFC in a
    line the guard does not even examine."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(SCRIPT), "-"],
                          input=CP1252_PATCH, capture_output=True, env=env)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    assert "UnicodeDecodeError" not in err, err[-400:]
    assert proc.returncode in (0, 1), f"rc={proc.returncode}: {err[-400:]}"
    assert "nothing to report" in out or "port_guard:" in out


def test_all_three_inputs_read_the_same_bytes_the_same_way(tmp_path):
    """File, --range and stdin must not disagree about a patch. Only the first
    two are compared here; --range builds its own diff and has no patch to be
    handed."""
    f = tmp_path / "cp1252.patch"
    f.write_bytes(CP1252_PATCH)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    from_file = subprocess.run([sys.executable, str(SCRIPT), str(f)],
                               capture_output=True, env=env).stdout
    from_pipe = subprocess.run([sys.executable, str(SCRIPT), "-"],
                               input=CP1252_PATCH, capture_output=True, env=env).stdout
    assert from_file == from_pipe


def test_a_finding_in_a_cp1252_line_is_still_found():
    """Replacement characters cost a rule at most one match, and this one is
    written to survive them: the same file is why check_third_party_name
    tolerates U+FFFD."""
    added = CP1252_PATCH.replace(b"-old line\n", b"").replace(
        b" % f\xfcr Belinda\n", b"+% f\xfcr Belinda\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(SCRIPT), "-"],
                          input=added, capture_output=True, env=env)
    out = proc.stdout.decode("utf-8", errors="replace")
    assert "third-party-name" in out, out[-300:]
    assert proc.returncode == 1


# --------------------------------------------------------------------------
# the in-line waiver
# --------------------------------------------------------------------------

def test_a_line_may_waive_itself_with_a_reason():
    patch = """\
diff --git a/tests/citations/test_no_local_paths.py b/tests/citations/test_no_local_paths.py
--- a/tests/citations/test_no_local_paths.py
+++ b/tests/citations/test_no_local_paths.py
@@ -1,1 +1,3 @@
 ctx
+    r"C:\\Users\\operator\\Orienta\\x.cif",  # port-guard: ok, fixture for the leak test
+    r"C:\\Users\\operator\\Orienta\\y.cif",
"""
    findings = scan(parse_patch(patch))
    assert len(findings) == 1, "only the un-waived line should be reported"
    assert findings[0].line_no == 3


def test_the_waiver_does_not_excuse_a_blocked_path():
    patch = """\
diff --git a/tasks/x.md b/tasks/x.md
--- /dev/null
+++ b/tasks/x.md
@@ -0,0 +1,1 @@
+anything  # port-guard: ok
"""
    findings = scan(parse_patch(patch))
    assert [f.kind for f in findings] == ["blocked-path"]


# --------------------------------------------------------------------------
# 6. lines written to us
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "    # In Ihrer EMsoftXtalGenerator-Klasse in Xtal_Generator_GUI.py:",
    "# Sebastian asked for this; Ralph should wire it next.",
    "that prevents Ralph from silently rewriting the oracle",
    'RESULT_URL = "https://github.com/your-repo"',
    "# Non-square support is planned for Phase B.",
    "# a research-agent should re-extract the EMSphInx formula",
])
def test_lines_addressed_to_us_are_reported(line):
    found = check_addressed_to_us("backend/x.py", 1, line)
    assert found is not None and found.kind == "addressed-to-us"


@pytest.mark.parametrize("line", [
    "# Phase 1 is the aluminium phase in this map",
    "the alpha phase and the S phase share a point group",
    'url = "https://github.com/SeSam-MUL/Orienta"',
])
def test_ordinary_lines_are_not_addressed_to_us(line):
    assert check_addressed_to_us("backend/x.py", 1, line) is None


# --------------------------------------------------------------------------
# regressions from the review of this guard
# --------------------------------------------------------------------------

def test_a_two_digit_plan_number_is_caught():
    """SP[0-9] with a word boundary matches SP3 and cannot match SP10."""
    assert check_plan_tokens("backend/x.py", 1, "# See SP10 of the roadmap") is not None
    assert check_plan_tokens("backend/x.py", 1, "# See SP3 of the roadmap") is not None


def test_a_tab_escape_is_not_a_unc_share():
    """CTF and ANG writers are full of these; reporting them all kills the guard."""
    line = r'    header = "\tPhaseName\tLaueGroup\tSpaceGroup"'
    assert check_absolute_path("Ai_Ml/ebsd_ai/data/ctf_parser.py", 1, line) is None


def test_a_real_unc_share_still_reports():
    line = 'WSL = r"' + "\\" * 2 + 'wsl$' + "\\" + 'Ubuntu' + "\\" + 'home"'
    assert check_absolute_path("backend/x.py", 1, line) is not None
    share = 'p = r"' + "\\" * 2 + 'wfsrv5.unileoben.domain' + "\\" + 'nem"'
    assert check_absolute_path("backend/x.py", 1, share) is not None


def test_a_german_code_comment_with_few_function_words_is_caught():
    """Content words count too: this line has only one function word."""
    line = "        # 2) HDF5-Datei schreiben (dieser Teil bleibt gleich)"
    assert check_german("crystal/Xtal_Generator_GUI.py", 1, line) is not None


def test_installer_translations_are_left_alone():
    line = 'de: "Die Anwendung wird jetzt entfernt, wenn Sie fortfahren",'
    assert check_german("electron/setup/locales.json", 1, line) is None
    assert check_german("electron/nsis/uninstall.nsh", 1, line) is None


def test_a_quoted_path_is_parsed_and_checked():
    """git quotes non-ASCII paths; an unquoted parser skips the file entirely."""
    # raw strings: the patch text must carry the six characters \303\234,
    # not the bytes they stand for
    patch = (
        r'diff --git "a/tasks/\303\234bersicht.md" "b/tasks/\303\234bersicht.md"' + "\n"
        "--- /dev/null\n"
        + r'+++ "b/tasks/\303\234bersicht.md"' + "\n"
        "@@ -0,0 +1,1 @@\n"
        "+text\n"
    )
    added = parse_patch(patch)
    assert "tasks/Übersicht.md" in added
    assert [f.kind for f in scan(added)] == ["blocked-path"]


def test_an_unparsable_diff_header_does_not_charge_lines_to_the_previous_file():
    patch = (
        "diff --git a/backend/ok.py b/backend/ok.py\n"
        "--- a/backend/ok.py\n"
        "+++ b/backend/ok.py\n"
        "@@ -1,1 +1,2 @@\n"
        " ctx\n"
        "diff --git something we do not understand\n"
        "@@ -1,1 +1,2 @@\n"
        '+p = r"C:' + "\\" + 'Users' + "\\" + 'operator' + "\\" + 'secret"\n'
    )
    findings = scan(parse_patch(patch))
    assert not any(f.path == "backend/ok.py" and f.kind == "absolute-path"
                   for f in findings), "a mystery header must not pollute the file before it"


def test_an_unreadable_patch_does_not_read_as_clean(tmp_path, capsys):
    """PowerShell's `git diff > out.patch` writes UTF-16; it must not exit 0."""
    p = tmp_path / "utf16.patch"
    p.write_bytes(SIMPLE_PATCH.encode("utf-16"))
    assert main([str(p)]) == 1
    assert "no diff headers found" in capsys.readouterr().out


def test_a_name_in_capitals_is_still_a_name():
    """A heading "NOTES FOR RALPH (Backend Agent)" must be caught too."""
    assert check_addressed_to_us("x.md", 1, "## 7. NOTES FOR RALPH (Backend Agent)") is not None
    assert check_addressed_to_us("x.md", 1, "written for claude code") is not None


def test_phase_a_stays_case_sensitive():
    """Case-insensitive here would flag ordinary EBSD prose."""
    assert check_addressed_to_us("x.py", 1, "# planned for Phase B") is not None
    assert check_addressed_to_us("x.py", 1, "the phase a particle belongs to") is None


# --------------------------------------------------------------------------
# dead secrets
# --------------------------------------------------------------------------

#: The WSL sudo value, built from its digits rather than written out, so this
#: file does not carry it either. scripts/port_guard.py is the one place in the
#: tree that spells it, and test_the_value_survives_in_no_portable_file below
#: asserts exactly that against the tree rather than against a fixture.
DEAD_VALUE = "".join(chr(ord("0") + n) for n in (1, 2, 3, 4))

#: Only the prefix, which is all the guard needs to recognise the key.
MP_KEY_PREFIX = "Fd6So1"


def test_the_dead_sudo_value_is_caught_where_it_really_appeared():
    """The three lines that carried it. Two of them are public today."""
    for line in (
        f'# Verify pocl.icd contains a valid library path (not garbage like "{DEAD_VALUE}")',
        f'   the OpenCL ICD files. That is the "ICD files contained {DEAD_VALUE}" finding.',
        f'contained {DEAD_VALUE}" report from February: {DEAD_VALUE} was the password.',
    ):
        assert check_dead_secret("x.md", 1, line) is not None, line


def test_the_dead_sudo_value_stays_quiet_on_ordinary_numbers():
    """Four common digits, so the context requirement is the whole design.

    Every line here is real, from this repository. Matching the value on its
    own reports all three, the guard gets waived at the release it was built
    for, and then it protects nothing.
    """
    for line in (
        f"    expect(P.killTree({DEAD_VALUE}, 'win32')).toEqual(...)",
        f'    name = _safe_filename("AlFe<>Si", "COD", "{DEAD_VALUE}")',
        f"    timeout_ms = {DEAD_VALUE}5",
        "# the password is kept by the OS keyring",
    ):
        assert check_dead_secret("x.py", 1, line) is None, line


def test_the_materials_project_key_is_caught_by_its_prefix():
    """No context needed: the prefix alone does not occur by accident."""
    assert check_dead_secret("x.py", 1, f'KEY = "{MP_KEY_PREFIX}abcd1234"') is not None
    assert check_dead_secret("x.py", 1, "KEY = os.environ['MP_API_KEY']") is None


def test_the_report_does_not_print_the_secret_it_reports():
    """A guard that prints the value in order to fix it has moved it, not
    removed it. That is the mistake this rule exists to catch: the installer
    used to echo the broken ICD content, which can BE the password, into
    logs/orienta.log, and that file ships inside the diagnostics zip."""
    found = check_dead_secret("CHANGELOG.md", 99, f"{DEAD_VALUE} was the password")
    assert found is not None
    assert DEAD_VALUE not in found.detail
    assert "<redacted>" in found.detail


def test_the_value_survives_in_no_portable_file():
    """The requirement itself, checked against the tree, not against a fixture.

    Held-back paths are skipped because the guard already blocks them whole,
    and the two files that spell the value on purpose are skipped by name.
    Anything else that matches would travel at the next port.
    """
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    binary = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".h5", ".sht", ".xtal",
              ".zip", ".pdf", ".pptx", ".xlsx", ".woff", ".woff2", ".ttf"}
    spelled_on_purpose = {"scripts/port_guard.py", "tests/test_port_guard.py"}

    offenders: list[str] = []
    for rel in tracked:
        if rel in spelled_on_purpose or check_blocked_path(rel) is not None:
            continue
        path = root / rel
        if path.suffix.lower() in binary:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if check_dead_secret(rel, line_no, line) is not None:
                offenders.append(f"{rel}:{line_no}")

    assert offenders == [], "would travel at the next port: " + ", ".join(offenders[:10])


# --------------------------------------------------------------------------
# the rules added after the 2026-09-25 audit
#
# Every "fires" case below is a line the audit actually found, and every
# "quiet" case is a line that actually occurs in this repository. That pairing
# is the test: a rule that cannot stay quiet gets waived at the release it was
# built for, and then it protects nothing.
# --------------------------------------------------------------------------

def test_a_credential_shaped_literal_is_reported():
    """The key sat in the published tree for a whole release."""
    assert check_credential("x.py", 1, 'MP_API_KEY = "Ab3Cd9Ef2Gh7Jk1Lm"') is not None
    assert check_credential("x.py", 1, 'API_KEY = "Ab3Cd9Ef2Gh7Jk1Lm"') is not None
    assert check_credential("x.js", 1, 'apiToken: "Ab3Cd9Ef2Gh7Jk1Lm"') is not None
    assert check_credential("x.py", 1, 'password = "1234567890123456"') is not None


def test_a_credential_rule_stays_quiet_on_ordinary_code():
    for benign in (
        'STORAGE_KEY = "orienta.theme.v2"',
        'i18nKey = "settings:apiKeys.providers.materialsProject.label"',
        'API_KEY = ""',
        'API_KEY = "YOUR_KEY_HERE"',
        'key = (os.environ.get("MP_API_KEY") or "").strip()',
        'SPLITTER_KEY = "cockpit-splitter-pos"',
    ):
        assert check_credential("x.py", 1, benign) is None, benign


def test_a_person_named_beside_unpublished_work_is_reported():
    """The audit's three person findings, in the shapes they really have."""
    assert check_third_party_name("a.m", 1, "% fuer Belinda") is not None
    assert check_third_party_name("a.m", 1, "%% Fuer paper Flo") is not None
    assert check_third_party_name("a.m", 1, "% Paper GHADIR 1.5 Mn") is not None
    assert check_third_party_name(
        "t.py", 1, 'P = Path(r"E:/Masterarbeit_Musterfrau_2026_07/x.h5oina")') is not None
    assert check_third_party_name("x.py", 1, "contact a.person@other.example.net") is not None


def test_naming_people_in_a_citation_is_not_a_finding():
    """NOTICE and CITATION name people on purpose; so does a licence."""
    for benign in (
        "Bachmann, R. Hielscher, H. Schaeben: Texture Analysis with MTEX",
        "#   John Iversen, 2005-10",
        "# for Windows users, see below",
        "// for Each element in the list",
        "    for Item in items:",
        "Co-Authored-By: someone <noreply@anthropic.com>",
    ):
        assert check_third_party_name("NOTICE.md", 1, benign) is None, benign


def test_an_internal_sample_designation_is_reported():
    """The largest class the audit found: each name is one measurement on one
    specimen, and the numbers beside it are that specimen's."""
    assert check_sample_name("x.py", 1, "    # measured on LoGainNi.h5") is not None
    assert check_sample_name("x.py", 1, "run on Probe B, 3x3 frame average") is not None
    assert check_sample_name("x.py", 1, "path = '70502_t2 area 3'") is not None


def test_aztec_dataset_words_are_not_sample_names():
    """They name a scan region in the vendor's own vocabulary, not a specimen.

    Listing them here contradicted AZTEC_TERMS, which strips the same words
    before the German count, and the guard proved it by reporting the line
    that declares them product vocabulary.
    """
    assert check_sample_name("x.md", 1, "Arbeitsbereich 1 crop, 80x48") is None
    assert check_sample_name("x.py", 1, "read the Elementverteilungsdaten") is None


def test_the_published_datasets_may_be_named():
    """SampleB, Scan1 and AA7050 went out through Zenodo under those names."""
    for benign in (
        "`SampleB_Al-extrusion.h5oina` | Al extrusion alloy",
        "| `Scan1_indexed.h5` | Scan 1 | pre-indexed dataset |",
        "| `AA7050_R_area1.h5oina` | AA7050 (Al-Zn-Mg-Cu) |",
        "for the sample set on disk",
    ):
        assert check_sample_name("sample_data/README.md", 1, benign) is None, benign


def test_a_database_record_id_outside_the_phase_library_is_reported():
    assert check_db_identifier("backend/x.py", 1, "phase sd_0302719 matches") is not None
    assert check_db_identifier("docs/x.md", 1, "see ICSD 99302 for the cell") is not None
    assert check_db_identifier("backend/x.py", 1, "mp-134 is the entry") is not None


def test_the_phase_library_may_carry_record_ids():
    """That is what a provenance file is for."""
    assert check_db_identifier("sample_data/phases/PROVENANCE.md", 1, "sd_0302719") is None
    assert check_db_identifier("backend/x.py", 1, "the sd card was full") is None


def test_a_line_pointing_into_the_private_tree_is_reported():
    """The blocked-path rule asks whether the FILE is private. It never asked
    whether a line points INTO the private tree, which is how a shipped
    docstring came to cite a folder the public repository never had."""
    assert check_private_tree_ref("backend/x.py", 1, "see tasks/todo.md for the plan") is not None
    assert check_private_tree_ref("x.md", 1, "described in docs/paper/methods.md") is not None
    assert check_private_tree_ref("x.py", 1, "load from Test_data/SampleB.h5oina") is not None
    assert check_private_tree_ref("x.py", 1, "the rule is in CLAUDE.md") is not None


def test_the_files_whose_job_is_to_name_those_paths_are_exempt():
    assert check_private_tree_ref(".gitignore", 1, "tasks/") is None
    assert check_private_tree_ref("scripts/port_guard.py", 1, '"tasks/",') is None
    assert check_private_tree_ref("x.py", 1, "the task queue is empty") is None


def test_session_bookkeeping_is_reported():
    assert check_session_residue("x.py", 1, "# 2026-09-25 session: measured 0.4") is not None
    assert check_session_residue("x.py", 1, "# the user reported this on Monday") is not None
    assert check_session_residue("x.py", 1, "# 602 pat/s on my machine") is not None
    assert check_session_residue("x.py", 1, "# Lever A skips the volume") is not None
    assert check_session_residue("x.py", 1, "# HIGH finding from the review") is not None


def test_ordinary_product_prose_is_not_bookkeeping():
    for benign in (
        "## v0.3.0 - 2026-09-14",
        "released 2026-09-14",
        # This is an EBSD application: "Phase 0" is its central noun. The
        # token fired 47 times across the tree with no true positive, while
        # PLAN_PHASE still covers the "Phase A".."Phase D" that means a plan.
        "Phase list - Phase 0 represents 'not indexed'",
        "phase 2 is missing its CIF",
        # a bounding box, not a machine
        "The auto-zoom draws the map inside this box",
        # a shipped UI string, in four languages
        "Not detected on this machine",
        "stage a of the solver runs first",
        "the phase map shows four phases",
        "Copyright (c) 2026",
    ):
        assert check_session_residue("CHANGELOG.md", 1, benign) is None, benign


# --------------------------------------------------------------------------
# what a line-by-line reading cannot see

MESSAGE_PATCH = """From abc1234 Mon Sep 25 12:00:00 2026
From: Someone <s@example.com>
Date: Thu, 25 Sep 2026 12:00:00 +0200
Subject: [PATCH 1/2] fix: the thing measured on LoGainNi

Sebastian said the numbers looked wrong, see tasks/todo.md.

Co-Authored-By: Claude <noreply@anthropic.com>
---
 a.py | 2 +-
 1 file changed

diff --git a/a.py b/a.py
index 111..222 100644
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-old
+new
"""


def test_the_commit_message_is_read_at_all():
    """18 of the audit's findings exist only in a message; the guard read none
    of them, because it only ever looked at patch lines."""
    msgs = parse_messages(MESSAGE_PATCH)
    text = " ".join(l for _, l in msgs["<commit message>"])
    assert "fix: the thing measured on LoGainNi" in text
    assert "tasks/todo.md" in text
    kinds = {f.kind for f in scan(msgs)}
    assert "sample-name" in kinds
    assert "private-tree-ref" in kinds


def test_git_trailers_are_skipped():
    """Not cosmetic. Co-Authored-By names Claude, so without this every single
    message trips `addressed-to-us` and the check reports 100% of commits,
    which carries exactly as much information as reporting none."""
    msgs = parse_messages(MESSAGE_PATCH)
    text = " ".join(l for _, l in msgs["<commit message>"])
    assert "Co-Authored-By" not in text
    assert "addressed-to-us" not in {f.kind for f in scan(msgs)}


def test_the_message_check_ends_at_the_diff():
    msgs = parse_messages(MESSAGE_PATCH)
    text = " ".join(l for _, l in msgs["<commit message>"])
    assert "diff --git" not in text and "+new" not in text


BINARY_PATCH = """diff --git a/docs/report.docx b/docs/report.docx
index 111..222 100644
Binary files a/docs/report.docx and b/docs/report.docx differ
diff --git a/tests/fixtures/scan.npz b/tests/fixtures/scan.npz
new file mode 100644
index 000..333
GIT binary patch
literal 12
"""


def test_a_binary_is_reported_with_the_question_it_needs():
    paths = parse_binaries(BINARY_PATCH)
    assert paths == ["docs/report.docx", "tests/fixtures/scan.npz"]
    docx = check_binary("docs/report.docx")
    npz = check_binary("tests/fixtures/scan.npz")
    assert docx is not None and "lastModifiedBy" in docx.detail
    assert npz is not None and "which specimen" in npz.detail
    assert check_binary("backend/api/main.py") is None


def test_the_target_repo_path_survives_a_drive_letter(tmp_path, monkeypatch):
    """`F:/repo:HEAD` split at the FIRST colon gives the repository "F", and
    then everything fails silently, because a repository that cannot be read
    looks exactly like a patch with nothing to report."""
    seen = {}

    def fake_run(cmd, **kw):
        if cmd[3] == "rev-parse":          # the up-front validation
            seen["repo"] = cmd[2]
            seen["rev"] = cmd[5].split("^")[0]   # ["git","-C",repo,"rev-parse","--verify",rev]
            class Ok:
                returncode = 0
                stdout = "deadbeef"
                stderr = ""
            return Ok()
        class R:
            returncode = 1
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr("scripts.port_guard.subprocess.run", fake_run)
    patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
             "@@ -1,1 +1,0 @@\n-a line long enough to count\n")
    check_withdrawals(patch, "F:/orienta-beta:HEAD")
    assert seen["repo"] == "F:/orienta-beta"
    assert seen["rev"] == "HEAD"


@pytest.mark.parametrize("target,repo,rev", [
    ("F:/orienta-beta", "F:/orienta-beta", "HEAD"),
    ("F:/orienta-beta:HEAD", "F:/orienta-beta", "HEAD"),
    # a revision may contain a slash, and origin/main is the obvious thing to
    # pass for a published repository. Rejecting it sent the whole string in
    # as the repository path, every git call failed, and the check reported
    # nothing -- which is what a clean patch set also looks like.
    ("F:/orienta-beta:origin/main", "F:/orienta-beta", "origin/main"),
    ("F:/orienta-beta:v0.2.6..origin/main", "F:/orienta-beta", "v0.2.6..origin/main"),
    ("/home/x/repo:feature/y..HEAD", "/home/x/repo", "feature/y..HEAD"),
])
def test_the_target_is_split_at_the_right_colon(target, repo, rev):
    from scripts.port_guard import _split_target
    assert _split_target(target) == (repo, rev)


def test_an_unreadable_target_fails_loudly(monkeypatch):
    """Silence here is indistinguishable from a clean patch set, which is the
    trap the docstring names. One rev-parse closes it."""
    def fake_run(cmd, **kw):
        class R:
            returncode = 128
            stdout = ""
            stderr = "fatal: Needed a single revision"
        return R()

    monkeypatch.setattr("scripts.port_guard.subprocess.run", fake_run)
    patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
             "@@ -1,1 +1,0 @@\n-a line long enough to count\n")
    with pytest.raises(SystemExit) as e:
        check_withdrawals(patch, "F:/orienta-beta:nosuchrev")
    assert "nosuchrev" in str(e.value)


def test_a_withdrawal_needs_the_line_to_be_newer_in_the_target(monkeypatch):
    """Without a base every ordinary edit qualifies: replacing a line is also
    deleting it. Measured on the real v0.3.0 port, the bare form reports 61
    files; with the base it reports 4, three of which are the real ones."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            returncode = 0
            # `show` returns the file, `diff` returns what the target added
            stdout = ("kept line that is long enough\n"
                      if cmd[3] == "show" else "+++ b/a.py\n+unrelated addition here\n")
        return R()

    monkeypatch.setattr("scripts.port_guard.subprocess.run", fake_run)
    patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
             "@@ -1,1 +1,0 @@\n-kept line that is long enough\n")
    # the deleted line is in the target, but the target did not add it in the
    # range, so it is an ordinary edit and not a withdrawal
    assert check_withdrawals(patch, "F:/repo:base..HEAD") == []


def test_comments_count_as_withdrawals(monkeypatch):
    """The first draft skipped comment lines and missed all three real
    withdrawals: the key resolver was mostly its own explanation, the licence
    correction WAS a comment, and the DOI was a line of prose."""
    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = ("# the key is never stored in this file\n"
                      if cmd[3] == "show" else
                      "+++ b/a.py\n+# the key is never stored in this file\n")
        return R()

    monkeypatch.setattr("scripts.port_guard.subprocess.run", fake_run)
    patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
             "@@ -1,1 +1,0 @@\n-# the key is never stored in this file\n")
    found = check_withdrawals(patch, "F:/repo:base..HEAD")
    assert len(found) == 1 and found[0].kind == "withdrawal"


def test_a_new_rule_can_never_make_findings_disappear():
    """format_report groups by kind. A kind missing from the order would be
    dropped from the printed report while still counting in the total."""
    odd = Finding("a-kind-nobody-listed", "x.py", 1, "detail")
    out = format_report([odd])
    assert "a-kind-nobody-listed" in out or "other" in out
    assert "detail" in out


def test_the_possessive_is_residue_except_where_the_reader_is_addressed():
    """The possessive is the signal, and that was measured rather than guessed.

    A NOTICE says software is "installed independently by the user" and means
    it; an EMsoft template asks whether "the user" should be e-mailed. Flagging
    every "the user" reported both. But dropping to a verb list alone lost six
    real findings from the audit corpus -- "at the user's real scan size", "in
    the user's words". The possessive recovers five of those six and leaves
    both false positives quiet.
    """
    for benign in ("MTEX must be installed independently by the user.",
                   "! should the user be notified by email when it finishes?",
                   "the user interface is available in four languages"):
        assert check_session_residue("NOTICE.md", 1, benign) is None, benign
    # the possessive in a document written TO the reader
    for benign in ("micromamba is unpacked onto the user's machine",
                   "installed on the user's machine"):
        for doc in ("NOTICE.md", "licenses/README.md", "docs/user-guide/EDS.md"):
            assert check_session_residue(doc, 1, benign) is None, f"{doc}: {benign}"
    # the same words in code are someone describing a session
    for residue in ("# Measured 3.5 s at the user's real scan size",
                    "Why this exists, in the user's words: EDS is ambiguous",
                    "# the user reported this on Monday"):
        assert check_session_residue("backend/x.py", 1, residue) is not None, residue


def test_the_name_rule_sees_the_file_as_it_is_on_disk():
    """DeformationAnalysis.m is cp1252, and read_sources decodes patches as
    UTF-8 with errors="replace", so the u-umlaut arrives as U+FFFD.

    The first version of this test asserted on an "ue" transliteration that
    does not occur in that file, so it certified a rule that would not have
    caught the line the branch exists to remove. This is the byte sequence
    the guard really receives.
    """
    as_the_guard_sees_it = "% f\ufffdr Belinda"
    assert check_third_party_name("matlab_testskripts/DeformationAnalysis.m", 160,
                                  as_the_guard_sees_it) is not None
    assert check_third_party_name("a.m", 1, "%% F\ufffdr paper Flo") is not None


def test_a_moved_line_is_not_a_withdrawal(monkeypatch):
    """Deleting a line in one place and adding it in another is a move."""
    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = ("moved line that is long enough\n" if cmd[3] == "show"
                      else "+++ b/a.py\n+moved line that is long enough\n")
        return R()

    monkeypatch.setattr("scripts.port_guard.subprocess.run", fake_run)
    patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
             "@@ -1,2 +1,2 @@\n-moved line that is long enough\n"
             " context\n+moved line that is long enough\n")
    assert check_withdrawals(patch, "F:/repo:base..HEAD") == []


def test_the_author_line_is_read():
    """git am carries From: into the public commit, so a third party's
    address in it is published as surely as one in the body."""
    patch = ("From abc Mon Sep 25 12:00:00 2026\n"
             "From: Someone Else <third.party@elsewhere.org>\n"
             "Subject: [PATCH] a change\n\nbody\n---\n")
    text = " ".join(l for _, l in parse_messages(patch)["<commit message>"])
    assert "third.party@elsewhere.org" in text


def test_an_encoded_subject_is_decoded_before_it_is_read():
    """git writes RFC 2047 and git am turns it back into German in the public
    commit. Scanning the encoded form destroys every word with an umlaut,
    which is most of what makes a German sentence recognisable."""
    patch = ("From abc Mon Sep 25 12:00:00 2026\n"
             "Subject: [PATCH] =?UTF-8?q?Der=20W=C3=A4chter=20pr=C3=BCft=20die?=\n"
             " =?UTF-8?q?=20Gr=C3=B6=C3=9Fe?=\n\nbody\n---\n")
    text = " ".join(l for _, l in parse_messages(patch)["<commit message>"])
    assert "W\u00e4chter" in text and "Gr\u00f6\u00dfe" in text
    assert "=?UTF-8?q?" not in text


def test_the_guard_does_not_report_itself():
    """Both files spell out every pattern they hunt for. Without this the
    guard reported itself 56 times on the commit that added the rules, and a
    report nobody reads protects nothing. The secret rules still run."""
    line = '    r"|LoGainNi|HiGainNi|Probe B|tasks/|sd_0302719"'
    for path in ("scripts/port_guard.py", "tests/test_port_guard.py"):
        assert not scan({path: [(1, line)]}), path
    # but a real credential in those files is still reported
    assert scan({"scripts/port_guard.py": [(1, 'MP_API_KEY = "Ab3Cd9Ef2Gh7Jk1Lm"')]})


def test_the_two_credential_guards_agree():
    """The port guard's credential patterns are a copy of the ones in
    tests/test_no_hardcoded_secrets.py, and they had already diverged: this
    copy applied the placeholder list to the NAME as well, which that file's
    docstring says must not be done, because API_SECRET contains "secret"."""
    import scripts.port_guard as pg
    import tests.test_no_hardcoded_secrets as secrets
    assert pg.CREDENTIAL_ASSIGNMENT.pattern == secrets.SECRET_ASSIGNMENT.pattern
    assert pg.CREDENTIAL_BENIGN_NAME.pattern == secrets.BENIGN_NAME.pattern
