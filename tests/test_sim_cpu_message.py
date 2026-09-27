"""The CPU Monte-Carlo message must not read like a failure on a Mac.

Mac tester (M3/M5), 2026-09-25: "something was up with PyTorch during the
simulation — is that installed?" It was installed and the run succeeded; what
they read was

    WARNING: no CUDA device — CPU MC path (EMsoft-free): using OUR numba/PyTorch
    CPU MC (the PyTorch loop is ~400x slower than the cupy kernel), capped ...

On a machine with no NVIDIA card that is the normal path, and the UI colours
any line containing "WARNING" orange (SimulationPage.jsx:1451), so the normal
path looked broken.

The two causes are now separate, because they are not equally serious:

  * no CUDA device      -> information. Nothing to act on.
  * CUDA but no cupy    -> still a warning. The user owns a GPU that is idle
                           and can install cupy to use it.

These tests read the source rather than running a Monte-Carlo simulation: the
branch needs a GPU probe, torch, and minutes of compute, and what is under
test is the wording, not the physics.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "backend/api/services/gpu_sim_runner.py"
SIM_PAGE = (Path(__file__).resolve().parents[1]
            / "frontend/src/components/Simulation/SimulationPage.jsx")


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8", errors="replace")


def _cpu_fallback_block() -> str:
    """The `if not _is_cuda: ... else: ...` pair that reports the CPU path."""
    src = _source()
    start = src.index("n_sim_d: int = min(max(1, totnum_el), GPU_MC_MAX_ELECTRONS)")
    end = src.index("mc_data = _run_our_mc(\"CPU MC\"", start)
    return src[start:end]


class TestTheNoGpuCaseIsInformation:
    def test_it_does_not_say_warning(self):
        block = _cpu_fallback_block()
        no_cuda = block[block.index("if not _is_cuda:"):block.index("else:")]
        assert "WARNING" not in no_cuda, (
            "the normal path on a Mac must not be coloured orange by "
            "SimulationPage's substring test:\n" + no_cuda
        )

    def test_it_names_the_engine_in_plain_words(self):
        no_cuda = _cpu_fallback_block()
        no_cuda = no_cuda[no_cuda.index("if not _is_cuda:"):no_cuda.index("else:")]
        # The tester's question was "is PyTorch installed?" — the answer has to
        # be in the sentence itself. "Orienta Engine" is the one name for the
        # built-in simulator, pinned by tests/test_engine_name_guard.py.
        assert "the Orienta Engine" in no_cuda
        assert "PyTorch/numba" in no_cuda
        assert "no CUDA GPU" in no_cuda

    def test_it_still_says_how_many_electrons(self):
        """The cap changes the result, so it must stay visible."""
        no_cuda = _cpu_fallback_block()
        no_cuda = no_cuda[no_cuda.index("if not _is_cuda:"):no_cuda.index("else:")]
        assert "n_sim_d:,} electrons" in no_cuda

    def test_it_logs_at_info_level(self):
        no_cuda = _cpu_fallback_block()
        no_cuda = no_cuda[no_cuda.index("if not _is_cuda:"):no_cuda.index("else:")]
        assert "logger.info(" in no_cuda
        assert "logger.warning(" not in no_cuda


class TestTheIdleGpuCaseStillWarns:
    def test_cuda_without_cupy_keeps_the_warning(self):
        """Not a regression: here there really is something to fix."""
        block = _cpu_fallback_block()
        else_branch = block[block.index("else:"):]
        assert "WARNING:" in else_branch
        assert "logger.warning(" in else_branch
        assert "cupy" in else_branch
        # the measured figure belongs here, where the comparison is real
        assert "400x" in else_branch


class TestNoOtherScaryTorchLineInThisPath:
    def test_warning_appears_exactly_once_in_the_runner(self):
        """b9 asked whether more torch messages read like errors. One left, and
        it is the idle-GPU one above."""
        hits = re.findall(r'"WARNING[: ]', _source())
        assert len(hits) == 1, f"{len(hits)} WARNING strings in gpu_sim_runner.py"

    def test_no_user_facing_line_calls_torch_a_problem(self):
        src = _source()
        for m in re.finditer(r"log_cb\(\s*(?:f?\"[^\"]*\"\s*)+\)", src):
            text = m.group(0)
            if "torch" in text.lower() or "pytorch" in text.lower():
                assert not re.search(r"\b(error|failed|broken|missing)\b", text, re.I), text


class TestTheProgressLineIsAlsoPlainWords:
    """The job status the user watches, which log_cb scanning does not reach.

    `progress_cb(..., "Monte Carlo (CPU, EMsoft-free): running…")` travels to
    job.progress_message and is rendered by SimulationPage for the whole slow
    step. A review found "EMsoft-free" still there two lines below the message
    this file was written about — the same internal vocabulary, in the string
    that is on screen longest.
    """

    def test_no_emsoft_free_in_a_string_the_user_sees(self):
        src = _source()
        offenders = [m.group(0) for m in re.finditer(
            r'_run_our_mc\([^)]*EMsoft-free[^)]*\)', src)]
        assert not offenders, offenders

    def test_the_progress_strings_name_the_engine_instead(self):
        src = _source()
        # the CPU line and its GPU twin, both in the pinned spelling
        assert src.count("Monte Carlo (CPU, Orienta Engine)") == 1
        assert src.count("Monte Carlo (GPU, Orienta Engine)") == 1


class TestTheUiColourRuleIsWhatWeThinkItIs:
    def test_the_page_still_colours_by_substring(self):
        """If this ever changes, the reason for the split above changes too."""
        page = SIM_PAGE.read_text(encoding="utf-8", errors="replace")
        assert "line.includes('WARNING')" in page


class TestTheModuleStillParses:
    def test_ast(self):
        ast.parse(_source())
