"""Without a readable config, a simulation must be refused — and it already is.

`_load_config` sets `self.config = None` when the .ini cannot be read. The
worker would then walk on: `_apply_params_to_config` returns **silently** (so
the parameters are never applied) and the next config access dies with

    AttributeError: 'NoneType' object has no attribute 'has_section'

several steps past the point where the cause was already known.

**That is not what a user hits today, and these tests say why.** The port review
reported it as a live defect; it is not. `start_simulation` refuses on the same
condition before it creates the job (`test_start_simulation_refuses_first`), it
is the only caller that starts the worker, and nothing after `__init__` can set
`self.config` back to `None`. `_require_config` is therefore a second guard at
the point of use, for a future entry point that does not go through
`start_simulation` — kept because it costs one comparison, pinned here so nobody
mistakes it for the fix to an observed crash.

Because both guards answer the same condition, they must not offer two
different remedies; `test_both_guards_give_the_same_remedy` holds them together.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from simulation.simulation_controller import SimulationController


def _controller(tmp_path: Path) -> SimulationController:
    """A controller whose config failed to load, without running __init__.

    __init__ is skipped on purpose: it copies the template when the .ini is
    missing, which is the very state under test.
    """
    c = SimulationController.__new__(SimulationController)
    c.config = None
    c.config_path = tmp_path / "emsphinx_config.ini"
    return c


def test_it_raises_instead_of_attributeerror(tmp_path):
    c = _controller(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        c._require_config()
    assert not isinstance(exc.value, AttributeError)


def test_the_message_names_the_file_it_could_not_read(tmp_path):
    c = _controller(tmp_path)
    with pytest.raises(RuntimeError, match=r"emsphinx_config\.ini"):
        c._require_config()


def test_it_does_not_claim_nothing_was_started(tmp_path):
    """It cannot know that.

    Where this guard can fire at all, `start_simulation` has already created
    the job, set it RUNNING and written a log line. EMsoft was not started; a
    job was. An earlier draft of this message promised "nothing was started",
    which would have been a false reassurance in a dialog.
    """
    c = _controller(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        c._require_config()
    text = str(exc.value)
    assert "Nothing was started" not in text
    assert "no partial output" not in text


def test_both_guards_give_the_same_remedy(tmp_path):
    """One condition, one remedy.

    `start_simulation` and `_require_config` both refuse when `config is None`.
    If their wording drifts apart, a user meets two different instructions for
    the same problem depending on which one fired.
    """
    c = _controller(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        c._require_config()
    mine = str(exc.value)

    earlier = inspect.getsource(SimulationController.start_simulation)
    assert "No configuration loaded. Please check:" in earlier
    for line in ("No configuration loaded. Please check:",
                 "Open Simulation Settings to configure paths."):
        assert line in mine, f"the two guards disagree on: {line}"


def test_a_loaded_config_passes_through(tmp_path):
    """The guard must be invisible when there is a config."""
    import configparser

    c = _controller(tmp_path)
    c.config = configparser.ConfigParser()
    assert c._require_config() is None


def test_the_run_path_calls_the_guard_before_touching_config():
    """Pin the ORDER: the guard has to come before the first config access.

    Without this, someone can keep the guard and move it below
    `self.config.has_section(...)`, which is exactly where the AttributeError
    would come from, and every test above still passes.
    """
    src = inspect.getsource(SimulationController._run_simulation_thread)
    guard = src.index("_require_config()")
    first_use = min(
        (src.index(t) for t in ("self.config.has_section", "self.config.set",
                                "self._apply_params_to_config")
         if t in src),
        default=len(src),
    )
    assert first_use < len(src), "premise: the worker really does use the config"
    assert guard < first_use, "the guard must run before the first config use"


def test_start_simulation_refuses_first(tmp_path):
    """Why `_require_config` is unreachable today — and must stay unreachable.

    This is the guard a user actually meets. It runs before the job exists, so
    nothing is created and no worker starts. If someone ever removes it, this
    test fails and `_require_config` stops being belt-and-braces.
    """
    c = _controller(tmp_path)
    c.jobs = {}
    c.job_counter = 0
    xtal = tmp_path / "Al.xtal"
    xtal.write_bytes(b"not a real xtal, never read: the guard fires first")

    with pytest.raises(RuntimeError, match="No configuration loaded"):
        c.start_simulation(str(xtal))

    assert c.jobs == {}, "no job may be created when there is no config"
    assert c.job_counter == 0


def test_only_start_simulation_starts_the_worker():
    """The reachability claim in this module's docstring, pinned.

    `_require_config` is dead code only as long as the worker has exactly one
    launcher. A second `Thread(target=self._run_simulation_thread, ...)`
    anywhere makes the guard live — and this test fail, so the docstring above
    gets corrected instead of quietly going stale.
    """
    src = inspect.getsource(SimulationController)
    assert src.count("target=self._run_simulation_thread") == 1
