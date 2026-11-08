"""The contract of the pattern-centre cost, in both PyEBSDIndex versions Orienta supports
(0.3.9.1 and 0.3.10.1). Run it in both."""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("pyebsdindex")

from tests import pc_synth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRUE_PC = (0.52, 0.31, 0.62)          # the PC the test renders the patterns at
SHAPE = (80, 80)


@pytest.fixture(scope="module")
def stack():
    with pc_synth.cpu_band_detection():
        pats = pc_synth.render_patterns(SHAPE, TRUE_PC, n=9, seed=7)
        indexer = pc_synth.make_indexer(SHAPE)
        from backend.api.services.pc_cost import JointPcCost
        yield pats, indexer, JointPcCost(indexer, pats)


# ------------------------------------------------------------------ shape and repeatability

def test_one_pc_gives_a_float_and_several_give_an_array(stack):
    _, _, cost = stack
    one = cost(TRUE_PC)
    assert isinstance(one, float)
    many = cost(np.array([TRUE_PC, (0.5, 0.5, 0.5)]))
    assert many.shape == (2,) and many[0] == pytest.approx(one, abs=1e-12)


def test_the_same_pc_costs_the_same_every_time(stack):
    """PyEBSDIndex 0.3.10.1 corrects the bands of its ``banddat`` argument in place, so a cost
    evaluated on one shared array drifts from call to call (0.12314, 0.12300, 0.12916 in the
    measurement this module's docstring quotes). Every evaluation here gets its own copy."""
    _, _, cost = stack
    pc = np.array(TRUE_PC) + np.array([0.0029, -0.0049, 0.0019])
    values = [cost(pc) for _ in range(4)]
    assert values == [values[0]] * 4


def test_the_cost_does_not_change_the_detected_bands(stack):
    _, _, cost = stack
    before = cost._banddat.copy()
    cost(TRUE_PC)
    cost(np.array([TRUE_PC, (0.6, 0.4, 0.7)]))
    assert before.tobytes() == cost._banddat.tobytes()


# ------------------------------------------------------------------ the minimum is where the test put it

def test_the_cost_is_lowest_at_the_pc_the_patterns_were_rendered_at(stack):
    """By construction (the test rendered the patterns at TRUE_PC): displacing the PC by 0.05 along
    any axis, or starting from the placeholder (0.5, 0.5, 0.5), must cost clearly more."""
    _, _, cost = stack
    at_truth = cost(TRUE_PC)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            moved = np.array(TRUE_PC)
            moved[axis] += sign * 0.05
            assert cost(moved) > at_truth + 1.0, (axis, sign)
    assert cost((0.5, 0.5, 0.5)) > at_truth + 1.0


def test_a_single_pattern_cost_uses_the_same_bands(stack):
    pats, indexer, cost = stack
    from backend.api.services.pc_cost import JointPcCost
    alone = JointPcCost(indexer, pats[3:4])
    assert cost.subset(3)(TRUE_PC) == pytest.approx(alone(TRUE_PC), abs=1e-12)
    assert cost.subset(3).n_patterns == 1
    with pytest.raises(IndexError):
        cost.subset(9)


def test_the_private_cost_returns_one_number_per_call(stack):
    """``JointPcCost`` takes the first element of whatever the private function returns
    (``np.atleast_1d(value)[0]``), which would hide a release that returns one value per pattern.
    So the shape is pinned here, on the raw call: one PC of shape (3,) gives exactly one finite
    number (shape (1,) in 0.3.9.1, 0-d in 0.3.10.1)."""
    _, indexer, cost = stack
    from backend.api.services.pc_cost import _private_cost
    value = _private_cost()(np.array(TRUE_PC, dtype=np.float64), indexer=indexer,
                            banddat=cost._banddat.copy())   # 0.3.10.1 corrects banddat in place
    assert np.size(value) == 1, np.shape(value)
    assert np.isfinite(float(np.asarray(value).reshape(-1)[0]))


def test_bad_input_is_refused(stack):
    _, _, cost = stack
    with pytest.raises(ValueError):
        cost((0.5, 0.5))
    from backend.api.services.pc_cost import JointPcCost
    with pytest.raises(ValueError):
        JointPcCost(stack[1], np.zeros((4,)))


# ------------------------------------------------------------------ the swarm still takes this cost

def test_the_swarm_class_accepts_the_cost_in_both_calling_styles(stack):
    """0.3.9.1 calls the function once per particle with a (3,) PC; 0.3.10.1 once per iteration with
    all particles as (n, 3). Both must work, and the swarm must stay inside its box."""
    _, _, cost = stack
    from pyebsdindex.pcopt import PSOOpt
    lo, hi = np.array([0.45, 0.25, 0.55]), np.array([0.6, 0.4, 0.7])
    swarm = PSOOpt(dimensions=3, n_particles=8, c1=3.5, c2=3.5, w=0.8,
                   hyperparammethod="auto", early_exit=1e-4)
    np.random.seed(3)
    best, where = swarm.optimize(cost, start=np.array(TRUE_PC), bounds=(lo, hi), niter=4, verbose=0)
    assert np.isfinite(float(best))
    where = np.asarray(where, dtype=float).reshape(-1)
    assert where.shape == (3,) and np.all(where >= lo - 1e-9) and np.all(where <= hi + 1e-9)


def test_the_swarm_surface_orienta_relies_on_is_still_there():
    from pyebsdindex.pcopt import PSOOpt
    init = inspect.signature(PSOOpt.__init__).parameters
    for name in ("dimensions", "n_particles", "c1", "c2", "w", "hyperparammethod", "early_exit"):
        assert name in init, name
    run = inspect.signature(PSOOpt.optimize).parameters
    for name in ("function", "start", "bounds", "niter", "verbose"):
        assert name in run, name


def test_find_bands_takes_a_stack_in_this_version(stack):
    pats, indexer, _ = stack
    found = indexer.bandDetectPlan.find_bands(pats)
    assert found.shape[0] == pats.shape[0] and found.ndim == 2


# ------------------------------------------------------------------ nobody else reaches for the private name

PRIVATE_COST = re.compile(r"_optfunction")      # no word boundary: also ``_optfunction_planar``


def _private_cost_offenders(files, reader):
    """The files of ``files`` (posix paths relative to ROOT) that mention the private cost, except
    this module's one home, the tests, and the measurement scripts (which are never shipped)."""
    home = "backend/api/services/pc_cost.py"
    return [f for f in files
            if f != home and f.split("/")[0] not in ("tests", "tasks") and PRIVATE_COST.search(reader(f))]


def _project_python_files():
    """Every Python file git knows, tracked or not yet added (but not ignored)."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.py"],
                         cwd=ROOT, check=True, capture_output=True).stdout
    files = sorted({p for p in out.decode("utf-8").split("\0") if p})
    assert len(files) > 100, "git ls-files found no project: is this a checkout?"
    return files


def test_only_this_module_touches_the_private_cost():
    def read(f):
        try:
            return (ROOT / f).read_text(encoding="utf-8", errors="replace")
        except OSError:                              # listed by git but deleted in the work tree
            return ""
    assert _private_cost_offenders(_project_python_files(), read) == []


def test_the_scan_pattern_catches_the_planar_variant_and_the_import_forms():
    sample = {
        "scripts/a.py": "from pyebsdindex.pcopt import _optfunction_planar\n",
        "simulation/b.py": "from pyebsdindex import pcopt\npcopt._optfunction(pc)\n",
        "Ai_Ml/c.py": "import pyebsdindex.pcopt as p\nf = getattr(p, '_optfunction')\n",
        "tools/d.py": "x = 1\n",
        "tasks/e.py": "from pyebsdindex.pcopt import _optfunction\n",
        "tests/f.py": "_optfunction\n",
        "backend/api/services/pc_cost.py": "pcopt._optfunction\n",
    }
    assert _private_cost_offenders(sorted(sample), sample.__getitem__) == [
        "Ai_Ml/c.py", "scripts/a.py", "simulation/b.py"]
