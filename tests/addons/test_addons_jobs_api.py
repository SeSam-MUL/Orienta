"""The run as a polled job: the same work, answered before it is finished.

Two properties carry this file and are tested directly rather than implied:

* **a refusal is not a job.** Every gate answers synchronously, so "the add-on
  is not enabled" arrives as a 409 with its reason and no job is created. A
  refusal turned into a failed job would have to be polled to discover that
  nothing was ever going to happen, and would be counted against the add-on in
  any statistic that reads job states;
* **the job route and the synchronous route are the same run.** Asserted by
  calling both for the same inputs and comparing what comes back, not by
  pinning a literal on the job side that the other side could drift away from.

The worker runs in a real thread, so every test that waits for one uses
``_await_job``: a bounded poll, never a sleep, and it fails naming the job's
last state rather than hanging the suite.
"""
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import addons as addons_route
from backend.api.routes import indexing as indexing_routes
from backend.api.routes.addons import ADDON_FAILED_STATUS
from backend.api.services.addons import jobs

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


@pytest.fixture(autouse=True)
def _addon_dirs(monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "runnable_addon"))


class _XMap:
    def __init__(self, values):
        self.prop = {"bc": values.ravel()}


class _Result:
    """The smallest stored result the context builder accepts.

    A quality channel is present because the fixture add-on only calls
    ``context.report`` on the branch that has one -- without it the progress
    test would assert against an add-on that never reports.
    """

    def __init__(self):
        self.original_shape = (3, 4)
        self.metadata = {}
        self.xmap = _XMap(np.arange(12, dtype=float))


@pytest.fixture
def registered_result():
    indexing_routes._result_registry["addon-job-test"] = _Result()
    yield "addon-job-test"
    indexing_routes._result_registry.pop("addon-job-test", None)


@pytest.fixture(autouse=True)
def _clear_map_store():
    addons_route._MAP_VALUES.clear()
    yield
    addons_route._MAP_VALUES.clear()


def _enable():
    client.post("/api/addons/runnable/enabled", json={"enabled": True})


def _start(result_id, params=None, key="addon.mean_quality"):
    return client.post("/api/addons/runnable/run-job", json={
        "analysis_key": key, "result_id": result_id, "params": params or {}})


def _await_job(job_id, timeout_s=20.0):
    """Poll until the job leaves ``running``; fail naming what it was doing.

    A bounded poll and not a sleep: a fixed sleep is either longer than the
    suite can afford or shorter than a loaded machine needs, and it turns a
    worker that never finishes into a green test one machine and a red one on
    the next.
    """
    deadline = time.monotonic() + timeout_s
    body = None
    while time.monotonic() < deadline:
        r = client.get(f"/api/addons/jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["state"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished: {body!r}")


def test_starting_a_job_answers_202_with_an_id_that_can_be_polled(
        registered_result):
    _enable()
    r = _start(registered_result)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    polled = client.get(f"/api/addons/jobs/{job_id}")
    assert polled.status_code == 200
    assert polled.json()["id"] == job_id
    assert polled.json()["name"] == "runnable"
    assert polled.json()["analysis_key"] == "addon.mean_quality"


def test_the_route_answers_while_the_run_is_still_going(registered_result,
                                                        monkeypatch):
    """The whole point of this route, and the only test that measures it.

    Every other test here polls until the job is finished, which a route that
    ran the analysis INLINE and answered afterwards would satisfy just as
    well -- measured: a reviewer replaced the worker thread with a shim that
    runs the target synchronously on the caller, and all fourteen tests stayed
    green. The 202 would still carry a job id, the poll would still say
    ``done``, and the page would still die at axios's 300 s on a
    dictionary-sized run, which is the entire reason this route exists.

    So this one holds the run open and asserts the answer arrives first: the
    202 comes back, the poll reads ``running``, and only then is the analysis
    released. Under an inline worker ``POST /run-job`` never returns to release
    it and the test fails on the wait rather than passing.
    """
    started, release = threading.Event(), threading.Event()
    real_run_analysis = addons_route.run_analysis

    def blocking(*args, **kwargs):
        started.set()
        assert release.wait(10), "the route did not answer while we waited"
        return real_run_analysis(*args, **kwargs)

    monkeypatch.setattr(addons_route, "run_analysis", blocking)
    _enable()
    r = _start(registered_result)
    assert r.status_code == 202, r.text          # returned while blocking
    job_id = r.json()["job_id"]
    assert started.wait(10), "the worker never reached the analysis"
    polled = client.get(f"/api/addons/jobs/{job_id}")
    assert polled.json()["state"] == "running", polled.text
    release.set()
    assert _await_job(job_id)["state"] == "done"


def test_the_poll_never_carries_outputs(registered_result):
    """The whole point of polling separately.

    A finished map's outputs are fetched once, from /result; re-sending them
    on every tick of a progress bar would make the poll the heaviest call in
    the page.
    """
    _enable()
    job_id = _start(registered_result).json()["job_id"]
    body = _await_job(job_id)
    assert body["state"] == "done", body
    assert "outputs" not in body
    assert "state" in body and "message" in body and "fraction" in body


def test_the_result_is_what_the_synchronous_route_returns(registered_result):
    """Same run, two doors. Compared, not pinned.

    A literal here could agree with a stale expectation while the two routes
    had drifted apart; comparing the two answers cannot.
    """
    # An undeclared parameter rides along deliberately: without it
    # ``ignored_params`` is [] on both sides, and comparing two empty lists
    # holds for an implementation that never fills it at all.
    params = {"scale": 2.0, "not_a_declared_param": 1}
    _enable()
    sync = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality",
        "result_id": registered_result, "params": params})
    assert sync.status_code == 200, sync.text
    assert sync.json()["ignored_params"], "the fixture must ignore something"

    job_id = _start(registered_result, params).json()["job_id"]
    _await_job(job_id)
    got = client.get(f"/api/addons/jobs/{job_id}/result")
    assert got.status_code == 200, got.text

    assert set(got.json()) == set(sync.json())
    by_key_job = {o["key"]: o for o in got.json()["outputs"]}
    by_key_sync = {o["key"]: o for o in sync.json()["outputs"]}
    assert set(by_key_job) == set(by_key_sync)
    assert by_key_job["mean"]["value"] == by_key_sync["mean"]["value"]
    assert got.json()["context"] == sync.json()["context"]
    assert got.json()["ignored_params"] == sync.json()["ignored_params"]


def test_a_maps_bytes_are_fetchable_after_a_job(registered_result):
    """The map URL in a job's outputs must resolve like any other.

    _serialise stores the values as a side effect, and it runs in the worker
    thread here rather than on the event loop -- a store that only worked on
    the synchronous path would leave every map in the page blank.
    """
    _enable()
    job_id = _start(registered_result).json()["job_id"]
    _await_job(job_id)
    outputs = client.get(f"/api/addons/jobs/{job_id}/result").json()["outputs"]
    a_map = next(o for o in outputs if o["kind"] == "map")
    fetched = client.get(a_map["values_url"])
    assert fetched.status_code == 200, fetched.text
    # Against what the output DECLARED, not against a dtype guessed here: the
    # front end decodes these bytes with exactly these two fields, so that is
    # the agreement worth testing. (An earlier draft assumed float32 and was
    # wrong -- the map arrives as the add-on's own float64.)
    assert len(fetched.content) == a_map["n_bytes"]
    assert np.frombuffer(fetched.content, dtype=np.dtype(a_map["dtype"])).size \
        == int(np.prod(a_map["shape"]))


def test_the_addons_progress_reaches_the_poll(registered_result):
    """context.report is the seam, and it is reached through build_context.

    The fixture add-on reports "scaling" at 0.5 on the branch that has a
    quality map. Threading the callback anywhere else -- a runner parameter,
    say -- leaves this None and the page's progress bar dead.
    """
    _enable()
    job_id = _start(registered_result).json()["job_id"]
    body = _await_job(job_id)
    assert body["message"] == "scaling"
    assert body["fraction"] == 0.5


def test_a_failing_addon_fails_the_job_and_names_itself(registered_result):
    _enable()
    job_id = _start(registered_result, {"fail": True}).json()["job_id"]
    body = _await_job(job_id)
    assert body["state"] == "failed"
    assert body["error"].startswith("runnable")
    assert body["reason"] == addons_route.REASON_ADDON_FAILED

    got = client.get(f"/api/addons/jobs/{job_id}/result")
    assert got.status_code == ADDON_FAILED_STATUS
    assert got.json()["reason"] == addons_route.REASON_ADDON_FAILED
    assert "runnable" in got.json()["detail"]


def test_a_crash_that_is_not_an_addon_error_still_finishes_the_job(
        registered_result, monkeypatch):
    """The catch-all, and why it is not paranoia.

    ``run_analysis`` wraps the add-on's own exceptions, so the remaining ways
    to crash are OURS -- ``_serialise`` here. Without the catch-all the thread
    dies, its traceback goes to stderr, and the job sits at ``running``
    forever: the page spins and nothing can clear it. Mutation: delete the
    except branch and the finally, and this goes red on the timeout.
    """
    def boom(*a, **k):
        raise TypeError("serialisation is ours to get wrong")

    monkeypatch.setattr(addons_route, "_serialise", boom)
    _enable()
    job_id = _start(registered_result).json()["job_id"]
    body = _await_job(job_id)
    assert body["state"] == "failed"
    assert body["error"].startswith("runnable")
    assert "TypeError" in body["error"]
    assert body["reason"] == addons_route.REASON_ADDON_FAILED


@pytest.mark.filterwarnings(
    # The KeyboardInterrupt is SUPPOSED to leave the thread: that is the whole
    # point -- pytest reports any exception escaping a thread, and this one
    # escaping is what makes the finally the only thing left to finish the job.
    "ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_something_that_is_not_an_exception_still_finishes_the_job(
        registered_result, monkeypatch):
    """The ``finally`` net, which the crash test above does NOT reach.

    That one is caught by ``except Exception`` and passes with the ``finally``
    deleted. A ``BaseException`` goes past the except, so only the net can
    finish the job -- and if nothing does, the job sits at ``running`` forever:
    ``prune_jobs`` has no stale cutoff precisely because this net is supposed
    to make that impossible.
    """
    def boom(*args, **kwargs):
        raise KeyboardInterrupt("not an Exception")

    monkeypatch.setattr(addons_route, "run_analysis", boom)
    _enable()
    job_id = _start(registered_result).json()["job_id"]
    body = _await_job(job_id)
    assert body["state"] == "failed"
    assert "without reporting a result" in body["error"]
    assert body["reason"] == addons_route.REASON_ADDON_FAILED


def test_a_worker_that_cannot_start_leaves_no_running_job(
        registered_result, monkeypatch):
    """``start_job`` runs before ``Thread.start``, which can refuse.

    Nothing caps how many of these threads run at once, so "can't start new
    thread" is a real answer -- and the job is already registered by then. A
    job left running here is not one stuck row: it falsifies the invariant
    ``prune_jobs`` relies on, so nothing will ever clean it up and
    ``/result`` answers "poll until it finishes" for the life of the process.

    The route's ``threading`` NAME is rebound, rather than ``Thread`` being
    replaced on the threading module: that module is shared, and the gates run
    through ``asyncio.to_thread``, whose pool creates threads too -- the
    request would then fail before it ever reached the line under test, and
    this test would pass for the wrong reason.
    """
    class Refuses:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(addons_route, "threading",
                        SimpleNamespace(Thread=Refuses))
    _enable()
    r = _start(registered_result)
    assert r.status_code == ADDON_FAILED_STATUS, r.text
    assert r.json()["reason"] == addons_route.REASON_ADDON_FAILED
    assert not [j for j in jobs._JOBS.values() if j["state"] == "running"], \
        "a job whose thread never started must not stay running"


def test_the_citation_is_recorded_by_the_job_path_too(registered_result):
    """``record_step`` fires only when ``run_analysis`` is given ``result=``.

    Omit it and the run succeeds, the outputs are right, and the sentence
    never appears -- the failure is invisible everywhere except here. Mutation:
    drop ``result=result`` in the worker and this is the only test that reddens.
    """
    _enable()
    job_id = _start(registered_result).json()["job_id"]
    _await_job(job_id)
    body = client.get(f"/api/citations/result/{registered_result}").json()
    assert "Mean pattern quality" in body["methods"]
    assert "10.5281/zenodo.7777777" in body["bibtex"]


def test_a_disabled_addon_is_refused_synchronously_and_makes_no_job(
        registered_result):
    """A refusal is not a failed job.

    The count is asserted, not just the status: a gate that ran after
    ``start_job`` would still answer 409 while leaving a job behind that
    nothing will ever finish.
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": False})
    before = len(jobs._JOBS)
    r = _start(registered_result)
    assert r.status_code == 409
    assert r.json()["reason"] == addons_route.REASON_ADDON_NOT_ENABLED
    assert len(jobs._JOBS) == before


def test_an_unknown_addon_and_an_unknown_result_are_refused_without_a_job(
        registered_result):
    _enable()
    before = len(jobs._JOBS)
    no_addon = client.post("/api/addons/nobody/run-job", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    no_result = _start("no-such-result")
    assert no_addon.status_code == 404
    assert no_addon.json()["reason"] == addons_route.REASON_ADDON_NOT_FOUND
    assert no_result.status_code == 404
    assert no_result.json()["reason"] == addons_route.REASON_RESULT_NOT_FOUND
    assert len(jobs._JOBS) == before


def _enable_in_the_trust_file(name):
    """Record the decision directly, because the enable route would refuse it.

    Both gates below also guard ``POST /enabled``, so the realistic case they
    exist for cannot be set up through the API: it is a decision recorded when
    this build DID accept the add-on, on an Orienta that no longer does.
    """
    trust_file = Path(os.environ["ORIENTA_ADDON_TRUST_FILE"])
    trust_file.write_text(json.dumps(
        {name: {"enabled": True, "version": "0.1.0", "doi": "",
                "crashes": 0}}), encoding="utf-8")


def test_an_incompatible_addon_is_refused_without_a_job(monkeypatch,
                                                        registered_result):
    """Compatibility is re-checked at run time, not only at enable time.

    And BEFORE the import: the fixture's module deliberately does not exist,
    so a gate that fired later would answer an import error naming
    ``never_imported_module`` instead of this 409.
    """
    _enable_in_the_trust_file("future")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "future_addon"))
    before = len(jobs._JOBS)
    r = client.post("/api/addons/future/run-job", json={
        "analysis_key": "addon.future", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 409, r.text
    assert r.json()["reason"] == addons_route.REASON_ADDON_INCOMPATIBLE
    assert ">=99" in r.json()["detail"]
    assert "never_imported_module" not in r.json()["detail"]
    assert len(jobs._JOBS) == before


_SHARED_KEY_MANIFEST = """\
api_version = 0
name = "{name}"
display_name = "{name}"
version = "0.1.0"
authors = ["T <t@example.org>"]
doi = ""

[[analyses]]
key = "addon.shared"
label = "Shared"
python_name = "{name}_module:analyse"
sentence = "{name} ran."
"""


def test_a_key_conflict_is_refused_without_a_job(tmp_path, monkeypatch,
                                                 registered_result):
    """Two add-ons declaring one key cannot both be cited honestly.

    The realistic arrival: the documented way to start an add-on is to fork
    the shipped example, and a fork that renames the package and keeps the key
    is the first thing anyone produces. Refused before the import -- neither
    fixture's module exists, so a later gate would name one.
    """
    for name in ("alice", "bob"):
        d = tmp_path / name
        d.mkdir()
        (d / "orienta-addon.toml").write_text(
            _SHARED_KEY_MANIFEST.format(name=name), encoding="utf-8")
    _enable_in_the_trust_file("alice")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(tmp_path))
    before = len(jobs._JOBS)
    r = client.post("/api/addons/alice/run-job", json={
        "analysis_key": "addon.shared", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 409, r.text
    assert r.json()["reason"] == addons_route.REASON_ANALYSIS_KEY_CONFLICT
    assert "bob" in r.json()["detail"]
    assert "alice_module" not in r.json()["detail"]
    assert len(jobs._JOBS) == before


def test_an_unknown_job_id_is_404_with_a_reason():
    for url in ("/api/addons/jobs/no-such-job",
                "/api/addons/jobs/no-such-job/result"):
        r = client.get(url)
        assert r.status_code == 404, url
        assert r.json()["reason"] == addons_route.REASON_JOB_NOT_FOUND
        assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_asking_for_the_result_of_a_running_job_says_so():
    """Not 404, and not an empty 200.

    A running job EXISTS, so 404 would be a lie; an empty success would be
    indistinguishable from an analysis that produced nothing.
    """
    job_id = jobs.start_job("runnable", "addon.mean_quality", "res-1")
    r = client.get(f"/api/addons/jobs/{job_id}/result")
    assert r.status_code == 409
    assert r.json()["reason"] == addons_route.REASON_JOB_UNFINISHED
