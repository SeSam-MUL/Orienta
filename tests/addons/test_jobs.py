import pytest
from backend.api.services.addons import jobs


def test_a_started_job_is_running_and_has_an_id():
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    assert isinstance(jid, str) and jid
    job = jobs.get_job(jid)
    assert job["state"] == "running"
    assert job["name"] == "bc-gmm"
    assert job["fraction"] is None


def test_progress_is_readable_while_the_job_runs():
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.report_progress(jid, "fitting", 0.5)
    job = jobs.get_job(jid)
    assert job["message"] == "fitting"
    assert job["fraction"] == 0.5


def test_a_fraction_outside_zero_to_one_is_dropped_not_raised_and_not_clamped():
    """An add-on's arithmetic bug must not destroy its run.

    Revision 2 had this raise ValueError. Both round-2 reviewers found the
    same consequence independently: the exception travels unwrapped through
    ``context.report`` into ``run_analysis``'s ``except Exception``, is counted
    by ``_note_crash``, and pushes the add-on towards the automatic disable at
    CRASH_LIMIT. A revision that exists to stop punishing correct add-ons must
    not add a new way to punish them.

    Not clamped either: 1.5 shown as 100 % would make the bug look like
    progress. The previous fraction stands, the message still updates, and the
    bad value is logged where a developer sees it and an add-on does not.
    """
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.report_progress(jid, "fitting", 0.25)
    jobs.report_progress(jid, "still fitting", 1.5)
    job = jobs.get_job(jid)
    assert job["fraction"] == 0.25
    assert job["message"] == "still fitting"
    assert job["state"] == "running"


@pytest.mark.parametrize("bad", ["50%", None.__class__, float("nan"), [0.5]])
def test_a_fraction_that_is_not_a_number_is_dropped_too(bad):
    """The range check must not be the thing that raises.

    Round 3 found `0.0 <= float(fraction) <= 1.0` still raising for
    `float("50%")` -- through `context.report`, into `except Exception`, into
    `_note_crash`: the exact chain the drop-instead-of-raise fix exists to
    break. A NaN is here because NaN fails every comparison silently, so a
    range check alone would let it through and JSON cannot carry it.
    """
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.report_progress(jid, "fitting", 0.25)
    jobs.report_progress(jid, "onwards", bad)
    job = jobs.get_job(jid)
    assert job["fraction"] == 0.25
    assert job["message"] == "onwards"


def test_progress_for_an_unknown_job_changes_nothing_and_does_not_raise():
    """The add-on holds this callback; a pruned job must not crash its run.

    Asserted, not merely executed: an earlier draft only called the function,
    which a `raise` inside it would still have satisfied at collection time.
    """
    live = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.report_progress("no-such-job", "fitting", 0.5)
    assert jobs.get_job("no-such-job") is None
    assert jobs.get_job(live)["message"] == ""


def test_a_finished_job_carries_outputs_ignored_params_and_context():
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(jid, outputs=[{"kind": "scalar", "key": "n_px"}],
                    ignored_params=("bogus",),
                    context={"step_size_um": 0.5, "quality_source": "native"})
    job = jobs.get_job(jid)
    assert job["state"] == "done"
    assert job["outputs"] == [{"kind": "scalar", "key": "n_px"}]
    assert job["ignored_params"] == ("bogus",)
    assert job["context"]["step_size_um"] == 0.5
    assert job["error"] is None


def test_a_failed_job_carries_the_message_and_the_reason():
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(jid, error="bc-gmm / addon.bc_gmm: ValueError: boom",
                    reason="addon_failed")
    job = jobs.get_job(jid)
    assert job["state"] == "failed"
    assert job["error"].startswith("bc-gmm /")
    assert job["reason"] == "addon_failed"


def test_progress_after_the_job_finished_is_ignored():
    """A thread that keeps reporting must not resurrect a finished job."""
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(jid, outputs=[])
    jobs.report_progress(jid, "late", 0.9)
    job = jobs.get_job(jid)
    assert job["state"] == "done"
    assert job["message"] == ""


def test_a_message_whose_str_raises_keeps_the_previous_one():
    """``str(message)`` runs the add-on's ``__str__``; it may raise.

    It was the one call left in report_progress that could travel back into a
    stranger's run and be counted against it by ``_note_crash`` -- the chain
    the whole drop-instead-of-raise rule exists to break.
    """
    class Hostile:
        def __str__(self):
            raise RuntimeError("no")

    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.report_progress(jid, "fitting", 0.25)
    jobs.report_progress(jid, Hostile(), 0.5)
    job = jobs.get_job(jid)
    assert job["message"] == "fitting"
    assert job["fraction"] == 0.5, "the fraction is still usable and must land"


def test_a_very_long_message_is_cut():
    """Polled on a timer, so an unbounded message is re-sent on every poll."""
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.report_progress(jid, "x" * 5000)
    assert len(jobs.get_job(jid)["message"]) < 600


def test_get_job_returns_a_copy_so_a_caller_cannot_mutate_the_registry():
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.get_job(jid)["state"] = "done"
    assert jobs.get_job(jid)["state"] == "running"


def test_the_copy_is_deep_so_outputs_and_context_are_not_shared():
    """A shallow ``dict(job)`` shares exactly the entries a caller walks into.

    The result route serialises ``outputs`` and ``context``; both are nested,
    and both would be the registry's own objects under a shallow copy. The
    docstring promised a copy, and the test above passes either way because it
    only overwrites a top-level key -- so it did not test the promise.
    """
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(jid, outputs=[{"kind": "scalar", "key": "n_px"}],
                    context={"step_size_um": 0.5})
    taken = jobs.get_job(jid)
    taken["context"]["step_size_um"] = 999
    taken["outputs"][0]["key"] = "clobbered"
    fresh = jobs.get_job(jid)
    assert fresh["context"]["step_size_um"] == 0.5
    assert fresh["outputs"][0]["key"] == "n_px"


def test_finishing_a_job_twice_keeps_the_first_result():
    """The worker's ``finally`` calls finish_job unconditionally.

    That call is what guarantees no job stays ``running`` after its thread
    ends (prune_jobs relies on it instead of a stale cutoff). It must not be
    able to overwrite the real result it exists to insure.
    """
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(jid, outputs=[{"kind": "scalar", "key": "n_px"}])
    jobs.finish_job(jid, error="worker ended without reporting a result",
                    reason="addon_failed")
    job = jobs.get_job(jid)
    assert job["state"] == "done"
    assert job["error"] is None
    assert job["outputs"] == [{"kind": "scalar", "key": "n_px"}]


def test_finishing_an_unknown_job_is_logged_not_silent(caplog):
    """A finished analysis with nowhere to go is a bug; silence keeps it one.

    Unlike report_progress, nothing here is held by add-on code, so there is
    nobody to protect from the event and no reason to swallow it.
    """
    with caplog.at_level("WARNING"):
        jobs.finish_job("no-such-job", outputs=[])
    assert any("no-such-job" in r.getMessage() for r in caplog.records), \
        caplog.text


def test_starting_a_job_prunes_old_finished_ones():
    """Pruning happens where the registry grows, so assert it THERE.

    An earlier draft called prune_jobs() explicitly after a second start_job
    and asserted `removed == 1`. Round 3 executed this file and found it fails
    9-passed-1-failed: start_job had already pruned, so the explicit call
    removes nothing. The trap is the cheap repair -- deleting the prune_jobs()
    call from start_job turns it green and silently undoes the fix. Asserting
    the effect instead of a count cannot be repaired that way.

    Reads ``_JOBS`` DIRECTLY and not through ``get_job``, which is the same
    trap one turn later: ``get_job`` now prunes as well, so asking it whether
    the old job is gone would answer yes even with ``start_job``'s sweep
    deleted -- the question would be answered by the observer.
    """
    old = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(old, outputs=[])
    jobs._JOBS[old]["finished_at"] -= 7200
    live = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-2")
    assert old not in jobs._JOBS, "start_job must prune finished jobs"
    assert live in jobs._JOBS


def test_looking_a_job_up_prunes_too_so_the_404_sentence_is_true():
    """The routes tell the user a finished job is dropped an hour later.

    With the sweep only in ``start_job``, that sentence described a policy
    that had not run: a user who runs one analysis and never another keeps
    that job -- and its outputs -- for the life of the process, and the only
    place the claim is ever READ is the 404 from a lookup. So the lookup
    prunes, and the claim is true at the moment it is made.

    Asserts through ``get_job`` on purpose, the opposite of the test above:
    here the observer IS the thing under test.
    """
    old = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(old, outputs=[])
    jobs._JOBS[old]["finished_at"] -= 7200
    assert jobs.get_job(old) is None
    assert old not in jobs._JOBS


def test_a_lookup_does_not_prune_a_job_that_is_still_young():
    """The other half: the sweep must not eat the job being polled.

    Without this, "prune on read" passes its own test by deleting everything.
    """
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(jid, outputs=[])
    assert jobs.get_job(jid) is not None
    assert jobs.get_job(jid, include_outputs=False) is not None


def test_prune_jobs_reports_how_many_it_removed():
    old = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.finish_job(old, outputs=[])
    jobs._JOBS[old]["finished_at"] -= 7200
    assert jobs.prune_jobs(max_age_s=3600) == 1
    assert jobs.prune_jobs(max_age_s=3600) == 0


def test_a_running_job_is_never_pruned_however_old():
    """Age is not what protects it -- the state is.

    ``finished_at`` is None while a job runs, so the age term reads 0 and is
    already older than any cutoff. Only the ``state != "running"`` guard keeps
    it, which is the point: a long dictionary run outlives any threshold worth
    setting, and there is deliberately no stale-running cutoff (see
    prune_jobs). Ageing ``started_at`` here would be decoration -- prune_jobs
    never reads it.
    """
    jid = jobs.start_job("bc-gmm", "addon.bc_gmm", "res-1")
    jobs.prune_jobs(max_age_s=1)
    assert jobs.get_job(jid) is not None
