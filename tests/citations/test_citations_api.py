from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_unknown_result_id_is_404():
    r = client.get("/api/citations/result/does-not-exist")
    assert r.status_code == 404


def test_returns_all_three_formats(stored_result_id):
    r = client.get(f"/api/citations/result/{stored_result_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["bibtex"].startswith("@")
    assert body["methods"]
    assert body["plain"]


def test_core_entries_appear_even_with_no_steps(stored_result_id_no_steps):
    """A result with an empty trail can still cite Orienta and the libraries."""
    body = client.get(
        f"/api/citations/result/{stored_result_id_no_steps}").json()
    assert "orienta" in body["bibtex"]


def test_undeclared_steps_are_listed(stored_result_id_undeclared):
    body = client.get(
        f"/api/citations/result/{stored_result_id_undeclared}").json()
    assert body["undeclared"] == ["some.addon.step"]
    assert "no citation declared" in body["methods"]


def test_entries_are_deduplicated_and_ordered(stored_result_id):
    body = client.get(f"/api/citations/result/{stored_result_id}").json()
    ids = [ln.split("{", 1)[1].split(",", 1)[0]
           for ln in body["bibtex"].splitlines() if ln.startswith("@")]
    assert len(ids) == len(set(ids))
    assert ids[0] == "orienta"


def test_answer_depends_only_on_the_stored_result(stored_result_id):
    """Spec §4, made to bite: the paragraph describes the run, not the session.

    Nothing a client sends can change it — there is no request body, and two
    identical GETs with different everything-else must agree.
    """
    a = client.get(f"/api/citations/result/{stored_result_id}").json()
    b = client.get(
        f"/api/citations/result/{stored_result_id}",
        headers={"Accept-Language": "de", "X-Anything": "else"},
    ).json()
    assert a["methods"] == b["methods"]
    assert a["bibtex"] == b["bibtex"]
    assert "dynamic" in a["methods"]      # the value the RUN recorded
