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


def test_unknown_result_id_explains_why(stored_result_id):
    r = client.get("/api/citations/result/does-not-exist")
    detail = r.json()["detail"]
    assert "evict" in detail or "restart" in detail


def test_normal_path_has_no_warnings(stored_result_id):
    body = client.get(f"/api/citations/result/{stored_result_id}").json()
    assert body["warnings"] == []


def test_broken_library_degrades_instead_of_500(
    stored_result_id, monkeypatch, caplog
):
    """load_library() can raise (missing/malformed CITATION.cff); the
    endpoint must still answer with what does not depend on it."""
    import logging

    def _boom(*args, **kwargs):
        raise FileNotFoundError("CITATION.cff")

    monkeypatch.setattr(
        "backend.api.routes.citations.load_library", _boom
    )

    with caplog.at_level(logging.ERROR, logger="backend.api.routes.citations"):
        r = client.get(f"/api/citations/result/{stored_result_id}")

    assert r.status_code == 200
    body = r.json()
    assert "dynamic" in body["methods"]   # methods come from steps.py, not the library
    assert body["bibtex"] == ""
    assert body["plain"] == ""
    assert body["warnings"]
    assert any("bibliography" in w for w in body["warnings"])
    assert any(
        "Failed to load the citation library" in rec.message
        for rec in caplog.records
    )


def test_a_repeated_undeclared_key_is_listed_once(register_result):
    """I2. The panel keys its <li> on the key, so a duplicate here is also a
    duplicate React key — and a list that reads like two unknown steps ran."""
    rid = register_result([
        {"key": "some.addon.step", "params": {}},
        {"key": "some.addon.step", "params": {"x": 1}},
    ])
    body = client.get(f"/api/citations/result/{rid}").json()
    assert body["undeclared"] == ["some.addon.step"]


def test_the_orienta_methods_note_renders_with_an_author_and_a_locator(
    register_result,
):
    """I1. This is the entry the whole feature exists to surface; it used to
    render as a bare title and a year — no author (render_plain fell back to
    `publisher`) and no DOI/URL, with `publisher` dropped by BibTeX because
    @techreport has no such field."""
    rid = register_result([
        {"key": "eds.chemistry_prior",
         "params": {"strength_by_phase": {"Si": 1.0}, "n_adjusted": 349}},
    ])
    body = client.get(f"/api/citations/result/{rid}").json()

    block = next(
        b for b in body["bibtex"].split("\n\n")
        if b.startswith("@techreport{orienta-eds-prior,")
    )
    assert "author = {" in block
    assert "Samberger, Sebastian" in block
    assert "url = {" in block
    assert "institution = {" in block     # not `publisher`, which styles drop
    assert "publisher = {" not in block
    assert "doi = {" not in block         # none exists; none is invented

    line = next(
        line for line in body["plain"].splitlines()
        if "EDS-Guided Phase Disambiguation" in line
    )
    assert line.startswith("Samberger, S.")
    assert "https://" in line
