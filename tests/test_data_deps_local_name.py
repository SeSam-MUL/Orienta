"""tests.data_deps.local_name: a scan named by an alias, the real name only in the test-data folder."""
import json

from tests.data_deps import LOCAL_NAMES, local_name


def test_the_alias_gives_the_path_the_folder_lists(tmp_path):
    (tmp_path / LOCAL_NAMES).write_text(json.dumps({"map_a": "sub/scan.ang"}), encoding="utf-8")
    assert local_name("map_a", tmp_path) == "sub/scan.ang"


def test_an_alias_the_folder_does_not_list_names_no_file(tmp_path):
    (tmp_path / LOCAL_NAMES).write_text(json.dumps({"map_a": "scan.ang"}), encoding="utf-8")
    assert local_name("map_b", tmp_path) == "missing-map_b"
    assert not (tmp_path / local_name("map_b", tmp_path)).exists()


def test_no_list_or_a_broken_one_names_no_file(tmp_path):
    assert local_name("map_a", tmp_path) == "missing-map_a"
    (tmp_path / LOCAL_NAMES).write_text("{not json", encoding="utf-8")
    assert local_name("map_a", tmp_path) == "missing-map_a"
    (tmp_path / LOCAL_NAMES).write_text(json.dumps(["map_a"]), encoding="utf-8")
    assert local_name("map_a", tmp_path) == "missing-map_a"


def test_the_folder_comes_from_orienta_test_data(tmp_path, monkeypatch):
    (tmp_path / LOCAL_NAMES).write_text(json.dumps({"map_a": "x.h5oina"}), encoding="utf-8")
    monkeypatch.setenv("ORIENTA_TEST_DATA", str(tmp_path))
    assert local_name("map_a") == "x.h5oina"
