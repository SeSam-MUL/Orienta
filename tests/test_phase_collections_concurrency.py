"""Concurrent writes to one collection file must not lose members.

Found in f7's user loop: an Alt-drag over twelve phases fired twelve parallel
`POST /members` at one collection. Every call is read-modify-write and nothing
locked, so they read the same "before", each added its own phase, and the last
write won: **twelve on screen, nine in the folder, every response 200.** The
frontend now sends one request per drag, but two windows -- or a second client,
or the app plus a script -- still hit the same file.

The shape matters more than the number. FastAPI runs a sync `def` endpoint in a
threadpool, so two requests genuinely overlap in one process, and `_write_atomic`
makes each individual write atomic without making read-modify-write atomic. An
atomic write is not a transaction; it only guarantees that nobody sees half a
file.

Lost updates are the worst kind of wrong here, because nothing reports them: the
user sees what they asked for, the file has less, and the next reload quietly
shows the smaller list. A phase missing from a collection changes which phases a
run considers.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import phase_collections as pc  # noqa: E402

#: Twelve, because twelve is what the user's drag actually sent.
N = 12


@pytest.fixture(autouse=True)
def store(tmp_path):
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    pc.collections_dir().mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        pc.set_collections_dir_for_test(None)


def _run(fns):
    """Start every call at once, so they really overlap."""
    ready = threading.Barrier(len(fns))
    errors = []

    def go(fn):
        ready.wait()
        try:
            fn()
        except Exception as exc:                      # recorded, not swallowed
            errors.append(f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=len(fns)) as pool:
        list(pool.map(go, fns))
    return errors


def test_twelve_concurrent_assigns_keep_all_twelve_phases():
    """The user's case, reproduced. Without a lock this lands around nine."""
    pc.save(pc.PhaseCollection(name="Matrix"))
    keys = [f"phase{i:02d}" for i in range(N)]

    errors = _run([lambda k=k: pc.assign([k], "Matrix") for k in keys])
    assert errors == [], errors

    (c,) = pc.load_all()[0]
    got = sorted(m.key for m in c.members)
    assert got == sorted(keys), (
        f"{len(got)} of {N} survived -- lost updates: "
        f"{sorted(set(keys) - set(got))}")


def test_concurrent_removals_all_take_effect():
    """The other direction: twelve parallel removals must all land, or a phase
    the user removed comes back on reload."""
    keys = [f"phase{i:02d}" for i in range(N)]
    pc.save(pc.PhaseCollection(
        name="Matrix", members=[pc.PhaseMember(key=k) for k in keys]))

    errors = _run([lambda k=k: pc.unassign_from("Matrix", [k]) for k in keys])
    assert errors == [], errors

    (c,) = pc.load_all()[0]
    assert [m.key for m in c.members] == [], (
        f"still present: {[m.key for m in c.members]}")


def test_concurrent_writes_to_different_collections_all_land():
    """A per-file lock must not serialise into a lost update across files, and a
    folder-wide lock must not deadlock. Either design has to pass this."""
    for i in range(N):
        pc.save(pc.PhaseCollection(name=f"C{i:02d}"))

    errors = _run([lambda i=i: pc.assign([f"p{i:02d}"], f"C{i:02d}")
                   for i in range(N)])
    assert errors == [], errors

    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {f"C{i:02d}": [f"p{i:02d}"] for i in range(N)}


def test_a_create_and_an_assign_racing_do_not_lose_either():
    """Mixed operations, because the lock has to cover `save` as well as the
    read-modify-write helpers -- `create` is a `save` of a fresh collection."""
    pc.save(pc.PhaseCollection(name="Matrix"))
    fns = [lambda: pc.save(pc.PhaseCollection(name="Other"))]
    fns += [lambda k=k: pc.assign([k], "Matrix")
            for k in (f"phase{i:02d}" for i in range(N - 1))]

    errors = _run(fns)
    assert errors == [], errors

    by = {c.name: sorted(m.key for m in c.members) for c in pc.load_all()[0]}
    assert set(by) == {"Matrix", "Other"}
    assert len(by["Matrix"]) == N - 1, by["Matrix"]


def test_every_mutating_function_is_serialised():
    """A structural guard: a new writer without the decorator is the same silent
    lost update, and the behavioural tests above only cover the writers they
    happen to call.

    The list is derived from the module rather than repeated here -- anything that
    calls `_write_atomic` mutates the folder, so anything that calls it must hold
    the lock.
    """
    import ast
    import inspect

    tree = ast.parse(Path(pc.__file__).read_text(encoding="utf-8"))
    writers = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = {n.func.id for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "_write_atomic" in calls and not node.name.startswith("_write"):
            writers.add(node.name)

    assert writers, "no writers found -- did _write_atomic get renamed?"
    missing = []
    for name in sorted(writers):
        fn = getattr(pc, name, None)
        if fn is None:                       # a nested helper, not public
            continue
        if not getattr(fn, "_serialised", False):
            missing.append(name)
    assert missing == [], (
        f"these mutate the folder without taking the write lock: {missing}")


def test_the_lock_is_not_held_across_a_read():
    """A guard against the cheap fix that would cost the app its responsiveness:
    taking the lock for the whole of `load_all` would serialise every listing
    behind every write. The lock belongs around read-modify-write, not around
    reading.
    """
    pc.save(pc.PhaseCollection(name="Matrix"))
    # Reading while the lock is held must not block.
    with pc._write_lock():
        done = threading.Event()

        def reader():
            pc.load_all()
            done.set()

        t = threading.Thread(target=reader)
        t.start()
        assert done.wait(timeout=5.0), (
            "load_all() blocked while a write lock was held")
        t.join()
