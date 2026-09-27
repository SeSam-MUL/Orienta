"""Ray's session directory, which is a socket-path-length problem in disguise.

Ray puts Unix domain sockets under its temp directory, and a socket path
cannot exceed about 104 bytes. A long TMPDIR therefore makes ``ray.init``
fail with an error that names none of this. Ray ignores TMPDIR on macOS (it
always uses /tmp) and uses named pipes on Windows, so the fix belongs to
Linux only -- and it must not fire when TMPDIR is short, because a fixed
path shared between users on one workstation is its own problem.
"""
import indexing_controller as ic

LONG = "/home/researcher/OneDrive - Montanuniversitaet Leoben/scratch/tmpdir"


def test_a_long_tmpdir_on_linux_gets_a_short_per_user_path():
    env = {"TMPDIR": LONG}
    chosen = ic._set_ray_temp_dir(env, platform="linux")
    assert chosen and chosen.startswith("/tmp/orienta-ray-")
    assert env["RAY_TMPDIR"] == chosen
    assert len(chosen) < 40


def test_a_short_tmpdir_is_left_alone():
    env = {"TMPDIR": "/tmp"}
    assert ic._set_ray_temp_dir(env, platform="linux") is None
    assert "RAY_TMPDIR" not in env


def test_a_users_own_choice_wins():
    env = {"TMPDIR": LONG, "RAY_TMPDIR": "/srv/fast/ray"}
    assert ic._set_ray_temp_dir(env, platform="linux") == "/srv/fast/ray"
    assert env["RAY_TMPDIR"] == "/srv/fast/ray"


def test_macos_and_windows_are_not_touched():
    # macOS ray ignores TMPDIR; Windows has no Unix sockets to overflow.
    for platform in ("darwin", "win32"):
        env = {"TMPDIR": LONG}
        assert ic._set_ray_temp_dir(env, platform=platform) is None
        assert "RAY_TMPDIR" not in env


def test_the_path_is_per_user_so_a_shared_machine_does_not_collide(monkeypatch):
    # /tmp is shared: two people on one workstation must not be handed the
    # same session directory. Windows has no getuid, so it is faked here --
    # the branch under test is the Linux one.
    monkeypatch.setattr(ic.os, "getuid", lambda: 1000, raising=False)
    env = {"TMPDIR": LONG}
    chosen = ic._set_ray_temp_dir(env, platform="linux")
    assert chosen.endswith("-1000"), chosen


def test_the_production_call_passes_no_platform_and_must_still_work():
    # The five tests above all pass platform=..., so the only branch
    # production uses was the only one never exercised -- and it raised
    # NameError, aborting every large Hough run, on Windows too.
    env = {"TMPDIR": LONG}
    result = ic._set_ray_temp_dir(env)
    assert result is None or result.startswith("/tmp/orienta-ray-")
