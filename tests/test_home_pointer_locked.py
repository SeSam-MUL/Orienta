"""The data-folder pointer, locked the way an antivirus scan locks it.

home.txt held open with share mode 0 while the shell starts. Reading it fails
(EBUSY or EPERM). The shell used to treat that as "no choice recorded" and
silently switch to the default folder -- a second 2.5 GB install, with no log
line. It must decide nothing, create nothing, keep the pointer, and say why.

Everything happens in a temporary directory with APPDATA and LOCALAPPDATA
redirected; the real folders of this machine are not touched. Windows only:
the lock is a Win32 sharing violation.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="a sharing violation is a Windows matter"),
    pytest.mark.skipif(NODE is None, reason="node is not installed"),
]

SCRIPT = r"""
const hc = require(process.argv[1]);
const ep = require(process.argv[2]);
let err = null;
const notice = hc.settleHomeAtStartup({ isPackaged: true, onError: (e) => { err = e; } });
console.log(JSON.stringify({
  notice, mustNotCreate: hc.mustNotCreateHome(notice), err: err && err.message,
  log: hc.startupLogLines({ notice, checkError: err, home: ep.orientaHome() }),
}));
"""


def _settle(appdata: Path, local: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "ORIENTA_HOME"}
    env["APPDATA"], env["LOCALAPPDATA"] = str(appdata), str(local)
    out = subprocess.run(
        [NODE, "-e", SCRIPT, str(REPO / "electron" / "home_choice.js"), str(REPO / "electron" / "update_endpoints.js")],
        capture_output=True, text=True, encoding="utf-8", env=env, check=True)
    return json.loads(out.stdout)


@pytest.fixture
def box(tmp_path):
    appdata, local = tmp_path / "appdata", tmp_path / "local"
    (appdata / "Orienta").mkdir(parents=True)
    local.mkdir()
    return appdata, local


def _lock(file: Path):
    handle = ctypes.windll.kernel32.CreateFileW(str(file), 0x80000000, 0, None, 3, 0x80, None)
    assert handle != -1, ctypes.GetLastError()
    return handle


def test_a_locked_pointer_decides_nothing_creates_nothing_and_says_why(box):
    appdata, local = box
    gone = local / "Orienta" / "Orienta" / "Orienta"          # a stale chain, as in the report
    pointer = appdata / "Orienta" / "home.txt"
    pointer.write_bytes(b"\xff\xfe" + f"{gone}\r\n".encode("utf-16-le"))
    handle = _lock(pointer)
    try:
        result = _settle(appdata, local)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    notice = result["notice"]
    assert notice["code"] == "unreachable" and notice["kind"] == "unreadable"
    assert result["mustNotCreate"] is True
    assert any("could not be read" in line for line in result["log"])
    assert pointer.exists() and not list((appdata / "Orienta").glob("home.txt.previous*"))
    assert not list(local.rglob("*")), "something was created in the default folder"


def test_the_same_pointer_unlocked_is_read_normally(box):
    appdata, local = box
    gone = local / "Orienta" / "Orienta"
    (appdata / "Orienta" / "home.txt").write_bytes(b"\xff\xfe" + f"{gone}\r\n".encode("utf-16-le"))
    result = _settle(appdata, local)
    # readable, and the folder is gone on a fixed disk: the stale route, not the unreadable one
    assert result["notice"]["code"] in ("staleFallback", "staleDefault")
