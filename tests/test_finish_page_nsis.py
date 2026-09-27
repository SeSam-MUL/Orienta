"""The desktop-shortcut checkbox, clicked for real.

A bench installer with a genuine MUI finish page includes
electron/nsis/installer.nsh UNCHANGED and inserts its `customFinishPage`. It
creates a shortcut in a temporary "desktop" (standing in for the one
electron-builder's install section creates), and the test drives the page
with the same Win32 messages the page uses itself — BM_GETCHECK, BM_SETCHECK,
BM_CLICK on "Fertigstellen". Nothing is installed; the real desktop is never
touched.

What it proves:
  * the checkbox is there, TICKED by default, and labelled in the installer's
    language (German here: the text is picked at run time from $LANGUAGE);
  * left ticked, the shortcut stays;
  * unticked, the shortcut electron-builder made is removed as the page closes.

Skipped where the NSIS compiler or electron-builder's plugins are not cached.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "electron" / "nsis" / "installer.nsh"
TEMPLATES = REPO / "frontend" / "node_modules" / "app-builder-lib" / "templates" / "nsis" / "include"
CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "electron-builder" / "Cache"


def _first(pattern: str) -> Path | None:
    found = sorted(CACHE.glob(pattern))
    return found[-1] if found else None


MAKENSIS = _first("nsis-*/*/Bin/makensis.exe")
PLUGINS = _first("nsis-resources-*/*/plugins/x86-unicode")

pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="Windows-only"),
    pytest.mark.skipif(MAKENSIS is None or PLUGINS is None,
                       reason="electron-builder's NSIS toolchain is not cached here"),
    pytest.mark.skipif(not TEMPLATES.is_dir(), reason="app-builder-lib is not installed"),
]

LABEL_DE = "Verknüpfung auf dem Desktop anlegen"

# Win32 messages, from PowerShell: find the bench's window, find the checkbox
# and the Finish button among its child windows by their text, read the box
# with BM_GETCHECK, untick it with BM_SETCHECK, press Finish with BM_CLICK.
#
# Not UI Automation: it reported every control on this page as a bare "Pane",
# with no toggle or invoke pattern to act through. These are the same messages
# the page itself uses — MUI reads the box with BM_GETCHECK as it closes.
DRIVE = textwrap.dedent(r"""
    param([int]$ProcessId, [string]$Label, [string]$Untick)
    Add-Type @'
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using System.Text;
    public static class W {
      public delegate bool Enum(IntPtr h, IntPtr l);
      [DllImport("user32.dll")] public static extern bool EnumWindows(Enum cb, IntPtr l);
      [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr p, Enum cb, IntPtr l);
      [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
      [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
      [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
      [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
      public static IntPtr Top(int pid) {
        IntPtr found = IntPtr.Zero;
        EnumWindows((h, l) => { uint p; GetWindowThreadProcessId(h, out p);
          if (p == pid && IsWindowVisible(h)) { found = h; return false; } return true; }, IntPtr.Zero);
        return found;
      }
      public static IntPtr Child(IntPtr top, string text) {
        IntPtr found = IntPtr.Zero;
        EnumChildWindows(top, (h, l) => { var s = new StringBuilder(512); GetWindowText(h, s, 512);
          if (s.ToString().Contains(text)) { found = h; return false; } return true; }, IntPtr.Zero);
        return found;
      }
    }
'@
    $top = [IntPtr]::Zero
    for ($i = 0; $i -lt 80 -and $top -eq [IntPtr]::Zero; $i++) {
        $top = [W]::Top($ProcessId); if ($top -eq [IntPtr]::Zero) { Start-Sleep -Milliseconds 250 } }
    if ($top -eq [IntPtr]::Zero) { Write-Output "NOWINDOW"; exit 2 }
    # MUI moves on from the install page to the finish page by itself, as the
    # real installer does, so the checkbox is simply waited for.
    $box = [IntPtr]::Zero
    for ($i = 0; $i -lt 80 -and $box -eq [IntPtr]::Zero; $i++) {
        $box = [W]::Child($top, $Label); if ($box -eq [IntPtr]::Zero) { Start-Sleep -Milliseconds 250 } }
    if ($box -eq [IntPtr]::Zero) { Write-Output "NOLABEL"; exit 3 }
    $state = [W]::SendMessage($box, 0x00F0, [IntPtr]::Zero, [IntPtr]::Zero)   # BM_GETCHECK
    Write-Output ("STATE=" + $state)
    if ($Untick -eq "yes") { [void][W]::SendMessage($box, 0x00F1, [IntPtr]::Zero, [IntPtr]::Zero) }  # BM_SETCHECK 0
    $finish = [W]::Child($top, "Fertigstellen")
    if ($finish -eq [IntPtr]::Zero) { Write-Output "NOFINISH"; exit 4 }
    [void][W]::SendMessage($finish, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)   # BM_CLICK
    Write-Output "PRESSED"
""")


def _bench(tmp: Path, desktop_link: Path, *, link_made_by_install: bool = True) -> Path:
    """An install page followed by the real finish page, as in the product.

    `link_made_by_install=False` is a REINSTALL: electron-builder keeps
    shortcuts as they were, so one the user removed earlier is not made again
    by the install section.
    """
    make_link = ('CreateShortCut "$newDesktopLink" "$appExe"'
                 if link_made_by_install else "; (a reinstall: no shortcut made)")
    script = tmp / "finish.nsi"
    script.write_text(textwrap.dedent(f"""        Unicode true
        !addincludedir "{TEMPLATES}"
        !addplugindir /x86-unicode "{PLUGINS}"
        !include "MUI2.nsh"
        !include "StdUtils.nsh"
        ; What electron-builder provides around the finish page:
        !define isUpdated "1 == 0"
        !define HIDE_RUN_AFTER_FINISH   ; nothing to launch in a bench
        !define APP_DESCRIPTION "Orienta bench"
        !define APP_ID "com.orienta.bench"
        Var launchLink
        Var newDesktopLink
        Var appExe
        !include "{WRAPPER}"
        Name "Orienta bench"
        OutFile "{tmp / 'finish.exe'}"
        RequestExecutionLevel user
        !insertmacro MUI_PAGE_INSTFILES
        !insertmacro customFinishPage
        !insertmacro MUI_LANGUAGE "German"
        Function .onInit
          StrCpy $LANGUAGE 1031
        FunctionEnd
        Section
          ; what electron-builder's install section sets and makes
          StrCpy $appExe "$SYSDIR\\notepad.exe"
          StrCpy $newDesktopLink "{desktop_link}"
          StrCpy $launchLink "$appExe"
          {make_link}
        SectionEnd
    """), encoding="utf-8-sig")
    build = subprocess.run([str(MAKENSIS), "/V1", "/WX", str(script)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-2000:]
    return tmp / "finish.exe"


def _press(tmp: Path, exe: Path, *, untick: bool) -> str:
    drive = tmp / "drive.ps1"
    drive.write_text(DRIVE, encoding="utf-8-sig")
    bench = subprocess.Popen([str(exe)])
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                              str(drive), "-ProcessId", str(bench.pid), "-Label", LABEL_DE,
                              "-Untick", "yes" if untick else "no"],
                             capture_output=True, text=True, timeout=60)
        bench.wait(timeout=30)
        return out.stdout
    finally:
        if bench.poll() is None:
            bench.kill()


def test_left_ticked_the_shortcut_stays(tmp_path):
    link = tmp_path / "desktop" / "Orienta.lnk"
    link.parent.mkdir()
    exe = _bench(tmp_path, link)
    out = _press(tmp_path, exe, untick=False)
    assert "STATE=1" in out, out            # ticked by default, German label found
    assert "PRESSED" in out, out
    assert link.exists()


def test_unticked_the_shortcut_is_removed(tmp_path):
    link = tmp_path / "desktop" / "Orienta.lnk"
    link.parent.mkdir()
    exe = _bench(tmp_path, link)
    out = _press(tmp_path, exe, untick=True)
    assert "STATE=1" in out, out
    assert "PRESSED" in out, out
    assert not link.exists()


def test_on_a_reinstall_a_ticked_box_makes_the_missing_shortcut(tmp_path):
    """Review finding: electron-builder keeps shortcuts as they were on a
    reinstall, so a shortcut removed on the first install was not made again
    while the box — ticked by default — said "create"."""
    link = tmp_path / "desktop" / "Orienta.lnk"
    link.parent.mkdir()
    exe = _bench(tmp_path, link, link_made_by_install=False)
    out = _press(tmp_path, exe, untick=False)
    assert "STATE=1" in out, out
    assert link.exists(), "ticked, but no shortcut"


def test_on_a_reinstall_an_unticked_box_leaves_the_desktop_empty(tmp_path):
    link = tmp_path / "desktop" / "Orienta.lnk"
    link.parent.mkdir()
    exe = _bench(tmp_path, link, link_made_by_install=False)
    _press(tmp_path, exe, untick=True)
    assert not link.exists()


def test_the_wrapper_keeps_electron_builders_creation_on():
    """The checkbox only ever REMOVES. electron-builder's uninstaller deletes
    the desktop link only while its own creation is enabled, so a shortcut
    made by us instead would outlive the application."""
    import json
    build = json.loads((REPO / "frontend" / "package.json").read_text(encoding="utf-8"))["build"]
    assert build["nsis"]["createDesktopShortcut"] is True
    assert WRAPPER.read_bytes()[:3] == b"\xef\xbb\xbf"
