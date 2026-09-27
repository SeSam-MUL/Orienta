"""Turning a path into a NAME, the same way on every platform.

``pathlib.Path`` is ``WindowsPath`` on Windows and ``PosixPath`` everywhere
else, and a backslash is an ordinary filename character on POSIX. So the same
string does not give the same name on two machines:

===============================================  ==========================
``C:\\Users\\operator\\Database\\Al.cif``          ``.stem``
===============================================  ==========================
on Windows                                       ``Al``
on Linux / macOS                                 ``C:\\Users\\operator\\...\\Al``
===============================================  ==========================

That second row is how an operator's username and directory layout end up in
a phase name — in the methods paragraph meant for a journal, and in every
exported ``.h5``. Found by the first Linux CI run of the Python suite
(``tests/citations/test_no_local_paths.py``), 2026-09-23.

Use these helpers wherever a path **that the backend did not just read off its
own filesystem** becomes something a person reads: paths out of a request
body, out of a stored result file, out of a config or checkpoint that may have
travelled from another machine. For a path the backend itself produced by
walking a directory, ``Path.stem`` is right and says so more clearly.

The trade-off, stated plainly: a file whose name genuinely contains a
backslash is legal on POSIX, and this reads it as a separator. That costs a
wrong display name in a case nobody has hit; the alternative costs a leaked
username in a case CI hit on the first run.
"""
from pathlib import PurePosixPath

__all__ = ["display_stem", "display_basename"]


def _as_posix(path) -> PurePosixPath:
    # PurePosixPath, not the platform Path: it applies pathlib's own rules for
    # suffixes and dotfiles (".gitignore" keeps its dot, "a.b.c" drops only
    # ".c") without asking which OS we are on.
    return PurePosixPath(str(path).replace("\\", "/"))


def display_stem(path) -> str:
    """File name without its last suffix, on any platform.

    >>> display_stem(r"C:\\Users\\operator\\Database\\Al.cif")
    'Al'
    >>> display_stem("/home/operator/Database/Al.cif")
    'Al'
    >>> display_stem("Al.cif")
    'Al'
    """
    return _as_posix(path).stem


def display_basename(path) -> str:
    """File name with its suffix, on any platform.

    >>> display_basename(r"C:\\Users\\operator\\Al (Al) [cF4].sht")
    'Al (Al) [cF4].sht'
    """
    return _as_posix(path).name
