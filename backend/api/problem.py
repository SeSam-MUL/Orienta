"""An HTTP error the interface can translate.

The body of an ``HTTPException`` reaches the screen: the frontend reads
``err.response.data.detail`` in 159 places and shows it. That text is English,
so a German user meets an English sentence — the M5 tester quoted one of them
verbatim ("No indexing result or analysis dataset available. Load data
first.").

The obvious fix, a dict in ``detail``, would change the shape those 159
readers see and turn every one of them into "[object Object]" on this route.
So the prose stays exactly where it was and the machine-readable code travels
in a header. Old readers are byte-for-byte unaffected; a caller that wants to
translate reads the header and falls back to the prose.

``X-Orienta-Code`` is listed in the CORS ``expose_headers`` so the dev server
on port 5173 can read it too; in the packaged app everything is same-origin.
"""
from __future__ import annotations

from fastapi import HTTPException

CODE_HEADER = "X-Orienta-Code"


def problem(status_code: int, code: str, message: str) -> HTTPException:
    """An HTTPException whose English prose is unchanged and whose code is a header.

    Parameters
    ----------
    status_code
        As before.
    code
        Stable identifier the interface translates on. Never shown.
    message
        The English sentence. Still the whole of ``detail``, so every existing
        reader keeps working.
    """
    return HTTPException(
        status_code=status_code,
        detail=message,
        headers={CODE_HEADER: code},
    )
