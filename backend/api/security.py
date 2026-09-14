"""Keep the local backend local.

The backend binds 127.0.0.1 and has no login: whoever reaches it is treated as
the user. That is right for the desktop app and for scripts on the same
machine, and wrong for a browser tab on some website, which can reach
127.0.0.1 just as well:

* a POST with only query parameters is a CORS "simple request" — sent without
  preflight, and the server acted on it (``/api/install/wsl-install`` deletes
  the WSL distribution);
* WebSockets have no CORS at all;
* DNS rebinding makes a foreign page same-origin, and then CORS protects
  nothing.

Two facts close all three. Browsers put an ``Origin`` header on every
non-GET request and on every WebSocket upgrade, and a ``Host`` header on
everything; a local process (curl, Electron's http.request, a test client)
sends no Origin. So a mutating request or WebSocket whose Origin is not a
local name is refused, a request whose Host is not a local name is refused,
and everything without an Origin keeps working. No token to hand around.

``ORIENTA_ALLOWED_HOSTS`` (comma separated) is ADDED to the local names — for
``uvicorn --host 0.0.0.0`` with a colleague on the LAN, list the name they
type into their browser; 127.0.0.1 and localhost keep working. Until that
variable is set, requests from a non-loopback client address are refused
too: the guard is anti-CSRF for browsers, not network access control, and
a LAN client can otherwise send ``Host: 127.0.0.1`` with no Origin at all.
GETs are not gated on Origin: they change nothing, and CORS already stops a
foreign page from reading the answer.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
LOOPBACK_CLIENTS = ("127.0.0.1", "::1", "localhost")
_READ_METHODS = ("GET", "HEAD", "OPTIONS")

# A host or origin netloc is letters, digits, dots, dashes, colons and IPv6
# brackets — nothing else. Spaces or commas mean two values were joined
# (a proxy folding duplicate headers), and "which one?" is not a question
# this guard should answer.
_NETLOC_RE = re.compile(r"^[A-Za-z0-9.\-:\[\]]+$")

_hosts_cache: frozenset[str] | None = None
_lan_mode: bool = False


def allowed_hosts() -> frozenset[str]:
    """Local host names plus ORIENTA_ALLOWED_HOSTS, lower-cased, without ports."""
    global _hosts_cache, _lan_mode
    if _hosts_cache is None:
        raw = os.environ.get("ORIENTA_ALLOWED_HOSTS", "")
        extra = {h.strip().lower() for h in raw.split(",") if h.strip()}
        _lan_mode = bool(extra)
        _hosts_cache = frozenset(set(DEFAULT_ALLOWED_HOSTS) | extra)
        if extra - set(DEFAULT_ALLOWED_HOSTS):
            logger.info("Allowed hosts extended via ORIENTA_ALLOWED_HOSTS: %s "
                        "(non-loopback clients are now accepted)",
                        ", ".join(sorted(extra)))
    return _hosts_cache


def lan_mode() -> bool:
    """True once ORIENTA_ALLOWED_HOSTS names anything: the operator opted in
    to non-loopback clients."""
    allowed_hosts()
    return _lan_mode


def reset_cache() -> None:
    """Forget the parsed environment (tests change it between cases)."""
    global _hosts_cache, _lan_mode
    _hosts_cache = None
    _lan_mode = False


def _host_of(netloc: str) -> str:
    """'localhost:8000' -> 'localhost'; '[::1]:8000' -> '[::1]'; junk -> ''."""
    netloc = netloc.strip().lower()
    if not _NETLOC_RE.match(netloc):
        return ""
    if netloc.startswith("["):
        end = netloc.find("]")
        return netloc[: end + 1] if end != -1 else ""
    return netloc.rsplit(":", 1)[0] if ":" in netloc else netloc


def client_is_local(client) -> bool:
    """ASGI ``scope['client']`` on the loopback interface (or unknown)?"""
    if not client:
        return True  # no transport address: an in-process call, not a socket
    return str(client[0]).lower() in LOOPBACK_CLIENTS


def host_is_local(host_header: str | None, hosts: frozenset[str] | set[str]) -> bool:
    if not host_header:
        return False
    return _host_of(host_header) in hosts


def origin_is_local(origin: str | None, hosts: frozenset[str] | set[str]) -> bool:
    """True for an absent Origin (not a browser) or a local one.

    An Origin is scheme + host [+ port] and nothing else: no path, no
    credentials, no query. Anything shaped differently is refused rather than
    guessed at — 'null' (redirect chains, sandboxed frames) included.
    """
    if origin is None:
        return True
    origin = origin.strip()
    if not origin:
        return False
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    if parts.path or parts.query or parts.fragment:
        return False
    if "@" in parts.netloc:
        return False
    return _host_of(parts.netloc) in hosts


class LocalOriginGuard:
    """Pure ASGI middleware: Host check on everything, Origin check on
    mutating HTTP requests and on every WebSocket upgrade."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        hosts = allowed_hosts()

        if not lan_mode() and not client_is_local(scope.get("client")):
            logger.warning("Refused %s %s from %s: not a loopback client "
                           "(set ORIENTA_ALLOWED_HOSTS for LAN use)",
                           kind, scope.get("path"), scope.get("client"))
            await self._refuse(scope, receive, send, 403,
                               "Refused: client is not on this machine. "
                               "Set ORIENTA_ALLOWED_HOSTS to serve other machines.")
            return

        if not host_is_local(headers.get("host"), hosts):
            logger.warning("Refused %s %s: Host %r is not a local name (DNS rebinding?)",
                           kind, scope.get("path"), headers.get("host"))
            await self._refuse(scope, receive, send, 400,
                               "Refused: Host header is not a local name.")
            return

        gated = kind == "websocket" or scope.get("method", "GET") not in _READ_METHODS
        if gated and not origin_is_local(headers.get("origin"), hosts):
            logger.warning("Refused %s %s: Origin %r is not local (cross-site request)",
                           kind, scope.get("path"), headers.get("origin"))
            await self._refuse(scope, receive, send, 403,
                               "Refused: request comes from a non-local web origin.")
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            # Closing before accept is the ASGI way to reject an upgrade.
            await send({"type": "websocket.close", "code": 1008, "reason": detail[:120]})
            return
        response = JSONResponse({"detail": detail}, status_code=status)
        await response(scope, receive, send)
