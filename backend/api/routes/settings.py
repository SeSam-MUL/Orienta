"""Settings API Routes — backed by per-machine user_config_manager.

Manages application settings:
- Server mode configuration (database root, enabled, offline mode)
- Manual paths for EMsoft/EMSphinx binaries
- API keys (Materials Project, …)
- System status (read-only, delegates to simulation system-status)

Storage location: per-machine user config (NOT in the project tree).
See `backend/api/services/user_config_manager.py` for details. The old
`<project_root>/data/server_config.json` and `data/manual_paths.json`
are auto-migrated and deleted on first load.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.services import user_config_manager as uc

logger = logging.getLogger(__name__)
router = APIRouter()


# Server subfolder names. These MUST match database.py's `_SERVER_SUBDIRS`,
# which is what the React Database Browser's sync / upload / download / browse
# actually read and write. The server uses an EBSD_ prefix for EVERY category
# (incl. CIF/XTAL) — unlike the LOCAL tree, where CIF/XTAL have no prefix.
#
# Previously these came from path_utils.DATABASE_SUBFOLDERS (cif -> "CIF_Library",
# xtal -> "XTAL_Library"), so Test Connection counted empty folders while the real
# data sat in EBSD_CIF_Library / EBSD_XTAL_Library — it reported "0 files" even
# after a successful sync. Importing the single source of truth keeps them aligned.
from backend.api.routes.database import _SERVER_SUBDIRS as SERVER_SUBDIRS  # noqa: E402


# --- Request models ---

class ServerConfigRequest(BaseModel):
    enabled: bool = True
    database_root: str = ""
    offline_mode: bool = False


class ManualPathsRequest(BaseModel):
    emsoft_bin_dir: str = ""
    emsphinx_dir: str = ""


class TestServerRequest(BaseModel):
    database_root: str


class ApiKeyRequest(BaseModel):
    """Update one or more API keys. Use an empty string to clear a key.
    Omit a field (or pass null) to leave it unchanged."""
    materials_project: Optional[str] = Field(
        None,
        description="Materials Project API key (free signup at https://materialsproject.org)",
    )


class ApiKeyTestRequest(BaseModel):
    name: str = Field(..., description="API-key name to test (e.g. 'materials_project')")


# --- Endpoints ---

@router.get("")
async def get_settings():
    """Return all application settings (with API keys REDACTED — never the raw key)."""
    cfg = uc.load_config()
    api_keys = cfg.get("api_keys", {})
    redacted = {
        name: {
            "configured": bool(value),
            "preview": uc.redact_key(value),
        }
        for name, value in api_keys.items()
    }
    return {
        "server": cfg.get("server", {}),
        "manual_paths": cfg.get("manual_paths", {}),
        "api_keys": redacted,
        "config_path": str(uc.config_path()),
    }


@router.put("/server")
async def save_server_config(body: ServerConfigRequest):
    """Persist server mode configuration."""
    data = body.model_dump()
    try:
        uc.update_section("server", data)
    except Exception as exc:
        logger.error("Failed to save server config: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not save settings: {exc}")
    return {"status": "ok", "saved": data}


@router.put("/paths")
async def save_manual_paths(body: ManualPathsRequest):
    """Persist manual binary paths that override auto-detection."""
    data = body.model_dump()
    try:
        uc.update_section("manual_paths", data)
    except Exception as exc:
        logger.error("Failed to save manual paths: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not save paths: {exc}")
    return {"status": "ok", "saved": data}


@router.put("/api-keys")
async def save_api_keys(body: ApiKeyRequest):
    """Update one or more API keys. Only fields present in the request are
    written; omit a field (or pass null) to leave it unchanged. Empty string
    clears a key."""
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400,
                            detail="No API-key fields provided.")
    try:
        uc.update_section("api_keys", data)
    except Exception as exc:
        logger.error("Failed to save API keys: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Could not save API keys: {exc}")
    return {
        "status": "ok",
        "saved": {k: uc.redact_key(v) for k, v in data.items()},
    }


@router.post("/api-keys/test")
async def test_api_key(body: ApiKeyTestRequest):
    """Test connectivity for a configured API key.

    Currently supports 'materials_project'. Returns success/failure +
    a short status message. NEVER returns the actual key value.
    """
    name = (body.name or "").strip()
    if not uc.has_api_key(name):
        return {"name": name, "success": False, "error": "no key configured"}
    if name == "materials_project":
        try:
            import asyncio
            from backend.api.services.crystal_hint_mp import test_mp_key
            # `test_mp_key` does a blocking `requests.get`; run it off the
            # event loop so other requests don't stall for up to 10s.
            ok, msg = await asyncio.to_thread(
                test_mp_key, uc.get_api_key("materials_project"),
            )
            return {"name": name, "success": ok, "message": msg}
        except ImportError:
            return {"name": name, "success": False,
                    "error": "MP client not available (crystal_hint_mp module missing)"}
        except Exception as exc:
            return {"name": name, "success": False, "error": str(exc)}
    return {"name": name, "success": False,
            "error": f"Testing for '{name}' not implemented"}


@router.post("/server/test")
async def test_server_connection(body: TestServerRequest):
    """Check whether the given database root is reachable and report subdirectory status."""
    root = Path(body.database_root)

    if not body.database_root:
        raise HTTPException(status_code=400, detail="database_root must not be empty")

    reachable = root.exists() and root.is_dir()

    directories: dict = {}
    for key, subdir_name in SERVER_SUBDIRS.items():
        subdir = root / subdir_name
        exists = subdir.exists() and subdir.is_dir()
        file_count = 0
        if exists:
            try:
                file_count = sum(1 for f in subdir.rglob("*") if f.is_file())
            except PermissionError:
                file_count = -1
        directories[key] = {
            "exists": exists,
            "path": str(subdir),
            "file_count": file_count,
        }

    return {"reachable": reachable, "directories": directories}


@router.post("/server/create-dirs")
async def create_server_dirs(body: TestServerRequest):
    """Create missing standard subdirectories on the server database root."""
    root = Path(body.database_root)

    if not body.database_root or not root.is_dir():
        raise HTTPException(status_code=400, detail="database_root is not a valid directory")

    created = []
    errors = []
    for key, subdir_name in SERVER_SUBDIRS.items():
        subdir = root / subdir_name
        if subdir.exists():
            continue
        try:
            subdir.mkdir(parents=True, exist_ok=True)
            created.append(subdir_name)
        except Exception as e:
            errors.append(f"{subdir_name}: {e}")

    return {"created": created, "errors": errors}
