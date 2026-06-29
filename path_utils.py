"""Cross-platform path utilities for Windows/WSL compatibility.

Converts Windows paths (drive letters, UNC) to Linux/WSL mount paths
and vice versa. All path resolution happens at READ time — stored
paths in QSettings remain in their original format.
"""

import logging
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union

logger = logging.getLogger(__name__)

# Cached result for is_wsl()
_is_wsl: bool | None = None


def is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    global _is_wsl
    if _is_wsl is not None:
        return _is_wsl
    if sys.platform != "linux":
        _is_wsl = False
        return False
    try:
        with open("/proc/version", "r") as f:
            _is_wsl = "microsoft" in f.read().lower()
    except OSError:
        _is_wsl = False
    return _is_wsl


def is_windows_path(path_str: str) -> bool:
    """Check if a path string looks like a Windows path."""
    if not path_str:
        return False
    # Drive letter: C:\, M:\, etc.
    if re.match(r'^[A-Za-z]:[\\\/]', path_str):
        return True
    # UNC path: \\server\share
    if path_str.startswith('\\\\') or path_str.startswith('//'):
        return True
    return False


def resolve_path(path_str: str) -> Path:
    """Convert any path to the current platform's format.

    On Windows: returns Path as-is.
    On Linux/WSL: converts Windows drive letters and UNC paths to /mnt/ paths.

    Examples (on Linux):
        'M:\\Austausch\\SHT' → Path('/mnt/m/Austausch/SHT')
        'C:\\Users\\user'    → Path('/mnt/c/Users/user')
        '/mnt/m/Austausch'   → Path('/mnt/m/Austausch')  (unchanged)
        ''                   → Path('')
    """
    if not path_str or not path_str.strip():
        return Path(path_str) if path_str else Path()

    path_str = path_str.strip()

    # On Windows, no conversion needed
    if sys.platform == "win32":
        return Path(path_str)

    # Already a Linux path
    if path_str.startswith('/'):
        return Path(path_str)

    # Drive letter path: M:\Austausch\... → /mnt/m/Austausch/...
    drive_match = re.match(r'^([A-Za-z]):[\\\/](.*)', path_str)
    if drive_match:
        drive = drive_match.group(1).lower()
        rest = drive_match.group(2).replace('\\', '/')
        linux_path = f"/mnt/{drive}/{rest}"
        logger.debug("Converted Windows path '%s' → '%s'", path_str, linux_path)
        return Path(linux_path)

    # UNC path: \\server\share\... → warn, not directly mountable
    if path_str.startswith('\\\\') or path_str.startswith('//'):
        logger.warning(
            "UNC path '%s' detected on Linux. UNC paths are not directly "
            "accessible in WSL. Please use a mounted drive letter instead.",
            path_str
        )
        # Best-effort: convert backslashes, return as-is
        return Path(path_str.replace('\\', '/'))

    # Unknown format, return as-is
    return Path(path_str)


def _is_valid_unix_path(path: str) -> bool:
    """Validate that a string looks like a real Unix path, not an error message.

    Used to guard against WSL returning error messages (e.g. when ext4.vhdx
    is missing) which would otherwise be saved as binary paths in the config.

    Returns True only if:
    - Starts with /
    - Contains only valid path characters (alphanumeric, /, -, _, .)
    - Does NOT contain error keywords (Fehler, Error, failed, etc.)
    """
    if not path or not isinstance(path, str):
        return False
    if not path.startswith('/'):
        return False
    # Reject strings containing error keywords (German + English)
    _lower = path.lower()
    for keyword in ('fehler', 'error', 'failed', 'cannot', 'permission',
                    'denied', 'invalid', 'not found', 'no such'):
        if keyword in _lower:
            return False
    # Must only contain valid path chars: /, alphanumeric, -, _, .
    if not re.match(r'^[/\w.\-]+$', path):
        return False
    return True


# Greek letter → ASCII mapping for filename sanitization
_GREEK_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'λ': 'lambda',
}


def sanitize_filename(text: str) -> str:
    """Sanitize a filename to ASCII-safe characters.

    Replaces Greek letters with their English names, replaces all other
    non-alphanumeric characters (except '-' and '_') with '_', collapses
    multiple underscores, and strips leading/trailing underscores.

    This is the SINGLE SOURCE OF TRUTH for filename sanitization.
    All modules must use this function (or import it) to ensure consistent
    naming across simulation, file sync, and discovery.

    Example: "α-(AlMnSi)" → "alpha-_AlMnSi"
    """
    for gr, rep in _GREEK_MAP.items():
        text = text.replace(gr, rep)
    text = re.sub(r'[^0-9A-Za-z\-\_]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')


def ekev_to_kv_label(ekev: float) -> int:
    """Map an accelerating voltage (kV, float) to the integer label used in filenames.

    SINGLE SOURCE OF TRUTH for the ``_E{kV}kV`` / ``{kV}kV`` integer that appears
    in MC ``.h5``, master ``.h5`` and ``.sht`` filenames — and in the
    ``scan_missing`` globs that detect those files.

    Uses ``round`` (banker's-rounding-free nearest) rather than truncation so the
    naming is stable around half-integer voltages: at ``ekev=19.9`` both the writer
    and the scan agree on ``20``.  An earlier divergence (GPU runner used
    ``round`` while the scan + EMsoft naming truncated) made "Simulate All Missing"
    re-queue forever at non-integer kV.

    At integer kV (the common case) ``round`` and truncation agree, so existing
    on-disk files (named under the old truncation convention) are unaffected.

    Example: 19.0 -> 19, 19.5 -> 20, 19.9 -> 20, 20.0 -> 20.
    """
    return int(round(float(ekev)))


# Standard subfolder names for the EBSD database root
DATABASE_SUBFOLDERS = {
    "sht_database": "EBSD_SHT_Database",
    "h5_cache": "EBSD_H5_Cache",
    "cif_library": "CIF_Library",
    "xtal_library": "XTAL_Library",
    "dictionary_library": "Dictionary_Library",
}


def discover_database_paths(root: str | Path) -> dict:
    """Auto-discover standard subfolders in a database root directory.

    Args:
        root: Path to the database root folder (e.g. /mnt/m/Austausch/EBSD_Database)

    Returns:
        Dict with keys from DATABASE_SUBFOLDERS.
        Each value is a dict: {"path": Path, "exists": bool}
    """
    root = resolve_path(str(root)) if isinstance(root, str) else Path(root)
    result = {}
    for key, folder_name in DATABASE_SUBFOLDERS.items():
        subfolder = root / folder_name
        result[key] = {
            "path": subfolder,
            "exists": subfolder.is_dir(),
        }
    return result


def get_local_database_path() -> Path:
    """Return the local Database/ folder inside the project root.

    Computed dynamically from the location of this file (path_utils.py lives
    in the project root), so it works on any machine without hardcoded paths.

    Returns:
        Path to Kikuchipy_GUI/Database/
    """
    return Path(__file__).parent / "Database"


def get_platform_placeholder(windows_example: str) -> str:
    """Return platform-appropriate placeholder text for path input fields.

    Converts a Windows path example to Linux format when on Linux.
    """
    if sys.platform == "win32":
        return windows_example
    return str(resolve_path(windows_example))
