"""
Guards that keep a test run out of the user's own files.

Deliberately NOT the development tree's conftest. That one carries fixtures for
a ``Test_data/`` directory this repository does not ship, and registers markers
``pytest.ini`` already declares; none of the tests here use either. What is
needed here is only the part that protects the machine the suite runs on.
"""

import os
import sys

# The suite imports modules that sit at the repository root (ebsd_utils, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing backend.api.main installs a rotating file handler at import time,
# so without this a test run writes into the real logs/orienta.log.
os.environ.setdefault("ORIENTA_NO_FILE_LOG", "1")

# The per-phase Hough reflector limits are stored under %APPDATA%/Kikuchipy.
# A test run must neither read nor overwrite what the user chose there.
os.environ.setdefault("ORIENTA_NO_PHASE_LIMIT_STORE", "1")
