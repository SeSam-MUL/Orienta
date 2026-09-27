"""A script-shaped module: it exits at import time, not at call time.

Activation is where an author meets this, because probe_import is the first
thing that runs their module at all.
"""
import sys

sys.exit(3)


def analyse(context):                       # pragma: no cover - never reached
    return []
