"""Imports nothing but the stdlib, so it is importable from Task 2 onward.

Its whole job is to be a module that COULD be imported, so the test that
asserts discovery did not import it is able to fail when discovery does.
"""
IMPORTED = True


def analyse(context):                       # pragma: no cover - never called
    return []
