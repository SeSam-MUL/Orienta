"""Imported by exactly one test, which proves the product — and not a test
fixture — is what makes a folder-installed add-on importable."""
from backend.api.services.addons.outputs import ScalarOutput


def analyse(context):
    return [ScalarOutput(key="ok", label="Imported", value=1.0)]
