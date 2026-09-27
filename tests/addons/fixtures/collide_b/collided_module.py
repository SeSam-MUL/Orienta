"""The other half of the collision. Returns a DIFFERENT number on purpose, so
"B silently ran A's code" is something a test can see rather than infer."""
from backend.api.services.addons.outputs import ScalarOutput

WHO = "b"


def analyse(context):
    return [ScalarOutput(key="who", label="Which add-on actually ran",
                         value=2.0)]
