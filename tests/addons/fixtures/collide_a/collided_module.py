"""One of two add-ons that ship a module of the SAME top-level name.

A top-level module name is one process-wide namespace, and import_module
answers from sys.modules first, so without the runner's identity check
whichever of the two ran first serves BOTH runs - and add-on B's results are
add-on A's work, attributed and cited to B.
"""
from backend.api.services.addons.outputs import ScalarOutput

WHO = "a"


def analyse(context):
    return [ScalarOutput(key="who", label="Which add-on actually ran",
                         value=1.0)]
