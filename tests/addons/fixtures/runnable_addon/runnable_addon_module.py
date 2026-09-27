"""A minimal add-on, used to prove the runner rather than to be useful."""
import numpy as np

from backend.api.services.addons.outputs import MapOutput, ScalarOutput


def analyse(context, scale: float = 1.0, fail: bool = False, report_dir=None):
    if fail:
        raise RuntimeError("this add-on chose to fail")
    quality = context.quality
    if quality is None:
        return [ScalarOutput(key="mean", label="Mean quality", value=0.0),
                ScalarOutput(key="n_px", label="Pixels used", value=0.0)]
    context.report("scaling", 0.5)
    return [
        MapOutput(key="scaled", label="Scaled quality",
                  values=np.asarray(quality) * scale),
        ScalarOutput(key="mean", label="Mean quality",
                     value=float(np.nanmean(quality)) * scale),
        ScalarOutput(key="n_px", label="Pixels used",
                     value=float(np.asarray(quality).size)),
        # Deliberately collides with the INPUT parameter of the same name,
        # and deliberately reports a different number, so the "input wins"
        # rule has something to be wrong about.
        ScalarOutput(key="scale", label="Scale, misreported on purpose",
                     value=float(scale) * 100.0),
    ]
