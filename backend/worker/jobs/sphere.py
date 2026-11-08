"""The master-pattern sphere, computed in the compute worker.

The same ``_build_payload`` the backend used to run, so the result is the same
to the bit. An out-of-memory error is translated HERE: exception chains do not
cross the process boundary, so the backend could not recognise it afterwards.
"""
from __future__ import annotations

from backend.worker.jobs import JobError


def preload() -> None:
    """The heavy imports of this job (torch through the renderer, the SHT reader).

    The worker runs this before it tells the backend that it has the job, so a
    slow first import (the 7-30 s of a cold start) counts as starting, not as
    computing.
    """
    from backend.api.services import sht_pattern_renderer, sht_sphere  # noqa: F401
    from backend.spherical_gpu.pipeline import sht_io  # noqa: F401


def run(args: dict, ctx) -> dict:
    from backend.api.services import sht_sphere
    from backend.api.services.sht_pattern_renderer import is_out_of_memory

    ctx.progress(None, "loading")
    try:
        return sht_sphere._build_payload(str(args["path"]), int(args["max_bandwidth"]),
                                         int(args["target_size"]))
    except sht_sphere.SphereOutOfMemoryError as e:
        raise JobError("sphereOutOfMemory", str(e)) from e
    except sht_sphere.SphereError as e:
        raise JobError("sphereFailed", str(e)) from e
    except FileNotFoundError as e:
        raise JobError("notFound", str(e)) from e
    except Exception as e:  # noqa: BLE001 -- only OOM is translated, the rest re-raised
        if is_out_of_memory(e):
            raise JobError("sphereOutOfMemory",
                           "not enough memory for the sphere at this resolution") from e
        raise
