"""Two analyses whose maps are told apart only by their numbers.

Same output key, same shape, same dtype. If the store cannot tell the two
analyses apart, one of these URLs serves the other's values and there is
nothing in the response -- not the label, not the shape, not the dtype -- that
would show it.
"""
import numpy as np

from backend.api.services.addons.outputs import MapOutput

#: Deliberately more than the cap the eviction test sets, so ``many`` fills the
#: store past it inside a single run.
N_MANY = 3


def first(context):
    return [MapOutput(key="m", label="First",
                      values=np.ones(context.shape, dtype=np.float64))]


def second(context):
    return [MapOutput(key="m", label="Second",
                      values=np.full(context.shape, 2.0, dtype=np.float64))]


def many(context):
    return [MapOutput(key=f"m{i}", label=f"Map {i}",
                      values=np.full(context.shape, float(i),
                                     dtype=np.float64))
            for i in range(N_MANY)]
