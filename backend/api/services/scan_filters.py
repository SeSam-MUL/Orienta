"""The neighbourhood filters of a phase map, written once for any scan topology.

``checkpoint_writer.CheckpointWriter.clean_phase_assignment`` cleans a phase map with three operations that look at
a point's neighbours: it labels same-phase clusters (``ndi.label``, 4-connectivity), takes a majority vote in a
window (``ndi.convolve`` with a box) and fills holes from the nearest indexed point (``distance_transform_edt`` on
index distance). On a hexagonal scan each of the three is wrong when read off the index grid: a point has six
neighbours, the window of a point depends on the parity of its row, and the nearest point in index space is not the
nearest on the sample. These are the same three operations on a ``Topology`` (``scan_neighbours``), so one definition
of "neighbour" serves a square and a hexagonal scan. Maps here are ``(n_rows, n_cols)`` arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from backend.api.services.scan_neighbours import Topology


#: Largest ``size`` a hexagonal scan accepts for the modal filter. The interface offers 3 and 5; the vote gathers
#: ``n_points x (3 r^2 + 3 r + 1)`` int64 for ``r = (size - 1) // 2`` rings, which is about 490 MB at size 5 on a
#: 1000 x 1200 map and 19 GB at size 51, so a request is refused above this before anything is computed.
MAX_HEX_MODAL_FILTER_SIZE = 7


def cluster_sizes(mask: np.ndarray, topology: Topology) -> np.ndarray:
    """For every point of ``mask`` the size of the cluster of neighbouring True points it belongs to; 0 elsewhere."""
    flat = np.asarray(mask, dtype=bool).ravel()
    a, b = topology.pairs()
    keep = flat[a] & flat[b]
    graph = coo_matrix((np.ones(int(keep.sum()), dtype=np.int8), (a[keep], b[keep])),
                       shape=(flat.size, flat.size))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return np.where(flat, sizes[labels], 0).reshape(np.shape(mask))


def modal_vote(ids: np.ndarray, topology: Topology, size: int) -> np.ndarray:
    """The majority phase id in the window around every point, -1 where no indexed point is in it.

    ``size`` is the side of the square window the interface offers (3 or 5); a hexagonal scan uses the disc that
    stands for it: ``(size - 1) // 2`` rings, i.e. the point and its 6 neighbours for 3, 19 points for 5. The
    point itself votes. A tie goes to the lowest id (``argmax``), as in the square code."""
    rings = max(1, (int(size) - 1) // 2)
    table = topology.table(rings=rings, include_self=True)
    flat = np.asarray(ids).ravel()
    present = np.unique(flat[flat >= 0])
    if present.size == 0:
        return np.asarray(ids).astype(np.int16).copy()
    gathered = flat[np.maximum(table, 0)]
    gathered[table < 0] = -1
    votes = np.stack([(gathered == pid).sum(axis=1) for pid in present])
    best = present[np.argmax(votes, axis=0)]
    return np.where(votes.sum(axis=0) > 0, best, -1).astype(np.int16).reshape(np.shape(ids))


def nearest_valid(valid: np.ndarray, topology: Topology, targets: np.ndarray | None = None) -> np.ndarray:
    """The flat index of the nearest True point of ``valid`` on the sample (a True point is its own).

    Without ``targets`` the answer is for every point, an ``(n_rows, n_cols)`` array. With ``targets``, a boolean
    map of the points whose neighbour is wanted, only those are looked up and the answer is the 1-D array of
    their indices in row-major order, i.e. ``nearest_valid(valid, topology)[targets]`` without the lookup of the
    other points (a cleanup asks for a few percent of the map).

    Equidistant points are broken arbitrarily, as the distance transform it replaces does."""
    x, y = topology.xy()
    pts = np.column_stack([x.ravel(), y.ravel()])
    where = np.flatnonzero(np.asarray(valid, dtype=bool).ravel())
    tree = cKDTree(pts[where])
    if targets is None:
        _, nearest = tree.query(pts, workers=-1)
        return where[nearest].reshape(np.shape(valid))
    asked = np.flatnonzero(np.asarray(targets, dtype=bool).ravel())
    _, nearest = tree.query(pts[asked], workers=-1)
    return where[nearest]
