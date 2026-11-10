"""An .h5oina read straight from its arrays, independent of Orienta's loaders.

Orienta's side of the MTEX comparison reads the scans this way: Euler angles, phases and
positions of ``/1/EBSD/Data``, the step of ``/1/EBSD/Header``, every phase taken as m-3m (the
maps compared are aluminium). Tests compare Orienta's own loading with it.
"""
import numpy as np


def xmap_from_h5oina(path):
    """(CrystalMap, step) of the .h5oina at ``path``."""
    import h5py
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    with h5py.File(path, "r") as f:
        d, hd = f["1/EBSD/Data"], f["1/EBSD/Header"]
        euler = np.asarray(d["Euler"], dtype=np.float64)
        ph = np.asarray(d["Phase"]).astype(int).ravel()
        x = np.asarray(d["X"], dtype=np.float64).ravel()
        y = np.asarray(d["Y"], dtype=np.float64).ravel()
        step = float(hd["X Step"][()][0])
        ids = sorted(int(k) for k in hd["Phases"].keys())
        names = []
        for k in ids:
            nm = hd[f"Phases/{k}/Phase Name"][()]
            nm = nm[0] if np.ndim(nm) else nm
            names.append(nm.decode() if isinstance(nm, bytes) else str(nm))
    phases = PhaseList(names=names, point_groups=["m-3m"] * len(ids), ids=[k - 1 for k in ids])
    return CrystalMap(rotations=Rotation.from_euler(euler), phase_id=ph - 1, x=x, y=y, phase_list=phases), step
