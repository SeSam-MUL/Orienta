import { useState } from 'react';
import useDataStore from '../../stores/useDataStore';
import { isHexScan } from '../../services/scanGeometry';

/**
 * The neighbourhood-averaging radius of a Crystal Hint mode.
 *
 * The average is taken over the (2r + 1) x (2r + 1) window of patterns around a point on the index
 * grid. On a hexagonal scan that window reaches up to 1.8 steps away and misses true neighbours, so
 * the backend refuses any radius above 0 there (`averaging_asked` in `hex_guard.py`). On such a
 * scan the radius that is USED is 0 and the control is locked, whatever was chosen before; the
 * choice itself is kept, so a square scan loaded later gets it back.
 *
 * @param {number} defaultRadius  the radius a square scan starts with
 * @returns {{avgRadius:number, setAvgRadius:Function, avgLocked:boolean}}
 */
export default function useAvgRadius(defaultRadius = 1) {
  const avgLocked = useDataStore((s) => isHexScan(s.scanGeometry));
  const [chosen, setChosen] = useState(defaultRadius);
  return { avgRadius: avgLocked ? 0 : chosen, setAvgRadius: setChosen, avgLocked };
}
