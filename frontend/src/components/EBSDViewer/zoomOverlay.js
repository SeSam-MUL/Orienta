/**
 * Overlay geometry for the zoomed EBSD-viewer overview.
 *
 * The overview image carries the zoom as a CSS transform (see zoomView.js in
 * ../EDS), but its ROI rectangle and crosshair live in a SIBLING overlay div
 * that is positioned with left/top/width/height in percent. Putting those
 * inside the transform would scale their border and cross arms along with the
 * image — a 2 px ROI border would render 32 px thick at 16x. So the overlay
 * stays untransformed and is moved numerically instead, by applying the exact
 * same view math in percent space.
 *
 * `ovImgRect` is the letterboxed image area (object-fit: contain) as a
 * percentage of the untransformed box. `zoomRectPct` maps it through a view.
 */

import { clampView } from '../EDS/zoomView';

/**
 * Apply a zoom view to a rect given in percent of the host box.
 *
 * Derivation — must stay in lockstep with `viewToTransform`: a content
 * fraction f of the box renders at screen fraction 0.5 + s*(f - c). In percent
 * that is 50 + s*(pct - 100*c), and a length simply scales by s.
 *
 * @param {{left:number,top:number,width:number,height:number}} rect  percent
 * @param {{scale:number,cx:number,cy:number}} view
 * @returns {{left:number,top:number,width:number,height:number}} percent
 */
export function zoomRectPct(rect, view) {
  const v = clampView(view);
  return {
    left: 50 + v.scale * (rect.left - 100 * v.cx),
    top: 50 + v.scale * (rect.top - 100 * v.cy),
    width: v.scale * rect.width,
    height: v.scale * rect.height,
  };
}
