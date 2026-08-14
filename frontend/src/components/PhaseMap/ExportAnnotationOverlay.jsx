import { useRef, useState } from 'react';
import AnnotationLayer from './annotations/AnnotationLayer';
import { applyPatch, withRemoved } from './annotations/exportAnnotEdits';

/**
 * The map's annotations, editable on top of the export preview.
 *
 * Mounted through the export dialog's `overlay` slot, which hands over the
 * rectangle the source image occupies in the preview. Annotation coordinates
 * are normalised [0..1] of that image, so a box sized to exactly that rectangle
 * is all `AnnotationLayer` needs — crop, zoom and border are already folded
 * into the rectangle by the dialog.
 *
 * The annotations here are a COPY: arranging a figure must not disturb the
 * working view behind the dialog. Closing the dialog drops the copy.
 */
export default function ExportAnnotationOverlay({
  rect, annotations, onChange, ctx, selectedId = null, onSelect = null,
}) {
  const hostRef = useRef(null);
  // Selection normally lives in the page, so the properties panel in the
  // dialog's settings column edits the same annotation the user clicked on the
  // preview. The local state is only a fallback for callers that do not care.
  const [ownSelected, setOwnSelected] = useState(null);
  const sel = onSelect ? selectedId : ownSelected;
  const select = onSelect || setOwnSelected;

  const update = (id, patch) => onChange(applyPatch(annotations, id, patch));
  const remove = (id) => {
    onChange(withRemoved(annotations, id));
    if (sel === id) select(null);
  };

  return (
    <div
      ref={hostRef}
      data-export-annotations
      style={{
        position: 'absolute',
        // Placed on the MAP, which may sit to the right of a colour-bar column
        // and below the sheet's top edge once the figure has a margin.
        left: rect?.left ?? 0,
        top: rect?.top ?? 0,
        width: rect?.width ?? '100%',
        height: rect?.height ?? '100%',
        // The layer itself is transparent to the pointer; its widgets are not.
        pointerEvents: 'none',
      }}
    >
      <AnnotationLayer
        annotations={annotations}
        selectedId={sel}
        onSelect={select}
        onDelete={remove}
        onUpdate={update}
        containerRef={hostRef}
        ctx={{
          ...ctx,
          // In the export the map is drawn unzoomed at full grid, so the scale
          // bar must measure against THAT, not against the zoomed on-screen
          // view it was arranged on.
          mapContentBbox: null,
          mapZoomScale: 1,
          mapBoxWidthPx: rect?.width || null,
        }}
      />
    </div>
  );
}
