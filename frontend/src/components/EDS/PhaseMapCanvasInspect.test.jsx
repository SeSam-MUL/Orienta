// @vitest-environment jsdom
/**
 * A click on the phase map must drive the same per-pixel panels every other
 * map on this page drives.
 *
 * The bug this guards: the phase map was the one clickable surface that only
 * updated its own readout. A user clicking a Cu-rich region saw the
 * candidate list for whatever pixel they had last clicked on a TILE — the
 * suggestion panel read "Pixel (51, 122)" while the map readout under their
 * cursor read "Pixel (73, 133)", and the phases listed belonged to neither
 * the region they were pointing at nor anything they could see.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k) => k }),
}));
vi.mock('../../services/api', () => ({ edsApi: {} }));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));

import { PhaseMapCanvas } from './PhaseMapPanel';

/** 1x1 transparent PNG — enough for the component to render an <img>. */
const PNG =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

function makeHandle() {
  return {
    phaseMap: { loaded: true, image: PNG, n_rows: 10, n_cols: 10 },
    hoveredPixel: null,
    setHoveredPixel: vi.fn(),
    region: { rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 },
    setRegion: vi.fn(),
    paintMode: 'rectangle',
    polygonVertices: [],
    setPolygonVertices: vi.fn(),
  };
}

/**
 * Give the map a layout box.
 *
 * Measured on the HOST, not on the <img>: since the zoom landed, the pointer
 * maths reads the untransformed outer box on purpose — `getBoundingClientRect`
 * already includes a CSS transform, so measuring the zoomed node would count
 * the zoom twice. Stubbing the img here would test a path the app no longer
 * takes.
 */
function prepareMap(container) {
  const img = container.querySelector('img');
  const host = img.closest('div[style*="overflow"]') || img.parentElement.parentElement;
  const rect = {
    left: 0, top: 0, width: 100, height: 100,
    right: 100, bottom: 100, x: 0, y: 0, toJSON: () => {},
  };
  host.getBoundingClientRect = () => rect;
  return img;
}

describe('clicking the phase map', () => {
  it('reports the clicked pixel to the page', () => {
    const onInspect = vi.fn();
    const { container } = render(
      <PhaseMapCanvas handle={makeHandle()} onInspect={onInspect} />,
    );
    const img = prepareMap(container);

    // Press and release on the same spot = inspect, not a region drag.
    fireEvent.mouseDown(img, { clientX: 35, clientY: 45 });
    fireEvent.mouseUp(img, { clientX: 35, clientY: 45 });

    expect(onInspect).toHaveBeenCalledTimes(1);
    const [row, col] = onInspect.mock.calls[0];
    expect(row).toBe(4);
    expect(col).toBe(3);
  });

  it('does not report a pixel when the user drags a region', () => {
    const onInspect = vi.fn();
    const handle = makeHandle();
    const { container } = render(
      <PhaseMapCanvas handle={handle} onInspect={onInspect} />,
    );
    const img = prepareMap(container);

    fireEvent.mouseDown(img, { clientX: 10, clientY: 10 });
    fireEvent.mouseMove(img, { clientX: 60, clientY: 70 });
    fireEvent.mouseUp(img, { clientX: 60, clientY: 70 });

    expect(onInspect).not.toHaveBeenCalled();
    expect(handle.setRegion).toHaveBeenCalledWith({
      rowStart: 1, rowEnd: 7, colStart: 1, colEnd: 6,
    });
  });

  it('works without an onInspect handler', () => {
    const { container } = render(<PhaseMapCanvas handle={makeHandle()} />);
    const img = prepareMap(container);
    expect(() => {
      fireEvent.mouseDown(img, { clientX: 35, clientY: 45 });
      fireEvent.mouseUp(img, { clientX: 35, clientY: 45 });
    }).not.toThrow();
  });
});
