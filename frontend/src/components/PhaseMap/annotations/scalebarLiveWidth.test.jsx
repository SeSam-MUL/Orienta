// @vitest-environment jsdom
/**
 * The on-screen scale bar has to follow the map's CURRENT width.
 *
 * `PhaseMapPage` built the annotation context in its render body:
 *
 *     mapBoxWidthPx: mapContainerRef.current
 *       ?.querySelector('[data-phasemap-map-box]')?.getBoundingClientRect().width
 *
 * — a DOM read frozen into a prop. `LayeredCanvas` recomputes its letterboxed
 * fit from its own ResizeObserver, so resizing the window changes how wide the
 * map is drawn without re-rendering the page. The bar then kept sizing itself
 * against the width the box had at the last page render: the label said 5 µm
 * and the bar was some other length, silently, and resizing the window before
 * exporting a figure is exactly what someone does.
 *
 * `AnnotationLayer` already runs a ResizeObserver for its own geometry, so it
 * measures the map box there too and that measurement wins.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, act } from '@testing-library/react';
import AnnotationLayer from './AnnotationLayer';

vi.mock('../../../theme/tokens', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff', cyan: '#0ff',
    green: '#0f0', red: '#f00', yellow: '#fd0', purple: '#a6f', orange: '#f80',
  },
}));

afterEach(cleanup);

const CONTAINER = { width: 900, height: 500 };
const NATIVE = { w: 120, h: 90 };     // scan grid
const STEP_X = 0.5;                    // µm per scan pixel

/** Size the container and the map box, then fire the observers. */
function stubSizes(container, mapBoxWidth) {
  container.getBoundingClientRect = () => ({
    width: CONTAINER.width, height: CONTAINER.height,
    top: 0, left: 0, right: CONTAINER.width, bottom: CONTAINER.height, x: 0, y: 0,
  });
  const box = container.querySelector('[data-phasemap-map-box]');
  box.getBoundingClientRect = () => ({
    width: mapBoxWidth, height: mapBoxWidth * (NATIVE.h / NATIVE.w),
    top: 0, left: 0, right: mapBoxWidth, bottom: 0, x: 0, y: 0,
  });
}

/** Width of the drawn bar, in CSS px. */
function barWidth(container) {
  const bar = container.querySelector('[data-annotation-widget] div div');
  return parseFloat(bar?.style?.width || 'NaN');
}

function Harness({ mapBoxWidth, staleCtxWidth }) {
  const ref = { current: null };
  return (
    <div
      ref={(el) => {
        ref.current = el;
        if (el) stubSizes(el, mapBoxWidth);
      }}
      style={{ position: 'relative' }}
    >
      <div data-phasemap-map-box />
      <AnnotationLayer
        annotations={[{
          id: 'sb', type: 'scalebar', x: 0.1, y: 0.8, w: 0.3, h: 0.1,
          props: { lengthUm: 10, fontSize: 12 },
        }]}
        selectedId={null}
        onSelect={() => {}}
        onDelete={() => {}}
        onUpdate={() => {}}
        containerRef={ref}
        ctx={{
          stepX: STEP_X,
          mapNativeSize: NATIVE,
          mapContentBbox: null,
          mapZoomScale: 1,
          // What the page froze in at its last render — deliberately stale.
          mapBoxWidthPx: staleCtxWidth,
        }}
      />
    </div>
  );
}

describe('the on-screen scale bar measures the map as it is NOW', () => {
  it('uses the live map-box width, not the one frozen into the context', () => {
    // 10 µm at 0.5 µm/px = 20 scan px of 120; the map box is 600 CSS px wide,
    // so the bar must be 20/120 * 600 = 100 px. The stale context claims the
    // box is 300 px wide, which would give 50.
    const { container } = render(<Harness mapBoxWidth={600} staleCtxWidth={300} />);
    expect(barWidth(container)).toBeCloseTo(100, 3);
  });

  it('falls back to the context where there is no map box to measure', () => {
    // The export dialog reuses these widgets over a preview that has no
    // `[data-phasemap-map-box]`; there the context IS the live answer.
    const ref = { current: null };
    const { container } = render(
      <div ref={(el) => {
        ref.current = el;
        if (el) {
          el.getBoundingClientRect = () => ({
            width: CONTAINER.width, height: CONTAINER.height,
            top: 0, left: 0, right: 0, bottom: 0, x: 0, y: 0,
          });
        }
      }}>
        <AnnotationLayer
          annotations={[{
            id: 'sb', type: 'scalebar', x: 0.1, y: 0.8, w: 0.3, h: 0.1,
            props: { lengthUm: 10, fontSize: 12 },
          }]}
          selectedId={null}
          onSelect={() => {}}
          onDelete={() => {}}
          onUpdate={() => {}}
          containerRef={ref}
          ctx={{
            stepX: STEP_X, mapNativeSize: NATIVE, mapContentBbox: null,
            mapZoomScale: 1, mapBoxWidthPx: 300,
          }}
        />
      </div>,
    );
    expect(barWidth(container)).toBeCloseTo(50, 3);
  });
});

describe('a scale bar with no scale says so', () => {
  it('draws no bar and no micrometre label when the map has no step size', () => {
    // `knownStepX` in PhaseMapPage is null until the step comes from the file
    // or the user types one. Drawing a full-width bar under a confident "10 µm"
    // — which is what happened — is the exact failure this annotation exists
    // to avoid.
    const ref = { current: null };
    const { container } = render(
      <div ref={(el) => {
        ref.current = el;
        if (el) {
          el.getBoundingClientRect = () => ({
            width: 900, height: 500, top: 0, left: 0, right: 0, bottom: 0, x: 0, y: 0,
          });
        }
      }}>
        <AnnotationLayer
          annotations={[{
            id: 'sb', type: 'scalebar', x: 0.1, y: 0.8, w: 0.3, h: 0.1,
            props: { lengthUm: 10, fontSize: 12 },
          }]}
          selectedId={null}
          onSelect={() => {}}
          onDelete={() => {}}
          onUpdate={() => {}}
          containerRef={ref}
          ctx={{
            stepX: null, mapNativeSize: NATIVE, mapContentBbox: null,
            mapZoomScale: 1, mapBoxWidthPx: 600,
          }}
        />
      </div>,
    );
    expect(container.querySelector('[data-scalebar-noscale]')).toBeTruthy();
    expect(container.textContent).not.toContain('10 µm');
  });
});
