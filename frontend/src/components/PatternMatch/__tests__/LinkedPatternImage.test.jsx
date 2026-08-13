// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import LinkedPatternImage from '../LinkedPatternImage';

const props = { src: 'data:,', alt: 'Exp', naturalSize: { w: 100, h: 100 },
  markers: [{ id: 'm1', n: 1, x: 0.5, y: 0.5 }], hover: null };

describe('LinkedPatternImage', () => {
  it('renders a marker node per marker', () => {
    const { container } = render(<LinkedPatternImage {...props} onHover={() => {}} onClick={() => {}} />);
    expect(container.querySelectorAll('[data-marker]')).toHaveLength(1);
    expect(container.querySelector('[data-marker]').textContent).toBe('1');
  });
  it('calls onClick with normalized coords', () => {
    const onClick = vi.fn();
    const { container } = render(<LinkedPatternImage {...props} onHover={() => {}} onClick={onClick} />);
    const wrap = container.querySelector('[data-linked-wrap]');
    wrap.getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 100 });
    fireEvent.click(wrap, { clientX: 50, clientY: 50 });
    expect(onClick).toHaveBeenCalledWith({ x: 0.5, y: 0.5 });
  });
  it('stays exactly as it was without zoom props', () => {
    const { container } = render(<LinkedPatternImage {...props} onHover={() => {}} onClick={() => {}} />);
    const img = container.querySelector('img');
    expect(img.style.transform).toBe('none');
    expect(container.querySelector('[data-pattern-zoom-badge]')).toBeNull();
  });

  it('transforms the image and shows a resettable badge when zoomed', () => {
    const onResetView = vi.fn();
    const { container } = render(
      <LinkedPatternImage {...props} onHover={() => {}} onClick={() => {}}
        view={{ scale: 2, cx: 0.5, cy: 0.5 }} onZoomAt={() => {}} onPan={() => {}}
        onResetView={onResetView} />,
    );
    expect(container.querySelector('img').style.transform).toContain('scale(2)');
    const badge = container.querySelector('[data-pattern-zoom-badge]');
    expect(badge.textContent).toBe('2.0×');
    fireEvent.click(badge);
    expect(onResetView).toHaveBeenCalled();
  });

  it('moves the marker with the zoom instead of leaving it behind', () => {
    // A marker at the top-left quarter must ride outward when the view is
    // magnified about the centre — the overlay sits outside the transform, so
    // this is the maths that keeps it on its feature.
    const at = (view) => {
      const { container } = render(
        <LinkedPatternImage {...props} markers={[{ id: 'm', n: 1, x: 0.25, y: 0.25 }]}
          onHover={() => {}} onClick={() => {}} view={view} onZoomAt={() => {}} />,
      );
      return parseFloat(container.querySelector('[data-marker]').style.left);
    };
    const plain = at(null);
    const zoomed = at({ scale: 2, cx: 0.5, cy: 0.5 });
    expect(zoomed).toBeLessThan(plain);          // pushed away from centre
    expect(zoomed).toBeCloseTo(50 + 2 * (plain - 50), 6);
  });

  it('clips the magnified image to its own box, whatever the caller styles', () => {
    // The bug this guards: without the clip the zoomed panel spilled across the
    // dialog and covered its neighbours.
    const { container } = render(
      <LinkedPatternImage {...props} onHover={() => {}} onClick={() => {}}
        style={{ overflow: 'visible' }}
        view={{ scale: 4, cx: 0.5, cy: 0.5 }} onZoomAt={() => {}} />,
    );
    expect(container.querySelector('[data-linked-wrap]').style.overflow).toBe('hidden');
  });

  it('renders a crosshair when hover is set', () => {
    const { container } = render(<LinkedPatternImage {...props} hover={{ x: 0.5, y: 0.5 }} onHover={() => {}} onClick={() => {}} />);
    expect(container.querySelector('[data-crosshair]')).toBeTruthy();
  });
});
