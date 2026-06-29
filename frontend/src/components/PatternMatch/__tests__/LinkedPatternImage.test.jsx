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
  it('renders a crosshair when hover is set', () => {
    const { container } = render(<LinkedPatternImage {...props} hover={{ x: 0.5, y: 0.5 }} onHover={() => {}} onClick={() => {}} />);
    expect(container.querySelector('[data-crosshair]')).toBeTruthy();
  });
});
