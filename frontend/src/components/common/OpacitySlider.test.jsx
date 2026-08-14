// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import OpacitySlider, { hexToRgb, trackBackground } from './OpacitySlider';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: { textSecondary: '#aaa' },
}));

describe('hexToRgb', () => {
  it('reads both short and long form', () => {
    expect(hexToRgb('#ff8800')).toEqual([255, 136, 0]);
    expect(hexToRgb('f80')).toEqual([255, 136, 0]);
  });

  it('falls back to black on anything unreadable', () => {
    expect(hexToRgb('rgb(1,2,3)')).toEqual([0, 0, 0]);
    expect(hexToRgb(null)).toEqual([0, 0, 0]);
  });
});

describe('trackBackground', () => {
  it('ramps from transparent to the chosen colour, over a checkerboard', () => {
    // The point of the control: the track shows THIS colour, not a fixed one.
    const bg = trackBackground('#bd93f9');
    expect(bg.backgroundImage).toContain('rgba(189,147,249,0)');
    expect(bg.backgroundImage).toContain('rgba(189,147,249,1)');
    expect(bg.backgroundImage).toContain('conic-gradient');
    expect(bg.backgroundSize).toBe('100% 100%, 10px 10px');
  });
});

describe('OpacitySlider', () => {
  it('is a real range input, so keyboard and click-to-jump still work', () => {
    render(<OpacitySlider value={0.6} onChange={() => {}} color="#112233" />);
    const input = screen.getByRole('slider');
    expect(input.type).toBe('range');
    expect(input.value).toBe('0.6');
    expect(input.getAttribute('aria-valuetext')).toBe('60%');
  });

  it('reports the new value as a fraction', () => {
    const onChange = vi.fn();
    render(<OpacitySlider value={0.6} onChange={onChange} />);
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0.25' } });
    expect(onChange).toHaveBeenCalledWith(0.25);
  });

  it('shows the percentage and clamps nonsense to the ends', () => {
    const { rerender } = render(<OpacitySlider value={2} onChange={() => {}} />);
    expect(screen.getByText('100%')).toBeTruthy();
    rerender(<OpacitySlider value={-1} onChange={() => {}} />);
    expect(screen.getByText('0%')).toBeTruthy();
    rerender(<OpacitySlider value={undefined} onChange={() => {}} />);
    expect(screen.getByText('0%')).toBeTruthy();
  });
});
