// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';
import ThresholdHistogram from './ThresholdHistogram';

vi.mock('./hooks/useThresholdHistogram', () => ({
  useThresholdHistogram: () => ({ hist: new Uint32Array([1, 2, 3, 4]), min: 0, max: 255 }),
}));
vi.mock('../../theme/components', () => ({
  colors: { bg: '#000', purple: '#a0f', cyan: '#0ff', bgSecondary: '#222', border: '#444', text: '#fff', textSecondary: '#aaa' },
}));

afterEach(cleanup);

describe('ThresholdHistogram', () => {
  it('renders histogram bars', () => {
    const { container } = render(<ThresholdHistogram bitmap={{}} threshold={{ min: 20, max: 200 }} onThresholdChange={() => {}} />);
    const bars = container.querySelectorAll('svg rect');
    expect(bars.length).toBeGreaterThan(3);
  });

  it('renders two sliders bound to threshold values', () => {
    const { container } = render(<ThresholdHistogram bitmap={{}} threshold={{ min: 20, max: 200 }} onThresholdChange={() => {}} />);
    const sliders = container.querySelectorAll('input[type=range]');
    expect(sliders.length).toBe(2);
    expect(sliders[0].value).toBe('20');
    expect(sliders[1].value).toBe('200');
  });

  it('fires onThresholdChange with new min on left-slider change', () => {
    const onChange = vi.fn();
    const { container } = render(<ThresholdHistogram bitmap={{}} threshold={{ min: 20, max: 200 }} onThresholdChange={onChange} />);
    const sliders = container.querySelectorAll('input[type=range]');
    fireEvent.change(sliders[0], { target: { value: '50' } });
    expect(onChange).toHaveBeenCalledWith({ min: 50, max: 200 });
  });

  it('fires onThresholdChange with new max on right-slider change', () => {
    const onChange = vi.fn();
    const { container } = render(<ThresholdHistogram bitmap={{}} threshold={{ min: 20, max: 200 }} onThresholdChange={onChange} />);
    const sliders = container.querySelectorAll('input[type=range]');
    fireEvent.change(sliders[1], { target: { value: '150' } });
    expect(onChange).toHaveBeenCalledWith({ min: 20, max: 150 });
  });
});
