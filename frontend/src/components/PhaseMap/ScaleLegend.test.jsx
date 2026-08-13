// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ScaleLegend, { formatScaleValue, stopsToGradient } from './ScaleLegend';

const CI_SCALE = {
  min: 0.1802, max: 0.4716, unit: '', cmap: 'RdYlGn',
  stops: ['#a50026', '#f46d43', '#ffffbf', '#66bd63', '#006837'],
};

describe('ScaleLegend', () => {
  it('states the range the map was painted with, not a nominal one', () => {
    // A confidence map stretched over 0.18..0.47 uses the whole colormap. A bar
    // labelled 0..1 would misstate every colour on it.
    render(<ScaleLegend label="CI (Best)" scale={CI_SCALE} />);
    expect(screen.getByText('0.472')).toBeTruthy();
    expect(screen.getByText('0.180')).toBeTruthy();
    expect(screen.getByText('0.326')).toBeTruthy();   // the midpoint
    expect(screen.getByText('CI (Best)')).toBeTruthy();
  });

  it('paints the gradient from the colormap the backend sampled', () => {
    const css = stopsToGradient(CI_SCALE.stops);
    expect(css).toContain('#a50026');
    expect(css).toContain('#006837');
    // Bottom is the minimum, like the numbers beside it.
    expect(css.startsWith('linear-gradient(to top,')).toBe(true);
  });

  it('shows a unit when the quantity has one', () => {
    render(<ScaleLegend label="PC Δx" scale={{ ...CI_SCALE, unit: 'px' }} />);
    expect(screen.getByText('px')).toBeTruthy();
  });

  it('draws nothing without a usable range — no bar beats a wrong bar', () => {
    const { container } = render(<ScaleLegend label="x" scale={null} />);
    expect(container.firstChild).toBeNull();
    const bad = render(<ScaleLegend label="x" scale={{ min: NaN, max: 1, stops: [] }} />);
    expect(bad.container.firstChild).toBeNull();
  });
});

describe('formatScaleValue', () => {
  it('keeps small numbers readable and large ones short', () => {
    expect(formatScaleValue(0.1802)).toBe('0.180');
    expect(formatScaleValue(1.5)).toBe('1.50');
    expect(formatScaleValue(42.7)).toBe('42.7');
    expect(formatScaleValue(255)).toBe('255');
    expect(formatScaleValue(0)).toBe('0');
  });

  it('falls back to exponent notation where digits would lie', () => {
    expect(formatScaleValue(0.00004)).toMatch(/e-/);
    expect(formatScaleValue(12345)).toMatch(/e\+/);
  });

  it('says nothing rather than NaN', () => {
    expect(formatScaleValue(NaN)).toBe('—');
    expect(formatScaleValue(undefined)).toBe('—');
  });
});
