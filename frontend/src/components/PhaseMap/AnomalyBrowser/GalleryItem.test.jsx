// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  forwardDiagApi: { thumbnail: vi.fn(() => Promise.resolve(null)) },
}));

import GalleryItem from './GalleryItem';

afterEach(() => cleanup());

const item = {
  rank: 1, row: 3, col: 4, phase_name: 'Al',
  ncc: 0.1234, local_anomaly: 0.5, pc_sensitivity: 0.25, pattern_residual: 0.75,
};

describe('GalleryItem', () => {
  it('formats the metrics of an ordinary item to three decimals', () => {
    const { container } = render(
      <GalleryItem item={item} resultId="r1" visible={false} />);
    expect(container.textContent).toContain('0.123');
    expect(container.textContent).toContain('0.500');
    expect(container.textContent).toContain('0.250');
    expect(container.textContent).toContain('0.750');
  });

  it('renders an item whose metrics are null (the route turns NaN into null) without throwing', () => {
    const { container } = render(
      <GalleryItem item={{ ...item, ncc: null, local_anomaly: null, pc_sensitivity: null,
        pattern_residual: undefined }} resultId="r1" visible={false} />);
    expect(container.textContent).toContain('Al');
    expect(container.textContent).not.toMatch(/null|undefined|NaN/);
  });
});
