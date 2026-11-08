// @vitest-environment jsdom
// A hexagonal result has no local-anomaly numbers: the browser answers `local_anomaly: null` for
// every item. The gallery row shows a dash there and still shows the other two metrics; with a
// number it shows the number to three decimals.
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';

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

const hexItem = (localAnomaly) => ({
  rank: 1, row: 4, col: 7, phase_name: 'Al',
  ncc: 0.5, local_anomaly: localAnomaly, pc_sensitivity: 0.25, pattern_residual: 0.75,
});

// `visible={false}`: the row does not ask for thumbnails, only the text is under test.
const renderItem = (localAnomaly) => render(
  <GalleryItem item={hexItem(localAnomaly)} resultId="r1" visible={false} />,
);

describe('GalleryItem local anomaly', () => {
  it('shows a dash, and the other two metrics, where the item has no local anomaly', () => {
    expect(() => renderItem(null)).not.toThrow();
    // hand-written: anom {{anom}} · pcs {{pcs}} · res {{res}} with anom '–', pcs 0.25, res 0.75
    expect(screen.getByText('anom – · pcs 0.250 · res 0.750')).toBeTruthy();
  });

  it('shows the local anomaly to three decimals when there is one', () => {
    renderItem(0.123);
    expect(screen.getByText('anom 0.123 · pcs 0.250 · res 0.750')).toBeTruthy();
  });
});
