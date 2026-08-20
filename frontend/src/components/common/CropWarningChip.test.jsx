// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import CropWarningChip from './CropWarningChip';
import '../../i18n';

describe('CropWarningChip', () => {
  it('says so, visibly, when a layer could not follow the crop', () => {
    render(<CropWarningChip status={{ cropped: false, reason: 'no step size' }} />);
    const el = screen.getByRole('status');
    expect(el.textContent).toMatch(/could not follow the crop/i);
    // The backend's precise reason rides along as the tooltip.
    expect(el.getAttribute('title')).toBe('no step size');
  });

  it('renders nothing for a layer that did follow the crop', () => {
    const { container } = render(<CropWarningChip status={{ cropped: true, reason: null }} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders nothing when there is no verdict at all', () => {
    const { container } = render(<CropWarningChip status={undefined} />);
    expect(container.innerHTML).toBe('');
  });
});
