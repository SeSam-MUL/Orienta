// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

const setSpec = vi.fn(() => Promise.resolve());
vi.mock('../../stores/useFrameStore', () => ({
  default: (sel) => sel({
    spec: { rotation: { mode: 'preset', preset: 'identity', axis: [0, 0, 1], angle_deg: 0, euler_deg: [0, 0, 0] },
            plot: { x_direction: 'east', z_into_plane: false, hemisphere: 'upper', projection: 'equal_area' },
            apply_to_export: false },
    frameSig: 'aaa', loaded: true, setSpec,
  }),
}));

import CoordinateSystemPanel from './CoordinateSystemPanel';

beforeEach(() => setSpec.mockClear());
afterEach(() => cleanup());

describe('CoordinateSystemPanel', () => {
  it('renders presets and a helper diagram', () => {
    render(<CoordinateSystemPanel />);
    expect(screen.getByText(/Coordinate System/i)).toBeInTheDocument();
    expect(screen.getByTestId('orientation-helper-svg')).toBeInTheDocument();
  });

  it('selecting a preset calls setSpec', () => {
    render(<CoordinateSystemPanel />);
    fireEvent.change(screen.getByLabelText(/Preset/i), { target: { value: 'aztec_oxford' } });
    expect(setSpec).toHaveBeenCalled();
  });
});
