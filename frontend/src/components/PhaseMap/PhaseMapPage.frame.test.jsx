// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, screen, cleanup } from '@testing-library/react';

// A full PhaseMapPage mount in jsdom is brittle (many providers/canvas).
// Per the Task-17 brief, this check verifies the shared CoordinateSystemPanel
// renders its titled group; the real wiring into PhaseMapPage is validated
// by the Task-18 manual E2E. Mock useFrameStore with a complete spec so the
// panel renders its body (matches the proven CoordinateSystemPanel.test shape).
vi.mock('./openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }), { virtual: true });
vi.mock('../../stores/useFrameStore', () => ({
  default: (sel) => sel({
    loadForFile: vi.fn(),
    frameSig: 'aaa',
    loaded: true,
    setSpec: vi.fn(),
    spec: {
      rotation: { mode: 'preset', preset: 'identity', axis: [0, 0, 1], angle_deg: 0, euler_deg: [0, 0, 0] },
      plot: { x_direction: 'east', z_into_plane: false, hemisphere: 'upper', projection: 'equal_area' },
      apply_to_export: false,
    },
  }),
}));

import CoordinateSystemPanel from '../common/CoordinateSystemPanel';

afterEach(() => cleanup());

describe('PhaseMap coordinate-system integration', () => {
  it('CoordinateSystemPanel is importable and renders a titled group', () => {
    render(<CoordinateSystemPanel />);
    expect(screen.getByText(/Coordinate System/i)).toBeInTheDocument();
  });
});
