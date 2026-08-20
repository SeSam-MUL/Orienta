/**
 * Tests for CropPanel — the strip under the overview that turns a drawn
 * selection into a dataset.
 *
 * The panel owns no selection state, so every case here is "given this bbox
 * and mask, what does it say and is the button honest about it":
 *   1. Reports the pixel count and the memory the crop will take.
 *   2. A masked selection reports the SELECTED count, not the box.
 *   3. Clicking hands the window (and the mask) to onCrop.
 *   4. Nothing drawn  → button disabled, panel says what to do.
 *   5. Mask selects nothing → button disabled.
 *   6. A non-finite bbox (NaN grid coordinates, which calcOverviewPos can
 *      produce when the overview container measures 0 wide) is treated as
 *      "nothing drawn" — NOT as a valid selection of size 0 B.
 *   7. Busy → button disabled and says so.
 *   8. Save-crop button: only for a crop, and only when a handler exists.
 */
// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import CropPanel from '../CropPanel';

afterEach(() => cleanup());

const BBOX = { row0: 4, col0: 6, rows: 8, cols: 10 };

describe('CropPanel', () => {
  it('says how many pixels and how much memory the crop will be', () => {
    render(<CropPanel bbox={BBOX} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} />);
    expect(screen.getByTestId('crop-summary').textContent).toContain('80');
    expect(screen.getByTestId('crop-summary').textContent).toContain('MB');
  });

  it('reports the selected count, not the box, for a masked selection', () => {
    const mask = new Uint8Array(80);
    mask.fill(1, 0, 17);
    render(<CropPanel bbox={BBOX} mask={mask} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} />);
    expect(screen.getByTestId('crop-summary').textContent).toContain('17');
  });

  it('hands the window and the mask to onCrop', () => {
    const onCrop = vi.fn();
    render(<CropPanel bbox={BBOX} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} shape="rect" onCrop={onCrop} />);
    fireEvent.click(screen.getByTestId('crop-apply'));
    expect(onCrop).toHaveBeenCalledWith(
      expect.objectContaining({ shape: 'rect', row0: 4, col0: 6, rows: 8, cols: 10 }),
    );
    expect(onCrop.mock.calls[0][0].mask).toBeNull();
  });

  it('sends the mask as a flat boolean array', () => {
    const onCrop = vi.fn();
    const mask = new Uint8Array(80);
    mask.fill(1, 0, 17);
    render(<CropPanel bbox={BBOX} mask={mask} patternShape={[128, 156]}
                      bytesPerPixel={1} shape="lasso" onCrop={onCrop} />);
    fireEvent.click(screen.getByTestId('crop-apply'));
    const payload = onCrop.mock.calls[0][0];
    expect(payload.shape).toBe('lasso');
    expect(payload.mask).toHaveLength(80);
    expect(payload.mask.filter(Boolean)).toHaveLength(17);
    expect(payload.mask[0]).toBe(true);
    expect(payload.mask[79]).toBe(false);
  });

  it('omits the size rather than claiming 0 B when the detector size is unknown', () => {
    render(<CropPanel bbox={BBOX} mask={null} patternShape={null}
                      bytesPerPixel={1} onCrop={() => {}} />);
    const text = screen.getByTestId('crop-summary').textContent;
    expect(text).toContain('80');
    expect(text).not.toContain('0 B');
    expect(screen.getByTestId('crop-apply')).not.toBeDisabled();
  });

  it('disables the button when nothing is drawn', () => {
    render(<CropPanel bbox={null} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} />);
    expect(screen.getByTestId('crop-apply')).toBeDisabled();
    expect(screen.getByTestId('crop-summary').textContent).toMatch(/selection/i);
  });

  it('disables the button when the mask selects nothing', () => {
    render(<CropPanel bbox={BBOX} mask={new Uint8Array(80)} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} />);
    expect(screen.getByTestId('crop-apply')).toBeDisabled();
  });

  it('treats a non-finite bbox as nothing drawn rather than a 0 B selection', () => {
    // boundsOf({r: NaN}) returns {row0: Infinity, col0: Infinity,
    // rows: NaN, cols: NaN} — truthy, so a null-check alone lets it through.
    const bad = { row0: Infinity, col0: Infinity, rows: NaN, cols: NaN };
    render(<CropPanel bbox={bad} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} />);
    expect(screen.getByTestId('crop-apply')).toBeDisabled();
    expect(screen.getByTestId('crop-summary').textContent).toMatch(/selection/i);
    expect(screen.getByTestId('crop-summary').textContent).not.toContain('0 B');
  });

  it('disables the button and says so while a crop is running', () => {
    render(<CropPanel bbox={BBOX} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} busy onCrop={() => {}} />);
    const button = screen.getByTestId('crop-apply');
    expect(button).toBeDisabled();
    expect(button.textContent).toMatch(/Cropping/i);
  });

  it('shows where a cropped dataset came from', () => {
    render(<CropPanel bbox={null} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}}
                      origin={{ source_file: 'D:/scans/Scan1.h5oina',
                                row0: 4, col0: 6, rows: 8, cols: 10 }} />);
    const text = screen.getByTestId('crop-origin').textContent;
    expect(text).toContain('Scan1.h5oina');
    expect(text).toContain('4');
    expect(text).toContain('11');   // row0 + rows - 1, inclusive
    expect(text).toContain('15');   // col0 + cols - 1, inclusive
  });

  it('shows no origin line for a full scan', () => {
    render(<CropPanel bbox={null} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} origin={null} />);
    expect(screen.queryByTestId('crop-origin')).toBeNull();
  });
  it('offers Save crop only for a cropped dataset', () => {
    const onExport = vi.fn();
    render(<CropPanel bbox={null} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} onExport={onExport}
                      origin={{ source_file: 'D:/scans/Scan1.h5oina',
                                row0: 4, col0: 6, rows: 8, cols: 10 }} />);
    fireEvent.click(screen.getByTestId('crop-export'));
    expect(onExport).toHaveBeenCalled();
  });

  it('shows no Save crop button for a full scan', () => {
    render(<CropPanel bbox={null} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}}
                      onExport={() => {}} origin={null} />);
    expect(screen.queryByTestId('crop-export')).toBeNull();
  });

  it('disables the Save crop button and says so while a save is running', () => {
    render(<CropPanel bbox={null} mask={null} patternShape={[128, 156]}
                      bytesPerPixel={1} onCrop={() => {}} onExport={() => {}}
                      saving
                      origin={{ source_file: 'D:/scans/Scan1.h5oina',
                                row0: 4, col0: 6, rows: 8, cols: 10 }} />);
    const button = screen.getByTestId('crop-export');
    expect(button).toBeDisabled();
    expect(button.textContent).toMatch(/Saving/i);
  });
});
