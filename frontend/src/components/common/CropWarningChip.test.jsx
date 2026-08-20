// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import CropWarningChip, { isCropWarning, cropWarningFor } from './CropWarningChip';
import LayerStackPanel from '../PhaseMap/LayerStackPanel';

// This project does not run vitest with `globals: true`, so testing-library
// never registers its own afterEach — without this, every render in the file
// piles up in one document and `getByRole('status')` sees them all.
afterEach(cleanup);

const REFUSED = { cropped: false, exact: false, reason: 'no step size' };
const CLAMPED = { cropped: true, exact: false, reason: 'clamped at the edge' };
const CLEAN = { cropped: true, exact: true, reason: null };

describe('isCropWarning', () => {
  it('is true for a refusal AND for a clamped cut-out', () => {
    // Both mis-place the image once LayeredCanvas stretches it to the map.
    expect(isCropWarning(REFUSED)).toBe(true);
    expect(isCropWarning(CLAMPED)).toBe(true);
  });

  it('is false for a faithful cut-out, and for no verdict at all', () => {
    expect(isCropWarning(CLEAN)).toBe(false);
    expect(isCropWarning(undefined)).toBe(false);
    expect(isCropWarning(null)).toBe(false);
  });
});

describe('cropWarningFor', () => {
  const statuses = new Map([['se:A', REFUSED], ['se:B', CLEAN]]);

  it('returns the verdict only when it is worth showing', () => {
    expect(cropWarningFor(statuses, 'se:A')).toBe(REFUSED);
    expect(cropWarningFor(statuses, 'se:B')).toBeNull();
    expect(cropWarningFor(statuses, 'se:missing')).toBeNull();
  });

  it('survives a caller that has no map yet', () => {
    expect(cropWarningFor(undefined, 'se:A')).toBeNull();
    expect(cropWarningFor(null, 'se:A')).toBeNull();
  });
});

describe('CropWarningChip', () => {
  it('says the image was not cropped at all, when it was refused', () => {
    render(<CropWarningChip status={REFUSED} />);
    const el = screen.getByRole('status');
    expect(el.textContent).toMatch(/could not follow the crop/i);
    // The backend's precise reason rides along as the tooltip.
    expect(el.getAttribute('title')).toBe('no step size');
  });

  it('says something DIFFERENT when the cut-out was merely clamped', () => {
    render(<CropWarningChip status={CLAMPED} />);
    const el = screen.getByRole('status');
    expect(el.textContent).toMatch(/clipped at its own edge/i);
    expect(el.textContent).not.toMatch(/could not follow the crop/i);
  });

  it('renders nothing for a layer that did follow the crop', () => {
    const { container } = render(<CropWarningChip status={CLEAN} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders nothing when there is no verdict at all', () => {
    const { container } = render(<CropWarningChip status={undefined} />);
    expect(container.innerHTML).toBe('');
  });
});

describe('the warning reaches a layer row', () => {
  // The real LayerStackPanel, the real chip, and the same two-line expression
  // both pages use for `renderLayerExtras`. What this does NOT cover is the
  // literal presence of those two lines inside EDSPage / PhaseMapPage — that
  // would need the whole page in jsdom.
  const LAYERS = [
    { id: 'se:A', label: 'SE A', visible: true, opacity: 1, blend: 'normal', key: 'a' },
    { id: 'se:B', label: 'SE B', visible: true, opacity: 1, blend: 'normal', key: 'b' },
  ];
  const statuses = new Map([['se:A', REFUSED], ['se:B', CLEAN]]);
  const noop = () => {};

  function renderPanel() {
    return render(
      <LayerStackPanel
        layers={LAYERS}
        bitmaps={new Map()}
        errors={new Map()}
        onSetOpacity={noop} onSetBlend={noop} onSetVisibility={noop}
        onRemove={noop} onReorder={noop} onAdd={noop} onUsePreset={noop}
        onSetSingleLayer={noop} onRetype={noop}
        renderLayerExtras={(layer) => {
          const cropWarn = cropWarningFor(statuses, layer.id);
          if (cropWarn) return <CropWarningChip status={cropWarn} />;
          return null;
        }}
      />,
    );
  }

  it('shows exactly one warning, on the layer that earned it', () => {
    renderPanel();
    const warnings = screen.getAllByRole('status');
    expect(warnings).toHaveLength(1);
    expect(warnings[0].getAttribute('title')).toBe('no step size');
    // …and it sits inside the offending layer's row, not loose in the panel.
    expect(warnings[0].closest('div').textContent).toMatch(/could not follow the crop/i);
  });
});
