// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import LayerStackPanel from './LayerStackPanel';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    accent: '#a0f', cyan: '#0ff', red: '#f00', green: '#0f0',
    text: '#fff', textSecondary: '#aaa',
  },
  spacing: { innerSpacing: 4 },
  CollapsibleGroup: ({ title, children }) => (
    <div><span>{title}</span>{children}</div>
  ),
  Select: ({ value, onChange, options = [], ...rest }) => (
    <select value={value} onChange={onChange} {...rest}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  ),
}));

vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ diagnosticsComputed: {}, refinementComputed: {} }),
}));
vi.mock('../../stores/useResultStore', () => ({
  default: (sel) => sel({ indexingResult: { result_id: 'r1' } }),
}));

// Pinned: the sentence an operator reads to know what the colours mean.
const LEGEND_EN =
  'How each pixel got its phase: cyan = the grain step decided this grain’s '
  + 'phase and wrote it here — it may be the phase the pixel already had, so '
  + 'cyan is not a count of what changed; orange = left alone, because the '
  + 'chemistry could not decide for the grain (interior too small to measure, '
  + 'composition uneven, or several phases fitting equally well); '
  + 'uncoloured = as indexing left it.';

function layer(id, label) {
  return { id, label, opacity: 1, blend: 'normal', visible: true };
}

const noop = () => {};
const baseProps = {
  bitmaps: new Map([['assignment-source', {}], ['bc', {}]]),
  errors: new Map(),
  onSetOpacity: noop, onSetBlend: noop, onSetVisibility: noop,
  onRemove: noop, onReorder: noop, onAdd: noop, onUsePreset: noop,
  onSetSingleLayer: noop,
  availableToAdd: [],
};

function rowOf(labelText) {
  return screen.getByText(labelText).closest('div[title]');
}

describe('LayerStackPanel — assignment-source provenance row', () => {
  it('shows the translated layer name', () => {
    render(<LayerStackPanel {...baseProps}
      layers={[layer('assignment-source', 'Assignment Source')]} />);
    expect(screen.getByText('Assignment Source')).toBeTruthy();
  });

  it('carries the legend, so the colours are readable without guessing', () => {
    render(<LayerStackPanel {...baseProps}
      layers={[layer('assignment-source', 'Assignment Source')]} />);
    expect(rowOf('Assignment Source').getAttribute('title'))
      .toBe(`Assignment Source — ${LEGEND_EN}`);
  });

  it('leaves a layer without a legend exactly as it was', () => {
    // Control that must survive: only definitions carrying a tipKey gain the
    // trailing legend; every other row keeps its bare label as the title.
    render(<LayerStackPanel {...baseProps} layers={[layer('bc', 'Band Contrast')]} />);
    expect(rowOf('Band Contrast').getAttribute('title')).toBe('Band Contrast');
  });

  it('says it is still loading rather than showing the legend early', () => {
    // A legend under a blank canvas invites reading a half-drawn map; the
    // loading title is the honest one until the bitmap is there.
    render(<LayerStackPanel {...baseProps} bitmaps={new Map()}
      layers={[layer('assignment-source', 'Assignment Source')]} />);
    const title = rowOf('Assignment Source').getAttribute('title');
    expect(title).toBe('Assignment Source (loading...)');
    expect(title).not.toContain('cyan');
  });

  it('a derived row follows the backend answer, not this session', () => {
    // The store mock says nothing was computed here. With the page's answer
    // in hand the row must not paint itself red anyway — that is what made a
    // perfectly drawable refinement layer look broken after a page reload.
    render(<LayerStackPanel {...baseProps}
      bitmaps={new Map([['pc-delta-y', {}]])}
      layers={[layer('pc-delta-y', 'PC Δy (px)')]}
      derivedReady={{ diagnostics: true, refinement: true }} />);
    expect(rowOf('PC Δy (px)').getAttribute('title')).toBe('PC Δy (px)');
  });

  it('and still says "refine first" when the backend says it is not there', () => {
    render(<LayerStackPanel {...baseProps}
      bitmaps={new Map([['pc-delta-y', {}]])}
      layers={[layer('pc-delta-y', 'PC Δy (px)')]}
      derivedReady={{ diagnostics: true, refinement: false }} />);
    // A gated row appends the requirement to its own name, so query the title.
    expect(screen.getByTitle("PC Δy (px) — Run 'Refine R + PC' first")).toBeTruthy();
  });

  it('without an answer the old store reading stands', () => {
    // Other pages pass no derivedReady; their behaviour must not shift.
    render(<LayerStackPanel {...baseProps}
      bitmaps={new Map([['forward-ncc', {}]])}
      layers={[layer('forward-ncc', 'Forward NCC')]} />);
    expect(screen.getByTitle("Forward NCC — Run 'Compute Diagnostics' first")).toBeTruthy();
  });

  it('an error still wins over the legend', () => {
    // A failed fetch is what the user needs to read first -- e.g. the 400 the
    // route returns for a result that did not come from the grain step.
    render(<LayerStackPanel {...baseProps}
      layers={[layer('assignment-source', 'Assignment Source')]}
      errors={new Map([['assignment-source', 'no assignment provenance']])} />);
    expect(rowOf('Assignment Source').getAttribute('title'))
      .toBe('Assignment Source: no assignment provenance');
  });
});
