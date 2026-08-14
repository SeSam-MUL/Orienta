// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import AnnotationToolbar from './AnnotationToolbar';

afterEach(() => cleanup());

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k) => k }),
}));

vi.mock('../../../theme/components', () => ({
  colors: { bg: '#111', border: '#444', text: '#fff', textSecondary: '#aaa' },
  spacing: { innerSpacing: 4 },
  CollapsibleGroup: ({ title, children }) => <div><span>{title}</span>{children}</div>,
}));

const ANNOT = {
  id: 'k1', type: 'colorkey', x: 1.04, y: 0.3, w: 0.4, h: 0.4,
  props: { bgColor: '#ffffff', bgOpacity: 0.9 },
};

function renderWith(annot) {
  return render(
    <AnnotationToolbar
      annotations={[annot]}
      selectedId={annot.id}
      onSelect={() => {}}
      onAdd={() => {}}
      onRemove={() => {}}
      onUpdate={() => {}}
      onClear={() => {}}
    />,
  );
}

describe('AnnotationToolbar — the sliders must be draggable', () => {
  it('keeps the same input element across a value change', () => {
    // The bug: `Row` was defined inside the component, so every render created
    // a new component type and React remounted the row. A remounted <input>
    // loses the pointer capture the browser grants on mousedown, which is why
    // the slider could be clicked but not dragged — the first mouse-move
    // ended the gesture. Element identity is the thing that has to hold.
    const { rerender } = renderWith(ANNOT);
    const before = screen.getByRole('slider');

    rerender(
      <AnnotationToolbar
        annotations={[{ ...ANNOT, props: { ...ANNOT.props, bgOpacity: 0.55 } }]}
        selectedId={ANNOT.id}
        onSelect={() => {}}
        onAdd={() => {}}
        onRemove={() => {}}
        onUpdate={() => {}}
        onClear={() => {}}
      />,
    );
    const after = screen.getByRole('slider');
    expect(after).toBe(before);
    expect(after.value).toBe('0.55');
  });

  it('reports every step of a drag, not just the first', () => {
    const onUpdate = vi.fn();
    render(
      <AnnotationToolbar
        annotations={[ANNOT]}
        selectedId={ANNOT.id}
        onSelect={() => {}}
        onAdd={() => {}}
        onRemove={() => {}}
        onUpdate={onUpdate}
        onClear={() => {}}
      />,
    );
    const slider = screen.getByRole('slider');
    fireEvent.mouseDown(slider);
    ['0.8', '0.6', '0.4'].forEach((v) => fireEvent.change(slider, { target: { value: v } }));
    fireEvent.mouseUp(slider);
    expect(onUpdate.mock.calls.map((c) => c[1].props.bgOpacity)).toEqual([0.8, 0.6, 0.4]);
  });
});
