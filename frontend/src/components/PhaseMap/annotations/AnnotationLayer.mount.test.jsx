// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { useRef } from 'react';
import AnnotationLayer from './AnnotationLayer';

// The stand-in calls hooks of its own on purpose. react-i18next's real
// useTranslation does too, and that matters here: React only rejects a growing
// hook list when the render before it had a non-empty one. A hookless mock
// hides exactly the bug this file is about.
vi.mock('react-i18next', async () => {
  const React = await import('react');
  return {
    useTranslation: () => {
      React.useRef(null);
      React.useState(0);
      return { t: (k) => k };
    },
  };
});

// The host starts unmeasured — that is the normal first paint when the layer
// mounts together with its container, as it does inside the export dialog.
let hostSize = { width: 0, height: 0 };

function Host() {
  const ref = useRef(null);
  return (
    <div ref={ref} data-host style={{ position: 'relative' }}>
      <AnnotationLayer
        annotations={[{ id: 'a1', type: 'title', x: 0.1, y: 0.1, w: 0.4, h: 0.1, props: {} }]}
        selectedId={null}
        onSelect={() => {}}
        onDelete={() => {}}
        onUpdate={() => {}}
        containerRef={ref}
        ctx={{}}
      />
    </div>
  );
}

describe('AnnotationLayer — mounting before the container has measured', () => {
  let errors;
  let spy;

  beforeEach(() => {
    hostSize = { width: 0, height: 0 };
    Element.prototype.getBoundingClientRect = function () {
      return this.hasAttribute('data-host')
        ? { ...hostSize, top: 0, left: 0, right: hostSize.width, bottom: hostSize.height, x: 0, y: 0 }
        : { width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0, x: 0, y: 0 };
    };
    errors = [];
    spy = vi.spyOn(console, 'error').mockImplementation((...a) => { errors.push(String(a[0])); });
  });
  afterEach(() => { spy.mockRestore(); });

  it('survives the container growing from zero to measured', () => {
    // Pre-fix this threw "Rendered more hooks than during the previous render":
    // the widget bailed out above its drag hooks while the box was still 0 wide,
    // then called them all on the paint after the measurement arrived.
    render(<Host />);
    expect(document.querySelectorAll('[data-annotation-widget]')).toHaveLength(0);

    act(() => {
      hostSize = { width: 640, height: 480 };
      window.dispatchEvent(new Event('resize'));
    });

    expect(errors.filter((e) => /hooks/i.test(e))).toEqual([]);
    expect(document.querySelectorAll('[data-annotation-widget]')).toHaveLength(1);
  });
});
