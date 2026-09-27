// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import SchemaForm, { defaultsFor } from './SchemaForm';

// This project does not run vitest with `globals: true`, so testing-library
// never registers its own afterEach — without this every render in the file
// piles up in one document and the queries below see all of them.
afterEach(cleanup);

const SCHEMA = {
  type: 'object',
  properties: {
    n_components: { type: 'integer', title: 'Components', default: 3,
                    minimum: 1, maximum: 9 },
    tolerance: { type: 'number', title: 'Tolerance', default: 0.5 },
    normalise: { type: 'boolean', title: 'Normalise', default: true },
    method: { type: 'string', title: 'Method', enum: ['full', 'tied'],
              default: 'full' },
    label: { type: 'string', title: 'Label' },
    report_dir: { type: 'string', format: 'path', title: 'Report folder' },
  },
};

function renderForm(overrides = {}) {
  const props = {
    schema: SCHEMA,
    value: defaultsFor(SCHEMA),
    onChange: vi.fn(),
    onUnsupported: vi.fn(),
    ...overrides,
  };
  const utils = render(<SchemaForm {...props} />);
  return { ...utils, props };
}

describe('defaultsFor', () => {
  it('returns exactly the declared defaults', () => {
    expect(defaultsFor(SCHEMA)).toEqual({
      n_components: 3, tolerance: 0.5, normalise: true, method: 'full',
    });
  });

  it('LEAVES OUT a property that declares no default', () => {
    // Not null, not undefined: absent. The runner merges declared defaults
    // into the provenance trail, so a key invented here would be written into
    // a methods paragraph as a fact about a run that nobody chose.
    const out = defaultsFor(SCHEMA);
    expect('label' in out).toBe(false);
    expect('report_dir' in out).toBe(false);
  });

  it('survives a schema with no properties at all', () => {
    expect(defaultsFor({ type: 'object' })).toEqual({});
    expect(defaultsFor(undefined)).toEqual({});
  });
});

describe('SchemaForm controls', () => {
  it('renders one control per declared type', () => {
    renderForm();
    expect(screen.getByLabelText('Components').type).toBe('number');
    expect(screen.getByLabelText('Tolerance').type).toBe('number');
    expect(screen.getByLabelText('Normalise').type).toBe('checkbox');
    expect(screen.getByLabelText('Method').tagName).toBe('SELECT');
    expect(screen.getByLabelText('Label').type).toBe('text');
    expect(screen.getByLabelText('Report folder').type).toBe('text');
  });

  it('an integer steps by one and a number does not', () => {
    // The distinction is the whole reason the two types are separate: a
    // component COUNT of 2.5 is not a value the add-on can be given.
    renderForm();
    expect(screen.getByLabelText('Components').step).toBe('1');
    expect(screen.getByLabelText('Tolerance').step).not.toBe('1');
  });

  it('carries minimum and maximum to the control', () => {
    renderForm();
    const n = screen.getByLabelText('Components');
    expect(n.min).toBe('1');
    expect(n.max).toBe('9');
  });

  it('offers the enum VALUES, not just a select', () => {
    // Asserting "a <select> exists" passes for a control offering the wrong
    // options entirely, which is the shape this test exists to refuse.
    renderForm();
    const values = Array.from(screen.getByLabelText('Method').options)
      .map((o) => o.value);
    expect(values).toEqual(['full', 'tied']);
  });

  it('offers a way to browse for a path parameter', () => {
    renderForm();
    expect(screen.getByRole('button', { name: /browse/i })).toBeTruthy();
  });

  it('pre-fills from the value it is given', () => {
    renderForm();
    expect(screen.getByLabelText('Components').value).toBe('3');
    expect(screen.getByLabelText('Normalise').checked).toBe(true);
    expect(screen.getByLabelText('Method').value).toBe('full');
  });
});

describe('SchemaForm text is the add-on author’s', () => {
  it('renders title and description verbatim, in the author’s language', () => {
    // Third-party text is NOT looked up in our locales: an add-on written in
    // German must read as its author wrote it, and a t() around it would
    // silently fall back to the key or to English.
    const schema = {
      type: 'object',
      properties: {
        schwelle: {
          type: 'number',
          title: 'Schwellenwert',
          description: 'Ab hier gilt ein Pixel als rekristallisiert.',
        },
      },
    };
    render(<SchemaForm schema={schema} value={{}} onChange={vi.fn()}
                       onUnsupported={vi.fn()} />);
    expect(screen.getByText('Schwellenwert')).toBeTruthy();
    expect(screen.getByText('Ab hier gilt ein Pixel als rekristallisiert.'))
      .toBeTruthy();
  });

  it('falls back to the property NAME when there is no title', () => {
    render(<SchemaForm schema={{ type: 'object', properties: { raw_key: { type: 'string' } } }}
                       value={{}} onChange={vi.fn()} onUnsupported={vi.fn()} />);
    expect(screen.getByLabelText('raw_key')).toBeTruthy();
  });
});

describe('SchemaForm reports what it cannot render', () => {
  it('shows a named row for an unsupported type and reports the name', () => {
    const onUnsupported = vi.fn();
    const schema = {
      type: 'object',
      properties: {
        ok: { type: 'string', title: 'Fine' },
        matrix: { type: 'array', title: 'Weights' },
      },
    };
    render(<SchemaForm schema={schema} value={{}} onChange={vi.fn()}
                       onUnsupported={onUnsupported} />);
    // NAMED, not silently dropped: a parameter that vanishes from a form is
    // read as "this add-on has no such setting", and the run would then be
    // started without it.
    expect(screen.getByText(/matrix/)).toBeTruthy();
    expect(onUnsupported).toHaveBeenCalledWith(['matrix']);
  });

  it('a schema with no properties renders empty and reports NOTHING unsupported', () => {
    // Asserted rather than assumed: "renders nothing" is also what a crashed
    // renderer does, and an empty form must not be read as a refusal.
    const onUnsupported = vi.fn();
    const { container } = render(
      <SchemaForm schema={{ type: 'object' }} value={{}} onChange={vi.fn()}
                  onUnsupported={onUnsupported} />);
    expect(container.querySelectorAll('input, select').length).toBe(0);
    expect(onUnsupported).toHaveBeenCalledWith([]);
  });

  it('reports on a schema CHANGE, not on every render', () => {
    // Task 8 puts this into state. A fresh array every render is a loop.
    const onUnsupported = vi.fn();
    const props = { schema: SCHEMA, value: {}, onChange: vi.fn(), onUnsupported };
    const { rerender } = render(<SchemaForm {...props} />);
    rerender(<SchemaForm {...props} value={{ tolerance: 1 }} />);
    rerender(<SchemaForm {...props} value={{ tolerance: 2 }} />);
    expect(onUnsupported).toHaveBeenCalledTimes(1);
  });
});

describe('SchemaForm edits', () => {
  it('hands back the whole next object', () => {
    const { props } = renderForm();
    fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'run A' } });
    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ label: 'run A', n_components: 3 }));
  });

  it('a number field emits a NUMBER, not the string the DOM gave it', () => {
    // JSON Schema says integer; an add-on doing arithmetic on "4" gets "44".
    const { props } = renderForm();
    fireEvent.change(screen.getByLabelText('Components'), { target: { value: '4' } });
    const [next] = props.onChange.mock.calls.at(-1);
    expect(next.n_components).toBe(4);
    expect(typeof next.n_components).toBe('number');
  });

  it('an emptied number field is absent, not NaN and not zero', () => {
    // Number('') is 0, and a 0 written into a run is a value the user never
    // chose; NaN cannot be sent as JSON at all.
    const { props } = renderForm();
    fireEvent.change(screen.getByLabelText('Components'), { target: { value: '' } });
    const [next] = props.onChange.mock.calls.at(-1);
    expect('n_components' in next).toBe(false);
  });

  it('a checkbox emits a boolean', () => {
    const { props } = renderForm();
    fireEvent.click(screen.getByLabelText('Normalise'));
    const [next] = props.onChange.mock.calls.at(-1);
    expect(next.normalise).toBe(false);
  });

  it('every control is disabled together', () => {
    renderForm({ disabled: true });
    expect(screen.getByLabelText('Components').disabled).toBe(true);
    expect(screen.getByLabelText('Normalise').disabled).toBe(true);
    expect(screen.getByLabelText('Method').disabled).toBe(true);
    expect(screen.getByRole('button', { name: /browse/i }).disabled).toBe(true);
  });
});
