// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, within }
  from '@testing-library/react';
import OutputPanels, { toDelimited, cellText, ROW_CAP } from './OutputPanels';

afterEach(cleanup);

const TABLE = {
  kind: 'table', key: 'components', label: 'Components',
  columns: ['component', 'weight', 'mean'],
  rows: [[1, 0.5, 120.4], [2, 0.3, 88.1], [3, 0.2, null]],
};

const SCALAR = { kind: 'scalar', key: 'n_px', label: 'Pixels used',
                 value: 10800, unit: 'px' };

describe('cellText', () => {
  it('renders an absent value as a dash, never as the word null', () => {
    // "null" in a results table reads as a value the add-on computed.
    expect(cellText(null)).toBe('—');
    expect(cellText(undefined)).toBe('—');
    expect(cellText(NaN)).toBe('—');
    expect(cellText(0)).toBe('0');        // and 0 is a number, not absence
    expect(cellText('')).toBe('');
  });
});

describe('toDelimited', () => {
  it('puts the column names on the first line', () => {
    const out = toDelimited(TABLE.columns, TABLE.rows, '\t');
    expect(out.split('\n')[0]).toBe('component\tweight\tmean');
  });

  it('writes one line per row, plus the header', () => {
    const out = toDelimited(TABLE.columns, TABLE.rows, '\t');
    expect(out.split('\n')).toHaveLength(TABLE.rows.length + 1);
  });

  it('writes an ABSENT cell as empty, not as the screen’s dash', () => {
    // A spreadsheet reading "—" gets text where a column of numbers was.
    const out = toDelimited(['a'], [[null]], ',');
    expect(out).toBe('a\n');
  });

  it('quotes a CSV cell that contains a comma, and doubles a quote', () => {
    const out = toDelimited(['a'], [['x,y'], ['say "hi"']], ',');
    expect(out.split('\n')[1]).toBe('"x,y"');
    expect(out.split('\n')[2]).toBe('"say ""hi"""');
  });

  it('a newline inside a cell cannot end the row', () => {
    // Otherwise a 3-row file silently becomes a 4-row one.
    const csv = toDelimited(['a'], [['line1\nline2']], ',');
    expect(csv.split('\n')).toHaveLength(2 + 1);   // header + quoted 2-line cell
    const tsv = toDelimited(['a'], [['line1\nline2']], '\t');
    expect(tsv.split('\n')).toHaveLength(2);       // header + one row
  });
});

describe('what the panel shows', () => {
  it('renders a table’s columns and rows in order', () => {
    render(<OutputPanels outputs={[TABLE]} onExport={vi.fn()} />);
    const headers = screen.getAllByRole('columnheader').map((h) => h.textContent);
    expect(headers).toEqual(['component', 'weight', 'mean']);
    const firstRow = screen.getAllByRole('row')[1];
    expect(within(firstRow).getAllByRole('cell').map((c) => c.textContent))
      .toEqual(['1', '0.5', '120.4']);
  });

  it('shows a dash for a null cell', () => {
    render(<OutputPanels outputs={[TABLE]} onExport={vi.fn()} />);
    const lastRow = screen.getAllByRole('row').at(-1);
    expect(within(lastRow).getAllByRole('cell').at(-1).textContent).toBe('—');
  });

  it('shows a scalar’s label, value AND unit', () => {
    // A number shown without its unit is one somebody writes into a paper
    // with the wrong one.
    render(<OutputPanels outputs={[SCALAR]} onExport={vi.fn()} />);
    const panel = screen.getByTestId('addon-outputs');
    expect(panel.textContent).toContain('Pixels used');
    expect(panel.textContent).toContain('10800');
    expect(panel.textContent).toContain('px');
  });

  it('says an empty table is empty', () => {
    // An empty table and a broken renderer look identical otherwise.
    render(<OutputPanels outputs={[{ ...TABLE, rows: [] }]} onExport={vi.fn()} />);
    expect(screen.getByText(/no rows/i)).toBeTruthy();
    expect(screen.queryAllByRole('row')).toHaveLength(0);
  });

  it('caps a long table AND says that it is capped', () => {
    const rows = Array.from({ length: ROW_CAP + 37 }, (_, i) => [i, 0, 0]);
    render(<OutputPanels outputs={[{ ...TABLE, rows }]} onExport={vi.fn()} />);
    expect(screen.getAllByRole('row')).toHaveLength(ROW_CAP + 1);   // + header
    expect(screen.getByText(new RegExp(String(rows.length)))).toBeTruthy();
  });

  it('renders nothing at all when there are no outputs', () => {
    const { container } = render(<OutputPanels outputs={[]} onExport={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it('lists a map output even before anything can draw it', () => {
    // An add-on that produced a map must not look like one that produced
    // nothing.
    render(<OutputPanels outputs={[{ kind: 'map', key: 'component_map',
                                     label: 'Component map' }]}
                         onExport={vi.fn()} />);
    expect(screen.getByText('Component map')).toBeTruthy();
  });
});

describe('getting the numbers out', () => {
  it('copy and save each ask for their own table', () => {
    const onExport = vi.fn();
    render(<OutputPanels outputs={[TABLE]} onExport={onExport} />);
    fireEvent.click(screen.getByRole('button', { name: /copy/i }));
    expect(onExport).toHaveBeenCalledWith('copy', TABLE);
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    expect(onExport).toHaveBeenCalledWith('save', TABLE);
  });

  it('the exported text carries EVERY row, not the rows on screen', () => {
    // The display is capped; a file that stopped where the display stops
    // would be complete-looking and short — the worst kind of wrong.
    const rows = Array.from({ length: ROW_CAP + 50 }, (_, i) => [i, 0, 0]);
    const text = toDelimited(TABLE.columns, rows, '\t');
    expect(text.split('\n')).toHaveLength(rows.length + 1);
  });
});
