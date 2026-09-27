/**
 * An add-on's outputs: scalars, tables, and a way to get the numbers OUT.
 *
 * Nothing generic exists for either — this app has twenty bespoke `<table>`
 * blocks and `theme/components.jsx` exports neither a table nor a labelled
 * number. So this is a small honest one for this page, deliberately not a
 * generalisation of the app's tables.
 *
 * THE EXPORT IS THE POINT, not a nicety. A table and a scalar live in one HTTP
 * response and are gone on reload, while the citation text beside them has had
 * a Copy button all along: the page would hand a user the sentence to cite and
 * keep the numbers it is about. Copy writes TSV (what a spreadsheet takes from
 * a clipboard); Save writes CSV through the same Electron path the image
 * export uses.
 *
 * Both carry EVERY row, not the rows on screen. The table is capped for
 * rendering — a hundred thousand rows would take the page down — and a file
 * that silently stopped where the display stops would be the worst kind of
 * wrong: complete-looking and short.
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, colors } from '../../theme/components';

//: Rows rendered before the table says it is showing a part. Not a limit on
//: what is exported.
export const ROW_CAP = 200;

/** A cell as text: never the word "null", never "undefined". */
export function cellText(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number' && !Number.isFinite(value)) return '—';
  return String(value);
}

/**
 * Rows and columns as delimited text, for a clipboard or a file.
 *
 * A cell is quoted only when it has to be, and an embedded quote is doubled —
 * the one rule every spreadsheet agrees on. A newline inside a cell would
 * otherwise end the row, which is how a 200-row file becomes a 201-row one
 * that nobody notices.
 *
 * `—` is for the SCREEN. An empty cell in a file is empty: a spreadsheet
 * reading an em dash gets text where a column of numbers was.
 */
export function toDelimited(columns, rows, sep) {
  const cell = (v) => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    if (sep === ',' && /[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    if (sep === '\t' && /[\t\n\r]/.test(s)) return s.replace(/[\t\n\r]+/g, ' ');
    return s;
  };
  const head = columns.map(cell).join(sep);
  const body = rows.map((r) => r.map(cell).join(sep));
  return [head, ...body].join('\n');
}

function ScalarRow({ output }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'baseline',
                  fontSize: '10pt', marginBottom: 4 }}>
      <span style={{ color: colors.textSecondary, minWidth: 160 }}>
        {output.label || output.key}
      </span>
      <strong>{cellText(output.value)}</strong>
      {/* The unit is not decoration: this contract keeps units in field names
          and an output's `unit`, and a number shown without one is a number
          somebody will write into a paper with the wrong one. */}
      {output.unit && (
        <span style={{ color: colors.textSecondary }}>{output.unit}</span>
      )}
    </div>
  );
}

function TableOutput({ output, onExport }) {
  const { t } = useTranslation('addons');
  const rows = output.rows || [];
  const columns = output.columns || [];
  const shown = rows.slice(0, ROW_CAP);

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <strong style={{ fontSize: '10pt' }}>{output.label || output.key}</strong>
        <span style={{ flex: 1 }} />
        <Button small onClick={() => onExport('copy', output)}>
          {t('outputs.copy')}
        </Button>
        <Button small onClick={() => onExport('save', output)}>
          {t('outputs.save')}
        </Button>
      </div>

      {rows.length === 0 ? (
        // An empty table and a broken renderer look identical, so it says
        // which it is.
        <div style={{ color: colors.textSecondary, fontSize: '9pt' }}>
          {t('outputs.emptyTable')}
        </div>
      ) : (
        <>
          <table style={{ borderCollapse: 'collapse', fontSize: '9pt',
                          marginTop: 4 }}>
            <thead>
              <tr>
                {columns.map((c) => (
                  <th key={c} style={{ textAlign: 'left', padding: '2px 8px',
                                       borderBottom: `1px solid ${colors.border}`,
                                       color: colors.textSecondary }}>
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => (
                    <td key={j} style={{ padding: '2px 8px' }}>{cellText(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > shown.length && (
            // Said, not implied. A table that just stops looks like the data
            // stop, and the export below carries the rest.
            <div style={{ color: colors.textSecondary, fontSize: '9pt' }}>
              {t('outputs.capped', { shown: shown.length, total: rows.length })}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function OutputPanels({ outputs = [], onExport, mapSlot }) {
  const { t } = useTranslation('addons');
  const scalars = outputs.filter((o) => o.kind === 'scalar');
  const tables = outputs.filter((o) => o.kind === 'table');
  const maps = outputs.filter((o) => o.kind === 'map');

  if (!outputs.length) return null;

  return (
    <div data-testid="addon-outputs" style={{ marginTop: 12 }}>
      {scalars.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          {scalars.map((o) => <ScalarRow key={o.key} output={o} />)}
        </div>
      )}
      {tables.map((o) => (
        <TableOutput key={o.key} output={o} onExport={onExport} />
      ))}
      {maps.length > 0 && (
        <div>
          <div style={{ color: colors.textSecondary, fontSize: '9pt' }}>
            {t('outputs.maps')}
          </div>
          {/* Filled by Task 10; the panel lists the maps either way so an
              add-on that produced one is not silently missing an output. */}
          {maps.map((o) => (
            <div key={o.key} style={{ fontSize: '10pt' }}>
              {mapSlot ? mapSlot(o) : (o.label || o.key)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export { ScalarRow, TableOutput };
