/**
 * A form built from an add-on's JSON Schema.
 *
 * Nothing else in this app renders a form from data -- every other panel knows
 * its own fields. Here the fields belong to somebody else's code, so the
 * schema is the only description we have, and this file is deliberately a
 * SUBSET of JSON Schema: object with properties, six control types, and
 * nothing else. No $ref, no allOf, no conditionals. What it cannot render it
 * says out loud rather than dropping, because a parameter missing from a form
 * reads as "this add-on has no such setting" and the run then starts without
 * it.
 *
 * All third-party text -- titles, descriptions, enum values -- is rendered
 * VERBATIM. It is never passed through i18n: an add-on written in German must
 * read as its author wrote it, and a lookup would fall back to the key or to
 * English and put words in the author's mouth.
 */
import { useEffect, useMemo } from 'react';
import { Input, NumberInput, Select, Checkbox, FormRow, Button, colors }
  from '../../theme/components';

const SUPPORTED = new Set(['integer', 'number', 'boolean', 'string']);

/**
 * The declared defaults, and ONLY those.
 *
 * A property without a `default` is ABSENT from the result -- not null, not
 * undefined. The runner merges the parameters it was given into the
 * provenance trail, so a key invented here would be recorded in a methods
 * paragraph as a fact about a run that nobody chose.
 */
export function defaultsFor(schema) {
  const props = (schema && schema.properties) || {};
  const out = {};
  for (const [name, prop] of Object.entries(props)) {
    if (prop && Object.prototype.hasOwnProperty.call(prop, 'default')) {
      out[name] = prop.default;
    }
  }
  return out;
}

function kindOf(prop) {
  if (!prop || typeof prop !== 'object') return 'unsupported';
  // enum wins over type: a string with an enum is a choice, and offering it as
  // free text invites a value the add-on will refuse.
  if (Array.isArray(prop.enum) && prop.enum.length) return 'enum';
  if (prop.type === 'string' && prop.format === 'path') return 'path';
  return SUPPORTED.has(prop.type) ? prop.type : 'unsupported';
}

export function unsupportedIn(schema) {
  const props = (schema && schema.properties) || {};
  return Object.entries(props)
    .filter(([, prop]) => kindOf(prop) === 'unsupported')
    .map(([name]) => name);
}

export default function SchemaForm({
  schema, value = {}, onChange, onUnsupported, disabled = false,
  onBrowse,
}) {
  const props = useMemo(
    () => Object.entries((schema && schema.properties) || {}), [schema]);

  // On the SCHEMA, not on every render. The caller keeps this in state to
  // decide whether a run may start, and a fresh array each render would set
  // state on every render -- a loop, not a report.
  useEffect(() => {
    if (onUnsupported) onUnsupported(unsupportedIn(schema));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schema]);

  const set = (name, next) => {
    const merged = { ...value };
    if (next === undefined) delete merged[name];
    else merged[name] = next;
    if (onChange) onChange(merged);
  };

  // An emptied number box is ABSENT, not 0 and not NaN. Number('') is 0, and
  // a zero written into a run is a value the user never chose; NaN cannot be
  // carried as JSON at all. Absent means "not given", which is exactly what
  // the box now says.
  const setNumber = (name, raw) => set(name, raw === '' ? undefined : Number(raw));

  return (
    <div>
      {props.map(([name, prop]) => {
        const kind = kindOf(prop);
        const label = prop.title || name;
        const current = value[name];
        const common = { disabled, 'aria-label': label };
        let control;

        if (kind === 'unsupported') {
          control = (
            <span style={{ color: colors.textSecondary, fontSize: '9pt' }}>
              {/* NAMED. A parameter that merely vanished would be read as a
                  setting the add-on does not have. */}
              {`${name}: this parameter type (${prop && prop.type
                ? prop.type : 'unknown'}) cannot be edited here`}
            </span>
          );
        } else if (kind === 'enum') {
          control = (
            <Select
              {...common}
              value={current === undefined ? '' : String(current)}
              options={prop.enum.map((v) => String(v))}
              onChange={(e) => set(name, e.target.value)}
            />
          );
        } else if (kind === 'boolean') {
          control = (
            <Checkbox
              {...common}
              checked={current === true}
              onChange={(e) => set(name, e.target.checked)}
            />
          );
        } else if (kind === 'integer' || kind === 'number') {
          control = (
            <NumberInput
              {...common}
              value={current === undefined ? '' : current}
              min={prop.minimum}
              max={prop.maximum}
              // An integer steps by one because a component COUNT of 2.5 is
              // not a value the add-on can be given; a number is free.
              step={kind === 'integer' ? 1 : 'any'}
              onChange={(e) => setNumber(name, e.target.value)}
            />
          );
        } else if (kind === 'path') {
          control = (
            <div style={{ display: 'flex', gap: 6 }}>
              <Input
                {...common}
                value={current === undefined ? '' : current}
                onChange={(e) => set(name, e.target.value)}
              />
              <Button small disabled={disabled}
                      onClick={() => onBrowse && onBrowse(name)}>
                Browse…
              </Button>
            </div>
          );
        } else {
          control = (
            <Input
              {...common}
              value={current === undefined ? '' : current}
              onChange={(e) => set(name, e.target.value)}
            />
          );
        }

        return (
          <div key={name}>
            <FormRow label={label}>{control}</FormRow>
            {prop && prop.description && (
              <div style={{
                color: colors.textSecondary,
                fontSize: '9pt',
                margin: '-6px 0 8px 88px',
              }}>
                {prop.description}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
