/**
 * The `⋯` menu on a group row.
 *
 * A REAL MENU AND NOT A `<select>` (spec §2.5.1). A select is a list of
 * values you pick one of; these are actions, and three of them are
 * destructive. A select also swallows the keyboard on Windows -- arrow keys
 * change the value as you move, so a keyboard user "visits" Delete on the
 * way past, and on some builds that commits it.
 *
 * So: a button that opens a `role="menu"`, arrow keys move between items,
 * Home and End jump, Escape closes and puts focus back on the button that
 * opened it, and clicking elsewhere closes it. An item that cannot be used
 * right now is disabled AND says why in its own label, rather than being
 * hidden -- a menu whose contents change between visits is a menu people
 * stop trusting.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { colors, spacing } from '../../theme/components';

const S = {
  wrap: { position: 'relative', display: 'inline-flex' },
  button: {
    background: 'transparent', border: 'none', color: colors.textSecondary,
    cursor: 'pointer', font: 'inherit', padding: '0 4px', lineHeight: 1,
  },
  menu: {
    position: 'absolute', top: '100%', right: 0, zIndex: 40, minWidth: 190,
    margin: 0, padding: 4, listStyle: 'none',
    background: colors.bgSecondary, color: colors.text,
    border: `1px solid ${colors.border}`, borderRadius: 3,
    boxShadow: '0 4px 14px rgba(0,0,0,0.35)',
  },
  item: {
    display: 'block', width: '100%', textAlign: 'left',
    background: 'transparent', border: 'none', color: colors.text,
    font: 'inherit', fontSize: '9pt', padding: `4px ${spacing.innerSpacing}px`,
    cursor: 'pointer', borderRadius: 2,
  },
  itemOff: { color: colors.textSecondary, cursor: 'default' },
  danger: { color: colors.red || '#ff6b6b' },
};

/**
 * @param items {label, onSelect, disabled?, danger?, key}[]
 */
export default function RowMenu({ label, items }) {
  const [open, setOpen] = useState(false);
  const [at, setAt] = useState(0);
  const wrapRef = useRef(null);
  const buttonRef = useRef(null);
  const itemRefs = useRef([]);
  const menuId = useId();

  const close = useCallback((refocus = true) => {
    setOpen(false);
    if (refocus && buttonRef.current) buttonRef.current.focus();
  }, []);

  // Clicking anywhere else closes it. `mousedown` and not `click`, so the
  // menu is gone before the thing under the pointer reacts -- otherwise a
  // click meant to dismiss also hits whatever is behind.
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const el = itemRefs.current[at];
    if (el) el.focus();
  }, [open, at]);

  const usable = items.map((it, i) => (it.disabled ? -1 : i)).filter((i) => i >= 0);

  const step = (dir) => {
    if (!usable.length) return;
    const here = usable.indexOf(at);
    const next = here === -1
      ? usable[0]
      : usable[(here + dir + usable.length) % usable.length];
    setAt(next);
  };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); step(1); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); step(-1); return; }
    if (e.key === 'Home') { e.preventDefault(); if (usable.length) setAt(usable[0]); return; }
    if (e.key === 'End') {
      e.preventDefault();
      if (usable.length) setAt(usable[usable.length - 1]);
    }
  };

  const openWith = (first) => {
    setAt(first === 'last' && usable.length ? usable[usable.length - 1]
      : (usable[0] ?? 0));
    setOpen(true);
  };

  return (
    <span style={S.wrap} ref={wrapRef}>
      <button
        type="button"
        ref={buttonRef}
        style={S.button}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={label}
        onClick={() => (open ? close(false) : openWith('first'))}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') { e.preventDefault(); openWith('first'); }
          if (e.key === 'ArrowUp') { e.preventDefault(); openWith('last'); }
        }}
      >
        ⋯
      </button>
      {open && (
        <ul style={S.menu} role="menu" id={menuId} aria-label={label}
            onKeyDown={onKeyDown}>
          {items.map((it, i) => (
            <li key={it.key} role="none">
              <button
                type="button"
                role="menuitem"
                ref={(el) => { itemRefs.current[i] = el; }}
                tabIndex={i === at ? 0 : -1}
                disabled={Boolean(it.disabled)}
                aria-disabled={it.disabled ? 'true' : undefined}
                style={{
                  ...S.item,
                  ...(it.disabled ? S.itemOff : null),
                  ...(it.danger && !it.disabled ? S.danger : null),
                }}
                onClick={() => { close(); it.onSelect(); }}
              >
                {it.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </span>
  );
}
