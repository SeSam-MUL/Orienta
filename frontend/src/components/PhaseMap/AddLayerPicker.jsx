/**
 * The "+ Add Layer" picker.
 *
 * Two things a dropdown could not do:
 *   - search, because the list is long once EDS elements and electron images
 *     are discovered (30+ entries on a full H5OINA);
 *   - hide what cannot be drawn. A layer whose data nobody loaded only ever
 *     produces an error chip in the stack, so those are out of the way by
 *     default and come back with one checkbox — with the missing piece named,
 *     because "where did Forward NCC go?" is the obvious next question.
 *
 * The list expands in flow rather than floating over the panel: the settings
 * column scrolls, and an absolutely positioned menu would be clipped by it.
 */
import { useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

/** Case- and diacritic-insensitive substring match over group + name. */
export function matchesQuery(option, query) {
  const q = (query || '').trim().toLowerCase();
  if (!q) return true;
  const hay = `${option.group ?? ''} ${option.name ?? ''} ${option.label ?? ''}`.toLowerCase();
  // Every word must appear, so "eds fe" finds "EDS: Fe" without demanding order.
  return q.split(/\s+/).every((word) => hay.includes(word));
}

/**
 * Split options into what the picker shows and what it holds back.
 * `showUnavailable` false keeps gated entries out of `visible` but still
 * counts them, so the checkbox can say how many are hidden.
 */
export function partitionOptions(options, query, showUnavailable) {
  const matched = (options || []).filter((o) => o.value && matchesQuery(o, query));
  const available = matched.filter((o) => !o.disabled);
  const gated = matched.filter((o) => o.disabled);
  return {
    visible: showUnavailable ? [...available, ...gated] : available,
    hiddenCount: showUnavailable ? 0 : gated.length,
    totalGated: (options || []).filter((o) => o.value && o.disabled).length,
  };
}

/** Group options for display, preserving the order they arrived in. */
export function groupOptions(options) {
  const out = [];
  const byGroup = new Map();
  for (const o of options) {
    const key = o.group ?? '';
    if (!byGroup.has(key)) {
      const entry = { group: key, items: [] };
      byGroup.set(key, entry);
      out.push(entry);
    }
    byGroup.get(key).items.push(o);
  }
  return out;
}

export default function AddLayerPicker({
  options = [], onAdd, disabled = false, disabledLabel = null,
  // Rendered beside the button, sharing its row. It belongs in here rather
  // than next to the whole picker so the open list can span the full panel
  // width instead of the button's half — layer names are long.
  trailing = null,
}) {
  const { t } = useTranslation('phasemap');
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [showUnavailable, setShowUnavailable] = useState(false);
  const inputRef = useRef(null);

  const { visible, hiddenCount, totalGated } = useMemo(
    () => partitionOptions(options, query, showUnavailable),
    [options, query, showUnavailable],
  );
  const groups = useMemo(() => groupOptions(visible), [visible]);

  // Gated entries stay clickable once revealed: "add Forward NCC, then compute
  // diagnostics" is a real order of work, and the layer row shows the gate.
  const pick = (opt) => {
    if (!opt) return;
    onAdd?.(opt.value);
    setQuery('');
    setOpen(false);
  };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') {
      setOpen(false);
      return;
    }
    if (e.key === 'Enter') {
      // Enter takes the first hit, which is what a search box is for.
      pick(visible.find((o) => !o.disabled));
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%', minWidth: 0 }}>
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <button
          type="button"
          onClick={() => {
            const next = !open;
            setOpen(next);
            if (next) setTimeout(() => inputRef.current?.focus(), 0);
          }}
          disabled={disabled}
          title={t('phasemap:hoverTips.addLayer')}
          aria-expanded={open}
          style={{
            flex: 1, minWidth: 0,
            fontSize: '9pt', height: 26, padding: '2px 6px',
            background: colors.bgTertiary,
            color: disabled ? colors.textSecondary : colors.text,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            cursor: disabled ? 'not-allowed' : 'pointer',
            textAlign: 'left',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}
        >
          {disabled && disabledLabel ? disabledLabel : t('phasemap:layers.addLayer')}
        </button>
        {trailing ? (
          <div style={{ flex: 1, minWidth: 0, display: 'flex' }}>{trailing}</div>
        ) : null}
      </div>

      {open && !disabled && (
        <div
          style={{
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            background: colors.bgSecondary,
            padding: 4,
            display: 'flex', flexDirection: 'column', gap: 4,
          }}
          onKeyDown={onKeyDown}
        >
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t('phasemap:layers.searchPlaceholder')}
            aria-label={t('phasemap:layers.searchPlaceholder')}
            style={{
              fontSize: '9pt', height: 24, padding: '2px 6px',
              background: colors.bg, color: colors.text,
              border: `1px solid ${colors.border}`, borderRadius: 3,
              minWidth: 0,
            }}
          />

          <label style={{
            display: 'flex', alignItems: 'center', gap: 5,
            fontSize: '8pt', color: colors.textSecondary, cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={showUnavailable}
              onChange={(e) => setShowUnavailable(e.target.checked)}
            />
            {totalGated > 0
              ? t('phasemap:layers.showUnavailableN', { n: totalGated })
              : t('phasemap:layers.showUnavailable')}
          </label>

          <div style={{ maxHeight: 260, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
            {visible.length === 0 && (
              <div style={{ fontSize: '8pt', color: colors.textSecondary, padding: '6px 4px' }}>
                {t('phasemap:layers.noMatches')}
              </div>
            )}
            {groups.map((g) => (
              <div key={g.group}>
                <div style={{
                  fontSize: '7.5pt', color: colors.textSecondary,
                  padding: '4px 4px 2px', textTransform: 'uppercase', letterSpacing: 0.4,
                }}>
                  {g.group}
                </div>
                {g.items.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    onClick={() => pick(o)}
                    title={o.disabled ? (o.tip || t('phasemap:layers.unavailable')) : o.label}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left',
                      fontSize: '9pt', padding: '3px 6px',
                      background: 'transparent',
                      color: o.disabled ? colors.textSecondary : colors.text,
                      border: 'none', borderRadius: 2,
                      cursor: 'pointer',
                      opacity: o.disabled ? 0.65 : 1,
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}
                  >
                    {o.name || o.label}
                    {o.disabled && o.tip ? (
                      <span style={{ fontSize: '7.5pt', fontStyle: 'italic' }}> — {o.tip}</span>
                    ) : null}
                  </button>
                ))}
              </div>
            ))}
          </div>

          {hiddenCount > 0 && (
            <div style={{ fontSize: '7.5pt', color: colors.textSecondary, padding: '0 4px 2px' }}>
              {t('phasemap:layers.hiddenCount', { n: hiddenCount })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
