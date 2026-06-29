/**
 * EdsOverlayPanel.jsx
 *
 * Panel for toggling EDS element overlays on the navigation canvas.
 * Default: uses element colors from useEdsColorStore as single-color RGBA overlays.
 * Optional "Heatmap" toggle: falls back to matplotlib colormaps.
 */

import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { edsApi } from '../../services/api';
import { colors as C, alpha } from '../../theme/tokens';
import useEdsColorStore from '../../stores/useEdsColorStore';

export default function EdsOverlayPanel({ enabled, available, onToggle, onLayersChange, fileKey }) {
  const { t } = useTranslation('indexing');
  const UNIT_OPTIONS = [
    { value: 'counts', label: t('eds.unitCounts') },
    { value: 'wt_pct', label: t('eds.unitWt') },
    { value: 'at_pct', label: t('eds.unitAt') },
  ];

  const CMAP_OPTIONS = [
    { value: 'hot', label: t('eds.cmapHot') },
    { value: 'viridis', label: t('eds.cmapViridis') },
    { value: 'plasma', label: t('eds.cmapPlasma') },
    { value: 'inferno', label: t('eds.cmapInferno') },
    { value: 'gray', label: t('eds.cmapGray') },
    { value: 'jet', label: t('eds.cmapJet') },
  ];

  const [elements, setElements] = useState([]);
  const [activeElements, setActiveElements] = useState({});
  const [unit, setUnit] = useState('counts');
  const [useElementColors, setUseElementColors] = useState(true);
  const [cmap, setCmap] = useState('hot');
  // Subscribe to the element→colour map so an "EDS Colors" edit recolours the
  // pills AND refetches active overlays (the colour is baked into the backend
  // PNG). getState().getColor() alone was a non-reactive snapshot, so edits
  // never showed until some other prop forced a re-render.
  const elementColors = useEdsColorStore((s) => s.colors);

  // Drop toggled elements + cached overlay images when the underlying dataset
  // changes — otherwise file-A's "Al" stays toggled with file-A's PNG painted
  // on top of file-B's navigation map.
  useEffect(() => {
    setActiveElements({});
  }, [fileKey]);

  // Fetch available elements (re-fetches on file change so we get the new
  // element list even if `available` stays true across the switch).
  useEffect(() => {
    if (!available) {
      setElements([]);
      return;
    }
    edsApi.elements().then(r => {
      const els = r.data?.elements || r.data || [];
      setElements(Array.isArray(els) ? els : []);
    }).catch(() => setElements([]));
  }, [available, fileKey]);

  // Clean element symbol for color lookup: "Window Integral Al Ka1" → "Al"
  const getElementInfo = useCallback((rawEl) => {
    const cleaned = rawEl.replace(/^Window Integral\s*/i, '').trim();
    const symbol = cleaned.split(/\s/)[0] || cleaned;
    const color = elementColors[symbol] || '#bd93f9';  // == old getColor(symbol)
    return { label: cleaned, symbol, color };
  }, [elementColors]);

  // Fetch a single element map (color overlay or heatmap)
  const fetchElementMap = useCallback(async (rawEl) => {
    const { color } = getElementInfo(rawEl);
    try {
      const res = await edsApi.getMap(
        rawEl,
        unit,
        useElementColors ? 'hot' : cmap,
        useElementColors ? color : ''
      );
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = `data:image/png;base64,${res.data.image}`;
      });
      return img;
    } catch {
      return null;
    }
  }, [unit, cmap, useElementColors, getElementInfo]);

  // Reload all active maps when unit/cmap/color-mode changes
  useEffect(() => {
    if (!enabled) return;
    const activeKeys = Object.entries(activeElements).filter(([, v]) => v.on).map(([k]) => k);
    if (activeKeys.length === 0) return;

    let cancelled = false;
    (async () => {
      const updates = {};
      for (const rawEl of activeKeys) {
        const img = await fetchElementMap(rawEl);
        if (cancelled) return;
        if (img) updates[rawEl] = img;
      }
      if (cancelled) return;
      setActiveElements(prev => {
        const next = { ...prev };
        for (const [k, img] of Object.entries(updates)) {
          if (next[k]?.on) next[k] = { ...next[k], image: img };
        }
        return next;
      });
    })();
    return () => { cancelled = true; };
  }, [unit, cmap, useElementColors, enabled, elementColors]); // eslint-disable-line react-hooks/exhaustive-deps

  // Toggle element on/off
  const toggleElement = useCallback(async (rawEl, on) => {
    if (on) {
      setActiveElements(prev => ({
        ...prev,
        [rawEl]: { on: true, opacity: prev[rawEl]?.opacity ?? 0.5, image: null },
      }));
      const img = await fetchElementMap(rawEl);
      if (img) {
        setActiveElements(prev => {
          if (!prev[rawEl]?.on) return prev;
          return { ...prev, [rawEl]: { ...prev[rawEl], image: img } };
        });
      }
    } else {
      setActiveElements(prev => ({
        ...prev,
        [rawEl]: { ...prev[rawEl], on: false, image: null },
      }));
    }
  }, [fetchElementMap]);

  const setOpacity = useCallback((el, opacity) => {
    setActiveElements(prev => ({
      ...prev,
      [el]: { ...prev[el], opacity },
    }));
  }, []);

  // Propagate layer changes to parent
  useEffect(() => {
    if (!enabled) { onLayersChange([]); return; }
    const layers = Object.entries(activeElements)
      .filter(([, v]) => v.on && v.image)
      .map(([el, v]) => ({ element: el, image: v.image, opacity: v.opacity }));
    onLayersChange(layers);
  }, [activeElements, enabled, onLayersChange]);

  if (!available) return null;

  const activeCount = Object.values(activeElements).filter(v => v.on).length;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 6,
      background: alpha(C.bgSecondary, 50),
      borderRadius: 6,
      padding: '8px 10px',
      border: `1px solid ${enabled ? alpha(C.cyan, 40) : C.border}`,
      transition: 'border-color 0.2s',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <label
          title={t('hoverTips.edsOverlayToggle')}
          style={{
          display: 'flex', alignItems: 'center', gap: 6,
          cursor: 'pointer', fontSize: '10pt',
          color: enabled ? C.cyan : C.textSecondary,
          fontWeight: 600, userSelect: 'none',
        }}>
          <input
            type="checkbox"
            checked={enabled}
            onChange={e => onToggle(e.target.checked)}
            style={{ accentColor: C.cyan, width: 15, height: 15 }}
          />
          {t('eds.overlay')}
        </label>
        {enabled && activeCount > 0 && (
          <span style={{
            fontSize: '8pt', color: C.cyan, fontWeight: 600,
            background: alpha(C.cyan, 15), borderRadius: 8,
            padding: '1px 7px',
          }}>
            {t('eds.active', { count: activeCount })}
          </span>
        )}
      </div>

      {enabled && elements.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {/* Controls row */}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
            <MiniSelect label={t('eds.unit')} value={unit} options={UNIT_OPTIONS} onChange={setUnit} title={t('hoverTips.edsUnit')} />
            <div style={{ flex: 1 }} />
            <label style={{
              display: 'flex', alignItems: 'center', gap: 4,
              cursor: 'pointer', fontSize: '8pt', color: C.textSecondary, userSelect: 'none',
            }} title={t('eds.heatmapTip')}>
              <input
                type="checkbox"
                checked={!useElementColors}
                onChange={e => setUseElementColors(!e.target.checked)}
                style={{ accentColor: C.purple, width: 12, height: 12 }}
              />
              {t('eds.heatmap')}
            </label>
            {!useElementColors && (
              <MiniSelect label="" value={cmap} options={CMAP_OPTIONS} onChange={setCmap} title={t('hoverTips.edsColormap')} />
            )}
          </div>

          {/* Element list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {elements.map(rawEl => {
              const { label, symbol, color } = getElementInfo(rawEl);
              const state = activeElements[rawEl] || { on: false, opacity: 0.5 };
              return (
                <ElementRow
                  key={rawEl}
                  rawEl={rawEl}
                  label={label}
                  symbol={symbol}
                  color={color}
                  isOn={!!state.on}
                  opacity={state.opacity ?? 0.5}
                  loading={state.on && !state.image}
                  onToggle={(on) => toggleElement(rawEl, on)}
                  onOpacity={(v) => setOpacity(rawEl, v)}
                  t={t}
                />
              );
            })}
          </div>
        </div>
      )}

      {enabled && elements.length === 0 && (
        <span style={{ color: C.textSecondary, fontSize: '8pt', fontStyle: 'italic' }}>
          {t('eds.noElements')}
        </span>
      )}
    </div>
  );
}

/** Single element row: color pill + name + opacity slider */
function ElementRow({ rawEl, label, symbol, color, isOn, opacity, loading, onToggle, onOpacity, t }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      title={t('hoverTips.edsElementRow', { element: label })}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '3px 6px', borderRadius: 4,
        background: isOn
          ? alpha(C.bgTertiary, 60)
          : hovered ? alpha(C.bgTertiary, 30) : 'transparent',
        cursor: 'pointer',
        transition: 'background 0.15s',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onToggle(!isOn)}
    >
      {/* Color indicator */}
      <span style={{
        width: 14, height: 14, borderRadius: 3, flexShrink: 0,
        background: color,
        border: isOn ? '2px solid rgba(255,255,255,0.9)' : '2px solid rgba(255,255,255,0.15)',
        boxShadow: isOn ? `0 0 6px ${color}` : 'none',
        transition: 'all 0.15s',
      }} />

      {/* Element name */}
      <span style={{
        color: isOn ? C.text : C.textSecondary,
        fontSize: '9pt', fontWeight: isOn ? 700 : 400,
        minWidth: 26,
        transition: 'color 0.15s',
      }}>
        {label}
      </span>

      {/* Loading dots */}
      {loading && (
        <span style={{ fontSize: '7pt', color: C.yellow, marginLeft: 'auto' }}>{t('eds.loading')}</span>
      )}

      {/* Opacity slider */}
      {isOn && !loading && (
        <>
          <input
            type="range" min={5} max={100}
            value={Math.round(opacity * 100)}
            onChange={e => { e.stopPropagation(); onOpacity(Number(e.target.value) / 100); }}
            onClick={e => e.stopPropagation()}
            style={{ flex: 1, height: 14, minWidth: 40, accentColor: color }}
            title={t('eds.opacity', { pct: Math.round(opacity * 100) })}
          />
          <span style={{ color: C.textSecondary, fontSize: '8pt', width: 28, textAlign: 'right' }}>
            {Math.round(opacity * 100)}%
          </span>
        </>
      )}
    </div>
  );
}

/** Compact labeled select */
function MiniSelect({ label, value, options, onChange, title }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
      {label && <span style={{ color: C.textSecondary, fontSize: '8pt' }}>{label}</span>}
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        title={title}
        style={{
          background: C.bg, border: `1px solid ${C.border}`,
          borderRadius: 3, color: C.text, fontSize: '8pt',
          padding: '1px 4px', height: 20, outline: 'none',
        }}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}
