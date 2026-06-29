import { useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import useEdsColorStore from '../../stores/useEdsColorStore';
import useDataStore from '../../stores/useDataStore';
import { ELEMENTS, F_BLOCK_MARKERS } from './periodicTableData';
import { colors } from '../../theme/tokens';

const CELL_SIZE = 40;
const GAP = 2;

// Build reverse lookups: full name → symbol, e.g. "Aluminium" → "Al"
const NAME_TO_SYMBOL = Object.fromEntries(
  ELEMENTS.map((el) => [el.name.toLowerCase(), el.symbol])
);
const ALL_SYMBOLS = new Set(ELEMENTS.map((el) => el.symbol));

/**
 * Normalize an EDS element label (e.g. "Al Kα1", "Aluminium", "Fe")
 * to its chemical symbol ("Al", "Fe").
 */
function toSymbol(label) {
  if (!label) return null;
  const s = label.trim();
  // Direct symbol match (e.g. "Fe", "Al")
  if (ALL_SYMBOLS.has(s)) return s;
  // "Fe Kα1" → first word "Fe"
  const first = s.split(/\s/)[0];
  if (ALL_SYMBOLS.has(first)) return first;
  // Full name match: "Aluminium" → "Al"
  const lower = s.toLowerCase();
  if (NAME_TO_SYMBOL[lower]) return NAME_TO_SYMBOL[lower];
  // Partial: "Aluminium Kα1" → try first word as name
  const firstLower = first.toLowerCase();
  if (NAME_TO_SYMBOL[firstLower]) return NAME_TO_SYMBOL[firstLower];
  return null;
}

function ElementCell({ element, color, inFile, onColorChange }) {
  const { t } = useTranslation('eds');
  const inputRef = useRef(null);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleChange = useCallback((e) => {
    onColorChange(element.symbol, e.target.value);
  }, [element.symbol, onColorChange]);

  const textColor = inFile ? '#fff' : colors.textSecondary;
  const bg = inFile ? color : `${color}30`;
  const border = inFile
    ? '2px solid #fff'
    : `1px solid ${colors.border}`;

  return (
    <div
      onClick={handleClick}
      title={t('periodicTable.cellTooltip', { name: element.name, z: element.z })}
      style={{
        gridColumn: element.col,
        gridRow: element.row,
        width: CELL_SIZE,
        height: CELL_SIZE,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 4,
        cursor: 'pointer',
        background: bg,
        border,
        boxShadow: inFile ? `0 0 6px ${color}66` : 'none',
        transition: 'transform 0.12s, box-shadow 0.12s',
        position: 'relative',
        opacity: inFile ? 1 : 0.5,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'scale(1.18)';
        e.currentTarget.style.zIndex = '10';
        e.currentTarget.style.boxShadow = `0 0 10px ${color}88`;
        e.currentTarget.style.opacity = '1';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
        e.currentTarget.style.zIndex = '1';
        e.currentTarget.style.boxShadow = inFile ? `0 0 6px ${color}66` : 'none';
        e.currentTarget.style.opacity = inFile ? '1' : '0.5';
      }}
    >
      <span style={{ fontSize: 7, color: textColor, opacity: 0.7, lineHeight: 1 }}>{element.z}</span>
      <span style={{ fontSize: 12, fontWeight: 700, color: textColor, lineHeight: 1.1 }}>{element.symbol}</span>
      <input
        ref={inputRef}
        type="color"
        value={color}
        onChange={handleChange}
        title={t('hoverTips.colorInput')}
        style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }}
        tabIndex={-1}
      />
    </div>
  );
}

function PeriodicTableDialog({ onClose }) {
  const { t } = useTranslation('eds');
  const { colors: elementColors, setColor, resetColors } = useEdsColorStore();
  const edsElements = useDataStore((s) => s.edsElements);

  const fileElements = new Set(
    (edsElements || [])
      .map((e) => {
        const raw = typeof e === 'string' ? e : e.name || e.element || String(e);
        return toSymbol(raw);
      })
      .filter(Boolean)
  );

  const handleColorChange = useCallback((symbol, color) => {
    setColor(symbol, color);
  }, [setColor]);

  return (
    <div style={{
      position: 'absolute',
      top: 40,
      left: 8,
      background: colors.bgSecondary,
      border: `1px solid ${colors.border}`,
      borderRadius: 8,
      padding: 16,
      zIndex: 90,
      boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      animation: 'fadeSlideIn 0.2s ease-out',
      overflowX: 'auto',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ color: colors.purple, fontWeight: 700, fontSize: 12 }}>
          {t('periodicTable.title')}
        </span>
        <button
          onClick={onClose}
          title={t('periodicTable.closeTooltip')}
          style={{
            background: 'transparent', border: 'none', color: colors.textSecondary,
            cursor: 'pointer', fontSize: 16, borderRadius: 3, padding: '0 4px',
            transition: 'color 0.12s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = colors.text; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = colors.textSecondary; }}
        >&times;</button>
      </div>

      {/* Main periodic table grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(18, ${CELL_SIZE}px)`,
        gridTemplateRows: `repeat(7, ${CELL_SIZE}px) 12px repeat(2, ${CELL_SIZE}px)`,
        gap: GAP,
        fontFamily: "'Segoe UI', system-ui, sans-serif",
      }}>
        {ELEMENTS.filter((el) => el.row <= 7).map((el) => (
          <ElementCell
            key={el.symbol}
            element={el}
            color={elementColors[el.symbol] || '#bd93f9'}
            inFile={fileElements.has(el.symbol)}
            onColorChange={handleColorChange}
          />
        ))}

        {/* F-block markers */}
        {F_BLOCK_MARKERS.map((m) => (
          <div key={m.label} style={{
            gridColumn: m.col, gridRow: m.row,
            width: CELL_SIZE, height: CELL_SIZE,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 7, color: colors.textSecondary,
            border: `1px dashed ${colors.border}`,
            borderRadius: 4,
          }}>{m.label}</div>
        ))}

        {/* Lanthanides & Actinides — rows 8 & 9, shifted to grid rows 9 & 10 */}
        {ELEMENTS.filter((el) => el.row >= 8).map((el) => (
          <ElementCell
            key={el.symbol}
            element={{ ...el, row: el.row + 1 }}
            color={elementColors[el.symbol] || '#bd93f9'}
            inFile={fileElements.has(el.symbol)}
            onColorChange={handleColorChange}
          />
        ))}
      </div>

      {/* Legend + Reset */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
        <div style={{ display: 'flex', gap: 16, fontSize: 10, color: colors.textSecondary }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#ff5555', border: '2px solid #fff' }} />
            {t('periodicTable.legendInFile')}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: `${colors.textSecondary}30`, border: `1px solid ${colors.border}` }} />
            {t('periodicTable.legendAvailable')}
          </span>
        </div>
        <button
          onClick={resetColors}
          title={t('hoverTips.resetColors')}
          style={{
            padding: '4px 14px', background: 'transparent',
            border: `1px solid ${colors.border}`, borderRadius: 4,
            color: colors.textSecondary, fontSize: 10, cursor: 'pointer',
            transition: 'border-color 0.12s, color 0.12s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = colors.purple; e.currentTarget.style.color = colors.text; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = colors.border; e.currentTarget.style.color = colors.textSecondary; }}
        >{t('periodicTable.resetDefaults')}</button>
      </div>
    </div>
  );
}

export default PeriodicTableDialog;
