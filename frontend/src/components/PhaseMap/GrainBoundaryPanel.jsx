import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { applyBandEdit, defaultBands } from './grainBoundaryBands';

/**
 * Settings for the grain-boundary overlay: three classes, each with its angle
 * range, colour and line width.
 *
 * The ranges are adjacent by construction — see grainBoundaryBands.js. Editing
 * one bound moves the shared cut point, so the panel can never show two classes
 * that claim the same angle.
 */
export default function GrainBoundaryPanel({ bands, onChange }) {
  const { t } = useTranslation('phasemap');
  const list = Array.isArray(bands) && bands.length === 3 ? bands : defaultBands();

  const NAMES = {
    sub: t('phasemap:grainBoundaries.sub'),
    lagb: t('phasemap:grainBoundaries.lagb'),
    hagb: t('phasemap:grainBoundaries.hagb'),
  };

  const edit = (index, field, value) => onChange(applyBandEdit(list, index, field, value));

  const numberStyle = {
    width: '100%', boxSizing: 'border-box',
    background: '#1f2937', color: '#e5e7eb',
    border: '1px solid #374151', borderRadius: 3,
    padding: '2px 4px', fontSize: '8.5pt', fontVariantNumeric: 'tabular-nums',
  };

  return (
    <div style={{
      marginTop: 4, padding: 6,
      background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 3,
    }}>
      <div style={{
        display: 'grid', gridTemplateColumns: '14px minmax(0,1fr) 46px 46px 26px 34px',
        gap: 4, alignItems: 'center',
        fontSize: '7.5pt', color: colors.textSecondary, marginBottom: 3,
      }}>
        <span />
        <span>{t('phasemap:grainBoundaries.classHeading')}</span>
        <span>{t('phasemap:grainBoundaries.fromHeading')}</span>
        <span>{t('phasemap:grainBoundaries.toHeading')}</span>
        <span>{t('phasemap:grainBoundaries.colorHeading')}</span>
        <span>{t('phasemap:grainBoundaries.widthHeading')}</span>
      </div>

      {list.map((band, i) => (
        <div
          key={band.id}
          style={{
            display: 'grid', gridTemplateColumns: '14px minmax(0,1fr) 46px 46px 26px 34px',
            gap: 4, alignItems: 'center', marginBottom: 3,
            opacity: band.on === false ? 0.5 : 1,
          }}
        >
          <input
            type="checkbox"
            checked={band.on !== false}
            onChange={(e) => edit(i, 'on', e.target.checked)}
            aria-label={NAMES[band.id] ?? band.id}
            title={t('phasemap:hoverTips.gbClassOn')}
            style={{ margin: 0 }}
          />
          <span style={{
            fontSize: '8.5pt', color: colors.text,
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>
            {NAMES[band.id] ?? band.id}
          </span>
          <input
            type="number" step="0.5" min="0.5" max="62.8"
            value={band.min}
            onChange={(e) => edit(i, 'min', e.target.value)}
            title={t('phasemap:hoverTips.gbFrom')}
            data-gb-min={band.id}
            style={numberStyle}
          />
          {band.max == null ? (
            // The top class has no upper end — every larger angle belongs to it.
            <span style={{ fontSize: '8.5pt', color: colors.textSecondary, textAlign: 'center' }}>∞</span>
          ) : (
            <input
              type="number" step="0.5" min="0.5" max="62.8"
              value={band.max}
              onChange={(e) => edit(i, 'max', e.target.value)}
              title={t('phasemap:hoverTips.gbTo')}
              data-gb-max={band.id}
              style={numberStyle}
            />
          )}
          <input
            type="color"
            value={band.color}
            onChange={(e) => edit(i, 'color', e.target.value)}
            title={t('phasemap:hoverTips.gbColor')}
            data-gb-color={band.id}
            style={{
              width: 24, height: 20, padding: 0, border: `1px solid ${colors.border}`,
              background: 'transparent', cursor: 'pointer',
            }}
          />
          <input
            type="number" step="1" min="1" max="8"
            value={band.width}
            onChange={(e) => edit(i, 'width', e.target.value)}
            title={t('phasemap:hoverTips.gbWidth')}
            data-gb-width={band.id}
            style={numberStyle}
          />
        </div>
      ))}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
        <span style={{ fontSize: '7.5pt', color: colors.textSecondary }}>
          {t('phasemap:grainBoundaries.unitHint')}
        </span>
        <button
          onClick={() => onChange(defaultBands())}
          title={t('phasemap:hoverTips.gbReset')}
          style={{
            background: 'transparent', border: 'none', color: colors.textSecondary,
            cursor: 'pointer', fontSize: '8pt', padding: 0, textDecoration: 'underline',
          }}
        >
          {t('phasemap:grainBoundaries.reset')}
        </button>
      </div>
    </div>
  );
}
