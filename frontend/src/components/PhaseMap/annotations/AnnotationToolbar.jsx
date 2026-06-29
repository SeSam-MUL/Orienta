import React from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing, CollapsibleGroup } from '../../../theme/components';

/**
 * Toolbar for adding / editing canvas annotations (Legend, Scalebar,
 * Title text, North-Arrow). Lives in the PhaseMapPage side column.
 *
 * Layout: Add row (four buttons) + selected-annotation props panel.
 * The props panel adapts to the selected annotation's type.
 */
function AnnotationToolbar({
  annotations,
  selectedId,
  onSelect,
  onAdd,
  onRemove,
  onUpdate,
  onClear,
}) {
  const { t } = useTranslation('phasemap');
  const TYPE_LABELS = {
    legend: t('phasemap:annotations.typeLegend'),
    scalebar: t('phasemap:annotations.typeScalebar'),
    title: t('phasemap:annotations.typeTitle'),
    arrow: t('phasemap:annotations.typeArrow'),
  };
  const selected = annotations.find((a) => a.id === selectedId) || null;
  const btn = {
    background: colors.bg,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 3,
    padding: '4px 8px',
    fontSize: '9pt',
    cursor: 'pointer',
  };
  const inputStyle = {
    background: '#1f2937',
    color: '#e5e7eb',
    border: '1px solid #374151',
    borderRadius: 3,
    padding: '3px 6px',
    fontSize: '9pt',
    width: '100%',
    boxSizing: 'border-box',
    fontVariantNumeric: 'tabular-nums',
  };
  const Row = ({ label, children }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
      <span style={{ fontSize: '8.5pt', color: colors.textSecondary, minWidth: 76 }}>
        {label}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>{children}</div>
    </div>
  );

  let propsPanel = null;
  if (selected) {
    const p = selected.props || {};
    const patchProp = (key, val) => onUpdate(selected.id, { props: { [key]: val } });
    switch (selected.type) {
      case 'legend':
        propsPanel = (
          <>
            <Row label={t('phasemap:annotations.fontSize')}>
              <input
                type="number" min={6} max={48} step={1}
                value={p.fontSize ?? 11}
                onChange={(e) => patchProp('fontSize', Number(e.target.value) || 11)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotFontSize')}
              />
            </Row>
            <Row label={t('phasemap:annotations.background')}>
              <input
                type="text"
                value={p.background ?? 'rgba(20,22,30,0.85)'}
                onChange={(e) => patchProp('background', e.target.value)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotBackground')}
              />
            </Row>
          </>
        );
        break;
      case 'scalebar':
        propsPanel = (
          <>
            <Row label={t('phasemap:annotations.lengthUm')}>
              <input
                type="number" min={0.1} step={0.5}
                value={p.lengthUm ?? 5}
                onChange={(e) => patchProp('lengthUm', Number(e.target.value) || 5)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotLengthUm')}
              />
            </Row>
            <Row label={t('phasemap:annotations.fontSize')}>
              <input
                type="number" min={6} max={48} step={1}
                value={p.fontSize ?? 12}
                onChange={(e) => patchProp('fontSize', Number(e.target.value) || 12)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotFontSize')}
              />
            </Row>
            <Row label={t('phasemap:annotations.barColor')}>
              <input
                type="color"
                value={p.barColor ?? '#ffffff'}
                onChange={(e) => patchProp('barColor', e.target.value)}
                style={{ ...inputStyle, padding: 0, width: '100%', height: 26 }}
                title={t('phasemap:hoverTips.annotBarColor')}
              />
            </Row>
            <Row label={t('phasemap:annotations.textColor')}>
              <input
                type="color"
                value={p.textColor ?? '#ffffff'}
                onChange={(e) => patchProp('textColor', e.target.value)}
                style={{ ...inputStyle, padding: 0, width: '100%', height: 26 }}
                title={t('phasemap:hoverTips.annotTextColor')}
              />
            </Row>
          </>
        );
        break;
      case 'title':
        propsPanel = (
          <>
            <Row label={t('phasemap:annotations.text')}>
              <input
                type="text"
                value={p.text ?? ''}
                onChange={(e) => patchProp('text', e.target.value)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotText')}
              />
            </Row>
            <Row label={t('phasemap:annotations.fontSize')}>
              <input
                type="number" min={8} max={72} step={1}
                value={p.fontSize ?? 16}
                onChange={(e) => patchProp('fontSize', Number(e.target.value) || 16)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotFontSize')}
              />
            </Row>
            <Row label={t('phasemap:annotations.color')}>
              <input
                type="color"
                value={p.color ?? '#ffffff'}
                onChange={(e) => patchProp('color', e.target.value)}
                style={{ ...inputStyle, padding: 0, width: '100%', height: 26 }}
                title={t('phasemap:hoverTips.annotColor')}
              />
            </Row>
          </>
        );
        break;
      case 'arrow':
        propsPanel = (
          <>
            <Row label={t('phasemap:annotations.label')}>
              <select
                value={p.label ?? 'ND'}
                onChange={(e) => patchProp('label', e.target.value)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotArrowLabel')}
              >
                <option value="ND">ND</option>
                <option value="RD">RD</option>
                <option value="TD">TD</option>
                <option value="X">X</option>
                <option value="Y">Y</option>
                <option value="Z">Z</option>
              </select>
            </Row>
            <Row label={t('phasemap:annotations.color')}>
              <input
                type="color"
                value={p.color ?? '#ffb86c'}
                onChange={(e) => patchProp('color', e.target.value)}
                style={{ ...inputStyle, padding: 0, width: '100%', height: 26 }}
                title={t('phasemap:hoverTips.annotColor')}
              />
            </Row>
            <Row label={t('phasemap:annotations.fontSize')}>
              <input
                type="number" min={6} max={36} step={1}
                value={p.fontSize ?? 11}
                onChange={(e) => patchProp('fontSize', Number(e.target.value) || 11)}
                style={inputStyle}
                title={t('phasemap:hoverTips.annotFontSize')}
              />
            </Row>
          </>
        );
        break;
      default:
        propsPanel = null;
    }
  }

  return (
    <CollapsibleGroup title={t('phasemap:annotations.title')}>
      <div style={{ marginTop: spacing.innerSpacing }}>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 6 }}>
          <button style={btn} onClick={() => onAdd('legend')} title={t('phasemap:hoverTips.addLegend')}>{t('phasemap:annotations.addLegend')}</button>
          <button style={btn} onClick={() => onAdd('scalebar')} title={t('phasemap:hoverTips.addScalebar')}>{t('phasemap:annotations.addScalebar')}</button>
          <button style={btn} onClick={() => onAdd('title')} title={t('phasemap:hoverTips.addTitle')}>{t('phasemap:annotations.addTitle')}</button>
          <button style={btn} onClick={() => onAdd('arrow')} title={t('phasemap:hoverTips.addArrow')}>{t('phasemap:annotations.addArrow')}</button>
        </div>
        {annotations.length === 0 && (
          <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
            {t('phasemap:annotations.empty')}
          </div>
        )}
        {annotations.length > 0 && (
          <div style={{ marginBottom: 6 }}>
            <div style={{ fontSize: '8pt', color: colors.textSecondary, marginBottom: 3 }}>
              {t('phasemap:annotations.count', { count: annotations.length })}
              {' · '}
              <button
                onClick={onClear}
                title={t('phasemap:hoverTips.clearAnnotations')}
                style={{
                  background: 'transparent', border: 'none',
                  color: '#ff7a7a', cursor: 'pointer',
                  fontSize: '8pt', padding: 0, textDecoration: 'underline',
                }}
              >{t('phasemap:annotations.clearAll')}</button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {annotations.map((a) => (
                <div
                  key={a.id}
                  onClick={() => onSelect(a.id)}
                  title={t('phasemap:hoverTips.selectAnnotation')}
                  style={{
                    display: 'flex', alignItems: 'center',
                    padding: '3px 6px',
                    fontSize: '8.5pt',
                    background: selectedId === a.id ? '#22273a' : 'transparent',
                    border: `1px solid ${selectedId === a.id ? '#50fa7b' : 'transparent'}`,
                    borderRadius: 3, cursor: 'pointer',
                  }}
                >
                  <span style={{ flex: 1, color: colors.text }}>
                    {TYPE_LABELS[a.type] || a.type}
                    {a.type === 'title' ? `: ${(a.props?.text || '').slice(0, 18)}` : ''}
                    {a.type === 'arrow' ? ` (${a.props?.label || 'ND'})` : ''}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onRemove(a.id); }}
                    aria-label={t('phasemap:hoverTips.removeAnnotation')}
                    title={t('phasemap:hoverTips.removeAnnotation')}
                    style={{
                      background: 'transparent', border: 'none',
                      color: colors.textSecondary, cursor: 'pointer',
                      fontSize: '10pt', padding: 0, lineHeight: 1,
                    }}
                  >×</button>
                </div>
              ))}
            </div>
          </div>
        )}
        {selected && propsPanel && (
          <div style={{
            padding: 8, marginTop: 4,
            background: colors.bg, borderRadius: 3,
            border: `1px solid ${colors.border}`,
          }}>
            <div style={{
              fontSize: '8pt', color: colors.textSecondary, marginBottom: 6,
              textTransform: 'uppercase', letterSpacing: 0.5,
            }}>
              {t('phasemap:annotations.propertiesHeading', { type: TYPE_LABELS[selected.type] || selected.type })}
            </div>
            {propsPanel}
          </div>
        )}
      </div>
    </CollapsibleGroup>
  );
}

export default AnnotationToolbar;
