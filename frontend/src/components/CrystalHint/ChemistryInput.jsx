/**
 * Chemistry input — element chips with add/remove + EDS-detected hint.
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

const S = {
  wrap: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    flexWrap: 'wrap',
  },
  label: {
    fontSize: 11,
    color: colors.textSecondary,
  },
  chip: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    background: colors.bgSecondary,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    padding: '2px 6px 2px 8px',
    borderRadius: 12,
    fontSize: 11,
  },
  chipRemove: {
    background: 'transparent',
    color: colors.textSecondary,
    border: 'none',
    cursor: 'pointer',
    fontSize: 12,
    padding: '0 2px',
    lineHeight: 1,
  },
  input: {
    background: 'transparent',
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    padding: '3px 6px',
    fontSize: 11,
    width: 50,
  },
  hint: {
    fontSize: 10,
    color: colors.textSecondary,
    fontStyle: 'italic',
  },
};

// Common element list for quick-pick (could be extended)
const COMMON_ELEMENTS = ['Al', 'Si', 'Cu', 'Fe', 'Mg', 'Mn', 'Zn', 'Cr', 'Ti', 'Ni', 'Co', 'V', 'C', 'O'];

export default function ChemistryInput({ elements, onAdd, onRemove, edsElements }) {
  const { t } = useTranslation('crystalhint');
  const [draft, setDraft] = useState('');

  const handleSubmit = (e) => {
    e?.preventDefault?.();
    const clean = (draft || '').trim();
    if (!clean) return;
    // Capitalize first letter only, lowercase rest
    const sym = clean[0].toUpperCase() + clean.slice(1).toLowerCase();
    onAdd(sym);
    setDraft('');
  };

  return (
    <div style={S.wrap}>
      <span style={S.label}>{t('chemistry.elements')}</span>
      {elements.map(el => (
        <span key={el} style={S.chip}>
          {el}
          <button
            style={S.chipRemove}
            onClick={() => onRemove(el)}
            title={t('chemistry.removeTooltip', { el })}
            aria-label={t('chemistry.removeTooltip', { el })}
          >×</button>
        </span>
      ))}
      <form onSubmit={handleSubmit} style={{ display: 'inline' }}>
        <input
          style={S.input}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          placeholder={t('chemistry.addPlaceholder')}
          title={t('chemistry.addTooltip')}
        />
      </form>
      {edsElements && edsElements.length > 0 && (
        <span style={S.hint} title={edsElements.join(', ')}>
          {t('chemistry.edsDetected', { count: edsElements.length })}
        </span>
      )}
    </div>
  );
}
