/**
 * Crystal Hint — detect crystal symmetry + suggest candidate phases for
 * EBSD patterns. New workflow tab with three sub-modes:
 *  - Single Pixel Deep (Mode 1)
 *  - Region / Whole Scan (Mode 2)
 *  - Quality Check (Mode 3)
 *
 * Spec: docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md
 */

import { useEffect, useMemo, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import useDataStore from '../../stores/useDataStore';
import { crystalHintApi } from '../../services/api';
import SinglePixelMode from './SinglePixelMode';
import RegionMode from './RegionMode';
import QualityCheckMode from './QualityCheckMode';
import ChemistryInput from './ChemistryInput';
import MaterialPresetPicker from './MaterialPresetPicker';

const S = {
  page: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    background: colors.bg,
    color: colors.text,
    overflow: 'hidden',
  },
  header: {
    flexShrink: 0,
    padding: '10px 16px',
    borderBottom: `1px solid ${colors.border}`,
    background: colors.bgSecondary,
  },
  title: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 600,
    color: colors.accent,
  },
  subtitle: {
    fontSize: '11px',
    color: colors.textSecondary,
    marginTop: 2,
  },
  chrome: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  modeTabs: {
    display: 'flex',
    gap: 4,
    borderBottom: `1px solid ${colors.border}`,
    padding: '0 16px',
    background: colors.bgSecondary,
    flexShrink: 0,
  },
  modeTab: (active, enabled = true) => ({
    background: active ? colors.bg : 'transparent',
    color: active ? colors.accent : enabled ? colors.text : colors.textSecondary,
    border: 'none',
    borderBottom: active ? `2px solid ${colors.accent}` : '2px solid transparent',
    padding: '8px 14px',
    fontSize: '12px',
    fontWeight: active ? 600 : 500,
    cursor: enabled ? 'pointer' : 'not-allowed',
    opacity: enabled ? 1 : 0.5,
    transition: 'all 0.15s',
  }),
  body: {
    flex: 1,
    overflow: 'auto',
    padding: 16,
  },
  placeholder: {
    padding: 24,
    background: colors.bgSecondary,
    border: `1px dashed ${colors.border}`,
    borderRadius: 8,
    color: colors.textSecondary,
    textAlign: 'center',
    margin: 16,
  },
};

const MODES = {
  single: { id: 'single', labelKey: 'modes.single', tooltipKey: 'modes.singleTooltip', enabled: true },
  region: { id: 'region', labelKey: 'modes.region', tooltipKey: 'modes.regionTooltip', enabled: true },
  quality: { id: 'quality', labelKey: 'modes.quality', tooltipKey: 'modes.qualityTooltip', enabled: true },
};

export default function CrystalHintPage({ isActive, onNavigate }) {
  const { t } = useTranslation('crystalhint');
  const [mode, setMode] = useState('single');
  const [presetKey, setPresetKey] = useState('AA226');
  const [presets, setPresets] = useState({});
  const [presetsLoading, setPresetsLoading] = useState(false);
  const [elements, setElements] = useState([]);
  const edsElements = useDataStore(s => s.edsElements);

  // Fetch presets once
  useEffect(() => {
    let cancelled = false;
    setPresetsLoading(true);
    crystalHintApi
      .listPresets()
      .then(res => {
        if (cancelled) return;
        setPresets(res.data || {});
      })
      .catch(err => {
        console.error('Failed to load presets', err);
      })
      .finally(() => {
        if (!cancelled) setPresetsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // When preset changes, prime elements (user can edit after)
  useEffect(() => {
    if (!presetKey || !presets[presetKey]) return;
    const presetEls = presets[presetKey].elements || [];
    // Merge with EDS-detected elements (de-duplicated)
    const edsSimplified = (edsElements || []).map(e => {
      // EDS labels look like "Fe Kα1" — strip to symbol
      const m = String(e).match(/^([A-Z][a-z]?)/);
      return m ? m[1] : null;
    }).filter(Boolean);
    const merged = Array.from(new Set([...presetEls, ...edsSimplified]));
    setElements(merged);
  }, [presetKey, presets, edsElements]);

  const presetEntry = presets[presetKey];

  const handleAddElement = useCallback((el) => {
    if (!el) return;
    setElements(prev => prev.includes(el) ? prev : [...prev, el]);
  }, []);

  const handleRemoveElement = useCallback((el) => {
    setElements(prev => prev.filter(e => e !== el));
  }, []);

  const sharedChrome = (
    <div style={S.chrome}>
      <MaterialPresetPicker
        presets={presets}
        loading={presetsLoading}
        value={presetKey}
        onChange={setPresetKey}
      />
      <ChemistryInput
        elements={elements}
        onAdd={handleAddElement}
        onRemove={handleRemoveElement}
        edsElements={edsElements || []}
      />
      {presetEntry && (
        <div style={{ fontSize: 11, color: colors.textSecondary, marginLeft: 'auto' }}>
          {t('page.presetSummary')}<strong>{presetEntry.label}</strong>
          {' · '}
          {t('page.expectedPhasesCount', { count: presetEntry.expected_phases?.length || 0 })}
        </div>
      )}
    </div>
  );

  return (
    <div style={S.page}>
      <div style={S.header}>
        <h1 style={S.title}>{t('page.title')}</h1>
        <div style={S.subtitle}>
          {t('page.subtitle')}
        </div>
        {sharedChrome}
      </div>

      <div style={S.modeTabs}>
        {Object.values(MODES).map(m => (
          <button
            key={m.id}
            style={S.modeTab(mode === m.id)}
            onClick={() => setMode(m.id)}
            title={t(m.tooltipKey)}
          >
            {t(m.labelKey)}
          </button>
        ))}
      </div>

      <div style={S.body}>
        {mode === 'single' && (
          <SinglePixelMode
            elements={elements}
            presetKey={presetKey}
            presetEntry={presetEntry}
            isActive={isActive}
            onNavigate={onNavigate}
          />
        )}
        {mode === 'region' && (
          <RegionMode
            elements={elements}
            presetKey={presetKey}
            presetEntry={presetEntry}
            isActive={isActive}
          />
        )}
        {mode === 'quality' && (
          <QualityCheckMode
            elements={elements}
            presetKey={presetKey}
            presetEntry={presetEntry}
            isActive={isActive}
          />
        )}
      </div>
    </div>
  );
}
