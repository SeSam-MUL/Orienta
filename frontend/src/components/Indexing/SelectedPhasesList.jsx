/**
 * SelectedPhasesList.jsx
 *
 * Shows selected phases with Remove/Add buttons.
 * For dictionary method: renders DictPhaseCard with full hierarchy.
 * For hough/spherical: renders a simple colored card per phase.
 */

import { useTranslation } from 'react-i18next';
import DictPhaseCard from './DictPhaseCard';
import DegeneracyWarningBanner from './DegeneracyWarningBanner';
import ReflectorBudget from './ReflectorBudget';
import { colors as C } from '../../theme/tokens';

const PHASE_COLORS = ['#c3e88d', '#82aaff', '#f78c6c', '#c792ea', '#ffcb6b', '#89ddff', '#ff5370'];

export default function SelectedPhasesList({
  phases = [],
  method = 'hough',
  allDiscoveredFiles = [],
  onRemovePhase,
  onAddClick,
  currentPc = null,
  onSelectDict,
  selectedDictPaths = {},
  degeneracy = { clusters: [], byPhaseIndex: {} },
  onReducePhases,
  onGenerateDict,
  detectorShape = null,
  geom = null,
  maxReflectors = {},
  onMaxReflectorsChange,
  nBands = 12,
}) {
  const { t } = useTranslation('indexing');
  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        borderRadius: 6,
        padding: '10px 12px',
        background: C.bgSecondary,
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 8,
        }}
      >
        <span style={{ color: C.text, fontSize: '10pt', fontWeight: 600 }}>
          {t('phases.selectedPhases')}
        </span>
        {phases.length > 1 && (
          <span
            style={{
              fontSize: '9pt',
              color: '#c3e88d',
              background: 'rgba(195,232,141,0.12)',
              border: '1px solid rgba(195,232,141,0.3)',
              borderRadius: 4,
              padding: '1px 7px',
            }}
          >
            {t('phases.multiPhaseActive')}
          </span>
        )}
      </div>

      {/* Degeneracy warning — non-blocking */}
      <DegeneracyWarningBanner degeneracy={degeneracy} onReducePhases={onReducePhases} />

      {/* Phase list */}
      {phases.length === 0 ? (
        <div
          style={{
            color: C.textSecondary,
            fontSize: '9pt',
            textAlign: 'center',
            padding: '12px 0',
          }}
        >
          {t('phases.noPhaseSelected')}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {phases.map((phase, idx) => {
            const color = PHASE_COLORS[idx % PHASE_COLORS.length];
            if (method === 'dictionary') {
              return (
                <DictPhaseCard
                  key={idx}
                  phase={phase}
                  allFiles={allDiscoveredFiles}
                  currentPc={currentPc}
                  color={color}
                  onRemove={() => onRemovePhase && onRemovePhase(idx)}
                  onSelectDict={onSelectDict}
                  selectedDictPath={selectedDictPaths[phase.path] || ''}
                  onGenerateDict={onGenerateDict}
                  detectorShape={detectorShape}
                  geom={geom}
                />
              );
            }
            // Simple card for hough / spherical
            return (
              <SimplePhaseCard
                key={idx}
                phase={phase}
                color={color}
                degeneracy={degeneracy.byPhaseIndex?.[idx]}
                onRemove={() => onRemovePhase && onRemovePhase(idx)}
                t={t}
                // Hough only: the band-triplet library is a Hough structure,
                // and neither Dictionary nor Spherical builds one.
                reflectorControl={method === 'hough' && phase?.path ? (
                  <ReflectorBudget
                    cifPath={phase.path}
                    value={maxReflectors[phase.path] ?? null}
                    nBands={nBands}
                    onChange={(n) => onMaxReflectorsChange?.(phase.path, n)}
                  />
                ) : null}
              />
            );
          })}
        </div>
      )}

      {/* Add button */}
      <button
        onClick={onAddClick}
        title={t('hoverTips.selectedPhasesAdd')}
        style={{
          marginTop: 8,
          width: '100%',
          padding: '6px 0',
          background: 'transparent',
          border: `1px dashed ${C.border}`,
          borderRadius: 4,
          color: C.textSecondary,
          fontSize: '9pt',
          cursor: 'pointer',
          transition: 'color 0.15s, border-color 0.15s',
        }}
        onMouseEnter={e => {
          e.currentTarget.style.color = C.text;
          e.currentTarget.style.borderColor = C.text;
        }}
        onMouseLeave={e => {
          e.currentTarget.style.color = C.textSecondary;
          e.currentTarget.style.borderColor = C.border;
        }}
      >
        {t('phases.addPhase')}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SimplePhaseCard — used for hough / spherical methods
// ---------------------------------------------------------------------------

function SimplePhaseCard({ phase, color, degeneracy, onRemove, t, reflectorControl = null }) {
  // Canonical phase-identity label (formula → structure → α/β tag) — matches the
  // Phase-Tester. Falls back to the raw formula for older/un-enriched entries.
  const displayLabel = phase?.display_label || phase?.formula || '—';
  const crystalSystem = phase?.crystal_system || '';
  const filename     = phase?.filename     || phase?.path?.split(/[\\/]/).pop() || '—';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 8,
        background: 'rgba(255,255,255,0.04)',
        border: `1px solid ${C.border}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 4,
        padding: '6px 8px',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span style={{ color, fontWeight: 700, fontSize: '10pt' }}>{displayLabel}</span>
          {crystalSystem && (
            <span style={{ color: C.textSecondary, fontSize: '9pt' }}>{crystalSystem}</span>
          )}
          {degeneracy && (
            <span
              title={
                degeneracy.tier === 'hard'
                  ? t('degeneracy.cannotSeparate', { partners: (degeneracy.partners || []).join(', ') })
                  : t('degeneracy.maybeHard', { partners: (degeneracy.partners || []).join(', ') })
              }
              style={{
                fontSize: '8pt', fontWeight: 700, cursor: 'help',
                color: degeneracy.tier === 'hard' ? '#ff5370' : '#ffcb6b',
              }}
            >
              ⚠
            </span>
          )}
        </div>
        <div
          style={{
            color: C.textSecondary,
            fontSize: '8pt',
            fontFamily: 'monospace',
            marginTop: 2,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {filename}
        </div>
        {reflectorControl}
      </div>
      <button
        onClick={onRemove}
        title={t('phases.removePhase')}
        style={{
          background: 'transparent',
          border: 'none',
          color: C.textSecondary,
          cursor: 'pointer',
          fontSize: '13pt',
          lineHeight: 1,
          padding: '0 2px',
          flexShrink: 0,
        }}
        onMouseEnter={e => { e.currentTarget.style.color = C.red; }}
        onMouseLeave={e => { e.currentTarget.style.color = C.textSecondary; }}
      >
        ×
      </button>
    </div>
  );
}
