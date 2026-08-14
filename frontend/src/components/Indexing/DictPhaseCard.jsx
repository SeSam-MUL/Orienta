/**
 * DictPhaseCard.jsx
 *
 * Hierarchical card for Dictionary indexing.
 * Shows: Phase header → Master Pattern → available Dictionaries with PC delta.
 *
 * Props:
 *   phase:     selected phase file object (the master pattern entry from discovery)
 *   allFiles:  array of all discovered files (masters + dictionaries)
 *   currentPc: [x, y, z] or null
 *   color:     border color string
 *   onRemove:  () => void
 */

import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import {
  mastersForPhase, dictsForPhase, masterPathForPhase,
  selectBestDict, dictShapeMatches, dictGeometryMatches, dictGeometryUnknown,
  pcDeltaPercent,
} from './dictionaryTargets';

const PC_COLORS = { good: '#c3e88d', ok: '#ffcb6b', bad: '#ff5370', unknown: '#666' };
const PC_BG     = { good: '#1a3a1a', ok: '#3a3a1a', bad: '#3a1a1a', unknown: '#2a2a2a' };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pcCategory(deltaPercent) {
  if (deltaPercent === null) return 'unknown';
  if (deltaPercent < 5)  return 'good';
  if (deltaPercent < 15) return 'ok';
  return 'bad';
}

/**
 * Strip the phase-formula prefix from a filename to get a compact label.
 * e.g. "Fe_20kV_128x156_2.0deg.h5" → "20kV_128x156_2.0deg"
 */
function abbreviateFilename(filename, formula) {
  if (!filename) return '';
  const name = filename.replace(/\.[^.]+$/, ''); // remove extension
  const prefix = formula ? formula.replace(/[^a-zA-Z0-9]/g, '') + '_' : '';
  return name.startsWith(prefix) ? name.slice(prefix.length) : name;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DictPhaseCard({ phase, allFiles = [], currentPc = null, color, onRemove, onSelectDict, selectedDictPath, onGenerateDict, detectorShape = null, geom = null }) {
  const { t } = useTranslation('indexing');
  const formulaLabel = phase?.formula  || '—';
  const spaceGroup   = phase?.space_group   || '';
  const crystalSystem = phase?.crystal_system || '';

  // Collect related masters and dictionaries by formula
  const { masters, dicts, bestDict, masterPath } = useMemo(() => {
    const ms = mastersForPhase(phase, allFiles);
    const ds = dictsForPhase(phase, allFiles);
    return {
      masters: ms,
      dicts: ds,
      // Same call the indexing request makes (resolveDictPathForPhase) — the
      // "chosen" badge below must name the file that will really be indexed.
      bestDict: selectBestDict(ds, currentPc, detectorShape, geom),
      masterPath: masterPathForPhase(phase, allFiles),
    };
  }, [allFiles, phase, currentPc, detectorShape, geom]);

  // Auto-select best dict on mount if none selected
  const activeDictPath = selectedDictPath || bestDict?.path || '';

  const masterEntry = masters[0] ?? phase; // fall back to the phase itself if needed

  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 4,
        background: 'rgba(255,255,255,0.04)',
        overflow: 'hidden',
      }}
    >
      {/* ---- Header ---- */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 8px',
          flexWrap: 'wrap',
        }}
      >
        <span style={{ color, fontWeight: 700, fontSize: '10pt' }}>{formulaLabel}</span>
        {spaceGroup && (
          <span style={{ color: '#82aaff', fontSize: '9pt' }}>{spaceGroup}</span>
        )}
        {crystalSystem && (
          <span style={{ color: C.textSecondary, fontSize: '9pt' }}>{crystalSystem}</span>
        )}
        <span
          style={{
            marginLeft: 'auto',
            color: C.textSecondary,
            fontSize: '8.5pt',
            whiteSpace: 'nowrap',
          }}
        >
          {t('dictCard.masterCount', { masters: masters.length, dicts: dicts.length })}
        </span>
        <button
          onClick={onRemove}
          title={t('dictCard.removePhase')}
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

      {/* ---- Master Pattern row ---- */}
      <MasterRow entry={masterEntry} t={t} />

      {/* PC not set warning */}
      {!currentPc && dicts.length > 0 && (
        <div
          style={{
            padding: '3px 8px 3px 20px',
            color: C.yellow,
            fontSize: '8pt',
          }}
        >
          {t('dictCard.pcNotSet')}
        </div>
      )}

      {/* ---- Dictionary rows ---- */}
      {dicts.length === 0 ? (
        <div
          style={{
            padding: '5px 8px 6px 24px',
            color: C.yellow,
            fontSize: '8.5pt',
          }}
        >
          {t('dictCard.noDictGenerated')}
        </div>
      ) : (
        dicts.map((d, i) => (
          <DictRow
            key={i}
            entry={d}
            formula={formulaLabel}
            currentPc={currentPc}
            isSelected={d.path === activeDictPath}
            fitsDetector={dictShapeMatches(d, detectorShape)}
            fitsGeometry={dictGeometryMatches(d, geom)}
            tiltUnknown={dictGeometryUnknown(d, geom)}
            neededShape={detectorShape}
            neededTilt={geom?.detectorTilt}
            neededSampleTilt={geom?.sampleTilt}
            onClick={() => onSelectDict && onSelectDict(phase, d)}
            t={t}
          />
        ))
      )}

      {/* ---- Generate a dictionary FOR THIS PHASE ----
          Per-phase, because a single page-level button can only ever seed one
          master and silently picked the first phase in the list. */}
      {onGenerateDict && (
        <GenerateDictRow
          disabled={!masterPath}
          onClick={() => onGenerateDict(phase, masterPath)}
          t={t}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// GenerateDictRow — per-phase "build one for my detector" action
// ---------------------------------------------------------------------------

function GenerateDictRow({ disabled, onClick, t }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 8px 5px 20px',
        borderTop: `1px solid rgba(255,255,255,0.05)`,
      }}
    >
      <button
        type="button"
        onClick={disabled ? undefined : onClick}
        disabled={disabled}
        title={disabled ? t('dictCard.generateNoMasterTip') : t('dictCard.generateForPhaseTip')}
        style={{
          background: 'transparent',
          border: `1px dashed ${disabled ? C.border : '#50fa7b66'}`,
          borderRadius: 3,
          color: disabled ? C.textSecondary : '#50fa7b',
          fontSize: '8pt',
          padding: '2px 8px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.5 : 1,
        }}
      >
        {t('dictCard.generateForPhase')}
      </button>
      {disabled && (
        <span style={{ color: C.textSecondary, fontSize: '7.5pt' }}>
          {t('dictCard.generateNoMaster')}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MasterRow
// ---------------------------------------------------------------------------

function MasterRow({ entry, t }) {
  const filename = entry?.filename || entry?.path?.split(/[\\/]/).pop() || '—';
  const energy   = entry?.energy   ? `${entry.energy} kV` : '';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '3px 8px 3px 12px',
        background: 'rgba(189,147,249,0.07)',
        borderTop: `1px solid rgba(189,147,249,0.15)`,
      }}
    >
      <Tag color="#bd93f9" bg="rgba(189,147,249,0.15)">{t('dictCard.tagMaster')}</Tag>
      <span
        style={{
          color: 'var(--text-primary)',
          fontSize: '8.5pt',
          fontFamily: 'monospace',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          flex: 1,
          minWidth: 0,
        }}
      >
        {filename}
      </span>
      {energy && (
        <span style={{ color: 'var(--text-secondary)', fontSize: '8pt', flexShrink: 0 }}>
          {energy}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DictRow
// ---------------------------------------------------------------------------

function DictRow({ entry, formula, currentPc, isSelected, onClick, t, fitsDetector = true, fitsGeometry = true, tiltUnknown = false, neededShape = null, neededTilt = null, neededSampleTilt = null }) {
  const filename  = entry?.filename || entry?.path?.split(/[\\/]/).pop() || '—';
  const shortName = abbreviateFilename(filename, formula);
  // Discovery reports `resolution_deg`; only older/inline entries use
  // `resolution`. Reading just the latter left this column permanently blank.
  const resolutionValue = entry?.resolution ?? entry?.resolution_deg;
  const resolution = resolutionValue != null ? `${resolutionValue}°` : '';
  const shape      = entry?.detector_shape
    ? `${entry.detector_shape[0]}×${entry.detector_shape[1]}`
    : '';

  const delta    = pcDeltaPercent(currentPc, entry?.pc);
  const cat      = pcCategory(delta);
  const deltaStr = delta !== null ? `${delta.toFixed(1)}%` : '—';

  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '3px 8px 3px 20px',
        borderTop: `1px solid rgba(255,255,255,0.05)`,
        background: isSelected ? 'rgba(195,232,141,0.08)' : 'transparent',
        cursor: 'pointer',
      }}
    >
      <Tag color="#50fa7b" bg="rgba(80,250,123,0.12)">{t('dictCard.tagDict')}</Tag>
      <span
        style={{
          color: 'var(--text-primary)',
          fontSize: '8pt',
          fontFamily: 'monospace',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          flex: 1,
          minWidth: 0,
        }}
        title={filename}
      >
        {shortName}
      </span>

      {/* PC delta badge */}
      <span
        style={{
          fontSize: '7.5pt',
          color: PC_COLORS[cat],
          background: PC_BG[cat],
          border: `1px solid ${PC_COLORS[cat]}44`,
          borderRadius: 3,
          padding: '1px 5px',
          flexShrink: 0,
        }}
        title={t('dictCard.pcDeviation')}
      >
        {deltaStr}
      </span>

      {resolution && (
        <span style={{ color: 'var(--text-secondary)', fontSize: '7.5pt', flexShrink: 0 }}>
          {resolution}
        </span>
      )}
      {shape && (
        <span
          style={{
            // A dictionary built for a different detector cannot be used at
            // all — the indexer rejects it. Say so here instead of letting the
            // run die halfway through.
            color: fitsDetector ? 'var(--text-secondary)' : PC_COLORS.bad,
            fontSize: '7.5pt',
            flexShrink: 0,
            fontWeight: fitsDetector ? 400 : 700,
          }}
          title={fitsDetector ? undefined : t('dictCard.shapeMismatch', {
            got: shape,
            needed: neededShape ? `${neededShape[0]}×${neededShape[1]}` : '?',
          })}
        >
          {shape}{fitsDetector ? '' : ' ⚠'}
        </span>
      )}
      {!fitsGeometry && (
        <span
          style={{
            color: PC_COLORS.bad, fontSize: '7.5pt', flexShrink: 0, fontWeight: 700,
          }}
          title={t('dictCard.tiltMismatch', {
            gotSample: entry?.sample_tilt != null ? `${Number(entry.sample_tilt).toFixed(1)}°` : '?',
            gotCam: entry?.detector_tilt != null ? `${Number(entry.detector_tilt).toFixed(2)}°` : '?',
            needSample: neededSampleTilt != null ? `${Number(neededSampleTilt).toFixed(1)}°` : '?',
            needCam: neededTilt != null ? `${Number(neededTilt).toFixed(2)}°` : '?',
          })}
        >
          {t('dictCard.tiltBadge')} ⚠
        </span>
      )}
      {tiltUnknown && fitsGeometry && (
        <span
          style={{ color: PC_COLORS.ok, fontSize: '7.5pt', flexShrink: 0 }}
          title={t('dictCard.tiltUnknownTip', {
            needed: neededTilt != null ? `${Number(neededTilt).toFixed(2)}°` : '?',
          })}
        >
          {t('dictCard.tiltUnknownBadge')}
        </span>
      )}
      {isSelected && (
        <span style={{ color: '#c3e88d', fontSize: '7.5pt', flexShrink: 0 }}>
          {t('dictCard.chosen')}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tag — small colored pill label
// ---------------------------------------------------------------------------

function Tag({ children, color, bg }) {
  return (
    <span
      style={{
        fontSize: '7.5pt',
        color,
        background: bg,
        border: `1px solid ${color}44`,
        borderRadius: 3,
        padding: '1px 5px',
        flexShrink: 0,
        fontWeight: 600,
      }}
    >
      {children}
    </span>
  );
}
