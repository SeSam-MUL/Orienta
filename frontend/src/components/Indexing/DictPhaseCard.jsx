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

const PC_COLORS = { good: '#c3e88d', ok: '#ffcb6b', bad: '#ff5370', unknown: '#666' };
const PC_BG     = { good: '#1a3a1a', ok: '#3a3a1a', bad: '#3a1a1a', unknown: '#2a2a2a' };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compute % difference between two PC vectors (Euclidean distance / √3 * 100). */
function pcDeltaPercent(pcA, pcB) {
  if (!pcA || !pcB || pcA.length < 3 || pcB.length < 3) return null;
  const dx = pcA[0] - pcB[0];
  const dy = pcA[1] - pcB[1];
  const dz = pcA[2] - pcB[2];
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  // Normalise to 0-100 % (√3 is the max Euclidean distance in unit cube)
  return (dist / Math.sqrt(3)) * 100;
}

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

/** Pick the best dictionary: lowest PC delta first, then finest (smallest) resolution. */
function selectBestDict(dicts, currentPc) {
  if (!dicts.length) return null;
  return dicts.reduce((best, d) => {
    const bestDelta = pcDeltaPercent(currentPc, best?.pc);
    const dDelta    = pcDeltaPercent(currentPc, d?.pc);
    // Prefer lower delta; if equal or both null, prefer finer resolution (smaller number)
    if (dDelta !== null && (bestDelta === null || dDelta < bestDelta)) return d;
    if (dDelta === bestDelta || (dDelta === null && bestDelta === null)) {
      const bestRes = best?.resolution ?? Infinity;
      const dRes    = d?.resolution    ?? Infinity;
      return dRes < bestRes ? d : best;
    }
    return best;
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DictPhaseCard({ phase, allFiles = [], currentPc = null, color, onRemove, onSelectDict, selectedDictPath, edsAvailable = false, edsStrengths = {}, onStrengthChange }) {
  const { t } = useTranslation('indexing');
  const formula      = (phase?.formula || '').toLowerCase();
  const formulaLabel = phase?.formula  || '—';
  const spaceGroup   = phase?.space_group   || '';
  const crystalSystem = phase?.crystal_system || '';

  // Collect related masters and dictionaries by formula
  const { masters, dicts, bestDict } = useMemo(() => {
    const ms = allFiles.filter(
      f => f.file_type === 'master' && (f.formula || '').toLowerCase() === formula
    );
    const ds = allFiles.filter(
      f => f.file_type === 'dictionary' && (f.formula || '').toLowerCase() === formula
    );
    return { masters: ms, dicts: ds, bestDict: selectBestDict(ds, currentPc) };
  }, [allFiles, formula, currentPc]);

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

      {/* ---- EDS chemistry-prior strength (only when EDS is available) ---- */}
      {edsAvailable && (
        <div style={{ padding: '2px 8px 4px 12px' }}>
          <label
            style={{ fontSize: '8pt', color: C.textSecondary }}
            title={t('hoverTips.phaseEdsStrength')}
          >
            {t('phases.edsStrength')}: {edsStrengths[phase.path] ?? 0}%
          </label>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={edsStrengths[phase.path] ?? 0}
            onChange={e => onStrengthChange && onStrengthChange(phase.path, Number(e.target.value))}
            title={t('hoverTips.phaseEdsStrength')}
            style={{ width: '100%' }}
          />
        </div>
      )}

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
            onClick={() => onSelectDict && onSelectDict(phase, d)}
            t={t}
          />
        ))
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

function DictRow({ entry, formula, currentPc, isSelected, onClick, t }) {
  const filename  = entry?.filename || entry?.path?.split(/[\\/]/).pop() || '—';
  const shortName = abbreviateFilename(filename, formula);
  const resolution = entry?.resolution != null ? `${entry.resolution}°` : '';
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
        <span style={{ color: 'var(--text-secondary)', fontSize: '7.5pt', flexShrink: 0 }}>
          {shape}
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
