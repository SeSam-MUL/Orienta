/**
 * Ranked candidate-phase list — shows local library matches sorted by score,
 * with badges for in-library SHT availability and "expected for preset".
 */

import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import { crystalHintApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import InfoTooltip from '../common/InfoTooltip';

const FILTER_LABEL_KEYS = {
  expected: 'candidate.filterExpected',
  sht: 'candidate.filterSht',
  all: 'candidate.filterAll',
};

const S = {
  card: {
    padding: 12,
    background: colors.bgSecondary,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
  },
  title: {
    fontSize: 13,
    fontWeight: 600,
    color: colors.accent,
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  item: {
    padding: 8,
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  itemTop: { display: 'flex', alignItems: 'baseline', gap: 8, flex: 1 },
  rank: {
    fontFamily: 'monospace',
    fontSize: 11,
    color: colors.textSecondary,
    minWidth: 20,
  },
  formula: {
    fontWeight: 600,
    fontSize: 13,
    color: colors.text,
  },
  meta: {
    fontSize: 11,
    color: colors.textSecondary,
  },
  scoreCol: {
    textAlign: 'right',
    minWidth: 56,
  },
  score: {
    fontSize: 13,
    fontWeight: 600,
    color: colors.accent,
    fontFamily: 'monospace',
  },
  badge: (color) => ({
    display: 'inline-block',
    padding: '1px 5px',
    fontSize: 9,
    background: `${color}22`,
    color: color,
    border: `1px solid ${color}44`,
    borderRadius: 3,
    marginLeft: 4,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    fontWeight: 600,
  }),
  empty: {
    padding: 16,
    textAlign: 'center',
    fontSize: 12,
    color: colors.textSecondary,
  },
  warning: {
    fontSize: 11,
    color: colors.orange,
    background: `${colors.orange}11`,
    padding: '4px 8px',
    borderRadius: 4,
    marginTop: 6,
  },
};

export default function CandidateList({
  matches = [],
  presetEntry,
  warnings = [],
  externalMatches = [],
  externalLoading = false,
  onNavigate,
}) {
  const { t } = useTranslation('crystalhint');
  const setPendingSimXtal = useDataStore(s => s.setPendingSimXtal);

  const handleOpenInSimulation = (cifPath) => {
    // Pre-fill the Simulation page's crystal-file field with this CIF.
    // The Simulation page reads `pendingSimXtal` on mount/activation
    // and clears it after consuming.
    setPendingSimXtal(cifPath);
    if (onNavigate) {
      onNavigate('simulation');
    }
  };
  // Per-match CIF download state, keyed by `${source}-${structure_id}`.
  //   undefined  → not yet requested
  //   "loading"  → request in flight
  //   {success, error, path?, warnings?} → completed
  const [cifState, setCifState] = useState({});
  const [filter, setFilter] = useState('expected');  // 'expected' | 'sht' | 'all'
  const [showAll, setShowAll] = useState(false);

  // Reorder + filter. Default ordering: expected first, then SHT-available,
  // then by score. The backend already boosts expected-preset scores so
  // sorting by score alone preserves the intent.
  const filteredMatches = useMemo(() => {
    let list = matches.slice();
    if (filter === 'expected') {
      list = list.filter(m => m.expected_in_preset);
    } else if (filter === 'sht') {
      list = list.filter(m => m.has_sht);
    }
    return list;
  }, [matches, filter]);

  // If "Expected only" returns nothing (no preset, or no matches match),
  // silently fall through to showing all matches so the user still sees
  // something useful. The filter chips stay where they are.
  const displayList = useMemo(() => {
    if (filteredMatches.length === 0 && filter !== 'all') {
      return matches;
    }
    return filteredMatches;
  }, [filteredMatches, filter, matches]);
  const visibleList = showAll ? displayList : displayList.slice(0, 5);

  // Detect a "tied top scores" situation — when the top N candidates all
  // have the same score AND the lattice estimator didn't help discriminate
  // (lattice_distance all 0), the ranking among them is essentially
  // alphabetical-by-n_atoms. The user should know that the score doesn't
  // mean "Al fits best" — it means "all of these are equally plausible
  // given the filters we have."
  const topTied = useMemo(() => {
    if (visibleList.length < 2) return 0;
    const topScore = visibleList[0].score;
    const topDist = visibleList[0].lattice_distance;
    let n = 0;
    for (const m of visibleList) {
      if (Math.abs(m.score - topScore) < 0.001 && Math.abs(m.lattice_distance - topDist) < 0.01) {
        n++;
      } else break;
    }
    return n;
  }, [visibleList]);

  const handleDownloadCif = async (m) => {
    const key = `${m.source}-${m.structure_id}`;
    setCifState(prev => ({ ...prev, [key]: 'loading' }));
    try {
      const res = await crystalHintApi.downloadCif({
        source: m.source,
        structureId: m.structure_id,
        save: true,
        overwrite: false,
      });
      setCifState(prev => ({ ...prev, [key]: { ...res.data } }));
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || t('candidate.downloadFailedDefault');
      setCifState(prev => ({ ...prev, [key]: { success: false, error: msg } }));
    }
  };
  return (
    <div style={S.card}>
      <div style={S.title}>
        <span style={{ display: 'inline-flex', alignItems: 'center' }}>
          {t('candidate.title')}
          <InfoTooltip>
            <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
              {t('candidate.scoreInfoTitle')}
            </div>
            <p style={{ margin: '4px 0' }}>
              {t('candidate.scoreInfoIntroPre')}<b>{t('candidate.scoreInfoIntroNot')}</b>{t('candidate.scoreInfoIntroPost')}
            </p>
            <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
              <li><b>{t('candidate.scoreInfoChemistryLabel')}</b> — {t('candidate.scoreInfoChemistry')}</li>
              <li><b>{t('candidate.scoreInfoCrystalSystemLabel')}</b> — {t('candidate.scoreInfoCrystalSystem')}</li>
              <li><b>{t('candidate.scoreInfoLatticeLabel')}</b> — {t('candidate.scoreInfoLattice')}</li>
              <li><b>{t('candidate.scoreInfoBoostLabel')}</b> {t('candidate.scoreInfoBoost')}</li>
            </ul>
            <p style={{ margin: '6px 0', color: colors.orange }}>
              {t('candidate.scoreInfoTiePre')}<b>{t('candidate.scoreInfoTieBold')}</b>{t('candidate.scoreInfoTiePost')}
            </p>
            <p style={{ margin: '4px 0', fontSize: 10, color: colors.textSecondary }}>
              {t('candidate.scoreInfoTieRanking')}
            </p>
          </InfoTooltip>
        </span>
        <span style={{ fontSize: 11, fontWeight: 400, color: colors.textSecondary }}>
          {t('candidate.matchCount', {
            shown: displayList.length,
            total: matches.length,
            matchWord: matches.length !== 1 ? t('candidate.matchWordPlural') : t('candidate.matchWordSingular'),
          })}
        </span>
      </div>

      {/* Tied-top warning: if N candidates share the same score+distance,
          surface that the score isn't a fit ranking. */}
      {topTied >= 3 && (
        <div style={{
          marginBottom: 8,
          padding: '6px 8px',
          fontSize: 11,
          background: `${colors.orange}15`,
          border: `1px solid ${colors.orange}55`,
          borderRadius: 4,
          color: colors.text,
        }}>
          {t('candidate.tiedWarningPre')}<b>{t('candidate.tiedWarningBold', { count: topTied })}</b>{t('candidate.tiedWarningPost', { score: visibleList[0].score.toFixed(2) })}
        </div>
      )}

      {/* Filter chips */}
      {matches.length > 0 && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }} title={t('candidate.filterChipsTooltip')}>
          {['expected', 'sht', 'all'].map(k => (
            <button
              key={k}
              onClick={() => { setFilter(k); setShowAll(false); }}
              title={t('candidate.filterChipsTooltip')}
              style={{
                padding: '3px 8px',
                fontSize: 10,
                background: filter === k ? colors.accent : 'transparent',
                color: filter === k ? '#fff' : colors.textSecondary,
                border: `1px solid ${filter === k ? colors.accent : colors.border}`,
                borderRadius: 12,
                cursor: 'pointer',
              }}
            >
              {t(FILTER_LABEL_KEYS[k])}
            </button>
          ))}
          {filter === 'expected' && filteredMatches.length === 0 && matches.length > 0 && (
            <span style={{ fontSize: 10, color: colors.orange, alignSelf: 'center' }}>
              {t('candidate.noExpectedMatch')}
            </span>
          )}
        </div>
      )}

      {matches.length === 0 && (
        <div style={S.empty}>
          {t('candidate.emptyPre')}<strong>{t('candidate.emptyBold')}</strong>{t('candidate.emptyPost')}
        </div>
      )}

      <div style={S.list}>
        {visibleList.map((m, idx) => (
          <div key={m.key} style={S.item}>
            <div style={S.itemTop}>
              <div style={S.rank}>#{idx + 1}</div>
              <div style={{ flex: 1 }}>
                <span style={S.formula} title={m.formula || m.key}>
                  {m.display_formula || m.formula || m.key}
                </span>
                {m.has_sht && <span style={S.badge(colors.green)}>{t('candidate.shtBadge')}</span>}
                {m.expected_in_preset && (
                  <span style={S.badge(colors.accent)}>{t('candidate.expectedBadge')}</span>
                )}
                {!m.system_match && (
                  <InfoTooltip inline content={(
                    <>
                      <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
                        {t('candidate.offSystemTitle')}
                      </div>
                      <p style={{ margin: '4px 0' }}>
                        {t('candidate.offSystemBody')}
                      </p>
                      <p style={{ margin: '4px 0', fontSize: 10,
                        color: colors.textSecondary }}>
                        {t('candidate.offSystemNote')}
                      </p>
                    </>
                  )}>
                    <span style={S.badge(colors.orange)}>{t('candidate.offSystemBadge')}</span>
                  </InfoTooltip>
                )}
                <div style={S.meta}>
                  {m.space_group || `SG ${m.space_group_number || '?'}`}
                  {' · '}
                  {m.crystal_system}
                  {m.lattice_a_A && ` · a=${m.lattice_a_A.toFixed(2)}Å`}
                  {' · '}
                  {(m.elements || []).join(', ')}
                </div>
                {m.degenerate_with && m.degenerate_with.length > 0 && (
                  <InfoTooltip inline content={(
                    <>
                      <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
                        {t('candidate.degenerateTitle')}
                      </div>
                      <p style={{ margin: '4px 0' }}>
                        {t('candidate.degenerateBody')}
                      </p>
                      <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
                        {m.degenerate_with.map((n, i) => (
                          <li key={i}>{n}</li>
                        ))}
                      </ul>
                    </>
                  )}>
                    <div style={{
                      fontSize: 11, color: colors.textSecondary,
                      marginTop: 2, cursor: 'help',
                      borderBottom: `1px dotted ${colors.border}`,
                      display: 'inline-block',
                    }}>
                      {t('candidate.alsoMatches', { count: m.degenerate_with.length })}
                      {m.degenerate_with.slice(0, 2).join(', ')}
                      {m.degenerate_with.length > 2 && ' …'}
                    </div>
                  </InfoTooltip>
                )}
              </div>
            </div>
            <div style={S.scoreCol}>
              <div style={S.score}>{Number(m.score).toFixed(2)}</div>
              {/* Per-phase fit breakdown — symmetry-NCC fit + d-spacing fit.
                  Shows WHY one phase ranks above another. Tooltip on
                  hover explains what the chips mean. */}
              {(m.symmetry_fit != null || m.dspacing_fit != null || m.chemistry_fit != null) && (
                <InfoTooltip inline content={(
                  <>
                    <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
                      {t('candidate.fitTitle')}
                    </div>
                    <p style={{ margin: '4px 0' }}>
                      {t('candidate.fitIntro')}
                    </p>
                    <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
                      <li>
                        <b style={{ color: colors.accent }}>{t('candidate.fitSymPre')}</b>{t('candidate.fitSymBody')}
                      </li>
                      <li>
                        <b style={{ color: colors.accent }}>{t('candidate.fitDspPre')}</b>{t('candidate.fitDspBody')}
                      </li>
                    </ul>
                    <p style={{ margin: '4px 0', fontSize: 10, color: colors.textSecondary }}>
                      {t('candidate.fitNote')}
                    </p>
                  </>
                )}>
                  <div style={{
                    display: 'flex', gap: 4, marginTop: 2, fontSize: 9,
                    cursor: 'help',
                  }}>
                    {m.symmetry_fit != null && (
                      <span style={{
                        padding: '1px 4px',
                        background: colors.bg,
                        border: `1px solid ${colors.border}`,
                        borderRadius: 3,
                        color: m.symmetry_fit >= 0.8 ? colors.green
                             : m.symmetry_fit >= 0.5 ? colors.accent
                             : colors.orange,
                        fontFamily: 'monospace',
                      }}>
                        sym {m.symmetry_fit.toFixed(2)}
                      </span>
                    )}
                    {m.dspacing_fit != null && (
                      <span style={{
                        padding: '1px 4px',
                        background: colors.bg,
                        border: `1px solid ${colors.border}`,
                        borderRadius: 3,
                        color: m.dspacing_fit >= 0.9 ? colors.green
                             : m.dspacing_fit >= 0.7 ? colors.accent
                             : colors.orange,
                        fontFamily: 'monospace',
                      }}>
                        dsp {m.dspacing_fit.toFixed(2)}
                      </span>
                    )}
                    {m.chemistry_fit != null && (
                      <span style={{
                        padding: '1px 4px',
                        background: colors.bg,
                        border: `1px solid ${colors.border}`,
                        borderRadius: 3,
                        color: m.chemistry_fit >= 0.8 ? colors.green
                             : m.chemistry_fit >= 0.5 ? colors.accent
                             : colors.orange,
                        fontFamily: 'monospace',
                      }}>
                        chem {m.chemistry_fit.toFixed(2)}
                      </span>
                    )}
                  </div>
                </InfoTooltip>
              )}
              <div style={S.meta}>
                {m.lattice_distance > 0 && t('candidate.deltaA', { pct: (m.lattice_distance * 100).toFixed(0) })}
              </div>
            </div>
          </div>
        ))}
      </div>

      {displayList.length > 5 && (
        <button
          onClick={() => setShowAll(s => !s)}
          title={showAll ? t('hoverTips.candidateShowFewer') : t('hoverTips.candidateShowAll')}
          style={{
            marginTop: 8, padding: '4px 10px', fontSize: 11,
            background: 'transparent', color: colors.accent,
            border: `1px solid ${colors.border}`, borderRadius: 4,
            cursor: 'pointer', width: '100%',
          }}
        >
          {showAll
            ? t('candidate.showFewer')
            : t('candidate.showAll', { count: displayList.length })}
        </button>
      )}

      {warnings && warnings.length > 0 && warnings.map((w, i) => (
        <div key={i} style={S.warning}>⚠ {w}</div>
      ))}

      {/* External matches — candidate phases NOT in the local library that
          match the symmetry + chemistry + lattice criteria. Sourced from
          BOTH the Crystallography Open Database (COD) and Materials Project
          (MP); each row carries a source badge. */}
      <div style={{ marginTop: 16, paddingTop: 12, borderTop: `1px dashed ${colors.border}` }}>
        <div style={{ ...S.title, marginBottom: 6, fontSize: 12 }}>
          <span>{t('candidate.externalTitle')}</span>
          {externalLoading && (
            <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('candidate.externalSearching')}</span>
          )}
          {!externalLoading && externalMatches.length > 0 && (
            <span style={{ fontSize: 11, color: colors.textSecondary }}>
              {t('candidate.externalCount', { count: externalMatches.length })}
            </span>
          )}
        </div>
        {!externalLoading && externalMatches.length === 0 && (
          <div style={{ fontSize: 11, color: colors.textSecondary, padding: '4px 0' }}>
            {t('candidate.externalEmpty')}
          </div>
        )}
        <div style={S.list}>
          {externalMatches.slice(0, 10).map((m, idx) => {
            const key = `${m.source}-${m.structure_id}`;
            const dlState = cifState[key];
            const isLoading = dlState === 'loading';
            const isDone = dlState && typeof dlState === 'object';
            return (
              <div key={key} style={S.item}>
                <div style={S.itemTop}>
                  <div style={S.rank}>#{idx + 1}</div>
                  <div style={{ flex: 1 }}>
                    <span style={S.formula}>{m.formula || m.structure_id}</span>
                    <span style={S.badge(colors.orange)}>{m.source}</span>
                    {/* "in library" badge: this external phase is structurally
                        already in the local SHT/CIF library — informational,
                        ranked below novel phases. */}
                    {m.in_local_library && (
                      <InfoTooltip inline content={(
                        <p style={{ margin: 0 }}>
                          {t('candidate.inLibraryBody')}
                        </p>
                      )}>
                        <span style={{
                          marginLeft: 4, padding: '1px 4px', fontSize: 9,
                          background: colors.bg,
                          border: `1px solid ${colors.border}`,
                          borderRadius: 3, color: colors.textSecondary,
                          cursor: 'help',
                        }}>{t('candidate.inLibraryBadge')}</span>
                      </InfoTooltip>
                    )}
                    {/* Fit chips on external matches too — same colour
                        coding as local. Lets the user see at a glance
                        which external candidate would actually fit. */}
                    {m.symmetry_fit != null && (
                      <span style={{
                        marginLeft: 4, padding: '1px 4px', fontSize: 9,
                        background: colors.bg,
                        border: `1px solid ${colors.border}`,
                        borderRadius: 3,
                        color: m.symmetry_fit >= 0.8 ? colors.green
                             : m.symmetry_fit >= 0.5 ? colors.accent
                             : colors.orange,
                        fontFamily: 'monospace',
                      }}>sym {m.symmetry_fit.toFixed(2)}</span>
                    )}
                    {m.dspacing_fit != null && (
                      <span style={{
                        marginLeft: 2, padding: '1px 4px', fontSize: 9,
                        background: colors.bg,
                        border: `1px solid ${colors.border}`,
                        borderRadius: 3,
                        color: m.dspacing_fit >= 0.9 ? colors.green
                             : m.dspacing_fit >= 0.7 ? colors.accent
                             : colors.orange,
                        fontFamily: 'monospace',
                      }}>dsp {m.dspacing_fit.toFixed(2)}</span>
                    )}
                    <span style={{ marginLeft: 6, fontSize: 11, color: colors.textSecondary }}>
                      {m.space_group || `SG ${m.space_group_number || '?'}`}
                      {m.a_A && ` · a=${m.a_A.toFixed(2)}Å`}
                    </span>
                    <div style={S.meta}>
                      {m.title || ''}
                      {m.cif_url && (
                        <>
                          {' · '}
                          <a
                            href={m.cif_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            title={t('candidate.viewCifTooltip')}
                            style={{ color: colors.accent, textDecoration: 'underline' }}
                          >
                            {t('candidate.viewCif', { id: m.structure_id })}
                          </a>
                        </>
                      )}
                    </div>
                    {isDone && dlState.success && (
                      <div style={{ fontSize: 10, color: colors.green, marginTop: 4 }}>
                        {t('candidate.savedToLibrary')}
                        {dlState.local_path && (
                          <span style={{ opacity: 0.7 }}>
                            : {dlState.local_path.split(/[\\/]/).pop()}
                          </span>
                        )}
                        {dlState.warnings?.length > 0 && (
                          <div style={{ color: colors.orange, marginTop: 2 }}>
                            {dlState.warnings.filter(w => !w.startsWith('Saved')).join(' · ')}
                          </div>
                        )}
                        <div style={{ color: colors.textSecondary, marginTop: 2 }}>
                          {t('candidate.nextStep')}
                        </div>
                        {onNavigate && dlState.local_path && (
                          <button
                            onClick={() => handleOpenInSimulation(dlState.local_path)}
                            title={t('candidate.openInSimulationTooltip')}
                            style={{
                              marginTop: 6,
                              background: colors.accent,
                              color: '#fff', border: 'none', borderRadius: 4,
                              padding: '4px 10px', fontSize: 10, fontWeight: 600,
                              cursor: 'pointer',
                            }}
                          >
                            {t('candidate.openInSimulation')}
                          </button>
                        )}
                      </div>
                    )}
                    {isDone && !dlState.success && (
                      <div style={{ fontSize: 10, color: colors.red, marginTop: 4 }}>
                        {t('candidate.downloadFailed', { msg: dlState.error || t('candidate.downloadFailedDefault') })}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleDownloadCif(m)}
                    disabled={isLoading || (isDone && dlState.success)}
                    title={t('candidate.downloadCifTooltip')}
                    style={{
                      background: isDone && dlState.success ? `${colors.green}22` : colors.accent,
                      color: isDone && dlState.success ? colors.green : '#fff',
                      border: 'none', borderRadius: 4,
                      padding: '4px 10px', fontSize: 10, fontWeight: 600,
                      cursor: isLoading || (isDone && dlState.success) ? 'default' : 'pointer',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {isLoading ? t('candidate.downloading') : isDone && dlState.success ? t('candidate.savedButton') : t('candidate.downloadCif')}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {presetEntry?.expected_phases?.length > 0 && (
        <div style={{ marginTop: 12, fontSize: 11, color: colors.textSecondary }}>
          {t('candidate.expectedNotMatchedPre')}<strong>{presetEntry.label}</strong>{t('candidate.expectedNotMatchedPost')}
          {presetEntry.expected_phases
            .filter(p => !matches.find(m =>
              (m.formula || m.key).toLowerCase().includes(p.name.toLowerCase().replace(/^(alpha|beta|gamma|theta)-/, ''))
            ))
            .map(p => p.name)
            .slice(0, 6)
            .join(', ') || t('candidate.expectedNotMatchedNone')}
        </div>
      )}
    </div>
  );
}
