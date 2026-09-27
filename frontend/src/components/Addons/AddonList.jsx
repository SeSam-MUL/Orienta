/**
 * One row per add-on the backend can see — including the ones it refused.
 *
 * A rejected manifest is SHOWN as rejected, never dropped: an add-on the user
 * installed and that simply does not appear is indistinguishable from one that
 * was never installed, and the row that most needs to be seen would be the one
 * missing. The backend already answers this way; the list only has to render
 * it.
 *
 * Every reason an add-on cannot be switched on is written next to the switch.
 * A disabled control with no explanation is read as a broken page.
 */
import { useTranslation } from 'react-i18next';
import { Button, colors } from '../../theme/components';
import { consentIsStale } from './addonIdentity';

export function blockedReason(addon, t) {
  // ONE function, so the badge and the disabled switch cannot disagree about
  // whether an add-on may be enabled — the "two sources, one updated" defect
  // this repo keeps meeting.
  //
  // Returns a localised HEADLINE and the server's own sentence beneath it.
  // The sentence is built in English by the backend and names paths, add-on
  // names and version ranges; translating it is not possible from here, but
  // leaving it as the only thing said made a German row read
  // "…Installiert in: … · another add-on also declares addon.bc_gmm". The
  // kind of problem is now said in the user's language and the specifics
  // stay verbatim — the same shape this repo settled on for backend messages
  // elsewhere: codes for the statement, prose as the fallback.
  if (!addon) return null;
  if (addon.error) {
    return { title: t('list.rejected'), detail: addon.error };
  }
  if (addon.conflict) {
    return { title: t('list.conflict'), detail: addon.conflict };
  }
  if (addon.compatible === false) {
    return { title: t('list.incompatible'), detail: addon.compatibility || '' };
  }
  return null;
}

function Row({ addon, onToggle, onRunAnalysis, busy }) {
  const { t } = useTranslation('addons');
  const blocked = blockedReason(addon, t);
  const runtimeOff = addon.disabled_by === 'runtime';
  const stale = consentIsStale(addon);
  // A backend that predates the field would otherwise render "3 of  failures"
  // — i18next drops a missing value rather than leaking {{limit}}, so the
  // sentence loses its second number and nobody can tell why.
  const limit = addon.crash_limit ?? '?';

  return (
    <div
      data-testid="addon-row"
      style={{
        border: `1px solid ${colors.border}`,
        borderRadius: 6,
        padding: 12,
        marginBottom: 10,
        opacity: addon.error ? 0.85 : 1,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        {/* The author's own words, rendered as given. */}
        <strong>{addon.display_name}</strong>
        {addon.version && (
          <span style={{ color: colors.textSecondary, fontSize: '9pt' }}>
            {addon.version}
          </span>
        )}
        {addon.enabled && (
          <span style={{ color: colors.success || colors.accent, fontSize: '9pt' }}>
            {t('list.enabled')}
          </span>
        )}
        <span style={{ flex: 1 }} />
        {!addon.error && (
          <Button
            small
            disabled={busy || (!addon.enabled && Boolean(blocked))}
            onClick={() => onToggle(addon)}
          >
            {addon.enabled ? t('list.disable') : t('list.enable')}
          </Button>
        )}
      </div>

      <div style={{ color: colors.textSecondary, fontSize: '9pt', marginTop: 6 }}>
        {t('list.thirdParty')}
      </div>

      {(addon.authors || []).length > 0 && (
        <div style={{ fontSize: '9pt', marginTop: 4 }}>
          {t('list.authors')}: {(addon.authors || []).join(', ')}
        </div>
      )}
      <div style={{ fontSize: '9pt' }}>
        DOI: {addon.doi || t('list.noDoi')}
      </div>
      {addon.source_path && (
        <div style={{ fontSize: '9pt', wordBreak: 'break-all' }}>
          {t('list.source')}: {addon.source_path}
        </div>
      )}

      {blocked && (
        <div role="status" style={{ color: colors.warning || colors.accent,
                                    fontSize: '9pt', marginTop: 6 }}>
          <div>{blocked.title}</div>
          {blocked.detail && (
            <div style={{ color: colors.textSecondary }}>{blocked.detail}</div>
          )}
        </div>
      )}
      {stale && (
        // The add-on on disk is not the one the recorded decision was about,
        // so the row says which fields moved and the switch will ask again.
        <div role="status" style={{ color: colors.warning || colors.accent,
                                    fontSize: '9pt', marginTop: 6 }}>
          {t('list.changedSinceConsent', { version: addon.known_version || '—',
                                           doi: addon.known_doi || '—' })}
        </div>
      )}

      {/* Without this the switch is simply off, for no visible reason, and
          the user's own decision is what they will assume. */}
      {runtimeOff && (
        <div role="status" style={{ color: colors.warning || colors.accent,
                                    fontSize: '9pt', marginTop: 6 }}>
          {t('list.disabledByRuntime', { count: addon.crashes,
                                         limit })}
        </div>
      )}
      {!runtimeOff && addon.crashes > 0 && (
        <div style={{ color: colors.textSecondary, fontSize: '9pt', marginTop: 6 }}>
          {t('list.failureCount', { count: addon.crashes, limit })}
        </div>
      )}

      {/* What the add-on actually DOES. A row that lists an author, a DOI and
          a switch, and never says what running it would compute, asks for a
          decision without giving the one fact it turns on. Only offered while
          the add-on is enabled: an analysis that cannot be started should not
          look startable. */}
      {addon.enabled && (addon.analyses || []).length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ color: colors.textSecondary, fontSize: '9pt' }}>
            {t('list.analyses')}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
            {(addon.analyses || []).map((a) => (
              <Button key={a.key} small
                      onClick={() => onRunAnalysis?.(addon, a)}>
                {/* The author's own label, verbatim. */}
                {a.label || a.key}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AddonList({ addons = [], onToggle, onRunAnalysis,
                                    busyName }) {
  return (
    <div>
      {addons.map((addon, i) => (
        <Row
          // A rejected row has no name, and two of them are possible, so the
          // key falls back to the source path and then to the index.
          key={addon.name || addon.source_path || i}
          addon={addon}
          onToggle={onToggle}
          onRunAnalysis={onRunAnalysis}
          busy={busyName === addon.name}
        />
      ))}
    </div>
  );
}
