/**
 * "Group: Al systems" in the toolbar — an INDICATOR, not a picker.
 *
 * What it replaces was a dropdown that both chose the active collection and
 * displayed it. Choosing moved to the library, where the group lives and
 * where you can see what is in it; this keeps the other half, and that half
 * is the one worth keeping: it is the only thing on an Indexing or a Phase
 * Test screen that says YOUR PHASE CHOICE IS NARROWED. Remove it and a run
 * over five phases instead of thirty-six looks exactly like a run over
 * thirty-six.
 *
 * IT CARRIES THE FAILURE, and that is why it was hardened rather than
 * dropped. The regression it survived was precisely this line lying: `GET /`
 * began sending ids where the filter compares names, the filter found
 * nothing and widened the run to the whole library, and this chip went on
 * naming a group that was not being applied. Now an active reference that
 * resolves to nothing says so here, in red, and the run is blocked with the
 * same reason.
 *
 * THE WORD IS "GROUP". Sebastian did not like "collection", and with the
 * manager dialog gone the word has no business surviving in the interface.
 * The routes and the files on disk keep their names; this is about what a
 * person reads.
 */
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import useCollectionStore from '../../stores/useCollectionStore';
import { activeMissing } from '../PhaseCollections/collectionFilter';
import { colors } from '../../theme/tokens';

const S = {
  base: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: 'transparent', border: `1px solid ${colors.border}`,
    borderRadius: 4, padding: '2px 8px', cursor: 'pointer',
    font: 'inherit', fontSize: '9pt', color: colors.text, whiteSpace: 'nowrap',
  },
  missing: { borderColor: colors.red || '#ff6b6b', color: colors.red || '#ff6b6b' },
  count: { color: colors.textSecondary, fontVariantNumeric: 'tabular-nums' },
};

function memberCount(c) {
  return c.effective_member_count ?? c.member_count ?? 0;
}

export default function ActiveGroupChip({ onOpenLibrary }) {
  const { t } = useTranslation('phaselibrary');
  const data = useCollectionStore((s) => s.data);
  const load = useCollectionStore((s) => s.load);
  const asked = useRef(false);

  /**
   * SOMEBODY HAS TO ASK THE SERVER, and it used to be the dropdown this
   * replaces. Removing that left nothing loading the store at start-up:
   * measured in the running app, the toolbar showed no chip at all until
   * some other page happened to load it, while `_local.json` had an active
   * group the whole time. An indicator that appears late is worse than
   * none, because the pages it should warn on are the ones you visit first.
   *
   * Once per mount, and BEFORE the early return below -- a hook after a
   * conditional return is the "Rendered more hooks than during the previous
   * render" crash this project shipped on 2026-08-14.
   */
  useEffect(() => {
    if (asked.current) return;
    asked.current = true;
    load();
  }, [load]);

  const collections = data?.collections || [];
  const active = data?.state?.active || null;

  // Nothing active is not a state worth a chip: every phase is on offer,
  // which is the ordinary case and needs no announcement.
  if (!active) return null;

  const missing = activeMissing(collections, active);
  const self = collections.find((c) => c.name === active);

  return (
    <button
      type="button"
      style={missing ? { ...S.base, ...S.missing } : S.base}
      onClick={onOpenLibrary}
      data-testid="active-group-chip"
      data-missing={missing ? 'yes' : 'no'}
      title={missing
        ? t('activeGroup.missingTip', { name: active })
        : t('activeGroup.tip', { name: active })}
    >
      {missing ? (
        <span>{t('activeGroup.missing', { name: active })}</span>
      ) : (
        <>
          <span>{t('activeGroup.label', { name: active })}</span>
          <span style={S.count}>
            {t('activeGroup.count', { count: memberCount(self || {}) })}
          </span>
        </>
      )}
    </button>
  );
}
