import { useTranslation } from 'react-i18next';
import { colors, spacing, CollapsibleGroup } from '../../theme/components';
import { findScaleBody, withScaleBody } from './annotations/exportAnnotEdits';

/**
 * Which scales the exported figure carries.
 *
 * One row per scale the picture COULD show — the IPF colour key and every
 * layer whose colours mean a number. Ticking a box puts that scale into the
 * figure as a body the user can then move, resize and put on a plate; clearing
 * it takes the body out again.
 *
 * The box is not a stored flag. It reads whether the body exists, so removing
 * one with its own × on the preview leaves the list telling the truth.
 *
 * `entries`: [{ type: 'colorkey' } | { type: 'valuescale', layerId }, label]
 */
export default function ExportScalePanel({
  entries, annotations, onChange, onSelect,
}) {
  const { t } = useTranslation(['phasemap', 'imageexport']);
  const list = Array.isArray(entries) ? entries : [];

  const toggle = (entry, present) => {
    const { annotations: next, id } = withScaleBody(annotations, entry, present);
    onChange(next);
    // Select what was just added, so the properties for it are one glance away
    // instead of needing a hunt on the picture.
    if (id && onSelect) onSelect(id);
    else if (!present && onSelect) onSelect(null);
  };

  return (
    <CollapsibleGroup title={t('phasemap:exportScales.title')}>
      <div style={{ marginTop: spacing.innerSpacing }} data-export-scale-panel>
        {list.length === 0 ? (
          <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
            {t('phasemap:exportScales.none')}
          </div>
        ) : (
          <>
            <div style={{ fontSize: '8pt', color: colors.textSecondary, marginBottom: 4 }}>
              {t('phasemap:exportScales.hint')}
            </div>
            {list.map((entry) => {
              const body = findScaleBody(annotations, entry);
              const key = entry.type === 'valuescale' ? `v:${entry.layerId}` : entry.type;
              return (
                <label
                  key={key}
                  title={t('phasemap:hoverTips.exportScaleToggle')}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    fontSize: '9pt', color: colors.text,
                    padding: '2px 0', cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={!!body}
                    onChange={(e) => toggle(entry, e.target.checked)}
                    style={{ margin: 0 }}
                    data-scale-toggle={key}
                  />
                  <span style={{
                    flex: 1, minWidth: 0,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {entry.label}
                  </span>
                  {body && (
                    <button
                      type="button"
                      onClick={(e) => { e.preventDefault(); onSelect?.(body.id); }}
                      title={t('phasemap:hoverTips.selectAnnotation')}
                      style={{
                        background: 'transparent', border: 'none', padding: 0,
                        color: colors.textSecondary, cursor: 'pointer', fontSize: '8pt',
                        textDecoration: 'underline',
                      }}
                    >
                      {t('phasemap:exportScales.edit')}
                    </button>
                  )}
                </label>
              );
            })}
          </>
        )}
      </div>
    </CollapsibleGroup>
  );
}
