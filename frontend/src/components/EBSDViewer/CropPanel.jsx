/**
 * The line under the overview that turns a drawn selection into a dataset.
 *
 * It owns no selection state — the viewer draws the shape and passes down the
 * bounding box and mask. This component only reports what the crop will cost
 * and fires the request. Keeping the drag out of here is what makes it
 * testable without a canvas.
 */
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { countSelected, estimateBytes, formatBytes } from './navSelection';

/**
 * A bbox we are willing to send to the backend.
 *
 * `boundsOf` is a faithful hull, not a validator: a point with a non-numeric
 * coordinate yields {row0: Infinity, col0: Infinity, rows: NaN, cols: NaN},
 * which is TRUTHY. Left unchecked that renders as a valid-looking selection of
 * "NaN of NaN pixels · 0 B" with the button enabled, and the request fails at
 * the backend. Non-finite grid coordinates are reachable in this codebase —
 * calcOverviewPos divides by the on-screen image size, which is 0 for one
 * frame after a container mounts — so "not null" is not enough.
 */
function isUsableBbox(bbox) {
  if (!bbox) return false;
  const { row0, col0, rows, cols } = bbox;
  if (![row0, col0, rows, cols].every(Number.isFinite)) return false;
  return rows > 0 && cols > 0 && row0 >= 0 && col0 >= 0;
}

export default function CropPanel({
  bbox, mask, patternShape, bytesPerPixel = 1, shape = 'rect',
  busy = false, origin = null, onCrop, onExport, saving = false,
}) {
  const { t } = useTranslation('ebsdviewer');

  const usable = isUsableBbox(bbox);
  const box = usable ? bbox : null;
  const selected = countSelected(mask, box);
  const total = box ? box.rows * box.cols : 0;
  const bytes = box ? estimateBytes(box.rows, box.cols, patternShape, bytesPerPixel) : 0;
  const disabled = busy || !box || selected === 0;

  const handleClick = () => {
    if (!box) return;
    onCrop({
      shape,
      row0: box.row0, col0: box.col0, rows: box.rows, cols: box.cols,
      mask: mask ? Array.from(mask, (v) => !!v) : null,
      materialise: true,
    });
  };

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
      padding: '4px 8px', flexShrink: 0,
      borderTop: `1px solid ${colors.border}`, background: colors.bg,
    }}>
      <span data-testid="crop-summary" style={{ fontSize: '8pt', color: colors.textSecondary }}>
        {/* estimateBytes returns 0 when the detector size is unknown. Printing
            that as "~0 B" would claim the crop is free; say nothing instead. */}
        {box
          ? t(bytes > 0 ? 'crop.summary' : 'crop.summaryNoSize', {
            selected: selected.toLocaleString(),
            total: total.toLocaleString(),
            size: formatBytes(bytes),
          })
          : t('crop.nothingSelected')}
      </span>
      <button
        type="button"
        data-testid="crop-apply"
        disabled={disabled}
        onClick={handleClick}
        style={{
          padding: '3px 10px', borderRadius: 4, fontSize: '8pt', fontWeight: 500,
          border: `1px solid ${colors.border}`,
          background: disabled ? colors.border : colors.accent,
          color: disabled ? colors.textSecondary : colors.textOnAccent,
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.55 : 1,
        }}
      >
        {busy ? t('crop.working') : t('crop.apply')}
      </button>
      {/* Gated on `origin`, i.e. on the backend saying this dataset IS a crop
          — the same fact the endpoint checks. A crop lives in memory and dies
          with the backend; this is the only way out of the process. */}
      {origin && onExport && (
        <button
          type="button"
          data-testid="crop-export"
          disabled={saving}
          onClick={onExport}
          title={t('crop.saveCropHint')}
          style={{
            padding: '3px 10px', borderRadius: 4, fontSize: '8pt', fontWeight: 500,
            border: `1px solid ${colors.border}`,
            background: colors.bgTertiary,
            color: saving ? colors.textSecondary : colors.text,
            cursor: saving ? 'wait' : 'pointer',
            opacity: saving ? 0.55 : 1,
          }}
        >
          {saving ? t('crop.saveCropWorking') : t('crop.saveCrop')}
        </button>
      )}
      <span style={{ flex: 1 }} />
      {origin && (
        <span data-testid="crop-origin" style={{ fontSize: '8pt', color: colors.yellow }}>
          {t('crop.origin', {
            parent: String(origin.source_file || '').split(/[\\/]/).pop() || '?',
            row0: origin.row0,
            row1: origin.row0 + origin.rows - 1,
            col0: origin.col0,
            col1: origin.col0 + origin.cols - 1,
          })}
        </span>
      )}
    </div>
  );
}
