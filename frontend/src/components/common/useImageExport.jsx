import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import ContextMenu from './ContextMenu';
import ImageExportDialog from './ImageExportDialog';
import { colors } from '../../theme/components';

/**
 * A file name reduced to its stem — the export dialog seeds both the caption and
 * the suggested file name from it. Splits on either slash, so a full Windows
 * path works as well as a bare name.
 */
export function exportStem(filename, fallback = 'export') {
  const base = String(filename || '').split(/[\\/]/).pop();
  return base.replace(/\.[^.]+$/, '') || fallback;
}

/**
 * Right-click → "Export image…" for a single visual.
 *
 * The pattern (menu state + build-can-fail + dialog) was already repeated in the
 * EDS page, the phase map and the EBSD viewer; this packages it so a viewer only
 * has to say WHAT to export:
 *
 *   const exp = useImageExport();
 *   <div onContextMenu={(e) => exp.openMenu(e, {
 *          build: () => canvas.toDataURL('image/png'),
 *          name: 'my-file', label: 'My file' })}>
 *   {exp.node}
 *
 * `build` returns a data/blob/http URL and may be async — capturing a WebGL or
 * Plotly view involves a read-back that resolves later. If it throws or resolves
 * to nothing the failure is shown instead of an empty dialog.
 */
export function useImageExport() {
  const { t } = useTranslation(['imageexport']);
  const [menu, setMenu] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  /**
   * `spec` is one export, or several when a visual can be exported more than one
   * way — the forward-simulation pair offers each panel on its own and both on
   * one sheet. Each entry may carry `menuLabel`; with one entry the generic
   * "Export image…" is used.
   */
  const openMenu = useCallback((e, spec) => {
    const specs = (Array.isArray(spec) ? spec : [spec]).filter(Boolean);
    if (!specs.length) return;
    e.preventDefault();
    // Some hosts (OrbitControls on the crystal canvas) also act on contextmenu.
    e.stopPropagation();
    setMenu({ x: e.clientX, y: e.clientY, specs });
  }, []);

  const run = useCallback(async (spec) => {
    setError(null);
    setBusy(true);
    try {
      const { build, name, label, menuLabel: _menuLabel, id: _id, ...rest } = spec;
      // `build` may hand back a bare URL or `{ src, ...dialogProps }` — the 3D
      // viewers use the second form to pass an auto-crop along.
      const out = await build();
      const { src, ...extra } = (typeof out === 'string' || !out) ? { src: out } : out;
      if (!src) throw new Error(t('imageexport:captureFailed'));
      setJob({ src, name, label, rest: { ...rest, ...extra } });
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setBusy(false);
    }
  }, [t]);

  const node = (
    <>
      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={menu.specs.map((spec, i) => ({
            // A lone entry keeps the plain id, so existing hooks into it hold.
            id: spec.id || (menu.specs.length === 1 ? 'export' : `export-${i}`),
            label: spec.menuLabel || t('imageexport:menuExport'),
            disabled: busy,
            onSelect: () => run(spec),
          }))}
        />
      )}
      {error && (
        <div
          data-image-export-error
          role="alert"
          onClick={() => setError(null)}
          style={{
            position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
            zIndex: 3600, background: colors.bgSecondary, border: `1px solid ${colors.red}`,
            color: colors.red, borderRadius: 6, padding: '8px 14px', fontSize: '9pt',
            cursor: 'pointer',
          }}
        >
          {error}
        </div>
      )}
      {job && (
        <ImageExportDialog
          open
          onClose={() => setJob(null)}
          src={job.src}
          title={job.label}
          defaultBaseName={job.name}
          annotations={{ label: job.label }}
          {...job.rest}
        />
      )}
    </>
  );

  return { openMenu, node, busy };
}

export default useImageExport;
