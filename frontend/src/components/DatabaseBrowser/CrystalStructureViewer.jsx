/**
 * CrystalStructureViewer — interactive 3D crystal structure for a local .cif/.xtal
 * in the Database Browser preview panel (parallel to MasterSphereViewer for .sht).
 *
 * Fetches GET /api/database/structure/{filename} and renders it with Three.js
 * (real shaded spheres, cylinder bonds, translucent coordination polyhedra, unit
 * cell) via the imperative scene in crystalScene3d.js, which is dynamically
 * imported so `three` stays out of the Database page's initial bundle.
 *
 * See docs/superpowers/specs/2026-07-14-cif-xtal-crystal-structure-viewer-design.md
 */
import { useState, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { dbApi } from '../../services/api';
import { colors } from '../../theme/components';
import { useImageExport, exportStem } from '../common/useImageExport';
import { trimUniformBackground } from '../common/imageExport';
import { useFullscreen, FullscreenButton } from '../common/useFullscreen';
import { elementSummary } from './crystalScene';

/** Resolve a `var(--x)` color to its computed hex (WebGL can't read CSS vars). */
function resolveCssVar(value, fallback) {
  if (typeof value !== 'string' || typeof window === 'undefined') return value || fallback;
  const mm = /var\((--[^),]+)/.exec(value);
  if (!mm) return value;
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(mm[1]).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
}

const HEIGHT = 380;
const overlayBox = {
  position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
  justifyContent: 'center', color: colors.textSecondary, fontSize: '9pt',
  textAlign: 'center', padding: 12, pointerEvents: 'none',
};

export default function CrystalStructureViewer({ filename, isLocal }) {
  const { t } = useTranslation('databasebrowser');
  const imageExport = useImageExport();
  const fsRef = useRef(null);
  const fs = useFullscreen(fsRef);
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);
  const [showBonds, setShowBonds] = useState(true);
  const [showCell, setShowCell] = useState(true);
  const [showPoly, setShowPoly] = useState(true);
  const [hidden, setHidden] = useState(() => new Set());
  const [tip, setTip] = useState(null);

  const bgColor = useMemo(() => resolveCssVar(colors.bgSecondary, '#21222c'), []);

  // Fetch the structure and pick sensible per-file defaults.
  useEffect(() => {
    if (!filename || !isLocal) { setPayload(null); setError(null); return undefined; }
    let cancelled = false;
    setLoading(true); setError(null); setPayload(null);
    dbApi.structure(filename)
      .then((r) => {
        if (cancelled) return;
        setPayload(r.data);
        const n = r.data?.atoms?.length || 0;
        setShowBonds(n <= 60);           // dense cells: bonds off, they become a hairball
        setShowCell(true);
        setShowPoly(true);
        setHidden(new Set());
      })
      .catch((err) => {
        if (!cancelled) setError(err?.response?.data?.detail || err?.message || t('crystalStructure.loadErrorFallback'));
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // `t` is only used for error strings and is stable in production; excluding
    // it avoids a refetch loop if the translator identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename, isLocal]);

  // Create the Three.js scene once (canvas mounts into the always-present container).
  useEffect(() => {
    let cancelled = false;
    let handle = null;
    const container = containerRef.current;
    if (!container) return undefined;
    import('./crystalScene3d')
      .then(({ createCrystalScene }) => {
        if (cancelled || !containerRef.current) return;
        handle = createCrystalScene(containerRef.current, {
          onHover: (atom, x, y) => setTip(atom ? { atom, x, y } : null),
          background: bgColor,
        });
        sceneRef.current = handle;
        setReady(true);
      })
      .catch(() => { if (!cancelled) setError(t('crystalStructure.loadErrorFallback')); });
    return () => {
      cancelled = true;
      if (handle) handle.dispose();
      sceneRef.current = null;
      setReady(false);
    };
    // Create the WebGL scene once on mount (bgColor is memoised/stable; `t` is
    // only the error fallback) — depending on either would dispose+rebuild the
    // scene on every translator/render change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load geometry when a new structure (or the scene) is ready.
  useEffect(() => {
    if (ready && payload && sceneRef.current) {
      sceneRef.current.update(payload, { showBonds, showCell, showPolyhedra: showPoly, hidden });
    }
    // Toggles are handled by the options effect below — don't rebuild on them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, payload]);

  // Cheap visibility flips (no geometry rebuild) when a toggle / legend changes.
  useEffect(() => {
    if (ready && payload && sceneRef.current) {
      sceneRef.current.setOptions({ showBonds, showCell, showPolyhedra: showPoly, hidden });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showBonds, showCell, showPoly, hidden]);

  const legend = useMemo(() => (payload ? elementSummary(payload.atoms) : []), [payload]);
  const hasPoly = payload?.polyhedra?.length > 0;

  const toggleElement = (el) => setHidden((prev) => {
    const next = new Set(prev);
    if (next.has(el)) next.delete(el); else next.add(el);
    return next;
  });

  const meta = payload?.meta || {};
  const sg = payload?.space_group || {};
  const metaLine = payload ? [
    meta.formula,
    sg.symbol ? `${sg.symbol} (#${sg.number})` : null,
    sg.crystal_system,
    t('crystalStructure.atomsCount', { count: meta.n_atoms }),
  ].filter(Boolean).join(' · ') : '';

  const ctrlLabel = { cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 };
  const iconBtn = {
    background: colors.bgTertiary, border: `1px solid ${colors.border}`, borderRadius: 4,
    color: colors.text, cursor: 'pointer', fontSize: '8pt', padding: '2px 8px',
  };

  return (
    <div
      ref={fsRef}
      style={fs.active ? {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: colors.bg,
        padding: 8,
        boxSizing: 'border-box',
      } : undefined}
    >
      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4, fontSize: '8pt', color: colors.textSecondary, flexWrap: 'wrap' }}>
        <label style={ctrlLabel} title={t('crystalStructure.bondsTooltip')}>
          <input type="checkbox" checked={showBonds} onChange={(e) => setShowBonds(e.target.checked)} /> {t('crystalStructure.bonds')}
        </label>
        <label style={ctrlLabel} title={t('crystalStructure.cellTooltip')}>
          <input type="checkbox" checked={showCell} onChange={(e) => setShowCell(e.target.checked)} /> {t('crystalStructure.cellBox')}
        </label>
        <label style={{ ...ctrlLabel, opacity: hasPoly ? 1 : 0.4 }} title={t('crystalStructure.polyhedraTooltip')}>
          <input type="checkbox" checked={showPoly && hasPoly} disabled={!hasPoly} onChange={(e) => setShowPoly(e.target.checked)} /> {t('crystalStructure.polyhedra')}
        </label>
        <div style={{ flex: 1 }} />
        <button type="button" style={iconBtn} onClick={() => sceneRef.current?.resetView()} title={t('crystalStructure.resetTooltip')}>⟲ {t('crystalStructure.reset')}</button>
        <button type="button" style={iconBtn} onClick={() => sceneRef.current?.screenshot()} title={t('crystalStructure.screenshotTooltip')}>⤓ PNG</button>
        <FullscreenButton active={fs.active} onToggle={fs.toggle} variant="plate" />
      </div>

      {/* Canvas + overlays (container is always present so the scene can mount) */}
      <div
        ref={containerRef}
        onContextMenu={(e) => {
          if (!isLocal || !sceneRef.current) return;
          imageExport.openMenu(e, {
            // Reads the view back as it stands — same camera, same hidden
            // elements, same toggles.
            build: async () => {
              const src = sceneRef.current?.toDataURL();
              return { src, defaultCrop: src ? await trimUniformBackground(src) : null };
            },
            name: exportStem(filename, 'crystal-structure'),
            label: exportStem(filename, 'crystal-structure'),
          });
        }}
        style={{
          position: 'relative',
          // Fullscreen: take whatever the row above leaves. A percentage would
          // resolve against a parent with no height of its own. The scene's
          // ResizeObserver picks the new size up either way.
          height: fs.active ? undefined : HEIGHT,
          flex: fs.active ? 1 : undefined,
          minHeight: 0,
          width: '100%',
          background: colors.bgSecondary, border: `1px solid ${colors.border}`,
          borderRadius: fs.active ? 0 : 4, overflow: 'hidden',
        }}
      >
        {/* Element legend (click to hide) */}
        {payload && (
          <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            {legend.map((e) => {
              const off = hidden.has(e.element);
              return (
                <button
                  key={e.element}
                  type="button"
                  onClick={() => toggleElement(e.element)}
                  title={t('crystalStructure.legendTooltip')}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6, background: 'transparent',
                    border: 'none', cursor: 'pointer', padding: '1px 2px', fontSize: '8pt',
                    color: off ? colors.textSecondary : colors.text,
                    textDecoration: off ? 'line-through' : 'none', opacity: off ? 0.55 : 1,
                  }}
                >
                  <span style={{ width: 11, height: 11, borderRadius: '50%', background: e.color, border: '1px solid rgba(0,0,0,0.3)' }} />
                  {e.element} <span style={{ opacity: 0.6 }}>×{e.count}</span>
                </button>
              );
            })}
          </div>
        )}

        {!isLocal && <div style={overlayBox}>{t('crystalStructure.downloadFirst')}</div>}
        {isLocal && loading && <div style={overlayBox}>{t('crystalStructure.parsing')}</div>}
        {isLocal && !loading && error && <div style={{ ...overlayBox, color: colors.red }}>{t('crystalStructure.loadError', { error })}</div>}
      </div>

      {/* Hover tooltip */}
      {tip && (
        <div style={{
          position: 'fixed', left: tip.x + 14, top: tip.y + 14, zIndex: 3000, pointerEvents: 'none',
          background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 4,
          padding: '4px 8px', fontSize: '8pt', color: colors.text, boxShadow: '0 4px 14px rgba(0,0,0,0.4)',
        }}>
          <div style={{ fontWeight: 700 }}>{tip.atom.label}</div>
          <div>{tip.atom.species.length > 1 ? tip.atom.species.map((s) => `${s.el} ${s.occ}`).join(' / ') : tip.atom.element}</div>
          <div style={{ opacity: 0.7 }}>({tip.atom.frac.join(', ')}) · occ {tip.atom.occ}</div>
        </div>
      )}

      {metaLine && (
        <div style={{ fontSize: '8pt', color: colors.textSecondary, textAlign: 'center', marginTop: 4 }}>
          {metaLine}
          {meta.disordered && <span style={{ color: colors.yellow, marginLeft: 6 }}>· {t('crystalStructure.disordered')}</span>}
        </div>
      )}
      <div style={{ fontSize: '8pt', color: colors.textSecondary, textAlign: 'center', marginTop: 2, opacity: 0.7 }}>
        {t('crystalStructure.hint')}
      </div>
      {imageExport.node}
    </div>
  );
}
