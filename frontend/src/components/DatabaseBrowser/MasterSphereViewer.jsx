/**
 * MasterSphereViewer — interactive, rotatable 3D master-pattern sphere for a .sht.
 *
 * Shown in the Database Browser preview when a local .sht is selected. Fetches the
 * reconstructed sphere grid from GET /api/database/sphere/{filename} and renders it
 * as a Plotly Surface (drag = rotate, scroll = zoom, toolbar = reset). Plotly is
 * lazy-loaded so its (large) bundle does not bloat the Database page's initial load.
 *
 * See docs/superpowers/specs/2026-06-22-sht-sphere-viewer-design.md
 */
import { useState, useEffect, useMemo, useRef, useCallback, Suspense, lazy } from 'react';
import { useTranslation } from 'react-i18next';
import { dbApi } from '../../services/api';
import { colors } from '../../theme/components';
import { useImageExport, exportStem } from '../common/useImageExport';
import { trimUniformBackground } from '../common/imageExport';
import { useFullscreen, fullscreenIcon } from '../common/useFullscreen';

// Lazy so the ~MB Plotly bundle is only pulled when a sphere is actually viewed.
// react-plotly.js is CJS; under Vite's ESM interop the dynamic import can resolve to
// either { default: Component } or { default: { default: Component } } (double-default).
// Unwrap defensively so React.lazy always receives the actual component.
const Plot = lazy(() =>
  import('react-plotly.js').then((m) => {
    const C = (m && m.default && m.default.default) ? m.default.default : (m.default || m);
    return { default: C };
  })
);

/** Resolve a `var(--x)` color to its computed hex (Plotly canvas can't read CSS vars). */
function resolveCssVar(value, fallback) {
  if (typeof value !== 'string' || typeof window === 'undefined') return value || fallback;
  const m = /var\((--[^),]+)/.exec(value);
  if (!m) return value;
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
}

/**
 * Resolution presets for the sphere reconstruction. Standard sends no params so the
 * backend keeps its own defaults (128/128) — this preserves the existing fetch
 * contract `dbApi.sphere(filename)` for the common case. Hoch/Max request a higher
 * SHT bandwidth + finer sample grid (both clamped server-side: bw 16–384, size 32–256).
 */
const RES_PRESETS = {
  standard: { labelKey: 'sphere.presetStandard', max_bandwidth: 128, target_size: 128 },
  high:     { labelKey: 'sphere.presetHigh',     max_bandwidth: 256, target_size: 192 },
  max:      { labelKey: 'sphere.presetMax',      max_bandwidth: 384, target_size: 256 },
};

const BOX = {
  height: 360,
  background: colors.bgSecondary,
  border: `1px solid ${colors.border}`,
  borderRadius: 4,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: colors.textSecondary,
  fontSize: '9pt',
};

/**
 * Build Plotly Surface arrays from the backend payload. The DH grid is
 * (n_phi, n_theta) with phi=azimuth[0,2π), theta=polar(0,π). We map each (φ,θ) to a
 * unit-sphere point and close the azimuth seam by repeating the first row so the
 * surface wraps with no gap.
 */
function buildSurface(payload) {
  const { grid, phi, theta } = payload;
  const nT = theta.length;
  const sinT = theta.map((t) => Math.sin(t));
  const cosT = theta.map((t) => Math.cos(t));
  const X = [], Y = [], Z = [], C = [];
  const pushRow = (ph, gridRow) => {
    const xr = new Array(nT), yr = new Array(nT), zr = new Array(nT);
    const cp = Math.cos(ph), sp = Math.sin(ph);
    for (let j = 0; j < nT; j++) {
      xr[j] = sinT[j] * cp;
      yr[j] = sinT[j] * sp;
      zr[j] = cosT[j];
    }
    X.push(xr); Y.push(yr); Z.push(zr); C.push(gridRow);
  };
  for (let i = 0; i < phi.length; i++) pushRow(phi[i], grid[i]);
  // Seam closure: repeat row 0 at φ = 2π (same point as φ = 0).
  pushRow(2 * Math.PI, grid[0]);
  return { X, Y, Z, C };
}

export default function MasterSphereViewer({ filename, isLocal }) {
  // 'common' too: the fullscreen label lives there, shared with the other viewers.
  const { t } = useTranslation(['databasebrowser', 'common']);
  const imageExport = useImageExport();
  // Plotly's graph div, handed over on init.
  const gdRef = useRef(null);
  const boxRef = useRef(null);
  const fs = useFullscreen(boxRef);
  // Plotly builds the modebar from `config`; a fresh object on every render
  // would make it rebuild the bar constantly. So the config is memoised on the
  // only thing that actually changes it — the icon/label — and the click goes
  // through a ref that always holds the current toggle.
  const toggleRef = useRef(fs.toggle);
  toggleRef.current = fs.toggle;

  /**
   * The sphere as a PNG data URL.
   *
   * Plotly builds its gl3d context WITHOUT `preserveDrawingBuffer` on desktop,
   * so reading its canvas with toDataURL() gives a blank image. The scene's own
   * `toImage` redraws and does a readPixels, which is the only route that
   * returns actual pixels.
   *
   * It is reached through `gd._fullLayout` — an internal handle — because
   * importing the `plotly.js` module here fails under Vite ("global is not
   * defined": the CJS entry expects Node globals, while react-plotly.js gets a
   * pre-bundled copy). Verified live. If a Plotly upgrade ever moves it, the
   * guard below turns that into a plain message rather than a blank export.
   */
  const captureSphere = useCallback(async () => {
    const scene = gdRef.current?._fullLayout?.scene?._scene;
    if (typeof scene?.toImage !== 'function') {
      throw new Error(t('databasebrowser:sphere.captureNotReady'));
    }

    // Shoot at three times the screen resolution, so the export dialog is
    // downscaling rather than enlarging a screen grab. Anything beyond that is
    // pointless: measured on this machine, a ratio of 12 came back SMALLER than
    // a ratio of 8 because the browser silently caps the drawing buffer.
    // The on-screen canvas keeps its own size throughout — nothing flickers.
    const base = scene.pixelRatio || 1;
    const glplot = scene.glplot || null;
    const prev = { scene: scene.pixelRatio, gl: glplot?.pixelRatio };
    let src;
    try {
      scene.pixelRatio = Math.min(base * 3, 6);
      if (glplot) glplot.pixelRatio = scene.pixelRatio;
      src = scene.toImage('png');
    } finally {
      scene.pixelRatio = prev.scene;
      if (glplot) glplot.pixelRatio = prev.gl;
    }

    // The sphere covers a fraction of a wide strip; open the dialog on the
    // sphere itself. "Whole image" undoes it.
    return { src, defaultCrop: await trimUniformBackground(src) };
  }, [t]);
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preset, setPreset] = useState('standard');

  useEffect(() => {
    if (!filename || !isLocal) { setPayload(null); setError(null); return; }
    let cancelled = false;
    setLoading(true); setError(null); setPayload(null);
    const p = RES_PRESETS[preset] || RES_PRESETS.standard;
    // Standard sends NO second arg → keeps the backend defaults and the original
    // single-arg `dbApi.sphere(filename)` contract; Hoch/Max request a sharper sphere.
    const fetchSphere = preset === 'standard'
      ? dbApi.sphere(filename)
      : dbApi.sphere(filename, { max_bandwidth: p.max_bandwidth, target_size: p.target_size });
    fetchSphere
      .then((r) => { if (!cancelled) setPayload(r.data); })
      .catch((err) => {
        if (!cancelled) setError(err?.response?.data?.detail || err?.message || t('databasebrowser:sphere.loadErrorFallback'));
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [filename, isLocal, preset]);

  const surface = useMemo(() => (payload ? buildSurface(payload) : null), [payload]);

  // Plotly draws to canvas/WebGL and CANNOT resolve CSS variables (the theme colors
  // are `var(--bg-secondary)` etc.), so a raw `colors.bgSecondary` makes the plot
  // background fall back to white. Resolve the var to its computed value here.
  const bgColor = useMemo(() => resolveCssVar(colors.bgSecondary, '#21222c'), []);

  if (!isLocal) return <div style={BOX}>{t('databasebrowser:sphere.downloadFirst')}</div>;
  if (loading) return <div style={BOX}>{t('databasebrowser:sphere.reconstructing')}</div>;
  if (error) return <div style={{ ...BOX, color: colors.red }}>{t('databasebrowser:sphere.loadError', { error })}</div>;
  if (!surface) return <div style={BOX}>{t('databasebrowser:sphere.selectLocal')}</div>;

  // Fullscreen as a modebar icon rather than a button floating over the plot:
  // same size, same hover, same alignment as the zoom/pan/reset icons, and it
  // stays reachable while fullscreen because the modebar goes fullscreen too.
  const plotConfig = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['toImage'],
    modeBarButtonsToAdd: [{
      name: 'fullscreen',
      title: fs.active ? t('common:fullscreen.exit') : t('common:fullscreen.enter'),
      icon: fullscreenIcon(fs.active),
      click: () => toggleRef.current?.(),
    }],
  };

  const meta = payload.meta || {};
  const metaLine = [
    meta.formula,
    meta.space_group != null ? t('databasebrowser:sphere.spaceGroupShort', { sg: meta.space_group }) : null,
    meta.point_group,
    meta.voltage_kV != null ? `${meta.voltage_kV} kV` : null,
  ].filter(Boolean).join(' · ');

  return (
    <div
      onContextMenu={(e) => {
        if (typeof gdRef.current?._fullLayout?.scene?._scene?.toImage !== 'function') return;
        imageExport.openMenu(e, {
          build: captureSphere,
          name: exportStem(filename, 'master-sphere'),
          label: exportStem(filename, 'master-sphere'),
        });
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginBottom: 4 }}>
        <label style={{ fontSize: '8pt', color: colors.textSecondary, marginRight: 4 }}
               htmlFor="sphere-res">{t('databasebrowser:sphere.resolutionLabel')}</label>
        <select id="sphere-res" aria-label={t('databasebrowser:sphere.resolutionLabel')}
                title={t('databasebrowser:sphere.resolutionTooltip')} value={preset}
                onChange={(e) => setPreset(e.target.value)}
                style={{ fontSize: '8pt' }}>
          {Object.entries(RES_PRESETS).map(([k, v]) => (
            <option key={k} value={k}>{t(`databasebrowser:${v.labelKey}`)}</option>
          ))}
        </select>
      </div>
      <div
        ref={boxRef}
        style={{
          position: 'relative',
          // Fullscreen replaces the fixed 360 px preview height, otherwise the
          // sphere would sit as a small box in the middle of a black screen.
          height: fs.active ? '100%' : undefined,
          background: fs.active ? bgColor : undefined,
        }}
      >
      <Suspense fallback={<div style={BOX}>{t('databasebrowser:sphere.loadingViewer')}</div>}>
        <Plot
          data={[{
            type: 'surface',
            x: surface.X, y: surface.Y, z: surface.Z,
            surfacecolor: surface.C,
            // high intensity → bright (white); low → dark.
            colorscale: [[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
            showscale: false,
            lighting: { ambient: 1.0, diffuse: 0.0, specular: 0.0, roughness: 1.0, fresnel: 0.0 },
            contours: { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } },
            hoverinfo: 'skip',
          }]}
          layout={{
            autosize: true,
            // In fullscreen the height comes from the box, not from a constant.
            height: fs.active ? undefined : 360,
            margin: { l: 0, r: 0, t: 0, b: 0 },
            paper_bgcolor: bgColor,
            plot_bgcolor: bgColor,
            scene: {
              xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false },
              aspectmode: 'data',
              bgcolor: bgColor,
              dragmode: 'orbit',
            },
          }}
          style={{ width: '100%', height: fs.active ? '100%' : undefined }}
          config={plotConfig}
          useResizeHandler
          onInitialized={(fig, gd) => { gdRef.current = gd; }}
        />
      </Suspense>
      </div>
      {metaLine && (
        <div style={{ fontSize: '8pt', color: colors.textSecondary, textAlign: 'center', marginTop: 4 }}>
          {metaLine}
          <span style={{ opacity: 0.6 }}>{` · ${t('databasebrowser:sphere.bandwidthSuffix', { bw: payload.bandwidth })}`}</span>
        </div>
      )}
      <div style={{ fontSize: '8pt', color: colors.textSecondary, textAlign: 'center', marginTop: 2, opacity: 0.7 }}>
        {t('databasebrowser:sphere.hint')}
      </div>
      {imageExport.node}
    </div>
  );
}
