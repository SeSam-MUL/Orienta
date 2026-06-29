/**
 * VantaBackground — Vanta.js FOG effect via CDN globals.
 * Three.js and Vanta.js are loaded via script tags in index.html.
 * This component just initializes the effect on mount and destroys on unmount.
 */
import { useRef, useEffect } from 'react';
import { useTheme } from '../../theme/ThemeProvider';
import { themes } from '../../theme/themes';

// Fallback matches the original hardcoded deep-teal look (scientific palette).
const DEFAULT_FOG = {
  highlightColor: 0x1a4a5a,
  midtoneColor: 0x18324a,
  lowlightColor: 0x0a1020,
  baseColor: 0x0f172a,
};

export default function VantaBackground() {
  const containerRef = useRef(null);
  const vantaRef = useRef(null);
  const { theme } = useTheme();

  // Re-init whenever the active theme changes so the fog matches the design.
  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;
    const fog = (themes[theme] && themes[theme].fog) || DEFAULT_FOG;

    // Wait for CDN scripts to load (THREE + VANTA)
    function tryInit() {
      if (cancelled) return;
      if (!window.VANTA || !window.VANTA.FOG || !window.THREE) {
        setTimeout(tryInit, 100);
        return;
      }

      try {
        vantaRef.current = window.VANTA.FOG({
          el: containerRef.current,
          THREE: window.THREE,
          mouseControls: false,
          touchControls: false,
          gyroControls: false,
          minHeight: 200,
          minWidth: 200,
          highlightColor: fog.highlightColor,
          midtoneColor: fog.midtoneColor,
          lowlightColor: fog.lowlightColor,
          baseColor: fog.baseColor,
          blurFactor: 0.4,
          speed: 0.8,
          zoom: 0.6,
        });
      } catch (err) {
        console.warn('Vanta.js FOG init failed:', err);
      }
    }

    tryInit();

    // Vanta doesn't auto-resize when container changes size (only on window resize).
    // Use ResizeObserver to trigger Vanta's resize() whenever the container resizes.
    const ro = new ResizeObserver(() => {
      if (vantaRef.current && vantaRef.current.resize) {
        vantaRef.current.resize();
      }
    });
    ro.observe(containerRef.current);

    return () => {
      cancelled = true;
      ro.disconnect();
      if (vantaRef.current) {
        vantaRef.current.destroy();
        vantaRef.current = null;
      }
    };
  }, [theme]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  );
}
