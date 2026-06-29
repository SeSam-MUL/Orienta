/**
 * WandererOverlay — Tiny CSS-animated wanderer silhouette with lantern.
 * Walks slowly left-to-right across the dashboard, behind the cards.
 * Placeholder for future Lottie-based character.
 */

const WALK_DURATION = 60; // seconds for one full crossing

export default function WandererOverlay() {
  return (
    <div style={{
      position: 'absolute',
      bottom: '35%',
      left: 0,
      right: 0,
      height: 30,
      zIndex: 1,
      pointerEvents: 'none',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute',
        bottom: 0,
        animation: `wander ${WALK_DURATION}s linear infinite`,
      }}>
        {/* Head */}
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: 'rgba(200, 200, 220, 0.4)',
          marginLeft: 3,
        }} />
        {/* Body */}
        <div style={{
          width: 0, height: 0,
          borderLeft: '6px solid transparent',
          borderRight: '6px solid transparent',
          borderBottom: '10px solid rgba(180, 180, 200, 0.3)',
          transform: 'rotate(180deg)',
        }} />
        {/* Lantern glow */}
        <div style={{
          position: 'absolute',
          top: 2, left: 14,
          width: 4, height: 4, borderRadius: '50%',
          background: 'rgba(255, 179, 71, 0.6)',
          boxShadow: '0 0 12px 6px rgba(255, 179, 71, 0.15), 0 0 4px 2px rgba(255, 179, 71, 0.3)',
          animation: 'lanternPulse 3.2s ease-in-out infinite',
        }} />
        {/* Lantern stick */}
        <div style={{
          position: 'absolute',
          top: 0, left: 12,
          width: 1, height: 10,
          background: 'rgba(180, 160, 140, 0.3)',
          transform: 'rotate(15deg)',
        }} />
      </div>

      <style>{`
        @keyframes wander {
          from { transform: translateX(-30px); }
          to { transform: translateX(calc(100vw + 30px)); }
        }
        @keyframes lanternPulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
