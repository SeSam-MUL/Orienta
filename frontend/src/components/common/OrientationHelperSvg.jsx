// frontend/src/components/common/OrientationHelperSvg.jsx
// Live RD/TD/ND helper diagram. Draws a sample square with X/Y axes per the
// plot convention so the user can see which way RD(X)/TD(Y)/ND(Z) point.
import { colors } from '../../theme/tokens';

export default function OrientationHelperSvg({ plot }) {
  const xNorth = plot?.x_direction === 'north';
  const zIn = !!plot?.z_into_plane;
  // X axis direction in SVG space (y grows downward).
  const xVec = xNorth ? { dx: 0, dy: -1, lx: 0, ly: -1 } : { dx: 1, dy: 0, lx: 1, ly: 0 };
  const yVec = xNorth ? { dx: 1, dy: 0 } : { dx: 0, dy: 1 };
  const cx = 70, cy = 70, L = 46;
  const ax = (v) => cx + v.dx * L, ay = (v) => cy + v.dy * L;
  return (
    <svg data-testid="orientation-helper-svg" width="140" height="140" viewBox="0 0 140 140"
         style={{ background: colors.bg, borderRadius: 6, border: `1px solid ${colors.border}` }}>
      <rect x="24" y="24" width="92" height="92" fill="none" stroke={colors.border} strokeDasharray="3 3" />
      {/* X axis (RD) */}
      <line x1={cx} y1={cy} x2={ax(xVec)} y2={ay(xVec)} stroke={colors.red} strokeWidth="2" markerEnd="url(#ah)" />
      <text x={ax(xVec)} y={ay(xVec)} dx={6} dy={4} fill={colors.red} fontSize="11">X · RD</text>
      {/* Y axis (TD) */}
      <line x1={cx} y1={cy} x2={ax(yVec)} y2={ay(yVec)} stroke={colors.green} strokeWidth="2" markerEnd="url(#ah)" />
      <text x={ax(yVec)} y={ay(yVec)} dx={6} dy={4} fill={colors.green} fontSize="11">Y · TD</text>
      {/* Z (ND) into/out of plane */}
      <circle cx={cx} cy={cy} r="6" fill="none" stroke={colors.cyan} strokeWidth="2" />
      {zIn
        ? <line x1={cx - 4} y1={cy - 4} x2={cx + 4} y2={cy + 4} stroke={colors.cyan} strokeWidth="2" />
        : <circle cx={cx} cy={cy} r="1.6" fill={colors.cyan} />}
      <text x={cx + 10} y={cy - 8} fill={colors.cyan} fontSize="11">Z · ND {zIn ? '⊗' : '⊙'}</text>
      <defs>
        <marker id="ah" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={colors.textSecondary} />
        </marker>
      </defs>
    </svg>
  );
}
