import { colors } from '../../theme/tokens';

const STATUS_COLORS = { pass: colors.green, warn: colors.yellow, fail: colors.red };
const STATUS_ICONS  = { pass: '\u2713',     warn: '\u26A0',      fail: '\u2717'  };

export default function PreFlightReport({ checks }) {
  if (!checks || checks.length === 0) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {checks.map((c, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: '10pt' }}>
          <span style={{ color: STATUS_COLORS[c.status], fontWeight: 'bold', width: 16 }}>
            {STATUS_ICONS[c.status]}
          </span>
          <span style={{ color: colors.text }}>{c.message}</span>
        </div>
      ))}
    </div>
  );
}
