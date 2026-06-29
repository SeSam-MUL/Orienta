import { colors } from '../../../theme/components';

export default function ColorLegend({ kind, min, max, phases }) {
  if (kind === 'scalar' && min != null) {
    return (
      <div style={{
        position: 'absolute', bottom: 8, left: 8,
        background: 'rgba(0,0,0,0.7)', border: `1px solid ${colors.border}`,
        padding: '4px 8px', fontSize: '8pt', color: colors.text,
        fontFamily: 'monospace',
      }}>
        {min.toFixed(2)} ─── {max.toFixed(2)}
      </div>
    );
  }
  if (kind === 'phase' && phases?.length) {
    return (
      <div style={{
        position: 'absolute', bottom: 8, left: 8,
        background: 'rgba(0,0,0,0.7)', border: `1px solid ${colors.border}`,
        padding: '4px 8px', fontSize: '8pt', color: colors.text,
        maxHeight: 120, overflowY: 'auto',
      }}>
        {phases.map((p) => (
          <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              width: 10, height: 10, display: 'inline-block',
              background: `rgb(${p.color.join(',')})`,
            }} />
            <span>{p.name}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}
