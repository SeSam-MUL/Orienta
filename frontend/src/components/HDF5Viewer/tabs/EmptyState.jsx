/**
 * EmptyState — shared empty-state placeholder used by tabs.
 *
 * Centered icon + message, fills available space.
 */

import { colors } from '../../../theme/components';

export default function EmptyState({ children, icon }) {
  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 6,
      fontSize: '10pt',
      color: colors.textSecondary,
      padding: 24,
      textAlign: 'center',
    }}>
      {icon && <div style={{ fontSize: 24, opacity: 0.25 }}>{icon}</div>}
      {children}
    </div>
  );
}
