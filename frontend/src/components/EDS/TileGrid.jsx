import Tile from './Tile';
import { colors } from '../../theme/components';

export default function TileGrid({ layers, bitmaps, errors, shape, onPixelClick, onRegionSelected, minTileWidth = 240, emptyMessage = 'Tick a layer in the panel on the left, or pick a preset.' }) {
  const visible = layers.filter((l) => l.visible);
  if (visible.length === 0) {
    return (
      <div style={{
        color: colors.textSecondary,
        padding: 24,
        textAlign: 'center',
        fontSize: '10pt',
      }}>
        {emptyMessage}
      </div>
    );
  }
  return (
    <div
      data-tile-grid
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fit, minmax(${minTileWidth}px, 1fr))`,
        gap: 10,
        padding: 4,
        alignContent: 'start',
        height: '100%',
        overflowY: 'auto',
      }}
    >
      {visible.map((l) => (
        <Tile
          key={l.id}
          layer={l}
          // Mask layers have no bitmap of their own. Hand the source bitmap
          // to Tile so it can synthesise the B/W mask preview itself.
          bitmap={l.kind === 'mask' ? bitmaps.get(l.isMaskFor) : bitmaps.get(l.id)}
          error={errors.get(l.id)}
          shape={shape}
          onPixelClick={onPixelClick}
          onRegionSelected={onRegionSelected}
        />
      ))}
    </div>
  );
}
