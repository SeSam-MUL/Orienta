/**
 * GalleryList — virtualized rendering of GalleryItem entries.
 *
 * Uses react-window v2's `List` (the old `FixedSizeList` was renamed/replaced
 * in 2.x). `rowComponent` receives `{ index, style, ...rowProps }` and
 * `rowProps` is the bag of extra props we want every row to see.
 */
import { List } from 'react-window';
import { useTranslation } from 'react-i18next';
import GalleryItem from './GalleryItem';

const ROW_HEIGHT = 96;

function Row({ index, style, items, resultId, onClick, onHover, onLeave }) {
  return (
    <div style={style}>
      <GalleryItem
        item={items[index]}
        resultId={resultId}
        visible={true}
        onClick={onClick}
        onHover={onHover}
        onLeave={onLeave}
      />
    </div>
  );
}

export default function GalleryList({ items, resultId, height, width, onClick, onHover, onLeave }) {
  const { t } = useTranslation('phasemap');
  if (!items || items.length === 0) {
    return <div style={{ padding: 12, color: '#888' }}>{t('phasemap:anomalyBrowser.noPixelsMatch')}</div>;
  }
  return (
    <List
      style={{ height, width }}
      rowCount={items.length}
      rowHeight={ROW_HEIGHT}
      rowComponent={Row}
      rowProps={{ items, resultId, onClick, onHover, onLeave }}
    />
  );
}
