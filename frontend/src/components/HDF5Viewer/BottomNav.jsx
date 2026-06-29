/**
 * BottomNav — navigation controls (prev/next + row/col spinboxes + position label),
 * flat-index slider/spinbox, and playback + bookmark controls.
 * Pure relocation from HDF5Viewer.jsx — no logic changes.
 */
import { useTranslation } from 'react-i18next';
import { colors, alpha, spacing, NumberInput, Select, Button, Label } from '../../theme/components';

const PLAYBACK_SPEEDS = [2, 5, 10, 20];

export default function BottomNav({
  isFileOpen, currentRow, currentCol, currentIndex,
  rows, cols, maxIndex,
  navigateRow, navigateCol, navigateByIndex, fetchAndSetPattern,
  playing, playSpeed, setPlaySpeed, togglePlayback,
  bookmarks, isCurrentBookmarked, addBookmark, clearBookmarks,
  patternCount, filePath,
}) {
  const { t } = useTranslation('hdf5viewer');
  return (
    <div style={{
      flexShrink: 0,
      background: colors.bgSecondary,
      borderTop: `1px solid ${colors.border}`,
      padding: `${spacing.innerMargin}px ${spacing.outerMargin}px`,
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
    }}>
      {/* Row 1: Row/Col spinboxes + prev/next arrows + position label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <button
          onClick={() => navigateByIndex(currentIndex - 1)}
          disabled={!isFileOpen || currentIndex === 0}
          title={t('bottomNav.prevTooltip')}
          style={{
            width: 26, height: 26,
            borderRadius: 3,
            border: `1px solid ${colors.border}`,
            background: colors.bgTertiary,
            color: isFileOpen && currentIndex > 0 ? colors.text : colors.textSecondary,
            cursor: isFileOpen && currentIndex > 0 ? 'pointer' : 'not-allowed',
            fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            transition: 'background 0.12s, border-color 0.12s',
          }}
          onMouseEnter={(e) => { if (isFileOpen && currentIndex > 0) e.currentTarget.style.background = alpha(colors.purple, 16); }}
          onMouseLeave={(e) => { e.currentTarget.style.background = colors.bgTertiary; }}
        >
          {'‹'}
        </button>

        <button
          onClick={() => navigateByIndex(currentIndex + 1)}
          disabled={!isFileOpen || currentIndex >= maxIndex}
          title={t('bottomNav.nextTooltip')}
          style={{
            width: 26, height: 26,
            borderRadius: 3,
            border: `1px solid ${colors.border}`,
            background: colors.bgTertiary,
            color: isFileOpen && currentIndex < maxIndex ? colors.text : colors.textSecondary,
            cursor: isFileOpen && currentIndex < maxIndex ? 'pointer' : 'not-allowed',
            fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            transition: 'background 0.12s, border-color 0.12s',
          }}
          onMouseEnter={(e) => { if (isFileOpen && currentIndex < maxIndex) e.currentTarget.style.background = alpha(colors.purple, 16); }}
          onMouseLeave={(e) => { e.currentTarget.style.background = colors.bgTertiary; }}
        >
          {'›'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Label small secondary style={{ flexShrink: 0 }}>{t('bottomNav.rowLabel')}</Label>
          <NumberInput
            value={currentRow}
            min={0}
            max={rows - 1}
            onChange={(e) => navigateRow(Number(e.target.value))}
            disabled={!isFileOpen}
            style={{ width: 60 }}
            title={t('bottomNav.rowTooltip')}
          />
          <span style={{ fontSize: '9pt', color: colors.textSecondary }}>/{rows - 1}</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Label small secondary style={{ flexShrink: 0 }}>{t('bottomNav.colLabel')}</Label>
          <NumberInput
            value={currentCol}
            min={0}
            max={cols - 1}
            onChange={(e) => navigateCol(Number(e.target.value))}
            disabled={!isFileOpen}
            style={{ width: 60 }}
            title={t('bottomNav.colTooltip')}
          />
          <span style={{ fontSize: '9pt', color: colors.textSecondary }}>/{cols - 1}</span>
        </div>

        <div style={{
          marginLeft: 'auto',
          fontSize: '9pt',
          color: colors.cyan,
          fontFamily: "'Courier New', monospace",
          flexShrink: 0,
        }}>
          {t('bottomNav.positionLabel', { row: currentRow, col: currentCol, index: currentIndex })}
        </div>
      </div>

      {/* Row 2: flat index slider + value spinbox */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Label small secondary style={{ flexShrink: 0 }}>{t('bottomNav.indexLabel')}</Label>
        <input
          type="range"
          min={0}
          max={maxIndex}
          value={currentIndex}
          onChange={(e) => navigateByIndex(Number(e.target.value))}
          disabled={!isFileOpen || patternCount === 0}
          title={t('bottomNav.indexSliderTooltip')}
          style={{
            flex: 1,
            accentColor: colors.purple,
            height: 4,
            cursor: isFileOpen && patternCount > 0 ? 'pointer' : 'not-allowed',
          }}
        />
        <NumberInput
          value={currentIndex}
          min={0}
          max={maxIndex}
          onChange={(e) => navigateByIndex(Number(e.target.value))}
          disabled={!isFileOpen}
          style={{ width: 70 }}
          title={t('bottomNav.indexInputTooltip')}
        />
      </div>

      {/* Row 3: Playback + Bookmark controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <button
          onClick={togglePlayback}
          disabled={!isFileOpen || patternCount === 0}
          title={playing ? t('bottomNav.pauseTooltip') : t('bottomNav.playTooltip')}
          style={{
            width: 28, height: 28,
            borderRadius: 3,
            border: `1px solid ${playing ? colors.green : colors.border}`,
            background: playing ? alpha(colors.green, 13) : colors.bgTertiary,
            color: playing ? colors.green : colors.text,
            cursor: isFileOpen ? 'pointer' : 'not-allowed',
            opacity: isFileOpen ? 1 : 0.45,
            fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.12s, border-color 0.12s, color 0.12s',
          }}
        >
          {playing ? '⏸' : '▶'}
        </button>

        <span title={t('bottomNav.speedTooltip')} style={{ display: 'inline-flex' }}>
          <Select
            value={playSpeed}
            onChange={(e) => setPlaySpeed(Number(e.target.value))}
            options={PLAYBACK_SPEEDS.map((fps) => ({ value: fps, label: t('bottomNav.fps', { fps }) }))}
            style={{ width: 80 }}
          />
        </span>

        <div style={{ flex: 1 }} />

        <Button
          small
          onClick={() => addBookmark({ index: currentIndex, row: currentRow, col: currentCol })}
          disabled={!isFileOpen || isCurrentBookmarked}
          variant={isCurrentBookmarked ? 'default' : 'purple'}
          title={t('bottomNav.bookmarkTooltip')}
        >
          {isCurrentBookmarked ? t('bottomNav.bookmarked') : t('bottomNav.bookmark')}
        </Button>

        <Button
          small
          onClick={clearBookmarks}
          disabled={bookmarks.length === 0}
          variant="danger"
          title={t('bottomNav.clearBookmarksTooltip')}
        >
          {t('bottomNav.clearBookmarks')}
        </Button>
      </div>

      {bookmarks.length === 0 && filePath && (
        <div style={{ fontSize: '8pt', color: colors.textSecondary, opacity: 0.6, paddingTop: 2 }}>
          {t('bottomNav.bookmarkHint')}
        </div>
      )}
      {bookmarks.length > 0 && (
        <div style={{
          display: 'flex', gap: 4, overflowX: 'auto', paddingBottom: 2, alignItems: 'center',
        }}>
          <Label small secondary style={{ flexShrink: 0 }}>{t('bottomNav.bookmarksLabel')}</Label>
          {bookmarks.map((bm) => (
            <button
              key={bm.index}
              onClick={() => fetchAndSetPattern(bm.row, bm.col, bm.index)}
              title={t('bottomNav.bookmarkChipTooltip', { index: bm.index, row: bm.row, col: bm.col })}
              style={{
                padding: '2px 8px',
                borderRadius: 3,
                border: `1px solid ${bm.index === currentIndex ? colors.purple : colors.border}`,
                background: bm.index === currentIndex ? alpha(colors.purple, 16) : colors.bgTertiary,
                color: bm.index === currentIndex ? colors.purple : colors.textSecondary,
                fontSize: '9pt',
                cursor: 'pointer',
                flexShrink: 0,
                fontFamily: "'Courier New', monospace",
                transition: 'background 0.12s, border-color 0.12s, color 0.12s',
              }}
              onMouseEnter={(e) => { if (bm.index !== currentIndex) { e.currentTarget.style.background = alpha(colors.purple, 10); e.currentTarget.style.color = colors.text; } }}
              onMouseLeave={(e) => { if (bm.index !== currentIndex) { e.currentTarget.style.background = colors.bgTertiary; e.currentTarget.style.color = colors.textSecondary; } }}
            >
              #{bm.index}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
