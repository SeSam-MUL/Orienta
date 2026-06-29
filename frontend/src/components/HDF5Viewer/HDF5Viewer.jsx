/**
 * HDF5Viewer — React port of gui/h5_viewer_gui.py
 *
 * Layout matches the PyQt5 original exactly:
 *   Top bar: path input + Open/Close buttons
 *   3-panel splitter: Overview minimap | Pattern canvas | Info tabs (EDS/Electron/Info)
 *   Bottom bar: Row/Col spinboxes, position label, flat-index slider
 *
 * State + lifecycle are split into 5 focused hooks under ./hooks:
 *   useH5File         — open/close/syncFromStatus
 *   useGridNavigation — row/col/index/pattern + race-safe fetch
 *   usePlayback       — auto-advance loop
 *   useBookmarks      — per-file localStorage list
 *   useScanResults    — backend /scan/quality + /scan/defects
 *
 * Uses ONLY shared components from theme/components.jsx — no local Button/Input/Label.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import useDataStore from '../../stores/useDataStore';
import useCockpitStore from '../../stores/useCockpitStore';
import { h5Api } from '../../services/api';
import { useConfirm, ConfirmDialog } from '../../theme/components';
import ScanResultsModal from './modals/ScanResultsModal';
import MetadataModal from './modals/MetadataModal';
import TreeModal from './modals/TreeModal';
import SummaryModal from './modals/SummaryModal';
import Cockpit from './Cockpit';
import { useH5File }         from './hooks/useH5File';
import { useGridNavigation } from './hooks/useGridNavigation';
import { usePlayback }       from './hooks/usePlayback';
import { useBookmarks }      from './hooks/useBookmarks';
import { useScanResults }    from './hooks/useScanResults';
import { patternToImgSrc }   from './utils/imageOps';

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
const ENHANCEMENT_DEFAULTS = {
  brightness: 0,
  contrast: 100,
  gamma: 100,
  invert: false,
  clahe: false,
};


// ---------------------------------------------------------------------------
// HDF5Viewer — main component
// ---------------------------------------------------------------------------
export default function HDF5Viewer({ onNavigate }) {
  // Store-level metadata (file feature flags, EDS/electron lists). These are
  // read directly from the store because multiple hooks/modals consume them.
  const {
    formatType, gridShape, patternCount, patternShape,
    hasEDS, hasElectronImages, edsElements, electronImages,
    hasPatterns, hasRawPatterns,
  } = useDataStore();
  const { t } = useTranslation('hdf5viewer');
  const [askConfirm, confirmProps] = useConfirm();

  // ---- Hooks -------------------------------------------------------------
  const {
    isOpen: isFileOpen, filePath,
    openLoading, openError, setOpenError,
    open, close, syncFromStatus,
  } = useH5File();

  const {
    row: currentRow, col: currentCol, index: currentIndex, pattern: currentPattern,
    patternLoading, rows, cols, maxIndex,
    navigateByIndex, navigateRow, navigateCol, fetchAndSetPattern, setPatternLoading,
  } = useGridNavigation({ gridShape, patternCount, isOpen: isFileOpen });

  const {
    playing, speed: playSpeed, setSpeed: setPlaySpeed, toggle: togglePlayback,
  } = usePlayback({ navigateByIndex, maxIndex, isOpen: isFileOpen });

  const {
    bookmarks, add: addBookmark, clear: clearBookmarks, isBookmarked,
  } = useBookmarks(filePath);

  const {
    quality: qualityResult, defects: defectResult,
    scanQuality, scanDefects, clearResults: clearScanResults,
  } = useScanResults();

  // ---- Local UI state (not part of any hook) -----------------------------
  const [pathInput, setPathInput]                 = useState('');
  const [enh, setEnh]                             = useState(ENHANCEMENT_DEFAULTS);
  const [showQualityModal, setShowQualityModal]   = useState(false);
  const [showSummaryModal, setShowSummaryModal]   = useState(false);
  const [showDefectModal,  setShowDefectModal]    = useState(false);
  const [showTreeModal,    setShowTreeModal]      = useState(false);
  const [showMetadataModal, setShowMetadataModal] = useState(false);

  // Memoized prop bag (used by future PixelInspector / Inspector tabs).
  // Kept around for parity with previous InfoTab integration.
  // eslint-disable-next-line no-unused-vars
  const fileInfo = useMemo(() => ({
    filePath, formatType, gridShape, patternCount, patternShape,
    hasEDS, hasElectronImages, edsElements, electronImages,
    hasPatterns, hasRawPatterns,
  }), [
    filePath, formatType, gridShape, patternCount, patternShape,
    hasEDS, hasElectronImages, edsElements, electronImages,
    hasPatterns, hasRawPatterns,
  ]);

  // ---------------------------------------------------------------------------
  // Mount: pick up file already opened by another module (EBSD Viewer etc.)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    syncFromStatus();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Another module (e.g. the Simulation page's "View Result" / "View Master
  // Pattern" buttons) can request a specific file by stashing its path in
  // sessionStorage('h5_preload_path') then navigating here. Open it on mount
  // and consume the request so it doesn't re-fire on the next open.
  useEffect(() => {
    let preload = null;
    try { preload = sessionStorage.getItem('h5_preload_path'); } catch { /* unavailable */ }
    if (!preload) return;
    try { sessionStorage.removeItem('h5_preload_path'); } catch { /* unavailable */ }
    setPathInput(preload);
    open(preload).catch(() => { /* error surfaces via openError state */ });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Pre-fill path input when the store already has a file (open elsewhere).
  useEffect(() => {
    if (isFileOpen && filePath && !pathInput) setPathInput(filePath);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFileOpen, filePath]);

  // ---------------------------------------------------------------------------
  // Auto-fetch first pattern when file was opened externally — useH5File.open
  // already fetches index 0, but a sync via /status -> store-update path needs
  // the same kick when there's no pattern yet.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!isFileOpen || currentPattern || patternLoading) return;
    setPatternLoading(true);
    h5Api.getPattern(currentIndex)
      .then((res) => {
        useDataStore.getState().setPosition(
          currentRow, currentCol, currentIndex, res.data?.image ?? null,
        );
      })
      .catch(() => {
        // Pattern not available — user can navigate to retry.
      })
      .finally(() => setPatternLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFileOpen]);

  // ---------------------------------------------------------------------------
  // Open / Close handlers — wire UI to useH5File
  // ---------------------------------------------------------------------------
  const handleOpen = useCallback(async () => {
    try {
      await open(pathInput);
    } catch {
      // Error surface is openError state; nothing to do here.
    }
  }, [open, pathInput]);

  const handleClose = useCallback(async () => {
    await close();
    setPathInput('');
    clearScanResults();
  }, [close, clearScanResults]);

  const handleBrowse = useCallback(async () => {
    if (window.electronAPI?.openFile) {
      const path = await window.electronAPI.openFile({ filters: [{ name: 'HDF5', extensions: ['h5', 'h5oina', 'hdf5'] }] });
      if (path) setPathInput(path);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Keyboard navigation
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const onKey = (e) => {
      if (!isFileOpen) return;
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      switch (e.key) {
        case 'ArrowRight': e.preventDefault(); navigateCol(currentCol + 1); break;
        case 'ArrowLeft':  e.preventDefault(); navigateCol(currentCol - 1); break;
        case 'ArrowDown':  e.preventDefault(); navigateRow(currentRow + 1); break;
        case 'ArrowUp':    e.preventDefault(); navigateRow(currentRow - 1); break;
        case 'PageDown':   e.preventDefault(); navigateByIndex(currentIndex + cols); break;
        case 'PageUp':     e.preventDefault(); navigateByIndex(currentIndex - cols); break;
        case 'Home':       e.preventDefault(); navigateByIndex(0); break;
        case 'End':        e.preventDefault(); navigateByIndex(maxIndex); break;
        default: break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isFileOpen, currentRow, currentCol, currentIndex, cols, maxIndex,
    navigateRow, navigateCol, navigateByIndex]);

  // ---------------------------------------------------------------------------
  // Export pattern
  // ---------------------------------------------------------------------------
  const handleExportPattern = useCallback(() => {
    if (!currentPattern) return;
    const a = document.createElement('a');
    a.href = patternToImgSrc(currentPattern);
    const base = filePath?.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? 'pattern';
    a.download = `${base}_r${currentRow}_c${currentCol}_i${currentIndex}.png`;
    a.click();
  }, [currentPattern, filePath, currentRow, currentCol, currentIndex]);

  // ---------------------------------------------------------------------------
  // Bookmarks (UI handlers — list state owned by useBookmarks)
  // ---------------------------------------------------------------------------
  const handleBookmark = useCallback(() => {
    if (!isFileOpen) return;
    addBookmark({ index: currentIndex, row: currentRow, col: currentCol });
  }, [isFileOpen, addBookmark, currentIndex, currentRow, currentCol]);

  const handleClearBookmarks = useCallback(() => {
    if (bookmarks.length === 0) return;
    askConfirm({
      title: t('bookmarksDialog.title'),
      message: t('bookmarksDialog.message', { count: bookmarks.length }),
      confirmLabel: t('bookmarksDialog.confirm'),
      onConfirm: clearBookmarks,
    });
  }, [bookmarks.length, clearBookmarks, askConfirm, t]);

  const isCurrentBookmarked = isBookmarked(currentIndex);

  // ---------------------------------------------------------------------------
  // Quality / Defect scans — triggered by buttons. The hook never throws;
  // it sets result.error on failure, which ScanResultsModal renders in red.
  // (console.warn from the hook is the dev-side fail-loud signal.)
  // ---------------------------------------------------------------------------
  const handleQualityScan = useCallback(async () => {
    if (!isFileOpen || patternCount === 0) return;
    setShowQualityModal(true);
    await scanQuality(50);
  }, [isFileOpen, patternCount, scanQuality]);

  const handleDefectScan = useCallback(async () => {
    if (!isFileOpen || patternCount === 0) return;
    setShowDefectModal(true);
    await scanDefects(60);
  }, [isFileOpen, patternCount, scanDefects]);

  // ---------------------------------------------------------------------------
  // Load to EBSD Viewer
  // ---------------------------------------------------------------------------
  const handleLoadToEBSD = useCallback(() => {
    if (!filePath || !onNavigate) return;
    try { sessionStorage.setItem('ebsd_preload_path', filePath); } catch { /* sessionStorage may be disabled */ }
    onNavigate('ebsdviewer');
  }, [filePath, onNavigate]);

  const handleLoadToIndexing = useCallback(() => {
    if (!filePath || !onNavigate) return;
    try {
      sessionStorage.setItem('indexing_preload_path', filePath);
      const sel = useCockpitStore.getState().selection;
      if (sel.mask) {
        // Compress as base64 for the next page; mask is Uint8Array (1 byte / pixel).
        // Build base64 in 8KB chunks to avoid RangeError on large masks
        // (spread `...mask` hits the JS argument-count limit on >500k pixels).
        let binary = '';
        const CHUNK = 8192;
        for (let i = 0; i < sel.mask.length; i += CHUNK) {
          binary += String.fromCharCode.apply(null, sel.mask.subarray(i, i + CHUNK));
        }
        const b64 = btoa(binary);
        sessionStorage.setItem('indexing_preload_mask', b64);
        sessionStorage.setItem('indexing_preload_grid', JSON.stringify(gridShape));
      } else {
        sessionStorage.removeItem('indexing_preload_mask');
        sessionStorage.removeItem('indexing_preload_grid');
      }
    } catch (err) { console.warn('preload-write failed', err); }
    onNavigate('indexing');
  }, [filePath, onNavigate, gridShape]);

  // ---------------------------------------------------------------------------
  // Enhancement helpers
  // ---------------------------------------------------------------------------
  const handleEnhReset = () => setEnh(ENHANCEMENT_DEFAULTS);

  const handleEnhAuto = useCallback(async () => {
    if (!currentPattern) return;
    const canvas = document.createElement('canvas');
    const img = new Image();
    await new Promise((res) => { img.onload = res; img.src = patternToImgSrc(currentPattern); });
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const lums = [];
    for (let i = 0; i < id.data.length; i += 4)
      lums.push(Math.round(0.299 * id.data[i] + 0.587 * id.data[i + 1] + 0.114 * id.data[i + 2]));
    lums.sort((a, b) => a - b);
    const p2  = lums[Math.floor(lums.length * 0.02)];
    const p98 = lums[Math.floor(lums.length * 0.98)];
    const range = Math.max(1, p98 - p2);
    const mid = (p2 + p98) / 2;
    setEnh((prev) => ({
      ...prev,
      brightness: Math.round(((128 - mid) / 255) * 100),
      contrast:   Math.min(300, Math.round((255 / range) * 100)),
    }));
  }, [currentPattern]);

  // ---------------------------------------------------------------------------
  // Render — Cockpit shell + modals
  // ---------------------------------------------------------------------------
  return (
    <>
      <Cockpit
        isFileOpen={isFileOpen}
        filePath={filePath}
        pathInput={pathInput} setPathInput={setPathInput}
        openLoading={openLoading} openError={openError} setOpenError={setOpenError}
        onOpen={handleOpen} onClose={handleClose} onBrowse={handleBrowse}
        onSummary={() => setShowSummaryModal(true)}
        onQualityScan={handleQualityScan} onDefectScan={handleDefectScan}
        onTree={() => setShowTreeModal(true)}
        onMetadata={() => setShowMetadataModal(true)}
        onLoadToEBSD={onNavigate ? handleLoadToEBSD : null}
        onLoadToIndexing={onNavigate ? handleLoadToIndexing : null}
        onExportPattern={handleExportPattern}
        currentPattern={currentPattern}
        currentRow={currentRow} currentCol={currentCol} currentIndex={currentIndex}
        rows={rows} cols={cols} maxIndex={maxIndex}
        gridShape={gridShape}
        navigateRow={navigateRow} navigateCol={navigateCol}
        navigateByIndex={navigateByIndex} fetchAndSetPattern={fetchAndSetPattern}
        onNavigateMap={(r, c) => fetchAndSetPattern(r, c, r * cols + c)}
        playing={playing} playSpeed={playSpeed} setPlaySpeed={setPlaySpeed}
        togglePlayback={togglePlayback}
        bookmarks={bookmarks} addBookmark={handleBookmark}
        clearBookmarks={handleClearBookmarks} isCurrentBookmarked={isCurrentBookmarked}
        patternCount={patternCount}
        patternShape={patternShape} patternLoading={patternLoading}
        enh={enh} setEnh={setEnh}
        handleEnhReset={handleEnhReset} handleEnhAuto={handleEnhAuto}
      />

      {/* Modals */}
      <ScanResultsModal open={showQualityModal} kind="quality" result={qualityResult}
        onGoTo={navigateByIndex} onClose={() => setShowQualityModal(false)} />
      <ScanResultsModal open={showDefectModal} kind="defects" result={defectResult}
        onClose={() => setShowDefectModal(false)} />
      {showSummaryModal && <SummaryModal onClose={() => setShowSummaryModal(false)} />}
      {showMetadataModal && (
        <MetadataModal isFileOpen={isFileOpen} onClose={() => setShowMetadataModal(false)} />
      )}
      {showTreeModal && (
        <TreeModal isFileOpen={isFileOpen} onClose={() => setShowTreeModal(false)} />
      )}
      <ConfirmDialog {...confirmProps} />
    </>
  );
}
