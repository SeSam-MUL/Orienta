/**
 * TopBar — file path + open/close + scan tools + handoff buttons.
 * Reuses behavior from the old HDF5Viewer top bar; just relocated.
 */
import { useTranslation } from 'react-i18next';
import { colors, spacing, alpha, Button, Label } from '../../theme/components';
import useCockpitStore from '../../stores/useCockpitStore';

export default function TopBar({
  isFileOpen, filePath, pathInput, setPathInput,
  openLoading, openError, setOpenError,
  onOpen, onClose, onBrowse,
  onSummary, onQualityScan, onDefectScan, onTree, onMetadata,
  onLoadToEBSD, onLoadToIndexing, currentPattern, onExportPattern,
}) {
  const { t } = useTranslation('hdf5viewer');
  const handlePathKeyDown = (e) => { if (e.key === 'Enter') onOpen(); };
  const resetUI = useCockpitStore((s) => s.resetUI);

  return (
    <>
      <div style={{
        display: 'flex', alignItems: 'center', gap: spacing.buttonSpacing,
        padding: `${spacing.innerMargin}px ${spacing.outerMargin}px`,
        background: colors.bgSecondary,
        borderBottom: `1px solid ${colors.border}`,
        flexShrink: 0, flexWrap: 'wrap',
      }}>
        <Label secondary small>{t('topbar.pathLabel')}</Label>
        <div style={{
          flex: 1, minWidth: 200, display: 'flex', alignItems: 'center',
          background: colors.bg, border: `1px solid ${colors.border}`,
          borderRadius: 4, height: spacing.buttonHeight, padding: '0 8px',
        }}>
          <input
            value={isFileOpen ? (filePath ?? '') : pathInput}
            onChange={(e) => !isFileOpen && setPathInput(e.target.value)}
            onKeyDown={handlePathKeyDown}
            readOnly={isFileOpen}
            placeholder={t('topbar.pathPlaceholder')}
            title={t('topbar.pathTooltip')}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: isFileOpen ? colors.green : colors.text,
              fontSize: '10pt', fontFamily: "'Courier New', monospace",
            }}
          />
        </div>

        {!isFileOpen && window.electronAPI?.openFile && (
          <Button onClick={onBrowse} title={t('topbar.browseTooltip')}>{t('topbar.browse')}</Button>
        )}
        {!isFileOpen && (
          <Button variant="primary" onClick={onOpen} disabled={openLoading || pathInput.trim() === ''} title={t('topbar.openTooltip')}>
            {openLoading ? t('topbar.opening') : t('topbar.open')}
          </Button>
        )}
        {isFileOpen && (
          <>
            <Button variant="danger" onClick={onClose} title={t('topbar.closeTooltip')}>{t('topbar.close')}</Button>
            <Button onClick={onExportPattern} disabled={!currentPattern} title={t('topbar.exportTooltip')}>{t('topbar.export')}</Button>
            <Button onClick={onSummary} title={t('topbar.summaryTooltip')}>{t('topbar.summary')}</Button>
            <Button onClick={onMetadata} title={t('topbar.metadataTooltip')}>{t('topbar.metadata')}</Button>
            <Button onClick={onTree} title={t('topbar.treeTooltip')}>{t('topbar.tree')}</Button>
            <Button onClick={onQualityScan} title={t('topbar.qualityTooltip')}>{t('topbar.quality')}</Button>
            <Button onClick={onDefectScan} title={t('topbar.defectTooltip')}>{t('topbar.defect')}</Button>
            <Button small onClick={resetUI} title={t('topbar.resetLayoutTooltip')}>{t('topbar.resetLayout')}</Button>
            {onLoadToEBSD && <Button variant="primary" onClick={onLoadToEBSD} title={t('topbar.toAnalysisTooltip')}>{t('topbar.toAnalysis')}</Button>}
            {onLoadToIndexing && <Button variant="primary" onClick={onLoadToIndexing} title={t('topbar.toIndexingTooltip')}>{t('topbar.toIndexing')}</Button>}
          </>
        )}
      </div>
      {openError && (
        <div role="alert" style={{
          padding: '6px 14px',
          background: alpha(colors.red, 9),
          borderBottom: `1px solid ${alpha(colors.red, 31)}`,
          fontSize: '9pt', color: colors.red, flexShrink: 0,
        }}>
          ⚠ {openError}
          <span onClick={() => setOpenError(null)} title={t('topbar.dismissErrorTooltip')} style={{ cursor: 'pointer', marginLeft: 8 }}>×</span>
        </div>
      )}
    </>
  );
}
