// The line every way of giving a PC ends in, for a PC that came from a saved calibration or from another file:
// "PC in use for <file>: ...", Show the bands, Go to indexing, the one step back, and what happened to a PC map.
// Anything the saving and applying of calibrations adds (Save as calibration..., Apply to other files...)
// arrives through `actions` and `children`; this file does not know about it.
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import PcGrant, { PcUndoDone } from '../PcGrant';
import { pcMapActionOf, pcMapNote } from '../pcMapNote';
import { noteStyle } from './CalibrationFacts';

/**
 * @param {string}    file     the dataset that got the PC
 * @param {number[]}  pc       the PC (Bruker)
 * @param {object}    previous `{pc, had_pc}` from the answer, or null (no step back)
 * @param {string}    mapAction  `pc_map_action` of the answer ("shifted" | "replaced" | "none")
 * @param {string}    copyKept the open copy that keeps a PC of its own when `file` is the file it was made from, or null
 * @param {Function}  onUndo   `(file) => Promise<{data} | {gone} | {error}>`, the page's put back
 * @param {Function}  onUndone called once the step back was made
 * @param {Array}     actions  extra buttons of PcGrant `{key, label, onClick}`
 * @param {node}      children what stands above the grant line (where the PC came from)
 */
export default function AppliedGrant({
  file, pc, previous, mapAction = null, copyKept = null, onUndo, onUndone = null, onGoToIndexing = null, renderBands = null,
  actions = [], children = null,
}) {
  const { t } = useTranslation('pcrefinement');
  const [bandsShown, setBandsShown] = useState(false);
  const [undone, setUndone] = useState(null);

  const undo = async () => {
    const answer = await onUndo(file);
    if (answer?.data) {
      const d = answer.data;
      setUndone({ previous: { pc: d.pc ?? null, had_pc: !!d.had_pc }, notRestored: Array.isArray(d.not_restored) ? d.not_restored : [] });
      setBandsShown(false);
      if (onUndone) onUndone(d);
      return undefined;
    }
    return answer;
  };

  if (undone) {
    return (
      <div data-testid="calibration-undone" style={{ marginTop: 6 }}>
        <PcUndoDone file={file} previous={undone.previous} notRestored={undone.notRestored} />
      </div>
    );
  }
  const note = pcMapNote(null, pcMapActionOf(mapAction));
  return (
    <div data-testid="calibration-grant" style={{ marginTop: 6 }}>
      {children}
      {note && (
        <div role="status" data-testid="calibration-map-note" style={noteStyle}>{t(`calibration.map.${note.key}`, note.params)}</div>
      )}
      {copyKept && (
        <div role="status" data-testid="calibration-copy-kept" style={noteStyle}>
          {t('calibration.apply.copyKeeps', { file, copy: copyKept })}
        </div>
      )}
      <PcGrant
        files={[file]}
        pc={pc}
        perPixel={pcMapActionOf(mapAction) === 'shifted'}
        previous={[previous]}
        actions={actions}
        bands={renderBands ? { shown: bandsShown, onToggle: () => setBandsShown((v) => !v) } : null}
        onGoToIndexing={onGoToIndexing}
        onUndo={onUndo ? undo : null}
      />
      {bandsShown && renderBands && renderBands(t('bands.caption'))}
    </div>
  );
}
